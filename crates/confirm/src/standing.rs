//! Standing approvals — pre-granted (agent, verb) consent that
//! bypasses the human-confirm prompt.
//!
//! ## Why this exists
//!
//! Reflexes fire signals unattended (fs changes, cron entries, system-state
//! transitions). Without standing approvals, every reflex firing that lands
//! a tier ≥ Destructive would block on a confirm prompt that no one is sitting
//! at to answer. Standing approvals let the user say once "this reflex,
//! with this agent identity, is allowed to do this verb" and the engine
//! auto-approves matching subsequent requests until the grant is revoked.
//!
//! ## Audit shape
//!
//! A re-grant after revoke creates a **new row** rather than mutating the
//! old one. This keeps the trail intact: you can answer "when was this
//! granted, when was it revoked, when did the user grant it again" by
//! walking the table in time order. The partial index on
//! `(agent_id, verb_ns, verb_action) WHERE revoked_at IS NULL` keeps the
//! hot-path lookup O(1).

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use storage::SqlitePool;
use uuid::Uuid;

use super::nonce::ConfirmError;

/// Key identifying *which* (agent, verb) tuple a standing approval
/// covers. Stored on [`crate::ApprovalSpec`] so the engine knows what
/// to look up.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrantKey {
    pub agent_id: String,
    pub verb_ns: String,
    pub verb_action: String,
}

impl GrantKey {
    pub fn new(
        agent_id: impl Into<String>,
        verb_ns: impl Into<String>,
        verb_action: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            verb_ns: verb_ns.into(),
            verb_action: verb_action.into(),
        }
    }
}

/// Scope qualifier boxing a grant to part of the request space. An empty
/// scope (all fields `None`) is the unscoped grant — it matches every
/// request for its verb. A scoped grant only matches requests whose
/// context satisfies *every* set qualifier; a request that carries no
/// value for a qualified dimension does **not** match (fail closed —
/// the user boxed the grant, so an unboxable request re-prompts).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrantScope {
    /// Path prefix (segment-boundary match: `/a/b` covers `/a/b` and
    /// `/a/b/...`, not `/a/bc`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prefix: Option<String>,
    /// Namespace, covering its `name/…` sub-namespaces — the same
    /// hierarchy rule recall and residency use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}

impl GrantScope {
    pub fn is_empty(&self) -> bool {
        self.path_prefix.is_none() && self.namespace.is_none()
    }

    /// True when the request context satisfies every set qualifier.
    pub fn matches(&self, path: Option<&str>, namespace: Option<&str>) -> bool {
        fn covers(prefix: &str, value: Option<&str>) -> bool {
            match value {
                Some(v) => {
                    v == prefix
                        || v.strip_prefix(prefix)
                            .is_some_and(|rest| rest.starts_with('/'))
                }
                None => false,
            }
        }
        if let Some(p) = &self.path_prefix {
            if !covers(p.trim_end_matches('/'), path) {
                return false;
            }
        }
        if let Some(ns) = &self.namespace {
            if !covers(ns, namespace) {
                return false;
            }
        }
        true
    }
}

/// One active standing approval row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StandingApproval {
    pub id: String,
    pub agent_id: String,
    pub verb_ns: String,
    pub verb_action: String,
    pub granted_at: DateTime<Utc>,
    pub note: Option<String>,
    /// When set, the grant stops matching after this instant (the next
    /// request re-prompts). Expired rows stay in the table for audit.
    pub expires_at: Option<DateTime<Utc>>,
    /// When set, the grant only matches requests inside the scope.
    pub scope: Option<GrantScope>,
}

/// Backend for standing approvals.
#[async_trait]
pub trait StandingApprovalStore: Send + Sync {
    /// True iff there is at least one non-revoked, unexpired, **unscoped**
    /// grant matching the key. Context-free callers cannot satisfy a
    /// scope qualifier, so scoped grants never match here (fail closed —
    /// use [`is_granted_for`](Self::is_granted_for) when the request
    /// context is known). Implementations should make this cheap — the
    /// engine calls it on every confirmation tier ≥
    /// requires_confirmation.
    async fn is_granted(&self, key: &GrantKey) -> Result<bool, ConfirmError>;

    /// Scope-aware check: true iff a non-revoked, unexpired grant
    /// matches the key *and* its scope (if any) admits the request's
    /// path/namespace context. The default delegates to
    /// [`is_granted`](Self::is_granted), i.e. only unscoped grants
    /// match — a safe floor for stores that don't model scopes.
    async fn is_granted_for(
        &self,
        key: &GrantKey,
        _path: Option<&str>,
        _namespace: Option<&str>,
    ) -> Result<bool, ConfirmError> {
        self.is_granted(key).await
    }

    /// Record a new grant; returns its id. Unscoped and non-expiring —
    /// delegates to [`grant_scoped`](Self::grant_scoped).
    async fn grant(&self, key: &GrantKey, note: Option<&str>) -> Result<String, ConfirmError> {
        self.grant_scoped(key, note, None, None).await
    }

    /// Record a new grant with an optional expiry instant and scope box;
    /// returns its id.
    async fn grant_scoped(
        &self,
        key: &GrantKey,
        note: Option<&str>,
        expires_at: Option<DateTime<Utc>>,
        scope: Option<GrantScope>,
    ) -> Result<String, ConfirmError>;

    /// Revoke a specific grant by id. Returns true iff a row was
    /// updated (false when the id is unknown or already revoked).
    async fn revoke(&self, id: &str) -> Result<bool, ConfirmError>;

    /// All non-revoked, unexpired grants, newest-first.
    async fn list_active(&self) -> Result<Vec<StandingApproval>, ConfirmError>;
}

/// SQLite-backed [`StandingApprovalStore`].
pub struct SqliteStandingApprovals {
    db: SqlitePool,
}

impl SqliteStandingApprovals {
    pub fn new(db: SqlitePool) -> Self {
        let store = Self { db };
        if let Err(e) = store.ensure_columns() {
            tracing::warn!("standing-approvals schema upgrade failed: {e}");
        }
        store
    }

    /// Bring the table (created by central migration v21) up to this
    /// crate's column set. Introspection-guarded `ALTER`s so the upgrade
    /// is idempotent — this crate owns its column evolution the same way
    /// audit/ganglia own their tables.
    fn ensure_columns(&self) -> Result<(), ConfirmError> {
        Ok(self.db.with_conn(|conn| {
            let mut existing = std::collections::HashSet::new();
            {
                let mut stmt = conn.prepare("PRAGMA table_info(standing_approvals)")?;
                let mut rows = stmt.query([])?;
                while let Some(row) = rows.next()? {
                    existing.insert(row.get::<_, String>(1)?);
                }
            }
            if !existing.contains("expires_at") {
                conn.execute_batch("ALTER TABLE standing_approvals ADD COLUMN expires_at TEXT;")?;
            }
            if !existing.contains("scope_json") {
                conn.execute_batch("ALTER TABLE standing_approvals ADD COLUMN scope_json TEXT;")?;
            }
            Ok(())
        })?)
    }
}

/// Shared row → struct mapping for the full column set.
fn row_to_approval(row: &rusqlite::Row<'_>) -> rusqlite::Result<StandingApproval> {
    let granted_at_raw: String = row.get(4)?;
    let expires_raw: Option<String> = row.get(6)?;
    let scope_raw: Option<String> = row.get(7)?;
    Ok(StandingApproval {
        id: row.get(0)?,
        agent_id: row.get(1)?,
        verb_ns: row.get(2)?,
        verb_action: row.get(3)?,
        granted_at: parse_ts(&granted_at_raw),
        note: row.get(5)?,
        expires_at: expires_raw.as_deref().map(parse_ts),
        scope: scope_raw.and_then(|s| serde_json::from_str(&s).ok()),
    })
}

fn parse_ts(raw: &str) -> DateTime<Utc> {
    // Migration v21 uses `datetime('now')` which writes
    // "YYYY-MM-DD HH:MM:SS" (no T, no offset); manual writes use
    // RFC3339. Accept both so the column stays portable.
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return dt.with_timezone(&Utc);
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S") {
        return naive.and_utc();
    }
    Utc::now()
}

#[async_trait]
impl StandingApprovalStore for SqliteStandingApprovals {
    async fn is_granted(&self, key: &GrantKey) -> Result<bool, ConfirmError> {
        // Context-free: only unscoped grants can match (fail closed).
        let agent_id = key.agent_id.clone();
        let verb_ns = key.verb_ns.clone();
        let verb_action = key.verb_action.clone();
        let now = Utc::now().to_rfc3339();
        let granted = self.db.with_conn(move |conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM standing_approvals
                 WHERE agent_id = ?1 AND verb_ns = ?2 AND verb_action = ?3
                   AND revoked_at IS NULL
                   AND (expires_at IS NULL OR expires_at > ?4)
                   AND scope_json IS NULL",
                params![agent_id, verb_ns, verb_action, now],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })?;
        Ok(granted)
    }

    async fn is_granted_for(
        &self,
        key: &GrantKey,
        path: Option<&str>,
        namespace: Option<&str>,
    ) -> Result<bool, ConfirmError> {
        let agent_id = key.agent_id.clone();
        let verb_ns = key.verb_ns.clone();
        let verb_action = key.verb_action.clone();
        let now = Utc::now().to_rfc3339();
        // Fetch active candidates; scope matching happens in Rust so the
        // matching rules live in exactly one place (GrantScope::matches).
        let scopes: Vec<Option<String>> = self.db.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT scope_json FROM standing_approvals
                 WHERE agent_id = ?1 AND verb_ns = ?2 AND verb_action = ?3
                   AND revoked_at IS NULL
                   AND (expires_at IS NULL OR expires_at > ?4)",
            )?;
            let rows = stmt
                .query_map(params![agent_id, verb_ns, verb_action, now], |row| {
                    row.get(0)
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        Ok(scopes.iter().any(|raw| match raw {
            None => true, // unscoped grant matches any context
            Some(json) => serde_json::from_str::<GrantScope>(json)
                .map(|scope| scope.matches(path, namespace))
                .unwrap_or(false), // unparseable scope fails closed
        }))
    }

    async fn grant_scoped(
        &self,
        key: &GrantKey,
        note: Option<&str>,
        expires_at: Option<DateTime<Utc>>,
        scope: Option<GrantScope>,
    ) -> Result<String, ConfirmError> {
        let id = Uuid::new_v4().to_string();
        let agent_id = key.agent_id.clone();
        let verb_ns = key.verb_ns.clone();
        let verb_action = key.verb_action.clone();
        let note_owned = note.map(|s| s.to_string());
        let expires_owned = expires_at.map(|t| t.to_rfc3339());
        // An empty scope is the unscoped grant — store NULL, not "{}".
        let scope_owned = scope
            .filter(|s| !s.is_empty())
            .and_then(|s| serde_json::to_string(&s).ok());
        let id_for_db = id.clone();
        self.db.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO standing_approvals
                    (id, agent_id, verb_ns, verb_action, granted_at, note, expires_at, scope_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    id_for_db,
                    agent_id,
                    verb_ns,
                    verb_action,
                    Utc::now().to_rfc3339(),
                    note_owned,
                    expires_owned,
                    scope_owned,
                ],
            )?;
            Ok(())
        })?;
        Ok(id)
    }

    async fn revoke(&self, id: &str) -> Result<bool, ConfirmError> {
        let id_owned = id.to_string();
        let updated = self.db.with_conn(move |conn| {
            let n = conn.execute(
                "UPDATE standing_approvals
                 SET revoked_at = ?2
                 WHERE id = ?1 AND revoked_at IS NULL",
                params![id_owned, Utc::now().to_rfc3339()],
            )?;
            Ok(n)
        })?;
        Ok(updated > 0)
    }

    async fn list_active(&self) -> Result<Vec<StandingApproval>, ConfirmError> {
        let now = Utc::now().to_rfc3339();
        let rows = self.db.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, agent_id, verb_ns, verb_action, granted_at, note, expires_at, scope_json
                 FROM standing_approvals
                 WHERE revoked_at IS NULL
                   AND (expires_at IS NULL OR expires_at > ?1)
                 ORDER BY granted_at DESC",
            )?;
            let mut out = Vec::new();
            let mut rows = stmt.query([now])?;
            while let Some(row) = rows.next()? {
                out.push(row_to_approval(row)?);
            }
            Ok(out)
        })?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SqliteStandingApprovals {
        SqliteStandingApprovals::new(SqlitePool::open_memory().unwrap())
    }

    #[tokio::test]
    async fn grant_then_lookup_reports_granted() {
        let s = store();
        let key = GrantKey::new("agent-a", "fs", "write");
        assert!(!s.is_granted(&key).await.unwrap());
        let _id = s.grant(&key, Some("nightly sync")).await.unwrap();
        assert!(s.is_granted(&key).await.unwrap());
    }

    #[tokio::test]
    async fn revoke_clears_grant() {
        let s = store();
        let key = GrantKey::new("agent-a", "fs", "write");
        let id = s.grant(&key, None).await.unwrap();
        assert!(s.is_granted(&key).await.unwrap());
        assert!(s.revoke(&id).await.unwrap());
        assert!(!s.is_granted(&key).await.unwrap());
    }

    #[tokio::test]
    async fn revoke_returns_false_for_unknown_id() {
        let s = store();
        assert!(!s.revoke("nonexistent").await.unwrap());
    }

    #[tokio::test]
    async fn revoke_returns_false_when_already_revoked() {
        let s = store();
        let key = GrantKey::new("a", "v", "x");
        let id = s.grant(&key, None).await.unwrap();
        assert!(s.revoke(&id).await.unwrap());
        assert!(!s.revoke(&id).await.unwrap(), "second revoke is a no-op");
    }

    #[tokio::test]
    async fn list_active_filters_revoked_rows() {
        let s = store();
        let k1 = GrantKey::new("a", "v", "x");
        let k2 = GrantKey::new("b", "v", "y");
        let id1 = s.grant(&k1, None).await.unwrap();
        let _id2 = s.grant(&k2, None).await.unwrap();
        assert_eq!(s.list_active().await.unwrap().len(), 2);
        s.revoke(&id1).await.unwrap();
        let active = s.list_active().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].agent_id, "b");
    }

    #[tokio::test]
    async fn re_grant_after_revoke_creates_separate_row() {
        let s = store();
        let key = GrantKey::new("a", "v", "x");
        let id1 = s.grant(&key, Some("first")).await.unwrap();
        s.revoke(&id1).await.unwrap();
        let id2 = s.grant(&key, Some("second")).await.unwrap();
        assert_ne!(id1, id2, "re-grant must be a separate row for audit");
        assert!(s.is_granted(&key).await.unwrap());
    }

    #[test]
    fn scope_matching_is_segment_boundary_and_fails_closed() {
        let scope = GrantScope {
            path_prefix: Some("/repo/app".into()),
            namespace: None,
        };
        assert!(scope.matches(Some("/repo/app"), None));
        assert!(scope.matches(Some("/repo/app/src/main.rs"), None));
        assert!(
            !scope.matches(Some("/repo/app2"), None),
            "segment, not prefix"
        );
        assert!(!scope.matches(Some("/other"), None));
        assert!(
            !scope.matches(None, None),
            "no context cannot satisfy a scope"
        );

        let ns_scope = GrantScope {
            path_prefix: None,
            namespace: Some("work".into()),
        };
        assert!(ns_scope.matches(None, Some("work")));
        assert!(ns_scope.matches(None, Some("work/projects")));
        assert!(!ns_scope.matches(None, Some("workshop")));
        assert!(!ns_scope.matches(None, None));

        assert!(
            GrantScope::default().matches(None, None),
            "empty scope is unscoped"
        );
    }

    /// DoD: an expired grant re-prompts — it stops matching every lookup
    /// and leaves the active list, while an unexpired TTL grant matches.
    #[tokio::test]
    async fn expired_grant_no_longer_matches() {
        let s = store();
        let key = GrantKey::new("agent-a", "fs", "write");

        let _live = s
            .grant_scoped(
                &key,
                None,
                Some(Utc::now() + chrono::Duration::hours(1)),
                None,
            )
            .await
            .unwrap();
        assert!(
            s.is_granted(&key).await.unwrap(),
            "unexpired TTL grant matches"
        );

        let s2 = store();
        let _dead = s2
            .grant_scoped(
                &key,
                None,
                Some(Utc::now() - chrono::Duration::seconds(1)),
                None,
            )
            .await
            .unwrap();
        assert!(
            !s2.is_granted(&key).await.unwrap(),
            "expired grant must not match"
        );
        assert!(
            !s2.is_granted_for(&key, Some("/x"), Some("personal"))
                .await
                .unwrap(),
            "expired grant must not match any context"
        );
        assert!(
            s2.list_active().await.unwrap().is_empty(),
            "expired grant must leave the active list"
        );
    }

    /// DoD: a scope mismatch re-prompts — the grant only matches inside
    /// its box, and never matches a context-free lookup.
    #[tokio::test]
    async fn scoped_grant_matches_only_inside_its_box() {
        let s = store();
        let key = GrantKey::new("agent-a", "shell", "exec");
        s.grant_scoped(
            &key,
            None,
            None,
            Some(GrantScope {
                path_prefix: Some("/repo/app".into()),
                namespace: None,
            }),
        )
        .await
        .unwrap();

        assert!(s
            .is_granted_for(&key, Some("/repo/app/scripts"), None)
            .await
            .unwrap());
        assert!(
            !s.is_granted_for(&key, Some("/elsewhere"), None)
                .await
                .unwrap(),
            "scope mismatch must re-prompt"
        );
        assert!(
            !s.is_granted_for(&key, None, None).await.unwrap(),
            "a request without path context cannot satisfy a path scope"
        );
        assert!(
            !s.is_granted(&key).await.unwrap(),
            "context-free is_granted must never match a scoped grant"
        );
    }

    #[tokio::test]
    async fn unscoped_grant_matches_any_context() {
        let s = store();
        let key = GrantKey::new("agent-a", "fs", "write");
        s.grant(&key, None).await.unwrap();
        assert!(s
            .is_granted_for(&key, Some("/anywhere"), Some("any"))
            .await
            .unwrap());
        assert!(s.is_granted_for(&key, None, None).await.unwrap());
        assert!(s.is_granted(&key).await.unwrap());
    }

    #[tokio::test]
    async fn list_active_carries_expiry_and_scope() {
        let s = store();
        let key = GrantKey::new("agent-a", "fs", "write");
        let exp = Utc::now() + chrono::Duration::hours(2);
        let scope = GrantScope {
            path_prefix: Some("/repo".into()),
            namespace: Some("work".into()),
        };
        s.grant_scoped(&key, Some("boxed"), Some(exp), Some(scope.clone()))
            .await
            .unwrap();
        let rows = s.list_active().await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].scope.as_ref(), Some(&scope));
        let got = rows[0].expires_at.expect("expiry must round-trip");
        assert!((got - exp).num_seconds().abs() <= 1);
    }

    #[tokio::test]
    async fn is_granted_distinguishes_different_verbs() {
        let s = store();
        s.grant(&GrantKey::new("a", "fs", "write"), None)
            .await
            .unwrap();
        assert!(s
            .is_granted(&GrantKey::new("a", "fs", "write"))
            .await
            .unwrap());
        assert!(!s
            .is_granted(&GrantKey::new("a", "fs", "read"))
            .await
            .unwrap());
        assert!(!s
            .is_granted(&GrantKey::new("b", "fs", "write"))
            .await
            .unwrap());
    }
}
