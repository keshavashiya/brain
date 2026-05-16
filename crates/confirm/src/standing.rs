//! Standing approvals — pre-granted (agent, verb) consent that
//! bypasses the human-confirm prompt.
//!
//! ## Why this exists
//!
//! Phase 5 introduces reflexes — triggers that fire signals unattended
//! (fs changes, cron entries, system-state transitions). Without
//! standing approvals, every reflex firing that lands a tier ≥
//! Destructive would block on a confirm prompt that no one is sitting
//! at to answer. Standing approvals let the user say once "this
//! reflex, with this agent identity, is allowed to do this verb" and
//! the engine auto-approves matching subsequent requests until the
//! grant is revoked.
//!
//! ## Audit shape
//!
//! A re-grant after revoke creates a **new row** rather than
//! mutating the old one. This keeps the trail intact: you can answer
//! "when was this granted, when was it revoked, when did the user
//! grant it again" by walking the table in time order. The partial
//! index on `(agent_id, verb_ns, verb_action) WHERE revoked_at IS
//! NULL` keeps the hot-path lookup O(1).
//!
//! ## What this slice does (PR-5f.1)
//!
//! - Migration v21 + `SqliteStandingApprovals` impl
//! - `ApprovalSpec::grant_key` so callers can opt in
//! - `SqliteConfirmationEngine::with_standing_approvals` to wire the
//!   check into the existing `request()` flow
//!
//! What it deliberately does **not** do (lands in PR-5f.2):
//!
//! - YAML config declaration of approvals at startup
//! - `/approval-revoke` slash command
//! - Wiring through the reflex pipeline end-to-end

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

/// One active standing approval row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StandingApproval {
    pub id: String,
    pub agent_id: String,
    pub verb_ns: String,
    pub verb_action: String,
    pub granted_at: DateTime<Utc>,
    pub note: Option<String>,
}

/// Backend for standing approvals.
#[async_trait]
pub trait StandingApprovalStore: Send + Sync {
    /// True iff there is at least one non-revoked grant matching the
    /// key. Implementations should make this cheap — the engine calls
    /// it on every confirmation tier ≥ requires_confirmation.
    async fn is_granted(&self, key: &GrantKey) -> Result<bool, ConfirmError>;

    /// Record a new grant; returns its id.
    async fn grant(&self, key: &GrantKey, note: Option<&str>) -> Result<String, ConfirmError>;

    /// Revoke a specific grant by id. Returns true iff a row was
    /// updated (false when the id is unknown or already revoked).
    async fn revoke(&self, id: &str) -> Result<bool, ConfirmError>;

    /// All non-revoked grants, newest-first.
    async fn list_active(&self) -> Result<Vec<StandingApproval>, ConfirmError>;
}

/// SQLite-backed [`StandingApprovalStore`].
pub struct SqliteStandingApprovals {
    db: SqlitePool,
}

impl SqliteStandingApprovals {
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }
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
        let agent_id = key.agent_id.clone();
        let verb_ns = key.verb_ns.clone();
        let verb_action = key.verb_action.clone();
        let granted = self.db.with_conn(move |conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM standing_approvals
                 WHERE agent_id = ?1 AND verb_ns = ?2 AND verb_action = ?3
                   AND revoked_at IS NULL",
                params![agent_id, verb_ns, verb_action],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })?;
        Ok(granted)
    }

    async fn grant(&self, key: &GrantKey, note: Option<&str>) -> Result<String, ConfirmError> {
        let id = Uuid::new_v4().to_string();
        let agent_id = key.agent_id.clone();
        let verb_ns = key.verb_ns.clone();
        let verb_action = key.verb_action.clone();
        let note_owned = note.map(|s| s.to_string());
        let id_for_db = id.clone();
        self.db.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO standing_approvals
                    (id, agent_id, verb_ns, verb_action, granted_at, note)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    id_for_db,
                    agent_id,
                    verb_ns,
                    verb_action,
                    Utc::now().to_rfc3339(),
                    note_owned,
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
        let rows = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, agent_id, verb_ns, verb_action, granted_at, note
                 FROM standing_approvals
                 WHERE revoked_at IS NULL
                 ORDER BY granted_at DESC",
            )?;
            let mut out = Vec::new();
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let granted_at_raw: String = row.get(4)?;
                out.push(StandingApproval {
                    id: row.get(0)?,
                    agent_id: row.get(1)?,
                    verb_ns: row.get(2)?,
                    verb_action: row.get(3)?,
                    granted_at: parse_ts(&granted_at_raw),
                    note: row.get(5)?,
                });
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
