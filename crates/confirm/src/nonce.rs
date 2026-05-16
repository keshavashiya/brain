//! Nonce-based approval workflow.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use storage::SqlitePool;
use thiserror::Error;
use tracing;
use uuid::Uuid;

use super::standing::{GrantKey, StandingApprovalStore};
use super::tier::ActionTier;
use super::timeout::EscalationPolicy;

/// Specification for an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalSpec {
    pub action_description: String,
    pub tier: ActionTier,
    pub nonce: String,
    pub timeout: std::time::Duration,
    pub escalation: EscalationPolicy,
    pub preferred_channel: Option<String>,
    pub alternatives: Vec<String>,
    /// Optional standing-approval key — when set and the engine has a
    /// [`StandingApprovalStore`] wired with a matching non-revoked
    /// grant, the request auto-approves without prompting. `None`
    /// preserves existing behavior (every tier ≥ Destructive blocks
    /// on user input).
    #[serde(default)]
    pub grant_key: Option<GrantKey>,
}

impl ApprovalSpec {
    pub fn new(action_description: impl Into<String>, tier: ActionTier) -> Self {
        Self {
            action_description: action_description.into(),
            tier,
            nonce: Uuid::new_v4().to_string(),
            timeout: tier.default_timeout(),
            escalation: EscalationPolicy::Abort,
            preferred_channel: None,
            alternatives: Vec::new(),
            grant_key: None,
        }
    }

    pub fn with_grant_key(mut self, key: GrantKey) -> Self {
        self.grant_key = Some(key);
        self
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_escalation(mut self, policy: EscalationPolicy) -> Self {
        self.escalation = policy;
        self
    }

    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.preferred_channel = Some(channel.into());
        self
    }

    pub fn with_alternatives(mut self, alternatives: Vec<String>) -> Self {
        self.alternatives = alternatives;
        self
    }
}

/// Decision made by the user on an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalDecision {
    Approve,
    Reject,
    RejectWithReason(String),
}

impl std::fmt::Display for ApprovalDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApprovalDecision::Approve => write!(f, "approve"),
            ApprovalDecision::Reject => write!(f, "reject"),
            ApprovalDecision::RejectWithReason(r) => write!(f, "reject ({r})"),
        }
    }
}

/// Outcome of an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalOutcome {
    Approved,
    Rejected { reason: String },
    TimedOut,
    Aborted { reason: String },
}

/// Status of a pending approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Pending {
        since: DateTime<Utc>,
    },
    Resolved {
        outcome: ApprovalOutcome,
        resolved_at: DateTime<Utc>,
    },
}

#[derive(Debug, Error)]
pub enum ConfirmError {
    #[error("Storage error: {0}")]
    Storage(#[from] storage::sqlite::SqliteError),
    #[error("Approval not found: {0}")]
    NotFound(String),
    #[error("Approval already resolved: {0}")]
    AlreadyResolved(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Timeout: {0}")]
    Timeout(String),
}

/// Human approval gates.
#[async_trait]
pub trait ConfirmationEngine: Send + Sync {
    /// Request approval for an action. Blocks until resolved or timed out.
    async fn request(&self, spec: ApprovalSpec) -> Result<ApprovalOutcome, ConfirmError>;

    /// Register a user response to an approval request.
    async fn respond(&self, nonce: &str, decision: ApprovalDecision) -> Result<(), ConfirmError>;

    /// Check status of a pending approval.
    async fn status(&self, nonce: &str) -> Result<ApprovalStatus, ConfirmError>;

    /// List all pending approvals.
    async fn pending(&self) -> Result<Vec<ApprovalSpec>, ConfirmError>;
}

/// Pending approval request stored in SQLite.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
struct PendingApproval {
    nonce: String,
    spec: ApprovalSpec,
    created_at: String,
    resolved: bool,
    outcome: Option<String>, // JSON-encoded ApprovalOutcome
}

/// SQLite-backed confirmation engine.
pub struct SqliteConfirmationEngine {
    db: SqlitePool,
    notifier: Option<std::sync::Arc<dyn crate::notifier::ApprovalNotifier>>,
    standing: Option<std::sync::Arc<dyn StandingApprovalStore>>,
}

impl SqliteConfirmationEngine {
    pub fn new(db: SqlitePool) -> Self {
        Self {
            db,
            notifier: None,
            standing: None,
        }
    }

    /// Attach an approval notifier — the engine will fire `notify()` once
    /// per pending request that requires explicit confirmation, so the
    /// user actually sees the prompt on their preferred channel. Without
    /// a notifier, the engine writes to SQLite and blocks until either
    /// `respond()` is called externally or the timeout expires.
    pub fn with_notifier(
        mut self,
        notifier: std::sync::Arc<dyn crate::notifier::ApprovalNotifier>,
    ) -> Self {
        self.notifier = Some(notifier);
        self
    }

    /// Attach a standing-approval store. When set, every request whose
    /// `spec.grant_key` matches an active grant auto-approves without
    /// notifying the user. Without a store, behavior is identical to
    /// the pre-Phase-5 engine.
    pub fn with_standing_approvals(
        mut self,
        store: std::sync::Arc<dyn StandingApprovalStore>,
    ) -> Self {
        self.standing = Some(store);
        self
    }

    /// True when the spec carries a `grant_key`, a store is wired, and
    /// the store reports a matching active grant. Storage errors are
    /// logged and treated as "not granted" — failing closed is safer
    /// than auto-approving on a flaky read.
    async fn standing_grants_request(&self, spec: &ApprovalSpec) -> bool {
        let Some(store) = &self.standing else {
            return false;
        };
        let Some(key) = &spec.grant_key else {
            return false;
        };
        match store.is_granted(key).await {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    agent_id = %key.agent_id,
                    verb_ns = %key.verb_ns,
                    verb_action = %key.verb_action,
                    "standing-approval lookup failed; failing closed (no bypass)"
                );
                false
            }
        }
    }

    pub fn ensure_tables(&self) -> Result<(), ConfirmError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS approval_requests (
                    nonce        TEXT PRIMARY KEY,
                    description  TEXT NOT NULL,
                    tier         TEXT NOT NULL,
                    timeout_secs INTEGER NOT NULL,
                    escalation   TEXT NOT NULL,
                    channel      TEXT,
                    alternatives TEXT,
                    created_at   TEXT NOT NULL,
                    resolved     INTEGER NOT NULL DEFAULT 0,
                    outcome      TEXT,
                    resolved_at  TEXT
                );
                "#,
            )?;
            Ok(())
        })?;
        Ok(())
    }
}

#[async_trait]
impl ConfirmationEngine for SqliteConfirmationEngine {
    async fn request(&self, spec: ApprovalSpec) -> Result<ApprovalOutcome, ConfirmError> {
        let nonce = spec.nonce.clone();

        // Store the pending request
        self.db.with_conn(|conn| {
            conn.execute(
                r#"INSERT INTO approval_requests (
                    nonce, description, tier, timeout_secs, escalation, channel, alternatives,
                    created_at, resolved
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0)"#,
                params![
                    spec.nonce,
                    spec.action_description,
                    spec.tier.to_string(),
                    spec.timeout.as_secs() as i64,
                    serde_json::to_string(&spec.escalation).unwrap_or_default(),
                    spec.preferred_channel,
                    serde_json::to_string(&spec.alternatives).unwrap_or_default(),
                    Utc::now().to_rfc3339(),
                ],
            )?;
            Ok(())
        })?;

        tracing::info!(nonce = %nonce, tier = %spec.tier, "approval request created");

        // Standing-approval bypass: if this spec carries a grant_key
        // and we have a store with a matching active grant, treat as
        // pre-approved. We still keep the row in the audit table — it
        // just resolves immediately rather than waiting for a human.
        let standing_bypass = self.standing_grants_request(&spec).await;

        // Push the prompt out through the channel layer (if wired) —
        // skipped when a standing approval already covers the request,
        // since no human action is required. Best-effort: a delivery
        // failure is logged but does not change request semantics.
        if spec.tier.requires_confirmation() && !standing_bypass {
            if let Some(notifier) = &self.notifier {
                if let Err(e) = notifier.notify(&spec).await {
                    tracing::warn!(
                        nonce = %nonce,
                        tier = %spec.tier,
                        error = %e,
                        "approval prompt delivery failed; user must use CLI/API to respond",
                    );
                }
            } else {
                tracing::debug!(
                    nonce = %nonce,
                    tier = %spec.tier,
                    "no ApprovalNotifier wired — request will rely on direct CLI/API response",
                );
            }
        }

        // Non-confirmatory tiers (Read/Write/Execute) auto-approve, as
        // do tiers covered by a standing approval. Destructive/External
        // without a standing bypass block here, polling until
        // `respond()` is called via the CLI or the timeout expires.
        if !spec.tier.requires_confirmation() || standing_bypass {
            if standing_bypass {
                tracing::info!(
                    nonce = %nonce,
                    tier = %spec.tier,
                    "auto-approving via standing approval"
                );
            } else {
                tracing::info!(nonce = %nonce, "auto-approving non-confirmatory action");
            }
            self.respond(&nonce, ApprovalDecision::Approve).await?;
        }

        let deadline = std::time::Instant::now() + spec.timeout;
        let poll_interval = std::time::Duration::from_millis(250);

        loop {
            let outcome_json: Option<String> = self
                .db
                .with_conn(|conn| {
                    let result = conn
                        .query_row(
                            "SELECT outcome FROM approval_requests WHERE nonce = ? AND resolved = 1",
                            [&nonce],
                            |row| row.get::<_, String>(0),
                        )
                        .ok();
                    Ok(result)
                })
                .map_err(ConfirmError::from)?;

            if let Some(json) = outcome_json {
                let outcome: ApprovalOutcome = serde_json::from_str(&json)
                    .map_err(|e| ConfirmError::InvalidData(format!("invalid outcome JSON: {e}")))?;
                return Ok(outcome);
            }

            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    nonce = %nonce,
                    tier = %spec.tier,
                    timeout_secs = spec.timeout.as_secs(),
                    "approval timed out"
                );
                // Mark the entry resolved as TimedOut so it no longer appears
                // in pending() and can be audited.
                let timeout_outcome = ApprovalOutcome::TimedOut;
                let outcome_json = serde_json::to_string(&timeout_outcome).unwrap_or_default();
                let now = Utc::now().to_rfc3339();
                let _ = self.db.with_conn(|conn| {
                    conn.execute(
                        "UPDATE approval_requests SET resolved = 1, outcome = ?, resolved_at = ? \
                         WHERE nonce = ? AND resolved = 0",
                        params![outcome_json, now, &nonce],
                    )?;
                    Ok(())
                });
                return Ok(timeout_outcome);
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn respond(&self, nonce: &str, decision: ApprovalDecision) -> Result<(), ConfirmError> {
        let outcome = match decision {
            ApprovalDecision::Approve => ApprovalOutcome::Approved,
            ApprovalDecision::Reject => ApprovalOutcome::Rejected {
                reason: "rejected by user".to_string(),
            },
            ApprovalDecision::RejectWithReason(ref r) => {
                ApprovalOutcome::Rejected { reason: r.clone() }
            }
        };

        let decision_str = decision.to_string();

        let outcome_json = serde_json::to_string(&outcome).unwrap_or_default();
        let now = Utc::now().to_rfc3339();

        self.db.with_conn(|conn| {
            let resolved: bool = conn.query_row(
                "SELECT resolved FROM approval_requests WHERE nonce = ?",
                [nonce],
                |row| row.get(0),
            ).map_err(|_| storage::sqlite::SqliteError::Rusqlite(
                rusqlite::Error::InvalidColumnName(format!("approval not found: {nonce}"))
            ))?;

            if resolved {
                return Err(storage::sqlite::SqliteError::Rusqlite(
                    rusqlite::Error::InvalidColumnName(format!("already resolved: {nonce}"))
                ));
            }

            conn.execute(
                "UPDATE approval_requests SET resolved = 1, outcome = ?, resolved_at = ? WHERE nonce = ?",
                params![outcome_json, now, nonce],
            )?;
            Ok(())
        })
        .map_err(|e| match e {
            storage::sqlite::SqliteError::Rusqlite(rusqlite::Error::InvalidColumnName(msg)) => {
                if msg.contains("not found") {
                    ConfirmError::NotFound(nonce.to_string())
                } else if msg.contains("already resolved") {
                    ConfirmError::AlreadyResolved(nonce.to_string())
                } else {
                    ConfirmError::InvalidData(msg)
                }
            }
            other => ConfirmError::Storage(other),
        })?;

        tracing::info!(nonce = %nonce, decision = %decision_str, "approval resolved");
        Ok(())
    }

    async fn status(&self, nonce: &str) -> Result<ApprovalStatus, ConfirmError> {
        self.db
            .with_conn(|conn| {
                let resolved: bool = conn
                    .query_row(
                        "SELECT resolved FROM approval_requests WHERE nonce = ?",
                        [nonce],
                        |row| row.get(0),
                    )
                    .map_err(|_| {
                        storage::sqlite::SqliteError::Rusqlite(rusqlite::Error::InvalidColumnName(
                            format!("approval not found: {nonce}"),
                        ))
                    })?;

                if resolved {
                    let outcome_json: String = conn.query_row(
                        "SELECT outcome FROM approval_requests WHERE nonce = ?",
                        [nonce],
                        |row| row.get(0),
                    )?;
                    let outcome: ApprovalOutcome =
                        serde_json::from_str(&outcome_json).map_err(|e| {
                            storage::sqlite::SqliteError::Rusqlite(
                                rusqlite::Error::InvalidColumnName(format!(
                                    "invalid outcome JSON: {e}"
                                )),
                            )
                        })?;
                    let resolved_at: String = conn.query_row(
                        "SELECT resolved_at FROM approval_requests WHERE nonce = ?",
                        [nonce],
                        |row| row.get(0),
                    )?;
                    let resolved_at = DateTime::parse_from_rfc3339(&resolved_at)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    Ok(ApprovalStatus::Resolved {
                        outcome,
                        resolved_at,
                    })
                } else {
                    let created_at: String = conn.query_row(
                        "SELECT created_at FROM approval_requests WHERE nonce = ?",
                        [nonce],
                        |row| row.get(0),
                    )?;
                    let created_at = DateTime::parse_from_rfc3339(&created_at)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now());

                    Ok(ApprovalStatus::Pending { since: created_at })
                }
            })
            .map_err(|e| match e {
                storage::sqlite::SqliteError::Rusqlite(rusqlite::Error::InvalidColumnName(msg)) => {
                    if msg.contains("not found") {
                        ConfirmError::NotFound(nonce.to_string())
                    } else {
                        ConfirmError::InvalidData(msg)
                    }
                }
                other => ConfirmError::Storage(other),
            })
    }

    async fn pending(&self) -> Result<Vec<ApprovalSpec>, ConfirmError> {
        self.db
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                "SELECT nonce, description, tier, timeout_secs, escalation, channel, alternatives
                 FROM approval_requests WHERE resolved = 0 ORDER BY created_at ASC",
            )?;
                let mut rows = stmt.query([])?;

                let mut pending = Vec::new();
                while let Some(row) = rows.next()? {
                    let nonce: String = row.get(0)?;
                    let description: String = row.get(1)?;
                    let tier_str: String = row.get(2)?;
                    let timeout_secs: i64 = row.get(3)?;
                    let channel: Option<String> = row.get(4)?;

                    let tier = match tier_str.as_str() {
                        "read" => ActionTier::Read,
                        "write" => ActionTier::Write,
                        "execute" => ActionTier::Execute,
                        "destructive" => ActionTier::Destructive,
                        "external" => ActionTier::External,
                        _ => ActionTier::Read,
                    };

                    let alternatives_json: String = row.get(5)?;
                    let alternatives: Vec<String> =
                        serde_json::from_str(&alternatives_json).unwrap_or_default();

                    pending.push(ApprovalSpec {
                        action_description: description,
                        tier,
                        nonce,
                        timeout: std::time::Duration::from_secs(timeout_secs as u64),
                        escalation: EscalationPolicy::Abort,
                        preferred_channel: channel,
                        alternatives,
                        // grant_key isn't persisted on approval_requests
                        // — the bypass decision is made at request-time
                        // from the standing_approvals table, not
                        // reconstructed from history.
                        grant_key: None,
                    });
                }
                Ok(pending)
            })
            .map_err(ConfirmError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> SqliteConfirmationEngine {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let engine = SqliteConfirmationEngine::new(pool);
        engine.ensure_tables().unwrap();
        engine
    }

    #[tokio::test]
    async fn test_auto_approve_read_tier() {
        let engine = test_engine();
        let spec = ApprovalSpec::new("read something", ActionTier::Read);
        let outcome = engine.request(spec).await.unwrap();
        assert!(matches!(outcome, ApprovalOutcome::Approved));
    }

    #[tokio::test]
    async fn test_destructive_approved_via_respond() {
        use std::sync::Arc;

        let engine = Arc::new(test_engine());
        let spec = ApprovalSpec::new("delete something", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_secs(5));
        let nonce = spec.nonce.clone();

        let engine_cloned = engine.clone();
        let nonce_cloned = nonce.clone();
        let responder = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            engine_cloned
                .respond(&nonce_cloned, ApprovalDecision::Approve)
                .await
                .unwrap();
        });

        let outcome = engine.request(spec).await.unwrap();
        responder.await.unwrap();
        assert!(matches!(outcome, ApprovalOutcome::Approved));
    }

    #[tokio::test]
    async fn test_destructive_rejected_via_respond() {
        use std::sync::Arc;

        let engine = Arc::new(test_engine());
        let spec = ApprovalSpec::new("delete something", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_secs(5));
        let nonce = spec.nonce.clone();

        let engine_cloned = engine.clone();
        let nonce_cloned = nonce.clone();
        let responder = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            engine_cloned
                .respond(&nonce_cloned, ApprovalDecision::Reject)
                .await
                .unwrap();
        });

        let outcome = engine.request(spec).await.unwrap();
        responder.await.unwrap();
        assert!(matches!(outcome, ApprovalOutcome::Rejected { .. }));
    }

    #[tokio::test]
    async fn standing_approval_bypasses_destructive_prompt() {
        use crate::standing::{GrantKey, SqliteStandingApprovals};
        use std::sync::Arc;

        let pool = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqliteStandingApprovals::new(pool.clone()));
        let key = GrantKey::new("agent-a", "fs", "write");
        store.grant(&key, Some("test")).await.unwrap();

        let engine = SqliteConfirmationEngine::new(pool).with_standing_approvals(store);
        engine.ensure_tables().unwrap();

        let spec = ApprovalSpec::new("write file", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_secs(1))
            .with_grant_key(key);

        let outcome = engine.request(spec).await.unwrap();
        assert!(
            matches!(outcome, ApprovalOutcome::Approved),
            "standing approval must bypass the destructive prompt"
        );
    }

    #[tokio::test]
    async fn standing_store_without_matching_grant_falls_through_to_timeout() {
        use crate::standing::{GrantKey, SqliteStandingApprovals};
        use std::sync::Arc;

        let pool = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqliteStandingApprovals::new(pool.clone()));
        // Grant a *different* verb so the lookup misses.
        store
            .grant(&GrantKey::new("agent-a", "fs", "read"), None)
            .await
            .unwrap();

        let engine = SqliteConfirmationEngine::new(pool).with_standing_approvals(store);
        engine.ensure_tables().unwrap();

        let spec = ApprovalSpec::new("write file", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_millis(300))
            .with_grant_key(GrantKey::new("agent-a", "fs", "write"));

        let outcome = engine.request(spec).await.unwrap();
        assert!(
            matches!(outcome, ApprovalOutcome::TimedOut),
            "missing grant must not bypass — destructive falls through to today's flow"
        );
    }

    #[tokio::test]
    async fn spec_without_grant_key_ignores_store() {
        use crate::standing::SqliteStandingApprovals;
        use std::sync::Arc;

        let pool = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqliteStandingApprovals::new(pool.clone()));
        // Even a wildcard-shaped grant can't match if the spec has no key.
        store
            .grant(
                &crate::standing::GrantKey::new("agent-a", "fs", "write"),
                None,
            )
            .await
            .unwrap();

        let engine = SqliteConfirmationEngine::new(pool).with_standing_approvals(store);
        engine.ensure_tables().unwrap();

        let spec = ApprovalSpec::new("write file", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_millis(300));
        // No `.with_grant_key(...)` — back-compat path.

        let outcome = engine.request(spec).await.unwrap();
        assert!(matches!(outcome, ApprovalOutcome::TimedOut));
    }

    #[tokio::test]
    async fn revoked_grant_does_not_bypass() {
        use crate::standing::{GrantKey, SqliteStandingApprovals};
        use std::sync::Arc;

        let pool = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqliteStandingApprovals::new(pool.clone()));
        let key = GrantKey::new("agent-a", "fs", "write");
        let id = store.grant(&key, None).await.unwrap();
        store.revoke(&id).await.unwrap();

        let engine = SqliteConfirmationEngine::new(pool).with_standing_approvals(store);
        engine.ensure_tables().unwrap();

        let spec = ApprovalSpec::new("write file", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_millis(300))
            .with_grant_key(key);

        let outcome = engine.request(spec).await.unwrap();
        assert!(matches!(outcome, ApprovalOutcome::TimedOut));
    }

    #[tokio::test]
    async fn test_destructive_times_out_without_response() {
        let engine = test_engine();
        let spec = ApprovalSpec::new("destructive action", ActionTier::Destructive)
            .with_timeout(std::time::Duration::from_millis(400));
        let nonce = spec.nonce.clone();

        let outcome = engine.request(spec).await.unwrap();
        assert!(
            matches!(outcome, ApprovalOutcome::TimedOut),
            "stub engine must NEVER auto-approve destructive actions"
        );

        // After timeout the entry is resolved.
        let status = engine.status(&nonce).await.unwrap();
        assert!(matches!(
            status,
            ApprovalStatus::Resolved {
                outcome: ApprovalOutcome::TimedOut,
                ..
            }
        ));
    }
}
