//! Audit trail schema and core trait.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use storage::SqlitePool;
use thiserror::Error;
use tracing;
use uuid::Uuid;

use super::query::AuditQuerySpec;
use super::rollback::RollbackPlan;

/// Action tier determines confirmation requirement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionTier {
    /// Read-only — never requires confirmation
    Read,
    /// Write — implicit confirmation, user can undo
    Write,
    /// Execute — sandboxed, reversible
    Execute,
    /// Destructive — explicit approval required
    Destructive,
    /// External — explicit approval + credential audit
    External,
}

impl std::fmt::Display for ActionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionTier::Read => write!(f, "read"),
            ActionTier::Write => write!(f, "write"),
            ActionTier::Execute => write!(f, "execute"),
            ActionTier::Destructive => write!(f, "destructive"),
            ActionTier::External => write!(f, "external"),
        }
    }
}

/// Outcome of an executed action.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Success,
    Failure,
    Cancelled,
    Timeout,
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditOutcome::Success => write!(f, "success"),
            AuditOutcome::Failure => write!(f, "failure"),
            AuditOutcome::Cancelled => write!(f, "cancelled"),
            AuditOutcome::Timeout => write!(f, "timeout"),
        }
    }
}

/// A single audit entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: String,
    pub source: String,   // "user", "ganglia", "orchestrator", "system"
    pub request: String,  // What was requested
    pub decision: String, // How it was decided
    pub action: String,   // What was executed
    pub tier: ActionTier,
    pub approved_by: Option<String>, // "user", "auto", "timeout_escalated"
    pub approval_nonce: Option<String>,
    pub stdout: Option<String>, // Truncated to 4KB
    pub stderr: Option<String>, // Truncated to 4KB
    pub exit_code: Option<i32>,
    pub duration_ms: Option<i64>,
    pub outcome: AuditOutcome,
    pub rollback: Option<String>, // Rollback coordinates (JSON)
    pub metadata: Option<String>, // Additional context (JSON)
}

impl AuditEntry {
    pub fn new(
        request: impl Into<String>,
        decision: impl Into<String>,
        action: impl Into<String>,
        tier: ActionTier,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now().to_rfc3339(),
            source: "system".to_string(),
            request: request.into(),
            decision: decision.into(),
            action: action.into(),
            tier,
            approved_by: None,
            approval_nonce: None,
            stdout: None,
            stderr: None,
            exit_code: None,
            duration_ms: None,
            outcome: AuditOutcome::Success,
            rollback: None,
            metadata: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn with_approval(mut self, approved_by: impl Into<String>, nonce: Option<String>) -> Self {
        self.approved_by = Some(approved_by.into());
        self.approval_nonce = nonce;
        self
    }

    pub fn with_execution(
        mut self,
        stdout: String,
        stderr: String,
        exit_code: i32,
        duration_ms: i64,
    ) -> Self {
        const MAX_OUTPUT: usize = 4096;
        self.stdout = Some(stdout.chars().take(MAX_OUTPUT).collect());
        self.stderr = Some(stderr.chars().take(MAX_OUTPUT).collect());
        self.exit_code = Some(exit_code);
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    pub fn with_rollback(mut self, rollback: RollbackPlan) -> Self {
        self.rollback = Some(serde_json::to_string(&rollback).unwrap_or_default());
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(serde_json::to_string(&metadata).unwrap_or_default());
        self
    }
}

#[derive(Debug, Error)]
pub enum AuditError {
    #[error("Storage error: {0}")]
    Storage(#[from] storage::sqlite::SqliteError),
    #[error("Entry not found: {0}")]
    NotFound(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Audit entries are immutable: {0}")]
    ImmutableViolation(String),
}

/// Immutable audit trail.
#[async_trait]
pub trait AuditTrail: Send + Sync {
    /// Record an audit entry. Returns the entry ID.
    async fn record(&self, entry: AuditEntry) -> Result<String, AuditError>;

    /// Query audit entries by specification.
    async fn query(&self, spec: AuditQuerySpec) -> Result<Vec<AuditEntry>, AuditError>;

    /// Summarize audit entries for a time window.
    async fn summarize(&self, window: chrono::Duration) -> Result<AuditSummary, AuditError>;

    /// Get rollback plan for an entry (if available).
    async fn rollback(&self, entry_id: &str) -> Result<Option<RollbackPlan>, AuditError>;

    /// Prune entries older than the given duration. Returns count pruned.
    async fn prune(&self, older_than: chrono::Duration) -> Result<usize, AuditError>;
}

/// Summary of audit activity for a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total_entries: usize,
    pub by_outcome: std::collections::HashMap<String, usize>,
    pub by_tier: std::collections::HashMap<String, usize>,
    pub by_source: std::collections::HashMap<String, usize>,
    pub avg_duration_ms: Option<f64>,
}

/// SQLite-backed audit trail implementation.
pub struct SqliteAuditTrail {
    db: SqlitePool,
}

impl SqliteAuditTrail {
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }

    pub fn ensure_tables(&self) -> Result<(), AuditError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS audit_entries (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    request     TEXT NOT NULL,
                    decision    TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    tier        TEXT NOT NULL,
                    approved_by TEXT,
                    approval_nonce TEXT,
                    stdout      TEXT,
                    stderr      TEXT,
                    exit_code   INTEGER,
                    duration_ms INTEGER,
                    outcome     TEXT NOT NULL,
                    rollback    TEXT,
                    metadata    TEXT
                );

                CREATE TRIGGER IF NOT EXISTS audit_no_update
                    BEFORE UPDATE ON audit_entries
                    BEGIN
                        SELECT RAISE(ABORT, 'audit entries are immutable');
                    END;

                CREATE TRIGGER IF NOT EXISTS audit_no_delete
                    BEFORE DELETE ON audit_entries
                    BEGIN
                        SELECT RAISE(ABORT, 'audit entries are immutable');
                    END;
                "#,
            )?;
            Ok(())
        })?;
        Ok(())
    }

    fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<AuditEntry> {
        let tier_str: String = row.get(6)?;
        let tier = match tier_str.as_str() {
            "read" => ActionTier::Read,
            "write" => ActionTier::Write,
            "execute" => ActionTier::Execute,
            "destructive" => ActionTier::Destructive,
            "external" => ActionTier::External,
            other => {
                return Err(rusqlite::Error::InvalidColumnName(format!(
                    "unknown tier: {other}"
                )))
            }
        };

        let outcome_str: String = row.get(14)?;
        let outcome = match outcome_str.as_str() {
            "success" => AuditOutcome::Success,
            "failure" => AuditOutcome::Failure,
            "cancelled" => AuditOutcome::Cancelled,
            "timeout" => AuditOutcome::Timeout,
            other => {
                return Err(rusqlite::Error::InvalidColumnName(format!(
                    "unknown outcome: {other}"
                )))
            }
        };

        Ok(AuditEntry {
            id: row.get(0)?,
            timestamp: row.get(1)?,
            source: row.get(2)?,
            request: row.get(3)?,
            decision: row.get(4)?,
            action: row.get(5)?,
            tier,
            approved_by: row.get::<_, Option<String>>(7)?.filter(|s| !s.is_empty()),
            approval_nonce: row.get::<_, Option<String>>(8)?.filter(|s| !s.is_empty()),
            stdout: row.get::<_, Option<String>>(9)?.filter(|s| !s.is_empty()),
            stderr: row.get::<_, Option<String>>(10)?.filter(|s| !s.is_empty()),
            exit_code: row.get::<_, Option<i32>>(11)?,
            duration_ms: row.get::<_, Option<i64>>(12)?,
            outcome,
            rollback: row.get::<_, Option<String>>(13)?.filter(|s| !s.is_empty()),
            metadata: row.get::<_, Option<String>>(15)?.filter(|s| !s.is_empty()),
        })
    }
}

#[async_trait]
impl AuditTrail for SqliteAuditTrail {
    async fn record(&self, entry: AuditEntry) -> Result<String, AuditError> {
        let id = entry.id.clone();
        let entry = Arc::new(entry);

        let entry = Arc::clone(&entry);
        self.db.with_conn(|conn| {
            conn.execute(
                r#"INSERT INTO audit_entries (
                    id, timestamp, source, request, decision, action, tier,
                    approved_by, approval_nonce, stdout, stderr, exit_code,
                    duration_ms, outcome, rollback, metadata
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)"#,
                params![
                    entry.id,
                    entry.timestamp,
                    entry.source,
                    entry.request,
                    entry.decision,
                    entry.action,
                    entry.tier.to_string(),
                    entry.approved_by,
                    entry.approval_nonce,
                    entry.stdout,
                    entry.stderr,
                    entry.exit_code,
                    entry.duration_ms,
                    entry.outcome.to_string(),
                    entry.rollback,
                    entry.metadata,
                ],
            )?;
            Ok(())
        })?;

        tracing::info!(id = %id, tier = %entry.tier, outcome = %entry.outcome, "audit entry recorded");
        Ok(id)
    }

    async fn query(&self, spec: AuditQuerySpec) -> Result<Vec<AuditEntry>, AuditError> {
        self.db
            .with_conn(|conn| {
                let mut sql = String::from(
                    "SELECT id, timestamp, source, request, decision, action, tier,
                        approved_by, approval_nonce, stdout, stderr, exit_code,
                        duration_ms, rollback, outcome, metadata
                 FROM audit_entries WHERE 1=1",
                );
                let mut param_values: Vec<String> = Vec::new();

                if let Some(since) = spec.since {
                    sql.push_str(" AND timestamp >= ?");
                    param_values.push(since.to_rfc3339());
                }
                if let Some(before) = spec.before {
                    sql.push_str(" AND timestamp < ?");
                    param_values.push(before.to_rfc3339());
                }
                if let Some(ref source) = spec.source {
                    sql.push_str(" AND source = ?");
                    param_values.push(source.clone());
                }
                if let Some(ref tier) = spec.tier {
                    sql.push_str(" AND tier = ?");
                    param_values.push(tier.to_string());
                }
                if let Some(ref outcome) = spec.outcome {
                    sql.push_str(" AND outcome = ?");
                    param_values.push(outcome.to_string());
                }
                if let Some(limit) = spec.limit {
                    sql.push_str(&format!(" LIMIT {limit}"));
                }
                sql.push_str(" ORDER BY timestamp DESC");

                let mut stmt = conn.prepare(&sql)?;
                let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values
                    .iter()
                    .map(|s| s as &dyn rusqlite::types::ToSql)
                    .collect();
                let mut rows = stmt.query(&param_refs[..])?;

                let mut entries = Vec::new();
                while let Some(row) = rows.next()? {
                    entries.push(Self::row_to_entry(row)?);
                }
                Ok(entries)
            })
            .map_err(AuditError::from)
    }

    async fn summarize(&self, window: chrono::Duration) -> Result<AuditSummary, AuditError> {
        self.db.with_conn(|conn| {
            let since = (Utc::now() - window).to_rfc3339();

            let total: i64 = conn.query_row(
                "SELECT COUNT(*) FROM audit_entries WHERE timestamp >= ?",
                [&since],
                |row| row.get(0),
            )?;

            let mut by_outcome = std::collections::HashMap::new();
            let mut stmt = conn.prepare(
                "SELECT outcome, COUNT(*) FROM audit_entries WHERE timestamp >= ? GROUP BY outcome",
            )?;
            let mut rows = stmt.query([&since])?;
            while let Some(row) = rows.next()? {
                let outcome: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                by_outcome.insert(outcome, count as usize);
            }

            let mut by_tier = std::collections::HashMap::new();
            let mut stmt = conn.prepare(
                "SELECT tier, COUNT(*) FROM audit_entries WHERE timestamp >= ? GROUP BY tier",
            )?;
            let mut rows = stmt.query([&since])?;
            while let Some(row) = rows.next()? {
                let tier: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                by_tier.insert(tier, count as usize);
            }

            let mut by_source = std::collections::HashMap::new();
            let mut stmt = conn.prepare(
                "SELECT source, COUNT(*) FROM audit_entries WHERE timestamp >= ? GROUP BY source",
            )?;
            let mut rows = stmt.query([&since])?;
            while let Some(row) = rows.next()? {
                let source: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                by_source.insert(source, count as usize);
            }

            let avg_duration_ms: Option<f64> = conn.query_row(
                "SELECT AVG(duration_ms) FROM audit_entries WHERE timestamp >= ? AND duration_ms IS NOT NULL",
                [&since],
                |row| row.get(0),
            ).ok();

            Ok(AuditSummary {
                total_entries: total as usize,
                by_outcome,
                by_tier,
                by_source,
                avg_duration_ms,
            })
        })
        .map_err(AuditError::from)
    }

    async fn rollback(&self, entry_id: &str) -> Result<Option<RollbackPlan>, AuditError> {
        self.db
            .with_conn(|conn| {
                let rollback_json: Option<String> = conn.query_row(
                    "SELECT rollback FROM audit_entries WHERE id = ?",
                    [entry_id],
                    |row| row.get(0),
                )?;

                match rollback_json {
                    Some(json) => serde_json::from_str(&json).map_err(|e| {
                        storage::sqlite::SqliteError::Rusqlite(rusqlite::Error::InvalidColumnName(
                            format!("invalid rollback JSON: {e}"),
                        ))
                    }),
                    None => Ok(None),
                }
            })
            .map_err(|e| match e {
                storage::sqlite::SqliteError::Rusqlite(rusqlite::Error::InvalidColumnName(msg)) => {
                    AuditError::InvalidData(msg)
                }
                other => AuditError::Storage(other),
            })
    }

    async fn prune(&self, _older_than: chrono::Duration) -> Result<usize, AuditError> {
        // Pruning requires bypassing the immutable trigger — we use a direct SQL
        // approach since the trigger prevents DELETE.
        // For now, return 0 — full implementation needs trigger bypass.
        tracing::warn!("audit prune not yet fully implemented");
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_trail() -> SqliteAuditTrail {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let trail = SqliteAuditTrail::new(pool);
        trail.ensure_tables().unwrap();
        trail
    }

    #[tokio::test]
    async fn test_record_and_query() {
        let trail = test_trail();
        let entry = AuditEntry::new(
            "test request",
            "test decision",
            "test action",
            ActionTier::Execute,
        );
        let id = trail.record(entry).await.unwrap();

        let results = trail.query(AuditQuerySpec::default()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[tokio::test]
    async fn test_immutability() {
        let trail = test_trail();
        let entry = AuditEntry::new("test", "decision", "action", ActionTier::Read);
        let id = trail.record(entry).await.unwrap();

        // Attempting to update should fail — tested via direct SQL
        let result = trail.db.with_conn(|conn| {
            Ok(conn.execute(
                "UPDATE audit_entries SET request = 'hacked' WHERE id = ?",
                [&id],
            )?)
        });
        assert!(result.is_err());
    }
}
