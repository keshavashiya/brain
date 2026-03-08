//! # Brain Cerebellum
//!
//! Procedure store — learned workflows and automation.
//!
//! Stores `(trigger_pattern, steps)` tuples in SQLite.  Incoming signals are
//! matched against known triggers; matching procedures contribute additional
//! context steps to the LLM prompt so Brain can auto-execute stored workflows.

use chrono::Utc;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use storage::SqlitePool;
use thiserror::Error;
use uuid::Uuid;

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum CerebellumError {
    #[error("Storage error: {0}")]
    Storage(#[from] storage::sqlite::SqliteError),

    #[error("Procedure not found: {0}")]
    NotFound(String),

    #[error("Invalid steps: {0}")]
    InvalidSteps(String),
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// A stored procedure — a trigger keyword / phrase mapped to a sequence of steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    /// Unique identifier.
    pub id: String,
    /// Trigger pattern: a keyword or short phrase that activates this procedure.
    pub trigger_pattern: String,
    /// Ordered list of action/context steps to include when the trigger fires.
    pub steps: Vec<String>,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// ISO 8601 last-updated timestamp.
    pub updated_at: String,
    /// How many times this procedure has been invoked.
    pub use_count: i64,
}

// ─── ProcedureStore ───────────────────────────────────────────────────────────

/// Stores and retrieves learned procedures backed by SQLite.
pub struct ProcedureStore {
    db: SqlitePool,
}

impl ProcedureStore {
    /// Create a new procedure store backed by the given SQLite pool.
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }

    /// Ensure the `procedures` table exists (idempotent).
    ///
    /// The table is created by the storage migration layer (`SqlitePool::migrate`).
    /// This method is a no-op when called on a fully-migrated pool, but kept as
    /// a public API so callers can safely call it without knowing migration state.
    pub fn ensure_tables(&self) -> Result<(), CerebellumError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS procedures (
                    id              TEXT PRIMARY KEY,
                    trigger_pattern TEXT NOT NULL,
                    steps_json      TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    use_count       INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_procedures_trigger
                    ON procedures(trigger_pattern);",
            )?;
            Ok(())
        })?;
        Ok(())
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Store a new procedure.
    ///
    /// `trigger` is a keyword/phrase that activates the procedure.
    /// `steps` is an ordered list of action descriptions.
    /// Returns the new procedure ID.
    pub fn store_procedure(
        &self,
        trigger: &str,
        steps: &[String],
    ) -> Result<String, CerebellumError> {
        if steps.is_empty() {
            return Err(CerebellumError::InvalidSteps(
                "Procedure must have at least one step".to_string(),
            ));
        }

        let id = Uuid::new_v4().to_string();
        let steps_json = serde_json::to_string(steps)
            .map_err(|e| CerebellumError::InvalidSteps(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO procedures (id, trigger_pattern, steps_json, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?4)",
                rusqlite::params![id, trigger, steps_json, now],
            )?;
            Ok(())
        })?;

        tracing::info!(trigger, id, "Procedure stored");
        Ok(id)
    }

    // ── Match ─────────────────────────────────────────────────────────────────

    /// Find all procedures whose trigger pattern appears in `input` (case-insensitive substring match).
    pub fn match_trigger(&self, input: &str) -> Result<Vec<Procedure>, CerebellumError> {
        let lower = input.to_lowercase();
        let all = self.list_procedures()?;
        Ok(all
            .into_iter()
            .filter(|p| lower.contains(&p.trigger_pattern.to_lowercase()))
            .collect())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Get a procedure by ID.
    pub fn get_procedure(&self, id: &str) -> Result<Procedure, CerebellumError> {
        let result = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT id, trigger_pattern, steps_json, created_at, updated_at, use_count
                 FROM procedures WHERE id = ?1",
                [id],
                row_to_procedure,
            )
            .optional()
            .map_err(|e| e.into())
        })?;
        result.ok_or_else(|| CerebellumError::NotFound(id.to_string()))
    }

    /// List all stored procedures ordered by trigger pattern.
    pub fn list_procedures(&self) -> Result<Vec<Procedure>, CerebellumError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, trigger_pattern, steps_json, created_at, updated_at, use_count
                 FROM procedures
                 ORDER BY trigger_pattern ASC",
            )?;
            let rows = stmt
                .query_map([], row_to_procedure)?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?)
    }

    // ── Mutate ────────────────────────────────────────────────────────────────

    /// Replace the steps of an existing procedure.
    pub fn update_steps(&self, id: &str, new_steps: &[String]) -> Result<(), CerebellumError> {
        if new_steps.is_empty() {
            return Err(CerebellumError::InvalidSteps(
                "Procedure must have at least one step".to_string(),
            ));
        }
        let steps_json = serde_json::to_string(new_steps)
            .map_err(|e| CerebellumError::InvalidSteps(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        let rows = self.db.with_conn(|conn| {
            let n = conn.execute(
                "UPDATE procedures SET steps_json = ?1, updated_at = ?2 WHERE id = ?3",
                rusqlite::params![steps_json, now, id],
            )?;
            Ok(n)
        })?;

        if rows == 0 {
            return Err(CerebellumError::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// Delete a procedure by ID.
    pub fn delete_procedure(&self, id: &str) -> Result<(), CerebellumError> {
        let rows = self.db.with_conn(|conn| {
            let n = conn.execute("DELETE FROM procedures WHERE id = ?1", [id])?;
            Ok(n)
        })?;
        if rows == 0 {
            return Err(CerebellumError::NotFound(id.to_string()));
        }
        tracing::info!(id, "Procedure deleted");
        Ok(())
    }

    /// Increment the use counter for a procedure (called each time it fires).
    pub fn record_execution(&self, id: &str) -> Result<(), CerebellumError> {
        self.db.with_conn(|conn| {
            conn.execute(
                "UPDATE procedures SET use_count = use_count + 1 WHERE id = ?1",
                [id],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Total number of stored procedures.
    pub fn count(&self) -> Result<i64, CerebellumError> {
        Ok(self.db.with_conn(|conn| {
            let n: i64 = conn.query_row("SELECT COUNT(*) FROM procedures", [], |row| row.get(0))?;
            Ok(n)
        })?)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn row_to_procedure(row: &rusqlite::Row<'_>) -> rusqlite::Result<Procedure> {
    let steps_json: String = row.get(2)?;
    let steps: Vec<String> = serde_json::from_str(&steps_json).unwrap_or_default();
    Ok(Procedure {
        id: row.get(0)?,
        trigger_pattern: row.get(1)?,
        steps,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        use_count: row.get(5)?,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> ProcedureStore {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let store = ProcedureStore::new(pool);
        store.ensure_tables().unwrap();
        store
    }

    #[test]
    fn test_ensure_tables_idempotent() {
        let store = test_store();
        store.ensure_tables().unwrap();
    }

    #[test]
    fn test_store_and_get() {
        let store = test_store();
        let id = store
            .store_procedure("standup", &["check PRs".into(), "review CI".into()])
            .unwrap();
        let proc = store.get_procedure(&id).unwrap();
        assert_eq!(proc.trigger_pattern, "standup");
        assert_eq!(proc.steps, vec!["check PRs", "review CI"]);
        assert_eq!(proc.use_count, 0);
    }

    #[test]
    fn test_empty_steps_rejected() {
        let store = test_store();
        let result = store.store_procedure("trigger", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_procedures() {
        let store = test_store();
        assert_eq!(store.count().unwrap(), 0);
        store
            .store_procedure("deploy", &["run tests".into(), "push to main".into()])
            .unwrap();
        store
            .store_procedure("review", &["open PR".into()])
            .unwrap();
        let procs = store.list_procedures().unwrap();
        assert_eq!(procs.len(), 2);
        // Ordered by trigger ASC
        assert_eq!(procs[0].trigger_pattern, "deploy");
        assert_eq!(procs[1].trigger_pattern, "review");
    }

    #[test]
    fn test_match_trigger_case_insensitive() {
        let store = test_store();
        store
            .store_procedure("standup", &["check Slack".into()])
            .unwrap();

        let matches = store.match_trigger("Doing my morning Standup").unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].trigger_pattern, "standup");
    }

    #[test]
    fn test_match_trigger_no_match() {
        let store = test_store();
        store
            .store_procedure("deploy", &["run tests".into()])
            .unwrap();
        let matches = store.match_trigger("What is the weather today?").unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_delete_procedure() {
        let store = test_store();
        let id = store
            .store_procedure("cleanup", &["archive logs".into()])
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);
        store.delete_procedure(&id).unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn test_delete_missing_returns_not_found() {
        let store = test_store();
        let result = store.delete_procedure("nonexistent-id");
        assert!(matches!(result, Err(CerebellumError::NotFound(_))));
    }

    #[test]
    fn test_record_execution_increments_count() {
        let store = test_store();
        let id = store.store_procedure("test", &["step 1".into()]).unwrap();
        store.record_execution(&id).unwrap();
        store.record_execution(&id).unwrap();
        let proc = store.get_procedure(&id).unwrap();
        assert_eq!(proc.use_count, 2);
    }

    #[test]
    fn test_update_steps() {
        let store = test_store();
        let id = store.store_procedure("old", &["step 1".into()]).unwrap();
        store
            .update_steps(&id, &["step A".into(), "step B".into()])
            .unwrap();
        let proc = store.get_procedure(&id).unwrap();
        assert_eq!(proc.steps, vec!["step A", "step B"]);
    }

    #[test]
    fn test_multiple_triggers_in_input() {
        let store = test_store();
        store.store_procedure("PR", &["check CI".into()]).unwrap();
        store
            .store_procedure("deploy", &["run tests".into()])
            .unwrap();

        // Both triggers appear in the input
        let matches = store
            .match_trigger("Review PR and then deploy to staging")
            .unwrap();
        assert_eq!(matches.len(), 2);
    }
}
