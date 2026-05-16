//! Dead-letter queue trait + in-memory implementation. Failures that
//! survive the retry budget land here so they can be replayed or
//! audited later. The `SqliteDlq` impl lives in `brainos-storage`
//! because it shares the workspace's SQLite pool; `resilience` keeps
//! only the trait and the in-memory variant so the crate's dep
//! footprint stays minimal.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Mutex;

/// One DLQ row. Mirrors the `dlq_entries` schema 1:1 so `SqliteDlq`
/// can `(entry.id, …)` map straight into a prepared insert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DlqEntry {
    pub id: String,
    pub tool_id: String,
    /// Request payload as canonical JSON — opaque to the queue but
    /// indexed by the audit replay path.
    pub request_json: String,
    pub error_message: String,
    /// Number of attempts that were burned before the failure was
    /// dead-lettered (retry budget + 1, typically).
    pub attempts: u32,
    pub dlq_at: DateTime<Utc>,
}

/// Errors any [`DeadLetterQueue`] impl can return.
#[derive(Debug, thiserror::Error)]
pub enum DlqError {
    #[error("dlq backend error: {0}")]
    Backend(String),
}

#[async_trait]
pub trait DeadLetterQueue: Send + Sync {
    /// Persist one DLQ entry.
    async fn enqueue(&self, entry: DlqEntry) -> Result<(), DlqError>;
    /// Return the `limit` most recent entries, newest first.
    async fn list_recent(&self, limit: usize) -> Result<Vec<DlqEntry>, DlqError>;
    /// Total entries currently held.
    async fn len(&self) -> Result<usize, DlqError>;
    /// Convenience derived from [`Self::len`]. Backends may override
    /// with a faster path.
    async fn is_empty(&self) -> Result<bool, DlqError> {
        Ok(self.len().await? == 0)
    }
}

/// Process-local DLQ for unit tests and dev runs without a SQLite
/// pool. Stores newest-first internally for cheap `list_recent`.
pub struct InMemoryDlq {
    entries: Mutex<Vec<DlqEntry>>,
}

impl InMemoryDlq {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }
}

impl Default for InMemoryDlq {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DeadLetterQueue for InMemoryDlq {
    async fn enqueue(&self, entry: DlqEntry) -> Result<(), DlqError> {
        let mut g = self.entries.lock().expect("dlq mutex poisoned");
        g.insert(0, entry);
        Ok(())
    }

    async fn list_recent(&self, limit: usize) -> Result<Vec<DlqEntry>, DlqError> {
        let g = self.entries.lock().expect("dlq mutex poisoned");
        Ok(g.iter().take(limit).cloned().collect())
    }

    async fn len(&self) -> Result<usize, DlqError> {
        Ok(self.entries.lock().expect("dlq mutex poisoned").len())
    }
}
