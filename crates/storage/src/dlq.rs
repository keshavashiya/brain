//! SQLite-backed [`DeadLetterQueue`] impl. Pairs with the
//! `dlq_entries` table from migration v19 and the
//! `brainos-resilience::DeadLetterQueue` trait.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use resilience::{DeadLetterQueue, DlqEntry, DlqError};

use crate::SqlitePool;

/// Persistent DLQ. Cloneable — wraps the shared pool.
#[derive(Clone)]
pub struct SqliteDlq {
    pool: Arc<SqlitePool>,
}

impl SqliteDlq {
    pub fn new(pool: Arc<SqlitePool>) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl DeadLetterQueue for SqliteDlq {
    async fn enqueue(&self, entry: DlqEntry) -> Result<(), DlqError> {
        let dlq_at = entry.dlq_at.to_rfc3339();
        self.pool
            .with_conn(|conn| {
                conn.execute(
                    "INSERT INTO dlq_entries
                       (id, tool_id, request_json, error_message, attempts, dlq_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        entry.id,
                        entry.tool_id,
                        entry.request_json,
                        entry.error_message,
                        entry.attempts,
                        dlq_at,
                    ],
                )?;
                Ok(())
            })
            .map_err(|e| DlqError::Backend(e.to_string()))
    }

    async fn list_recent(&self, limit: usize) -> Result<Vec<DlqEntry>, DlqError> {
        self.pool
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, tool_id, request_json, error_message, attempts, dlq_at
                     FROM dlq_entries
                     ORDER BY dlq_at DESC
                     LIMIT ?1",
                )?;
                let rows = stmt.query_map([limit as i64], |row| {
                    let dlq_at_str: String = row.get(5)?;
                    Ok(DlqEntryRow {
                        id: row.get(0)?,
                        tool_id: row.get(1)?,
                        request_json: row.get(2)?,
                        error_message: row.get(3)?,
                        attempts: row.get::<_, i64>(4)? as u32,
                        dlq_at: dlq_at_str,
                    })
                })?;
                let mut out = Vec::new();
                for r in rows {
                    out.push(r?);
                }
                Ok(out)
            })
            .map_err(|e| DlqError::Backend(e.to_string()))?
            .into_iter()
            .map(|r| {
                Ok(DlqEntry {
                    id: r.id,
                    tool_id: r.tool_id,
                    request_json: r.request_json,
                    error_message: r.error_message,
                    attempts: r.attempts,
                    dlq_at: parse_ts(&r.dlq_at)?,
                })
            })
            .collect()
    }

    async fn purge(&self, ids: &[String]) -> Result<usize, DlqError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let owned: Vec<String> = ids.to_vec();
        self.pool
            .with_conn(move |conn| {
                let mut removed = 0usize;
                let mut stmt = conn.prepare("DELETE FROM dlq_entries WHERE id = ?1")?;
                for id in &owned {
                    removed += stmt.execute(rusqlite::params![id])?;
                }
                Ok(removed)
            })
            .map_err(|e| DlqError::Backend(e.to_string()))
    }

    async fn len(&self) -> Result<usize, DlqError> {
        self.pool
            .with_conn(|conn| {
                let n: i64 =
                    conn.query_row("SELECT COUNT(*) FROM dlq_entries", [], |row| row.get(0))?;
                Ok(n as usize)
            })
            .map_err(|e| DlqError::Backend(e.to_string()))
    }
}

struct DlqEntryRow {
    id: String,
    tool_id: String,
    request_json: String,
    error_message: String,
    attempts: u32,
    dlq_at: String,
}

fn parse_ts(s: &str) -> Result<DateTime<Utc>, DlqError> {
    // Prefer RFC3339 (what we wrote); fall back to the SQLite
    // `datetime('now')` default ("YYYY-MM-DD HH:MM:SS") for rows
    // inserted directly via SQL or older defaults.
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(naive.and_utc());
    }
    Err(DlqError::Backend(format!("unparseable dlq_at: {s}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tool: &str, msg: &str, attempts: u32) -> DlqEntry {
        DlqEntry {
            id: uuid::Uuid::new_v4().to_string(),
            tool_id: tool.to_string(),
            request_json: r#"{"x":1}"#.to_string(),
            error_message: msg.to_string(),
            attempts,
            dlq_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn sqlite_dlq_enqueue_then_list_recent_newest_first() {
        let pool = Arc::new(SqlitePool::open_memory().expect("memory pool"));
        let dlq = SqliteDlq::new(pool);
        dlq.enqueue(entry("mcp:a:x", "first", 3)).await.unwrap();
        // Ensure ordering deterministic — second insert is strictly later.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        dlq.enqueue(entry("mcp:b:y", "second", 5)).await.unwrap();
        let recent = dlq.list_recent(10).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].error_message, "second");
        assert_eq!(recent[1].error_message, "first");
        assert_eq!(dlq.len().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn sqlite_dlq_list_recent_respects_limit() {
        let pool = Arc::new(SqlitePool::open_memory().expect("memory pool"));
        let dlq = SqliteDlq::new(pool);
        for i in 0..5 {
            dlq.enqueue(entry("t", &format!("e{i}"), 1)).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        let recent = dlq.list_recent(3).await.unwrap();
        assert_eq!(recent.len(), 3);
        // Newest-first ordering: e4, e3, e2.
        assert_eq!(recent[0].error_message, "e4");
        assert_eq!(recent[2].error_message, "e2");
    }

    #[tokio::test]
    async fn sqlite_dlq_purge_removes_only_named_ids() {
        let pool = Arc::new(SqlitePool::open_memory().expect("memory pool"));
        let dlq = SqliteDlq::new(pool);
        let a = entry("mcp:a:x", "alpha", 1);
        let b = entry("mcp:b:y", "beta", 1);
        let c = entry("mcp:c:z", "gamma", 1);
        let target = b.id.clone();
        dlq.enqueue(a).await.unwrap();
        dlq.enqueue(b).await.unwrap();
        dlq.enqueue(c).await.unwrap();
        assert_eq!(dlq.len().await.unwrap(), 3);

        let removed = dlq.purge(&[target]).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(dlq.len().await.unwrap(), 2);

        let recent = dlq.list_recent(10).await.unwrap();
        let messages: Vec<_> = recent.iter().map(|e| e.error_message.as_str()).collect();
        assert!(messages.contains(&"alpha"));
        assert!(messages.contains(&"gamma"));
        assert!(!messages.contains(&"beta"));
    }

    #[tokio::test]
    async fn sqlite_dlq_purge_empty_input_is_no_op() {
        let pool = Arc::new(SqlitePool::open_memory().expect("memory pool"));
        let dlq = SqliteDlq::new(pool);
        dlq.enqueue(entry("t", "x", 1)).await.unwrap();
        let removed = dlq.purge(&[]).await.unwrap();
        assert_eq!(removed, 0);
        assert_eq!(dlq.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_dlq_purge_unknown_id_is_no_op() {
        let pool = Arc::new(SqlitePool::open_memory().expect("memory pool"));
        let dlq = SqliteDlq::new(pool);
        dlq.enqueue(entry("t", "x", 1)).await.unwrap();
        let removed = dlq.purge(&["ghost".to_string()]).await.unwrap();
        assert_eq!(removed, 0);
        assert_eq!(dlq.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_dlq_preserves_all_fields() {
        let pool = Arc::new(SqlitePool::open_memory().expect("memory pool"));
        let dlq = SqliteDlq::new(pool);
        let e = entry("mcp:echo:echo", "boom", 7);
        let expected = e.clone();
        dlq.enqueue(e).await.unwrap();
        let recent = dlq.list_recent(1).await.unwrap();
        assert_eq!(recent.len(), 1);
        let got = &recent[0];
        assert_eq!(got.id, expected.id);
        assert_eq!(got.tool_id, expected.tool_id);
        assert_eq!(got.request_json, expected.request_json);
        assert_eq!(got.error_message, expected.error_message);
        assert_eq!(got.attempts, expected.attempts);
        // Timestamps round-trip within sub-second tolerance.
        let drift = (got.dlq_at - expected.dlq_at).num_milliseconds().abs();
        assert!(drift < 1_000, "timestamp drift {drift}ms");
    }
}
