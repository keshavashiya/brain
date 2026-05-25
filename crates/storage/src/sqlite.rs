//! SQLite storage backend.
//!
//! Provides connection management, schema migrations,
//! and typed CRUD operations for all Brain data:
//! - Episodes (conversations)
//! - Semantic facts (user model, extracted knowledge)
//! - Sessions (conversation grouping)

use std::path::Path;
#[cfg(feature = "encryption")]
use std::sync::Arc;

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Connection;
use thiserror::Error;
use tracing::info;
use uuid::Uuid;

#[cfg(feature = "encryption")]
use crate::encryption::Encryptor;

mod migrations;

#[cfg(test)]
mod tests;

/// Errors from the SQLite storage layer.
#[derive(Debug, Error)]
pub enum SqliteError {
    #[error("SQLite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("connection pool unavailable: {0}")]
    Pool(String),

    /// Retained for back-compat with downstream `match` arms. New code
    /// should emit [`SqliteError::Pool`] for connection-acquisition
    /// failures; with the r2d2 pool we no longer hold a `Mutex` that
    /// can poison. Treat any inbound `LockPoisoned` as equivalent to
    /// `Pool` for routing decisions.
    #[error("Lock poisoned")]
    LockPoisoned,

    #[error("Migration failed: {0}")]
    Migration(String),
}

impl From<r2d2::Error> for SqliteError {
    fn from(e: r2d2::Error) -> Self {
        SqliteError::Pool(e.to_string())
    }
}

/// A semantic fact for export/import operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportedFact {
    pub id: String,
    pub namespace: String,
    pub category: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source_episode_id: Option<String>,
}

/// An episodic memory entry for export/import operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportedEpisode {
    pub id: String,
    pub session_id: String,
    pub session_channel: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub importance: f64,
    pub reinforcement_count: i32,
}

fn default_namespace() -> String {
    "personal".to_string()
}

/// A notification queued for delivery to the user.
#[derive(Debug, Clone)]
pub struct Notification {
    pub id: String,
    pub content: String,
    pub priority: i32,
    pub triggered_by: String,
    pub created_at: String,
    pub delivered_at: Option<String>,
    pub channel: Option<String>,
}

/// Thread-safe SQLite connection wrapper.
///
/// Backed by an `r2d2` pool of `rusqlite::Connection`s with WAL mode
/// enabled, so concurrent reads can proceed in parallel (writes are
/// still serialized at the SQLite level). Replaces the previous
/// `Arc<Mutex<Connection>>` shape that funnelled every operation
/// through a single mutex (Wave F, Issue 68).
///
/// When an `Encryptor` is set, `content` columns are transparently
/// encrypted on write and decrypted on read by the store layers.
#[derive(Clone)]
pub struct SqlitePool {
    pool: Pool<SqliteConnectionManager>,
    #[cfg(feature = "encryption")]
    encryptor: Option<Arc<Encryptor>>,
}

/// PRAGMAs applied to every connection the pool hands out for a
/// file-backed database. `journal_mode = WAL` is the load-bearing one
/// — that's what lets the pool actually give us concurrent reads.
const FILE_PRAGMAS: &str = "
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;
    PRAGMA cache_size = -8000;
";

/// In-memory variant — no WAL filesystem to worry about, just FK +
/// foreign-keys consistency.
const MEMORY_PRAGMAS: &str = "
    PRAGMA foreign_keys = ON;
";

/// Pool size for file-backed databases. Enough connections to absorb
/// burst concurrent reads (WS streaming + reflex + HTTP at the same
/// time) without bloating memory; SQLite writes still serialize so
/// over-sizing buys nothing on the write side.
const FILE_POOL_SIZE: u32 = 8;

/// Persisted scheduling intent (persist-only mode, no internal runtime).
#[derive(Debug, Clone)]
pub struct ScheduledIntent {
    pub id: String,
    pub description: String,
    pub cron: Option<String>,
    pub namespace: String,
    pub created_at: String,
    pub status: String,
    pub metadata: Option<String>,
}

impl SqlitePool {
    /// Open a new SQLite database at the given path.
    ///
    /// - Creates the file if it doesn't exist
    /// - Enables WAL mode for concurrent reads
    /// - Enables foreign keys
    /// - Runs all schema migrations
    pub fn open(path: &Path) -> Result<Self, SqliteError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SqliteError::Migration(format!("Cannot create directory {}: {e}", parent.display()))
            })?;
        }

        let manager = SqliteConnectionManager::file(path)
            .with_init(|c: &mut Connection| c.execute_batch(FILE_PRAGMAS));
        let pool = Pool::builder().max_size(FILE_POOL_SIZE).build(manager)?;

        let p = Self {
            pool,
            #[cfg(feature = "encryption")]
            encryptor: None,
        };

        // Run migrations on one borrowed connection — DDL is persisted
        // to the file so subsequent pool checkouts see the new schema.
        p.migrate()?;

        info!(
            "SQLite database opened at {} (pool size {FILE_POOL_SIZE})",
            path.display()
        );
        Ok(p)
    }

    /// Open an in-memory database (for testing).
    ///
    /// Pool size is forced to 1 — multiple `Connection::open_in_memory`
    /// handles each get a *fresh* DB, which would break any test that
    /// writes from one checkout and reads from another. Single
    /// connection preserves the legacy single-connection semantics.
    pub fn open_memory() -> Result<Self, SqliteError> {
        let manager = SqliteConnectionManager::memory()
            .with_init(|c: &mut Connection| c.execute_batch(MEMORY_PRAGMAS));
        let pool = Pool::builder().max_size(1).build(manager)?;

        let p = Self {
            pool,
            #[cfg(feature = "encryption")]
            encryptor: None,
        };

        p.migrate()?;
        Ok(p)
    }

    /// Execute a closure with a connection borrowed from the pool.
    ///
    /// The connection returns to the pool when the closure exits.
    /// Multiple concurrent readers can hold distinct connections at
    /// the same time; writes still serialize at the SQLite level
    /// (single-writer is fundamental to SQLite, WAL or not).
    pub fn with_conn<F, T>(&self, f: F) -> Result<T, SqliteError>
    where
        F: FnOnce(&Connection) -> Result<T, SqliteError>,
    {
        let conn = self.pool.get()?;
        f(&conn)
    }

    /// Attach an encryptor to this pool (builder pattern).
    ///
    /// Once set, `encrypt_content` / `decrypt_content` are active on all
    /// store layers that use this pool.
    #[cfg(feature = "encryption")]
    pub fn with_encryptor(mut self, enc: Encryptor) -> Self {
        self.encryptor = Some(Arc::new(enc));
        self
    }

    /// Returns true if an encryptor is active.
    pub fn is_encrypted(&self) -> bool {
        #[cfg(feature = "encryption")]
        {
            self.encryptor.is_some()
        }
        #[cfg(not(feature = "encryption"))]
        {
            false
        }
    }

    /// Encrypt a string if encryption is enabled, otherwise return as-is.
    pub fn encrypt_content(&self, plaintext: &str) -> String {
        #[cfg(feature = "encryption")]
        {
            if let Some(enc) = &self.encryptor {
                return enc
                    .encrypt_string(plaintext)
                    .unwrap_or_else(|_| plaintext.to_string());
            }
        }
        plaintext.to_string()
    }

    /// Decrypt a string if encryption is enabled.
    ///
    /// Falls back to returning the input unchanged if decryption fails
    /// (e.g. legacy plaintext rows written before encryption was enabled).
    pub fn decrypt_content(&self, maybe_ciphertext: &str) -> String {
        #[cfg(feature = "encryption")]
        {
            if let Some(enc) = &self.encryptor {
                return enc
                    .decrypt_string(maybe_ciphertext)
                    .unwrap_or_else(|_| maybe_ciphertext.to_string());
            }
        }
        maybe_ciphertext.to_string()
    }

    /// Try to decrypt a string, returning `None` if decryption fails.
    ///
    /// Unlike `decrypt_content`, this does NOT fall back to returning raw
    /// ciphertext. Use this at read boundaries to filter out rows that
    /// were encrypted with a different key or are corrupted.
    pub fn try_decrypt_content(&self, maybe_ciphertext: &str) -> Option<String> {
        #[cfg(feature = "encryption")]
        {
            if let Some(enc) = &self.encryptor {
                return enc.decrypt_string(maybe_ciphertext).ok();
            }
        }
        Some(maybe_ciphertext.to_string())
    }

    /// Flush the WAL file into the main database file.
    ///
    /// Should be called on graceful shutdown to ensure all committed writes are
    /// fully persisted and the WAL file is clean. Uses `TRUNCATE` mode which
    /// also resets the WAL to zero size.
    pub fn wal_checkpoint(&self) -> Result<(), SqliteError> {
        self.with_conn(|conn| {
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
            Ok(())
        })
    }

    /// Persist a scheduled intent and return its generated ID.
    pub fn insert_scheduled_intent(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
        metadata: Option<&str>,
    ) -> Result<String, SqliteError> {
        let id = Uuid::new_v4().to_string();
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO scheduled_intents (id, description, cron, namespace, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![id, description, cron, namespace, metadata],
            )?;
            Ok(())
        })?;
        Ok(id)
    }

    /// List scheduled intents, optionally filtered by namespace.
    pub fn list_scheduled_intents(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<ScheduledIntent>, SqliteError> {
        self.with_conn(|conn| {
            let mut intents = Vec::new();
            if let Some(ns) = namespace {
                let mut stmt = conn.prepare(
                    "SELECT id, description, cron, namespace, created_at, status, metadata
                     FROM scheduled_intents
                     WHERE namespace = ?1 OR namespace LIKE ?2
                     ORDER BY created_at DESC",
                )?;
                let prefix = format!("{}/%", ns);
                let rows = stmt.query_map([ns, &prefix], |row| {
                    Ok(ScheduledIntent {
                        id: row.get(0)?,
                        description: row.get(1)?,
                        cron: row.get(2)?,
                        namespace: row.get(3)?,
                        created_at: row.get(4)?,
                        status: row.get(5)?,
                        metadata: row.get(6)?,
                    })
                })?;
                for row in rows {
                    intents.push(row?);
                }
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, description, cron, namespace, created_at, status, metadata
                     FROM scheduled_intents
                     ORDER BY created_at DESC",
                )?;
                let rows = stmt.query_map([], |row| {
                    Ok(ScheduledIntent {
                        id: row.get(0)?,
                        description: row.get(1)?,
                        cron: row.get(2)?,
                        namespace: row.get(3)?,
                        created_at: row.get(4)?,
                        status: row.get(5)?,
                        metadata: row.get(6)?,
                    })
                })?;
                for row in rows {
                    intents.push(row?);
                }
            }
            Ok(intents)
        })
    }

    /// Update a scheduled intent status. Returns true when a row was updated.
    pub fn update_scheduled_intent_status(
        &self,
        id: &str,
        status: &str,
    ) -> Result<bool, SqliteError> {
        self.with_conn(|conn| {
            let affected = conn.execute(
                "UPDATE scheduled_intents SET status = ?2 WHERE id = ?1",
                rusqlite::params![id, status],
            )?;
            Ok(affected > 0)
        })
    }

    /// Cancel a scheduled intent (set status to "cancelled").
    pub fn cancel_scheduled_intent(&self, id: &str) -> Result<bool, SqliteError> {
        self.update_scheduled_intent_status(id, "cancelled")
    }

    /// Return all scheduled intents with status `"scheduled"` (i.e. pending execution).
    pub fn due_scheduled_intents(&self) -> Result<Vec<ScheduledIntent>, SqliteError> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, description, cron, namespace, created_at, status, metadata
                 FROM scheduled_intents
                 WHERE status = 'scheduled'
                 ORDER BY created_at ASC",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok(ScheduledIntent {
                    id: row.get(0)?,
                    description: row.get(1)?,
                    cron: row.get(2)?,
                    namespace: row.get(3)?,
                    created_at: row.get(4)?,
                    status: row.get(5)?,
                    metadata: row.get(6)?,
                })
            })?;
            Ok(rows.filter_map(|r| r.ok()).collect())
        })
    }

    /// Insert a notification into the outbox for later delivery.
    pub fn insert_notification(
        &self,
        content: &str,
        priority: i32,
        triggered_by: &str,
        channel: Option<&str>,
    ) -> Result<String, SqliteError> {
        let id = Uuid::new_v4().to_string();
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO notification_outbox (id, content, priority, triggered_by, channel)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![id, content, priority, triggered_by, channel],
            )?;
            Ok(())
        })?;
        Ok(id)
    }

    /// Fetch all pending (undelivered) notifications, ordered by priority then age.
    pub fn pending_notifications(&self, limit: usize) -> Result<Vec<Notification>, SqliteError> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, content, priority, triggered_by, created_at, delivered_at, channel
                 FROM notification_outbox
                 WHERE delivered_at IS NULL
                 ORDER BY priority DESC, created_at ASC
                 LIMIT ?1",
            )?;
            let rows = stmt
                .query_map([limit as i64], |row| {
                    Ok(Notification {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        priority: row.get(2)?,
                        triggered_by: row.get(3)?,
                        created_at: row.get(4)?,
                        delivered_at: row.get(5)?,
                        channel: row.get(6)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Mark a notification as delivered (sets `delivered_at` to now).
    pub fn mark_notification_delivered(&self, id: &str) -> Result<bool, SqliteError> {
        self.with_conn(|conn| {
            let affected = conn.execute(
                "UPDATE notification_outbox SET delivered_at = datetime('now') WHERE id = ?1 AND delivered_at IS NULL",
                [id],
            )?;
            Ok(affected > 0)
        })
    }

    /// Mark multiple notifications as delivered in a single UPDATE.
    /// Returns the count of notifications actually marked delivered.
    pub fn mark_notifications_delivered(&self, ids: &[String]) -> Result<usize, SqliteError> {
        if ids.is_empty() {
            return Ok(0);
        }
        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "UPDATE notification_outbox SET delivered_at = datetime('now') \
             WHERE delivered_at IS NULL AND id IN ({placeholders})"
        );
        self.with_conn(|conn| {
            let params: Vec<&dyn rusqlite::types::ToSql> = ids
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            let affected = conn.execute(&sql, params.as_slice())?;
            Ok(affected)
        })
    }

    /// Prune old delivered notifications and stale undelivered ones.
    pub fn prune_notifications(&self, max_age_days: u32) -> Result<usize, SqliteError> {
        self.with_conn(|conn| {
            let deleted = conn.execute(
                "DELETE FROM notification_outbox
                 WHERE (delivered_at IS NOT NULL AND created_at < datetime('now', ?1))
                    OR created_at < datetime('now', ?1)",
                [format!("-{max_age_days} days")],
            )?;
            Ok(deleted)
        })
    }

    // ── Export / Import ──────────────────────────────────────────────────────

    /// Export all semantic facts ordered by ID.
    pub fn export_all_facts(&self) -> Result<Vec<ExportedFact>, SqliteError> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, namespace, category, subject, predicate, object,
                        confidence, source_episode_id
                 FROM semantic_facts
                 ORDER BY id ASC",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    Ok(ExportedFact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: row.get(5)?,
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Export all episodes with session info, ordered by timestamp.
    pub fn export_all_episodes(&self) -> Result<Vec<ExportedEpisode>, SqliteError> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT e.id, e.session_id, COALESCE(s.channel, 'cli'),
                        e.namespace, e.role, e.content, e.timestamp,
                        e.importance, e.reinforcement_count
                 FROM episodes e
                 LEFT JOIN sessions s ON s.id = e.session_id
                 ORDER BY e.timestamp ASC",
            )?;
            let rows = stmt
                .query_map([], |row| {
                    Ok(ExportedEpisode {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        session_channel: row.get(2)?,
                        namespace: row.get(3)?,
                        role: row.get(4)?,
                        content: row.get(5)?,
                        timestamp: row.get(6)?,
                        importance: row.get(7)?,
                        reinforcement_count: row.get(8)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })
    }

    /// Import facts (ON CONFLICT DO NOTHING). Returns (imported_count, new_indices).
    pub fn import_facts(&self, facts: &[ExportedFact]) -> Result<(usize, Vec<usize>), SqliteError> {
        self.with_conn(|conn| {
            let mut imported = 0usize;
            let mut new_indices = Vec::new();
            for (idx, f) in facts.iter().enumerate() {
                let n = conn.execute(
                    "INSERT INTO semantic_facts
                        (id, namespace, category, subject, predicate, object,
                         confidence, source_episode_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                     ON CONFLICT(id) DO NOTHING",
                    rusqlite::params![
                        f.id,
                        f.namespace,
                        f.category,
                        f.subject,
                        f.predicate,
                        f.object,
                        f.confidence,
                        f.source_episode_id
                    ],
                )?;
                if n > 0 {
                    new_indices.push(idx);
                }
                imported += n;
            }
            Ok((imported, new_indices))
        })
    }

    /// Import episodes (ON CONFLICT DO NOTHING). Returns count of newly imported episodes.
    pub fn import_episodes(&self, episodes: &[ExportedEpisode]) -> Result<usize, SqliteError> {
        self.with_conn(|conn| {
            // Ensure sessions exist first
            let mut sessions: std::collections::HashMap<String, String> =
                std::collections::HashMap::new();
            for ep in episodes {
                sessions
                    .entry(ep.session_id.clone())
                    .or_insert_with(|| ep.session_channel.clone());
            }
            for (sid, channel) in &sessions {
                conn.execute(
                    "INSERT INTO sessions (id, channel) VALUES (?1, ?2)
                     ON CONFLICT(id) DO NOTHING",
                    rusqlite::params![sid, channel],
                )?;
            }

            let mut imported = 0usize;
            for e in episodes {
                let n = conn.execute(
                    "INSERT INTO episodes
                        (id, session_id, namespace, role, content, timestamp,
                         importance, reinforcement_count)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                     ON CONFLICT(id) DO NOTHING",
                    rusqlite::params![
                        e.id,
                        e.session_id,
                        e.namespace,
                        e.role,
                        e.content,
                        e.timestamp,
                        e.importance,
                        e.reinforcement_count
                    ],
                )?;
                imported += n;
            }
            Ok(imported)
        })
    }

    /// Get table row counts for status display.
    pub fn table_stats(&self) -> Result<Vec<(String, i64)>, SqliteError> {
        self.with_conn(|conn| {
            let mut stats = Vec::new();
            // Whitelist approach: each match arm is both the table name and its SQL,
            // preventing any possibility of SQL injection via table name interpolation.
            for table in &[
                "sessions",
                "episodes",
                "semantic_facts",
                "episode_promotions",
                "scheduled_intents",
                "notification_outbox",
                "user_profile",
                "procedures",
                "audit_log",
            ] {
                let sql = match *table {
                    "sessions" => "SELECT COUNT(*) FROM sessions",
                    "episodes" => "SELECT COUNT(*) FROM episodes",
                    "semantic_facts" => "SELECT COUNT(*) FROM semantic_facts",
                    "episode_promotions" => "SELECT COUNT(*) FROM episode_promotions",
                    "scheduled_intents" => "SELECT COUNT(*) FROM scheduled_intents",
                    "notification_outbox" => "SELECT COUNT(*) FROM notification_outbox",
                    "user_profile" => "SELECT COUNT(*) FROM user_profile",
                    "procedures" => "SELECT COUNT(*) FROM procedures",
                    "audit_log" => "SELECT COUNT(*) FROM audit_log",
                    _ => continue,
                };
                let count: i64 = conn.query_row(sql, [], |row| row.get(0)).unwrap_or(0);
                stats.push((table.to_string(), count));
            }

            Ok(stats)
        })
    }
}
