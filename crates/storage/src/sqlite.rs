//! SQLite storage backend.
//!
//! Provides connection management, schema migrations,
//! and typed CRUD operations for all Brain data:
//! - Episodes (conversations)
//! - Semantic facts (user model, extracted knowledge)
//! - Sessions (conversation grouping)

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::Connection;
use thiserror::Error;
use tracing::info;
use uuid::Uuid;

use crate::encryption::Encryptor;

/// Errors from the SQLite storage layer.
#[derive(Debug, Error)]
pub enum SqliteError {
    #[error("SQLite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("Lock poisoned")]
    LockPoisoned,

    #[error("Migration failed: {0}")]
    Migration(String),
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
/// Uses a `Mutex<Connection>` — sufficient for our single-process,
/// moderate-write workload. If we ever need concurrent writers,
/// switch to `r2d2` or WAL mode (already enabled).
///
/// When an `Encryptor` is set, `content` columns are transparently
/// encrypted on write and decrypted on read by the store layers.
#[derive(Clone)]
pub struct SqlitePool {
    conn: Arc<Mutex<Connection>>,
    encryptor: Option<Arc<Encryptor>>,
}

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

        let conn = Connection::open(path)?;

        // Performance and safety pragmas
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 5000;
            PRAGMA cache_size = -8000;
            ",
        )?;

        let pool = Self {
            conn: Arc::new(Mutex::new(conn)),
            encryptor: None,
        };

        // Run migrations
        pool.migrate()?;

        info!("SQLite database opened at {}", path.display());
        Ok(pool)
    }

    /// Open an in-memory database (for testing).
    pub fn open_memory() -> Result<Self, SqliteError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            ",
        )?;

        let pool = Self {
            conn: Arc::new(Mutex::new(conn)),
            encryptor: None,
        };

        pool.migrate()?;
        Ok(pool)
    }

    /// Execute a closure with an exclusive lock on the connection.
    pub fn with_conn<F, T>(&self, f: F) -> Result<T, SqliteError>
    where
        F: FnOnce(&Connection) -> Result<T, SqliteError>,
    {
        let conn = self.conn.lock().map_err(|_| SqliteError::LockPoisoned)?;
        f(&conn)
    }

    /// Attach an encryptor to this pool (builder pattern).
    ///
    /// Once set, `encrypt_content` / `decrypt_content` are active on all
    /// store layers that use this pool.
    pub fn with_encryptor(mut self, enc: Encryptor) -> Self {
        self.encryptor = Some(Arc::new(enc));
        self
    }

    /// Returns true if an encryptor is active.
    pub fn is_encrypted(&self) -> bool {
        self.encryptor.is_some()
    }

    /// Encrypt a string if encryption is enabled, otherwise return as-is.
    pub fn encrypt_content(&self, plaintext: &str) -> String {
        if let Some(enc) = &self.encryptor {
            enc.encrypt_string(plaintext)
                .unwrap_or_else(|_| plaintext.to_string())
        } else {
            plaintext.to_string()
        }
    }

    /// Decrypt a string if encryption is enabled.
    ///
    /// Falls back to returning the input unchanged if decryption fails
    /// (e.g. legacy plaintext rows written before encryption was enabled).
    pub fn decrypt_content(&self, maybe_ciphertext: &str) -> String {
        if let Some(enc) = &self.encryptor {
            enc.decrypt_string(maybe_ciphertext)
                .unwrap_or_else(|_| maybe_ciphertext.to_string())
        } else {
            maybe_ciphertext.to_string()
        }
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
                     WHERE namespace = ?1
                     ORDER BY created_at DESC",
                )?;
                let rows = stmt.query_map([ns], |row| {
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

    /// Run all schema migrations.
    fn migrate(&self) -> Result<(), SqliteError> {
        self.with_conn(|conn| {
            // Create migrations tracking table
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
                );",
            )?;

            let current_version: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(version), 0) FROM _migrations",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            let migrations = Self::migrations();

            for (version, name, sql) in &migrations {
                if *version > current_version {
                    info!("Running migration {version}: {name}");
                    conn.execute_batch(sql).map_err(|e| {
                        SqliteError::Migration(format!("Migration {version} ({name}) failed: {e}"))
                    })?;

                    conn.execute(
                        "INSERT INTO _migrations (version, name) VALUES (?1, ?2)",
                        rusqlite::params![version, name],
                    )?;
                }
            }

            if current_version < migrations.last().map_or(0, |m| m.0) {
                info!(
                    "Migrations complete (v{current_version} → v{})",
                    migrations.last().unwrap().0
                );
            }

            Ok(())
        })
    }

    /// All schema migrations in order.
    fn migrations() -> Vec<(i64, &'static str, &'static str)> {
        vec![
            (
                1,
                "create_sessions",
                "
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL DEFAULT (datetime('now')),
                    ended_at TEXT,
                    channel TEXT NOT NULL DEFAULT 'cli',
                    metadata TEXT
                );
            ",
            ),
            (
                2,
                "create_episodes",
                "
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    importance REAL NOT NULL DEFAULT 0.5,
                    decay_rate REAL NOT NULL DEFAULT 0.1,
                    reinforcement_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance DESC);
            ",
            ),
            (
                3,
                "create_episodes_fts",
                "
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    content,
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                );
            ",
            ),
            (
                4,
                "create_semantic_facts",
                "
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source_episode_id TEXT REFERENCES episodes(id),
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    superseded_by TEXT REFERENCES semantic_facts(id)
                );

                CREATE INDEX IF NOT EXISTS idx_facts_category ON semantic_facts(category);
                CREATE INDEX IF NOT EXISTS idx_facts_subject ON semantic_facts(subject);
            ",
            ),
            (
                5,
                "create_user_profile",
                "
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    source TEXT,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            ",
            ),
            (
                6,
                "create_procedures",
                "
                CREATE TABLE IF NOT EXISTS procedures (
                    id              TEXT PRIMARY KEY,
                    trigger_pattern TEXT NOT NULL,
                    steps_json      TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    use_count       INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_procedures_trigger
                    ON procedures(trigger_pattern);
            ",
            ),
            (
                7,
                "create_audit_log",
                "
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    action TEXT NOT NULL,
                    details TEXT,
                    prev_hash TEXT,
                    hash TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
            ",
            ),
            (
                10,
                "add_namespace_to_semantic_facts",
                "
                ALTER TABLE semantic_facts ADD COLUMN namespace TEXT NOT NULL DEFAULT 'personal';
                CREATE INDEX IF NOT EXISTS idx_facts_namespace ON semantic_facts(namespace);
            ",
            ),
            (
                11,
                "add_namespace_to_episodes",
                "
                ALTER TABLE episodes ADD COLUMN namespace TEXT NOT NULL DEFAULT 'personal';
                CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace);
            ",
            ),
            (
                12,
                "rebuild_procedures_table",
                "
                DROP TABLE IF EXISTS procedures;
                CREATE TABLE IF NOT EXISTS procedures (
                    id              TEXT PRIMARY KEY,
                    trigger_pattern TEXT NOT NULL,
                    steps_json      TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    use_count       INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_procedures_trigger
                    ON procedures(trigger_pattern);
            ",
            ),
            (
                13,
                "create_episode_promotions",
                "
                CREATE TABLE IF NOT EXISTS episode_promotions (
                    episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
                    fact_id TEXT NOT NULL REFERENCES semantic_facts(id) ON DELETE CASCADE,
                    promoted_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            ",
            ),
            (
                14,
                "create_scheduled_intents",
                "
                CREATE TABLE IF NOT EXISTS scheduled_intents (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    cron TEXT,
                    namespace TEXT NOT NULL DEFAULT 'personal',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_intents_namespace
                    ON scheduled_intents(namespace);
                CREATE INDEX IF NOT EXISTS idx_scheduled_intents_status
                    ON scheduled_intents(status);
            ",
            ),
            (
                15,
                "create_notification_outbox",
                "
                CREATE TABLE IF NOT EXISTS notification_outbox (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1,
                    triggered_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    delivered_at TEXT,
                    channel TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_pending
                    ON notification_outbox(delivered_at, priority, created_at)
                    WHERE delivered_at IS NULL;
            ",
            ),
            (
                16,
                "add_agent_column",
                "
                ALTER TABLE episodes ADD COLUMN agent TEXT;
                ALTER TABLE semantic_facts ADD COLUMN agent TEXT;
            ",
            ),
        ]
    }

    /// Get the current schema version.
    pub fn schema_version(&self) -> Result<i64, SqliteError> {
        self.with_conn(|conn| {
            let version: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(version), 0) FROM _migrations",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            Ok(version)
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

    /// Prune old delivered notifications and stale undelivered ones.
    pub fn prune_notifications(&self, max_age_days: u32) -> Result<usize, SqliteError> {
        self.with_conn(|conn| {
            let deleted = conn.execute(
                "DELETE FROM notification_outbox
                 WHERE delivered_at IS NOT NULL
                    OR created_at < datetime('now', ?1)",
                [format!("-{max_age_days} days")],
            )?;
            Ok(deleted)
        })
    }

    /// Get table row counts for status display.
    pub fn table_stats(&self) -> Result<Vec<(String, i64)>, SqliteError> {
        self.with_conn(|conn| {
            let tables = [
                "sessions",
                "episodes",
                "semantic_facts",
                "episode_promotions",
                "scheduled_intents",
                "notification_outbox",
                "user_profile",
                "procedures",
                "audit_log",
            ];

            let mut stats = Vec::new();
            for table in &tables {
                let count: i64 = conn
                    .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                        row.get(0)
                    })
                    .unwrap_or(0);
                stats.push((table.to_string(), count));
            }

            Ok(stats)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_memory() {
        let pool = SqlitePool::open_memory().unwrap();
        let version = pool.schema_version().unwrap();
        assert_eq!(version, 16); // All migrations applied
    }

    #[test]
    fn test_migrations_idempotent() {
        let pool = SqlitePool::open_memory().unwrap();
        // Running migrate again should be a no-op
        pool.migrate().unwrap();
        assert_eq!(pool.schema_version().unwrap(), 16);
    }

    #[test]
    fn test_table_stats_empty() {
        let pool = SqlitePool::open_memory().unwrap();
        let stats = pool.table_stats().unwrap();
        assert_eq!(stats.len(), 9);
        for (_, count) in &stats {
            assert_eq!(*count, 0);
        }
    }

    #[test]
    fn test_scheduled_intent_lifecycle() {
        let pool = SqlitePool::open_memory().unwrap();
        let id = pool
            .insert_scheduled_intent(
                "deploy release",
                Some("0 9 * * 1-5"),
                "work",
                Some(r#"{"source":"test"}"#),
            )
            .unwrap();

        let all = pool.list_scheduled_intents(None).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, id);
        assert_eq!(all[0].namespace, "work");
        assert_eq!(all[0].status, "scheduled");

        let personal = pool.list_scheduled_intents(Some("personal")).unwrap();
        assert!(personal.is_empty());

        let work = pool.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].description, "deploy release");
        assert_eq!(work[0].cron.as_deref(), Some("0 9 * * 1-5"));
        assert!(work[0].created_at.contains(':'));
        assert_eq!(work[0].metadata.as_deref(), Some(r#"{"source":"test"}"#));

        let updated = pool
            .update_scheduled_intent_status(&id, "cancelled")
            .unwrap();
        assert!(updated);

        let work_after = pool.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(work_after[0].status, "cancelled");
    }

    #[test]
    fn test_insert_and_query_session() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES (?1, ?2)",
                rusqlite::params!["sess001", "cli"],
            )?;

            let channel: String = conn.query_row(
                "SELECT channel FROM sessions WHERE id = ?1",
                ["sess001"],
                |row| row.get(0),
            )?;
            assert_eq!(channel, "cli");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_insert_episode_with_fk() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            // Insert session first (FK constraint)
            conn.execute("INSERT INTO sessions (id) VALUES (?1)", ["sess001"])?;

            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params!["ep001", "sess001", "user", "Hello Brain!"],
            )?;

            let content: String = conn.query_row(
                "SELECT content FROM episodes WHERE id = ?1",
                ["ep001"],
                |row| row.get(0),
            )?;
            assert_eq!(content, "Hello Brain!");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_fk_constraint_enforced() {
        let pool = SqlitePool::open_memory().unwrap();
        let result = pool.with_conn(|conn| {
            // Insert episode without session — should fail
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params!["ep001", "nonexistent", "user", "Hello"],
            )?;
            Ok(())
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_semantic_fact_insert() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["fact001", "personal", "user", "name_is", "Keshav"],
            )?;

            let obj: String = conn.query_row(
                "SELECT object FROM semantic_facts WHERE subject = ?1 AND predicate = ?2",
                rusqlite::params!["user", "name_is"],
                |row| row.get(0),
            )?;
            assert_eq!(obj, "Keshav");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_namespace_column_on_semantic_facts() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            // Insert facts in two different namespaces
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params!["factw1", "work", "user", "role_is", "developer", "work"],
            )?;
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params!["factp1", "personal", "user", "name_is", "Keshav", "personal"],
            )?;

            // Query work namespace — should only return work fact
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM semantic_facts WHERE namespace = 'work'",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(count, 1, "work namespace should have 1 fact");

            // Query personal namespace — should only return personal fact
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM semantic_facts WHERE namespace = 'personal'",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(count, 1, "personal namespace should have 1 fact");

            // Namespace isolation: work search should not return personal facts
            let found: bool = conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM semantic_facts
                     WHERE namespace = 'work' AND predicate = 'name_is'",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(false);
            assert!(!found, "work namespace must not contain personal facts");

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_namespace_default_is_personal() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            // Insert without specifying namespace — should default to 'personal'
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["factdefault", "personal", "user", "likes", "Rust"],
            )?;

            let ns: String = conn.query_row(
                "SELECT namespace FROM semantic_facts WHERE id = 'factdefault'",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(ns, "personal", "default namespace should be 'personal'");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_notification_outbox_lifecycle() {
        let pool = SqlitePool::open_memory().unwrap();

        // Insert two notifications at different priorities
        let id1 = pool
            .insert_notification("Low priority nudge", 1, "habit:morning_review", None)
            .unwrap();
        let id2 = pool
            .insert_notification("High priority reminder", 3, "open_loop:todo", Some("slack"))
            .unwrap();

        // Pending should return both, highest priority first
        let pending = pool.pending_notifications(10).unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].id, id2, "higher priority should come first");
        assert_eq!(pending[1].id, id1);
        assert!(pending[0].delivered_at.is_none());
        assert_eq!(pending[1].channel, None);
        assert_eq!(pending[0].channel.as_deref(), Some("slack"));

        // Mark one as delivered
        assert!(pool.mark_notification_delivered(&id2).unwrap());
        let pending = pool.pending_notifications(10).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, id1);

        // Idempotency: marking the same one again returns false
        assert!(!pool.mark_notification_delivered(&id2).unwrap());
    }

    #[test]
    fn test_notification_prune() {
        let pool = SqlitePool::open_memory().unwrap();
        let id = pool
            .insert_notification("test", 1, "test", None)
            .unwrap();
        pool.mark_notification_delivered(&id).unwrap();

        // Prune delivered notifications (max_age_days=0 would prune nothing recent,
        // but delivered_at IS NOT NULL clause catches delivered ones)
        let pruned = pool.prune_notifications(365).unwrap();
        assert_eq!(pruned, 1);
        assert!(pool.pending_notifications(10).unwrap().is_empty());
    }

    #[test]
    fn test_list_namespaces_with_counts() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            // Insert facts across three namespaces
            for i in 0..3 {
                conn.execute(
                    "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
                     VALUES (?1, 'personal', 'user', 'fact', ?2, 'personal')",
                    rusqlite::params![format!("p{i}"), format!("val{i}")],
                )?;
            }
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
                 VALUES ('w1', 'work', 'user', 'role', 'dev', 'work')",
                [],
            )?;

            // Count facts per namespace
            let mut stmt = conn.prepare(
                "SELECT namespace, COUNT(*) as cnt FROM semantic_facts
                 WHERE superseded_by IS NULL
                 GROUP BY namespace ORDER BY namespace",
            )?;
            let rows: Vec<(String, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            assert_eq!(rows.len(), 2, "should have 2 namespaces");
            let personal = rows.iter().find(|(ns, _)| ns == "personal").unwrap();
            assert_eq!(personal.1, 3, "personal should have 3 facts");
            let work = rows.iter().find(|(ns, _)| ns == "work").unwrap();
            assert_eq!(work.1, 1, "work should have 1 fact");

            Ok(())
        })
        .unwrap();
    }
}
