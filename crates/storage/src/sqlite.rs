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

/// Thread-safe SQLite connection wrapper.
///
/// Uses a `Mutex<Connection>` — sufficient for our single-process,
/// moderate-write workload. If we ever need concurrent writers,
/// switch to `r2d2` or WAL mode (already enabled).
#[derive(Clone)]
pub struct SqlitePool {
    conn: Arc<Mutex<Connection>>,
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
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    steps TEXT NOT NULL,
                    trigger_pattern TEXT,
                    repetition_count INTEGER NOT NULL DEFAULT 0,
                    last_executed TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
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

    /// Get table row counts for status display.
    pub fn table_stats(&self) -> Result<Vec<(String, i64)>, SqliteError> {
        self.with_conn(|conn| {
            let tables = [
                "sessions",
                "episodes",
                "semantic_facts",
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
        assert_eq!(version, 7); // All 7 migrations applied
    }

    #[test]
    fn test_migrations_idempotent() {
        let pool = SqlitePool::open_memory().unwrap();
        // Running migrate again should be a no-op
        pool.migrate().unwrap();
        assert_eq!(pool.schema_version().unwrap(), 7);
    }

    #[test]
    fn test_table_stats_empty() {
        let pool = SqlitePool::open_memory().unwrap();
        let stats = pool.table_stats().unwrap();
        assert_eq!(stats.len(), 6);
        for (_, count) in &stats {
            assert_eq!(*count, 0);
        }
    }

    #[test]
    fn test_insert_and_query_session() {
        let pool = SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES (?1, ?2)",
                rusqlite::params!["sess-001", "cli"],
            )?;

            let channel: String = conn.query_row(
                "SELECT channel FROM sessions WHERE id = ?1",
                ["sess-001"],
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
            conn.execute("INSERT INTO sessions (id) VALUES (?1)", ["sess-001"])?;

            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params!["ep-001", "sess-001", "user", "Hello Brain!"],
            )?;

            let content: String = conn.query_row(
                "SELECT content FROM episodes WHERE id = ?1",
                ["ep-001"],
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
                rusqlite::params!["ep-001", "nonexistent", "user", "Hello"],
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
                rusqlite::params!["fact-001", "personal", "user", "name_is", "Keshav"],
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
}
