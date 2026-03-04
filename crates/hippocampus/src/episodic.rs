//! Episodic memory — SQLite-backed conversation store.
//!
//! Stores conversations as timestamped episodes with importance
//! scoring, decay rates, and reinforcement tracking.

use chrono::Utc;
use storage::SqlitePool;
use thiserror::Error;
use uuid::Uuid;

/// Errors from the episodic memory layer.
#[derive(Debug, Error)]
pub enum EpisodicError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] storage::sqlite::SqliteError),

    #[error("Episode not found: {0}")]
    NotFound(String),
}

/// A single conversational episode (message).
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub importance: f64,
    pub decay_rate: f64,
    pub reinforcement_count: i32,
    pub last_accessed: Option<String>,
}

/// A conversation session.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub channel: String,
}

/// A BM25 full-text search result.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub episode_id: String,
    pub content: String,
    pub rank: f64,
    /// ISO 8601 timestamp of when this episode was stored.
    pub timestamp: String,
}

/// Sanitize user input for FTS5 MATCH queries.
///
/// Strips characters that are special in FTS5 syntax (`"`, `*`, `+`, `-`,
/// `(`, `)`, `^`, `/`, `?`, `:`, `~`, `{`, `}`, `[`, `]`) and joins the
/// remaining alphanumeric tokens with spaces. Returns an empty string if
/// no searchable tokens remain.
fn sanitize_fts5_query(query: &str) -> String {
    query
        .chars()
        .map(|c| {
            if "\"*+-()^/?:~{}[]".contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Episodic memory store — manages conversations via SQLite.
pub struct EpisodicStore {
    db: SqlitePool,
}

impl EpisodicStore {
    /// Create a new episodic store backed by the given SQLite pool.
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }

    /// Get a reference to the underlying SQLite pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.db
    }

    /// Create a new conversation session.
    pub fn create_session(&self, channel: &str) -> Result<String, EpisodicError> {
        let id = Uuid::new_v4().to_string();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES (?1, ?2)",
                rusqlite::params![id, channel],
            )?;
            Ok(id.clone())
        })?;
        Ok(id)
    }

    /// End a conversation session.
    pub fn end_session(&self, session_id: &str) -> Result<(), EpisodicError> {
        let now = Utc::now().to_rfc3339();
        self.db.with_conn(|conn| {
            conn.execute(
                "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
                rusqlite::params![now, session_id],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: &str) -> Result<Session, EpisodicError> {
        let result = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT id, started_at, ended_at, channel FROM sessions WHERE id = ?1",
                [session_id],
                |row| {
                    Ok(Session {
                        id: row.get(0)?,
                        started_at: row.get(1)?,
                        ended_at: row.get(2)?,
                        channel: row.get(3)?,
                    })
                },
            )
            .map_err(|e| e.into())
        });
        match result {
            Ok(session) => Ok(session),
            Err(storage::sqlite::SqliteError::Rusqlite(rusqlite::Error::QueryReturnedNoRows)) => {
                Err(EpisodicError::NotFound(session_id.to_string()))
            }
            Err(e) => Err(EpisodicError::Sqlite(e)),
        }
    }

    /// Store an episode (message) in episodic memory.
    pub fn store_episode(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        importance: f64,
    ) -> Result<String, EpisodicError> {
        let id = Uuid::new_v4().to_string();
        let encrypted_content = self.db.encrypt_content(content);
        let is_encrypted = self.db.is_encrypted();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, importance)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![id, session_id, role, encrypted_content, importance],
            )?;

            // FTS5 indexes plaintext for BM25 search.
            // When encryption is enabled, skip FTS indexing — BM25 falls back
            // to empty results while vector search continues to work.
            if !is_encrypted {
                conn.execute(
                    "INSERT INTO episodes_fts (rowid, content) VALUES (last_insert_rowid(), ?1)",
                    [content],
                )?;
            }

            Ok(id.clone())
        })?;
        Ok(id)
    }

    /// Get the most recent episodes for a session.
    pub fn get_session_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<Episode>, EpisodicError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, role, content, timestamp,
                        importance, decay_rate, reinforcement_count, last_accessed
                 FROM episodes
                 WHERE session_id = ?1
                 ORDER BY timestamp ASC
                 LIMIT ?2",
            )?;

            let episodes = stmt
                .query_map(rusqlite::params![session_id, limit as i64], |row| {
                    let raw: String = row.get(3)?;
                    Ok(Episode {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        role: row.get(2)?,
                        content: pool.decrypt_content(&raw),
                        timestamp: row.get(4)?,
                        importance: row.get(5)?,
                        decay_rate: row.get(6)?,
                        reinforcement_count: row.get(7)?,
                        last_accessed: row.get(8)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(episodes)
        })?)
    }

    /// Reinforce a memory — bumps reinforcement count and updates last_accessed.
    ///
    /// Called each time a memory is recalled, making it resist decay longer.
    pub fn reinforce(&self, episode_id: &str) -> Result<(), EpisodicError> {
        let now = Utc::now().to_rfc3339();
        let rows = self.db.with_conn(|conn| {
            let rows = conn.execute(
                "UPDATE episodes SET reinforcement_count = reinforcement_count + 1,
                        last_accessed = ?1
                 WHERE id = ?2",
                rusqlite::params![now, episode_id],
            )?;
            Ok(rows)
        })?;
        if rows == 0 {
            return Err(EpisodicError::NotFound(episode_id.to_string()));
        }
        Ok(())
    }

    /// Search episodes by full-text query using BM25 ranking.
    pub fn search_bm25(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>, EpisodicError> {
        let sanitized = sanitize_fts5_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT e.id, f.content, f.rank, e.timestamp
                 FROM episodes_fts f
                 JOIN episodes e ON e.rowid = f.rowid
                 WHERE episodes_fts MATCH ?1
                 ORDER BY f.rank
                 LIMIT ?2",
            )?;

            let results = stmt
                .query_map(rusqlite::params![sanitized, limit as i64], |row| {
                    Ok(FtsResult {
                        episode_id: row.get(0)?,
                        content: row.get(1)?,
                        rank: row.get(2)?,
                        timestamp: row.get(3)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(results)
        })?)
    }

    /// Get total episode count.
    pub fn count(&self) -> Result<i64, EpisodicError> {
        Ok(self.db.with_conn(|conn| {
            let count: i64 =
                conn.query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?;
            Ok(count)
        })?)
    }

    /// Get recent episodes across all sessions.
    pub fn recent(&self, limit: usize) -> Result<Vec<Episode>, EpisodicError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, role, content, timestamp,
                        importance, decay_rate, reinforcement_count, last_accessed
                 FROM episodes
                 ORDER BY timestamp DESC
                 LIMIT ?1",
            )?;

            let episodes = stmt
                .query_map([limit as i64], |row| {
                    let raw: String = row.get(3)?;
                    Ok(Episode {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        role: row.get(2)?,
                        content: pool.decrypt_content(&raw),
                        timestamp: row.get(4)?,
                        importance: row.get(5)?,
                        decay_rate: row.get(6)?,
                        reinforcement_count: row.get(7)?,
                        last_accessed: row.get(8)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(episodes)
        })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> EpisodicStore {
        let pool = SqlitePool::open_memory().unwrap();
        EpisodicStore::new(pool)
    }

    #[test]
    fn test_create_session() {
        let store = test_store();
        let id = store.create_session("cli").unwrap();
        assert!(!id.is_empty());

        let session = store.get_session(&id).unwrap();
        assert_eq!(session.channel, "cli");
        assert!(session.ended_at.is_none());
    }

    #[test]
    fn test_end_session() {
        let store = test_store();
        let id = store.create_session("cli").unwrap();
        store.end_session(&id).unwrap();

        let session = store.get_session(&id).unwrap();
        assert!(session.ended_at.is_some());
    }

    #[test]
    fn test_store_and_retrieve_episodes() {
        let store = test_store();
        let session = store.create_session("cli").unwrap();

        store
            .store_episode(&session, "user", "Hello Brain!", 0.5)
            .unwrap();
        store
            .store_episode(&session, "assistant", "Hello! How can I help?", 0.5)
            .unwrap();
        store
            .store_episode(&session, "user", "What's the weather?", 0.3)
            .unwrap();

        let history = store.get_session_history(&session, 10).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[0].content, "Hello Brain!");
        assert_eq!(history[1].role, "assistant");
    }

    #[test]
    fn test_episode_count() {
        let store = test_store();
        let session = store.create_session("cli").unwrap();

        assert_eq!(store.count().unwrap(), 0);
        store
            .store_episode(&session, "user", "Test message", 0.5)
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[test]
    fn test_reinforce() {
        let store = test_store();
        let session = store.create_session("cli").unwrap();
        let ep_id = store
            .store_episode(&session, "user", "Important fact", 0.8)
            .unwrap();

        // Initial reinforcement count is 0
        let history = store.get_session_history(&session, 10).unwrap();
        assert_eq!(history[0].reinforcement_count, 0);

        // Reinforce
        store.reinforce(&ep_id).unwrap();
        store.reinforce(&ep_id).unwrap();

        let history = store.get_session_history(&session, 10).unwrap();
        assert_eq!(history[0].reinforcement_count, 2);
        assert!(history[0].last_accessed.is_some());
    }

    #[test]
    fn test_bm25_search() {
        let store = test_store();
        let session = store.create_session("cli").unwrap();

        store
            .store_episode(&session, "user", "I love programming in Rust", 0.7)
            .unwrap();
        store
            .store_episode(&session, "user", "Python is great for scripting", 0.5)
            .unwrap();
        store
            .store_episode(&session, "user", "Rust has amazing performance", 0.8)
            .unwrap();

        let results = store.search_bm25("Rust", 10).unwrap();
        assert_eq!(results.len(), 2);
        // Both results should contain "Rust"
        assert!(results.iter().all(|r| r.content.contains("Rust")));
    }

    #[test]
    fn test_recent_episodes() {
        let store = test_store();
        let s1 = store.create_session("cli").unwrap();
        let s2 = store.create_session("whatsapp").unwrap();

        store
            .store_episode(&s1, "user", "First message", 0.5)
            .unwrap();
        store
            .store_episode(&s2, "user", "Second message", 0.5)
            .unwrap();

        let recent = store.recent(10).unwrap();
        assert_eq!(recent.len(), 2);
        // Both messages should be present (order depends on timestamp precision)
        let contents: Vec<&str> = recent.iter().map(|e| e.content.as_str()).collect();
        assert!(contents.contains(&"First message"));
        assert!(contents.contains(&"Second message"));
    }
}
