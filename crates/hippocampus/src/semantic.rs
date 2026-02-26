//! Semantic memory — RuVector-backed vector memory.
//!
//! Stores extracted facts, user model data, and knowledge
//! as vector embeddings for similarity-based retrieval.

use storage::{RuVectorStore, SqlitePool, VectorResult};
use thiserror::Error;
use uuid::Uuid;

/// Errors from the semantic memory layer.
#[derive(Debug, Error)]
pub enum SemanticError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] storage::sqlite::SqliteError),

    #[error("RuVector error: {0}")]
    RuVector(#[from] storage::ruvector::RuVectorError),

    #[error("Fact not found: {0}")]
    NotFound(String),
}

/// A semantic fact — a structured piece of knowledge.
#[derive(Debug, Clone)]
pub struct Fact {
    pub id: String,
    pub category: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source_episode_id: Option<String>,
}

/// A vector search result with the associated fact.
#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub fact: Fact,
    pub distance: f32,
}

/// Semantic memory store — dual-writes to SQLite + RuVector.
///
/// SQLite stores the structured fact data (subject-predicate-object),
/// while RuVector stores the vector embeddings for similarity search.
pub struct SemanticStore {
    db: SqlitePool,
    ruv: RuVectorStore,
}

impl SemanticStore {
    /// Create a new semantic store.
    pub fn new(db: SqlitePool, ruv: RuVectorStore) -> Self {
        Self { db, ruv }
    }

    /// Store a new fact in both SQLite and RuVector.
    ///
    /// The `vector` should be the embedding of the fact's content
    /// (typically: "{subject} {predicate} {object}").
    pub async fn store_fact(
        &self,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source_episode_id: Option<&str>,
        vector: Vec<f32>,
    ) -> Result<String, SemanticError> {
        let id = Uuid::new_v4().to_string();
        let content = format!("{subject} {predicate} {object}");
        let now = chrono::Utc::now().to_rfc3339();

        // Write to SQLite
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object, confidence, source_episode_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![id, category, subject, predicate, object, confidence, source_episode_id],
            )?;
            Ok(())
        })?;

        // Write vector to RuVector
        self.ruv
            .add_vectors(
                "facts_vec",
                vec![id.clone()],
                vec![content],
                vec![vector],
                vec![now],
                "semantic",
            )
            .await?;

        Ok(id)
    }

    /// Search for similar facts by vector.
    ///
    /// Returns facts ranked by vector similarity (closest first).
    pub async fn search_similar(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<SemanticResult>, SemanticError> {
        let ruv_results: Vec<VectorResult> = self
            .ruv
            .search("facts_vec", query_vector, top_k)
            .await?;

        let mut results = Vec::new();
        for vr in ruv_results {
            // Look up the full fact from SQLite
            let fact_opt = self.get_fact(&vr.id)?;
            if let Some(fact) = fact_opt {
                results.push(SemanticResult {
                    fact,
                    distance: vr.distance,
                });
            }
        }

        Ok(results)
    }

    /// Get a fact by ID from SQLite.
    pub fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let result = conn.query_row(
                "SELECT id, category, subject, predicate, object, confidence, source_episode_id
                 FROM semantic_facts WHERE id = ?1",
                [fact_id],
                |row| {
                    Ok(Fact {
                        id: row.get(0)?,
                        category: row.get(1)?,
                        subject: row.get(2)?,
                        predicate: row.get(3)?,
                        object: row.get(4)?,
                        confidence: row.get(5)?,
                        source_episode_id: row.get(6)?,
                    })
                },
            );
            match result {
                Ok(fact) => Ok(Some(fact)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })?)
    }

    /// Get all facts by category.
    pub fn get_facts_by_category(&self, category: &str) -> Result<Vec<Fact>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, category, subject, predicate, object, confidence, source_episode_id
                 FROM semantic_facts WHERE category = ?1
                 ORDER BY updated_at DESC",
            )?;

            let facts = stmt
                .query_map([category], |row| {
                    Ok(Fact {
                        id: row.get(0)?,
                        category: row.get(1)?,
                        subject: row.get(2)?,
                        predicate: row.get(3)?,
                        object: row.get(4)?,
                        confidence: row.get(5)?,
                        source_episode_id: row.get(6)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(facts)
        })?)
    }

    /// Get all facts about a specific subject.
    pub fn get_facts_about(&self, subject: &str) -> Result<Vec<Fact>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, category, subject, predicate, object, confidence, source_episode_id
                 FROM semantic_facts WHERE subject = ?1
                 ORDER BY confidence DESC",
            )?;

            let facts = stmt
                .query_map([subject], |row| {
                    Ok(Fact {
                        id: row.get(0)?,
                        category: row.get(1)?,
                        subject: row.get(2)?,
                        predicate: row.get(3)?,
                        object: row.get(4)?,
                        confidence: row.get(5)?,
                        source_episode_id: row.get(6)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(facts)
        })?)
    }

    /// Update a fact (supersedes the old version).
    pub async fn update_fact(
        &self,
        old_fact_id: &str,
        new_object: &str,
        new_vector: Vec<f32>,
    ) -> Result<String, SemanticError> {
        // Get the old fact for its metadata
        let old_fact = self
            .get_fact(old_fact_id)?
            .ok_or_else(|| SemanticError::NotFound(old_fact_id.to_string()))?;

        // Store new fact
        let new_id = self
            .store_fact(
                &old_fact.category,
                &old_fact.subject,
                &old_fact.predicate,
                new_object,
                old_fact.confidence,
                old_fact.source_episode_id.as_deref(),
                new_vector,
            )
            .await?;

        // Mark old fact as superseded
        self.db.with_conn(|conn| {
            conn.execute(
                "UPDATE semantic_facts SET superseded_by = ?1 WHERE id = ?2",
                rusqlite::params![new_id, old_fact_id],
            )?;
            Ok(())
        })?;

        Ok(new_id)
    }

    /// List all active (non-superseded) facts.
    pub fn list_all(&self) -> Result<Vec<Fact>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, category, subject, predicate, object, confidence, source_episode_id
                 FROM semantic_facts WHERE superseded_by IS NULL
                 ORDER BY rowid DESC",
            )?;

            let facts = stmt
                .query_map([], |row| {
                    Ok(Fact {
                        id: row.get(0)?,
                        category: row.get(1)?,
                        subject: row.get(2)?,
                        predicate: row.get(3)?,
                        object: row.get(4)?,
                        confidence: row.get(5)?,
                        source_episode_id: row.get(6)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(facts)
        })?)
    }

    /// Count total active facts.
    pub fn count(&self) -> Result<i64, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL",
                [],
                |row| row.get(0),
            )?;
            Ok(count)
        })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_store() -> (SemanticStore, tempfile::TempDir) {
        let db = SqlitePool::open_memory().unwrap();
        let ruv_dir = tempfile::tempdir().unwrap();
        let ruv = RuVectorStore::open(ruv_dir.path()).await.unwrap();
        ruv.ensure_tables().await.unwrap();
        (SemanticStore::new(db, ruv), ruv_dir)
    }

    fn dummy_vector() -> Vec<f32> {
        vec![0.1; 384]
    }

    #[tokio::test]
    async fn test_store_and_get_fact() {
        let (store, _dir) = test_store().await;

        let id = store
            .store_fact(
                "personal",
                "user",
                "name_is",
                "Keshav",
                1.0,
                None,
                dummy_vector(),
            )
            .await
            .unwrap();

        let fact = store.get_fact(&id).unwrap().unwrap();
        assert_eq!(fact.subject, "user");
        assert_eq!(fact.predicate, "name_is");
        assert_eq!(fact.object, "Keshav");
        assert_eq!(fact.category, "personal");
    }

    #[tokio::test]
    async fn test_get_facts_by_category() {
        let (store, _dir) = test_store().await;

        store.store_fact("personal", "user", "name_is", "Keshav", 1.0, None, dummy_vector()).await.unwrap();
        store.store_fact("personal", "user", "likes", "Rust", 0.9, None, dummy_vector()).await.unwrap();
        store.store_fact("work", "user", "role_is", "developer", 0.8, None, dummy_vector()).await.unwrap();

        let personal = store.get_facts_by_category("personal").unwrap();
        assert_eq!(personal.len(), 2);

        let work = store.get_facts_by_category("work").unwrap();
        assert_eq!(work.len(), 1);
    }

    #[tokio::test]
    async fn test_get_facts_about() {
        let (store, _dir) = test_store().await;

        store.store_fact("personal", "user", "name_is", "Keshav", 1.0, None, dummy_vector()).await.unwrap();
        store.store_fact("personal", "user", "likes", "Rust", 0.9, None, dummy_vector()).await.unwrap();
        store.store_fact("personal", "Alice", "knows", "user", 0.5, None, dummy_vector()).await.unwrap();

        let about_user = store.get_facts_about("user").unwrap();
        assert_eq!(about_user.len(), 2);
    }

    #[tokio::test]
    async fn test_fact_count() {
        let (store, _dir) = test_store().await;

        assert_eq!(store.count().unwrap(), 0);
        store.store_fact("test", "a", "b", "c", 1.0, None, dummy_vector()).await.unwrap();
        assert_eq!(store.count().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_vector_search() {
        let (store, _dir) = test_store().await;

        // Insert facts with different vectors
        let mut v1 = vec![0.0f32; 384];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; 384];
        v2[1] = 1.0;

        store.store_fact("test", "rust", "is", "fast", 1.0, None, v1.clone()).await.unwrap();
        store.store_fact("test", "python", "is", "popular", 1.0, None, v2).await.unwrap();

        // Search with v1 — should find "rust is fast" first
        let results = store.search_similar(v1, 2).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].fact.subject, "rust");
    }

    #[tokio::test]
    async fn test_update_fact() {
        let (store, _dir) = test_store().await;

        let old_id = store
            .store_fact("personal", "user", "location", "NYC", 1.0, None, dummy_vector())
            .await
            .unwrap();

        let new_id = store
            .update_fact(&old_id, "SF", dummy_vector())
            .await
            .unwrap();

        // New fact should have the updated value
        let new_fact = store.get_fact(&new_id).unwrap().unwrap();
        assert_eq!(new_fact.object, "SF");

        // Active count should still be 1 (old is superseded)
        assert_eq!(store.count().unwrap(), 1);
    }
}
