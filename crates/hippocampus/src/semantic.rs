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
    pub namespace: String,
    pub category: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source_episode_id: Option<String>,
    /// Originating AI agent (e.g. "claude-code", "opencode"). None for direct user input.
    pub agent: Option<String>,
}

/// A vector search result with the associated fact.
#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub fact: Fact,
    pub distance: f32,
    /// When this fact was last updated (ISO 8601).
    pub created_at: String,
}

/// Semantic memory store — dual-writes to SQLite + RuVector.
///
/// SQLite stores the structured fact data (subject-predicate-object),
/// while RuVector stores the vector embeddings for similarity search.
#[derive(Clone)]
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
    /// The `namespace` scopes the fact (e.g. "personal", "work").
    #[allow(clippy::too_many_arguments)]
    pub async fn store_fact(
        &self,
        namespace: &str,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source_episode_id: Option<&str>,
        vector: Vec<f32>,
        agent: Option<&str>,
    ) -> Result<String, SemanticError> {
        let content = format!("{subject} {predicate} {object}");
        let now = chrono::Utc::now().to_rfc3339();

        // Deduplication: check for highly similar facts in the same namespace and category.
        let similar = self.search_similar(vector.clone(), 1, Some(namespace), agent).await?;
        if let Some(hit) = similar.first() {
            // Distance < 0.1 means similarity > 0.9 (cosine distance = 1 - similarity)
            if hit.distance < 0.1 && hit.fact.category == category {
                // If the content is identical, just return the existing ID.
                if hit.fact.subject == subject && hit.fact.predicate == predicate && hit.fact.object == object {
                    return Ok(hit.fact.id.clone());
                }
                // Otherwise, mark the existing fact as superseded and insert the new one.
                let id = self.do_store_fact(namespace, category, subject, predicate, object, confidence, source_episode_id, vector, agent, &content, &now).await?;
                self.db.with_conn(|conn| {
                    conn.execute(
                        "UPDATE semantic_facts SET superseded_by = ?1 WHERE id = ?2",
                        rusqlite::params![id, hit.fact.id],
                    )?;
                    Ok(())
                })?;
                return Ok(id);
            }
        }

        self.do_store_fact(namespace, category, subject, predicate, object, confidence, source_episode_id, vector, agent, &content, &now).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn do_store_fact(
        &self,
        namespace: &str,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source_episode_id: Option<&str>,
        vector: Vec<f32>,
        agent: Option<&str>,
        content: &str,
        now: &str,
    ) -> Result<String, SemanticError> {
        let id = Uuid::new_v4().to_string();

        // Encrypt the object field (the main content) if encryption is enabled.
        let stored_object = self.db.encrypt_content(object);

        // Write to SQLite
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO semantic_facts (id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![id, namespace, category, subject, predicate, stored_object, confidence, source_episode_id, agent],
            )?;
            Ok(())
        })?;

        // Write vector to RuVector
        self.ruv
            .add_vectors(
                "facts_vec",
                vec![id.clone()],
                vec![content.to_string()],
                vec![vector],
                vec![now.to_string()],
                "semantic",
            )
            .await?;

        Ok(id)
    }

    /// Search for similar facts by vector, optionally scoped to a namespace.
    ///
    /// Returns facts ranked by vector similarity (closest first).
    /// If `namespace` is `None`, results from all namespaces are returned.
    pub async fn search_similar(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        namespace: Option<&str>,
        agent: Option<&str>,
    ) -> Result<Vec<SemanticResult>, SemanticError> {
        // Fetch more candidates so we have enough after filtering
        let fetch_k = if namespace.is_some() || agent.is_some() {
            top_k * 4
        } else {
            top_k
        };
        let ruv_results: Vec<VectorResult> =
            self.ruv.search("facts_vec", query_vector, fetch_k).await?;

        let mut results = Vec::new();
        for vr in ruv_results {
            if results.len() >= top_k {
                break;
            }
            // Look up the full fact + timestamp from SQLite
            let fact_opt = self.get_fact_with_timestamp(&vr.id)?;
            if let Some((fact, created_at)) = fact_opt {
                // Filter by namespace if specified
                if namespace.is_some_and(|ns| ns != fact.namespace) {
                    continue;
                }
                // Filter by agent if specified
                if agent.is_some_and(|a| fact.agent.as_deref() != Some(a)) {
                    continue;
                }
                results.push(SemanticResult {
                    fact,
                    distance: vr.distance,
                    created_at,
                });
            }
        }

        Ok(results)
    }

    /// Get a fact by ID from SQLite.
    pub fn get_fact(&self, fact_id: &str) -> Result<Option<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let result = conn.query_row(
                "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                 FROM semantic_facts WHERE id = ?1",
                [fact_id],
                |row| {
                    let raw_object: String = row.get(5)?;
                    Ok(Fact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: pool.decrypt_content(&raw_object),
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                        agent: row.get(8)?,
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

    /// Get a fact and its updated_at timestamp by ID.
    fn get_fact_with_timestamp(
        &self,
        fact_id: &str,
    ) -> Result<Option<(Fact, String)>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let result = conn.query_row(
                "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, updated_at, agent
                 FROM semantic_facts WHERE id = ?1",
                [fact_id],
                |row| {
                    let raw_object: String = row.get(5)?;
                    let updated_at: String = row.get(8)?;
                    Ok((Fact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: pool.decrypt_content(&raw_object),
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                        agent: row.get(9)?,
                    }, updated_at))
                },
            );
            match result {
                Ok(pair) => Ok(Some(pair)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })?)
    }

    /// Get all facts by category, optionally filtered by namespace.
    pub fn get_facts_by_category(
        &self,
        category: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match namespace {
                Some(ns) => (
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts 
                     WHERE category = ?1 AND (namespace = ?2 OR namespace LIKE ?3) AND superseded_by IS NULL
                     ORDER BY updated_at DESC".to_string(),
                    vec![Box::new(category.to_string()), Box::new(ns.to_string()), Box::new(format!("{}/%", ns))],
                ),
                None => (
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts WHERE category = ?1 AND superseded_by IS NULL
                     ORDER BY updated_at DESC".to_string(),
                    vec![Box::new(category.to_string())],
                ),
            };

            let mut stmt = conn.prepare(&sql)?;
            let params_ref: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

            let facts = stmt
                .query_map(params_ref.as_slice(), |row| {
                    let raw_object: String = row.get(5)?;
                    Ok(Fact {
                        id: row.get(0)?,
                        namespace: row.get(1)?,
                        category: row.get(2)?,
                        subject: row.get(3)?,
                        predicate: row.get(4)?,
                        object: pool.decrypt_content(&raw_object),
                        confidence: row.get(6)?,
                        source_episode_id: row.get(7)?,
                        agent: row.get(8)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(facts)
        })?)
    }

    /// Get all facts about a specific subject.
    pub fn get_facts_about(&self, subject: &str) -> Result<Vec<Fact>, SemanticError> {
        self.get_facts_about_in_namespace(subject, None)
    }

    /// Get facts about a specific subject, optionally filtered by namespace.
    pub fn get_facts_about_in_namespace(
        &self,
        subject: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let row_to_fact = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Fact> {
                let raw_object: String = row.get(5)?;
                Ok(Fact {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    category: row.get(2)?,
                    subject: row.get(3)?,
                    predicate: row.get(4)?,
                    object: pool.decrypt_content(&raw_object),
                    confidence: row.get(6)?,
                    source_episode_id: row.get(7)?,
                    agent: row.get(8)?,
                })
            };

            let facts: Vec<Fact> = if let Some(ns) = namespace {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE subject = ?1 AND (namespace = ?2 OR namespace LIKE ?3)
                     ORDER BY confidence DESC",
                )?;
                let prefix = format!("{}/%", ns);
                let rows = stmt.query_map(rusqlite::params![subject, ns, &prefix], row_to_fact)?;
                rows.collect::<Result<Vec<_>, _>>()?
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE subject = ?1
                     ORDER BY confidence DESC",
                )?;
                let rows = stmt.query_map([subject], row_to_fact)?;
                rows.collect::<Result<Vec<_>, _>>()?
            };

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

        // Store new fact (preserve namespace and agent)
        let new_id = self
            .store_fact(
                &old_fact.namespace,
                &old_fact.category,
                &old_fact.subject,
                &old_fact.predicate,
                new_object,
                old_fact.confidence,
                old_fact.source_episode_id.as_deref(),
                new_vector,
                old_fact.agent.as_deref(),
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

    /// List all active (non-superseded) facts, optionally scoped to a namespace.
    pub fn list_all(&self) -> Result<Vec<Fact>, SemanticError> {
        self.list_by_namespace(None)
    }

    /// List all active facts in a specific namespace.
    pub fn list_by_namespace(&self, namespace: Option<&str>) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let row_to_fact = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Fact> {
                let raw_object: String = row.get(5)?;
                Ok(Fact {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    category: row.get(2)?,
                    subject: row.get(3)?,
                    predicate: row.get(4)?,
                    object: pool.decrypt_content(&raw_object),
                    confidence: row.get(6)?,
                    source_episode_id: row.get(7)?,
                    agent: row.get(8)?,
                })
            };

            let facts: Vec<Fact> = if let Some(ns) = namespace {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts WHERE superseded_by IS NULL AND namespace = ?1
                     ORDER BY rowid DESC",
                )?;
                let rows = stmt.query_map([ns], row_to_fact)?.collect::<Result<Vec<_>, _>>()?;
                rows
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts WHERE superseded_by IS NULL
                     ORDER BY rowid DESC",
                )?;
                let rows = stmt.query_map([], row_to_fact)?.collect::<Result<Vec<_>, _>>()?;
                rows
            };
            Ok(facts)
        })?)
    }

    /// List all namespaces with their fact and episode counts.
    ///
    /// Returns `(namespace, fact_count, episode_count)` tuples.
    pub fn list_namespaces(&self) -> Result<Vec<NamespaceStats>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT namespace, COUNT(*) as fact_count FROM semantic_facts
                 WHERE superseded_by IS NULL
                 GROUP BY namespace ORDER BY namespace",
            )?;
            let fact_ns: Vec<(String, i64)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            let mut stmt2 = conn.prepare(
                "SELECT namespace, COUNT(*) as ep_count FROM episodes
                 GROUP BY namespace ORDER BY namespace",
            )?;
            let ep_ns: Vec<(String, i64)> = stmt2
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;

            // Merge both lists by namespace
            let mut map: std::collections::HashMap<String, (i64, i64)> =
                std::collections::HashMap::new();
            for (ns, cnt) in &fact_ns {
                map.entry(ns.clone()).or_default().0 = *cnt;
            }
            for (ns, cnt) in &ep_ns {
                map.entry(ns.clone()).or_default().1 = *cnt;
            }

            let mut result: Vec<NamespaceStats> = map
                .into_iter()
                .map(|(namespace, (fact_count, episode_count))| NamespaceStats {
                    namespace,
                    fact_count,
                    episode_count,
                })
                .collect();
            result.sort_by(|a, b| a.namespace.cmp(&b.namespace));
            Ok(result)
        })?)
    }

    /// Delete a fact from both SQLite and RuVector.
    pub async fn delete_fact(&self, fact_id: &str) -> Result<(), SemanticError> {
        // Delete from SQLite
        self.db.with_conn(|conn| {
            conn.execute("DELETE FROM semantic_facts WHERE id = ?1", [fact_id])?;
            Ok(())
        })?;

        // Delete from RuVector
        self.ruv.delete("facts_vec", fact_id).await?;

        Ok(())
    }

    /// Find facts whose subject, predicate, or object contains the query string.
    ///
    /// Used by the Forget intent to find facts matching a target description.
    pub fn find_facts_matching(
        &self,
        query: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<Fact>, SemanticError> {
        let pool = &self.db;
        let pattern = format!("%{query}%");
        Ok(self.db.with_conn(|conn| {
            let row_to_fact = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Fact> {
                let raw_object: String = row.get(5)?;
                Ok(Fact {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    category: row.get(2)?,
                    subject: row.get(3)?,
                    predicate: row.get(4)?,
                    object: pool.decrypt_content(&raw_object),
                    confidence: row.get(6)?,
                    source_episode_id: row.get(7)?,
                    agent: row.get(8)?,
                })
            };

            let facts: Vec<Fact> = if let Some(ns) = namespace {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE superseded_by IS NULL
                       AND namespace = ?2
                       AND (subject LIKE ?1 OR predicate LIKE ?1 OR object LIKE ?1)
                     ORDER BY rowid DESC
                     LIMIT 50",
                )?;
                let rows = stmt.query_map(rusqlite::params![&pattern, ns], row_to_fact)?;
                rows.collect::<Result<Vec<_>, _>>()?
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent
                     FROM semantic_facts
                     WHERE superseded_by IS NULL
                       AND (subject LIKE ?1 OR predicate LIKE ?1 OR object LIKE ?1)
                     ORDER BY rowid DESC
                     LIMIT 50",
                )?;
                let rows = stmt.query_map([&pattern], row_to_fact)?;
                rows.collect::<Result<Vec<_>, _>>()?
            };

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

/// Statistics for a single namespace.
#[derive(Debug, Clone)]
pub struct NamespaceStats {
    pub namespace: String,
    pub fact_count: i64,
    pub episode_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_store() -> (SemanticStore, tempfile::TempDir) {
        let db = SqlitePool::open_memory().unwrap();
        let ruv_dir = tempfile::tempdir().unwrap();
        let ruv = RuVectorStore::open(ruv_dir.path(), 384).await.unwrap();
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
                "personal",
                "user",
                "name_is",
                "Keshav",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();

        let fact = store.get_fact(&id).unwrap().unwrap();
        assert_eq!(fact.namespace, "personal");
        assert_eq!(fact.subject, "user");
        assert_eq!(fact.predicate, "name_is");
        assert_eq!(fact.object, "Keshav");
        assert_eq!(fact.category, "personal");
    }

    #[tokio::test]
    async fn test_get_facts_by_category() {
        let (store, _dir) = test_store().await;

        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "name_is",
                "Keshav",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "likes",
                "Rust",
                0.9,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "personal",
                "work",
                "user",
                "role_is",
                "developer",
                0.8,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();

        // Without namespace filter — returns all
        let personal = store.get_facts_by_category("personal", None).unwrap();
        assert_eq!(personal.len(), 2);

        let work = store.get_facts_by_category("work", None).unwrap();
        assert_eq!(work.len(), 1);

        // With namespace filter — scoped correctly
        let scoped = store
            .get_facts_by_category("personal", Some("personal"))
            .unwrap();
        assert_eq!(scoped.len(), 2);

        // Cross-namespace isolation: "work" category stored under "personal" namespace
        let cross = store.get_facts_by_category("work", Some("other")).unwrap();
        assert_eq!(cross.len(), 0);
    }

    #[tokio::test]
    async fn test_get_facts_about() {
        let (store, _dir) = test_store().await;

        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "name_is",
                "Keshav",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "likes",
                "Rust",
                0.9,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "personal",
                "personal",
                "Alice",
                "knows",
                "user",
                0.5,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();

        let about_user = store.get_facts_about("user").unwrap();
        assert_eq!(about_user.len(), 2);
    }

    #[tokio::test]
    async fn test_fact_count() {
        let (store, _dir) = test_store().await;

        assert_eq!(store.count().unwrap(), 0);
        store
            .store_fact(
                "personal",
                "test",
                "a",
                "b",
                "c",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
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

        store
            .store_fact(
                "personal",
                "test",
                "rust",
                "is",
                "fast",
                1.0,
                None,
                v1.clone(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "personal", "test", "python", "is", "popular", 1.0, None, v2, None,
            )
            .await
            .unwrap();

        // Search with v1 — should find "rust is fast" first
        let results = store.search_similar(v1, 2, None, None).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].fact.subject, "rust");
    }

    #[tokio::test]
    async fn test_update_fact() {
        let (store, _dir) = test_store().await;

        let old_id = store
            .store_fact(
                "personal",
                "personal",
                "user",
                "location",
                "NYC",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();

        let new_id = store
            .update_fact(&old_id, "SF", dummy_vector())
            .await
            .unwrap();

        // New fact should have the updated value
        let new_fact = store.get_fact(&new_id).unwrap().unwrap();
        assert_eq!(new_fact.object, "SF");
        assert_eq!(new_fact.namespace, "personal"); // namespace preserved

        // Active count should still be 1 (old is superseded)
        assert_eq!(store.count().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_namespace_isolation() {
        let (store, _dir) = test_store().await;

        // Store facts in different namespaces
        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "hobby",
                "coding",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "work",
                "work",
                "user",
                "role",
                "developer",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();

        // list_by_namespace filters correctly
        let personal = store.list_by_namespace(Some("personal")).unwrap();
        assert_eq!(personal.len(), 1);
        assert_eq!(personal[0].namespace, "personal");

        let work = store.list_by_namespace(Some("work")).unwrap();
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].namespace, "work");

        // list_all returns both
        let all = store.list_all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_list_namespaces() {
        let (store, _dir) = test_store().await;

        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "hobby",
                "coding",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "personal",
                "personal",
                "user",
                "name",
                "Keshav",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();
        store
            .store_fact(
                "work",
                "work",
                "user",
                "role",
                "developer",
                1.0,
                None,
                dummy_vector(),
                None,
            )
            .await
            .unwrap();

        let namespaces = store.list_namespaces().unwrap();
        assert_eq!(namespaces.len(), 2);

        let personal_ns = namespaces
            .iter()
            .find(|n| n.namespace == "personal")
            .unwrap();
        assert_eq!(personal_ns.fact_count, 2);

        let work_ns = namespaces.iter().find(|n| n.namespace == "work").unwrap();
        assert_eq!(work_ns.fact_count, 1);
    }
}
