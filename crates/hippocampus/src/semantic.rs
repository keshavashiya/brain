//! Semantic memory — RuVector-backed vector memory.
//!
//! Stores extracted facts, user model data, and knowledge
//! as vector embeddings for similarity-based retrieval.

use storage::{RuVectorStore, SqlitePool, VectorResult};
use thiserror::Error;
use uuid::Uuid;

mod query;

#[cfg(test)]
mod tests;

pub use query::NamespaceStats;

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
    /// Originating AI agent — opaque id set by the caller. `None` for
    /// direct user input.
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
    /// Write lock to prevent TOCTOU races during dedup-then-insert.
    write_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
}

impl SemanticStore {
    /// Create a new semantic store.
    pub fn new(db: SqlitePool, ruv: RuVectorStore) -> Self {
        Self {
            db,
            ruv,
            write_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    /// Clone of the underlying vector store handle. `RuVectorStore` is
    /// `Arc`-backed, so the clone shares the same tables — used to give
    /// the graph write-path / recall a `graph_vec` collection on the
    /// same store without re-opening it.
    pub fn vector_store(&self) -> RuVectorStore {
        self.ruv.clone()
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

        let _guard = self.write_lock.lock().await;

        let similar = self
            .search_similar(vector.clone(), 1, Some(namespace), agent)
            .await?;
        if let Some(hit) = similar.first() {
            if hit.distance < 0.1 && hit.fact.category == category {
                if hit.fact.subject == subject
                    && hit.fact.predicate == predicate
                    && hit.fact.object == object
                {
                    return Ok(hit.fact.id.clone());
                }

                let id = self
                    .do_store_fact(
                        namespace,
                        category,
                        subject,
                        predicate,
                        object,
                        confidence,
                        source_episode_id,
                        vector,
                        agent,
                        &content,
                        &now,
                    )
                    .await?;
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

        self.do_store_fact(
            namespace,
            category,
            subject,
            predicate,
            object,
            confidence,
            source_episode_id,
            vector,
            agent,
            &content,
            &now,
        )
        .await
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

        let stored_object = self.db.encrypt_content(object);

        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO semantic_facts (id, namespace, category, subject, predicate, object, confidence, source_episode_id, agent)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![id, namespace, category, subject, predicate, stored_object, confidence, source_episode_id, agent],
            )?;
            Ok(())
        })?;

        let ruv_result = self
            .ruv
            .add_vectors(
                "facts_vec",
                vec![id.clone()],
                vec![content.to_string()],
                vec![vector],
                vec![now.to_string()],
                "semantic",
            )
            .await;

        if let Err(e) = ruv_result {
            self.db.with_conn(|conn| {
                conn.execute("DELETE FROM semantic_facts WHERE id = ?1", [&id])?;
                Ok(())
            })?;
            return Err(SemanticError::RuVector(e));
        }

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
        let fetch_k = if namespace.is_some() || agent.is_some() {
            top_k * 4
        } else {
            top_k
        };
        let ruv_results: Vec<VectorResult> =
            self.ruv.search("facts_vec", query_vector, fetch_k).await?;

        if ruv_results.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<&str> = ruv_results.iter().map(|vr| vr.id.as_str()).collect();
        let placeholders: String = (1..=ids.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT id, namespace, category, subject, predicate, object, confidence, source_episode_id, updated_at, agent, superseded_by
             FROM semantic_facts WHERE id IN ({placeholders})"
        );

        let pool = &self.db;
        let fact_map: std::collections::HashMap<String, Option<(Fact, String)>> =
            self.db.with_conn(|conn| {
                let mut stmt = conn.prepare(&sql)?;
                let params: Vec<&dyn rusqlite::types::ToSql> = ids
                    .iter()
                    .map(|id| id as &dyn rusqlite::types::ToSql)
                    .collect();
                let rows = stmt.query_map(params.as_slice(), |row| {
                    let raw_object: String = row.get(5)?;
                    let updated_at: String = row.get(8)?;
                    let superseded_by: Option<String> = row.get(10)?;
                    Ok((
                        Fact {
                            id: row.get(0)?,
                            namespace: row.get(1)?,
                            category: row.get(2)?,
                            subject: row.get(3)?,
                            predicate: row.get(4)?,
                            object: String::new(),
                            confidence: row.get(6)?,
                            source_episode_id: row.get(7)?,
                            agent: row.get(9)?,
                        },
                        raw_object,
                        updated_at,
                        superseded_by,
                    ))
                })?;

                let mut map = std::collections::HashMap::new();
                for (mut fact, raw_object, updated_at, superseded_by) in rows.flatten() {
                    if superseded_by.is_some() {
                        map.insert(fact.id.clone(), None);
                        continue;
                    }
                    match pool.try_decrypt_content(&raw_object) {
                        Some(obj) => {
                            fact.object = obj;
                            map.insert(fact.id.clone(), Some((fact, updated_at)));
                        }
                        None => {
                            map.insert(fact.id.clone(), None);
                        }
                    }
                }
                Ok(map)
            })?;

        let mut results = Vec::new();
        for vr in &ruv_results {
            if results.len() >= top_k {
                break;
            }
            if let Some(Some((ref fact, ref created_at))) = fact_map.get(&vr.id) {
                if namespace.is_some_and(|ns| ns != fact.namespace) {
                    continue;
                }
                if agent.is_some_and(|a| fact.agent.as_deref() != Some(a)) {
                    continue;
                }
                results.push(SemanticResult {
                    fact: fact.clone(),
                    distance: vr.distance,
                    created_at: created_at.clone(),
                });
            }
        }

        Ok(results)
    }

    /// Update a fact (supersedes the old version).
    pub async fn update_fact(
        &self,
        old_fact_id: &str,
        new_object: &str,
        new_vector: Vec<f32>,
    ) -> Result<String, SemanticError> {
        let old_fact = self
            .get_fact(old_fact_id)?
            .ok_or_else(|| SemanticError::NotFound(old_fact_id.to_string()))?;

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

        self.db.with_conn(|conn| {
            conn.execute(
                "UPDATE semantic_facts SET superseded_by = ?1 WHERE id = ?2",
                rusqlite::params![new_id, old_fact_id],
            )?;
            Ok(())
        })?;

        Ok(new_id)
    }

    /// Insert a pre-existing fact's vector into the index (for import/re-embed).
    ///
    /// Does NOT write to SQLite — only adds the vector to RuVector.
    /// Used by import to re-embed facts that already exist in SQLite.
    pub async fn add_vector(
        &self,
        fact_id: &str,
        content: &str,
        vector: Vec<f32>,
        source: &str,
    ) -> Result<(), SemanticError> {
        let now = chrono::Utc::now().to_rfc3339();
        self.ruv
            .add_vectors(
                "facts_vec",
                vec![fact_id.to_string()],
                vec![content.to_string()],
                vec![vector],
                vec![now],
                source,
            )
            .await?;
        Ok(())
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
