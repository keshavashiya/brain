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

/// Canonicalise a predicate to lower `snake_case` so formatting variants
/// ("Server Address Is", "server-address-is", "server__address__is") all
/// collapse to one key. Unicode letters/digits are kept (lower-cased); every
/// other run of characters becomes a single `_`, and leading/trailing `_` are
/// dropped. This is the first line of defence against the model filing the same
/// relation under cosmetically different predicates.
pub fn normalize_predicate(predicate: &str) -> String {
    let mut out = String::with_capacity(predicate.len());
    let mut pending_separator = false;
    for ch in predicate.trim().chars() {
        if ch.is_alphanumeric() {
            if pending_separator && !out.is_empty() {
                out.push('_');
            }
            pending_separator = false;
            out.extend(ch.to_lowercase());
        } else {
            pending_separator = true;
        }
    }
    out
}

/// True when an object reads as a concrete value (IP, version, path, model
/// number, date, phone) rather than a generic concept. Used to gate
/// cross-predicate dedup: two predicates pointing at the *same concrete value*
/// ("server_at"/"server_address_is" → `10.4.2.19`) are almost certainly the
/// same fact, whereas the same generic word under two predicates
/// ("likes coffee" / "dislikes coffee", "born_in Paris" / "lives_in Paris")
/// can mean genuinely different things and must be left alone.
fn object_is_value_like(object: &str) -> bool {
    object.chars().any(|c| c.is_ascii_digit()) || object.contains(['.', ':', '/'])
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
        // Canonicalise the predicate up front so formatting variants don't
        // masquerade as distinct facts (and so the dedup keys below are stable).
        let subject = subject.trim();
        let predicate_norm = normalize_predicate(predicate);
        let predicate = predicate_norm.as_str();
        let object = object.trim();
        let content = format!("{subject} {predicate} {object}");
        let now = chrono::Utc::now().to_rfc3339();

        let _guard = self.write_lock.lock().await;

        // ── Deterministic dedup (embedder-independent) ──────────────────────
        // The vector path below only collapses *near-identical embeddings*,
        // which the deterministic fallback embedder doesn't produce and which
        // shift whenever the model picks a different predicate word for the same
        // fact ("server_at" vs "server_address_is"). Catch the high-confidence
        // duplicates up front by exact structured comparison over the subject's
        // own active facts.
        let existing = self.active_facts_for_subject(namespace, subject)?;
        let mut supersede: Vec<String> = Vec::new();
        for (id, cand_predicate, cand_object) in &existing {
            let same_object = cand_object.eq_ignore_ascii_case(object);
            if cand_predicate == predicate {
                // Exact restatement of an existing (subject, predicate, object)
                // is a no-op. A *changed* object under the same predicate is
                // left to the vector path's supersession, so multi-valued
                // predicates (skills, likes, projects) aren't clobbered here.
                if same_object {
                    return Ok(id.clone());
                }
            } else if same_object && object_is_value_like(object) {
                // Same concrete value under a different predicate → the model
                // re-filed one fact under a synonym predicate. Collapse it.
                supersede.push(id.clone());
            }
        }
        if !supersede.is_empty() {
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
                for old in &supersede {
                    conn.execute(
                        "UPDATE semantic_facts SET superseded_by = ?1 WHERE id = ?2",
                        rusqlite::params![id, old],
                    )?;
                }
                Ok(())
            })?;
            return Ok(id);
        }

        // ── Fuzzy near-dup via vector similarity ────────────────────────────
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

    /// Active (non-superseded, non-quarantined) facts about `subject` in
    /// `namespace`, as `(id, predicate, decrypted_object)` triples — the
    /// minimal shape the deterministic dedup in [`Self::store_fact`] needs.
    /// Bounded by the number of facts about a single subject. Objects that
    /// fail to decrypt (wrong key / corruption) are skipped, exactly as the
    /// recall queries do.
    fn active_facts_for_subject(
        &self,
        namespace: &str,
        subject: &str,
    ) -> Result<Vec<(String, String, String)>, SemanticError> {
        let pool = &self.db;
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, predicate, object FROM semantic_facts
                 WHERE namespace = ?1 AND subject = ?2 AND superseded_by IS NULL
                   AND id NOT IN (SELECT row_id FROM memory_quarantine WHERE kind = 'fact')",
            )?;
            let rows = stmt.query_map(rusqlite::params![namespace, subject], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            let mut out = Vec::new();
            for (id, predicate, raw_object) in rows.flatten() {
                if let Some(object) = pool.try_decrypt_content(&raw_object) {
                    out.push((id, predicate, object));
                }
            }
            Ok(out)
        })?)
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
             FROM semantic_facts WHERE id IN ({placeholders})
               AND id NOT IN (SELECT row_id FROM memory_quarantine WHERE kind = 'fact')"
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
    /// Quarantine a fact: written by an agent nobody vouched for, it is
    /// excluded from search, recall, and listings until the writer is
    /// approved. The row itself is untouched — the content stays
    /// auditable and releasable.
    pub fn quarantine_fact(&self, fact_id: &str, agent: &str) -> Result<(), SemanticError> {
        Ok(self.db.with_conn(|conn| {
            conn.execute(
                "INSERT OR IGNORE INTO memory_quarantine (kind, row_id, agent)
                 VALUES ('fact', ?1, ?2)",
                rusqlite::params![fact_id, agent],
            )?;
            Ok(())
        })?)
    }

    /// Release every quarantined fact written by `agent` (the writer was
    /// approved). Returns how many were released.
    pub fn release_quarantined_facts(&self, agent: &str) -> Result<usize, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let n = conn.execute(
                "DELETE FROM memory_quarantine WHERE kind = 'fact' AND agent = ?1",
                [agent],
            )?;
            Ok(n)
        })?)
    }

    /// Quarantined fact counts per agent, for the review surfaces
    /// (`/grants`, the capability digest).
    pub fn quarantined_fact_counts(&self) -> Result<Vec<(String, i64)>, SemanticError> {
        Ok(self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT agent, COUNT(*) FROM memory_quarantine
                 WHERE kind = 'fact' GROUP BY agent ORDER BY agent",
            )?;
            let rows = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?)
    }

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
