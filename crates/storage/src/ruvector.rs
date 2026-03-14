//! RuVector — backed by ruvector-core HNSW vector database.
//!
//! Wraps [`ruvector_core::VectorDB`] with the multi-table interface that the
//! rest of Brain uses.  Each logical table maps to one `VectorDB` persisted at
//! `<root>/<table_name>.db`.
//!
//! # Storage layout
//! ```text
//! ~/.brain/ruvector/
//!   facts_vec.db     -- semantic fact vectors (HNSW)
//!   episodes_vec.db  -- episode vectors (HNSW)
//! ```

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use thiserror::Error;
use tracing::{info, warn};

use ruvector_core::{
    types::{DbOptions, HnswConfig as RuvHnswConfig},
    DistanceMetric, SearchQuery, VectorDB, VectorEntry,
};

/// Default vector dimension.
/// Override by passing the actual embedding model dimension to [`RuVectorStore::open`].
pub const VECTOR_DIM: usize = 768;
const VECTOR_NORM_EPS: f32 = 1e-12;
const INSERT_JITTER_EPS: f32 = 1e-2;

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum RuVectorError {
    #[error("Vector DB error: {0}")]
    Db(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Lock poisoned")]
    LockPoisoned,
}

impl From<ruvector_core::error::RuvectorError> for RuVectorError {
    fn from(e: ruvector_core::error::RuvectorError) -> Self {
        RuVectorError::Db(e.to_string())
    }
}

// ─── Public result type ───────────────────────────────────────────────────────

/// A single vector search result.
#[derive(Debug, Clone)]
pub struct VectorResult {
    /// ID of the stored vector (matches the fact/episode ID in SQLite).
    pub id: String,
    /// Cosine distance (lower = more similar).
    pub distance: f32,
}

// ─── Store ───────────────────────────────────────────────────────────────────

/// RuVector store — manages multiple per-table `VectorDB` instances.
#[derive(Clone)]
pub struct RuVectorStore {
    root: PathBuf,
    /// Dimension of the embedding vectors (must match the active embedding model).
    dimensions: usize,
    tables: Arc<RwLock<HashMap<String, VectorDB>>>,
}

impl RuVectorStore {
    /// Open (or create) a RuVector store at the given directory.
    ///
    /// `dimensions` must equal the output dimension of the embedding model in use.
    /// Passing the wrong dimension will cause `Dimension mismatch` errors on insert.
    /// Use [`VECTOR_DIM`] as the default (384) when the embedding provider is not
    /// yet known, and prefer probing the actual embedder output at startup.
    pub async fn open(path: &Path, dimensions: usize) -> Result<Self, RuVectorError> {
        std::fs::create_dir_all(path)?;
        info!(
            "RuVector store opened at {} (dim={})",
            path.display(),
            dimensions
        );
        Ok(Self {
            root: path.to_path_buf(),
            dimensions,
            tables: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn make_db(&self, table_name: &str) -> Result<VectorDB, RuVectorError> {
        let db_path = self.root.join(format!("{table_name}.db"));
        let options = DbOptions {
            dimensions: self.dimensions,
            distance_metric: DistanceMetric::Cosine,
            storage_path: db_path.to_string_lossy().into_owned(),
            hnsw_config: Some(RuvHnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 50,
                max_elements: 10_000_000,
            }),
            quantization: None,
        };
        VectorDB::new(options).map_err(Into::into)
    }

    fn get_or_create_db(&self, table_name: &str) -> Result<(), RuVectorError> {
        let has = self
            .tables
            .read()
            .map_err(|_| RuVectorError::LockPoisoned)?
            .contains_key(table_name);

        if !has {
            let db = self.make_db(table_name)?;
            self.tables
                .write()
                .map_err(|_| RuVectorError::LockPoisoned)?
                .insert(table_name.to_string(), db);
        }
        Ok(())
    }

    /// Ensure the standard vector tables exist (idempotent).
    pub async fn ensure_tables(&self) -> Result<(), RuVectorError> {
        for name in &["facts_vec", "episodes_vec"] {
            self.get_or_create_db(name)?;
            info!("Ensured RuVector table: {name}");
        }
        Ok(())
    }

    /// Add vectors to a table. `ids` and `vectors` must have the same length.
    pub async fn add_vectors(
        &self,
        table_name: &str,
        ids: Vec<String>,
        _contents: Vec<String>,
        vectors: Vec<Vec<f32>>,
        _timestamps: Vec<String>,
        _source_type: &str,
    ) -> Result<(), RuVectorError> {
        self.get_or_create_db(table_name)?;
        let tables = self
            .tables
            .read()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        let db = tables
            .get(table_name)
            .ok_or_else(|| RuVectorError::TableNotFound(table_name.to_string()))?;

        let count = ids.len();
        for (id, vector) in ids.into_iter().zip(vectors.into_iter()) {
            let safe_vector = sanitize_vector_for_insert(vector, self.dimensions, &id);
            let entry = VectorEntry {
                id: Some(id),
                vector: safe_vector,
                metadata: None,
            };
            db.insert(entry)?;
        }
        info!("Added {count} vectors to '{table_name}'");
        Ok(())
    }

    /// Search for the most similar vectors using cosine distance.
    ///
    /// Returns results sorted by distance ascending (closest first).
    pub async fn search(
        &self,
        table_name: &str,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<VectorResult>, RuVectorError> {
        // Lazy-open on first use so search does not fail with TableNotFound
        // when callers skipped an explicit ensure_tables() step.
        self.get_or_create_db(table_name)?;

        let tables = self
            .tables
            .read()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        let db = tables
            .get(table_name)
            .ok_or_else(|| RuVectorError::TableNotFound(table_name.to_string()))?;

        let safe_query = sanitize_vector_for_query(query_vector, self.dimensions, table_name);
        let results = db.search(SearchQuery {
            vector: safe_query,
            k: top_k,
            filter: None,
            ef_search: None,
        })?;

        Ok(results
            .into_iter()
            .map(|r| VectorResult {
                id: r.id,
                distance: sanitize_distance(r.score),
            })
            .collect())
    }

    /// Delete a vector by ID from a table.
    pub async fn delete(&self, table_name: &str, id: &str) -> Result<(), RuVectorError> {
        let tables = self
            .tables
            .read()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        if let Some(db) = tables.get(table_name) {
            db.delete(id)?;
        }
        Ok(())
    }

    /// Get the row count for a table.
    pub async fn table_count(&self, table_name: &str) -> Result<usize, RuVectorError> {
        let tables = self
            .tables
            .read()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        Ok(tables
            .get(table_name)
            .map(|db| db.len().unwrap_or(0))
            .unwrap_or(0))
    }

    /// List all open table names.
    pub async fn table_names(&self) -> Result<Vec<String>, RuVectorError> {
        Ok(self
            .tables
            .read()
            .map_err(|_| RuVectorError::LockPoisoned)?
            .keys()
            .cloned()
            .collect())
    }
}

fn sanitize_distance(score: f32) -> f32 {
    if !score.is_finite() {
        return f32::MAX;
    }
    if score < 0.0 {
        return 0.0;
    }
    score
}

fn sanitize_vector_for_insert(vector: Vec<f32>, dimensions: usize, id: &str) -> Vec<f32> {
    let mut out = sanitize_vector_for_query(vector, dimensions, id);
    apply_insert_jitter(&mut out, id);
    normalize_in_place_or_fallback(&mut out, id);
    out
}

fn sanitize_vector_for_query(vector: Vec<f32>, dimensions: usize, seed: &str) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }
    if vector.len() != dimensions || vector.iter().any(|x| !x.is_finite()) {
        warn!(
            expected_dim = dimensions,
            got_dim = vector.len(),
            "Invalid embedding shape/value; using deterministic fallback"
        );
        return deterministic_fallback_vector(seed, dimensions);
    }

    let mut out = vector;
    if !normalize_in_place_or_fallback(&mut out, seed) {
        return deterministic_fallback_vector(seed, dimensions);
    }
    out
}

fn normalize_in_place_or_fallback(vector: &mut [f32], seed: &str) -> bool {
    if vector.is_empty() {
        return true;
    }

    let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq <= VECTOR_NORM_EPS {
        let fallback = deterministic_fallback_vector(seed, vector.len());
        vector.copy_from_slice(&fallback);
        return false;
    }

    let norm = norm_sq.sqrt();
    for v in vector.iter_mut() {
        *v /= norm;
    }
    true
}

fn apply_insert_jitter(vector: &mut [f32], id: &str) {
    if vector.is_empty() {
        return;
    }

    // Deterministic id-based perturbation to avoid pathological duplicate vectors.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in id.as_bytes() {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }

    let idx_a = (hash as usize) % vector.len();
    let idx_b = (hash.rotate_left(17) as usize) % vector.len();
    let sign_a = if (hash & 1) == 0 { 1.0 } else { -1.0 };
    let sign_b = if ((hash >> 1) & 1) == 0 { -1.0 } else { 1.0 };
    vector[idx_a] += sign_a * INSERT_JITTER_EPS;
    vector[idx_b] += sign_b * INSERT_JITTER_EPS * 0.5;
}

fn deterministic_fallback_vector(seed: &str, dimensions: usize) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }

    let mut state: u64 = 0xcbf29ce484222325;
    for b in seed.as_bytes() {
        state ^= u64::from(*b);
        state = state.wrapping_mul(0x100000001b3);
    }
    if state == 0 {
        state = 1;
    }

    let mut out = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545f4914f6cdd1d);
        let unit = (r as f64 / u64::MAX as f64) as f32;
        out.push(unit * 2.0 - 1.0);
    }

    let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= VECTOR_NORM_EPS {
        let mut unit = vec![0.0_f32; dimensions];
        unit[0] = 1.0;
        return unit;
    }
    for v in &mut out {
        *v /= norm;
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store() -> (RuVectorStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = RuVectorStore::open(dir.path(), VECTOR_DIM).await.unwrap();
        (store, dir)
    }

    fn unit_vec(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; VECTOR_DIM];
        v[axis] = 1.0;
        v
    }

    #[tokio::test]
    async fn test_open_and_ensure_tables() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        let mut tables = store.table_names().await.unwrap();
        tables.sort();
        assert!(tables.contains(&"episodes_vec".to_string()));
        assert!(tables.contains(&"facts_vec".to_string()));
    }

    #[tokio::test]
    async fn test_ensure_tables_idempotent() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();
        store.ensure_tables().await.unwrap();
    }

    #[tokio::test]
    async fn test_add_and_count() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        store
            .add_vectors(
                "episodes_vec",
                vec!["ep001".into()],
                vec![],
                vec![unit_vec(0)],
                vec![],
                "episodic",
            )
            .await
            .unwrap();

        assert_eq!(store.table_count("episodes_vec").await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_vector_search() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        let v1 = unit_vec(0);
        let v2 = unit_vec(1);
        let mut v3 = vec![0.0f32; VECTOR_DIM];
        v3[0] = 0.9;
        v3[1] = 0.1;

        store
            .add_vectors(
                "facts_vec",
                vec!["f1".into(), "f2".into(), "f3".into()],
                vec![],
                vec![v1.clone(), v2, v3],
                vec![],
                "semantic",
            )
            .await
            .unwrap();

        let results = store.search("facts_vec", v1, 2).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "f1");
    }

    #[tokio::test]
    async fn test_delete() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        store
            .add_vectors(
                "facts_vec",
                vec!["f1".into()],
                vec![],
                vec![unit_vec(0)],
                vec![],
                "semantic",
            )
            .await
            .unwrap();

        assert_eq!(store.table_count("facts_vec").await.unwrap(), 1);
        store.delete("facts_vec", "f1").await.unwrap();
        assert_eq!(store.table_count("facts_vec").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_identical_vectors_with_different_ids_do_not_panic() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        let repeated = unit_vec(0);
        for i in 0..64 {
            store
                .add_vectors(
                    "facts_vec",
                    vec![format!("dup-{i}")],
                    vec![],
                    vec![repeated.clone()],
                    vec![],
                    "semantic",
                )
                .await
                .unwrap();
        }

        let results = store.search("facts_vec", unit_vec(0), 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.distance.is_finite()));
    }

    #[tokio::test]
    async fn test_invalid_or_zero_vectors_are_sanitized() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        store
            .add_vectors(
                "facts_vec",
                vec!["zero".into(), "nan".into()],
                vec![],
                vec![vec![0.0_f32; VECTOR_DIM], vec![f32::NAN; VECTOR_DIM]],
                vec![],
                "semantic",
            )
            .await
            .unwrap();

        let results = store
            .search("facts_vec", vec![0.0_f32; VECTOR_DIM], 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.distance.is_finite()));
        assert!(results.iter().all(|r| r.distance >= 0.0));
    }
}
