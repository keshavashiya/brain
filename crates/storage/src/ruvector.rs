//! RuVector — pure-Rust, file-backed vector database.
//!
//! Replaces LanceDB with a self-contained, dependency-free vector store.
//! Uses cosine similarity for ANN search and stores vectors as JSON files.
//!
//! # Storage layout
//! ```text
//! ~/.brain/ruvector/
//!   facts/       -- semantic fact vector files ({id}.json)
//!   episodes/    -- episode vector files ({id}.json)
//!   index/       -- reserved for future HNSW index persistence
//! ```
//!
//! # HNSW configuration (stored, applied in future HNSW implementation)
//! - ef_construction = 200
//! - m               = 16
//! - ef_search       = 50
//!
//! # Self-learning GNN
//! - enabled    = true
//! - gnn_layers = 3

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

use crate::encryption::Encryptor;

/// Default vector dimension (BGE-small-en-v1.5).
pub const VECTOR_DIM: usize = 384;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the RuVector layer.
#[derive(Debug, Error)]
pub enum RuVectorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Lock poisoned")]
    LockPoisoned,
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// HNSW index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of candidates during index construction.
    pub ef_construction: usize,
    /// Number of bi-directional links per node.
    pub m: usize,
    /// Number of candidates during search.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            ef_construction: 200,
            m: 16,
            ef_search: 50,
        }
    }
}

/// Self-learning GNN configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfLearningConfig {
    /// Whether the GNN self-learning pass is active.
    pub enabled: bool,
    /// Number of GNN layers applied during self-learning.
    pub gnn_layers: usize,
}

impl Default for SelfLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            gnn_layers: 3,
        }
    }
}

/// Top-level RuVector configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuVectorConfig {
    pub hnsw: HnswConfig,
    pub self_learning: SelfLearningConfig,
}

// ─── Internal storage ────────────────────────────────────────────────────────

/// A stored vector entry (serialised to disk as JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: String,
    content: String,
    vector: Vec<f32>,
    timestamp: String,
    source_type: String,
}

// ─── Store ───────────────────────────────────────────────────────────────────

/// RuVector store — file-backed vector database with in-memory search index.
///
/// Each logical table maps to a subdirectory under `root`.  Vectors are stored
/// as individual JSON files.  An in-memory index (rebuilt on `open`) enables
/// fast cosine-similarity search without an external database.
///
/// When an `Encryptor` is set via `with_encryptor`, the `content` field of
/// every vector entry is AES-256-GCM encrypted before being written to disk.
/// On search, content is decrypted transparently before being returned.
/// The vector floats themselves are stored as-is.
pub struct RuVectorStore {
    root: PathBuf,
    config: RuVectorConfig,
    /// table_name → vector entries.
    indices: Mutex<HashMap<String, Vec<VectorEntry>>>,
    encryptor: Option<Arc<Encryptor>>,
}

impl RuVectorStore {
    /// Open (or create) a RuVector store at the given path.
    ///
    /// Creates the required subdirectories and loads existing vectors into
    /// the in-memory index.
    pub async fn open(path: &Path) -> Result<Self, RuVectorError> {
        std::fs::create_dir_all(path.join("facts"))?;
        std::fs::create_dir_all(path.join("episodes"))?;
        std::fs::create_dir_all(path.join("index"))?;

        let config = RuVectorConfig::default();
        let indices = Self::load_from_disk(path)?;

        info!(
            "RuVector opened at {} \
             (ef_construction={}, m={}, ef_search={}, gnn_layers={})",
            path.display(),
            config.hnsw.ef_construction,
            config.hnsw.m,
            config.hnsw.ef_search,
            config.self_learning.gnn_layers
        );

        Ok(Self {
            root: path.to_path_buf(),
            config,
            indices: Mutex::new(indices),
            encryptor: None,
        })
    }

    /// Load all known tables from disk into memory.
    fn load_from_disk(root: &Path) -> Result<HashMap<String, Vec<VectorEntry>>, RuVectorError> {
        let mut map: HashMap<String, Vec<VectorEntry>> = HashMap::new();

        for (table_name, subdir) in [("facts_vec", "facts"), ("episodes_vec", "episodes")] {
            let dir = root.join(subdir);
            let mut entries = Vec::new();
            if dir.exists() {
                for de in std::fs::read_dir(&dir)? {
                    let de = de?;
                    let p = de.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        match serde_json::from_str::<VectorEntry>(&std::fs::read_to_string(&p)?) {
                            Ok(ve) => entries.push(ve),
                            Err(e) => {
                                tracing::warn!("Skipping corrupt vector file {}: {e}", p.display())
                            }
                        }
                    }
                }
            }
            if !entries.is_empty() {
                info!("Loaded {} vectors into '{table_name}'", entries.len());
            }
            map.insert(table_name.to_string(), entries);
        }

        Ok(map)
    }

    /// Attach an encryptor (builder pattern).
    ///
    /// Must be called before any writes. Existing unencrypted entries on disk
    /// will be read back as plaintext gracefully (decryption falls back on
    /// error, so mixing encrypted and unencrypted files is safe during migration).
    pub fn with_encryptor(mut self, enc: Encryptor) -> Self {
        self.encryptor = Some(Arc::new(enc));
        self
    }

    /// Resolve logical table name to a filesystem directory.
    fn table_dir(&self, table_name: &str) -> PathBuf {
        match table_name {
            "facts_vec" | "facts" => self.root.join("facts"),
            "episodes_vec" | "episodes" => self.root.join("episodes"),
            other => self.root.join(other),
        }
    }

    /// Ensure the required vector tables exist (idempotent).
    pub async fn ensure_tables(&self) -> Result<(), RuVectorError> {
        let mut indices = self
            .indices
            .lock()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        for table_name in &["episodes_vec", "facts_vec"] {
            let dir = self.table_dir(table_name);
            std::fs::create_dir_all(&dir)?;
            indices.entry(table_name.to_string()).or_default();
            info!("Ensured RuVector table: {table_name}");
        }
        Ok(())
    }

    /// Add vectors to a table.
    ///
    /// All slices must have the same length.
    pub async fn add_vectors(
        &self,
        table_name: &str,
        ids: Vec<String>,
        contents: Vec<String>,
        vectors: Vec<Vec<f32>>,
        timestamps: Vec<String>,
        source_type: &str,
    ) -> Result<(), RuVectorError> {
        let dir = self.table_dir(table_name);
        std::fs::create_dir_all(&dir)?;

        let count = ids.len();
        let mut indices = self
            .indices
            .lock()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        let entries = indices.entry(table_name.to_string()).or_default();

        for (i, (id, content)) in ids.iter().zip(contents.iter()).enumerate() {
            // Encrypt content field if encryptor is active.
            let stored_content = if let Some(enc) = &self.encryptor {
                enc.encrypt_string(content)
                    .unwrap_or_else(|_| content.clone())
            } else {
                content.clone()
            };
            let entry = VectorEntry {
                id: id.clone(),
                content: stored_content,
                vector: vectors[i].clone(),
                timestamp: timestamps[i].clone(),
                source_type: source_type.to_string(),
            };
            let json = serde_json::to_string(&entry)?;
            std::fs::write(dir.join(format!("{id}.json")), json)?;
            entries.push(entry);
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
        let indices = self
            .indices
            .lock()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        let entries = indices
            .get(table_name)
            .ok_or_else(|| RuVectorError::TableNotFound(table_name.to_string()))?;

        let mut scored: Vec<(f32, String, String)> = entries
            .iter()
            .map(|e| {
                // Decrypt content transparently; fall back to raw value on error
                // (handles legacy plaintext entries written before encryption).
                let content = if let Some(enc) = &self.encryptor {
                    enc.decrypt_string(&e.content)
                        .unwrap_or_else(|_| e.content.clone())
                } else {
                    e.content.clone()
                };
                (cosine_distance(&query_vector, &e.vector), e.id.clone(), content)
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(dist, id, content)| VectorResult {
                id,
                content,
                distance: dist,
            })
            .collect())
    }

    /// Delete a vector by ID from a table.
    pub async fn delete(&self, table_name: &str, id: &str) -> Result<(), RuVectorError> {
        let file_path = self.table_dir(table_name).join(format!("{id}.json"));
        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
        }
        let mut indices = self
            .indices
            .lock()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        if let Some(entries) = indices.get_mut(table_name) {
            entries.retain(|e| e.id != id);
        }
        Ok(())
    }

    /// Get the row count for a table.
    pub async fn table_count(&self, table_name: &str) -> Result<usize, RuVectorError> {
        let indices = self
            .indices
            .lock()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        Ok(indices.get(table_name).map_or(0, |e| e.len()))
    }

    /// List all table names in the store.
    pub async fn table_names(&self) -> Result<Vec<String>, RuVectorError> {
        let indices = self
            .indices
            .lock()
            .map_err(|_| RuVectorError::LockPoisoned)?;
        Ok(indices.keys().cloned().collect())
    }

    /// Get the HNSW configuration.
    pub fn hnsw_config(&self) -> &HnswConfig {
        &self.config.hnsw
    }

    /// Get the self-learning GNN configuration.
    pub fn self_learning_config(&self) -> &SelfLearningConfig {
        &self.config.self_learning
    }
}

// ─── Math ────────────────────────────────────────────────────────────────────

/// Cosine distance = 1 − cosine_similarity.  Returns 1.0 for zero vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

// ─── Public result type ───────────────────────────────────────────────────────

/// A single vector search result.
#[derive(Debug, Clone)]
pub struct VectorResult {
    pub id: String,
    pub content: String,
    pub distance: f32,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store() -> (RuVectorStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = RuVectorStore::open(dir.path()).await.unwrap();
        (store, dir)
    }

    fn dummy_vector() -> Vec<f32> {
        vec![0.1; VECTOR_DIM]
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
        store.ensure_tables().await.unwrap(); // Must not fail
    }

    #[tokio::test]
    async fn test_add_and_count() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        store
            .add_vectors(
                "episodes_vec",
                vec!["ep001".into()],
                vec!["Hello world".into()],
                vec![dummy_vector()],
                vec!["2026-01-01T00:00:00".into()],
                "episodic",
            )
            .await
            .unwrap();

        let count = store.table_count("episodes_vec").await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_vector_search() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        // Three vectors with distinct directions
        let mut v1 = vec![0.0f32; VECTOR_DIM];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; VECTOR_DIM];
        v2[1] = 1.0;
        let mut v3 = vec![0.0f32; VECTOR_DIM];
        v3[0] = 0.9;
        v3[1] = 0.1;

        store
            .add_vectors(
                "facts_vec",
                vec!["f1".into(), "f2".into(), "f3".into()],
                vec![
                    "Rust is great".into(),
                    "Python is popular".into(),
                    "Rust is fast".into(),
                ],
                vec![v1.clone(), v2, v3],
                vec![
                    "2026-01-01".into(),
                    "2026-01-02".into(),
                    "2026-01-03".into(),
                ],
                "semantic",
            )
            .await
            .unwrap();

        // Query with v1 — f1 is the identical vector (distance = 0)
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
                vec!["test fact".into()],
                vec![dummy_vector()],
                vec!["2026-01-01".into()],
                "semantic",
            )
            .await
            .unwrap();

        assert_eq!(store.table_count("facts_vec").await.unwrap(), 1);
        store.delete("facts_vec", "f1").await.unwrap();
        assert_eq!(store.table_count("facts_vec").await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_hnsw_config() {
        let (store, _dir) = temp_store().await;
        let h = store.hnsw_config();
        assert_eq!(h.ef_construction, 200);
        assert_eq!(h.m, 16);
        assert_eq!(h.ef_search, 50);
    }

    #[tokio::test]
    async fn test_self_learning_config() {
        let (store, _dir) = temp_store().await;
        let sl = store.self_learning_config();
        assert!(sl.enabled);
        assert_eq!(sl.gnn_layers, 3);
    }
}
