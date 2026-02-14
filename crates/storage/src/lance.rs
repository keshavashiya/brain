//! LanceDB vector storage backend.
//!
//! Provides vector table management, ANN search,
//! and semantic memory storage with 384-dim embeddings.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Float32Array, RecordBatch, RecordBatchIterator, StringArray, FixedSizeListArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::query::{ExecutableQuery, QueryBase};
use thiserror::Error;
use tracing::info;

/// Default vector dimensions (BGE-small-en-v1.5).
const VECTOR_DIM: i32 = 384;

/// Errors from the LanceDB layer.
#[derive(Debug, Error)]
pub enum LanceError {
    #[error("LanceDB error: {0}")]
    Lance(#[from] lancedb::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Table not found: {0}")]
    TableNotFound(String),
}

/// LanceDB vector store wrapper.
///
/// Manages two tables:
/// - `episodes_vec` -- vector embeddings for episodic memory
/// - `facts_vec` -- vector embeddings for semantic facts
pub struct LanceStore {
    db: lancedb::Connection,
}

impl LanceStore {
    /// Open a LanceDB connection at the given directory.
    pub async fn open(path: &Path) -> Result<Self, LanceError> {
        std::fs::create_dir_all(path).map_err(|e| {
            LanceError::Lance(lancedb::Error::Runtime {
                message: format!("Cannot create LanceDB directory: {e}"),
            })
        })?;

        let db = lancedb::connect(path.to_str().unwrap_or("."))
            .execute()
            .await?;

        info!("LanceDB opened at {}", path.display());
        Ok(Self { db })
    }

    /// Get the Arrow schema for vector tables.
    fn vector_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    VECTOR_DIM,
                ),
                false,
            ),
            Field::new("timestamp", DataType::Utf8, false),
            Field::new("source_type", DataType::Utf8, false),
        ]))
    }

    /// Ensure the required vector tables exist.
    pub async fn ensure_tables(&self) -> Result<(), LanceError> {
        let table_names = self.db.table_names().execute().await?;

        for table_name in &["episodes_vec", "facts_vec"] {
            if !table_names.contains(&table_name.to_string()) {
                let schema = Self::vector_schema();
                // Create empty table with schema
                let batch = RecordBatch::new_empty(schema.clone());
                let batches = RecordBatchIterator::new(
                    vec![Ok(batch)],
                    schema,
                );
                self.db.create_table(*table_name, Box::new(batches))
                    .execute()
                    .await?;
                info!("Created LanceDB table: {table_name}");
            }
        }

        Ok(())
    }

    /// Add vectors to a table.
    pub async fn add_vectors(
        &self,
        table_name: &str,
        ids: Vec<String>,
        contents: Vec<String>,
        vectors: Vec<Vec<f32>>,
        timestamps: Vec<String>,
        source_type: &str,
    ) -> Result<(), LanceError> {
        let count = ids.len();
        let schema = Self::vector_schema();

        let id_array = Arc::new(StringArray::from(ids));
        let content_array = Arc::new(StringArray::from(contents));
        let timestamp_array = Arc::new(StringArray::from(timestamps));
        let source_type_array = Arc::new(StringArray::from(vec![source_type.to_string(); count]));

        // Flatten vectors into a single buffer for FixedSizeListArray
        let flat_values: Vec<f32> = vectors.into_iter().flatten().collect();
        let values_array = Arc::new(Float32Array::from(flat_values));
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = Arc::new(FixedSizeListArray::new(
            field, VECTOR_DIM, values_array, None,
        ));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array, content_array, vector_array, timestamp_array, source_type_array],
        )?;

        let table = self.db.open_table(table_name).execute().await?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(Box::new(batches)).execute().await?;

        info!("Added {count} vectors to {table_name}");
        Ok(())
    }

    /// Search for similar vectors using ANN.
    ///
    /// Returns (id, content, distance) tuples sorted by distance (ascending).
    pub async fn search(
        &self,
        table_name: &str,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<VectorResult>, LanceError> {
        let table = self.db.open_table(table_name).execute().await?;

        let results = table
            .vector_search(query_vector)?
            .limit(top_k)
            .execute()
            .await?;

        use futures::TryStreamExt;
        let batches: Vec<RecordBatch> = results.try_collect().await?;

        let mut output = Vec::new();
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let contents = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(ids), Some(contents), Some(distances)) = (ids, contents, distances) {
                for i in 0..batch.num_rows() {
                    output.push(VectorResult {
                        id: ids.value(i).to_string(),
                        content: contents.value(i).to_string(),
                        distance: distances.value(i),
                    });
                }
            }
        }

        Ok(output)
    }

    /// Get the row count for a table.
    pub async fn table_count(&self, table_name: &str) -> Result<usize, LanceError> {
        let table = self.db.open_table(table_name).execute().await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }

    /// List all table names.
    pub async fn table_names(&self) -> Result<Vec<String>, LanceError> {
        Ok(self.db.table_names().execute().await?)
    }
}

/// A single vector search result.
#[derive(Debug, Clone)]
pub struct VectorResult {
    pub id: String,
    pub content: String,
    pub distance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store() -> (LanceStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = LanceStore::open(dir.path()).await.unwrap();
        (store, dir)
    }

    fn dummy_vector() -> Vec<f32> {
        vec![0.1; VECTOR_DIM as usize]
    }

    #[tokio::test]
    async fn test_open_and_ensure_tables() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        let tables = store.table_names().await.unwrap();
        assert!(tables.contains(&"episodes_vec".to_string()));
        assert!(tables.contains(&"facts_vec".to_string()));
    }

    #[tokio::test]
    async fn test_ensure_tables_idempotent() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();
        store.ensure_tables().await.unwrap(); // Should not fail
    }

    #[tokio::test]
    async fn test_add_and_count() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        store.add_vectors(
            "episodes_vec",
            vec!["ep-001".into()],
            vec!["Hello world".into()],
            vec![dummy_vector()],
            vec!["2026-01-01T00:00:00".into()],
            "episodic",
        ).await.unwrap();

        let count = store.table_count("episodes_vec").await.unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_vector_search() {
        let (store, _dir) = temp_store().await;
        store.ensure_tables().await.unwrap();

        // Insert 3 vectors with slightly different values
        let mut v1 = vec![0.0f32; VECTOR_DIM as usize];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; VECTOR_DIM as usize];
        v2[1] = 1.0;
        let mut v3 = vec![0.0f32; VECTOR_DIM as usize];
        v3[0] = 0.9;
        v3[1] = 0.1;

        store.add_vectors(
            "facts_vec",
            vec!["f1".into(), "f2".into(), "f3".into()],
            vec!["Rust is great".into(), "Python is popular".into(), "Rust is fast".into()],
            vec![v1.clone(), v2, v3],
            vec!["2026-01-01".into(), "2026-01-02".into(), "2026-01-03".into()],
            "semantic",
        ).await.unwrap();

        // Query with v1 -- should return f1 as closest
        let results = store.search("facts_vec", v1, 2).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "f1");
    }
}
