//! # Brain Storage
//!
//! Storage abstraction layer providing:
//! - SQLite for episodic memory, user profiles, procedures, and tasks
//! - RuVector for vector-based semantic memory (HNSW + GNN self-learning)
//! - Encryption (deferred to v1.1)
//! - Schema migrations

pub mod encryption;
pub mod ruvector;
pub mod sqlite;

pub use encryption::Encryptor;
pub use ruvector::{RuVectorStore, VectorResult};
pub use sqlite::SqlitePool;
