//! # Brain Storage
//!
//! Storage abstraction layer providing:
//! - SQLite for episodic memory, user profiles, procedures, and tasks
//! - RuVector for vector-based semantic memory (HNSW + GNN self-learning)
//! - Encryption (deferred to v1.1)
//! - Schema migrations

pub mod encryption;
pub mod sqlite;
pub mod lance; // TODO: Replace with RuVector integration

pub use encryption::Encryptor;
pub use lance::{LanceStore, VectorResult}; // TODO: Replace with RuVectorStore
pub use sqlite::SqlitePool;
