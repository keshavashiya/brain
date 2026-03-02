//! # Brain Storage
//!
//! Storage abstraction layer providing:
//! - SQLite for episodic memory, semantic facts, procedures, and FTS5 index
//! - RuVector for vector-based semantic memory (HNSW)
//! - Encryption at rest (AES-256-GCM + Argon2id)
//! - Schema migrations

pub mod encryption;
pub mod ruvector;
pub mod sqlite;

pub use encryption::Encryptor;
pub use ruvector::{RuVectorStore, VectorResult};
pub use sqlite::SqlitePool;
