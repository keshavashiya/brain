//! # Brain Storage
//!
//! Storage abstraction layer providing:
//! - SQLite for episodic memory, user profiles, procedures, and tasks
//! - LanceDB for vector-based semantic memory
//! - AES-256-GCM encryption for data at rest
//! - Schema migrations

pub mod encryption;
pub mod lance;
pub mod sqlite;

pub use encryption::Encryptor;
pub use lance::{LanceStore, VectorResult};
pub use sqlite::SqlitePool;
