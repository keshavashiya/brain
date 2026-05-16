//! # Brain Hippocampus
//!
//! Memory engine providing:
//! - Episodic memory (conversation storage with decay)
//! - Semantic memory (fact storage with vector embeddings)
//! - Procedural memory (learned workflows)
//! - Importance scoring (keyword-based, no LLM)
//! - Embedding pipeline (Ollama / OpenAI-compatible)
//! - Hybrid search (vector ANN + BM25 FTS5 + RRF fusion)
//! - Memory consolidation (sleep cycle)

pub mod compactor;
pub mod consolidation;
pub mod dual_memory;
pub mod embedding;
pub mod episodic;
pub mod graph;
pub mod importance;
pub mod search;
pub mod semantic;

pub use compactor::{CompactConfig, CompactStats, Compactor, DefaultCompactor};
pub use consolidation::{
    ConsolidationConfig, ConsolidationReport, Consolidator, PromotionCandidate,
};
pub use dual_memory::{DualMemoryError, DualMemoryReader, MemoryEntry};
pub use embedding::{Embedder, EmbeddingError};
pub use episodic::{Episode, EpisodicStore, Session};
pub use graph::{Edge, EdgeKind, EpisodicGraph, GraphError, Node, NodeKind, SqliteGraph};
pub use importance::{ImportanceScorer, ImportanceSignals};
pub use search::{Memory, MemorySource, RecallConfig, RecallEngine};
pub use semantic::{Fact, NamespaceStats, SemanticResult, SemanticStore};
