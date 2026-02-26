//! # Brain Hippocampus
//!
//! Memory engine providing:
//! - Episodic memory (conversation storage with decay)
//! - Semantic memory (fact storage with vector embeddings)
//! - Procedural memory (learned workflows)
//! - Importance scoring (keyword-based, no LLM)
//! - ONNX embedding pipeline (BGE-small-en-v1.5)
//! - Hybrid search (vector ANN + BM25 FTS5 + RRF fusion)
//! - Memory consolidation (sleep cycle)

pub mod consolidation;
pub mod embedding;
pub mod episodic;
pub mod importance;
pub mod procedural;
pub mod search;
pub mod semantic;

pub use consolidation::{ConsolidationConfig, ConsolidationReport, Consolidator};
pub use embedding::{Embedder, EMBEDDING_DIM};
pub use episodic::{Episode, EpisodicStore, Session};
pub use importance::{ImportanceScorer, ImportanceSignals};
pub use search::{Memory, MemorySource, RecallConfig, RecallEngine};
pub use semantic::{Fact, NamespaceStats, SemanticResult, SemanticStore};
