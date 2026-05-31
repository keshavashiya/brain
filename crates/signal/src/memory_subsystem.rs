//! The memory subsystem bundled out of [`SignalProcessor`](crate::SignalProcessor).
//!
//! Episodic + semantic stores, the embedding provider and its query cache,
//! the recall engine, and the dual-memory reader are one cohesive subsystem;
//! grouping them keeps the processor's top-level field list a roster of
//! distinct collaborators instead of the internals of this one.
//!
//! Fields are `pub(crate)` because the pipeline modules (`recall`,
//! `pipeline::*`, `terminal_graph_mirror`) reach them directly through
//! `self.memory.<field>` rather than going through accessors.

use std::sync::Arc;

/// Memory collaborators owned by the [`SignalProcessor`](crate::SignalProcessor).
pub(crate) struct MemorySubsystem {
    pub(crate) episodic: hippocampus::EpisodicStore,
    pub(crate) semantic: Option<hippocampus::SemanticStore>,
    /// Embedding provider. `Embedder::embed` takes `&self`, so no external
    /// lock is needed — concurrent embed calls are safe and the HTTP client
    /// inside the provider already serializes appropriately. `Arc`-shared so
    /// the terminal graph sink can embed node bodies through the same
    /// provider without minting a second one.
    pub(crate) embedder: Option<Arc<hippocampus::Embedder>>,
    /// Actual output dimension of the active embedding provider (probed at startup).
    pub(crate) embedding_dim: usize,
    /// LRU cache for embedded query/text vectors. Keyed by a fast hash of
    /// the input text; only successful embeddings are cached (fallback
    /// vectors are deterministic and skipped to avoid polluting the cache
    /// during transient provider outages).
    pub(crate) embedding_cache: std::sync::Mutex<lru::LruCache<u64, Arc<Vec<f32>>>>,
    pub(crate) recall_engine: hippocampus::RecallEngine,
    /// Dual-memory reader — graph-first, legacy-fallback point lookups.
    /// Wired so future read paths can resolve an id against the
    /// episodic graph without bypassing the legacy `episodes` table for
    /// content written before the graph schema landed.
    pub(crate) dual_memory_reader: Option<hippocampus::DualMemoryReader>,
}
