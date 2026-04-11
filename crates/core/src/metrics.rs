//! Shared subsystem metrics used across subsystems for Prometheus exposition.
//!
//! Kept in `brain_core` so that subsystems (signal, hippocampus, backends) can
//! increment counters without taking a dependency on the HTTP adapter. The HTTP
//! adapter is the only component that renders these as Prometheus text.

use std::sync::atomic::{AtomicU64, Ordering};

/// Cross-subsystem counters covering memory activity, embedding, consolidation,
/// circuit-breaker events, and intent classification. Safe to `Arc<>` and share.
#[derive(Default, Debug)]
pub struct SubsystemMetrics {
    // Embedding
    pub embedding_requests_total: AtomicU64,
    pub embedding_fallbacks_total: AtomicU64,

    // Consolidation
    pub consolidation_runs_total: AtomicU64,
    pub consolidation_pruned_total: AtomicU64,
    pub consolidation_promoted_total: AtomicU64,

    // Circuit breaker
    pub circuit_open_total: AtomicU64,
    pub circuit_resets_total: AtomicU64,

    // Thalamus
    pub intent_classifications_total: AtomicU64,
    pub intent_llm_fallbacks_total: AtomicU64,
}

impl SubsystemMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn inc_embedding_request(&self) {
        self.embedding_requests_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_embedding_fallback(&self) {
        self.embedding_fallbacks_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_consolidation_run(&self) {
        self.consolidation_runs_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_consolidation_pruned(&self, n: u64) {
        self.consolidation_pruned_total
            .fetch_add(n, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_consolidation_promoted(&self, n: u64) {
        self.consolidation_promoted_total
            .fetch_add(n, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_circuit_open(&self) {
        self.circuit_open_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_circuit_reset(&self) {
        self.circuit_resets_total.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_intent_classification(&self) {
        self.intent_classifications_total
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inc_intent_llm_fallback(&self) {
        self.intent_llm_fallbacks_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot of all counters for rendering.
    pub fn snapshot(&self) -> SubsystemSnapshot {
        SubsystemSnapshot {
            embedding_requests_total: self.embedding_requests_total.load(Ordering::Relaxed),
            embedding_fallbacks_total: self.embedding_fallbacks_total.load(Ordering::Relaxed),
            consolidation_runs_total: self.consolidation_runs_total.load(Ordering::Relaxed),
            consolidation_pruned_total: self.consolidation_pruned_total.load(Ordering::Relaxed),
            consolidation_promoted_total: self.consolidation_promoted_total.load(Ordering::Relaxed),
            circuit_open_total: self.circuit_open_total.load(Ordering::Relaxed),
            circuit_resets_total: self.circuit_resets_total.load(Ordering::Relaxed),
            intent_classifications_total: self.intent_classifications_total.load(Ordering::Relaxed),
            intent_llm_fallbacks_total: self.intent_llm_fallbacks_total.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SubsystemSnapshot {
    pub embedding_requests_total: u64,
    pub embedding_fallbacks_total: u64,
    pub consolidation_runs_total: u64,
    pub consolidation_pruned_total: u64,
    pub consolidation_promoted_total: u64,
    pub circuit_open_total: u64,
    pub circuit_resets_total: u64,
    pub intent_classifications_total: u64,
    pub intent_llm_fallbacks_total: u64,
}
