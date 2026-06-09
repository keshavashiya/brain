//! Shared subsystem metrics used across subsystems for Prometheus exposition.
//!
//! A dependency-free leaf crate so that subsystems (signal, backends) can
//! increment counters without taking a dependency on `core` or the HTTP
//! adapter. The HTTP adapter is the only component that renders these as
//! Prometheus text.

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

/// Sentinel stored in a gauge whose value is currently unavailable — either a
/// platform that can't probe it (Windows degrades to `None`) or the window
/// before the first sample lands. Decodes to `None` in the snapshot. `u64::MAX`
/// is safe because no real RSS / disk byte-count, connection count, or CPU
/// milli-percent ever reaches it.
const GAUGE_UNSET: u64 = u64::MAX;

/// Scale factor for the fixed-point CPU-percent encoding: percent is stored as
/// integer milli-percent (`42.5%` → `42_500`) so the gauge stays a lock-free
/// `AtomicU64`. Keeps three decimal places, ample for a utilisation reading.
const CPU_PCT_SCALE: f64 = 1000.0;

/// Sampled runtime resource gauges — the *level* counterparts to
/// [`SubsystemMetrics`]' counters. Filled in by the resource sampler (one
/// bounded background task) and read via [`ResourceMetrics::snapshot`] for
/// rendering and threshold checks. Safe to `Arc<>` and share.
///
/// Every gauge is a *level* (current value), not a monotonic counter, so the
/// sampler overwrites with [`set_*`](ResourceMetrics::set_rss_bytes) rather
/// than adding. Unavailable gauges hold [`GAUGE_UNSET`] and read back as
/// `None`; a freshly-constructed store reads all-`None` until the first sample.
#[derive(Debug)]
pub struct ResourceMetrics {
    /// Resident set size, in bytes.
    rss_bytes: AtomicU64,
    /// Process CPU utilisation, fixed-point milli-percent (see [`CPU_PCT_SCALE`]).
    cpu_millipct: AtomicU64,
    /// Open SQLite connections held by the pool.
    open_connections: AtomicU64,
    /// `~/.brain` data-directory disk usage, in bytes.
    disk_bytes: AtomicU64,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        // All gauges start unavailable — nothing has been sampled yet.
        Self {
            rss_bytes: AtomicU64::new(GAUGE_UNSET),
            cpu_millipct: AtomicU64::new(GAUGE_UNSET),
            open_connections: AtomicU64::new(GAUGE_UNSET),
            disk_bytes: AtomicU64::new(GAUGE_UNSET),
        }
    }
}

impl ResourceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn store(slot: &AtomicU64, value: Option<u64>) {
        slot.store(value.unwrap_or(GAUGE_UNSET), Ordering::Relaxed);
    }

    #[inline]
    fn load(slot: &AtomicU64) -> Option<u64> {
        match slot.load(Ordering::Relaxed) {
            GAUGE_UNSET => None,
            v => Some(v),
        }
    }

    /// Set (or clear, with `None`) the resident-set-size gauge, in bytes.
    #[inline]
    pub fn set_rss_bytes(&self, bytes: Option<u64>) {
        Self::store(&self.rss_bytes, bytes);
    }

    /// Set (or clear, with `None`) the CPU-utilisation gauge, in percent.
    /// Stored as fixed-point milli-percent; negatives clamp to `0`.
    #[inline]
    pub fn set_cpu_pct(&self, pct: Option<f64>) {
        let encoded = pct.map(|p| (p.max(0.0) * CPU_PCT_SCALE).round() as u64);
        Self::store(&self.cpu_millipct, encoded);
    }

    /// Set (or clear, with `None`) the open-SQLite-connections gauge.
    #[inline]
    pub fn set_open_connections(&self, n: Option<u64>) {
        Self::store(&self.open_connections, n);
    }

    /// Set (or clear, with `None`) the `~/.brain` disk-usage gauge, in bytes.
    #[inline]
    pub fn set_disk_bytes(&self, bytes: Option<u64>) {
        Self::store(&self.disk_bytes, bytes);
    }

    /// Snapshot of all gauges for rendering and threshold checks. Unavailable
    /// gauges read as `None`.
    pub fn snapshot(&self) -> ResourceSnapshot {
        ResourceSnapshot {
            rss_bytes: Self::load(&self.rss_bytes),
            cpu_pct: Self::load(&self.cpu_millipct).map(|m| m as f64 / CPU_PCT_SCALE),
            open_connections: Self::load(&self.open_connections),
            disk_bytes: Self::load(&self.disk_bytes),
        }
    }
}

/// Point-in-time view of [`ResourceMetrics`]. A `None` gauge was unavailable at
/// sample time (unsupported platform or pre-first-sample).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ResourceSnapshot {
    pub rss_bytes: Option<u64>,
    pub cpu_pct: Option<f64>,
    pub open_connections: Option<u64>,
    pub disk_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_gauges_read_unavailable() {
        let m = ResourceMetrics::new();
        assert_eq!(m.snapshot(), ResourceSnapshot::default());
    }

    #[test]
    fn set_gauges_round_trip() {
        let m = ResourceMetrics::new();
        m.set_rss_bytes(Some(512 * 1024 * 1024));
        m.set_open_connections(Some(4));
        m.set_disk_bytes(Some(1_234_567));

        let snap = m.snapshot();
        assert_eq!(snap.rss_bytes, Some(512 * 1024 * 1024));
        assert_eq!(snap.open_connections, Some(4));
        assert_eq!(snap.disk_bytes, Some(1_234_567));
    }

    #[test]
    fn cpu_fixed_point_round_trips_to_three_decimals() {
        let m = ResourceMetrics::new();
        m.set_cpu_pct(Some(42.5));
        assert_eq!(m.snapshot().cpu_pct, Some(42.5));

        // Multi-core busy loop: >100% is representable.
        m.set_cpu_pct(Some(237.125));
        assert_eq!(m.snapshot().cpu_pct, Some(237.125));

        // Negative readings clamp to zero rather than wrapping the cast.
        m.set_cpu_pct(Some(-1.0));
        assert_eq!(m.snapshot().cpu_pct, Some(0.0));
    }

    #[test]
    fn none_clears_a_previously_set_gauge() {
        let m = ResourceMetrics::new();
        m.set_rss_bytes(Some(1));
        assert_eq!(m.snapshot().rss_bytes, Some(1));
        m.set_rss_bytes(None);
        assert_eq!(m.snapshot().rss_bytes, None);
    }
}
