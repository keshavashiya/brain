//! Atomic counters and gauge caching exposed at `GET /metrics`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Cached gauge snapshot so `/metrics` doesn't hit SQLite on every scrape.
#[derive(Default)]
pub struct GaugeCache {
    pub fact_count: i64,
    pub episode_count: i64,
    pub last_updated_ms: u64,
}

const GAUGE_TTL_MS: u64 = 10_000;

/// Atomic counters exposed at `GET /metrics` in Prometheus text format.
#[derive(Default)]
pub struct Metrics {
    /// Total POST /v1/signals requests processed.
    pub signals_total: AtomicU64,
    /// Signals that returned a non-5xx response.
    pub signals_ok: AtomicU64,
    /// Signals that returned a 5xx error.
    pub signals_error: AtomicU64,
    /// Total POST /v1/memory/search requests.
    pub search_total: AtomicU64,
    /// Total GET /v1/memory/facts requests.
    pub facts_total: AtomicU64,
    /// Cumulative POST /v1/signals processing time in milliseconds.
    pub signals_latency_ms_total: AtomicU64,
    /// Cached memory gauges (refreshed on scrape, 10s TTL).
    pub gauge_cache: std::sync::Mutex<GaugeCache>,
}

impl Metrics {
    /// Refresh cached fact/episode gauges from SQLite (throttled to every 10s).
    pub fn refresh_gauges(&self, processor: &signal::SignalProcessor) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut guard = self.gauge_cache.lock().unwrap_or_else(|e| e.into_inner());
        if now_ms.saturating_sub(guard.last_updated_ms) < GAUGE_TTL_MS {
            return;
        }
        let pool = processor.episodic().pool();
        let fact_count = pool
            .with_conn(|conn| {
                let mut stmt = conn.prepare("SELECT COUNT(*) FROM semantic_facts")?;
                let row: i64 = stmt.query_row([], |r| r.get(0))?;
                Ok(row)
            })
            .unwrap_or(0);
        let episode_count = pool
            .with_conn(|conn| {
                let mut stmt = conn.prepare("SELECT COUNT(*) FROM episodes")?;
                let row: i64 = stmt.query_row([], |r| r.get(0))?;
                Ok(row)
            })
            .unwrap_or(0);
        guard.fact_count = fact_count;
        guard.episode_count = episode_count;
        guard.last_updated_ms = now_ms;
    }

    /// Render counters as Prometheus plain-text format (text/plain; version=0.0.4).
    pub fn render(&self, subsystems: &brain::metrics::SubsystemMetrics) -> String {
        let signals_total = self.signals_total.load(Ordering::Relaxed);
        let signals_ok = self.signals_ok.load(Ordering::Relaxed);
        let signals_error = self.signals_error.load(Ordering::Relaxed);
        let search_total = self.search_total.load(Ordering::Relaxed);
        let facts_total = self.facts_total.load(Ordering::Relaxed);
        let latency_ms = self.signals_latency_ms_total.load(Ordering::Relaxed);
        let sub = subsystems.snapshot();
        let (fact_count, episode_count) = {
            let g = self.gauge_cache.lock().unwrap_or_else(|e| e.into_inner());
            (g.fact_count, g.episode_count)
        };

        format!(
            "# HELP brain_signals_total Total signal requests received.\n\
             # TYPE brain_signals_total counter\n\
             brain_signals_total {signals_total}\n\
             # HELP brain_signals_ok_total Successful signal requests.\n\
             # TYPE brain_signals_ok_total counter\n\
             brain_signals_ok_total {signals_ok}\n\
             # HELP brain_signals_error_total Failed signal requests (5xx).\n\
             # TYPE brain_signals_error_total counter\n\
             brain_signals_error_total {signals_error}\n\
             # HELP brain_search_total Total memory search requests.\n\
             # TYPE brain_search_total counter\n\
             brain_search_total {search_total}\n\
             # HELP brain_facts_total Total memory facts requests.\n\
             # TYPE brain_facts_total counter\n\
             brain_facts_total {facts_total}\n\
             # HELP brain_signals_latency_ms_total Cumulative signal processing latency in ms.\n\
             # TYPE brain_signals_latency_ms_total counter\n\
             brain_signals_latency_ms_total {latency_ms}\n\
             # HELP brain_facts_count Total semantic facts currently stored.\n\
             # TYPE brain_facts_count gauge\n\
             brain_facts_count {fact_count}\n\
             # HELP brain_episodes_count Total episodes currently stored.\n\
             # TYPE brain_episodes_count gauge\n\
             brain_episodes_count {episode_count}\n\
             # HELP brain_embedding_requests_total Total embedding requests.\n\
             # TYPE brain_embedding_requests_total counter\n\
             brain_embedding_requests_total {embed_req}\n\
             # HELP brain_embedding_fallbacks_total Times deterministic fallback vectors were used.\n\
             # TYPE brain_embedding_fallbacks_total counter\n\
             brain_embedding_fallbacks_total {embed_fb}\n\
             # HELP brain_consolidation_runs_total Number of consolidation cycles completed.\n\
             # TYPE brain_consolidation_runs_total counter\n\
             brain_consolidation_runs_total {cons_runs}\n\
             # HELP brain_consolidation_pruned_total Episodes pruned during consolidation.\n\
             # TYPE brain_consolidation_pruned_total counter\n\
             brain_consolidation_pruned_total {cons_pruned}\n\
             # HELP brain_consolidation_promoted_total Episodes promoted to semantic facts.\n\
             # TYPE brain_consolidation_promoted_total counter\n\
             brain_consolidation_promoted_total {cons_promoted}\n\
             # HELP brain_circuit_open_total Times a circuit breaker opened.\n\
             # TYPE brain_circuit_open_total counter\n\
             brain_circuit_open_total {cb_open}\n\
             # HELP brain_circuit_resets_total Times a circuit breaker reset on success.\n\
             # TYPE brain_circuit_resets_total counter\n\
             brain_circuit_resets_total {cb_reset}\n\
             # HELP brain_intent_classifications_total Total intent classifications.\n\
             # TYPE brain_intent_classifications_total counter\n\
             brain_intent_classifications_total {intents}\n\
             # HELP brain_intent_llm_fallbacks_total Intents resolved via LLM fallback.\n\
             # TYPE brain_intent_llm_fallbacks_total counter\n\
             brain_intent_llm_fallbacks_total {intent_llm}\n",
            embed_req = sub.embedding_requests_total,
            embed_fb = sub.embedding_fallbacks_total,
            cons_runs = sub.consolidation_runs_total,
            cons_pruned = sub.consolidation_pruned_total,
            cons_promoted = sub.consolidation_promoted_total,
            cb_open = sub.circuit_open_total,
            cb_reset = sub.circuit_resets_total,
            intents = sub.intent_classifications_total,
            intent_llm = sub.intent_llm_fallbacks_total,
        )
    }
}
