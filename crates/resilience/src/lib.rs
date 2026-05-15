//! # Brain Resilience
//!
//! Resilience primitives for tool dispatch — currently a single Hystrix-style
//! three-state circuit breaker (`Closed → Open → HalfOpen`). Higher-level
//! layers (Tower-based stack: Timeout / RateLimit / LoopDetector / Retry /
//! DLQ) will compose on top.
//!
//! The breaker is observer-aware: every state transition emits a
//! `BrainEvent::BreakerStateChange` so the Live tab, `brain tail`, and
//! remote subscribers can render breaker health alongside tool calls.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::Utc;
use observe::{BrainEvent, Observer};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

// ─── Config + state ─────────────────────────────────────────────────────────

/// Tuning knobs for a [`CircuitBreaker`].
#[derive(Debug, Clone)]
pub struct BreakerConfig {
    /// Consecutive failures from `Closed` that flip the breaker to `Open`.
    pub failure_threshold: u32,
    /// How long an `Open` breaker stays open before allowing a `HalfOpen`
    /// probe.
    pub open_duration: Duration,
    /// Consecutive successes from `HalfOpen` that flip the breaker back to
    /// `Closed`.
    pub half_open_required_successes: u32,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            open_duration: Duration::from_secs(30),
            half_open_required_successes: 1,
        }
    }
}

/// Three-state Hystrix machine. `Open` carries `opened_at` so the breaker
/// can transparently transition itself to `HalfOpen` once enough wall-clock
/// time has passed — callers do not need to drive the state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen,
}

impl BreakerState {
    /// Lowercase wire form for [`BrainEvent::BreakerStateChange`].
    pub fn as_str(&self) -> &'static str {
        match self {
            BreakerState::Closed => "closed",
            BreakerState::Open { .. } => "open",
            BreakerState::HalfOpen => "half_open",
        }
    }
}

struct Inner {
    state: BreakerState,
    /// Consecutive failures while in `Closed`. Reset on success.
    consecutive_failures: u32,
    /// Consecutive successes while in `HalfOpen`. Reset on failure.
    probe_successes: u32,
}

// ─── Breaker ────────────────────────────────────────────────────────────────

/// Per-target circuit breaker. The `tool_id` is opaque to the breaker but
/// is what consumers (the capability router, audit log, Live tab) key on,
/// so it is included in every emitted [`BrainEvent::BreakerStateChange`].
pub struct CircuitBreaker {
    tool_id: String,
    config: BreakerConfig,
    inner: Mutex<Inner>,
    observer: Option<Arc<dyn Observer>>,
}

impl CircuitBreaker {
    pub fn new(tool_id: impl Into<String>, config: BreakerConfig) -> Self {
        Self {
            tool_id: tool_id.into(),
            config,
            inner: Mutex::new(Inner {
                state: BreakerState::Closed,
                consecutive_failures: 0,
                probe_successes: 0,
            }),
            observer: None,
        }
    }

    /// Wire an `Observer` so state transitions emit
    /// `BrainEvent::BreakerStateChange`. Unwired breakers stay silent.
    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    pub fn config(&self) -> &BreakerConfig {
        &self.config
    }

    /// Snapshot the current state. Does **not** drive the
    /// `Open → HalfOpen` transition; use [`Self::is_open`] for that.
    pub async fn state(&self) -> BreakerState {
        self.inner.lock().await.state
    }

    /// Returns `true` only when the breaker is actively rejecting calls.
    /// `Open` transparently transitions to `HalfOpen` once
    /// `config.open_duration` has elapsed since the open transition —
    /// callers in the hot path can rely on this method alone for gating.
    pub async fn is_open(&self) -> bool {
        let mut inner = self.inner.lock().await;
        if let BreakerState::Open { opened_at } = inner.state {
            if opened_at.elapsed() >= self.config.open_duration {
                self.transition(&mut inner, BreakerState::HalfOpen).await;
                return false;
            }
            return true;
        }
        false
    }

    pub async fn record_success(&self) {
        let mut inner = self.inner.lock().await;
        match inner.state {
            BreakerState::Closed => {
                inner.consecutive_failures = 0;
            }
            BreakerState::HalfOpen => {
                inner.probe_successes += 1;
                if inner.probe_successes >= self.config.half_open_required_successes {
                    self.transition(&mut inner, BreakerState::Closed).await;
                }
            }
            BreakerState::Open { .. } => {
                // Shouldn't fire — the caller is expected to gate via
                // `is_open` before issuing the request. Treat as a probe
                // (move to HalfOpen with one success banked) so a manual
                // success doesn't get lost.
                self.transition(&mut inner, BreakerState::HalfOpen).await;
                inner.probe_successes = 1;
                if inner.probe_successes >= self.config.half_open_required_successes {
                    self.transition(&mut inner, BreakerState::Closed).await;
                }
            }
        }
    }

    pub async fn record_failure(&self) {
        let mut inner = self.inner.lock().await;
        match inner.state {
            BreakerState::Closed => {
                inner.consecutive_failures += 1;
                if inner.consecutive_failures >= self.config.failure_threshold {
                    self.transition(
                        &mut inner,
                        BreakerState::Open {
                            opened_at: Instant::now(),
                        },
                    )
                    .await;
                }
            }
            BreakerState::HalfOpen => {
                self.transition(
                    &mut inner,
                    BreakerState::Open {
                        opened_at: Instant::now(),
                    },
                )
                .await;
            }
            BreakerState::Open { .. } => {
                // Already open — no-op. Renewing `opened_at` would extend
                // the cooldown unfairly when retries pile in.
            }
        }
    }

    async fn transition(&self, inner: &mut Inner, to: BreakerState) {
        let from = inner.state;
        if from == to {
            return;
        }
        inner.state = to;
        match to {
            BreakerState::Closed => {
                inner.consecutive_failures = 0;
                inner.probe_successes = 0;
                info!(tool_id = %self.tool_id, "Circuit breaker closed");
            }
            BreakerState::Open { .. } => {
                inner.probe_successes = 0;
                warn!(
                    tool_id = %self.tool_id,
                    open_duration_secs = self.config.open_duration.as_secs(),
                    "Circuit breaker OPEN",
                );
            }
            BreakerState::HalfOpen => {
                inner.probe_successes = 0;
                debug!(tool_id = %self.tool_id, "Circuit breaker entering HalfOpen probe");
            }
        }
        if let Some(observer) = &self.observer {
            let _ = observer
                .publish(BrainEvent::BreakerStateChange {
                    id: Uuid::new_v4(),
                    tool_id: self.tool_id.clone(),
                    from: from.as_str().to_string(),
                    to: to.as_str().to_string(),
                    ts: Utc::now(),
                })
                .await;
        }
    }
}

// ─── Per-tool registry ──────────────────────────────────────────────────────

/// Owns one [`CircuitBreaker`] per `tool_id`. The capability router queries
/// it via the [`intent::BreakerCheck`] impl to exclude `Open` tools from
/// scoring; the dispatch site records success/failure after each tool call.
pub struct BreakerRegistry {
    breakers: RwLock<HashMap<String, Arc<CircuitBreaker>>>,
    config: BreakerConfig,
    observer: Option<Arc<dyn Observer>>,
}

impl BreakerRegistry {
    /// Build a registry that lazily creates per-tool breakers with the
    /// provided default config. Each breaker is wired to the same observer
    /// (if any) so all transitions reach the bus.
    pub fn new(config: BreakerConfig) -> Self {
        Self {
            breakers: RwLock::new(HashMap::new()),
            config,
            observer: None,
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Look up the breaker for `tool_id`, creating one on first access.
    pub async fn get_or_create(&self, tool_id: &str) -> Arc<CircuitBreaker> {
        if let Some(existing) = self.breakers.read().await.get(tool_id) {
            return existing.clone();
        }
        let mut guard = self.breakers.write().await;
        // Re-check under the write lock — another task may have raced us.
        if let Some(existing) = guard.get(tool_id) {
            return existing.clone();
        }
        let mut cb = CircuitBreaker::new(tool_id, self.config.clone());
        if let Some(obs) = &self.observer {
            cb = cb.with_observer(obs.clone());
        }
        let arc = Arc::new(cb);
        guard.insert(tool_id.to_string(), arc.clone());
        arc
    }

    /// Snapshot lookup — `None` if no breaker has been minted yet.
    pub async fn get(&self, tool_id: &str) -> Option<Arc<CircuitBreaker>> {
        self.breakers.read().await.get(tool_id).cloned()
    }

    /// Convenience wrapper: mint the breaker for `tool_id` if needed and
    /// record a success.
    pub async fn record_success(&self, tool_id: &str) {
        self.get_or_create(tool_id).await.record_success().await;
    }

    /// Convenience wrapper: mint the breaker for `tool_id` if needed and
    /// record a failure.
    pub async fn record_failure(&self, tool_id: &str) {
        self.get_or_create(tool_id).await.record_failure().await;
    }

    pub async fn len(&self) -> usize {
        self.breakers.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.breakers.read().await.is_empty()
    }
}

#[async_trait]
impl intent::BreakerCheck for BreakerRegistry {
    async fn is_open(&self, tool_id: &str) -> bool {
        // Only query existing breakers — never mint one just to read state.
        // A tool with no recorded outcomes is considered closed.
        match self.get(tool_id).await {
            Some(cb) => cb.is_open().await,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests;
