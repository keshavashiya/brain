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

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use observe::{BrainEvent, Observer};
use tokio::sync::Mutex;
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

#[cfg(test)]
mod tests;
