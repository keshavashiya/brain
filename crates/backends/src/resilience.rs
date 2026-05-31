//! Resilience helpers for HTTP backends — pairs the canonical
//! [`resilience::CircuitBreaker`] with retry + reqwest-aware
//! transient-error detection. Re-exports `CircuitBreaker` so existing
//! backend code keeps the `use crate::resilience::CircuitBreaker` import
//! shape.
//!
//! Metrics: each HTTP backend wires its breaker to a [`MetricsObserver`]
//! so `BreakerStateChange` events drive the per-subsystem
//! `inc_circuit_open` / `inc_circuit_reset` counters exported on
//! `/metrics`. The observer also rebroadcasts via a small local channel
//! so anything that does subscribe — today nothing — still sees the
//! events.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use metrics::SubsystemMetrics;
use observe::{BrainEvent, ObserveError, Observer};
use resilience::BreakerConfig;
use tokio::sync::broadcast;

pub use resilience::CircuitBreaker;

/// Default capacity for the metrics-observer rebroadcast channel. Tiny
/// because nothing in-tree actually subscribes; the value exists only to
/// satisfy `Observer::subscribe`.
const REBROADCAST_CAPACITY: usize = 16;

/// `Observer` that converts `BreakerStateChange` events from a circuit
/// breaker into backend-side `SubsystemMetrics` counters. Anything else
/// is rebroadcast unchanged so the observer composes cleanly if another
/// listener is ever attached.
pub struct MetricsObserver {
    metrics: Arc<SubsystemMetrics>,
    tx: broadcast::Sender<BrainEvent>,
}

impl MetricsObserver {
    pub fn new(metrics: Arc<SubsystemMetrics>) -> Arc<Self> {
        let (tx, _) = broadcast::channel(REBROADCAST_CAPACITY);
        Arc::new(Self { metrics, tx })
    }
}

#[async_trait]
impl Observer for MetricsObserver {
    async fn publish(&self, ev: BrainEvent) -> Result<(), ObserveError> {
        if let BrainEvent::BreakerStateChange { to, .. } = &ev {
            match to.as_str() {
                "open" => self.metrics.inc_circuit_open(),
                "closed" => self.metrics.inc_circuit_reset(),
                _ => {}
            }
        }
        // send() returns Err when there are no receivers; not a problem.
        let _ = self.tx.send(ev);
        Ok(())
    }

    fn subscribe(&self) -> broadcast::Receiver<BrainEvent> {
        self.tx.subscribe()
    }
}

/// Build a `CircuitBreaker` for an HTTP backend with a familiar
/// `(name, failure_threshold, cooldown_secs)` shape and an optional
/// metrics handle. `cooldown_secs` becomes
/// [`BreakerConfig::open_duration`]; the half-open probe needs one
/// success to close, matching the prior atomic-breaker semantics.
pub fn http_breaker(
    name: &str,
    failure_threshold: u32,
    cooldown_secs: u64,
    metrics: Option<Arc<SubsystemMetrics>>,
) -> CircuitBreaker {
    let cfg = BreakerConfig {
        failure_threshold,
        open_duration: Duration::from_secs(cooldown_secs),
        half_open_required_successes: 1,
    };
    let mut cb = CircuitBreaker::new(name, cfg);
    if let Some(m) = metrics {
        cb = cb.with_observer(MetricsObserver::new(m));
    }
    cb
}

/// Returns true if the HTTP status is transient (worth retrying):
/// any 5xx, 408 Request Timeout, or 429 Too Many Requests.
fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS // 429
        || status == reqwest::StatusCode::REQUEST_TIMEOUT // 408
}

/// Returns true if the request-error is transient (worth retrying).
/// Treats timeouts and connect errors as transient; for status-bearing
/// errors, defers to [`is_transient_status`].
fn is_transient(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    err.status().is_some_and(is_transient_status)
}

/// Send an HTTP request with retry + circuit breaker.
pub async fn resilient_send<F>(
    build_request: F,
    circuit_breaker: &CircuitBreaker,
    max_retries: u32,
    retry_base_ms: u64,
) -> Result<reqwest::Response, cortex::actions::ActionError>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    if circuit_breaker.is_open().await {
        return Err(cortex::actions::ActionError::ExecutionFailed(format!(
            "{} circuit breaker is open — backend disabled until cooldown elapses",
            circuit_breaker.tool_id()
        )));
    }

    let attempts = 1 + max_retries;
    let mut last_err = None;

    for attempt in 0..attempts {
        if attempt > 0 {
            let delay = retry_base_ms * (1u64 << (attempt - 1).min(5));
            tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
        }

        match build_request().send().await {
            Ok(response) => {
                if response.status().is_success() || !is_transient_status(response.status()) {
                    circuit_breaker.record_success().await;
                    return Ok(response);
                }
                let status = response.status();
                tracing::debug!(
                    backend = %circuit_breaker.tool_id(),
                    attempt = attempt + 1,
                    status = %status,
                    "Transient HTTP error, will retry"
                );
                last_err = Some(format!("HTTP {}", status));
            }
            Err(e) => {
                if !is_transient(&e) {
                    circuit_breaker.record_failure().await;
                    return Err(cortex::actions::ActionError::ExecutionFailed(e.to_string()));
                }
                tracing::debug!(
                    backend = %circuit_breaker.tool_id(),
                    attempt = attempt + 1,
                    error = %e,
                    "Transient error, will retry"
                );
                last_err = Some(e.to_string());
            }
        }
    }

    circuit_breaker.record_failure().await;
    Err(cortex::actions::ActionError::ExecutionFailed(
        last_err.unwrap_or_else(|| "all retry attempts exhausted".to_string()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(threshold: u32, cooldown_secs: u64) -> BreakerConfig {
        BreakerConfig {
            failure_threshold: threshold,
            open_duration: Duration::from_secs(cooldown_secs),
            half_open_required_successes: 1,
        }
    }

    #[tokio::test]
    async fn http_breaker_closed_by_default() {
        let cb = http_breaker("test", 3, 60, None);
        assert!(!cb.is_open().await);
    }

    #[tokio::test]
    async fn breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new("test", cfg(3, 60));
        cb.record_failure().await;
        cb.record_failure().await;
        assert!(
            !cb.is_open().await,
            "should still be closed below threshold"
        );
        cb.record_failure().await;
        assert!(cb.is_open().await, "should be open at threshold");
    }

    #[tokio::test]
    async fn breaker_resets_on_success() {
        let cb = CircuitBreaker::new("test", cfg(3, 60));
        cb.record_failure().await;
        cb.record_failure().await;
        cb.record_success().await;
        assert!(!cb.is_open().await);
        cb.record_failure().await;
        cb.record_failure().await;
        assert!(!cb.is_open().await, "should be closed — counter was reset");
    }

    #[tokio::test]
    async fn breaker_half_open_after_cooldown() {
        let cb = CircuitBreaker::new("test", cfg(2, 0));
        cb.record_failure().await;
        cb.record_failure().await;
        // open_duration == 0 means the next is_open() call transitions to
        // HalfOpen and returns false.
        assert!(
            !cb.is_open().await,
            "should be half-open after zero cooldown"
        );
    }
}
