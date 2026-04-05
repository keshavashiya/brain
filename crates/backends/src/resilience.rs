//! Resilience primitives — circuit breaker and retry logic for HTTP backends.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Tracks consecutive failures and opens a circuit after a threshold is reached.
pub struct CircuitBreaker {
    consecutive_failures: AtomicU32,
    last_failure_epoch_ms: AtomicU64,
    threshold: u32,
    cooldown_ms: u64,
    pub name: String,
}

impl CircuitBreaker {
    pub fn new(name: &str, threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            last_failure_epoch_ms: AtomicU64::new(0),
            threshold,
            cooldown_ms: cooldown_secs * 1000,
            name: name.to_string(),
        }
    }

    pub fn is_open(&self) -> bool {
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        if failures < self.threshold {
            return false;
        }
        let last_fail = self.last_failure_epoch_ms.load(Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        if now.saturating_sub(last_fail) >= self.cooldown_ms {
            return false;
        }
        true
    }

    pub fn record_success(&self) {
        let prev = self.consecutive_failures.swap(0, Ordering::Relaxed);
        if prev >= self.threshold {
            tracing::info!(backend = %self.name, "Circuit breaker closed (backend recovered)");
        }
    }

    pub fn record_failure(&self) {
        let prev = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_failure_epoch_ms.store(now, Ordering::Relaxed);
        if prev + 1 == self.threshold {
            tracing::warn!(
                backend = %self.name,
                threshold = self.threshold,
                cooldown_secs = self.cooldown_ms / 1000,
                "Circuit breaker OPEN — backend disabled until cooldown elapses"
            );
        }
    }
}

/// Returns true if the error is transient (worth retrying).
fn is_transient(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    if let Some(status) = err.status() {
        return status.is_server_error();
    }
    false
}

/// Returns true if the HTTP status is transient (worth retrying).
fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
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
    if circuit_breaker.is_open() {
        return Err(cortex::actions::ActionError::ExecutionFailed(format!(
            "{} circuit breaker is open — backend disabled until cooldown elapses",
            circuit_breaker.name
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
                    circuit_breaker.record_success();
                    return Ok(response);
                }
                let status = response.status();
                tracing::debug!(
                    backend = %circuit_breaker.name,
                    attempt = attempt + 1,
                    status = %status,
                    "Transient HTTP error, will retry"
                );
                last_err = Some(format!("HTTP {}", status));
            }
            Err(e) => {
                if !is_transient(&e) {
                    circuit_breaker.record_failure();
                    return Err(cortex::actions::ActionError::ExecutionFailed(e.to_string()));
                }
                tracing::debug!(
                    backend = %circuit_breaker.name,
                    attempt = attempt + 1,
                    error = %e,
                    "Transient error, will retry"
                );
                last_err = Some(e.to_string());
            }
        }
    }

    circuit_breaker.record_failure();
    Err(cortex::actions::ActionError::ExecutionFailed(
        last_err.unwrap_or_else(|| "all retry attempts exhausted".to_string()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_by_default() {
        let cb = CircuitBreaker::new("test", 3, 60);
        assert!(!cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new("test", 3, 60);
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open(), "should still be closed below threshold");
        cb.record_failure();
        assert!(cb.is_open(), "should be open at threshold");
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let cb = CircuitBreaker::new("test", 3, 60);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert!(!cb.is_open());
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open(), "should be closed — counter was reset");
    }

    #[test]
    fn test_circuit_breaker_half_open_after_cooldown() {
        let cb = CircuitBreaker::new("test", 2, 0);
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open(), "should be half-open after zero cooldown");
    }
}
