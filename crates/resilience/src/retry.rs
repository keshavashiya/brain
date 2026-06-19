//! Async retry combinator. Pure-data — composes with any
//! `Future<Output = Result<T, E>>` factory without taking a dep on
//! Tower or `tower-retry`. Backoff is exponential with a cap and an
//! optional full-jitter randomization; an optional
//! [`intent::BreakerCheck`] is consulted between attempts so a retry
//! cycle aborts cleanly the moment the breaker opens.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use tracing::debug;

/// Tuning knobs for [`retry`].
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Total attempts including the first call. Must be ≥ 1; values
    /// below 1 are treated as 1 (single-shot, no retry).
    pub max_attempts: u32,
    /// First-retry delay before jitter. Subsequent delays double up to
    /// `max_delay`.
    pub base_delay: Duration,
    /// Hard ceiling on the exponential backoff.
    pub max_delay: Duration,
    /// 0.0 = deterministic backoff; 1.0 = full jitter (uniform random
    /// over `[0, computed_delay]`). Clamped to `[0.0, 1.0]`.
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter_factor: 1.0,
        }
    }
}

/// Reason the retry cycle returned without exhausting `max_attempts`.
#[derive(Debug)]
pub enum RetryOutcome<E> {
    /// All attempts produced an error; this is the final one.
    Exhausted(E),
    /// The breaker tied to this tool went `Open` between attempts; the
    /// inner cycle stopped. Carries the most recent error.
    BreakerOpenAbort(BreakerOpenAbort<E>),
}

/// Companion struct for [`RetryOutcome::BreakerOpenAbort`].
#[derive(Debug)]
pub struct BreakerOpenAbort<E> {
    pub tool_id: String,
    pub last_error: E,
}

impl<E> RetryOutcome<E> {
    pub fn last_error(self) -> E {
        match self {
            RetryOutcome::Exhausted(e) => e,
            RetryOutcome::BreakerOpenAbort(a) => a.last_error,
        }
    }
}

/// Run `f` up to `config.max_attempts` times. The first invocation is
/// not delayed; subsequent attempts wait `min(base_delay * 2^(n-1),
/// max_delay)` with optional full jitter.
///
/// If `breaker_check` is `Some`, [`intent::BreakerCheck::is_open`] is
/// consulted between attempts; an open breaker short-circuits the
/// cycle and returns [`RetryOutcome::BreakerOpenAbort`] carrying the
/// last error.
///
/// On the first `Ok(T)`, the function returns immediately.
pub async fn retry<F, Fut, T, E>(
    config: &RetryConfig,
    breaker_check: Option<(Arc<dyn intent::BreakerCheck>, &str)>,
    mut f: F,
) -> Result<T, RetryOutcome<E>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let attempts = config.max_attempts.max(1);
    let mut last_err: Option<E> = None;

    for attempt in 0..attempts {
        if attempt > 0 {
            // Gate against the breaker before sleeping — a fresh Open
            // signal lets us abort without burning the backoff.
            if let Some((check, tool_id)) = &breaker_check {
                if check.is_open(tool_id).await {
                    debug!(tool_id = %tool_id, "Retry abort: breaker is open");
                    return Err(RetryOutcome::BreakerOpenAbort(BreakerOpenAbort {
                        tool_id: tool_id.to_string(),
                        last_error: last_err
                            .expect("invariant: last_err is set on every non-first attempt"),
                    }));
                }
            }
            let delay = compute_delay(config, attempt);
            tokio::time::sleep(delay).await;
        }

        match f().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                debug!(
                    attempt = attempt + 1,
                    max = attempts,
                    "retry: attempt failed"
                );
                last_err = Some(e);
            }
        }
    }

    Err(RetryOutcome::Exhausted(
        last_err.expect("invariant: at least one attempt ran"),
    ))
}

/// Exposed so tests can pin the backoff curve without invoking the
/// actual retry machinery.
pub fn compute_delay(config: &RetryConfig, attempt: u32) -> Duration {
    debug_assert!(attempt > 0, "attempt index is 1-based for delay math");
    let shift = (attempt - 1).min(31);
    let exp_ms = config.base_delay.as_millis().saturating_mul(1u128 << shift);
    let cap_ms = config.max_delay.as_millis();
    let bounded_ms = exp_ms.min(cap_ms);
    let bounded_ms_u64: u64 = bounded_ms.try_into().unwrap_or(u64::MAX);
    let base = Duration::from_millis(bounded_ms_u64);
    let jitter = config.jitter_factor.clamp(0.0, 1.0);
    if jitter == 0.0 || bounded_ms_u64 == 0 {
        return base;
    }
    // Full-jitter scheme (Marc Brooker / AWS): uniform over [0, delay]
    // when jitter_factor == 1.0; partial otherwise.
    let max_jitter_ms = (bounded_ms_u64 as f64 * jitter as f64) as u64;
    let drawn = rand::random_range(0..=max_jitter_ms);
    let deterministic_floor = bounded_ms_u64.saturating_sub(max_jitter_ms);
    Duration::from_millis(deterministic_floor + drawn)
}
