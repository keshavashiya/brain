//! Async timeout combinator. Thin wrapper over
//! [`tokio::time::timeout`] that returns a typed error distinct from
//! the inner future's error so callers can tell "the operation timed
//! out" apart from "the operation failed."
//!
//! Cancellation safety: when the deadline expires, the inner future is
//! **dropped**. Callers must ensure the wrapped future is cancellation
//! safe (no half-applied side effects on drop). Tool-call surfaces in
//! `mcphost` and `terminal` are cancellation safe by construction —
//! they don't hold mutable external state across `.await` points.

use std::future::Future;
use std::time::Duration;

/// Outcome of [`timeout`]. `Elapsed` is the deadline-hit case;
/// `Inner` carries the inner future's own error verbatim.
#[derive(Debug)]
pub enum TimeoutError<E> {
    /// The deadline elapsed before the inner future completed.
    Elapsed,
    /// The inner future completed with an error before the deadline.
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for TimeoutError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeoutError::Elapsed => write!(f, "operation timed out"),
            TimeoutError::Inner(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for TimeoutError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TimeoutError::Elapsed => None,
            TimeoutError::Inner(e) => Some(e),
        }
    }
}

/// Run `fut` with a deadline. Returns:
/// - `Ok(T)` if the inner future completes within `duration` with `Ok`.
/// - `Err(TimeoutError::Inner(e))` if it completes within the deadline
///   with `Err(e)`.
/// - `Err(TimeoutError::Elapsed)` if the deadline expires first — the
///   inner future is dropped.
pub async fn timeout<F, T, E>(duration: Duration, fut: F) -> Result<T, TimeoutError<E>>
where
    F: Future<Output = Result<T, E>>,
{
    match tokio::time::timeout(duration, fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(TimeoutError::Inner(e)),
        Err(_elapsed) => Err(TimeoutError::Elapsed),
    }
}
