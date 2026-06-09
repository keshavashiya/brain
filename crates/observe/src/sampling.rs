//! Log sampling for high-volume, low-information log lines.
//!
//! Some call sites fire on every iteration of a hot loop — a resource
//! sampler heartbeat, a per-request embedding probe — where logging *every*
//! occurrence drowns the signal and inflates log volume without adding
//! information. [`LogSampler`] gates such sites so only 1 in N lines is
//! emitted, while the value behind each line (a metric, an event) is still
//! recorded every time.
//!
//! It is deliberately tiny and dependency-free: a single relaxed atomic
//! counter. Sampling decisions don't need cross-thread ordering guarantees —
//! an occasional off-by-one under concurrency only shifts *which* line is
//! kept, never the long-run 1-in-N rate.

use std::sync::atomic::{AtomicU64, Ordering};

/// A 1-in-N gate for high-volume log lines. Construct once per high-volume
/// site (or share via `Arc`) and call [`LogSampler::should_emit`] at the log
/// point:
///
/// ```
/// use brainos_observe::LogSampler;
///
/// let sampler = LogSampler::one_in(100);
/// for i in 0..250 {
///     if sampler.should_emit() {
///         // ~1 in 100 of these run; the other 99 are skipped.
///         let _ = i;
///     }
/// }
/// ```
#[derive(Debug)]
pub struct LogSampler {
    /// Emit one line for every `one_in_n` calls. Clamped to a minimum of 1,
    /// so a misconfigured `0` means "emit everything" rather than "never".
    one_in_n: u64,
    count: AtomicU64,
}

impl LogSampler {
    /// Emit 1 in `n` calls. `n <= 1` disables sampling (every call emits).
    pub fn one_in(n: u32) -> Self {
        Self {
            one_in_n: (n as u64).max(1),
            count: AtomicU64::new(0),
        }
    }

    /// A sampler that emits on every call — the explicit "sampling off" form.
    pub fn unsampled() -> Self {
        Self::one_in(1)
    }

    /// Whether this call should emit its log line. Returns `true` on the
    /// 1st call and every `one_in_n`-th call thereafter, so a freshly built
    /// sampler always logs its first occurrence (you never lose the first
    /// instance of a newly-hot site).
    pub fn should_emit(&self) -> bool {
        let n = self.count.fetch_add(1, Ordering::Relaxed);
        n.is_multiple_of(self.one_in_n)
    }

    /// The configured rate (post-clamp); 1 means unsampled.
    pub fn rate(&self) -> u64 {
        self.one_in_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_one_emits_every_call() {
        let s = LogSampler::one_in(1);
        assert!((0..10).all(|_| s.should_emit()));
    }

    #[test]
    fn zero_is_clamped_to_emit_everything() {
        let s = LogSampler::one_in(0);
        assert_eq!(s.rate(), 1);
        assert!(s.should_emit());
    }

    #[test]
    fn emits_first_then_every_nth() {
        let s = LogSampler::one_in(4);
        // calls:        0    1      2      3      4    5
        // should_emit: true false  false  false  true false
        let pattern: Vec<bool> = (0..6).map(|_| s.should_emit()).collect();
        assert_eq!(
            pattern,
            vec![true, false, false, false, true, false],
            "first call emits, then every 4th"
        );
    }

    #[test]
    fn long_run_rate_is_one_in_n() {
        let s = LogSampler::one_in(10);
        let emitted = (0..1000).filter(|_| s.should_emit()).count();
        assert_eq!(emitted, 100, "1000 calls at 1-in-10 → 100 emitted");
    }

    #[test]
    fn unsampled_is_rate_one() {
        assert_eq!(LogSampler::unsampled().rate(), 1);
    }
}
