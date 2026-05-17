//! Per-tool token-bucket rate limiter. Pure-data (no background task)
//! — refill is computed on every acquire from `(tokens, last_refill)`.
//! Avoids tying a `tokio::spawn` to registry lifetime and keeps the
//! limiter usable inside test reactors without a runtime tick driver.
//!
//! Acquire semantics: the bucket starts full at `burst_capacity` and
//! refills at `tokens_per_refill / refill_interval` tokens per second.
//! `acquire` consumes one token; if none are available it sleeps for
//! exactly the gap until the next whole token, then re-checks (which
//! handles spurious concurrent drains).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use tracing::debug;

/// Tuning knobs for a [`RateLimiter`].
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Number of tokens added per [`Self::refill_interval`]. Combined
    /// with the interval, this is the steady-state rate.
    pub tokens_per_refill: u32,
    /// How often a full `tokens_per_refill` worth of tokens accrues.
    /// Refill is computed proportionally, so a 1-second interval with
    /// 10 tokens behaves identically to a 100ms interval with 1 token.
    pub refill_interval: Duration,
    /// Maximum tokens the bucket holds — the burst ceiling.
    pub burst_capacity: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            tokens_per_refill: 10,
            refill_interval: Duration::from_secs(1),
            burst_capacity: 20,
        }
    }
}

impl RateLimitConfig {
    /// Steady-state tokens per second derived from
    /// `tokens_per_refill / refill_interval`. Returns `f64::INFINITY`
    /// if the interval is zero (treated as "no rate limit").
    fn refill_rate_per_sec(&self) -> f64 {
        let secs = self.refill_interval.as_secs_f64();
        if secs <= 0.0 {
            f64::INFINITY
        } else {
            self.tokens_per_refill as f64 / secs
        }
    }
}

struct Bucket {
    /// Available tokens; fractional during partial refills.
    tokens: f64,
    /// Last instant the bucket was refilled.
    last_refill: Instant,
}

/// Per-target token-bucket rate limiter. The `tool_id` is opaque to
/// the limiter — kept for symmetry with [`crate::CircuitBreaker`] so
/// log lines correlate.
pub struct RateLimiter {
    tool_id: String,
    config: RateLimitConfig,
    inner: Mutex<Bucket>,
}

impl RateLimiter {
    pub fn new(tool_id: impl Into<String>, config: RateLimitConfig) -> Self {
        let burst = config.burst_capacity as f64;
        Self {
            tool_id: tool_id.into(),
            config,
            inner: Mutex::new(Bucket {
                tokens: burst,
                last_refill: Instant::now(),
            }),
        }
    }

    pub fn tool_id(&self) -> &str {
        &self.tool_id
    }

    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Block until one token is available, then consume it.
    pub async fn acquire(&self) {
        let rate = self.config.refill_rate_per_sec();
        let burst = self.config.burst_capacity as f64;
        loop {
            let wait = {
                let mut bucket = self.inner.lock().expect("rate-limit bucket poisoned");
                let now = Instant::now();
                let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
                let refilled = (bucket.tokens + elapsed * rate).min(burst);
                bucket.last_refill = now;
                if refilled >= 1.0 {
                    bucket.tokens = refilled - 1.0;
                    return;
                }
                bucket.tokens = refilled;
                let needed = 1.0 - refilled;
                let secs = if rate.is_finite() && rate > 0.0 {
                    needed / rate
                } else {
                    0.0
                };
                Duration::from_secs_f64(secs)
            };
            debug!(tool_id = %self.tool_id, wait_ms = wait.as_millis() as u64, "rate limiter awaiting refill");
            tokio::time::sleep(wait).await;
        }
    }

    /// Non-blocking acquire: returns `true` if a token was consumed,
    /// `false` if the bucket is empty. Sync — the critical section is
    /// pure arithmetic and never crosses an await point, so a `std`
    /// mutex is the right primitive.
    pub fn try_acquire(&self) -> bool {
        let rate = self.config.refill_rate_per_sec();
        let burst = self.config.burst_capacity as f64;
        let mut bucket = self.inner.lock().expect("rate-limit bucket poisoned");
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        let refilled = (bucket.tokens + elapsed * rate).min(burst);
        bucket.last_refill = now;
        if refilled >= 1.0 {
            bucket.tokens = refilled - 1.0;
            true
        } else {
            bucket.tokens = refilled;
            false
        }
    }

    /// Inspect available tokens (refill is applied first). Test helper
    /// — production code should use [`Self::acquire`] / [`Self::try_acquire`].
    pub fn available(&self) -> f64 {
        let rate = self.config.refill_rate_per_sec();
        let burst = self.config.burst_capacity as f64;
        let mut bucket = self.inner.lock().expect("rate-limit bucket poisoned");
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        let refilled = (bucket.tokens + elapsed * rate).min(burst);
        bucket.tokens = refilled;
        bucket.last_refill = now;
        refilled
    }
}

/// Owns one [`RateLimiter`] per `tool_id`. Mirrors
/// [`crate::BreakerRegistry`] so the resilience layer can wire both
/// from the same configuration surface.
pub struct RateLimitRegistry {
    limiters: RwLock<HashMap<String, Arc<RateLimiter>>>,
    config: RateLimitConfig,
}

impl RateLimitRegistry {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            limiters: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Look up the limiter for `tool_id`, creating one on first access.
    pub fn get_or_create(&self, tool_id: &str) -> Arc<RateLimiter> {
        if let Some(existing) = self
            .limiters
            .read()
            .expect("rate-limit registry poisoned")
            .get(tool_id)
        {
            return existing.clone();
        }
        let mut guard = self.limiters.write().expect("rate-limit registry poisoned");
        if let Some(existing) = guard.get(tool_id) {
            return existing.clone();
        }
        let arc = Arc::new(RateLimiter::new(tool_id, self.config.clone()));
        guard.insert(tool_id.to_string(), arc.clone());
        arc
    }

    /// Snapshot lookup — `None` if no limiter has been minted yet.
    pub fn get(&self, tool_id: &str) -> Option<Arc<RateLimiter>> {
        self.limiters
            .read()
            .expect("rate-limit registry poisoned")
            .get(tool_id)
            .cloned()
    }

    /// Convenience: mint the limiter for `tool_id` if needed and block
    /// for a token.
    pub async fn acquire(&self, tool_id: &str) {
        self.get_or_create(tool_id).acquire().await;
    }

    pub fn len(&self) -> usize {
        self.limiters
            .read()
            .expect("rate-limit registry poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.limiters
            .read()
            .expect("rate-limit registry poisoned")
            .is_empty()
    }
}
