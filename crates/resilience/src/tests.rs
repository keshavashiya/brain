use std::sync::Arc;
use std::time::Duration;

use observe::{BrainEvent, BroadcastObserver, Observer};
use tokio::time::sleep;

use super::*;

fn fast_config() -> BreakerConfig {
    BreakerConfig {
        failure_threshold: 2,
        open_duration: Duration::from_millis(20),
        half_open_required_successes: 1,
    }
}

#[tokio::test]
async fn defaults_to_closed() {
    let cb = CircuitBreaker::new("t", BreakerConfig::default());
    assert!(!cb.is_open().await);
    assert_eq!(cb.state().await, BreakerState::Closed);
}

#[tokio::test]
async fn closes_to_open_at_threshold() {
    let cb = CircuitBreaker::new("t", fast_config());
    cb.record_failure().await;
    assert!(!cb.is_open().await, "one failure must not open");
    cb.record_failure().await;
    assert!(cb.is_open().await, "threshold reached — must be open");
}

#[tokio::test]
async fn success_in_closed_resets_failure_count() {
    let cb = CircuitBreaker::new("t", fast_config());
    cb.record_failure().await;
    cb.record_success().await;
    cb.record_failure().await;
    assert!(!cb.is_open().await, "success must have reset counter");
}

#[tokio::test]
async fn open_transitions_to_half_open_after_duration() {
    let cb = CircuitBreaker::new("t", fast_config());
    cb.record_failure().await;
    cb.record_failure().await;
    assert!(cb.is_open().await);
    sleep(Duration::from_millis(30)).await;
    assert!(!cb.is_open().await, "must auto-transition to HalfOpen");
    assert_eq!(cb.state().await, BreakerState::HalfOpen);
}

#[tokio::test]
async fn half_open_success_closes_the_breaker() {
    let cb = CircuitBreaker::new("t", fast_config());
    cb.record_failure().await;
    cb.record_failure().await;
    sleep(Duration::from_millis(30)).await;
    let _ = cb.is_open().await; // drive Open → HalfOpen
    cb.record_success().await;
    assert_eq!(cb.state().await, BreakerState::Closed);
}

#[tokio::test]
async fn half_open_failure_reopens_the_breaker() {
    let cb = CircuitBreaker::new("t", fast_config());
    cb.record_failure().await;
    cb.record_failure().await;
    sleep(Duration::from_millis(30)).await;
    let _ = cb.is_open().await; // drive Open → HalfOpen
    cb.record_failure().await;
    assert!(matches!(cb.state().await, BreakerState::Open { .. }));
}

#[tokio::test]
async fn observer_sees_state_transitions() {
    let broadcast = BroadcastObserver::new();
    let mut rx = broadcast.subscribe();
    let cb = CircuitBreaker::new("mcp:echo:echo", fast_config())
        .with_observer(broadcast as Arc<dyn Observer>);

    cb.record_failure().await;
    cb.record_failure().await; // closed -> open
    sleep(Duration::from_millis(30)).await;
    let _ = cb.is_open().await; // open -> half_open
    cb.record_success().await; // half_open -> closed

    let mut transitions: Vec<(String, String)> = Vec::new();
    for _ in 0..3 {
        match tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {
            Ok(Ok(BrainEvent::BreakerStateChange {
                tool_id, from, to, ..
            })) => {
                assert_eq!(tool_id, "mcp:echo:echo");
                transitions.push((from, to));
            }
            other => panic!("expected BreakerStateChange, got {other:?}"),
        }
    }
    assert_eq!(
        transitions,
        vec![
            ("closed".into(), "open".into()),
            ("open".into(), "half_open".into()),
            ("half_open".into(), "closed".into()),
        ]
    );
}

#[tokio::test]
async fn open_state_is_idempotent_under_repeated_failure() {
    let cb = CircuitBreaker::new("t", fast_config());
    cb.record_failure().await;
    cb.record_failure().await; // opens
    let opened_at = match cb.state().await {
        BreakerState::Open { opened_at } => opened_at,
        other => panic!("expected Open, got {other:?}"),
    };
    cb.record_failure().await; // should NOT renew opened_at
    let again = match cb.state().await {
        BreakerState::Open { opened_at } => opened_at,
        other => panic!("expected Open, got {other:?}"),
    };
    assert_eq!(
        opened_at, again,
        "open must be idempotent on extra failures"
    );
}

#[test]
fn breaker_state_wire_strings_are_stable() {
    assert_eq!(BreakerState::Closed.as_str(), "closed");
    assert_eq!(
        BreakerState::Open {
            opened_at: std::time::Instant::now(),
        }
        .as_str(),
        "open"
    );
    assert_eq!(BreakerState::HalfOpen.as_str(), "half_open");
}

#[tokio::test]
async fn registry_creates_breakers_lazily() {
    let reg = BreakerRegistry::new(fast_config());
    assert!(reg.is_empty().await);
    assert!(reg.get("missing").await.is_none());
    let cb = reg.get_or_create("mcp:echo:echo").await;
    assert_eq!(cb.tool_id(), "mcp:echo:echo");
    assert_eq!(reg.len().await, 1);
    let again = reg.get_or_create("mcp:echo:echo").await;
    assert!(
        Arc::ptr_eq(&cb, &again),
        "must return same Arc on second call"
    );
}

#[tokio::test]
async fn registry_is_breaker_check_reports_open() {
    use intent::BreakerCheck;
    let reg = BreakerRegistry::new(fast_config());
    // Unknown tool — never minted — must report closed.
    assert!(!reg.is_open("unknown").await);
    // Mint and trip it.
    reg.record_failure("mcp:t:a").await;
    reg.record_failure("mcp:t:a").await;
    assert!(reg.is_open("mcp:t:a").await);
    assert!(!reg.is_open("mcp:t:b").await);
}

#[tokio::test]
async fn registry_record_success_resets_failures() {
    let reg = BreakerRegistry::new(fast_config());
    reg.record_failure("t").await;
    reg.record_success("t").await;
    reg.record_failure("t").await;
    use intent::BreakerCheck;
    assert!(!reg.is_open("t").await, "success must have reset");
}

#[tokio::test]
async fn registry_forwards_observer_to_new_breakers() {
    let broadcast = BroadcastObserver::new();
    let mut rx = broadcast.subscribe();
    let reg = BreakerRegistry::new(fast_config()).with_observer(broadcast as Arc<dyn Observer>);
    reg.record_failure("t").await;
    reg.record_failure("t").await; // opens
    match tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {
        Ok(Ok(BrainEvent::BreakerStateChange { tool_id, to, .. })) => {
            assert_eq!(tool_id, "t");
            assert_eq!(to, "open");
        }
        other => panic!("expected BreakerStateChange, got {other:?}"),
    }
}

// ─── Retry tests ────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU32, Ordering};

use crate::retry::compute_delay;

fn retry_config(attempts: u32, base_ms: u64, jitter: f32) -> RetryConfig {
    RetryConfig {
        max_attempts: attempts,
        base_delay: Duration::from_millis(base_ms),
        max_delay: Duration::from_millis(base_ms * 8),
        jitter_factor: jitter,
    }
}

#[tokio::test]
async fn retry_returns_first_success_without_delay() {
    let calls = Arc::new(AtomicU32::new(0));
    let cfg = retry_config(5, 0, 0.0);
    let result: Result<u32, _> = retry(&cfg, None, || {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<u32, &'static str>(42)
        }
    })
    .await;
    assert_eq!(result.ok(), Some(42));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn retry_succeeds_after_transient_failures() {
    let calls = Arc::new(AtomicU32::new(0));
    let cfg = retry_config(5, 0, 0.0);
    let result: Result<&'static str, _> = retry(&cfg, None, || {
        let calls = calls.clone();
        async move {
            let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
            if n < 3 {
                Err("flake")
            } else {
                Ok("ok")
            }
        }
    })
    .await;
    assert_eq!(result.ok(), Some("ok"));
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_exhausts_and_returns_last_error() {
    let calls = Arc::new(AtomicU32::new(0));
    let cfg = retry_config(3, 0, 0.0);
    let result: Result<(), _> = retry(&cfg, None, || {
        let calls = calls.clone();
        async move {
            let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
            Err::<(), String>(format!("attempt {n}"))
        }
    })
    .await;
    match result {
        Err(RetryOutcome::Exhausted(e)) => assert_eq!(e, "attempt 3"),
        other => panic!("expected Exhausted, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_max_attempts_zero_promoted_to_one() {
    let calls = Arc::new(AtomicU32::new(0));
    let mut cfg = retry_config(0, 0, 0.0);
    cfg.max_attempts = 0; // treated as 1
    let result: Result<(), _> = retry(&cfg, None, || {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Err::<(), &'static str>("once")
        }
    })
    .await;
    assert!(matches!(result, Err(RetryOutcome::Exhausted("once"))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct AlwaysOpen;
#[async_trait::async_trait]
impl intent::BreakerCheck for AlwaysOpen {
    async fn is_open(&self, _tool_id: &str) -> bool {
        true
    }
}

struct OpensAfter(AtomicU32);
#[async_trait::async_trait]
impl intent::BreakerCheck for OpensAfter {
    async fn is_open(&self, _tool_id: &str) -> bool {
        self.0.fetch_add(1, Ordering::SeqCst) >= 1
    }
}

#[tokio::test]
async fn retry_aborts_when_breaker_opens_mid_cycle() {
    let calls = Arc::new(AtomicU32::new(0));
    let cfg = retry_config(5, 1, 0.0);
    let breaker: Arc<dyn intent::BreakerCheck> = Arc::new(OpensAfter(AtomicU32::new(0)));
    let result: Result<(), _> = retry(&cfg, Some((breaker, "mcp:t:a")), || {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Err::<(), &'static str>("transient")
        }
    })
    .await;
    match result {
        Err(RetryOutcome::BreakerOpenAbort(abort)) => {
            assert_eq!(abort.tool_id, "mcp:t:a");
            assert_eq!(abort.last_error, "transient");
        }
        other => panic!("expected BreakerOpenAbort, got {other:?}"),
    }
    // Two attempts ran: the first (no breaker check) and the second
    // (breaker reports closed on first probe, opens on second). On the
    // third we abort before calling f.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn retry_breaker_always_open_aborts_after_first_attempt() {
    let calls = Arc::new(AtomicU32::new(0));
    let cfg = retry_config(5, 1, 0.0);
    let breaker: Arc<dyn intent::BreakerCheck> = Arc::new(AlwaysOpen);
    let result: Result<(), _> = retry(&cfg, Some((breaker, "tool:x")), || {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Err::<(), &'static str>("e")
        }
    })
    .await;
    assert!(matches!(result, Err(RetryOutcome::BreakerOpenAbort(_))));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "second attempt aborted by breaker"
    );
}

#[test]
fn compute_delay_caps_at_max_delay() {
    let cfg = RetryConfig {
        max_attempts: 10,
        base_delay: Duration::from_millis(10),
        max_delay: Duration::from_millis(40),
        jitter_factor: 0.0,
    };
    assert_eq!(compute_delay(&cfg, 1), Duration::from_millis(10));
    assert_eq!(compute_delay(&cfg, 2), Duration::from_millis(20));
    assert_eq!(compute_delay(&cfg, 3), Duration::from_millis(40));
    assert_eq!(compute_delay(&cfg, 4), Duration::from_millis(40), "capped");
    assert_eq!(
        compute_delay(&cfg, 8),
        Duration::from_millis(40),
        "still capped"
    );
}

#[test]
fn compute_delay_full_jitter_stays_within_range() {
    let cfg = RetryConfig {
        max_attempts: 10,
        base_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(1),
        jitter_factor: 1.0,
    };
    for _ in 0..32 {
        let d = compute_delay(&cfg, 2).as_millis();
        // attempt 2 → 200ms base; with full jitter, in [0, 200].
        assert!(d <= 200, "{d}ms exceeded ceiling");
    }
}

// ─── Timeout tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn timeout_returns_ok_when_inner_completes_in_time() {
    let result: Result<u32, TimeoutError<&'static str>> =
        timeout(Duration::from_millis(50), async { Ok(7) }).await;
    assert!(matches!(result, Ok(7)));
}

#[tokio::test]
async fn timeout_returns_inner_error_when_inner_fails_in_time() {
    let result: Result<u32, TimeoutError<&'static str>> =
        timeout(Duration::from_millis(50), async { Err("boom") }).await;
    match result {
        Err(TimeoutError::Inner(e)) => assert_eq!(e, "boom"),
        other => panic!("expected Inner, got {other:?}"),
    }
}

#[tokio::test]
async fn timeout_returns_elapsed_when_deadline_hits_first() {
    let result: Result<u32, TimeoutError<&'static str>> =
        timeout(Duration::from_millis(10), async {
            sleep(Duration::from_millis(80)).await;
            Ok(1)
        })
        .await;
    assert!(matches!(result, Err(TimeoutError::Elapsed)));
}

#[tokio::test]
async fn timeout_drops_inner_future_on_expiry() {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct DropFlag(Arc<AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let dropped = Arc::new(AtomicBool::new(false));
    let flag = DropFlag(dropped.clone());
    let result: Result<(), TimeoutError<&'static str>> =
        timeout(Duration::from_millis(10), async move {
            let _flag = flag; // moved into the future
            sleep(Duration::from_millis(80)).await;
            Ok(())
        })
        .await;
    assert!(matches!(result, Err(TimeoutError::Elapsed)));
    assert!(
        dropped.load(Ordering::SeqCst),
        "inner future must be dropped on expiry"
    );
}

#[test]
fn timeout_error_display_distinguishes_elapsed_and_inner() {
    let elapsed: TimeoutError<&'static str> = TimeoutError::Elapsed;
    assert_eq!(elapsed.to_string(), "operation timed out");
    let inner: TimeoutError<&'static str> = TimeoutError::Inner("nope");
    assert_eq!(inner.to_string(), "nope");
}

// ─── RateLimit tests ────────────────────────────────────────────────────────

fn rl_config(tokens_per_refill: u32, refill_ms: u64, burst: u32) -> RateLimitConfig {
    RateLimitConfig {
        tokens_per_refill,
        refill_interval: Duration::from_millis(refill_ms),
        burst_capacity: burst,
    }
}

#[tokio::test]
async fn rate_limiter_allows_burst_up_to_capacity() {
    // 1 token/sec refill, burst 5 — five instant acquires should
    // complete well under a refill interval.
    let rl = RateLimiter::new("t", rl_config(1, 1000, 5));
    let start = std::time::Instant::now();
    for _ in 0..5 {
        rl.acquire().await;
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(50),
        "burst of 5 should be ~instant, took {elapsed:?}"
    );
}

#[tokio::test]
async fn rate_limiter_blocks_when_drained() {
    // 100 tokens/sec → ~10ms per token, burst 1.
    let rl = RateLimiter::new("t", rl_config(100, 1000, 1));
    rl.acquire().await; // drains
    let start = std::time::Instant::now();
    rl.acquire().await;
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_millis(8),
        "second acquire should have waited ~10ms, got {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_millis(60),
        "second acquire should not have waited a full second, got {elapsed:?}"
    );
}

#[tokio::test]
async fn rate_limiter_refills_over_time() {
    let rl = RateLimiter::new("t", rl_config(50, 1000, 2)); // 50 tok/s, burst 2
    rl.acquire().await;
    rl.acquire().await; // drains
    assert!(!rl.try_acquire().await, "should be drained");
    sleep(Duration::from_millis(60)).await; // ≥ 3 tokens accrue, capped at burst=2
    assert!(
        rl.try_acquire().await,
        "first refill token should be present"
    );
    assert!(
        rl.try_acquire().await,
        "second refill token should be present (capped at burst)"
    );
    assert!(!rl.try_acquire().await, "third try should miss — capped");
}

#[tokio::test]
async fn rate_limiter_try_acquire_does_not_block() {
    let rl = RateLimiter::new("t", rl_config(1, 10_000, 1)); // 0.1 tok/s
    assert!(rl.try_acquire().await, "first token should be present");
    assert!(
        !rl.try_acquire().await,
        "second try should fail without sleeping"
    );
}

#[tokio::test]
async fn rate_limit_registry_creates_limiters_lazily() {
    let reg = RateLimitRegistry::new(rl_config(10, 1000, 10));
    assert!(reg.is_empty().await);
    assert!(reg.get("missing").await.is_none());
    let rl = reg.get_or_create("mcp:echo:echo").await;
    assert_eq!(rl.tool_id(), "mcp:echo:echo");
    assert_eq!(reg.len().await, 1);
    let again = reg.get_or_create("mcp:echo:echo").await;
    assert!(Arc::ptr_eq(&rl, &again), "same Arc on second call");
}

#[tokio::test]
async fn rate_limit_registry_isolates_per_tool() {
    let reg = RateLimitRegistry::new(rl_config(1, 10_000, 1));
    reg.acquire("a").await;
    // 'a' is drained; 'b' should still have its own full bucket.
    let rl_b = reg.get_or_create("b").await;
    assert!(
        rl_b.try_acquire().await,
        "tool b should not be affected by tool a's drain"
    );
}

#[test]
fn rate_limit_config_default_is_sane() {
    let c = RateLimitConfig::default();
    assert!(c.burst_capacity >= c.tokens_per_refill);
    assert!(c.refill_interval > Duration::ZERO);
}

// ─── LoopDetector tests ─────────────────────────────────────────────────────

use crate::loop_detector::canonical_json_for_test;
use serde_json::json;

fn ld_config(window: usize, threshold: u32) -> LoopDetectorConfig {
    LoopDetectorConfig { window, threshold }
}

#[tokio::test]
async fn loop_detector_passes_under_threshold() {
    let det = LoopDetector::new(ld_config(8, 4));
    let args = json!({"x": 1});
    for _ in 0..4 {
        det.check("p1", "mcp:t:a", &args).await.unwrap();
    }
    assert_eq!(det.window_len("p1").await, 4);
}

#[tokio::test]
async fn loop_detector_trips_past_threshold() {
    let det = LoopDetector::new(ld_config(8, 4));
    let args = json!({"x": 1});
    for _ in 0..4 {
        det.check("p1", "mcp:t:a", &args).await.unwrap();
    }
    let err = det.check("p1", "mcp:t:a", &args).await.unwrap_err();
    match err {
        LoopDetectorError::LoopDetected {
            tool_id,
            count,
            window,
        } => {
            assert_eq!(tool_id, "mcp:t:a");
            assert_eq!(count, 5);
            assert_eq!(window, 8);
        }
    }
}

#[tokio::test]
async fn loop_detector_distinguishes_different_args() {
    let det = LoopDetector::new(ld_config(8, 2));
    // Same tool, distinct args — each hash count stays at 1, never trips.
    for i in 0..6 {
        det.check("p1", "mcp:t:a", &json!({"x": i})).await.unwrap();
    }
}

#[tokio::test]
async fn loop_detector_is_scoped_per_principal() {
    let det = LoopDetector::new(ld_config(8, 2));
    let args = json!({"x": 1});
    // p1 maxes out three times → would trip on the 3rd.
    det.check("p1", "mcp:t:a", &args).await.unwrap();
    det.check("p1", "mcp:t:a", &args).await.unwrap();
    // p2 shares the tool but has its own window — should not be affected.
    det.check("p2", "mcp:t:a", &args).await.unwrap();
    det.check("p2", "mcp:t:a", &args).await.unwrap();
    // p2's third still passes (count == 3, threshold 2 → trips on >2 means 3rd is err).
    let p2_third = det.check("p2", "mcp:t:a", &args).await;
    assert!(p2_third.is_err(), "p2 should trip on its own 3rd");
    // And p1's window state did not affect p2's count.
    assert_eq!(det.window_len("p1").await, 2);
}

#[tokio::test]
async fn loop_detector_window_evicts_old_entries() {
    // window=3, threshold=3. Sequence a,a,b,a,a totals 4 a's; without
    // eviction count(a)=4>3 would trip. With window=3 the trailing
    // window is [a,b,a,a]→[b,a,a], window-scoped count(a)=2 → no trip.
    let det = LoopDetector::new(ld_config(3, 3));
    let a = json!({"x": 1});
    let b = json!({"x": 2});
    det.check("p", "t", &a).await.unwrap(); // [a]
    det.check("p", "t", &a).await.unwrap(); // [a,a]
    det.check("p", "t", &b).await.unwrap(); // [a,a,b]
    det.check("p", "t", &a).await.unwrap(); // [a,b,a]   count(a)=2
    det.check("p", "t", &a).await.unwrap(); // [b,a,a]   count(a)=2
    assert_eq!(det.window_len("p").await, 3);
}

#[tokio::test]
async fn loop_detector_canonicalizes_object_key_order() {
    let det = LoopDetector::new(ld_config(8, 2));
    let v1 = json!({"a": 1, "b": 2});
    let v2 = json!({"b": 2, "a": 1});
    // Same canonical shape — three calls should trip on the third.
    det.check("p", "t", &v1).await.unwrap();
    det.check("p", "t", &v2).await.unwrap();
    let err = det.check("p", "t", &v1).await.unwrap_err();
    assert!(matches!(
        err,
        LoopDetectorError::LoopDetected { count: 3, .. }
    ));
}

#[tokio::test]
async fn loop_detector_reset_clears_principal_state() {
    let det = LoopDetector::new(ld_config(4, 2));
    let args = json!({"x": 1});
    det.check("p", "t", &args).await.unwrap();
    det.check("p", "t", &args).await.unwrap();
    det.reset("p").await;
    // Counter starts over — even though three total calls have happened.
    det.check("p", "t", &args).await.unwrap();
    det.check("p", "t", &args).await.unwrap();
    assert_eq!(det.window_len("p").await, 2);
}

#[tokio::test]
async fn loop_detector_observer_sees_error_on_trip() {
    let broadcast = BroadcastObserver::new();
    let mut rx = broadcast.subscribe();
    let det = LoopDetector::new(ld_config(4, 2)).with_observer(broadcast as Arc<dyn Observer>);
    let args = json!({"x": 1});
    det.check("p", "mcp:t:a", &args).await.unwrap();
    det.check("p", "mcp:t:a", &args).await.unwrap();
    let _ = det.check("p", "mcp:t:a", &args).await.unwrap_err();
    match tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {
        Ok(Ok(BrainEvent::Error {
            source, message, ..
        })) => {
            assert_eq!(source, "loop_detector");
            assert!(message.contains("mcp:t:a"));
            assert!(message.contains("repeated 3"));
            assert!(message.contains("window 4"));
        }
        other => panic!("expected loop_detector Error, got {other:?}"),
    }
}

#[test]
fn loop_detector_canonical_json_sorts_keys_recursively() {
    let v = json!({"b": {"y": 1, "x": 2}, "a": [3, {"d": 4, "c": 5}]});
    let canon = canonical_json_for_test(&v);
    assert_eq!(canon, r#"{"a":[3,{"c":5,"d":4}],"b":{"x":2,"y":1}}"#);
}

// ─── InMemoryDlq tests ──────────────────────────────────────────────────────

use chrono::Utc;

fn dlq_entry(tool: &str, msg: &str, attempts: u32) -> DlqEntry {
    DlqEntry {
        id: uuid::Uuid::new_v4().to_string(),
        tool_id: tool.to_string(),
        request_json: r#"{"x":1}"#.to_string(),
        error_message: msg.to_string(),
        attempts,
        dlq_at: Utc::now(),
    }
}

#[tokio::test]
async fn in_memory_dlq_starts_empty() {
    let q = InMemoryDlq::new();
    assert_eq!(q.len().await.unwrap(), 0);
    assert!(q.list_recent(10).await.unwrap().is_empty());
}

#[tokio::test]
async fn in_memory_dlq_enqueue_orders_newest_first() {
    let q = InMemoryDlq::new();
    q.enqueue(dlq_entry("a", "first", 3)).await.unwrap();
    q.enqueue(dlq_entry("b", "second", 5)).await.unwrap();
    let recent = q.list_recent(10).await.unwrap();
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].error_message, "second");
    assert_eq!(recent[1].error_message, "first");
}

#[tokio::test]
async fn in_memory_dlq_list_recent_respects_limit() {
    let q = InMemoryDlq::new();
    for i in 0..5 {
        q.enqueue(dlq_entry("t", &format!("e{i}"), 1))
            .await
            .unwrap();
    }
    let recent = q.list_recent(3).await.unwrap();
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0].error_message, "e4");
    assert_eq!(recent[2].error_message, "e2");
}

#[tokio::test]
async fn in_memory_dlq_can_be_used_via_trait_object() {
    let q: Arc<dyn DeadLetterQueue> = Arc::new(InMemoryDlq::new());
    q.enqueue(dlq_entry("t", "boom", 2)).await.unwrap();
    assert_eq!(q.len().await.unwrap(), 1);
}
