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
