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
