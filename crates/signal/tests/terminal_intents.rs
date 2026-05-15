//! End-to-end tests for the terminal-session intents through the
//! `SignalProcessor` pipeline. cfg(unix) — PTY spawn requires a Unix host.

#![cfg(unix)]

use std::sync::Arc;

use brain_core::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalSource};
use terminal::TerminalBridge;

async fn make_processor() -> SignalProcessor {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    // Leak the tempdir so it outlives the test (cleanup at process exit).
    std::mem::forget(temp);
    processor
}

#[tokio::test]
async fn list_terminal_sessions_with_no_bridge_reports_not_configured() {
    let processor = make_processor().await;
    let signal = Signal::new(SignalSource::Cli, "cli", "user", "/terminal-list");
    let resp = processor.process(signal).await.unwrap();
    match resp.response {
        ResponseContent::Text(t) => {
            assert!(
                t.contains("not configured"),
                "expected 'not configured', got: {t}"
            );
        }
        other => panic!("expected Text, got {other:?}"),
    }
}

#[tokio::test]
async fn open_list_close_round_trip_through_pipeline() {
    let bridge = Arc::new(TerminalBridge::new());
    let processor = make_processor().await.with_terminal_bridge(bridge.clone());

    // 1. Open via `/terminal-open` slash command.
    let open_signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "/terminal-open /bin/sh -c sleep_30_for_test_purposes",
    );
    let open_resp = processor.process(open_signal).await.unwrap();
    let open_text = match open_resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected Text, got {other:?}"),
    };
    assert!(
        open_text.starts_with("Opened terminal session "),
        "got: {open_text}"
    );

    // Extract the UUID from the response — first whitespace-separated
    // token after "session ".
    let session_id = open_text
        .strip_prefix("Opened terminal session ")
        .and_then(|rest| rest.split_whitespace().next())
        .expect("session id in response")
        .to_string();
    assert_eq!(bridge.sessions().len().await, 1);

    // 2. List shows the session.
    let list_signal = Signal::new(SignalSource::Cli, "cli", "user", "/terminal-list");
    let list_resp = processor.process(list_signal).await.unwrap();
    let list_text = match list_resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected Text, got {other:?}"),
    };
    assert!(list_text.contains("1 active terminal session"));
    assert!(list_text.contains(&session_id));

    // 3. Close via slash command.
    let close_signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        format!("/terminal-close {session_id}"),
    );
    let close_resp = processor.process(close_signal).await.unwrap();
    let close_text = match close_resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected Text, got {other:?}"),
    };
    assert!(close_text.contains("Closed terminal session"));
    assert!(close_text.contains(&session_id));
    assert_eq!(bridge.sessions().len().await, 0);
}

#[tokio::test]
async fn close_unknown_session_reports_failure() {
    let bridge = Arc::new(TerminalBridge::new());
    let processor = make_processor().await.with_terminal_bridge(bridge.clone());

    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "/terminal-close not-a-real-session",
    );
    let resp = processor.process(signal).await.unwrap();
    let text = match resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected Text, got {other:?}"),
    };
    assert!(
        text.contains("Failed to close"),
        "expected failure message, got: {text}"
    );
}
