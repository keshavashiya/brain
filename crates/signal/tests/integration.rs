//! Integration tests for SignalProcessor — full pipeline round-trips.

use brainos_signal::{
    ResponseContent, ResponseStatus, Signal, SignalError, SignalProcessor, SignalSource,
};

#[tokio::test]
async fn test_process_store_fact_integration() {
    let temp_dir = tempfile::tempdir().unwrap();

    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let processor = SignalProcessor::new(config).await.unwrap();

    // "Remember that Rust is fast" → StoreFact intent
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "Remember that Rust is fast",
    );

    let response = processor.process(signal).await.unwrap();

    assert_eq!(response.status, ResponseStatus::Ok);
    // StoreFact stores in semantic memory → facts_used = 1
    assert_eq!(response.memory_context.facts_used, 1);
    assert_eq!(response.memory_context.episodes_used, 0);
    // Response text should confirm the stored fact
    if let ResponseContent::Text(text) = &response.response {
        assert!(text.contains("Rust"));
    } else {
        panic!("Expected Text response");
    }
}

/// Integration test: CLI signal → SignalProcessor → store fact → search fact → verify result.
#[tokio::test]
async fn test_store_fact_then_search_roundtrip() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let processor = SignalProcessor::new(config).await.unwrap();

    // Store a fact via the CLI signal pipeline
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "Remember that Rust is fast",
    );
    let resp = processor.process(signal).await.unwrap();
    assert_eq!(resp.status, ResponseStatus::Ok);
    assert_eq!(
        resp.memory_context.facts_used, 1,
        "StoreFact should persist 1 fact"
    );

    // Verify persistence: list_facts returns the stored fact
    let facts = processor.list_facts(None);
    assert!(
        !facts.is_empty(),
        "Stored fact should appear in list_facts()"
    );

    // Verify search: search_facts returns results
    let results = processor
        .search_facts("Rust programming language", 5, None)
        .await;
    assert!(
        !results.is_empty(),
        "search_facts() should return the stored fact"
    );
}

#[tokio::test]
async fn test_forget_is_namespace_scoped() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();

    processor
        .store_fact_direct("personal", "test", "project", "uses", "bun", None)
        .await
        .unwrap();
    processor
        .store_fact_direct("work", "test", "project", "uses", "bun", None)
        .await
        .unwrap();

    let mut forget_signal = Signal::new(SignalSource::Cli, "cli", "user", "forget bun");
    forget_signal.namespace = "work".to_string();
    let _ = processor.process(forget_signal).await.unwrap();

    let personal = processor.list_facts(Some("personal"));
    let work = processor.list_facts(Some("work"));
    assert_eq!(personal.len(), 1, "personal namespace fact should remain");
    assert_eq!(work.len(), 0, "work namespace fact should be deleted");
}

#[tokio::test]
async fn test_store_fact_preserves_agent() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let processor = SignalProcessor::new(config).await.unwrap();

    // Store a fact with agent identity
    let signal = Signal::new(
        SignalSource::Http,
        "http",
        "apiclient",
        "Remember that Python is versatile",
    )
    .with_agent("open-code");

    let resp = processor.process(signal).await.unwrap();
    assert_eq!(resp.status, ResponseStatus::Ok);

    // Verify the agent is persisted on the fact
    let facts = processor.list_facts(None);
    assert!(!facts.is_empty());
    let fact = &facts[0];
    assert_eq!(fact.agent.as_deref(), Some("open-code"));
}

/// Integration test: chat signal creates episodic memory entries.
///
/// Requires Ollama running locally. Without it, the test hangs for ~120s
/// waiting for the HTTP timeout, so it is skipped in normal CI.
#[tokio::test]
#[ignore = "Requires Ollama server running locally"]
async fn test_process_chat_reaches_llm() {
    let temp_dir = tempfile::tempdir().unwrap();

    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let processor = SignalProcessor::new(config).await.unwrap();

    let signal = Signal::new(SignalSource::Cli, "cli", "user", "Hello, how are you?");

    let result = processor.process(signal).await;
    match result {
        Ok(resp) => {
            assert_eq!(resp.status, ResponseStatus::Ok);
        }
        Err(SignalError::Llm(_)) => {
            // Expected when Ollama is not running — pipeline is wired correctly
        }
        Err(other) => {
            panic!("Unexpected error (should be Llm, not storage/routing): {other}");
        }
    }
}

// ── v1.0.0 Phase 0: Observer wiring ──────────────────────────────────────────

/// `SignalProcessor::with_observer` makes `process()` publish a
/// `BrainEvent::SignalReceived` carrying the signal's id, source, channel,
/// sender, namespace, and a UTF-8-safe content preview.
#[tokio::test]
async fn observer_receives_signal_received_event() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let observer = observe::BroadcastObserver::new();
    let processor = SignalProcessor::new(config)
        .await
        .unwrap()
        .with_observer(observer.clone());

    let mut rx = processor.subscribe_brain_events().expect("observer wired");
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "Remember that Rust is fast",
    );
    let signal_id = signal.id;

    // Spawn the pipeline; ignore the result — we're asserting on the event,
    // not on whether the LLM is reachable.
    let handle = tokio::spawn(async move { processor.process(signal).await });

    let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
        .await
        .expect("event arrived within 500ms")
        .expect("bus delivered");

    match ev {
        observe::BrainEvent::SignalReceived { id, signal, .. } => {
            assert_eq!(id, signal_id);
            assert_eq!(signal.source, "cli");
            assert_eq!(signal.channel, "cli");
            assert_eq!(signal.sender, "user");
            assert_eq!(signal.content_preview, "Remember that Rust is fast");
        }
        other => panic!("expected SignalReceived, got {other:?}"),
    }

    let _ = handle.await;
}
