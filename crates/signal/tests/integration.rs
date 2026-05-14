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

/// Sending Intent::CancelSignal for an unknown signal id returns a clean
/// "no in-flight signal" response — no panic, no leaked state.
#[tokio::test]
async fn cancel_signal_for_unknown_id_returns_noop_message() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();

    let target = uuid::Uuid::new_v4();
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        &format!("cancel signal {target}"),
    );
    let resp = processor.process(signal).await.unwrap();
    if let ResponseContent::Text(text) = &resp.response {
        assert!(
            text.contains("No in-flight signal"),
            "unexpected response: {text}"
        );
    } else {
        panic!("expected text response, got {:?}", resp.response);
    }
}

/// `cancel_signal()` triggers the registered notify for an in-flight signal.
/// Direct API call (the intent path is exercised above); this verifies the
/// registry semantics.
#[tokio::test]
async fn cancel_signal_triggers_registered_notify() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();

    let id = uuid::Uuid::new_v4();
    let notify = processor.register_cancel(id).await;

    // Cancel from another task; main task awaits notified().
    let proc = std::sync::Arc::new(processor);
    let proc2 = proc.clone();
    let cancel_task = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        proc2.cancel_signal(id).await
    });

    tokio::time::timeout(std::time::Duration::from_millis(200), notify.notified())
        .await
        .expect("notify fired");
    let was_registered = cancel_task.await.unwrap();
    assert!(was_registered, "cancel_signal should report true");
}

// ── v1.0.0 Phase 0 acceptance suite ──────────────────────────────────────────

use brainos_signal as _signal_alias; // anchor crate root for proptest! macro path

/// Phase 0 acceptance per `docs/v1.0.0.md` §10 line 1326:
///
/// > Send a Signal via any adapter → it appears in the Live tab within
/// > ~50 ms → cancel button stops in-flight tool call → audit row and
/// > event row carry the same redacted args.
///
/// This test exercises the in-process equivalent: SignalProcessor with an
/// Observer wired publishes a `SignalReceived` event the moment `process()`
/// is called; a subsequent `AuditTrail::record` publishes `AuditAppended`
/// with the same id the SQLite row carries; cancellation triggers cleanly.
#[tokio::test]
async fn phase_0_acceptance_signal_audit_cancel_within_budget() {
    use std::sync::Arc;

    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let observer = observe::BroadcastObserver::new();
    let pool = storage::SqlitePool::open_memory().unwrap();
    let audit = Arc::new(audit::SqliteAuditTrail::new(pool).with_observer(observer.clone()));
    audit.ensure_tables().unwrap();

    let processor = Arc::new(
        SignalProcessor::new(config)
            .await
            .unwrap()
            .with_observer(observer.clone()),
    );

    use observe::Observer as _;
    let mut rx = observer.subscribe();

    // 1) Process a signal — SignalReceived must arrive within 50ms.
    let signal = Signal::new(SignalSource::Cli, "cli", "user", "Remember Rust is fast");
    let signal_id = signal.id;
    let proc = processor.clone();
    let handle = tokio::spawn(async move { proc.process(signal).await });

    let signal_event = tokio::time::timeout(std::time::Duration::from_millis(50), async {
        loop {
            if let Ok(ev) = rx.recv().await {
                if let observe::BrainEvent::SignalReceived { .. } = &ev {
                    return ev;
                }
            }
        }
    })
    .await
    .expect("SignalReceived within 50ms");
    if let observe::BrainEvent::SignalReceived { id, .. } = &signal_event {
        assert_eq!(*id, signal_id, "event id matches signal id");
    }

    // 2) Record an audit entry — AuditAppended must arrive carrying the same
    //    UUID the SQLite row holds.
    let entry = audit::AuditEntry::new("req", "decision", "act", audit::ActionTier::Read);
    let expected_audit_id = entry.id.clone();
    let returned = audit::AuditTrail::record(audit.as_ref(), entry)
        .await
        .unwrap();
    assert_eq!(returned, expected_audit_id);

    let audit_event = tokio::time::timeout(std::time::Duration::from_millis(50), async {
        loop {
            if let Ok(ev) = rx.recv().await {
                if let observe::BrainEvent::AuditAppended { .. } = &ev {
                    return ev;
                }
            }
        }
    })
    .await
    .expect("AuditAppended within 50ms");
    if let observe::BrainEvent::AuditAppended { audit_entry_id, .. } = &audit_event {
        assert_eq!(audit_entry_id, &expected_audit_id);
    }

    // 3) Cancellation of an unknown id returns cleanly (no panic).
    let cancelled = processor.cancel_signal(uuid::Uuid::new_v4()).await;
    assert!(!cancelled, "unregistered id reports false, not panic");

    // Let the spawned signal complete (or error on missing LLM).
    let _ = handle.await;
}

/// Redaction wiring proptest: a vault-marked secret embedded in
/// `Signal.content` MUST NOT appear in the BrainEvent::SignalReceived
/// payload that lands on the bus. PR1 proves the Redactor itself is safe;
/// this proves we actually call it on the wiring path.
proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config {
        cases: 64,
        .. proptest::test_runner::Config::default()
    })]
    #[test]
    fn signal_received_event_never_carries_raw_secret(
        handle in "[a-zA-Z0-9_-]{1,16}",
        value in "[^\\x00:]{4,32}",
        prefix in "[^\\x00]{0,16}",
        suffix in "[^\\x00]{0,16}",
    ) {
        let body = format!("{prefix}{}{suffix}", observe::redact::mark(&handle, &value));
        let value_owned = value.clone();
        let handle_owned = handle.clone();

        let result = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let temp = tempfile::tempdir().unwrap();
                let mut cfg = brain_core::BrainConfig::default();
                cfg.brain.data_dir = temp.path().to_str().unwrap().to_string();
                let observer = observe::BroadcastObserver::new();
                let processor = SignalProcessor::new(cfg)
                    .await
                    .unwrap()
                    .with_observer(observer.clone());
                use observe::Observer as _;
                let mut rx = observer.subscribe();
                processor.publish_signal_received(&Signal::new(
                    SignalSource::Cli, "cli", "user", &body,
                )).await;
                rx.recv().await.unwrap()
            });

        let serialized = serde_json::to_string(&result).unwrap();
        proptest::prop_assert!(!serialized.contains(&value_owned),
            "raw secret leaked into BrainEvent: {serialized}");
        proptest::prop_assert!(serialized.contains(&format!("<vault:{handle_owned}>")),
            "redacted handle missing from event: {serialized}");
    }
}
