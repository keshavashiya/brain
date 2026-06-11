//! Integration tests for SignalProcessor — full pipeline round-trips.

use brainos_signal::{
    ResponseContent, ResponseStatus, Signal, SignalError, SignalProcessor, SignalSource,
};

#[tokio::test]
async fn test_process_store_fact_integration() {
    let temp_dir = tempfile::tempdir().unwrap();

    let mut config = brain::BrainConfig::default();
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
    let mut config = brain::BrainConfig::default();
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
    let mut config = brain::BrainConfig::default();
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
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    // Vouch for the writer so the fact lands live — unvouched agents are
    // quarantined and excluded from listings (covered in memory_quarantine.rs).
    config
        .memory
        .trust
        .agents
        .insert("open-code".to_string(), 1.0);

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

    let mut config = brain::BrainConfig::default();
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
    let mut config = brain::BrainConfig::default();
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
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();

    let target = uuid::Uuid::new_v4();
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        format!("cancel signal {target}"),
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
    let mut config = brain::BrainConfig::default();
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
    let mut config = brain::BrainConfig::default();
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

// Redaction wiring proptest: a vault-marked secret embedded in
// `Signal.content` MUST NOT appear in the BrainEvent::SignalReceived
// payload that lands on the bus. PR1 proves the Redactor itself is safe;
// this proves we actually call it on the wiring path.
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
                let mut cfg = brain::BrainConfig::default();
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

// ── v1.0.0 Phase 1 acceptance: two agents, two tiers ──────────────────────────

/// Phase 1 acceptance per `docs/v1.0.0.md` §10 line 1337:
///
/// > Two agents (`claude-code` with `Execute` tier, `cursor` with `Read` tier)
/// > invoke the same `shell.exec` intent. The first runs; the second escalates
/// > to user and surfaces in the Live tab as `ConfirmationRequested
/// > { reason: "agent_id=cursor missing scope shell.exec" }`.
#[tokio::test]
async fn phase_1_two_agents_one_runs_one_escalates() {
    use identity::IdentityStore as _;
    use std::sync::Arc;

    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    // Identity config: claude-code holds shell.exec at Execute; cursor at Read.
    let id_cfg: identity::IdentityConfig = serde_yaml::from_str(
        r#"
        user_id: keshav
        principals:
          - agent_id: claude-code
            scopes: [shell.exec]
            tier: execute
          - agent_id: cursor
            scopes: [shell.exec]
            tier: read
        "#,
    )
    .unwrap();
    let identity_store = Arc::new(identity::ConfigIdentityStore::from_config(id_cfg));

    let observer = observe::BroadcastObserver::new();
    let processor = SignalProcessor::new(config)
        .await
        .unwrap()
        .with_observer(observer.clone())
        .with_identity_store(identity_store.clone());

    use observe::Observer as _;
    let mut rx = observer.subscribe();

    let claude = identity_store
        .principal_for(&identity::AgentHint::AgentId("claude-code".into()))
        .await
        .unwrap();
    let cursor = identity_store
        .principal_for(&identity::AgentHint::AgentId("cursor".into()))
        .await
        .unwrap();

    // claude-code: Execute-tier → should pass the identity gate. The
    // command itself fails because no sandbox is wired in this test, but
    // the *gate* let it through — that's what we're asserting.
    let claude_signal =
        Signal::new(SignalSource::Cli, "cli", "user", "run echo hi").with_principal(claude);
    let claude_resp = processor.process(claude_signal).await.unwrap();
    assert!(
        !response_text(&claude_resp).contains("Approval required"),
        "claude-code with Execute tier should NOT be escalated: {:?}",
        claude_resp
    );

    // cursor: Read-tier → identity gate escalates. Response text says so,
    // and a ConfirmationRequested BrainEvent fires.
    let cursor_signal =
        Signal::new(SignalSource::Cli, "cli", "user", "run echo hi").with_principal(cursor);
    let cursor_id = cursor_signal.id;
    let cursor_resp = processor.process(cursor_signal).await.unwrap();
    let text = response_text(&cursor_resp);
    assert!(
        text.contains("Approval required"),
        "expected escalation, got: {text}"
    );
    assert!(
        text.contains("tier=read") || text.contains("missing scope") || text.contains("tier"),
        "escalation reason should mention tier or scope: {text}"
    );

    // Drain the bus until we see the ConfirmationRequested for cursor.
    let event = tokio::time::timeout(std::time::Duration::from_millis(200), async {
        loop {
            if let Ok(ev) = rx.recv().await {
                if let observe::BrainEvent::ConfirmationRequested { id, .. } = &ev {
                    if *id == cursor_id {
                        return ev;
                    }
                }
            }
        }
    })
    .await
    .expect("ConfirmationRequested arrived");
    if let observe::BrainEvent::ConfirmationRequested { reason, .. } = event {
        assert!(
            reason.contains("cursor"),
            "reason should name the agent: {reason}"
        );
    }
}

fn response_text(resp: &brainos_signal::SignalResponse) -> String {
    match &resp.response {
        brainos_signal::ResponseContent::Text(t) => t.clone(),
        brainos_signal::ResponseContent::Error(e) => e.clone(),
        other => format!("{other:?}"),
    }
}

/// Happy-path Phase 1 acceptance complement: signal with a principal that
/// satisfies the gate flows through unimpeded, and an audit entry written
/// alongside carries the same principal in both the SQLite row and the
/// `BrainEvent::AuditAppended` payload.
///
/// Tier B already covers the denied path (cursor is escalated). This is the
/// other half: confirms the full chain works when the principal is allowed
/// (audit-write sites pulling from `Signal.principal` directly is a broader
/// rewire, queued behind Phase 1 — this test constructs the audit entry
/// manually to exercise the persistence + observer path end to end).
#[tokio::test]
async fn phase_1_happy_path_audit_carries_principal() {
    use identity::IdentityStore as _;
    use std::sync::Arc;

    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

    let id_cfg: identity::IdentityConfig = serde_yaml::from_str(
        r#"
        user_id: keshav
        principals:
          - agent_id: claude-code
            scopes: ["shell.exec", "memory.store"]
            tier: execute
        "#,
    )
    .unwrap();
    let identity_store = Arc::new(identity::ConfigIdentityStore::from_config(id_cfg));

    let observer = observe::BroadcastObserver::new();
    let pool = storage::SqlitePool::open_memory().unwrap();
    let audit_trail = Arc::new(audit::SqliteAuditTrail::new(pool).with_observer(observer.clone()));
    audit_trail.ensure_tables().unwrap();

    let processor = SignalProcessor::new(config)
        .await
        .unwrap()
        .with_observer(observer.clone())
        .with_identity_store(identity_store.clone());

    let claude = identity_store
        .principal_for(&identity::AgentHint::AgentId("claude-code".into()))
        .await
        .unwrap();

    // Signal with Execute-tier principal — should pass the gate (StoreFact
    // → memory.store @ Write, satisfied by tier=Execute and scope memory.store).
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "Remember that Rust is fast",
    )
    .with_principal(claude.clone());
    let resp = processor.process(signal).await.unwrap();
    assert_eq!(
        resp.status,
        ResponseStatus::Ok,
        "expected Ok, got {:?}: {}",
        resp.status,
        response_text(&resp)
    );

    // Now simulate a handler writing an audit row tagged with the same
    // principal (the wire-up step is broader than Phase 1).
    let entry = audit::AuditEntry::new(
        "remember rust",
        "store-fact",
        "memory.store",
        audit::ActionTier::Write,
    )
    .with_principal(claude.clone());
    let entry_id = audit::AuditTrail::record(audit_trail.as_ref(), entry)
        .await
        .unwrap();

    // Round-trip from SQLite must preserve the full principal.
    let rows = audit::AuditTrail::query(audit_trail.as_ref(), audit::AuditQuerySpec::default())
        .await
        .unwrap();
    let row = rows.iter().find(|r| r.id == entry_id).unwrap();
    let stored = row.principal.as_ref().unwrap();
    assert_eq!(stored.agent_id, claude.agent_id);
    assert_eq!(stored.user_id, claude.user_id);
    assert_eq!(stored.tier, claude.tier);

    // Drain the bus to find the AuditAppended event for this entry.
    use observe::Observer as _;
    let mut rx = observer.subscribe();

    // Record one more so a fresh subscriber definitely sees an event.
    let probe_entry = audit::AuditEntry::new("probe", "d", "a", audit::ActionTier::Read)
        .with_principal(claude.clone());
    audit::AuditTrail::record(audit_trail.as_ref(), probe_entry)
        .await
        .unwrap();

    let event = tokio::time::timeout(std::time::Duration::from_millis(100), async {
        loop {
            if let Ok(ev) = rx.recv().await {
                if matches!(ev, observe::BrainEvent::AuditAppended { .. }) {
                    return ev;
                }
            }
        }
    })
    .await
    .expect("AuditAppended within 100ms");
    if let observe::BrainEvent::AuditAppended { principal, .. } = event {
        let summary = principal.expect("event carries principal");
        assert_eq!(summary.agent_id, "claude-code");
        assert_eq!(summary.user_id, "keshav");
    }
}

#[tokio::test]
async fn tool_registry_accessor_round_trips() {
    use std::sync::Arc;
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    assert!(processor.tool_registry().is_none());

    let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
    registry
        .register(intent::ToolDescriptor {
            tool_id: "mcp:fs:read".into(),
            source: intent::ToolSource::McpServer {
                server: "fs".into(),
            },
            verb: intent::Verb::new("fs", "read"),
            description: "Read a file".into(),
            input_schema: serde_json::json!({ "type": "object" }),
            output_schema: None,
            capabilities: vec!["fs.read".into()],
            annotations: intent::ToolAnnotations::default(),
            usage: intent::ToolUsage::default(),
            embedding: None,
        })
        .await
        .unwrap();

    let processor = processor.with_tool_registry(registry);
    let wired = processor.tool_registry().expect("registry wired");
    assert_eq!(wired.list().await.len(), 1);
    assert!(wired.get("mcp:fs:read").await.is_some());
}
