use super::agents::build_agent_registry;
use super::processor::build_processor;

// Minimal config producer that disables auto-discovery and provides
// one manual subprocess delegate — keeps the test offline and tight.
fn cfg_with_manual_delegate() -> brain::BrainConfig {
    let mut c = brain::BrainConfig::default();
    c.agents.auto_discovery = false;
    c.agents.delegates = vec![brain::config::AgentEntry {
        name: "hand-wired".to_string(),
        kind: "subprocess".to_string(),
        alias: None,
        binary: "/usr/bin/true".to_string(),
        args: vec![],
        workdir: None,
        prompt_via_stdin: true,
        tags: vec!["test".to_string()],
    }];
    c.agents.fallbacks = vec!["hand-wired".to_string(), "ghost".to_string()];
    c
}

#[tokio::test]
async fn build_agent_registry_records_manual_entry_in_status() {
    let cfg = cfg_with_manual_delegate();
    let reg = build_agent_registry(&cfg).await.expect("registry builds");
    assert!(reg.contains("hand-wired"), "manual delegate not registered");
    match reg.agent_status("hand-wired") {
        Some(delegate::RegistryAgentStatus::Registered { source, .. }) => {
            assert_eq!(*source, delegate::AgentSource::Manual);
        }
        other => panic!("expected Manual Registered, got {other:?}"),
    }
    // Fallback "ghost" is bogus — known_agents should not gain a phantom
    // entry from the validator pass (validator only warns).
    assert!(reg.agent_status("ghost").is_none());
}

#[tokio::test]
async fn build_agent_registry_empty_when_fully_disabled() {
    let mut cfg = brain::BrainConfig::default();
    cfg.agents.auto_discovery = false;
    let reg = build_agent_registry(&cfg).await.expect("registry builds");
    assert!(reg.is_empty());
    assert!(reg.known_agents().is_empty());
}

// Confirms `build_processor` constructs and injects a
// `BroadcastObserver` so the pipeline's `publish_signal_received`
// actually reaches subscribers. Without this wiring,
// `subscribe_brain_events` returns `None` and every pipeline
// observability path is dead.
#[tokio::test]
async fn observer_publishes_on_signal() {
    use signal::types::{Signal, SignalSource};

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let mut rx = processor
        .subscribe_brain_events()
        .expect("observer wired by build_processor");

    let signal = Signal::new(SignalSource::Cli, "test", "tester", "hello");
    let proc = std::sync::Arc::new(processor);
    let proc_for_task = proc.clone();
    // Drive the pipeline in a task; we only need the first SignalReceived
    // publish, which fires before any classification/LLM work.
    tokio::spawn(async move {
        let _ = proc_for_task.process(signal).await;
    });

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("observer event arrived within timeout")
        .expect("broadcast recv");
    assert_eq!(event.kind(), "signal_received");
}

// Confirms `build_processor` wires `ConfigIdentityStore` and that a
// YAML-declared principal materialises through `principal_for`. Without
// this wiring, `processor.identity_store()` returns `None`, so every
// adapter's `auth::resolve_principal` short-circuits to `None` and the
// pipeline's identity gate is a no-op — even when adapters hold a valid
// `(api-key → agent_id)` mapping.
#[tokio::test]
async fn identity_store_wired_with_configured_principal() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;
    cfg.identity = identity::IdentityConfig {
        user_id: "keshav".into(),
        principals: vec![identity::PrincipalConfig {
            agent_id: "claude-code".into(),
            scopes: vec!["shell.exec".into()],
            tier: identity::Tier::Execute,
            path_allowlist: vec![],
            constraints: vec![],
        }],
    };

    let processor = build_processor(&cfg).await.expect("processor builds");
    let store = processor
        .identity_store()
        .expect("identity store wired by build_processor")
        .clone();

    let principal = store
        .principal_for(&identity::AgentHint::AgentId("claude-code".into()))
        .await
        .expect("configured agent resolves");
    assert_eq!(principal.tier, identity::Tier::Execute);
    assert!(principal.scopes.iter().any(|s| s == "shell.exec"));

    // Unknown agents must surface as `UnknownAgent` so adapters fall
    // back to anonymous — not silently pass with a fabricated principal.
    let err = store
        .principal_for(&identity::AgentHint::AgentId("ghost".into()))
        .await
        .expect_err("unknown agent fails closed");
    assert!(matches!(err, identity::IdentityError::UnknownAgent(_)));
}

// Confirms `build_processor` wires a `TerminalBridge` so
// `Intent::OpenTerminalSession` / `ListTerminalSessions` /
// `CloseTerminalSession` have a real backend. Without this, every
// terminal intent handler returned "Terminal Bridge not configured"
// and the bridge's gRPC server (when spawned by cmd_serve) had no
// shared registry to serve.
#[tokio::test]
async fn terminal_bridge_wired() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let bridge = processor
        .terminal_bridge()
        .expect("terminal bridge wired by build_processor")
        .clone();
    // Sessions registry starts empty out of the box.
    assert!(bridge.sessions().is_empty().await);
}

// Confirms `build_processor` attaches a `HippocampusTerminalSink`
// to the bridge so a real session lifecycle leaves nodes in the
// episodic graph backed by the same SQLite pool the rest of the
// processor uses. Without this, terminal activity is invisible to
// recall/audit even though the bridge itself works.
#[cfg(unix)]
#[tokio::test]
async fn terminal_bridge_mirrors_session_lifecycle_into_graph() {
    use hippocampus::EpisodicGraph;

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let bridge = processor
        .terminal_bridge()
        .expect("terminal bridge wired")
        .clone();
    let svc = bridge.svc();
    let handle = svc
        .open_via_pipeline(
            terminal::pb::OpenRequest {
                program: "/bin/sh".to_string(),
                args: vec!["-c".into(), "exit 0".into()],
                env: Default::default(),
                cwd: String::new(),
                initial_size: Some(terminal::pb::PtySize {
                    rows: 24,
                    cols: 80,
                    pixel_width: 0,
                    pixel_height: 0,
                }),
                client_id: String::new(),
                set_controlling_tty: false,
            },
            None,
        )
        .await
        .expect("open session");
    svc.close_via_pipeline(&handle.session_id)
        .await
        .expect("close session");

    // The graph sink lives behind the bridge — verify by reading
    // the same pool through a fresh `SqliteGraph` handle. Three
    // nodes (tool_call + open_event + close_event) are the
    // documented per-lifecycle output.
    let graph = hippocampus::SqliteGraph::new(processor.episodic().pool().clone());
    let nodes = graph.list_all_nodes().expect("list nodes");
    assert_eq!(
        nodes.len(),
        3,
        "expected 3 nodes after one session lifecycle, got {}: {:?}",
        nodes.len(),
        nodes.iter().map(|n| n.kind.as_str()).collect::<Vec<_>>()
    );
}

// Confirms `build_processor` wires a `DualMemoryReader` that
// resolves an id against the same SQLite pool the processor owns,
// preferring the graph. Without this wiring, callers asking for a
// memory by id would have to know whether it lives in the legacy
// `episodes` table or the graph nodes/edges schema, and graph-only
// content would be unreachable through the reader facade.
#[tokio::test]
async fn dual_memory_reader_wired_with_graph_first_lookup() {
    use hippocampus::EpisodicGraph;

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let reader = processor
        .dual_memory_reader()
        .expect("dual-memory reader wired by build_processor");

    // Inject a graph node directly via the shared pool and confirm
    // the reader resolves it as a Graph entry, not a Legacy one.
    let graph = hippocampus::SqliteGraph::new(processor.episodic().pool().clone());
    let node = hippocampus::Node::new(
        hippocampus::NodeKind::new("fact"),
        serde_json::json!({"sample": true}),
        "personal",
        None,
    );
    graph.add_node(&node).expect("graph insert");

    let entry = reader
        .read_by_id(&node.id)
        .expect("dual read succeeds")
        .expect("inserted node is reachable");
    assert!(
        entry.is_graph(),
        "graph-first lookup should return MemoryEntry::Graph, got {entry:?}"
    );

    // Unknown ids resolve to `None` — neither side has the row.
    assert!(reader
        .read_by_id("ghost-id-not-in-any-table")
        .expect("dual read on missing id")
        .is_none());
}

// Confirms `build_processor` wires the capability kernel — a
// `ToolRegistry` and a `DefaultIntentRouter` on top of it. Without
// both wired, every `Intent::ToolCall` fell through to the
// deterministic placeholder and no MCP mount could publish its
// catalog into a routable surface. The registry is also seeded with
// the kernel's native capabilities at boot, so it is non-empty even
// before any MCP server mounts.
#[tokio::test]
async fn capability_kernel_wired_with_native_seed() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let registry = processor
        .tool_registry()
        .expect("tool registry wired by build_processor")
        .clone();
    assert!(processor.intent_router().is_some(), "intent router wired");

    let tools = registry.list().await;
    // Native capabilities are seeded; no MCP-sourced tool exists yet.
    assert!(
        tools
            .iter()
            .any(|t| t.verb == intent::Verb::new("memory", "store")),
        "memory.store native capability is seeded at boot"
    );
    assert!(
        !tools
            .iter()
            .any(|t| matches!(t.source, intent::ToolSource::McpServer { .. })),
        "no MCP server is mounted on a default install"
    );
    // Seeded native descriptors carry the usage enrichment.
    let store = tools
        .iter()
        .find(|t| t.verb == intent::Verb::new("memory", "store"))
        .unwrap();
    assert_eq!(store.usage.tier.as_deref(), Some("write"));
}

// Confirms `build_processor` wires a `BreakerRegistry` and that the
// same `Arc` reaches both the processor surface and (transitively)
// the router's `BreakerCheck` view. Recording failures above the
// configured threshold must flip the breaker to `Open`, which is
// exactly the signal the router consults to skip a sick tool.
#[tokio::test]
async fn breaker_registry_wired_and_shared() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let registry = processor
        .breaker_registry()
        .expect("breaker registry wired by build_processor")
        .clone();

    // No outcomes recorded yet — registry is empty and untouched
    // tool ids report as closed via the BreakerCheck surface.
    assert!(registry.is_empty().await);
    use intent::BreakerCheck;
    assert!(!registry.is_open("mcp:test:never-called").await);

    // Pushing the configured number of failures must flip the
    // breaker `Open` for the affected tool only.
    let threshold = cfg.actions.resilience.circuit_breaker_threshold;
    for _ in 0..threshold {
        registry.record_failure("mcp:test:sick").await;
    }
    assert!(
        registry.is_open("mcp:test:sick").await,
        "breaker should be Open after {threshold} consecutive failures"
    );
    assert!(
        !registry.is_open("mcp:test:healthy").await,
        "untouched tool must stay closed"
    );
}

// Confirms `build_processor` wires an `MCPHost` so MCP intents and the
// capability router's `Mcp` route resolver have a real backend. Without
// this, every `Intent::MountMcpServer` / `ListMcpServers` /
// `UnmountMcpServer` returned "MCP host not configured" regardless of
// YAML or runtime input.
#[tokio::test]
async fn mcp_host_wired_and_empty_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;

    let processor = build_processor(&cfg).await.expect("processor builds");
    let host = processor
        .mcp_host()
        .expect("mcp host wired by build_processor")
        .clone();
    assert!(
        host.list_servers().await.is_empty(),
        "default install mounts no MCP servers"
    );
}

// Umbrella check: a single `build_processor` call must populate
// every optional injection slot the processor exposes. The
// per-subsystem tests above each verify one slot in isolation;
// this one fails fast when a new `with_*` lands without a
// corresponding bootstrap call, or when an existing wiring
// regresses to `None`. Drives one signal through the pipeline so
// the observer side is exercised end-to-end too.
#[tokio::test]
async fn build_processor_populates_every_injection_slot() {
    use signal::types::{Signal, SignalSource};

    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = brain::BrainConfig::default();
    cfg.brain.data_dir = tmp.path().to_str().unwrap().to_string();
    cfg.agents.auto_discovery = false;
    cfg.identity = identity::IdentityConfig {
        user_id: "keshav".into(),
        principals: vec![identity::PrincipalConfig {
            agent_id: "claude-code".into(),
            scopes: vec!["shell.exec".into()],
            tier: identity::Tier::Execute,
            path_allowlist: vec![],
            constraints: vec![],
        }],
    };

    let processor = build_processor(&cfg).await.expect("processor builds");

    // Subscribe BEFORE driving the signal so the SignalReceived
    // event the pipeline publishes is reliably observed.
    let mut events = processor
        .subscribe_brain_events()
        .expect("observer wired and exposing a brain-event stream");

    // Every wired subsystem must be reachable through its accessor.
    assert!(
        processor.identity_store().is_some(),
        "identity store must be wired"
    );
    assert!(
        processor.terminal_bridge().is_some(),
        "terminal bridge must be wired"
    );
    assert!(
        processor.tool_registry().is_some(),
        "tool registry must be wired"
    );
    assert!(
        processor.intent_router().is_some(),
        "intent router must be wired"
    );
    assert!(processor.mcp_host().is_some(), "mcp host must be wired");
    assert!(
        processor.breaker_registry().is_some(),
        "breaker registry must be wired"
    );
    assert!(
        processor.client_rate_limits().is_some(),
        "client rate-limit registry must be wired"
    );
    assert!(
        processor.dual_memory_reader().is_some(),
        "dual-memory reader must be wired"
    );
    assert!(processor.dlq().is_some(), "DLQ must be wired");
    assert!(
        processor.audit_trail().is_some(),
        "audit trail must be wired"
    );
    assert!(
        processor.confirmation_engine().is_some(),
        "confirmation engine must be wired"
    );
    assert!(
        processor.cost_budget().is_some(),
        "cost budget must be wired"
    );
    assert!(
        processor.orchestrator().is_some(),
        "task orchestrator must be wired"
    );
    assert!(
        processor.channel_router().is_some(),
        "channel router must be wired"
    );
    assert!(
        processor.channel_dispatcher().is_some(),
        "channel dispatcher must be wired"
    );
    assert!(
        processor.standing_approvals().is_some(),
        "standing-approval store must be wired"
    );

    // Identity resolves the configured agent (proves the store is
    // not just instantiated but populated from `config.identity`).
    let store = processor.identity_store().unwrap().clone();
    let principal = store
        .principal_for(&identity::AgentHint::AgentId("claude-code".into()))
        .await
        .expect("configured principal resolves");
    assert_eq!(principal.tier, identity::Tier::Execute);

    // MCP host is queryable even with no mounts (proves the inner
    // `RmcpHost` is reachable through the `ResilientMcpHost`
    // decorator that's now in front of it).
    assert!(
        processor
            .mcp_host()
            .unwrap()
            .list_servers()
            .await
            .is_empty(),
        "no MCP servers should be mounted by default"
    );

    // Drive a signal end-to-end and confirm the observer sees the
    // pipeline publish — this is the integration check that the
    // observer hand-off survives the full processing path.
    let proc = std::sync::Arc::new(processor);
    let driver = proc.clone();
    tokio::spawn(async move {
        let signal = Signal::new(SignalSource::Cli, "wiring", "tester", "hello");
        let _ = driver.process(signal).await;
    });

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), events.recv())
        .await
        .expect("observer event must arrive within timeout")
        .expect("broadcast recv");
    assert_eq!(event.kind(), "signal_received");
}
