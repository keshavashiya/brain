//! The canonical `build_processor` entry point: constructs a fully-wired
//! `SignalProcessor` from config, delegating the action dispatcher, safety
//! infrastructure, and agent registry to the sibling modules.

use std::sync::Arc;

#[cfg(feature = "encryption")]
use crate::encryption::resolve_encryptor;
// `backends::*` (used in `super::dispatcher`) re-exports a `resilience`
// submodule that would shadow the `brainos-resilience` extern crate. Name the
// breaker types via the `::` prefix to keep them reachable regardless.
use ::resilience::{BreakerConfig, BreakerRegistry};

use super::dispatcher::build_action_dispatcher;
use super::safety::wire_safety_infrastructure;

/// Build a fully-wired `SignalProcessor` from config.
///
/// This is the canonical bootstrap path. All CLI commands that need a processor
/// should call this instead of wiring backends individually.
///
/// What gets wired:
/// - Encryption (if enabled)
/// - Action dispatcher with memory backend
/// - Web search backend (searxng / tavily / custom)
/// - Scheduling backend
/// - Messaging backend (webhooks)
/// - Safety infrastructure (audit, confirm, budget, sandbox)
///
/// What is NOT wired here (caller-specific):
/// - Notification router (only needed for `brain serve`)
/// - Background tasks (consolidation, proactivity — only `brain serve`)
/// - Credential vault (wired separately via `wire_vault`)
pub async fn build_processor(
    config: &brain::BrainConfig,
) -> anyhow::Result<signal::SignalProcessor> {
    #[cfg(feature = "encryption")]
    let mut processor = {
        let encryptor = resolve_encryptor(config)?;
        signal::SignalProcessor::new_with_encryptor(config.clone(), encryptor).await?
    };
    #[cfg(not(feature = "encryption"))]
    let mut processor = signal::SignalProcessor::new(config.clone()).await?;

    // Observability bus — every consequential pipeline event publishes through
    // this observer. Wired here so HTTP/WS/gRPC adapter SSE/event streams can
    // subscribe via `processor.subscribe_brain_events()`. Until this is wired,
    // every `if let Some(observer) = ...` guard inside the pipeline silently
    // skipped publication. Default capacity (4096) has lag-drop semantics so
    // slow subscribers can't backpressure the pipeline.
    let observer: Arc<dyn observe::Observer> = observe::BroadcastObserver::new();
    processor = processor.with_observer(observer.clone());
    tracing::info!("Observability bus wired (BroadcastObserver, capacity=4096)");

    // Principal & identity — adapters resolve a `Principal` from their auth
    // context (api-key → agent_id → `IdentityStore::principal_for`). Without
    // the store wired, every `processor.identity_store()` call returns `None`
    // and the authorization gate is a no-op. An empty `identity:` config is
    // the explicit-anonymous default: signals carry no principal and the
    // gate is skipped.
    let identity_store: Arc<dyn identity::IdentityStore> = Arc::new(
        identity::ConfigIdentityStore::from_config(config.identity.clone()),
    );
    processor = processor.with_identity_store(identity_store);
    tracing::info!(
        principals = config.identity.principals.len(),
        "Identity store wired (ConfigIdentityStore)"
    );

    // Terminal Bridge — registry + observer wiring for
    // `Intent::OpenTerminalSession` / `ListTerminalSessions` /
    // `CloseTerminalSession`. Always wired so the intent handlers have a
    // real backend; in-process callers go through the pipeline's identity
    // gate just like HTTP/WS/gRPC/MCP signal handlers. Network-side
    // `TerminalAuth` (api-key → agent_id → principal) is attached in
    // `cmd_serve` at gRPC-server spawn, matching how every other adapter
    // wires its per-request authentication. The graph sink mirrors every
    // session lifecycle into the episodic graph as a
    // `tool_call → terminal_event(open) → terminal_event(close)` chain
    // so recall / audit can reason about terminal activity the same way
    // it does about other tool calls.
    let graph: Arc<dyn hippocampus::EpisodicGraph> = Arc::new(hippocampus::SqliteGraph::new(
        processor.episodic().pool().clone(),
    ));
    // When a semantic store is present, share its vector store + the
    // processor's embedder so mirrored nodes are embedded into `graph_vec`
    // and gain a `vector_id` — this is what lets terminal activity surface
    // through ANN recall, not just FTS. Degraded (no semantic store)
    // installs still mirror nodes; they just skip the ANN link.
    let mut sink = signal::terminal_graph_mirror::HippocampusTerminalSink::new(graph.clone());
    if let Some(semantic) = processor.semantic() {
        sink = sink.with_embedding(
            processor.embedder(),
            semantic.vector_store(),
            processor.embedding_dim(),
        );
    }
    let graph_sink: Arc<dyn terminal::TerminalGraphSink> = Arc::new(sink);
    let terminal_bridge = terminal::TerminalBridge::new()
        .with_observer(observer.clone())
        .with_graph_sink(graph_sink);
    processor = processor.with_terminal_bridge(Arc::new(terminal_bridge));
    tracing::info!("Terminal Bridge wired (registry + observer + graph sink)");

    // Dual-memory reader — graph-first, legacy-fallback point lookup
    // facade. Wired so callers reading a memory by id get the graph
    // version when present and the legacy `episodes` row otherwise,
    // without having to know where the row lives. Both inner handles
    // share the processor's SQLite pool, so writes from either side
    // become visible to subsequent reads.
    let legacy = Arc::new(hippocampus::EpisodicStore::new(
        processor.episodic().pool().clone(),
    ));
    let mut dual_reader = hippocampus::DualMemoryReader::dual(legacy, graph);
    // Attach the shared vector store so recall's graph-ANN half can query
    // `graph_vec` (the FTS half needs no extra handle). Skipped on degraded
    // installs without a semantic store — graph recall then runs FTS-only.
    if let Some(semantic) = processor.semantic() {
        dual_reader = dual_reader.with_vector_store(semantic.vector_store());
    }
    processor = processor.with_dual_memory_reader(dual_reader);
    tracing::info!("Dual-memory reader wired (graph first, legacy fallback)");

    // Per-tool circuit breaker registry — minted before the router and
    // the MCP host so both share one source of truth. The router
    // queries it via `intent::BreakerCheck` to exclude `Open` tools
    // from scoring; the `ResilientMcpHost` decorator records
    // success/failure on every call so the breaker reflects ground
    // truth. Observer is threaded in so state transitions land on the
    // same bus SSE subscribes to.
    let breaker_cfg = BreakerConfig {
        failure_threshold: config.actions.resilience.circuit_breaker_threshold,
        open_duration: std::time::Duration::from_secs(
            config.actions.resilience.circuit_breaker_cooldown_secs,
        ),
        ..BreakerConfig::default()
    };
    let breakers = Arc::new(BreakerRegistry::new(breaker_cfg).with_observer(observer.clone()));
    tracing::info!(
        failure_threshold = config.actions.resilience.circuit_breaker_threshold,
        cooldown_secs = config.actions.resilience.circuit_breaker_cooldown_secs,
        "Breaker registry wired"
    );

    // Per-client rate-limit registry (Issue 51). Wired here so HTTP / WS /
    // gRPC adapters can read it off the processor and apply throttling at
    // the edge.
    let rl_cfg = &config.access.rate_limit;
    if rl_cfg.enabled {
        let rate_cfg = ::resilience::RateLimitConfig {
            tokens_per_refill: rl_cfg.tokens_per_refill,
            refill_interval: std::time::Duration::from_millis(rl_cfg.refill_interval_ms),
            burst_capacity: rl_cfg.burst_capacity,
        };
        let limits = Arc::new(::resilience::RateLimitRegistry::new(rate_cfg));
        processor = processor.with_client_rate_limits(limits);
        tracing::info!(
            tokens_per_refill = rl_cfg.tokens_per_refill,
            refill_interval_ms = rl_cfg.refill_interval_ms,
            burst_capacity = rl_cfg.burst_capacity,
            "Client rate limiter wired"
        );
    } else {
        tracing::info!("Client rate limiter disabled by config");
    }

    // Capability kernel — workspace-wide registry of every tool the host
    // can dispatch to, plus the intent router that resolves a classified
    // `IntentToken` to a concrete `ToolRoute`. The same `ToolRegistry`
    // is threaded into the MCP host below so every server mount
    // auto-populates it; the mcphost-side `ToolCapabilityIndex` keeps the
    // host's own per-server lookup hot. The breaker registry is wired
    // into the router so `Open` tools are skipped during scoring.
    let tool_registry: Arc<dyn intent::ToolRegistry> =
        Arc::new(intent::InMemoryToolRegistry::new());
    let breaker_check: Arc<dyn intent::BreakerCheck> = breakers.clone();
    let intent_router: Arc<dyn intent::IntentRouter> = Arc::new(
        intent::DefaultIntentRouter::new(tool_registry.clone()).with_breakers(breaker_check),
    );
    let mcp_capability_index: Arc<dyn mcphost::ToolCapabilityIndex> =
        Arc::new(mcphost::InMemoryToolCapabilityIndex::new());
    processor = processor
        .with_tool_registry(tool_registry.clone())
        .with_intent_router(intent_router);
    tracing::info!("Capability kernel wired (registry + DefaultIntentRouter + breakers)");

    // Seed the registry with the kernel's *native* capabilities (action
    // backends + terminal) so the one manifest the MCP host also
    // populates describes the built-in tools, not just mounted servers.
    // This is what the SOUL capability digest and external `tools/list`
    // read. Awareness only — execution stays gated.
    crate::capabilities::register_native_capabilities(&tool_registry, config).await;

    // MCP host — always wired so `Intent::MountMcpServer` /
    // `ListMcpServers` / `UnmountMcpServer` and the `mcp:{server}:{tool}`
    // route resolver have a real backend instead of the
    // "MCP host not configured" placeholder. Without this, every MCP
    // intent handler short-circuited and the capability router's `Mcp`
    // arm dropped tool calls. Empty until callers mount a server at
    // runtime; the observer is threaded in so the host's rug-pull
    // (`tools/list` hash change) and refresh-failure events land on the
    // same bus the SSE stream subscribes to. The tool registry +
    // capability index let mounts populate the workspace-wide and
    // host-local catalogs the router queries. The host is wrapped in
    // `ResilientMcpHost` so every `call` records breaker outcomes and
    // `Open` tools fail fast at the host boundary as well — keeps the
    // breaker state honest even when callers bypass the router.
    let rmcp_inner: Arc<dyn mcphost::MCPHost> = Arc::new(
        mcphost::RmcpHost::new()
            .with_observer(observer.clone())
            .with_tool_registry(tool_registry)
            .with_capability_index(mcp_capability_index),
    );
    // Persistent dead-letter queue — exhausted MCP retries land here
    // so the serve loop's drain task (cli::serve::spawn_dlq_drain) can
    // replay them later. The same `Arc` is threaded through the
    // resilient decorator (writer) and onto the processor (reader for
    // the drainer) so both surfaces see one consistent backlog.
    let dlq: Arc<dyn ::resilience::DeadLetterQueue> = Arc::new(storage::SqliteDlq::new(Arc::new(
        processor.episodic().pool().clone(),
    )));

    let mcp_host: Arc<dyn mcphost::MCPHost> = Arc::new(
        mcphost::ResilientMcpHost::new(rmcp_inner)
            .with_breakers(breakers.clone())
            .with_dlq(dlq.clone()),
    );
    processor = processor
        .with_mcp_host(mcp_host)
        .with_breaker_registry(breakers)
        .with_dlq(dlq);
    tracing::info!("MCP host wired (RmcpHost + ResilientMcpHost decorator with breakers + DLQ)");

    // Sandbox executor — hoisted up here (was inside
    // `wire_safety_infrastructure`) so the action dispatcher can route
    // `Action::ExecuteCommand` through the same executor instead of
    // falling back to raw `tokio::process::Command` (Issue 121).
    let exec_timeout = std::time::Duration::from_secs(config.security.exec_timeout_seconds as u64);
    let sandbox =
        sandbox::IsolatedSandbox::new(config.security.exec_allowlist.clone(), exec_timeout)
            .with_allowed_paths(vec![
                std::path::PathBuf::from(&config.brain.data_dir),
                std::env::current_dir().unwrap_or_default(),
            ]);
    let sandbox_executor: Arc<dyn sandbox::SandboxExecutor> = Arc::new(sandbox);
    tracing::info!(
        allowlist_size = config.security.exec_allowlist.len(),
        timeout_s = config.security.exec_timeout_seconds,
        "Sandbox executor wired (isolated: rlimits + allowlist)"
    );

    let action_dispatcher = build_action_dispatcher(config, &processor)?
        .with_sandbox_executor(sandbox_executor.clone());
    processor = processor.with_action_dispatcher(action_dispatcher);

    // ── Safety infrastructure ───────────────────────────────────────────
    processor = wire_safety_infrastructure(processor, config, sandbox_executor).await?;

    // Product self-model — grounds the SOUL in Brain's own surface (real CLI
    // commands walked from clap, in-chat signals walked from the SIGNALS table,
    // config schema sliced from the embedded defaults, policy invariants) so it
    // answers product questions from code-derived truth instead of fabricating
    // commands/signals/config keys.
    let self_model = selfmodel::ProductSelfModel::new(
        crate::command_catalog::build(),
        crate::chat::signal_catalog(),
        brain::BrainConfig::default_config_content(),
    );
    processor = processor.with_product_self_model(Arc::new(self_model));
    tracing::info!("Product self-model wired (commands + signals + config schema + policy)");

    Ok(processor)
}
