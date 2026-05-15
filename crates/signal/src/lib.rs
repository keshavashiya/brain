//! # Brain Signal Processor
//!
//! Central hub that converts all input signals (CLI, HTTP, WebSocket, MCP, gRPC)
//! into a unified Signal type and routes them through the Brain pipeline.
//!
//! The SignalProcessor wires together:
//! - Thalamus (intent classification)
//! - Amygdala (importance scoring)
//! - Hippocampus (episodic + semantic memory)
//! - Cortex (LLM reasoning + context assembly)
//! - NotificationRouter (proactive delivery)

pub mod approval;
pub mod authz;
pub mod notification;
pub mod types;

mod budget_guard;
mod constructors;
mod exchange;
mod extract;
mod pipeline;
mod recall;
mod render;
mod streaming;
mod wiring;

pub use budget_guard::{check_llm_input, record_llm_usage, BudgetGate};

pub use approval::ChannelApprovalNotifier;

// Re-export all public types so `use signal::X;` continues to work.
pub use types::*;

// ─── Signal Processor ─────────────────────────────────────────────────────────

/// Central processor that wires all Brain subsystems together.
///
/// One instance is shared across all adapters. Each incoming Signal is routed
/// through intent classification → importance scoring → memory → LLM → response.
pub struct SignalProcessor {
    config: brain_core::BrainConfig,
    classifier: thalamus::IntentClassifier,
    importance: amygdala::ImportanceScorer,
    episodic: hippocampus::EpisodicStore,
    semantic: Option<hippocampus::SemanticStore>,
    embedder: tokio::sync::Mutex<Option<hippocampus::Embedder>>,
    /// Actual output dimension of the active embedding provider (probed at startup).
    embedding_dim: usize,
    recall_engine: hippocampus::RecallEngine,
    llm: std::sync::Arc<dyn cortex::LlmProvider>,
    context_assembler: cortex::context::ContextAssembler,
    procedures: cerebellum::ProcedureStore,
    events_tx: tokio::sync::broadcast::Sender<SignalProcessedEvent>,
    /// Notification router for proactive message delivery (set via builder).
    notification_router: Option<notification::NotificationRouter>,
    /// Action dispatcher for executing tool intents (set via builder).
    action_dispatcher: Option<cortex::actions::ActionDispatcher>,
    /// Cross-subsystem metrics (embedding, consolidation, circuit breaker, intent).
    metrics: std::sync::Arc<brain_core::metrics::SubsystemMetrics>,

    // ── Phase 1: Safety infrastructure (opt-in via builder) ──────────────
    /// Immutable audit trail — records every consequential action.
    audit_trail: Option<std::sync::Arc<dyn audit::AuditTrail>>,
    /// Confirmation engine — human approval gates for destructive/external actions.
    confirmation_engine: Option<std::sync::Arc<dyn confirm::ConfirmationEngine>>,
    /// Cost budget — per-action and rolling ceilings on LLM tokens, API calls, sandbox time.
    cost_budget: Option<std::sync::Arc<dyn budget::CostBudget>>,
    /// Sandbox executor — isolated command execution with resource limits.
    sandbox_executor: Option<std::sync::Arc<dyn sandbox::SandboxExecutor>>,
    /// Credential vault — secure credential storage and injection.
    credential_vault: Option<std::sync::Arc<dyn vault::CredentialVault>>,
    /// Task orchestrator — decomposes requests into executable plans (Phase 2).
    orchestrator: Option<std::sync::Arc<orchestrate::TaskOrchestrator>>,

    // ── Channel intelligence (opt-in via builder) ────────────────────────
    /// Channel router — selects best-available surface for proactive delivery.
    channel_router: Option<std::sync::Arc<dyn channel::ChannelRouter>>,
    /// Channel preference store — learned weights per (namespace, category).
    channel_preferences: Option<std::sync::Arc<dyn channel::ChannelPreferenceStore>>,
    /// Confirmation correlator — resolves approve/reject messages from any channel.
    confirmation_correlator: Option<std::sync::Arc<channel::ConfirmationCorrelator>>,
    /// Channel dispatcher — owns transport handles and performs actual delivery.
    channel_dispatcher: Option<std::sync::Arc<channel::ChannelDispatcher>>,

    // ── Agent delegation (Phase 3) ───────────────────────────────────────
    /// Registry of specialist agent delegates (Claude Code, custom subprocess, etc.).
    agent_registry: Option<std::sync::Arc<delegate::AgentRegistry>>,

    // ── Observability (v1.0.0 Phase 0) ───────────────────────────────────
    /// Optional event bus. When set, the pipeline publishes structured
    /// `BrainEvent`s for the Live tab, `brain tail`, and remote subscribers.
    /// Coexists with the legacy `events_tx` `SignalProcessedEvent` bus during
    /// the Phase-0 → Phase-2 transition; that bus is removed once all
    /// consumers (httpadapter, grpcadapter) migrate to subscribing through
    /// `Observer`.
    observer: Option<std::sync::Arc<dyn observe::Observer>>,
    /// In-flight signal cancellation registry. `process()` registers a
    /// `Notify` keyed by `Signal.id` at entry and removes it on completion
    /// via the `CancelGuard` RAII. `Intent::CancelSignal` looks up the
    /// notify and triggers it; the LLM-generation step listens via
    /// `tokio::select!` and aborts. Structured concurrency at every
    /// checkpoint lands with the Phase 6 orchestrator rewrite.
    cancel_registry: std::sync::Arc<
        tokio::sync::Mutex<
            std::collections::HashMap<uuid::Uuid, std::sync::Arc<tokio::sync::Notify>>,
        >,
    >,
    /// Authorization store (v1.0.0 Phase 1, `docs/v1.0.0.md` §7). When set
    /// and the incoming `Signal` carries a `Principal`, the pipeline gates
    /// the classified intent through `IdentityStore::check` before
    /// executing it. Unwired = back-compat (no enforcement).
    identity_store: Option<std::sync::Arc<dyn identity::IdentityStore>>,

    // ── Terminal Bridge (Motor cortex) ──────────────────────────────────
    /// Optional Terminal Bridge for `OpenTerminalSession` /
    /// `ListTerminalSessions` / `CloseTerminalSession` intents. When
    /// unwired, those intents return a "not configured" response.
    terminal_bridge: Option<std::sync::Arc<terminal::TerminalBridge>>,

    // ── MCP Host (Motor cortex) ────────────────────────────────────────
    /// Optional MCP host for `MountMcpServer` / `UnmountMcpServer` /
    /// `ListMcpServers` intents. When unwired, those intents return a
    /// "not configured" response.
    mcp_host: Option<std::sync::Arc<dyn mcphost::MCPHost>>,

    // ── Capability Kernel (Phase 3) ────────────────────────────────────
    /// Tool registry the capability router resolves [`intent::IntentToken`]s
    /// against. Populated by the MCP host and native backends at mount /
    /// registration time. When unwired, `Intent::ToolCall` returns the
    /// router-not-configured placeholder.
    tool_registry: Option<std::sync::Arc<dyn intent::ToolRegistry>>,
    /// Capability router that resolves [`intent::IntentToken`]s into
    /// [`intent::ToolRoute`]s. The pipeline's `handle_tool_call` arm calls
    /// `resolve` and dispatches the returned route. Without a router,
    /// `Intent::ToolCall` falls back to a deterministic placeholder.
    intent_router: Option<std::sync::Arc<dyn intent::IntentRouter>>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_signal_new() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello world");
        assert!(!signal.id.is_nil());
        assert_eq!(signal.source, SignalSource::Cli);
        assert_eq!(signal.channel, "cli");
        assert_eq!(signal.sender, "user");
        assert_eq!(signal.content, "hello world");
        assert!(signal.metadata.is_empty());
    }

    #[test]
    fn test_signal_response_ok() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "success");
        assert_eq!(resp.signal_id, id);
        assert_eq!(resp.status, ResponseStatus::Ok);
        assert!(matches!(resp.response, ResponseContent::Text(_)));
        assert_eq!(resp.memory_context.facts_used, 0);
        assert_eq!(resp.memory_context.episodes_used, 0);
    }

    #[test]
    fn test_signal_response_error() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::error(id, "something went wrong");
        assert_eq!(resp.status, ResponseStatus::Error);
        assert!(matches!(resp.response, ResponseContent::Error(_)));
    }

    #[test]
    fn test_memory_context_default() {
        let ctx = MemoryContext::default();
        assert_eq!(ctx.facts_used, 0);
        assert_eq!(ctx.episodes_used, 0);
    }

    #[test]
    fn test_signal_source_serde() {
        let sources = vec![
            SignalSource::Cli,
            SignalSource::Http,
            SignalSource::WebSocket,
            SignalSource::Mcp,
            SignalSource::Grpc,
        ];
        for s in &sources {
            let json = serde_json::to_string(s).unwrap();
            let back: SignalSource = serde_json::from_str(&json).unwrap();
            assert_eq!(s, &back);
        }
    }

    #[test]
    fn test_signal_serde() {
        let signal = Signal::new(SignalSource::Http, "http", "apiclient", "Remember coffee");
        let json = serde_json::to_string(&signal).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(signal.id, back.id);
        assert_eq!(signal.content, back.content);
    }

    #[test]
    fn test_signal_with_agent() {
        let signal =
            Signal::new(SignalSource::Http, "http", "apiclient", "hello").with_agent("claude-code");
        assert_eq!(signal.agent.as_deref(), Some("claude-code"));

        // Serialization round-trip preserves agent
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("claude-code"));
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agent.as_deref(), Some("claude-code"));
    }

    #[test]
    fn test_signal_without_agent_omits_field() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello");
        assert!(signal.agent.is_none());
        let json = serde_json::to_string(&signal).unwrap();
        // skip_serializing_if = "Option::is_none" should omit agent entirely
        assert!(!json.contains("agent"));
    }

    #[test]
    fn test_signal_response_serde() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "hello");
        let json = serde_json::to_string(&resp).unwrap();
        let back: SignalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.signal_id, back.signal_id);
        assert_eq!(resp.status, back.status);
    }
}
