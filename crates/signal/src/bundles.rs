//! Opt-in capability bundles owned by [`crate::SignalProcessor`].
//!
//! Issue 107 (Wave C SOLID refactor): the processor had grown to 39 fields,
//! the vast majority opt-in via builder methods. The fields naturally cluster
//! along responsibility lines that were already documented as section comments
//! in `lib.rs`. Each bundle here owns one such cluster, and `SignalProcessor`
//! holds them as named sub-structs.
//!
//! Builder/accessor methods on `SignalProcessor` stay verb-stable — they just
//! delegate `self.field = Some(x)` → `self.bundle.field = Some(x)`. Internal
//! pipeline code reaches in through `self.bundle.field` directly (fields are
//! `pub(crate)`); there is no behavioural change.

use std::sync::Arc;

/// Approval / accounting / sandbox gates that fire before consequential
/// actions reach the outside world. Every field is opt-in; unwired ones leave
/// the corresponding gate disabled (pipeline falls through to the legacy
/// "no-X configured" behaviour).
#[derive(Default)]
pub(crate) struct SafetyBundle {
    /// Immutable audit trail — records every consequential action.
    pub(crate) audit_trail: Option<Arc<dyn audit::AuditTrail>>,
    /// Confirmation engine — human approval gates for destructive/external actions.
    pub(crate) confirmation_engine: Option<Arc<dyn confirm::ConfirmationEngine>>,
    /// Cost budget — per-action and rolling ceilings on LLM tokens, API calls, sandbox time.
    pub(crate) cost_budget: Option<Arc<dyn budget::CostBudget>>,
    /// Sandbox executor — isolated command execution with resource limits.
    pub(crate) sandbox_executor: Option<Arc<dyn sandbox::SandboxExecutor>>,
    /// Dead-letter queue — exhausted tool-call retries land here. The same
    /// `Arc` is wired into the serve loop's drain task and the MCP host's
    /// decorator.
    pub(crate) dlq: Option<Arc<dyn resilience::DeadLetterQueue>>,
    /// Standing-approval store, exposed to `/approval-list` and
    /// `/approval-revoke` and wired into the `ConfirmationEngine` so the
    /// bypass check and slash commands see one consistent table.
    pub(crate) standing_approvals: Option<Arc<dyn confirm::StandingApprovalStore>>,
    /// Optional override for the inline confirmation gate's per-request
    /// timeout. Defaults to the tier-driven value
    /// ([`brain::security::ActionTier::default_timeout`]). Tests shorten this
    /// so the no-bypass path doesn't take 60s+.
    pub(crate) confirmation_timeout: Option<std::time::Duration>,
}

/// Cross-channel routing, preference, and delivery infrastructure for
/// proactive notifications and confirmation prompts.
#[derive(Default)]
pub(crate) struct ChannelBundle {
    /// Notification router for proactive message delivery.
    pub(crate) notification_router: Option<crate::notification::NotificationRouter>,
    /// Channel router — selects best-available surface for proactive delivery.
    pub(crate) channel_router: Option<Arc<dyn channel::ChannelRouter>>,
    /// Channel preference store — learned weights per (namespace, category).
    pub(crate) channel_preferences: Option<Arc<dyn channel::ChannelPreferenceStore>>,
    /// Confirmation correlator — resolves approve/reject messages from any channel.
    pub(crate) confirmation_correlator: Option<Arc<channel::ConfirmationCorrelator>>,
    /// Channel dispatcher — owns transport handles and performs actual delivery.
    pub(crate) channel_dispatcher: Option<Arc<channel::ChannelDispatcher>>,
}

/// Motor cortex + capability kernel: action dispatch, terminal/MCP hosts,
/// the tool registry and capability router, plus the per-tool breaker
/// registry that gates dispatch.
#[derive(Default)]
pub(crate) struct CapabilityBundle {
    /// Action dispatcher for executing tool intents.
    pub(crate) action_dispatcher: Option<cortex::actions::ActionDispatcher>,
    /// Terminal Bridge for `OpenTerminalSession` / `ListTerminalSessions` /
    /// `CloseTerminalSession` intents. When unwired, those intents return a
    /// "not configured" response.
    pub(crate) terminal_bridge: Option<Arc<terminal::TerminalBridge>>,
    /// MCP host for `MountMcpServer` / `UnmountMcpServer` / `ListMcpServers`
    /// intents.
    pub(crate) mcp_host: Option<Arc<dyn mcphost::MCPHost>>,
    /// Tool registry the capability router resolves
    /// [`intent::IntentToken`]s against. Populated by the MCP host and
    /// native backends at mount / registration time.
    pub(crate) tool_registry: Option<Arc<dyn intent::ToolRegistry>>,
    /// Capability router that resolves [`intent::IntentToken`]s into
    /// [`intent::ToolRoute`]s. Without one, `Intent::ToolCall` falls back to
    /// the deterministic placeholder.
    pub(crate) intent_router: Option<Arc<dyn intent::IntentRouter>>,
    /// Per-tool breaker registry. Wired into the router (to exclude `Open`
    /// tools from scoring) and into the dispatch site (to record success /
    /// failure after each MCP call).
    pub(crate) breaker_registry: Option<Arc<resilience::BreakerRegistry>>,
}

/// Observability + cancellation: structured `BrainEvent` bus, the legacy
/// `SignalProcessedEvent` bus, and the in-flight cancellation registry.
pub(crate) struct ObservabilityBundle {
    /// Optional event bus. When set, the pipeline publishes structured
    /// `BrainEvent`s for the Live tab, `brain tail`, and remote subscribers.
    /// Coexists with the legacy `events_tx` `SignalProcessedEvent` bus while
    /// callers migrate over.
    pub(crate) observer: Option<Arc<dyn observe::Observer>>,
    /// Legacy event bus, used by adapters that have not migrated to
    /// `observer` yet.
    pub(crate) events_tx: tokio::sync::broadcast::Sender<crate::types::SignalProcessedEvent>,
    /// In-flight signal cancellation registry. `process()` registers a
    /// `Notify` keyed by `Signal.id` at entry and removes it on completion
    /// via the `CancelGuard` RAII. `Intent::CancelSignal` looks up the notify
    /// and triggers it; the LLM-generation step listens via `tokio::select!`
    /// and aborts.
    pub(crate) cancel_registry:
        Arc<tokio::sync::Mutex<std::collections::HashMap<uuid::Uuid, Arc<tokio::sync::Notify>>>>,
}

impl ObservabilityBundle {
    /// Build with a fresh broadcast channel (capacity matches the legacy
    /// constructor value) and an empty cancel registry. `observer` stays
    /// `None` until wired via the builder.
    pub(crate) fn new() -> Self {
        // Capacity 4096 with lag-drop semantics — matches the pre-refactor
        // `tokio::sync::broadcast::channel(4096)` site in `constructors.rs`.
        let (events_tx, _) = tokio::sync::broadcast::channel(4096);
        Self {
            observer: None,
            events_tx,
            cancel_registry: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }
}
