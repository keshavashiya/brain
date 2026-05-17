//! Wiring, accessors, and scheduled intent management for SignalProcessor.

use std::sync::Arc;

use crate::notification;
use crate::SignalProcessor;

impl SignalProcessor {
    /// Expose the config (for adapter use).
    pub fn config(&self) -> &brain_core::BrainConfig {
        &self.config
    }

    /// Expose the episodic store (for adapter use).
    pub fn episodic(&self) -> &hippocampus::EpisodicStore {
        &self.episodic
    }

    /// Expose the semantic store (for adapter use).
    pub fn semantic(&self) -> Option<&hippocampus::SemanticStore> {
        self.semantic.as_ref()
    }

    /// Expose the LLM provider (for adapter use).
    pub fn llm(&self) -> &Arc<dyn cortex::LlmProvider> {
        &self.llm
    }

    /// Expose the context assembler (for adapter use).
    pub fn context_assembler(&self) -> &cortex::context::ContextAssembler {
        &self.context_assembler
    }

    /// Expose the embedding dimension (for adapter use).
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Expose the subsystem metrics handle (for adapter use / instrumentation).
    pub fn metrics(&self) -> &Arc<brain_core::metrics::SubsystemMetrics> {
        &self.metrics
    }

    /// Get a cloneable handle to the LLM provider (for adapter use).
    pub fn llm_arc(&self) -> Arc<dyn cortex::LlmProvider> {
        self.llm.clone()
    }

    /// Attach a notification router (builder pattern).
    pub fn with_notification_router(mut self, router: notification::NotificationRouter) -> Self {
        self.notification_router = Some(router);
        self
    }

    /// Expose the notification router.
    pub fn notification_router(&self) -> Option<&notification::NotificationRouter> {
        self.notification_router.as_ref()
    }

    /// Attach an action dispatcher for executing tool intents (builder pattern).
    pub fn with_action_dispatcher(mut self, dispatcher: cortex::actions::ActionDispatcher) -> Self {
        self.action_dispatcher = Some(dispatcher);
        self
    }

    /// Set the namespace used by the action dispatcher (if attached).
    ///
    /// Call this before `prepare()` when the active namespace changes
    /// (e.g. CLI session namespace switch).
    pub fn set_action_namespace(&mut self, ns: &str) {
        if let Some(d) = &mut self.action_dispatcher {
            d.set_namespace(ns);
        }
    }

    /// Flush all in-flight writes and checkpoint the SQLite WAL.
    ///
    /// Call this on graceful shutdown to ensure no committed data is lost.
    /// Safe to call from any async context; completes synchronously on the
    /// calling thread (WAL checkpoint is a fast O(WAL-size) operation).
    pub fn shutdown(&self) {
        if let Err(e) = self.episodic.pool().wal_checkpoint() {
            tracing::warn!("WAL checkpoint on shutdown failed: {e}");
        } else {
            tracing::info!("SQLite WAL checkpoint complete");
        }
    }

    /// Expose the procedure store (for adapter / MCP use).
    pub fn procedures(&self) -> &cerebellum::ProcedureStore {
        &self.procedures
    }

    // ── Safety infrastructure builder methods ───────────────────────────

    /// Attach an audit trail (builder pattern).
    pub fn with_audit_trail(mut self, trail: Arc<dyn audit::AuditTrail>) -> Self {
        self.audit_trail = Some(trail);
        self
    }

    /// Expose the audit trail.
    pub fn audit_trail(&self) -> Option<&Arc<dyn audit::AuditTrail>> {
        self.audit_trail.as_ref()
    }

    /// Attach a confirmation engine (builder pattern).
    pub fn with_confirmation_engine(
        mut self,
        engine: Arc<dyn confirm::ConfirmationEngine>,
    ) -> Self {
        self.confirmation_engine = Some(engine);
        self
    }

    /// Expose the confirmation engine.
    pub fn confirmation_engine(&self) -> Option<&Arc<dyn confirm::ConfirmationEngine>> {
        self.confirmation_engine.as_ref()
    }

    /// Attach a cost budget (builder pattern).
    pub fn with_cost_budget(mut self, budget: Arc<dyn budget::CostBudget>) -> Self {
        self.cost_budget = Some(budget);
        self
    }

    /// Expose the cost budget.
    pub fn cost_budget(&self) -> Option<&Arc<dyn budget::CostBudget>> {
        self.cost_budget.as_ref()
    }

    /// Attach a sandbox executor (builder pattern).
    pub fn with_sandbox_executor(mut self, executor: Arc<dyn sandbox::SandboxExecutor>) -> Self {
        self.sandbox_executor = Some(executor);
        self
    }

    /// Attach a dual-memory reader (builder pattern). Reads prefer the
    /// graph and fall back to the legacy `episodes` table so callers
    /// can resolve a memory id without caring which side it lives on.
    pub fn with_dual_memory_reader(mut self, reader: hippocampus::DualMemoryReader) -> Self {
        self.dual_memory_reader = Some(reader);
        self
    }

    /// Expose the dual-memory reader.
    pub fn dual_memory_reader(&self) -> Option<&hippocampus::DualMemoryReader> {
        self.dual_memory_reader.as_ref()
    }

    /// Attach a task orchestrator (builder pattern).
    pub fn with_orchestrator(mut self, orch: Arc<orchestrate::TaskOrchestrator>) -> Self {
        self.orchestrator = Some(orch);
        self
    }

    /// Expose the task orchestrator.
    pub fn orchestrator(&self) -> Option<&Arc<orchestrate::TaskOrchestrator>> {
        self.orchestrator.as_ref()
    }

    // ── Channel intelligence ─────────────────────────────────────────────

    /// Attach a channel router (builder pattern).
    pub fn with_channel_router(mut self, router: Arc<dyn channel::ChannelRouter>) -> Self {
        self.channel_router = Some(router);
        self
    }

    /// Expose the channel router.
    pub fn channel_router(&self) -> Option<&Arc<dyn channel::ChannelRouter>> {
        self.channel_router.as_ref()
    }

    /// Attach a channel preference store (builder pattern).
    pub fn with_channel_preferences(
        mut self,
        preferences: Arc<dyn channel::ChannelPreferenceStore>,
    ) -> Self {
        self.channel_preferences = Some(preferences);
        self
    }

    /// Expose the channel preference store.
    pub fn channel_preferences(&self) -> Option<&Arc<dyn channel::ChannelPreferenceStore>> {
        self.channel_preferences.as_ref()
    }

    /// Attach a confirmation correlator (builder pattern).
    pub fn with_confirmation_correlator(
        mut self,
        correlator: Arc<channel::ConfirmationCorrelator>,
    ) -> Self {
        self.confirmation_correlator = Some(correlator);
        self
    }

    /// Expose the confirmation correlator.
    pub fn confirmation_correlator(&self) -> Option<&Arc<channel::ConfirmationCorrelator>> {
        self.confirmation_correlator.as_ref()
    }

    /// Attach a channel dispatcher (builder pattern). The dispatcher owns
    /// transport handles and performs delivery — both the orchestrator's
    /// `Notify` step and the confirm engine's approval prompts route
    /// through it.
    pub fn with_channel_dispatcher(mut self, dispatcher: Arc<channel::ChannelDispatcher>) -> Self {
        self.channel_dispatcher = Some(dispatcher);
        self
    }

    /// Expose the channel dispatcher.
    pub fn channel_dispatcher(&self) -> Option<&Arc<channel::ChannelDispatcher>> {
        self.channel_dispatcher.as_ref()
    }

    // ── Agent delegation ──────────────────────────────────────────────────

    /// Attach an agent registry (builder pattern). Orchestrator-managed
    /// `StepAction::Implement` steps dispatch through this registry.
    pub fn with_agent_registry(mut self, registry: Arc<delegate::AgentRegistry>) -> Self {
        self.agent_registry = Some(registry);
        self
    }

    /// Expose the agent registry.
    pub fn agent_registry(&self) -> Option<&Arc<delegate::AgentRegistry>> {
        self.agent_registry.as_ref()
    }

    // ── Scheduled intent management ───────────────────────────────────────────

    /// List scheduled intents, optionally filtered by namespace.
    pub fn list_scheduled_intents(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<storage::ScheduledIntent>, crate::types::SignalError> {
        self.episodic
            .pool()
            .list_scheduled_intents(namespace)
            .map_err(|e| crate::types::SignalError::Storage(e.to_string()))
    }

    /// Cancel a scheduled intent by ID. Returns true if the intent was found and cancelled.
    pub fn cancel_scheduled_intent(&self, id: &str) -> Result<bool, crate::types::SignalError> {
        self.episodic
            .pool()
            .cancel_scheduled_intent(id)
            .map_err(|e| crate::types::SignalError::Storage(e.to_string()))
    }

    /// Subscribe to live signal-processing events.
    pub fn subscribe_events(
        &self,
    ) -> tokio::sync::broadcast::Receiver<crate::types::SignalProcessedEvent> {
        self.events_tx.subscribe()
    }

    /// Attach an observability event bus (builder pattern). When set, the
    /// pipeline publishes structured `BrainEvent`s alongside the legacy
    /// `SignalProcessedEvent` bus.
    pub fn with_observer(mut self, observer: Arc<dyn observe::Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Expose the configured observability bus, if any.
    pub fn observer(&self) -> Option<&Arc<dyn observe::Observer>> {
        self.observer.as_ref()
    }

    /// Subscribe to the structured `BrainEvent` bus. Returns `None` if no
    /// observer was wired via `with_observer`.
    pub fn subscribe_brain_events(
        &self,
    ) -> Option<tokio::sync::broadcast::Receiver<observe::BrainEvent>> {
        self.observer.as_ref().map(|o| o.subscribe())
    }

    /// Attach an `IdentityStore`. When wired and a
    /// `Signal` carries a `Principal`, the pipeline gates the classified
    /// intent through `IdentityStore::check` before executing.
    /// See the identity crate's docs for verb/tier semantics.
    pub fn with_identity_store(mut self, store: Arc<dyn identity::IdentityStore>) -> Self {
        self.identity_store = Some(store);
        self
    }

    /// Expose the configured identity store, if any.
    pub fn identity_store(&self) -> Option<&Arc<dyn identity::IdentityStore>> {
        self.identity_store.as_ref()
    }

    /// Attach a Terminal Bridge so `OpenTerminalSession` /
    /// `ListTerminalSessions` / `CloseTerminalSession` intents can drive
    /// real PTY sessions. Without this, the three intents return a
    /// "Terminal Bridge not configured" response.
    pub fn with_terminal_bridge(mut self, bridge: Arc<terminal::TerminalBridge>) -> Self {
        self.terminal_bridge = Some(bridge);
        self
    }

    /// Expose the configured terminal bridge, if any.
    pub fn terminal_bridge(&self) -> Option<&Arc<terminal::TerminalBridge>> {
        self.terminal_bridge.as_ref()
    }

    /// Attach an MCP host so `MountMcpServer` / `UnmountMcpServer` /
    /// `ListMcpServers` intents can drive real MCP server lifecycle.
    /// Without this, the three intents return a "MCP host not configured"
    /// response.
    pub fn with_mcp_host(mut self, host: Arc<dyn mcphost::MCPHost>) -> Self {
        self.mcp_host = Some(host);
        self
    }

    /// Expose the configured MCP host, if any.
    pub fn mcp_host(&self) -> Option<&Arc<dyn mcphost::MCPHost>> {
        self.mcp_host.as_ref()
    }

    /// Attach a tool registry. Populated by the MCP host and native
    /// backends at mount / registration time; the capability router (when
    /// wired) resolves `Intent::ToolCall` against this registry. Without it
    /// the router cannot enumerate tools and falls back to the
    /// router-not-configured placeholder.
    pub fn with_tool_registry(mut self, registry: Arc<dyn intent::ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    /// Expose the configured tool registry, if any.
    pub fn tool_registry(&self) -> Option<&Arc<dyn intent::ToolRegistry>> {
        self.tool_registry.as_ref()
    }

    /// Attach a capability router. The pipeline's `Intent::ToolCall` arm
    /// calls `router.resolve(&token)` and dispatches the returned
    /// `ToolRoute` through the wired MCP host / terminal bridge / native
    /// backends. Without a router, `Intent::ToolCall` falls back to the
    /// deterministic placeholder.
    pub fn with_intent_router(mut self, router: Arc<dyn intent::IntentRouter>) -> Self {
        self.intent_router = Some(router);
        self
    }

    /// Expose the configured intent router, if any.
    pub fn intent_router(&self) -> Option<&Arc<dyn intent::IntentRouter>> {
        self.intent_router.as_ref()
    }

    /// Attach a per-tool breaker registry. The pipeline records success /
    /// failure into this registry after every tool dispatch; the router
    /// queries it (via `intent::BreakerCheck`) to skip `Open` tools.
    /// Wiring the router and the registry separately is intentional —
    /// callers compose the two via `DefaultIntentRouter::with_breakers`.
    pub fn with_breaker_registry(mut self, registry: Arc<resilience::BreakerRegistry>) -> Self {
        self.breaker_registry = Some(registry);
        self
    }

    /// Expose the configured breaker registry, if any.
    pub fn breaker_registry(&self) -> Option<&Arc<resilience::BreakerRegistry>> {
        self.breaker_registry.as_ref()
    }

    /// Attach a standing-approval store. Wire the same `Arc` into the
    /// `ConfirmationEngine` so the bypass check and the slash commands
    /// see one consistent table.
    pub fn with_standing_approvals(
        mut self,
        store: Arc<dyn confirm::StandingApprovalStore>,
    ) -> Self {
        self.standing_approvals = Some(store);
        self
    }

    /// Expose the configured standing-approval store, if any.
    pub fn standing_approvals(&self) -> Option<&Arc<dyn confirm::StandingApprovalStore>> {
        self.standing_approvals.as_ref()
    }

    /// Override the inline confirmation gate's per-request timeout.
    /// Production leaves this `None`, deferring to the tier's default;
    /// tests pin a short value so the no-bypass path returns promptly.
    pub fn with_confirmation_timeout(mut self, t: std::time::Duration) -> Self {
        self.confirmation_timeout = Some(t);
        self
    }
}
