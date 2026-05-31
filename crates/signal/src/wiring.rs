//! Wiring, accessors, and scheduled intent management for SignalProcessor.

use std::sync::Arc;

use crate::notification;
use crate::SignalProcessor;

impl SignalProcessor {
    /// Expose the config (for adapter use).
    pub fn config(&self) -> &brain::BrainConfig {
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

    /// Expose the embedding provider (for the terminal graph sink, which
    /// embeds node bodies through the same provider). `Arc`-shared clone.
    pub fn embedder(&self) -> Option<Arc<hippocampus::Embedder>> {
        self.embedder.clone()
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
    pub fn metrics(&self) -> &Arc<brain::metrics::SubsystemMetrics> {
        &self.metrics
    }

    /// Get a cloneable handle to the LLM provider (for adapter use).
    pub fn llm_arc(&self) -> Arc<dyn cortex::LlmProvider> {
        self.llm.clone()
    }

    /// Attach a notification router (builder pattern).
    pub fn with_notification_router(mut self, router: notification::NotificationRouter) -> Self {
        self.channels.notification_router = Some(router);
        self
    }

    /// Expose the notification router.
    pub fn notification_router(&self) -> Option<&notification::NotificationRouter> {
        self.channels.notification_router.as_ref()
    }

    /// Attach an action dispatcher for executing tool intents (builder pattern).
    pub fn with_action_dispatcher(mut self, dispatcher: cortex::actions::ActionDispatcher) -> Self {
        self.capability.action_dispatcher = Some(dispatcher);
        self
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

    /// Expose the learned capability-fitness store. Used by the tool-loop
    /// advertiser (ranking nudge) and the capability digest (proven-tools
    /// line), and written to by `dispatch_tool_route` after each dispatch.
    pub(crate) fn fitness(&self) -> &cerebellum::CapabilityFitnessStore {
        &self.fitness
    }

    // ── Safety infrastructure builder methods ───────────────────────────

    /// Attach an audit trail (builder pattern).
    pub fn with_audit_trail(mut self, trail: Arc<dyn audit::AuditTrail>) -> Self {
        self.safety.audit_trail = Some(trail);
        self
    }

    /// Expose the audit trail.
    pub fn audit_trail(&self) -> Option<&Arc<dyn audit::AuditTrail>> {
        self.safety.audit_trail.as_ref()
    }

    /// Attach a confirmation engine (builder pattern).
    pub fn with_confirmation_engine(
        mut self,
        engine: Arc<dyn confirm::ConfirmationEngine>,
    ) -> Self {
        self.safety.confirmation_engine = Some(engine);
        self
    }

    /// Expose the confirmation engine.
    pub fn confirmation_engine(&self) -> Option<&Arc<dyn confirm::ConfirmationEngine>> {
        self.safety.confirmation_engine.as_ref()
    }

    /// Attach a cost budget (builder pattern).
    pub fn with_cost_budget(mut self, budget: Arc<dyn budget::CostBudget>) -> Self {
        self.safety.cost_budget = Some(budget);
        self
    }

    /// Expose the cost budget.
    pub fn cost_budget(&self) -> Option<&Arc<dyn budget::CostBudget>> {
        self.safety.cost_budget.as_ref()
    }

    /// Attach a sandbox executor (builder pattern).
    pub fn with_sandbox_executor(mut self, executor: Arc<dyn sandbox::SandboxExecutor>) -> Self {
        self.safety.sandbox_executor = Some(executor);
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

    /// Attach a dead-letter queue (builder pattern). The same `Arc` is
    /// expected to be wired into the `ResilientMcpHost` decorator so
    /// the host's enqueue path and the serve loop's drain task see one
    /// consistent backlog.
    pub fn with_dlq(mut self, dlq: Arc<dyn resilience::DeadLetterQueue>) -> Self {
        self.safety.dlq = Some(dlq);
        self
    }

    /// Expose the dead-letter queue.
    pub fn dlq(&self) -> Option<&Arc<dyn resilience::DeadLetterQueue>> {
        self.safety.dlq.as_ref()
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
        self.channels.channel_router = Some(router);
        self
    }

    /// Expose the channel router.
    pub fn channel_router(&self) -> Option<&Arc<dyn channel::ChannelRouter>> {
        self.channels.channel_router.as_ref()
    }

    /// Attach a channel preference store (builder pattern).
    pub fn with_channel_preferences(
        mut self,
        preferences: Arc<dyn channel::ChannelPreferenceStore>,
    ) -> Self {
        self.channels.channel_preferences = Some(preferences);
        self
    }

    /// Expose the channel preference store.
    pub fn channel_preferences(&self) -> Option<&Arc<dyn channel::ChannelPreferenceStore>> {
        self.channels.channel_preferences.as_ref()
    }

    /// Attach a confirmation correlator (builder pattern).
    pub fn with_confirmation_correlator(
        mut self,
        correlator: Arc<channel::ConfirmationCorrelator>,
    ) -> Self {
        self.channels.confirmation_correlator = Some(correlator);
        self
    }

    /// Expose the confirmation correlator.
    pub fn confirmation_correlator(&self) -> Option<&Arc<channel::ConfirmationCorrelator>> {
        self.channels.confirmation_correlator.as_ref()
    }

    /// Attach a channel dispatcher (builder pattern). The dispatcher owns
    /// transport handles and performs delivery — both the orchestrator's
    /// `Notify` step and the confirm engine's approval prompts route
    /// through it.
    pub fn with_channel_dispatcher(mut self, dispatcher: Arc<channel::ChannelDispatcher>) -> Self {
        self.channels.channel_dispatcher = Some(dispatcher);
        self
    }

    /// Expose the channel dispatcher.
    pub fn channel_dispatcher(&self) -> Option<&Arc<channel::ChannelDispatcher>> {
        self.channels.channel_dispatcher.as_ref()
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

    /// Expose the runtime proactivity toggle. The CLI bootstrap hands
    /// this `Arc` to the ganglia habit-engine and open-loop background
    /// tasks so they can skip generation when the user disables nudges
    /// at runtime. Spawn-time wiring still respects the startup config —
    /// the flag is a per-tick guard, not a re-spawn trigger.
    pub fn proactivity_enabled(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.proactivity_enabled.clone()
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
        self.observability.events_tx.subscribe()
    }

    /// Attach an observability event bus (builder pattern). When set, the
    /// pipeline publishes structured `BrainEvent`s alongside the legacy
    /// `SignalProcessedEvent` bus.
    pub fn with_observer(mut self, observer: Arc<dyn observe::Observer>) -> Self {
        self.observability.observer = Some(observer);
        self
    }

    /// Expose the configured observability bus, if any.
    pub fn observer(&self) -> Option<&Arc<dyn observe::Observer>> {
        self.observability.observer.as_ref()
    }

    /// Subscribe to the structured `BrainEvent` bus. Returns `None` if no
    /// observer was wired via `with_observer`.
    pub fn subscribe_brain_events(
        &self,
    ) -> Option<tokio::sync::broadcast::Receiver<observe::BrainEvent>> {
        self.observability.observer.as_ref().map(|o| o.subscribe())
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

    /// Attach Brain's product self-model (builder pattern). When wired, the chat
    /// prompt carries an authoritative "About Brain" grounding section (real CLI
    /// commands, config schema, policy) so the SOUL stops fabricating the
    /// product's own surface. Built from code at bootstrap.
    pub fn with_product_self_model(mut self, model: Arc<brain::ProductSelfModel>) -> Self {
        self.product_self_model = Some(model);
        self
    }

    /// Expose the configured product self-model, if any.
    pub fn product_self_model(&self) -> Option<&Arc<brain::ProductSelfModel>> {
        self.product_self_model.as_ref()
    }

    /// Attach a Terminal Bridge so `OpenTerminalSession` /
    /// `ListTerminalSessions` / `CloseTerminalSession` intents can drive
    /// real PTY sessions. Without this, the three intents return a
    /// "Terminal Bridge not configured" response.
    pub fn with_terminal_bridge(mut self, bridge: Arc<terminal::TerminalBridge>) -> Self {
        self.capability.terminal_bridge = Some(bridge);
        self
    }

    /// Expose the configured terminal bridge, if any.
    pub fn terminal_bridge(&self) -> Option<&Arc<terminal::TerminalBridge>> {
        self.capability.terminal_bridge.as_ref()
    }

    /// Attach an MCP host so `MountMcpServer` / `UnmountMcpServer` /
    /// `ListMcpServers` intents can drive real MCP server lifecycle.
    /// Without this, the three intents return a "MCP host not configured"
    /// response.
    pub fn with_mcp_host(mut self, host: Arc<dyn mcphost::MCPHost>) -> Self {
        self.capability.mcp_host = Some(host);
        self
    }

    /// Expose the configured MCP host, if any.
    pub fn mcp_host(&self) -> Option<&Arc<dyn mcphost::MCPHost>> {
        self.capability.mcp_host.as_ref()
    }

    /// Attach a tool registry. Populated by the MCP host and native
    /// backends at mount / registration time; the capability router (when
    /// wired) resolves `Intent::ToolCall` against this registry. Without it
    /// the router cannot enumerate tools and falls back to the
    /// router-not-configured placeholder.
    pub fn with_tool_registry(mut self, registry: Arc<dyn intent::ToolRegistry>) -> Self {
        self.capability.tool_registry = Some(registry);
        self
    }

    /// Expose the configured tool registry, if any.
    pub fn tool_registry(&self) -> Option<&Arc<dyn intent::ToolRegistry>> {
        self.capability.tool_registry.as_ref()
    }

    /// Attach a capability router. The pipeline's `Intent::ToolCall` arm
    /// calls `router.resolve(&token)` and dispatches the returned
    /// `ToolRoute` through the wired MCP host / terminal bridge / native
    /// backends. Without a router, `Intent::ToolCall` falls back to the
    /// deterministic placeholder.
    pub fn with_intent_router(mut self, router: Arc<dyn intent::IntentRouter>) -> Self {
        self.capability.intent_router = Some(router);
        self
    }

    /// Expose the configured intent router, if any.
    pub fn intent_router(&self) -> Option<&Arc<dyn intent::IntentRouter>> {
        self.capability.intent_router.as_ref()
    }

    /// Attach a per-tool breaker registry. The pipeline records success /
    /// failure into this registry after every tool dispatch; the router
    /// queries it (via `intent::BreakerCheck`) to skip `Open` tools.
    /// Wiring the router and the registry separately is intentional —
    /// callers compose the two via `DefaultIntentRouter::with_breakers`.
    pub fn with_breaker_registry(mut self, registry: Arc<resilience::BreakerRegistry>) -> Self {
        self.capability.breaker_registry = Some(registry);
        self
    }

    /// Expose the configured breaker registry, if any.
    pub fn breaker_registry(&self) -> Option<&Arc<resilience::BreakerRegistry>> {
        self.capability.breaker_registry.as_ref()
    }

    /// Attach the per-client rate-limit registry. Adapters (HTTP / WS /
    /// gRPC) call `.get_or_create(api_key).try_acquire()` per request and
    /// reject with the protocol's analog of HTTP 429 when the bucket is
    /// drained. Wiring is optional — without a registry, rate limiting
    /// is disabled.
    pub fn with_client_rate_limits(mut self, registry: Arc<resilience::RateLimitRegistry>) -> Self {
        self.client_rate_limits = Some(registry);
        self
    }

    /// Expose the configured per-client rate-limit registry, if any.
    pub fn client_rate_limits(&self) -> Option<&Arc<resilience::RateLimitRegistry>> {
        self.client_rate_limits.as_ref()
    }

    /// Attach a standing-approval store. Wire the same `Arc` into the
    /// `ConfirmationEngine` so the bypass check and the slash commands
    /// see one consistent table.
    pub fn with_standing_approvals(
        mut self,
        store: Arc<dyn confirm::StandingApprovalStore>,
    ) -> Self {
        self.safety.standing_approvals = Some(store);
        self
    }

    /// Expose the configured standing-approval store, if any.
    pub fn standing_approvals(&self) -> Option<&Arc<dyn confirm::StandingApprovalStore>> {
        self.safety.standing_approvals.as_ref()
    }

    /// Override the inline confirmation gate's per-request timeout.
    /// Production leaves this `None`, deferring to the tier's default;
    /// tests pin a short value so the no-bypass path returns promptly.
    pub fn with_confirmation_timeout(mut self, t: std::time::Duration) -> Self {
        self.safety.confirmation_timeout = Some(t);
        self
    }
}
