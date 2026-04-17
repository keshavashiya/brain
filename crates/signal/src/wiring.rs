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

    // ── Phase 1: Safety infrastructure builder methods ──────────────────

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

    /// Expose the sandbox executor.
    pub fn sandbox_executor(&self) -> Option<&Arc<dyn sandbox::SandboxExecutor>> {
        self.sandbox_executor.as_ref()
    }

    /// Attach a credential vault (builder pattern).
    pub fn with_credential_vault(mut self, vault: Arc<dyn vault::CredentialVault>) -> Self {
        self.credential_vault = Some(vault);
        self
    }

    /// Expose the credential vault.
    pub fn credential_vault(&self) -> Option<&Arc<dyn vault::CredentialVault>> {
        self.credential_vault.as_ref()
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
}
