//! Task orchestrator — the execution loop that coordinates decomposition,
//! approval, execution, and outcome synthesis.
//!
//! The `TaskOrchestrator` `impl` is split across sibling modules sharing the
//! `pub(crate)` fields: construction + cancellation-token helpers here,
//! `plan`/queries/state-machine in `crate::lifecycle`, the execution loop in
//! `crate::execute`, corrective replanning in `crate::replan`, and the
//! per-`StepAction` handlers in `crate::actions` / `crate::aggregation`.

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::decompose::{DecompositionError, TaskDecomposer};
use crate::state::{StepState, TaskState};

#[derive(Debug, Error)]
pub enum OrchestrateError {
    #[error("Decomposition failed: {0}")]
    Decomposition(#[from] DecompositionError),
    #[error("Graph error: {0}")]
    Graph(#[from] crate::graph::GraphError),
    #[error("Sandbox error: {0}")]
    Sandbox(String),
    #[error("Confirmation error: {0}")]
    Confirmation(String),
    #[error("Budget exceeded: {0}")]
    BudgetExceeded(String),
    #[error("Audit error: {0}")]
    Audit(String),
    #[error("Task not found: {0}")]
    TaskNotFound(String),
    #[error("Task cancelled")]
    Cancelled,
}

/// The task orchestrator — manages the full lifecycle of task plans.
///
/// Fields are `pub(crate)` so per-action handlers (`crate::actions`) and
/// aggregation helpers (`crate::aggregation`) can split `impl` across
/// sibling modules. Outside the `orchestrate` crate the struct's surface
/// is the public methods only.
pub struct TaskOrchestrator {
    pub(crate) decomposer: Arc<dyn TaskDecomposer>,
    pub(crate) audit: Option<Arc<dyn audit::AuditTrail>>,
    pub(crate) confirm: Option<Arc<dyn confirm::ConfirmationEngine>>,
    pub(crate) budget: Option<Arc<dyn budget::CostBudget>>,
    pub(crate) sandbox: Option<Arc<dyn sandbox::SandboxExecutor>>,
    pub(crate) agents: Option<Arc<delegate::AgentRegistry>>,
    /// LLM provider for `Research` / `Review` step types.
    pub(crate) llm: Option<Arc<dyn cortex::LlmProvider>>,
    /// Channel dispatcher for `Notify` step types.
    pub(crate) dispatcher: Option<Arc<channel::ChannelDispatcher>>,
    /// Episodic memory store — captures delegation outcomes so future
    /// runs can recall them.
    pub(crate) episodic: Option<Arc<hippocampus::EpisodicStore>>,
    /// Default fallback chain applied to every delegation. Individual
    /// step failures follow this chain unless overridden in the future.
    pub(crate) delegation_policy: delegate::EscalationPolicy,
    /// Cached binary allowlist used to rebuild a `DecompositionContext`
    /// inside the replan-on-failure loop. Populated by the wiring
    /// layer; empty by default (no allowlist constraint surfaced to
    /// the LLM during replan).
    pub(crate) available_tools: Vec<String>,
    /// Active tasks indexed by task ID.
    pub(crate) tasks: RwLock<HashMap<String, TaskState>>,
    /// Observer bus for `BrainEvent::TaskStateChange` emissions. When
    /// unwired, transitions still update the in-memory state and the
    /// optional persistence pool, but no event goes out — existing tests
    /// can keep building bare orchestrators.
    pub(crate) observer: Option<Arc<dyn observe::Observer>>,
    /// SQLite pool used to append rows to the `task_states` audit table
    /// (migration v22). When unwired, the state-machine history lives
    /// only in memory.
    pub(crate) state_pool: Option<storage::SqlitePool>,
    /// Per-task cancellation tokens (PR-6b). Created on `plan()`,
    /// observed at every orchestrator checkpoint (the execute loop, the
    /// confirmation wait, the per-action future, the replan LLM call),
    /// and fired by `cancel()` so in-flight child futures abort within
    /// one polling cycle instead of waiting for the current step to
    /// finish.
    pub(crate) cancel_tokens: RwLock<HashMap<String, CancellationToken>>,
    /// Maximum number of independent ready steps the execute loop runs
    /// concurrently within one wave. `1` (the default) preserves strictly
    /// sequential execution — the wiring layer raises it from config so
    /// production exploits the DAG's parallel branches. Confirmation
    /// prompts within a wave are always resolved sequentially regardless
    /// of this value; only the approved actions run concurrently.
    pub(crate) max_parallel_steps: usize,
}

/// Maximum number of replan-on-failure attempts per task. Bounds LLM
/// cost when the model keeps producing plans the sandbox refuses.
pub(crate) const MAX_REPLAN_ATTEMPTS: u32 = 2;

impl TaskOrchestrator {
    pub fn new(decomposer: Arc<dyn TaskDecomposer>) -> Self {
        Self {
            decomposer,
            audit: None,
            confirm: None,
            budget: None,
            sandbox: None,
            agents: None,
            llm: None,
            dispatcher: None,
            episodic: None,
            delegation_policy: delegate::EscalationPolicy::default(),
            available_tools: Vec::new(),
            tasks: RwLock::new(HashMap::new()),
            observer: None,
            state_pool: None,
            cancel_tokens: RwLock::new(HashMap::new()),
            max_parallel_steps: 1,
        }
    }

    /// Look up the per-task cancellation token. Returns a fresh
    /// (never-cancelled) token for unknown task IDs so callers that pre-
    /// date the cancel-token map (e.g. tasks constructed by tests that
    /// inject directly into `self.tasks`) keep their old behavior — they
    /// just never observe a cancel signal.
    pub(crate) async fn cancel_token_for(&self, task_id: &str) -> CancellationToken {
        self.cancel_tokens
            .read()
            .await
            .get(task_id)
            .cloned()
            .unwrap_or_else(CancellationToken::new)
    }

    /// Mark a single step `Cancelled` under a brief write lock. Used by
    /// the cancellation arms of `execute_step` so the per-step state
    /// reflects the abort even when `cancel()` raced ahead (which would
    /// have flipped it to Cancelled already — overwriting Cancelled with
    /// Cancelled is a no-op).
    pub(crate) async fn mark_step_cancelled(&self, task_id: &str, step_id: &str) {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(task_id) {
            task.set_step_state(step_id, StepState::Cancelled);
        }
    }

    /// Cache the sandbox's binary allowlist so the replan-on-failure
    /// loop can include it in its corrective LLM call. Without this the
    /// replan call has no allowlist context and may suggest binaries
    /// the sandbox would reject.
    pub fn with_available_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    pub fn with_audit(mut self, audit: Arc<dyn audit::AuditTrail>) -> Self {
        self.audit = Some(audit);
        self
    }

    pub fn with_confirmation(mut self, confirm: Arc<dyn confirm::ConfirmationEngine>) -> Self {
        self.confirm = Some(confirm);
        self
    }

    pub fn with_budget(mut self, budget: Arc<dyn budget::CostBudget>) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn with_sandbox(mut self, sandbox: Arc<dyn sandbox::SandboxExecutor>) -> Self {
        self.sandbox = Some(sandbox);
        self
    }

    /// Attach the agent registry — enables `StepAction::Implement`
    /// dispatch to specialist delegates.
    pub fn with_agents(mut self, agents: Arc<delegate::AgentRegistry>) -> Self {
        self.agents = Some(agents);
        self
    }

    /// Attach an LLM provider so `Research` and `Review` steps actually
    /// run a model call instead of returning a no-op string.
    pub fn with_llm(mut self, llm: Arc<dyn cortex::LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Attach a channel dispatcher so `Notify` steps actually deliver
    /// the message to the user's preferred channel.
    pub fn with_channel_dispatcher(mut self, dispatcher: Arc<channel::ChannelDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Attach an episodic memory store — delegate outcomes are recorded
    /// so they're searchable in future sessions.
    pub fn with_episodic(mut self, store: Arc<hippocampus::EpisodicStore>) -> Self {
        self.episodic = Some(store);
        self
    }

    /// Attach an observer bus so phase transitions emit
    /// [`observe::BrainEvent::TaskStateChange`]. Unwired = silent.
    pub fn with_observer(mut self, observer: Arc<dyn observe::Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Attach a SQLite pool so phase transitions append rows to the
    /// `task_states` audit table (migration v22). Unwired = in-memory
    /// history only.
    pub fn with_state_pool(mut self, pool: storage::SqlitePool) -> Self {
        self.state_pool = Some(pool);
        self
    }

    /// Override the default delegation escalation policy.
    pub fn with_delegation_policy(mut self, policy: delegate::EscalationPolicy) -> Self {
        self.delegation_policy = policy;
        self
    }

    /// Set the maximum number of independent ready steps to run
    /// concurrently per wave. Clamped to at least `1` so `0` can never
    /// stall the loop; `1` keeps execution strictly sequential.
    pub fn with_max_parallel_steps(mut self, max: usize) -> Self {
        self.max_parallel_steps = max.max(1);
        self
    }
}

#[cfg(test)]
mod tests;
