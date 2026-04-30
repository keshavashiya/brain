//! Task orchestrator — the execution loop that coordinates decomposition,
//! approval, execution, and outcome synthesis.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing;

use crate::decompose::{DecompositionContext, DecompositionError, TaskDecomposer};
use crate::graph::TaskGraph;
use crate::state::{StepOutcome, StepState, TaskPhase, TaskState};
use crate::step::StepAction;
use crate::synthesize;

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
    /// runs can recall them. Phase 3 result aggregation.
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

    /// Override the default delegation escalation policy.
    pub fn with_delegation_policy(mut self, policy: delegate::EscalationPolicy) -> Self {
        self.delegation_policy = policy;
        self
    }

    /// Decompose a user request into a task plan.
    /// Returns the task ID and a formatted plan for user review.
    pub async fn plan(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<(String, String), OrchestrateError> {
        tracing::info!(request = %request, "Decomposing task");

        let steps = self.decomposer.decompose(request, context).await?;
        let graph = TaskGraph::from_steps(steps)?;

        let task_id = uuid::Uuid::new_v4().to_string();
        let mut task_state = TaskState::new(task_id.clone(), request.to_string(), graph);
        task_state.phase = TaskPhase::AwaitingApproval;

        let plan_text = synthesize::format_plan_for_approval(&task_state);

        // Record in audit trail
        if let Some(audit) = &self.audit {
            let entry = audit::AuditEntry::new(
                request,
                "decomposed into task plan",
                &plan_text,
                audit::ActionTier::Read,
            )
            .with_source("orchestrator");
            if let Err(e) = audit.record(entry).await {
                tracing::warn!("Failed to audit task plan: {e}");
            }
        }

        self.tasks.write().await.insert(task_id.clone(), task_state);

        tracing::info!(task_id = %task_id, "Task plan created");
        Ok((task_id, plan_text))
    }

    /// Execute a previously planned task (after user approval).
    pub async fn execute(&self, task_id: &str) -> Result<String, OrchestrateError> {
        // Transition to executing phase
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .ok_or_else(|| OrchestrateError::TaskNotFound(task_id.to_string()))?;
            task.phase = TaskPhase::Executing;
        }

        tracing::info!(task_id = %task_id, "Starting task execution");

        // Execute steps in topological order, respecting dependencies.
        //
        // `ready_steps` is computed against the *succeeded* set, not the
        // terminal set — a failed step must NOT unblock its dependents.
        // Failure cascades are handled below by marking dependents
        // `Skipped` so the loop still terminates without busy-looping.
        loop {
            let ready_steps = {
                let tasks = self.tasks.read().await;
                let task = tasks
                    .get(task_id)
                    .expect("invariant: task inserted by plan(); only state changes after");

                if task.is_complete() {
                    break;
                }

                let succeeded: HashSet<String> = task
                    .step_states
                    .iter()
                    .filter(|(_, s)| s.is_success())
                    .map(|(id, _)| id.clone())
                    .collect();
                // `ready_steps` only checks dep-satisfaction — it does
                // NOT exclude steps that are already terminal. Without
                // this filter a Failed step (which is not in `succeeded`
                // and has no missing deps) would be picked as "ready"
                // again on the next iteration, re-running the failure
                // and re-triggering the replan loop. Only steps whose
                // current state is Pending may be (re)scheduled.
                task.graph
                    .ready_steps(&succeeded)
                    .into_iter()
                    .filter(|id| {
                        matches!(
                            task.step_states.get(id),
                            Some(StepState::Pending) | Some(StepState::Ready)
                        )
                    })
                    .collect::<Vec<_>>()
            };

            if ready_steps.is_empty() {
                // No ready steps but not complete — some steps must be blocked
                // (running or awaiting confirmation). Break to avoid busy-loop.
                break;
            }

            // Execute ready steps (sequentially for now; parallel in future)
            for step_id in &ready_steps {
                self.execute_step(task_id, step_id).await?;
            }
        }

        // Generate summary
        let tasks = self.tasks.read().await;
        let task = tasks
            .get(task_id)
            .expect("invariant: task inserted by plan() and never removed");
        let summary = synthesize::summarize_task(task);

        Ok(summary)
    }

    /// Execute a single step.
    async fn execute_step(&self, task_id: &str, step_id: &str) -> Result<(), OrchestrateError> {
        let (action, tier, description) = {
            let tasks = self.tasks.read().await;
            let task = tasks
                .get(task_id)
                .expect("invariant: task_id always corresponds to a planned task");
            let step = task
                .graph
                .steps
                .get(step_id)
                .expect("invariant: step_id sourced from task.graph.ready_steps()");
            (step.action.clone(), step.tier, step.description.clone())
        };

        // Mark as running
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .expect("invariant: task_id always corresponds to a planned task");
            task.set_step_state(
                step_id,
                StepState::Running {
                    started_at: Utc::now(),
                },
            );
        }

        tracing::info!(task_id = %task_id, step_id = %step_id, step = %description, "Executing step");

        // Check confirmation for destructive/external tiers
        if tier.requires_confirmation() {
            if let Some(confirm) = &self.confirm {
                let spec = confirm::ApprovalSpec::new(&description, tier);
                let nonce = spec.nonce.clone();

                // Mark as awaiting confirmation
                {
                    let mut tasks = self.tasks.write().await;
                    let task = tasks
                        .get_mut(task_id)
                        .expect("invariant: task_id always corresponds to a planned task");
                    task.set_step_state(
                        step_id,
                        StepState::AwaitingConfirmation {
                            nonce: nonce.clone(),
                            since: Utc::now(),
                        },
                    );
                }

                match confirm.request(spec).await {
                    Ok(confirm::ApprovalOutcome::Approved) => {
                        tracing::info!(step = %description, "Step approved");
                    }
                    Ok(outcome) => {
                        let reason = format!("Approval denied: {outcome:?}");
                        let mut tasks = self.tasks.write().await;
                        let task = tasks
                            .get_mut(task_id)
                            .expect("invariant: task_id always corresponds to a planned task");
                        task.set_step_state(step_id, StepState::Cancelled);
                        tracing::info!(step = %description, reason = %reason, "Step cancelled");
                        return Ok(());
                    }
                    Err(e) => {
                        let mut tasks = self.tasks.write().await;
                        let task = tasks
                            .get_mut(task_id)
                            .expect("invariant: task_id always corresponds to a planned task");
                        task.set_step_state(
                            step_id,
                            StepState::Failed {
                                error: format!("Confirmation error: {e}"),
                                retryable: true,
                                failed_at: Utc::now(),
                            },
                        );
                        return Ok(());
                    }
                }
            }
        }

        // Execute the action
        let result = match &action {
            StepAction::Execute { command, workdir } | StepAction::Test { command, workdir } => {
                self.execute_sandbox_step(command, workdir).await
            }
            StepAction::Shell { command, workdir } => {
                self.execute_shell_step(command, workdir).await
            }
            StepAction::Research { query } => self.execute_research_step(query).await,
            StepAction::Plan { output } => {
                // A `Plan` step that carries no output is effectively a
                // no-op — the LLM emitted a step the executor cannot
                // perform but marked it `plan` so it would silently
                // succeed. Treat that as an honest failure so the user
                // sees that nothing happened, instead of a "succeeded"
                // count that masks an empty result.
                let trimmed = output.trim();
                if trimmed.is_empty() {
                    Err(format!(
                        "Plan step '{description}' had no output to produce — \
                         the planner did not specify what this step should write. \
                         Re-plan with concrete steps (research/execute/implement)."
                    ))
                } else {
                    Ok(StepOutcome {
                        stdout: output.clone(),
                        stderr: String::new(),
                        exit_code: None,
                        artifacts: vec![],
                        summary: summarize_first_line(trimmed),
                    })
                }
            }
            StepAction::Implement { spec, agent } => {
                self.delegate_implement_step(spec, agent).await
            }
            StepAction::Review { artifact } => self.execute_review_step(artifact).await,
            StepAction::Notify { channel, message } => {
                self.execute_notify_step(channel, message).await
            }
        };

        // Update step state
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .expect("invariant: task_id always corresponds to a planned task");

        match result {
            Ok(outcome) => {
                // Record in audit trail
                if let Some(audit) = &self.audit {
                    let entry = audit::AuditEntry::new(
                        &description,
                        "step executed",
                        &outcome.summary,
                        tier,
                    )
                    .with_source("orchestrator")
                    .with_execution(
                        outcome.stdout.clone(),
                        outcome.stderr.clone(),
                        outcome.exit_code.unwrap_or(0),
                        0, // duration tracked elsewhere
                    );
                    if let Err(e) = audit.record(entry).await {
                        tracing::warn!("Failed to audit step outcome: {e}");
                    }
                }

                task.set_step_state(
                    step_id,
                    StepState::Completed {
                        outcome,
                        completed_at: Utc::now(),
                    },
                );
            }
            Err(error) => {
                // Mirror the success-path audit write so failed steps
                // are recorded in the audit trail too — otherwise a
                // sandbox exit-1 disappears from history once we lifted
                // it out of the Ok arm.
                if let Some(audit) = &self.audit {
                    let entry = audit::AuditEntry::new(&description, "step failed", &error, tier)
                        .with_source("orchestrator")
                        .with_outcome(audit::AuditOutcome::Failure);
                    if let Err(e) = audit.record(entry).await {
                        tracing::warn!("Failed to audit step failure: {e}");
                    }
                }

                task.set_step_state(
                    step_id,
                    StepState::Failed {
                        error: error.clone(),
                        retryable: true,
                        failed_at: Utc::now(),
                    },
                );

                // Mark all transitive dependents `Skipped` so the loop
                // terminates and the user sees an honest status instead
                // of cascading attempts against missing inputs.
                let dependents = task.graph.transitive_dependents(step_id);
                let reason = format!("dependency {step_id} failed");
                for dep_id in dependents {
                    if let Some(state) = task.step_states.get(&dep_id) {
                        if !state.is_terminal() {
                            task.set_step_state(
                                &dep_id,
                                StepState::Skipped {
                                    reason: reason.clone(),
                                },
                            );
                        }
                    }
                }

                // Drop the write lock before the (potentially slow) LLM
                // replan call below. We still own a snapshot of the
                // fields the replan needs.
                drop(tasks);

                // Try to repair the plan if we still have replan budget.
                // Best-effort: a replan failure leaves the task in the
                // standard "failed step + skipped dependents" state.
                self.try_replan_after_failure(task_id, step_id, &description, &error)
                    .await;

                // Re-acquire the lock to mark the task complete (or not).
                let mut tasks = self.tasks.write().await;
                let task = tasks
                    .get_mut(task_id)
                    .expect("invariant: task_id always corresponds to a planned task");
                if task.is_complete() {
                    task.phase = TaskPhase::Completed;
                    task.completed_at = Some(Utc::now());
                    tracing::info!(task_id = %task_id, "Task completed");
                }
                return Ok(());
            }
        }

        // Check if task is complete (success path).
        if task.is_complete() {
            task.phase = TaskPhase::Completed;
            task.completed_at = Some(Utc::now());
            tracing::info!(task_id = %task_id, "Task completed");
        }

        Ok(())
    }

    /// Best-effort corrective replan after a step failure. Asks the
    /// decomposer for a fresh sub-plan given the original goal +
    /// what's already succeeded + the actual error, then splices the
    /// new steps into the graph so the execution loop picks them up
    /// next iteration. Bounded by `MAX_REPLAN_ATTEMPTS`.
    pub(crate) async fn try_replan_after_failure(
        &self,
        task_id: &str,
        failed_step_id: &str,
        failed_step_description: &str,
        error: &str,
    ) {
        // Snapshot the fields we need under a short read lock.
        let (request, completed, attempts) = {
            let tasks = self.tasks.read().await;
            let task = match tasks.get(task_id) {
                Some(t) => t,
                None => return,
            };
            if task.replan_attempts >= MAX_REPLAN_ATTEMPTS {
                tracing::info!(
                    task_id = %task_id,
                    attempts = task.replan_attempts,
                    "replan budget exhausted; leaving plan in failed state"
                );
                return;
            }
            // Stdout per completed step, capped so a single noisy step
            // can't dominate the prompt. The replan LLM uses these to
            // ground its next step in the real data prior steps produced.
            const PER_STEP_OUTPUT_LIMIT: usize = 1500;
            let completed: Vec<crate::decompose::CompletedStepRecap> = task
                .graph
                .topological_order()
                .into_iter()
                .filter_map(|id| {
                    let state = task.step_states.get(&id)?;
                    let StepState::Completed { outcome, .. } = state else {
                        return None;
                    };
                    let step = task.graph.steps.get(&id)?;
                    let trimmed = outcome.stdout.trim();
                    let excerpt = if trimmed.len() > PER_STEP_OUTPUT_LIMIT {
                        let head = &trimmed[..PER_STEP_OUTPUT_LIMIT];
                        format!("{head}\n…[truncated]")
                    } else {
                        trimmed.to_string()
                    };
                    Some(crate::decompose::CompletedStepRecap {
                        description: step.description.clone(),
                        output_excerpt: excerpt,
                    })
                })
                .collect();
            (task.request.clone(), completed, task.replan_attempts)
        };

        let context = crate::decompose::DecompositionContext {
            available_tools: self.available_tools.clone(),
            ..Default::default()
        };
        let repair = crate::decompose::RepairContext {
            original_request: request,
            failed_step: failed_step_description.to_string(),
            error: error.to_string(),
            completed,
        };

        tracing::info!(
            task_id = %task_id,
            failed_step_id = %failed_step_id,
            attempt = attempts + 1,
            max = MAX_REPLAN_ATTEMPTS,
            "attempting replan after step failure"
        );

        let new_steps = match self.decomposer.replan_after_failure(repair, context).await {
            Ok(steps) if !steps.is_empty() => steps,
            Ok(_) => {
                tracing::info!(task_id = %task_id, "replan returned empty plan; skipping");
                return;
            }
            Err(e) => {
                tracing::warn!(task_id = %task_id, error = %e, "replan failed; leaving plan as-is");
                return;
            }
        };

        // Splice the new steps in. Each new step's depends_on already
        // references its sibling new steps via UUIDs from build_task_step
        // (via the sequential-fallback in replan_after_failure), so the
        // first new step has no deps and runs immediately on the next
        // execute() loop iteration.
        let mut tasks = self.tasks.write().await;
        let task = match tasks.get_mut(task_id) {
            Some(t) => t,
            None => return,
        };

        let new_ids: Vec<String> = new_steps.iter().map(|s| s.id.clone()).collect();
        match task.graph.add_steps(new_steps) {
            Ok(()) => {
                for id in &new_ids {
                    task.step_states
                        .insert(id.clone(), crate::state::StepState::Pending);
                }
                task.replan_attempts += 1;
                tracing::info!(
                    task_id = %task_id,
                    spliced = new_ids.len(),
                    total_attempts = task.replan_attempts,
                    "replan succeeded; new steps spliced into graph"
                );
            }
            Err(e) => {
                tracing::warn!(task_id = %task_id, error = %e, "splicing replan steps failed");
            }
        }
    }

    /// Get the current state of a task.
    pub async fn get_task(&self, task_id: &str) -> Option<TaskState> {
        self.tasks.read().await.get(task_id).cloned()
    }

    /// Return task IDs currently in the `AwaitingApproval` phase. Used by
    /// the signal pipeline to resolve bare `approve` / `reject` (no id)
    /// to the single pending plan when there's exactly one.
    pub async fn pending_approvals(&self) -> Vec<String> {
        self.tasks
            .read()
            .await
            .iter()
            .filter(|(_, t)| t.phase == TaskPhase::AwaitingApproval)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// List all active tasks.
    pub async fn list_tasks(&self) -> Vec<(String, String, TaskPhase)> {
        self.tasks
            .read()
            .await
            .iter()
            .map(|(id, t)| (id.clone(), t.request.clone(), t.phase))
            .collect()
    }

    /// Cancel a task.
    pub async fn cancel(&self, task_id: &str) -> Result<(), OrchestrateError> {
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .ok_or_else(|| OrchestrateError::TaskNotFound(task_id.to_string()))?;
        task.phase = TaskPhase::Cancelled;
        for (_, state) in task.step_states.iter_mut() {
            if !state.is_terminal() {
                *state = StepState::Cancelled;
            }
        }
        Ok(())
    }
}

/// First non-empty line of `s` truncated to 160 chars — used for short
/// step summaries surfaced in the user-facing task report.
fn summarize_first_line(s: &str) -> String {
    let line = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("Plan produced");
    if line.chars().count() > 160 {
        let truncated: String = line.chars().take(157).collect();
        format!("{truncated}…")
    } else {
        line.to_string()
    }
}

// `audit::ActionTier`, `confirm::ActionTier`, and `sandbox::ActionTier`
// are now all re-exports of `brain_core::ActionTier`. The previous
// `convert_tier` / `convert_audit_tier` / local `RequiresConfirmation`
// trait existed solely to bridge the three former duplicate enums.
// `requires_confirmation()` is now an inherent method on the canonical
// type — no shim required.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompose::DecompositionContext;
    use crate::step::{StepAction, TaskStep};

    /// A mock decomposer that returns a fixed set of steps.
    struct MockDecomposer {
        steps: Vec<TaskStep>,
    }

    #[async_trait::async_trait]
    impl TaskDecomposer for MockDecomposer {
        async fn decompose(
            &self,
            _request: &str,
            _context: DecompositionContext,
        ) -> Result<Vec<TaskStep>, DecompositionError> {
            Ok(self.steps.clone())
        }
    }

    fn test_steps() -> Vec<TaskStep> {
        vec![
            TaskStep {
                id: "s1".to_string(),
                description: "Research".to_string(),
                action: StepAction::Research {
                    query: "test".to_string(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            },
            TaskStep {
                id: "s2".to_string(),
                description: "Test".to_string(),
                action: StepAction::Execute {
                    command: "echo hello".to_string(),
                    workdir: "/tmp".into(),
                },
                depends_on: vec!["s1".to_string()],
                tier: audit::ActionTier::Execute,
                estimated_tokens: 0,
            },
        ]
    }

    #[tokio::test]
    async fn test_plan_creates_task() {
        let decomposer = Arc::new(MockDecomposer {
            steps: test_steps(),
        });
        let orchestrator = TaskOrchestrator::new(decomposer);

        let (task_id, plan_text) = orchestrator
            .plan("build something", DecompositionContext::default())
            .await
            .unwrap();

        assert!(!task_id.is_empty());
        assert!(plan_text.contains("Research"));
        assert!(plan_text.contains("Test"));

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(task.phase, TaskPhase::AwaitingApproval);
    }

    #[tokio::test]
    async fn test_execute_runs_steps() {
        let sandbox = Arc::new(sandbox::StubSandbox::new());
        let decomposer = Arc::new(MockDecomposer {
            steps: test_steps(),
        });
        let orchestrator = TaskOrchestrator::new(decomposer).with_sandbox(sandbox);

        let (task_id, _) = orchestrator
            .plan("build something", DecompositionContext::default())
            .await
            .unwrap();

        let summary = orchestrator.execute(&task_id).await.unwrap();
        assert!(summary.contains("Completed"));

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(task.phase, TaskPhase::Completed);
        assert!(task.all_succeeded());
    }

    #[tokio::test]
    async fn test_implement_step_dispatches_through_registry() {
        use async_trait::async_trait;
        use chrono::Utc;
        use delegate::{
            AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
            AgentTaskStatus,
        };

        struct StubAgent;

        #[async_trait]
        impl AgentDelegate for StubAgent {
            fn name(&self) -> &str {
                "stub"
            }
            fn capabilities(&self) -> AgentCapabilities {
                AgentCapabilities::default()
            }
            async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
                let now = Utc::now();
                Ok(AgentResult {
                    task_id: task.id,
                    status: AgentTaskStatus::Succeeded,
                    summary: format!("stubbed: {}", task.description),
                    artifacts: vec![],
                    stdout: "ok".to_string(),
                    stderr: String::new(),
                    exit_code: Some(0),
                    started_at: now,
                    completed_at: now,
                })
            }
        }

        let mut registry = AgentRegistry::new();
        registry.register(Arc::new(StubAgent));
        let registry = Arc::new(registry);

        let implement_step = TaskStep {
            id: "impl".to_string(),
            description: "Implement feature".to_string(),
            action: StepAction::Implement {
                spec: "write a README".to_string(),
                agent: "stub".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Write,
            estimated_tokens: 0,
        };
        let decomposer = Arc::new(MockDecomposer {
            steps: vec![implement_step],
        });
        let orchestrator = TaskOrchestrator::new(decomposer).with_agents(registry);

        let (task_id, _) = orchestrator
            .plan("build it", DecompositionContext::default())
            .await
            .unwrap();
        let summary = orchestrator.execute(&task_id).await.unwrap();
        assert!(summary.contains("Completed"));

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert!(task.all_succeeded());
        let step = task.step_states.get("impl").unwrap();
        match step {
            StepState::Completed { outcome, .. } => {
                assert!(outcome.summary.contains("stub"));
                assert!(outcome.summary.contains("write a README"));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_implement_step_without_registry_fails() {
        let implement_step = TaskStep {
            id: "impl".to_string(),
            description: "Implement feature".to_string(),
            action: StepAction::Implement {
                spec: "do the thing".to_string(),
                agent: "ghost".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Write,
            estimated_tokens: 0,
        };
        let decomposer = Arc::new(MockDecomposer {
            steps: vec![implement_step],
        });
        let orchestrator = TaskOrchestrator::new(decomposer);

        let (task_id, _) = orchestrator
            .plan("build it", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.execute(&task_id).await.unwrap();

        let task = orchestrator.get_task(&task_id).await.unwrap();
        let step = task.step_states.get("impl").unwrap();
        assert!(
            matches!(step, StepState::Failed { .. }),
            "expected Failed without registry, got {step:?}"
        );
    }

    #[tokio::test]
    async fn failed_step_skips_dependents_instead_of_running_them() {
        // Regression: previously `is_terminal()` was used to decide which
        // deps were satisfied, so a Failed step unblocked its dependents
        // and they ran against missing inputs. Now they should be Skipped.
        let steps = vec![
            TaskStep {
                id: "s1".to_string(),
                description: "fail".to_string(),
                action: StepAction::Implement {
                    spec: "won't matter".to_string(),
                    agent: "missing".to_string(), // no registry → fails
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            },
            TaskStep {
                id: "s2".to_string(),
                description: "depends on s1".to_string(),
                action: StepAction::Plan {
                    output: "should not run".to_string(),
                },
                depends_on: vec!["s1".to_string()],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            },
            TaskStep {
                id: "s3".to_string(),
                description: "depends on s2".to_string(),
                action: StepAction::Plan {
                    output: "should not run".to_string(),
                },
                depends_on: vec!["s2".to_string()],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            },
        ];
        let decomposer = Arc::new(MockDecomposer { steps });
        let orchestrator = TaskOrchestrator::new(decomposer);

        let (task_id, _) = orchestrator
            .plan("anything", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.execute(&task_id).await.unwrap();

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert!(matches!(
            task.step_states.get("s1"),
            Some(StepState::Failed { .. })
        ));
        assert!(
            matches!(task.step_states.get("s2"), Some(StepState::Skipped { .. })),
            "s2 should be Skipped after s1 failed, got {:?}",
            task.step_states.get("s2")
        );
        assert!(
            matches!(task.step_states.get("s3"), Some(StepState::Skipped { .. })),
            "s3 should be transitively Skipped, got {:?}",
            task.step_states.get("s3")
        );
        assert_eq!(task.phase, TaskPhase::Completed);
    }

    #[tokio::test]
    async fn nonzero_exit_marks_step_failed_and_skips_dependents() {
        // Regression for the daemon RCA: a sandbox command that returns
        // exit_code != 0 used to be recorded as `Completed` because the
        // executor returned `Ok(StepOutcome { exit_code: Some(1), .. })`.
        // It must now be marked Failed so dependents cascade-skip.
        let sandbox = Arc::new(sandbox::StubSandbox::new());
        let steps = vec![
            TaskStep {
                id: "fail".to_string(),
                description: "always-fail command".to_string(),
                action: StepAction::Execute {
                    command: "false".to_string(),
                    workdir: "/tmp".into(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Execute,
                estimated_tokens: 0,
            },
            TaskStep {
                id: "after".to_string(),
                description: "should be skipped".to_string(),
                action: StepAction::Plan {
                    output: "must not run".to_string(),
                },
                depends_on: vec!["fail".to_string()],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            },
        ];
        let decomposer = Arc::new(MockDecomposer { steps });
        let orchestrator = TaskOrchestrator::new(decomposer).with_sandbox(sandbox);

        let (task_id, _) = orchestrator
            .plan("anything", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.execute(&task_id).await.unwrap();

        let task = orchestrator.get_task(&task_id).await.unwrap();
        let fail = task.step_states.get("fail").unwrap();
        assert!(
            matches!(fail, StepState::Failed { .. }),
            "non-zero exit must mark step Failed, got {fail:?}"
        );
        let after = task.step_states.get("after").unwrap();
        assert!(
            matches!(after, StepState::Skipped { .. }),
            "dependent must be Skipped, got {after:?}"
        );
    }

    #[tokio::test]
    async fn replan_on_failure_splices_corrective_steps() {
        // After a step fails, the orchestrator should call the
        // decomposer's replan_after_failure hook and splice the
        // returned steps into the graph so they execute next.
        use crate::decompose::RepairContext;

        struct ReplanDecomposer {
            initial: Vec<TaskStep>,
            replan_called: std::sync::atomic::AtomicUsize,
            replan_steps: Vec<TaskStep>,
        }

        #[async_trait::async_trait]
        impl TaskDecomposer for ReplanDecomposer {
            async fn decompose(
                &self,
                _request: &str,
                _context: DecompositionContext,
            ) -> Result<Vec<TaskStep>, crate::decompose::DecompositionError> {
                Ok(self.initial.clone())
            }
            async fn replan_after_failure(
                &self,
                _repair: RepairContext,
                _context: DecompositionContext,
            ) -> Result<Vec<TaskStep>, crate::decompose::DecompositionError> {
                self.replan_called
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(self.replan_steps.clone())
            }
        }

        // Initial plan: one step that always fails.
        let initial = vec![TaskStep {
            id: "fail".to_string(),
            description: "missing-agent step".to_string(),
            action: StepAction::Implement {
                spec: "doomed".to_string(),
                agent: "ghost".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }];
        // Replan plan: a single Plan step that always succeeds.
        let replan_steps = vec![TaskStep {
            id: "replan-1".to_string(),
            description: "corrective step".to_string(),
            action: StepAction::Plan {
                output: "fixed it".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }];

        let decomposer = Arc::new(ReplanDecomposer {
            initial,
            replan_called: std::sync::atomic::AtomicUsize::new(0),
            replan_steps: replan_steps.clone(),
        });
        let decomposer_handle = decomposer.clone();
        let orchestrator = TaskOrchestrator::new(decomposer);

        let (task_id, _) = orchestrator
            .plan("anything", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.execute(&task_id).await.unwrap();

        assert_eq!(
            decomposer_handle
                .replan_called
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "decomposer.replan_after_failure must be invoked exactly once"
        );

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(
            task.replan_attempts, 1,
            "task.replan_attempts must increment after a successful splice"
        );
        // The original step stays Failed; the replanned step succeeds.
        assert!(matches!(
            task.step_states.get("fail"),
            Some(StepState::Failed { .. })
        ));
        assert!(matches!(
            task.step_states.get("replan-1"),
            Some(StepState::Completed { .. })
        ));
        // Task is now complete with mixed outcomes.
        assert_eq!(task.phase, TaskPhase::Completed);
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let decomposer = Arc::new(MockDecomposer {
            steps: test_steps(),
        });
        let orchestrator = TaskOrchestrator::new(decomposer);

        let (task_id, _) = orchestrator
            .plan("build something", DecompositionContext::default())
            .await
            .unwrap();

        orchestrator.cancel(&task_id).await.unwrap();

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(task.phase, TaskPhase::Cancelled);
    }

    #[tokio::test]
    async fn notify_with_no_channels_is_soft_success() {
        // When the dispatcher has no transports registered, the router
        // returns NoChannelAvailable. The orchestrator must NOT fail the
        // step — replan-on-failure produces Notify steps as its honest
        // "I cannot do this" path, and a hard failure here recurses into
        // more Notify steps until the replan budget is exhausted (see
        // brain.log:1036–1043 for the user-visible cascade).
        let db = storage::SqlitePool::open_memory().unwrap();
        let prefs = Arc::new(channel::SqlitePreferenceStore::new(db));
        prefs.ensure_tables().unwrap();
        let router: Arc<dyn channel::ChannelRouter> =
            Arc::new(channel::DefaultChannelRouter::new(prefs));
        let dispatcher = Arc::new(channel::ChannelDispatcher::new(router));

        let decomposer = Arc::new(MockDecomposer {
            steps: test_steps(),
        });
        let orchestrator = TaskOrchestrator::new(decomposer).with_channel_dispatcher(dispatcher);

        let outcome = orchestrator
            .execute_notify_step("default", "PDF cannot be parsed: pdftotext missing")
            .await
            .expect("notify must not fail when no channels are configured");
        assert!(outcome.summary.contains("no external channel"));
        assert!(outcome.summary.contains("pdftotext missing"));
    }
}
