//! Task orchestrator — the execution loop that coordinates decomposition,
//! approval, execution, and outcome synthesis.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::Utc;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
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
        let task_state = TaskState::new(task_id.clone(), request.to_string(), graph);

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
        self.cancel_tokens
            .write()
            .await
            .insert(task_id.clone(), CancellationToken::new());

        // State-machine: emit the initial `planning` entry, then
        // transition to AwaitingApproval. Both events are visible to the
        // observer and persisted to `task_states` (if a pool is wired).
        self.record_initial_planning(&task_id).await;
        self.transition_phase(&task_id, TaskPhase::AwaitingApproval)
            .await;

        tracing::info!(task_id = %task_id, "Task plan created");
        Ok((task_id, plan_text))
    }

    /// Execute a previously planned task (after user approval).
    pub async fn execute(&self, task_id: &str) -> Result<String, OrchestrateError> {
        // Confirm the task exists before any state work so a wrong
        // task_id never produces a phantom transition event.
        {
            let tasks = self.tasks.read().await;
            if !tasks.contains_key(task_id) {
                return Err(OrchestrateError::TaskNotFound(task_id.to_string()));
            }
        }
        // PR-6b: clone the task's cancellation token up-front. Every
        // checkpoint below — top of loop, per-step dispatch, the per-
        // action future, the confirmation wait, the replan LLM call —
        // races against `token.cancelled()` so a `cancel()` call mid-
        // step aborts within one polling cycle.
        let token = self.cancel_token_for(task_id).await;
        if token.is_cancelled() {
            // Cancel fired before execute() even started; honor it.
            return Ok(synthesize::summarize_task(
                self.tasks
                    .read()
                    .await
                    .get(task_id)
                    .expect("invariant: task_id is present (checked above)"),
            ));
        }
        self.transition_phase(task_id, TaskPhase::Executing).await;

        tracing::info!(task_id = %task_id, "Starting task execution");

        // Execute steps in topological order, respecting dependencies.
        //
        // `ready_steps` is computed against the *succeeded* set, not the
        // terminal set — a failed step must NOT unblock its dependents.
        // Failure cascades are handled below by marking dependents
        // `Skipped` so the loop still terminates without busy-looping.
        loop {
            if token.is_cancelled() {
                tracing::info!(task_id = %task_id, "execute loop observed cancellation");
                break;
            }
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
                if token.is_cancelled() {
                    break;
                }
                self.execute_step(task_id, step_id, &token).await?;
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
    async fn execute_step(
        &self,
        task_id: &str,
        step_id: &str,
        token: &CancellationToken,
    ) -> Result<(), OrchestrateError> {
        // Pre-flight: if cancellation already fired (e.g. between the
        // outer loop's check and us entering this fn), mark the step
        // cancelled and bail without touching the action handlers.
        if token.is_cancelled() {
            self.mark_step_cancelled(task_id, step_id).await;
            return Ok(());
        }
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

                // PR-6b: race the confirmation wait against the task
                // token. A `cancel()` mid-prompt aborts the wait so the
                // step doesn't block forever on a confirmation that will
                // never come.
                let confirm_outcome = tokio::select! {
                    biased;
                    _ = token.cancelled() => {
                        self.mark_step_cancelled(task_id, step_id).await;
                        return Ok(());
                    }
                    r = confirm.request(spec) => r,
                };
                match confirm_outcome {
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

        // Execute the action. PR-6b: race against `token.cancelled()`
        // so an in-flight sandbox/LLM/delegate call aborts mid-flight.
        // Dropping the action future is cancel-safe — none of the
        // handlers hold mutable global state past an await.
        let result = tokio::select! {
            biased;
            _ = token.cancelled() => {
                self.mark_step_cancelled(task_id, step_id).await;
                return Ok(());
            }
            r = async { match &action {
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
        } } => r,
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
                self.try_replan_after_failure(task_id, step_id, &description, &error, token)
                    .await;

                // Re-check completion + drive the canonical
                // Reconciling → (Completed | Failed) shape under the
                // state-machine helper.
                let (done, all_succeeded) = {
                    let tasks = self.tasks.read().await;
                    let task = tasks
                        .get(task_id)
                        .expect("invariant: task_id always corresponds to a planned task");
                    (task.is_complete(), task.all_succeeded())
                };
                if done {
                    self.transition_phase(task_id, TaskPhase::Reconciling).await;
                    let terminal = if all_succeeded {
                        TaskPhase::Completed
                    } else {
                        TaskPhase::Failed
                    };
                    self.transition_phase(task_id, terminal).await;
                    tracing::info!(task_id = %task_id, terminal = %terminal.as_str(), "Task complete");
                }
                return Ok(());
            }
        }

        // Drop the write lock before the I/O-bound transition_phase
        // calls below — they take their own lock for the brief in-mem
        // flip and we don't want to hold the executor's lock through
        // the audit-row write and observer publish.
        drop(tasks);
        let (done, all_succeeded) = {
            let tasks = self.tasks.read().await;
            let task = tasks
                .get(task_id)
                .expect("invariant: task_id always corresponds to a planned task");
            (task.is_complete(), task.all_succeeded())
        };
        if done {
            self.transition_phase(task_id, TaskPhase::Reconciling).await;
            let terminal = if all_succeeded {
                TaskPhase::Completed
            } else {
                TaskPhase::Failed
            };
            self.transition_phase(task_id, terminal).await;
            tracing::info!(task_id = %task_id, terminal = %terminal.as_str(), "Task complete");
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
        token: &CancellationToken,
    ) {
        // PR-6b: don't burn an LLM call on a task that just got
        // cancelled. The execute loop's next iteration will see the
        // cancellation and break anyway, but skipping the replan saves
        // the round-trip.
        if token.is_cancelled() {
            return;
        }
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

        let replan_call = self.decomposer.replan_after_failure(repair, context);
        let new_steps = tokio::select! {
            biased;
            _ = token.cancelled() => {
                tracing::info!(task_id = %task_id, "replan aborted by cancellation");
                return;
            }
            r = replan_call => match r {
                Ok(steps) if !steps.is_empty() => steps,
                Ok(_) => {
                    tracing::info!(task_id = %task_id, "replan returned empty plan; skipping");
                    return;
                }
                Err(e) => {
                    tracing::warn!(task_id = %task_id, error = %e, "replan failed; leaving plan as-is");
                    return;
                }
            },
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

    /// Cancel a task. Flips all non-terminal step states to `Cancelled`,
    /// transitions the task phase to `Cancelled`, and (PR-6b) fires the
    /// per-task [`CancellationToken`] so any in-flight step future
    /// observing the token aborts within one polling cycle — without
    /// PR-6b, cancellation would have to wait for the current step to
    /// finish on its own.
    pub async fn cancel(&self, task_id: &str) -> Result<(), OrchestrateError> {
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .ok_or_else(|| OrchestrateError::TaskNotFound(task_id.to_string()))?;
            for (_, state) in task.step_states.iter_mut() {
                if !state.is_terminal() {
                    *state = StepState::Cancelled;
                }
            }
        }
        self.transition_phase(task_id, TaskPhase::Cancelled).await;
        // Fire the cancellation token AFTER state has already been
        // flipped to Cancelled — that way a select-loser that races to
        // overwrite step state with Cancelled is a no-op, not a write
        // that could clobber a Completed/Failed transition that
        // legitimately landed first.
        if let Some(t) = self.cancel_tokens.read().await.get(task_id) {
            t.cancel();
        }
        Ok(())
    }

    /// State-machine helper. The single canonical mutator of
    /// [`TaskState::phase`]: takes the write lock just long enough to
    /// flip the in-memory field, then releases it before doing
    /// I/O-bound work (audit row write + observer publish). Idempotent
    /// for terminal transitions — if a task is already in a terminal
    /// phase, the helper is a no-op so cancel-then-complete races stay
    /// well-defined.
    pub(crate) async fn transition_phase(&self, task_id: &str, to: TaskPhase) {
        // Read prior phase + write the new one under one lock. The
        // bound block guarantees the guard drops before the async I/O
        // below so other handlers aren't blocked on the disk write.
        let from = {
            let mut tasks = self.tasks.write().await;
            let task = match tasks.get_mut(task_id) {
                Some(t) => t,
                None => return,
            };
            if task.phase.is_terminal() && task.phase != to {
                // Already done — refuse to flip out of a terminal
                // state so a late completion doesn't overwrite a
                // cancellation.
                tracing::debug!(
                    task_id = %task_id,
                    from = %task.phase.as_str(),
                    to = %to.as_str(),
                    "ignoring transition out of terminal state"
                );
                return;
            }
            if task.phase == to {
                return;
            }
            let from = task.phase;
            task.phase = to;
            if to.is_terminal() {
                task.completed_at = Some(Utc::now());
            }
            from
        };

        // Audit table append (best-effort — a write failure is logged
        // and we proceed so the in-memory phase update isn't undone).
        if let Some(pool) = &self.state_pool {
            let task_id_owned = task_id.to_string();
            let state_str = to.as_str();
            let res = pool.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO task_states (task_id, state) VALUES (?1, ?2)",
                    rusqlite::params![task_id_owned, state_str],
                )?;
                Ok(())
            });
            if let Err(e) = res {
                tracing::warn!(
                    task_id = %task_id,
                    state = %to.as_str(),
                    error = %e,
                    "task_states row append failed"
                );
            }
        }

        // Observer publish (best-effort, same rationale).
        if let Some(observer) = &self.observer {
            let event = observe::BrainEvent::TaskStateChange {
                id: uuid::Uuid::new_v4(),
                task_id: task_id.to_string(),
                from: from.as_str().to_string(),
                to: to.as_str().to_string(),
                ts: Utc::now(),
            };
            let _ = observer.publish(event).await;
        }

        tracing::info!(
            task_id = %task_id,
            from = %from.as_str(),
            to = %to.as_str(),
            "task phase transition"
        );
    }

    /// Convenience: emit the initial Planning transition (`from = "none"`).
    /// Called from [`plan`] right after the task is inserted into the
    /// active map, so the audit table records the task's birth before
    /// any subsequent state moves.
    pub(crate) async fn record_initial_planning(&self, task_id: &str) {
        if let Some(pool) = &self.state_pool {
            let task_id_owned = task_id.to_string();
            let res = pool.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO task_states (task_id, state) VALUES (?1, 'planning')",
                    rusqlite::params![task_id_owned],
                )?;
                Ok(())
            });
            if let Err(e) = res {
                tracing::warn!(
                    task_id = %task_id,
                    error = %e,
                    "initial planning state append failed"
                );
            }
        }
        if let Some(observer) = &self.observer {
            let event = observe::BrainEvent::TaskStateChange {
                id: uuid::Uuid::new_v4(),
                task_id: task_id.to_string(),
                from: "none".into(),
                to: "planning".into(),
                ts: Utc::now(),
            };
            let _ = observer.publish(event).await;
        }
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

    // ── State-machine acceptance ────────────────────────────────────────

    #[tokio::test]
    async fn phase6_state_machine_emits_canonical_transitions_and_persists_rows() {
        use observe::{BrainEvent, BroadcastObserver, Observer};

        let pool = storage::SqlitePool::open_memory().unwrap();
        let observer_arc = BroadcastObserver::new();
        // Subscribe BEFORE the orchestrator runs so the broadcast
        // channel has a live receiver — without a subscriber,
        // `BroadcastObserver::publish` returns `Err(BusClosed)` and
        // the orchestrator's best-effort send swallows it.
        let mut rx = observer_arc.subscribe();
        let observer: Arc<dyn Observer> = observer_arc.clone();

        // One no-op Plan step is the cheapest path through execute()
        // that exercises is_complete() + all_succeeded() → Reconciling
        // → Completed without dragging in the sandbox / LLM / agent
        // registry. Plan-with-non-empty-output succeeds cleanly.
        let decomposer = Arc::new(MockDecomposer {
            steps: vec![TaskStep {
                id: "s1".to_string(),
                description: "no-op step".to_string(),
                action: StepAction::Plan {
                    output: "did nothing observable".to_string(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            }],
        });
        let orchestrator = TaskOrchestrator::new(decomposer)
            .with_observer(observer)
            .with_state_pool(pool.clone());

        let (task_id, _plan) = orchestrator
            .plan("phase6 smoke", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.execute(&task_id).await.unwrap();

        // Drain observed transitions.
        let mut transitions: Vec<(String, String)> = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            if let BrainEvent::TaskStateChange { from, to, .. } = ev {
                transitions.push((from, to));
            }
        }

        let expected: Vec<(&str, &str)> = vec![
            ("none", "planning"),
            ("planning", "awaiting_approval"),
            ("awaiting_approval", "executing"),
            ("executing", "reconciling"),
            ("reconciling", "completed"),
        ];
        let observed: Vec<(&str, &str)> = transitions
            .iter()
            .map(|(f, t)| (f.as_str(), t.as_str()))
            .collect();
        assert_eq!(observed, expected, "transition sequence mismatch");

        // The `task_states` audit table should mirror the events,
        // newest-last. ORDER BY id ASC preserves insertion order even
        // if two transitions land in the same wall-clock second.
        let states_in_db = pool
            .with_conn(|conn| {
                let mut stmt = conn
                    .prepare("SELECT state FROM task_states WHERE task_id = ?1 ORDER BY id ASC")?;
                let states: Vec<String> = stmt
                    .query_map([&task_id], |r| r.get::<_, String>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(states)
            })
            .unwrap();
        assert_eq!(
            states_in_db,
            vec![
                "planning".to_string(),
                "awaiting_approval".to_string(),
                "executing".to_string(),
                "reconciling".to_string(),
                "completed".to_string(),
            ],
            "task_states audit rows must mirror the emitted events"
        );

        // Final in-memory phase agrees with the table.
        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(task.phase, TaskPhase::Completed);
        assert!(task.completed_at.is_some());
    }

    #[tokio::test]
    async fn phase6_failed_step_lands_in_failed_terminal_state() {
        // Empty-output Plan step is the deterministic failure path —
        // the executor treats it as "honest failure" so we don't need
        // to mock a flaky sandbox.
        use observe::{BroadcastObserver, Observer};

        let pool = storage::SqlitePool::open_memory().unwrap();
        let observer_arc = BroadcastObserver::new();
        let _rx = observer_arc.subscribe();
        let observer: Arc<dyn Observer> = observer_arc.clone();

        let decomposer = Arc::new(MockDecomposer {
            steps: vec![TaskStep {
                id: "s1".to_string(),
                description: "failing step".to_string(),
                action: StepAction::Plan {
                    output: String::new(), // empty → fail
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            }],
        });
        let orchestrator = TaskOrchestrator::new(decomposer)
            .with_observer(observer)
            .with_state_pool(pool.clone());

        let (task_id, _) = orchestrator
            .plan("phase6 fail", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.execute(&task_id).await.unwrap();

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(
            task.phase,
            TaskPhase::Failed,
            "task with a failed step must land in Failed, not Completed"
        );

        // Sanity: the final row in task_states is `failed`.
        let last_state: String = pool
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT state FROM task_states WHERE task_id = ?1 ORDER BY id DESC LIMIT 1",
                    [&task_id],
                    |r| r.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap();
        assert_eq!(last_state, "failed");
    }

    #[tokio::test]
    async fn phase6_terminal_transitions_are_idempotent() {
        // After Cancelled, a stray Completed transition must not flip
        // the phase back. Protects against a slow-completing step that
        // returns after the user has already cancelled.
        let decomposer = Arc::new(MockDecomposer {
            steps: vec![TaskStep {
                id: "s1".to_string(),
                description: "any".to_string(),
                action: StepAction::Plan {
                    output: "ok".to_string(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            }],
        });
        let orchestrator = TaskOrchestrator::new(decomposer);
        let (task_id, _) = orchestrator
            .plan(
                "phase6 cancel-then-late-completion",
                DecompositionContext::default(),
            )
            .await
            .unwrap();
        orchestrator.cancel(&task_id).await.unwrap();
        orchestrator
            .transition_phase(&task_id, TaskPhase::Completed)
            .await;
        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(
            task.phase,
            TaskPhase::Cancelled,
            "late Completed transition must not overwrite Cancelled"
        );
    }

    // ── PR-6b: CancellationToken propagation ────────────────────────────

    #[tokio::test]
    async fn pr6b_cancel_aborts_in_flight_step_within_one_polling_cycle() {
        // A step that would otherwise sleep for an hour must abort
        // promptly once `cancel()` fires. Acceptance criterion:
        // execute() returns within a bounded wall-clock window
        // after cancel() — we use 2s to give CI breathing room.
        use async_trait::async_trait;
        use chrono::Utc;
        use delegate::{
            AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
            AgentTaskStatus,
        };
        use std::time::{Duration, Instant};

        struct SlowAgent;
        #[async_trait]
        impl AgentDelegate for SlowAgent {
            fn name(&self) -> &str {
                "slow"
            }
            fn capabilities(&self) -> AgentCapabilities {
                AgentCapabilities::default()
            }
            async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
                // Sleep for an hour — well beyond any reasonable test
                // timeout. Drop-on-cancel from tokio::select! must
                // interrupt this so execute() can return.
                tokio::time::sleep(Duration::from_secs(3600)).await;
                let now = Utc::now();
                Ok(AgentResult {
                    task_id: task.id,
                    status: AgentTaskStatus::Succeeded,
                    summary: "unreachable".to_string(),
                    artifacts: vec![],
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: Some(0),
                    started_at: now,
                    completed_at: now,
                })
            }
        }

        let mut registry = AgentRegistry::new();
        registry.register(Arc::new(SlowAgent));
        let registry = Arc::new(registry);

        let decomposer = Arc::new(MockDecomposer {
            steps: vec![TaskStep {
                id: "slow".to_string(),
                description: "long-running step".to_string(),
                action: StepAction::Implement {
                    spec: "do nothing forever".to_string(),
                    agent: "slow".to_string(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            }],
        });
        let orchestrator = Arc::new(TaskOrchestrator::new(decomposer).with_agents(registry));

        let (task_id, _) = orchestrator
            .plan("pr6b mid-step cancel", DecompositionContext::default())
            .await
            .unwrap();

        let exec_orch = orchestrator.clone();
        let exec_task_id = task_id.clone();
        let exec_handle =
            tokio::spawn(async move { exec_orch.execute(&exec_task_id).await.unwrap() });

        // Let execute() actually enter the slow step before we cancel —
        // otherwise the cancel could race ahead of the spawn and just
        // hit the outer-loop pre-check, which wouldn't exercise the
        // mid-step abort path we're testing.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let cancel_at = Instant::now();
        orchestrator.cancel(&task_id).await.unwrap();

        // execute() must return shortly after cancel — well under the
        // 3600s the SlowAgent would otherwise sleep for.
        let _summary = tokio::time::timeout(Duration::from_secs(2), exec_handle)
            .await
            .expect("execute() must return within 2s of cancel; did the token thread through?")
            .expect("execute task panicked");
        let elapsed = cancel_at.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "execute() returned but took {elapsed:?} after cancel — cancellation should be near-instant"
        );

        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(
            task.phase,
            TaskPhase::Cancelled,
            "task must land in Cancelled after mid-step cancel"
        );
        // The slow step itself should be Cancelled, not lingering in
        // Running. cancel() flips state, and the select-loser's
        // mark_step_cancelled overwrite is a no-op — either way the
        // observed state is Cancelled.
        assert!(
            matches!(task.step_states.get("slow"), Some(StepState::Cancelled)),
            "in-flight step must be Cancelled, got {:?}",
            task.step_states.get("slow")
        );
    }

    #[tokio::test]
    async fn pr6b_cancel_before_execute_exits_without_running_steps() {
        // If cancel() fires after plan() but before execute() starts,
        // execute() should observe the cancellation and exit without
        // touching any step handlers. The task lands Cancelled.
        let decomposer = Arc::new(MockDecomposer {
            steps: test_steps(),
        });
        let orchestrator = TaskOrchestrator::new(decomposer);

        let (task_id, _) = orchestrator
            .plan("pr6b pre-execute cancel", DecompositionContext::default())
            .await
            .unwrap();
        orchestrator.cancel(&task_id).await.unwrap();

        let _summary = orchestrator.execute(&task_id).await.unwrap();
        let task = orchestrator.get_task(&task_id).await.unwrap();
        assert_eq!(task.phase, TaskPhase::Cancelled);
        // No step should have advanced past Cancelled — set by cancel()
        // before execute() ran.
        for (id, state) in &task.step_states {
            assert!(
                matches!(state, StepState::Cancelled),
                "step {id} should be Cancelled, got {state:?}"
            );
        }
    }

    #[tokio::test]
    async fn pr6b_cancel_token_fires_when_cancel_called() {
        // Lower-level invariant: cancel() must actually fire the
        // per-task token so future select! consumers (e.g. an external
        // bridge that wires its own cancel-aware future) observe it.
        let decomposer = Arc::new(MockDecomposer {
            steps: test_steps(),
        });
        let orchestrator = TaskOrchestrator::new(decomposer);
        let (task_id, _) = orchestrator
            .plan("pr6b token fires", DecompositionContext::default())
            .await
            .unwrap();
        let token = orchestrator.cancel_token_for(&task_id).await;
        assert!(!token.is_cancelled(), "token must start uncancelled");
        orchestrator.cancel(&task_id).await.unwrap();
        assert!(
            token.is_cancelled(),
            "cancel() must fire the per-task token"
        );
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
        // State-machine: a task with any non-succeeded step
        // lands in `Failed`, not `Completed`.
        assert_eq!(task.phase, TaskPhase::Failed);
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
        // Mixed-outcome task lands in `Failed` — the
        // original failure is recorded even though the replan
        // succeeded. (The user can still see the replanned-step
        // success in the per-step states.)
        assert_eq!(task.phase, TaskPhase::Failed);
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
