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
pub struct TaskOrchestrator {
    decomposer: Arc<dyn TaskDecomposer>,
    audit: Option<Arc<dyn audit::AuditTrail>>,
    confirm: Option<Arc<dyn confirm::ConfirmationEngine>>,
    budget: Option<Arc<dyn budget::CostBudget>>,
    sandbox: Option<Arc<dyn sandbox::SandboxExecutor>>,
    agents: Option<Arc<delegate::AgentRegistry>>,
    /// Default fallback chain applied to every delegation. Individual
    /// step failures follow this chain unless overridden in the future.
    delegation_policy: delegate::EscalationPolicy,
    /// Active tasks indexed by task ID.
    tasks: RwLock<HashMap<String, TaskState>>,
}

impl TaskOrchestrator {
    pub fn new(decomposer: Arc<dyn TaskDecomposer>) -> Self {
        Self {
            decomposer,
            audit: None,
            confirm: None,
            budget: None,
            sandbox: None,
            agents: None,
            delegation_policy: delegate::EscalationPolicy::default(),
            tasks: RwLock::new(HashMap::new()),
        }
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

        // Execute steps in topological order, respecting dependencies
        loop {
            let ready_steps = {
                let tasks = self.tasks.read().await;
                let task = tasks.get(task_id).unwrap();

                if task.is_complete() {
                    break;
                }

                let completed: HashSet<String> = task
                    .step_states
                    .iter()
                    .filter(|(_, s)| s.is_terminal())
                    .map(|(id, _)| id.clone())
                    .collect();
                task.graph.ready_steps(&completed)
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
        let task = tasks.get(task_id).unwrap();
        let summary = synthesize::summarize_task(task);

        Ok(summary)
    }

    /// Execute a single step.
    async fn execute_step(&self, task_id: &str, step_id: &str) -> Result<(), OrchestrateError> {
        let (action, tier, description) = {
            let tasks = self.tasks.read().await;
            let task = tasks.get(task_id).unwrap();
            let step = task.graph.steps.get(step_id).unwrap();
            (step.action.clone(), step.tier, step.description.clone())
        };

        // Mark as running
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks.get_mut(task_id).unwrap();
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
                let spec = confirm::ApprovalSpec::new(&description, convert_tier(tier));
                let nonce = spec.nonce.clone();

                // Mark as awaiting confirmation
                {
                    let mut tasks = self.tasks.write().await;
                    let task = tasks.get_mut(task_id).unwrap();
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
                        let task = tasks.get_mut(task_id).unwrap();
                        task.set_step_state(step_id, StepState::Cancelled);
                        tracing::info!(step = %description, reason = %reason, "Step cancelled");
                        return Ok(());
                    }
                    Err(e) => {
                        let mut tasks = self.tasks.write().await;
                        let task = tasks.get_mut(task_id).unwrap();
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
            StepAction::Research { query } => Ok(StepOutcome {
                stdout: format!("Research query: {query}"),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Researched: {query}"),
            }),
            StepAction::Plan { output } => Ok(StepOutcome {
                stdout: output.clone(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: "Plan produced".to_string(),
            }),
            StepAction::Implement { spec, agent } => {
                self.delegate_implement_step(spec, agent).await
            }
            StepAction::Review { artifact } => Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![artifact.clone()],
                summary: format!("Review requested: {artifact}"),
            }),
            StepAction::Notify { channel, message } => Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Notified {channel}: {message}"),
            }),
        };

        // Update step state
        let mut tasks = self.tasks.write().await;
        let task = tasks.get_mut(task_id).unwrap();

        match result {
            Ok(outcome) => {
                // Record in audit trail
                if let Some(audit) = &self.audit {
                    let entry = audit::AuditEntry::new(
                        &description,
                        "step executed",
                        &outcome.summary,
                        convert_audit_tier(tier),
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
                task.set_step_state(
                    step_id,
                    StepState::Failed {
                        error: error.clone(),
                        retryable: true,
                        failed_at: Utc::now(),
                    },
                );
            }
        }

        // Check if task is complete
        if task.is_complete() {
            task.phase = TaskPhase::Completed;
            task.completed_at = Some(Utc::now());
            tracing::info!(task_id = %task_id, "Task completed");
        }

        Ok(())
    }

    /// Execute a command in the sandbox.
    async fn execute_sandbox_step(
        &self,
        command: &str,
        workdir: &std::path::Path,
    ) -> Result<StepOutcome, String> {
        let sandbox = match &self.sandbox {
            Some(s) => s,
            None => {
                return Err("Sandbox not available".to_string());
            }
        };

        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty command".to_string());
        }

        let cmd = sandbox::SandboxCommand::new(
            parts[0],
            parts[1..].iter().map(|s| s.to_string()).collect(),
        )
        .with_workdir(workdir.to_path_buf());

        match sandbox.run(cmd).await {
            Ok(outcome) => Ok(StepOutcome {
                stdout: outcome.stdout,
                stderr: outcome.stderr,
                exit_code: Some(outcome.exit_code),
                artifacts: vec![],
                summary: if outcome.exit_code == 0 {
                    format!("Command succeeded: {command}")
                } else {
                    format!("Command failed (exit {}): {command}", outcome.exit_code)
                },
            }),
            Err(e) => Err(format!("Sandbox execution failed: {e}")),
        }
    }

    /// Hand the step off to a registered [`AgentDelegate`]. Failures are
    /// run through the configured escalation policy — a primary hang or
    /// launch failure transparently falls over to the declared fallback
    /// chain; anything the chain can't recover becomes a human escalation
    /// recorded as a failed step outcome.
    async fn delegate_implement_step(
        &self,
        spec: &str,
        agent: &str,
    ) -> Result<StepOutcome, String> {
        let registry = self
            .agents
            .as_ref()
            .ok_or_else(|| "Agent registry not attached to orchestrator".to_string())?;

        let primary = registry
            .get(agent)
            .map_err(|e| format!("Delegate '{agent}' unavailable: {e}"))?;

        let task = delegate::AgentTask::new(spec);
        let outcome =
            delegate::run_with_escalation(primary, registry.as_ref(), task, &self.delegation_policy)
                .await;

        match outcome {
            delegate::EscalationOutcome::Succeeded(result) => Ok(StepOutcome {
                stdout: result.stdout,
                stderr: result.stderr,
                exit_code: result.exit_code,
                artifacts: result
                    .artifacts
                    .iter()
                    .map(|a| a.reference.clone())
                    .collect(),
                summary: format!("{agent}: {}", result.summary),
            }),
            delegate::EscalationOutcome::Recovered { via, result } => Ok(StepOutcome {
                stdout: result.stdout,
                stderr: result.stderr,
                exit_code: result.exit_code,
                artifacts: result
                    .artifacts
                    .iter()
                    .map(|a| a.reference.clone())
                    .collect(),
                summary: format!("{agent} failed; recovered via {via}: {}", result.summary),
            }),
            delegate::EscalationOutcome::EscalateToHuman { reason } => Err(reason),
        }
    }

    /// Get the current state of a task.
    pub async fn get_task(&self, task_id: &str) -> Option<TaskState> {
        self.tasks.read().await.get(task_id).cloned()
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

/// Convert sandbox ActionTier to confirm ActionTier.
/// These are the same enum duplicated across crates — convert between them.
fn convert_tier(tier: audit::ActionTier) -> confirm::ActionTier {
    match tier {
        audit::ActionTier::Read => confirm::ActionTier::Read,
        audit::ActionTier::Write => confirm::ActionTier::Write,
        audit::ActionTier::Execute => confirm::ActionTier::Execute,
        audit::ActionTier::Destructive => confirm::ActionTier::Destructive,
        audit::ActionTier::External => confirm::ActionTier::External,
    }
}

/// Convert to audit ActionTier for audit entries.
fn convert_audit_tier(tier: audit::ActionTier) -> audit::ActionTier {
    tier
}

/// Check if the tier requires confirmation (using confirm crate's logic).
trait RequiresConfirmation {
    fn requires_confirmation(self) -> bool;
}

impl RequiresConfirmation for audit::ActionTier {
    fn requires_confirmation(self) -> bool {
        matches!(
            self,
            audit::ActionTier::Destructive | audit::ActionTier::External
        )
    }
}

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
}
