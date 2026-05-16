//! Task and step state machine.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::graph::TaskGraph;

/// State of a single step in the task plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StepState {
    /// Waiting for dependencies.
    Pending,
    /// Dependencies met, awaiting execution slot.
    Ready,
    /// Currently executing.
    Running { started_at: DateTime<Utc> },
    /// Waiting for human approval.
    AwaitingConfirmation { nonce: String, since: DateTime<Utc> },
    /// Completed successfully.
    Completed {
        outcome: StepOutcome,
        completed_at: DateTime<Utc>,
    },
    /// Failed.
    Failed {
        error: String,
        retryable: bool,
        failed_at: DateTime<Utc>,
    },
    /// Skipped (dependency failed, user chose to skip).
    Skipped { reason: String },
    /// Cancelled by user or timeout.
    Cancelled,
}

impl StepState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            StepState::Completed { .. }
                | StepState::Failed { .. }
                | StepState::Skipped { .. }
                | StepState::Cancelled
        )
    }

    pub fn is_success(&self) -> bool {
        matches!(self, StepState::Completed { .. })
    }
}

/// Outcome of a completed step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutcome {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub artifacts: Vec<String>,
    pub summary: String,
}

/// Overall phase of the task.
///
/// The canonical state machine is
/// `Planning → Executing → Reconciling → (Completed | Failed)`.
/// `AwaitingApproval` is a side-channel state entered before `Executing`
/// when the plan needs human consent; `Cancelled` is a terminal state
/// reachable from anywhere via [`crate::TaskOrchestrator::cancel`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskPhase {
    /// Plan is being assembled.
    Planning,
    /// Waiting for user approval of the plan.
    AwaitingApproval,
    /// Steps are being executed.
    Executing,
    /// All steps complete; verifying the world matches what was planned
    /// before committing to a terminal phase. The explicit "world-drift
    /// detection" hook; today it's a brief no-op transition that lands
    /// on `Completed` or `Failed` based on step outcomes.
    Reconciling,
    /// All steps completed successfully.
    Completed,
    /// At least one step failed and the orchestrator gave up
    /// (replan budget exhausted or non-retryable error).
    Failed,
    /// Task was cancelled.
    Cancelled,
}

impl TaskPhase {
    /// Stable wire string used by [`observe::BrainEvent::TaskStateChange`]
    /// and the `task_states` audit table. Snake-case matches the serde
    /// representation; keep them in sync if either changes.
    pub fn as_str(self) -> &'static str {
        match self {
            TaskPhase::Planning => "planning",
            TaskPhase::AwaitingApproval => "awaiting_approval",
            TaskPhase::Executing => "executing",
            TaskPhase::Reconciling => "reconciling",
            TaskPhase::Completed => "completed",
            TaskPhase::Failed => "failed",
            TaskPhase::Cancelled => "cancelled",
        }
    }

    /// True for phases the orchestrator never transitions out of.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TaskPhase::Completed | TaskPhase::Failed | TaskPhase::Cancelled
        )
    }
}

/// Complete state of a task — graph + per-step states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskState {
    /// Unique task ID.
    pub id: String,
    /// Original user request.
    pub request: String,
    /// The task graph (steps + dependencies).
    pub graph: TaskGraph,
    /// Per-step state.
    pub step_states: HashMap<String, StepState>,
    /// When the task was created.
    pub created_at: DateTime<Utc>,
    /// When the task completed (if it has).
    pub completed_at: Option<DateTime<Utc>>,
    /// Current phase.
    pub phase: TaskPhase,
    /// How many times the orchestrator has invoked the
    /// replan-on-failure loop for this task. Capped to bound LLM cost.
    #[serde(default)]
    pub replan_attempts: u32,
}

impl TaskState {
    /// Create a new task in the Planning phase.
    pub fn new(id: String, request: String, graph: TaskGraph) -> Self {
        let step_states: HashMap<String, StepState> = graph
            .steps
            .keys()
            .map(|id| (id.clone(), StepState::Pending))
            .collect();

        Self {
            id,
            request,
            graph,
            step_states,
            created_at: Utc::now(),
            completed_at: None,
            phase: TaskPhase::Planning,
            replan_attempts: 0,
        }
    }

    /// Transition a step to a new state.
    pub fn set_step_state(&mut self, step_id: &str, state: StepState) {
        self.step_states.insert(step_id.to_string(), state);
    }

    /// Check whether all steps are in a terminal state.
    pub fn is_complete(&self) -> bool {
        self.step_states.values().all(|s| s.is_terminal())
    }

    /// Check whether all steps succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.step_states.values().all(|s| s.is_success())
    }

    /// Count steps by state category.
    pub fn counts(&self) -> TaskCounts {
        let mut c = TaskCounts::default();
        for state in self.step_states.values() {
            match state {
                StepState::Pending => c.pending += 1,
                StepState::Ready => c.ready += 1,
                StepState::Running { .. } => c.running += 1,
                StepState::AwaitingConfirmation { .. } => c.awaiting += 1,
                StepState::Completed { .. } => c.completed += 1,
                StepState::Failed { .. } => c.failed += 1,
                StepState::Skipped { .. } => c.skipped += 1,
                StepState::Cancelled => c.cancelled += 1,
            }
        }
        c
    }
}

/// Step counts by state.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TaskCounts {
    pub pending: usize,
    pub ready: usize,
    pub running: usize,
    pub awaiting: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub cancelled: usize,
}

impl TaskCounts {
    pub fn total(&self) -> usize {
        self.pending
            + self.ready
            + self.running
            + self.awaiting
            + self.completed
            + self.failed
            + self.skipped
            + self.cancelled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TaskGraph;
    use crate::step::{StepAction, TaskStep};

    fn simple_graph() -> TaskGraph {
        let steps = vec![
            TaskStep {
                id: "s1".to_string(),
                description: "Step 1".to_string(),
                action: StepAction::Plan {
                    output: "plan".to_string(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Execute,
                estimated_tokens: 0,
            },
            TaskStep {
                id: "s2".to_string(),
                description: "Step 2".to_string(),
                action: StepAction::Test {
                    command: "cargo test".to_string(),
                    workdir: "/tmp".into(),
                },
                depends_on: vec!["s1".to_string()],
                tier: audit::ActionTier::Execute,
                estimated_tokens: 0,
            },
        ];
        TaskGraph::from_steps(steps).unwrap()
    }

    #[test]
    fn test_new_task_state() {
        let graph = simple_graph();
        let state = TaskState::new("t1".to_string(), "build it".to_string(), graph);
        assert_eq!(state.phase, TaskPhase::Planning);
        assert!(!state.is_complete());
        let counts = state.counts();
        assert_eq!(counts.pending, 2);
        assert_eq!(counts.total(), 2);
    }

    #[test]
    fn test_step_transitions() {
        let graph = simple_graph();
        let mut state = TaskState::new("t1".to_string(), "build it".to_string(), graph);

        state.set_step_state(
            "s1",
            StepState::Completed {
                outcome: StepOutcome {
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: Some(0),
                    artifacts: vec![],
                    summary: "done".to_string(),
                },
                completed_at: Utc::now(),
            },
        );
        assert!(!state.is_complete());

        state.set_step_state(
            "s2",
            StepState::Completed {
                outcome: StepOutcome {
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: Some(0),
                    artifacts: vec![],
                    summary: "done".to_string(),
                },
                completed_at: Utc::now(),
            },
        );
        assert!(state.is_complete());
        assert!(state.all_succeeded());
    }
}
