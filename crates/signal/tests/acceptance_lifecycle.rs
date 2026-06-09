//! Acceptance suite — orchestrated task **happy-path lifecycle**.
//!
//! The companion to `phase6_acceptance` (which exercises mid-step
//! cancellation). Here a planned single-step task runs to completion and
//! the bus must emit the full canonical phase sequence
//!
//!   none → planning → awaiting_approval → executing → reconciling → completed
//!
//! with the task and its step both landing in the success terminal. This
//! pins the orchestrator's state machine end-to-end: every PR that touches
//! planning, execution, or reconciliation must keep this green.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use delegate::{
    AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
    AgentTaskStatus,
};
use observe::{BrainEvent, BroadcastObserver, Observer};
use orchestrate::{
    DecompositionContext, DecompositionError, StepAction, StepState, TaskDecomposer,
    TaskOrchestrator, TaskPhase, TaskStep,
};

/// An agent that completes immediately with a success result — the
/// opposite of `phase6_acceptance`'s `SlowAgent`.
struct FastAgent;

#[async_trait]
impl AgentDelegate for FastAgent {
    fn name(&self) -> &str {
        "fast"
    }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::default()
    }
    async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        let now = Utc::now();
        Ok(AgentResult {
            task_id: task.id,
            status: AgentTaskStatus::Succeeded,
            summary: "did the thing".to_string(),
            artifacts: vec![],
            stdout: "ok".to_string(),
            stderr: String::new(),
            exit_code: Some(0),
            started_at: now,
            completed_at: now,
        })
    }
}

struct OneStepDecomposer;

#[async_trait]
impl TaskDecomposer for OneStepDecomposer {
    async fn decompose(
        &self,
        _request: &str,
        _context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        Ok(vec![TaskStep {
            id: "only".to_string(),
            description: "the one step".to_string(),
            action: StepAction::Implement {
                spec: "lifecycle acceptance".to_string(),
                agent: "fast".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }])
    }
}

#[tokio::test]
async fn task_runs_full_lifecycle_to_completed() {
    // ── Wire orchestrator with observer + fast delegate ────────────────
    let mut registry = AgentRegistry::new();
    registry.register(Arc::new(FastAgent));
    let registry = Arc::new(registry);

    let observer_arc = BroadcastObserver::new();
    let mut rx = observer_arc.subscribe();
    let observer: Arc<dyn Observer> = observer_arc.clone();

    let orchestrator = TaskOrchestrator::new(Arc::new(OneStepDecomposer))
        .with_agents(registry)
        .with_observer(observer);

    // ── plan() drives none → planning → awaiting_approval ──────────────
    let (task_id, plan_text) = orchestrator
        .plan("do the lifecycle task", DecompositionContext::default())
        .await
        .unwrap();
    assert!(
        !plan_text.is_empty(),
        "plan() should return a human-readable plan summary"
    );

    // The freshly planned task is visible to the approval surface.
    let pending = orchestrator.pending_approvals().await;
    assert!(
        pending.contains(&task_id),
        "freshly planned task must appear in pending_approvals, got {pending:?}"
    );

    // ── execute() drives executing → reconciling → completed ───────────
    let summary = tokio::time::timeout(Duration::from_secs(5), orchestrator.execute(&task_id))
        .await
        .expect("execute() should finish promptly for a one-step fast task")
        .unwrap();
    assert!(
        !summary.is_empty(),
        "execute() should return a result summary"
    );

    // ── Terminal state: task Completed, the step Completed ─────────────
    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(
        task.phase,
        TaskPhase::Completed,
        "one-step task with a succeeding agent must land Completed"
    );
    assert!(
        matches!(
            task.step_states.get("only"),
            Some(StepState::Completed { .. })
        ),
        "the single step must land Completed, got {:?}",
        task.step_states.get("only")
    );

    // ── Bus emitted the full canonical happy-path sequence ─────────────
    let mut transitions: Vec<(String, String)> = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if let BrainEvent::TaskStateChange { from, to, .. } = ev {
            transitions.push((from, to));
        }
    }
    let observed: Vec<(&str, &str)> = transitions
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    let expected: Vec<(&str, &str)> = vec![
        ("none", "planning"),
        ("planning", "awaiting_approval"),
        ("awaiting_approval", "executing"),
        ("executing", "reconciling"),
        ("reconciling", "completed"),
    ];
    assert_eq!(
        observed, expected,
        "happy-path lifecycle must emit the full none→…→completed sequence"
    );
}
