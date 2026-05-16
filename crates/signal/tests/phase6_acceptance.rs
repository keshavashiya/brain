//! v1.0.0 Phase 6 acceptance test (PR-6c).
//!
//! End-to-end: a long-running orchestrated task is cancelled mid-step
//! via `/task-cancel <id>` flowing through [`SignalProcessor::process`].
//! The acceptance criterion (per the v1.0 roadmap):
//!
//! - The in-flight step's future is dropped within one polling cycle
//!   of the cancel — well under the 3600s the SlowAgent would
//!   otherwise have spent in `tokio::time::sleep`.
//! - The bus emits the canonical state transitions ending at
//!   `cancelled`. Cancellation skips the `reconciling` phase by
//!   construction: `TaskOrchestrator::cancel` flips state directly to
//!   `Cancelled`, because there is no "verify the world matches the
//!   plan" step when the plan was forcibly torn down.
//! - The signal response reflects the successful cancel.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use brain_core::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use chrono::Utc;
use delegate::{
    AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
    AgentTaskStatus,
};
use identity::{AgentId, Principal, Tier, UserId};
use observe::{BrainEvent, BroadcastObserver, Observer};
use orchestrate::{
    DecompositionContext, DecompositionError, StepAction, TaskDecomposer, TaskOrchestrator,
    TaskPhase, TaskStep,
};

const AGENT: &str = "phase6-agent";

/// Stub delegate that never returns under normal flow. Dropping the
/// future on cancel is cancel-safe (the sleep just gets dropped).
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

struct OneStepDecomposer;

#[async_trait]
impl TaskDecomposer for OneStepDecomposer {
    async fn decompose(
        &self,
        _request: &str,
        _context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        Ok(vec![TaskStep {
            id: "slow".to_string(),
            description: "long-running step".to_string(),
            action: StepAction::Implement {
                spec: "phase6 acceptance".to_string(),
                agent: "slow".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }])
    }
}

fn principal_for(agent: &str) -> Principal {
    Principal {
        user_id: UserId("test-user".into()),
        agent_id: AgentId(agent.into()),
        scopes: vec!["*".into()],
        tier: Tier::Write,
    }
}

fn text(resp: SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        ResponseContent::Error(t) => t,
        other => panic!("expected text/error, got {other:?}"),
    }
}

#[tokio::test]
async fn pr6c_task_cancel_slash_aborts_mid_step_via_signal_pipeline() {
    // ── Wire orchestrator with observer + slow delegate ────────────────
    let mut registry = AgentRegistry::new();
    registry.register(Arc::new(SlowAgent));
    let registry = Arc::new(registry);

    let observer_arc = BroadcastObserver::new();
    let mut rx = observer_arc.subscribe();
    let observer: Arc<dyn Observer> = observer_arc.clone();

    let orchestrator = Arc::new(
        TaskOrchestrator::new(Arc::new(OneStepDecomposer))
            .with_agents(registry)
            .with_observer(observer),
    );

    // ── Build a SignalProcessor over the same orchestrator ─────────────
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    let processor = processor.with_orchestrator(orchestrator.clone());

    // ── Plant a task and spawn execute() in the background ─────────────
    let (task_id, _plan_text) = orchestrator
        .plan("phase6 long task", DecompositionContext::default())
        .await
        .unwrap();
    let exec_orch = orchestrator.clone();
    let exec_task_id = task_id.clone();
    let exec_handle = tokio::spawn(async move { exec_orch.execute(&exec_task_id).await.unwrap() });

    // Give execute() time to enter the SlowAgent's sleep — otherwise the
    // cancel could land before the step starts, bypassing the mid-flight
    // abort path we're exercising.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ── Drive the cancel through SignalProcessor::process ──────────────
    let mut signal = Signal::new(
        SignalSource::Cli,
        "user",
        AGENT,
        format!("/task-cancel {task_id}"),
    );
    signal.principal = Some(principal_for(AGENT));

    let cancel_at = Instant::now();
    let resp = processor.process(signal).await.unwrap();
    let elapsed_signal = cancel_at.elapsed();

    // SignalProcessor's handle_cancel_task returns the orchestrator's
    // ack synchronously after `cancel()` resolves; cancel() is fast
    // (in-memory state flip + token.cancel()).
    assert!(
        elapsed_signal < Duration::from_millis(500),
        "/task-cancel signal should return promptly; took {elapsed_signal:?}"
    );
    let body = text(resp);
    assert!(
        body.contains("cancelled"),
        "response should confirm cancellation; got: {body}"
    );
    assert!(
        body.contains(&task_id),
        "response should name the cancelled task; got: {body}"
    );

    // ── execute() must abort within one polling cycle of cancel ────────
    let _summary = tokio::time::timeout(Duration::from_secs(2), exec_handle)
        .await
        .expect("execute() must return within 2s of /task-cancel — did PR-6b's token thread through to the pipeline?")
        .expect("execute task panicked");
    let elapsed_total = cancel_at.elapsed();
    assert!(
        elapsed_total < Duration::from_secs(2),
        "execute() returned but took {elapsed_total:?} after cancel — cancellation should be near-instant"
    );

    // ── Task lands cleanly in Cancelled, step too ──────────────────────
    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(
        task.phase,
        TaskPhase::Cancelled,
        "task must land in Cancelled after mid-step /task-cancel"
    );
    assert!(
        matches!(
            task.step_states.get("slow"),
            Some(orchestrate::StepState::Cancelled)
        ),
        "in-flight step must be Cancelled, got {:?}",
        task.step_states.get("slow")
    );

    // ── Bus emitted the canonical Phase-6 transition sequence ──────────
    // Cancellation skips Reconciling: cancel() transitions straight to
    // Cancelled because there's no "verify the world" step when the
    // plan was forcibly torn down.
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
        ("executing", "cancelled"),
    ];
    assert_eq!(
        observed, expected,
        "Phase 6 transition sequence mismatch — cancel must drive executing → cancelled directly"
    );

    drop(temp);
}
