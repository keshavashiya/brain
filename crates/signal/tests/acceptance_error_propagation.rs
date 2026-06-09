//! Acceptance suite — **error propagation** through the pipeline.
//!
//! A failure in a downstream handler must reach the user *with fidelity* —
//! the specific backend error, not a swallowed/generic "something went
//! wrong". This suite drives two failure flavors through the public
//! `SignalProcessor::process` surface, both via delegation (a clean,
//! deterministic injection point):
//!
//! - the delegate returns `Err(AgentError)` — the concrete error message
//!   must appear verbatim in the response, and
//! - the delegate returns `Ok` with a `Failed` status — the non-success
//!   status must be surfaced honestly rather than reported as success.
//!
//! The complementary "breaker trips → HumanConfirm" propagation path is
//! covered by `tool_breaker_dispatch`.

use std::sync::Arc;

use async_trait::async_trait;
use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use chrono::Utc;
use delegate::{
    AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
    AgentTaskStatus,
};
use identity::{AgentId, Principal, Tier, UserId};

const CALLER: &str = "error-propagation-acceptance";
const ERR_MARKER: &str = "disk exploded at sector 42";

/// Always returns a concrete, recognizable `AgentError`.
struct FailingAgent;

#[async_trait]
impl AgentDelegate for FailingAgent {
    fn name(&self) -> &str {
        "boom"
    }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::default()
    }
    async fn delegate(&self, _task: AgentTask) -> Result<AgentResult, AgentError> {
        Err(AgentError::Launch(ERR_MARKER.to_string()))
    }
}

/// Returns `Ok` but with a non-success status — the "soft failure" the
/// pipeline must not paper over as success.
struct SoftFailAgent;

#[async_trait]
impl AgentDelegate for SoftFailAgent {
    fn name(&self) -> &str {
        "softfail"
    }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::default()
    }
    async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        let now = Utc::now();
        Ok(AgentResult {
            task_id: task.id,
            status: AgentTaskStatus::Failed,
            summary: "could not complete the task".to_string(),
            artifacts: vec![],
            stdout: String::new(),
            stderr: "boom".to_string(),
            exit_code: Some(1),
            started_at: now,
            completed_at: now,
        })
    }
}

fn principal() -> Principal {
    Principal {
        user_id: UserId("test-user".into()),
        agent_id: AgentId(CALLER.into()),
        scopes: vec!["*".into()],
        tier: Tier::Execute,
    }
}

fn text(resp: SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        ResponseContent::Error(t) => t,
        other => panic!("expected text/error, got {other:?}"),
    }
}

async fn processor_with(agent: Arc<dyn AgentDelegate>) -> SignalProcessor {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    std::mem::forget(temp);

    let mut registry = AgentRegistry::new();
    registry.register(agent);

    SignalProcessor::new(config)
        .await
        .unwrap()
        .with_agent_registry(Arc::new(registry))
}

#[tokio::test]
async fn delegate_error_reaches_user_verbatim() {
    let processor = processor_with(Arc::new(FailingAgent)).await;

    let mut signal = Signal::new(SignalSource::Cli, "user", CALLER, "@boom do the thing");
    signal.principal = Some(principal());

    let body = text(processor.process(signal).await.unwrap());

    assert!(
        body.contains("boom") && body.to_lowercase().contains("fail"),
        "response should report that delegate 'boom' failed; got: {body}"
    );
    assert!(
        body.contains(ERR_MARKER),
        "the concrete backend error must propagate verbatim, not be swallowed; got: {body}"
    );
}

#[tokio::test]
async fn soft_failure_status_is_surfaced_not_masked() {
    let processor = processor_with(Arc::new(SoftFailAgent)).await;

    let mut signal = Signal::new(SignalSource::Cli, "user", CALLER, "@softfail do the thing");
    signal.principal = Some(principal());

    let body = text(processor.process(signal).await.unwrap());

    // The Failed status must be visible — the response must not read as a
    // plain success that hides the non-zero outcome.
    assert!(
        body.contains("Failed"),
        "non-success delegate status must be surfaced in the response; got: {body}"
    );
}
