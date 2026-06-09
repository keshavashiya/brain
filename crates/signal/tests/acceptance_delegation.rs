//! Acceptance suite — single-shot **delegation** end-to-end.
//!
//! `@<agent> <prompt>` (and `delegate to <agent>: <prompt>`) classify to
//! `Intent::DelegateTask`, authorize against `agent/delegate @ Tier::Execute`,
//! and dispatch through the wired `AgentRegistry` to the named delegate. This
//! suite drives that whole path through the public `SignalProcessor::process`
//! surface and asserts:
//!
//! - the registered agent actually receives the prompt body, and
//! - its result summary is reflected back in the signal response.
//!
//! An unregistered target is also exercised — it must fail honestly with the
//! roster, not silently succeed.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use chrono::Utc;
use delegate::{
    AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
    AgentTaskStatus,
};
use identity::{AgentId, Principal, Tier, UserId};

const CALLER: &str = "delegation-acceptance";

/// Records the prompt it was handed, then returns a success result whose
/// summary echoes a recognizable marker.
struct RecordingAgent {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentDelegate for RecordingAgent {
    fn name(&self) -> &str {
        "recorder"
    }
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::default()
    }
    async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        self.seen.lock().unwrap().push(task.description.clone());
        let now = Utc::now();
        Ok(AgentResult {
            task_id: task.id,
            status: AgentTaskStatus::Succeeded,
            summary: "RECORDER-DID-IT".to_string(),
            artifacts: vec![],
            stdout: String::new(),
            stderr: String::new(),
            exit_code: Some(0),
            started_at: now,
            completed_at: now,
        })
    }
}

fn principal() -> Principal {
    Principal {
        user_id: UserId("test-user".into()),
        agent_id: AgentId(CALLER.into()),
        // Tier::Execute clears the agent/delegate gate without a
        // confirmation prompt (only Destructive/External require one).
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

async fn make_processor(seen: Arc<Mutex<Vec<String>>>) -> SignalProcessor {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    // Keep `temp` alive for the processor's lifetime by leaking it — the
    // test process is short-lived and the OS reclaims the tempdir on exit.
    std::mem::forget(temp);

    let mut registry = AgentRegistry::new();
    registry.register(Arc::new(RecordingAgent { seen }));

    SignalProcessor::new(config)
        .await
        .unwrap()
        .with_agent_registry(Arc::new(registry))
}

#[tokio::test]
async fn delegation_reaches_agent_and_returns_summary() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let processor = make_processor(seen.clone()).await;

    let mut signal = Signal::new(
        SignalSource::Cli,
        "user",
        CALLER,
        "@recorder summarize the meeting notes",
    );
    signal.principal = Some(principal());

    let resp = processor.process(signal).await.unwrap();
    let body = text(resp);

    // The agent received the distilled prompt body (not the `@recorder` prefix).
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        1,
        "agent should have been delegated to exactly once"
    );
    assert_eq!(
        seen[0], "summarize the meeting notes",
        "agent must receive the prompt body with the @mention stripped"
    );

    // The response names the agent and carries its result summary.
    assert!(
        body.contains("recorder"),
        "response should name the delegate; got: {body}"
    );
    assert!(
        body.contains("RECORDER-DID-IT"),
        "response should surface the agent's result summary; got: {body}"
    );
}

#[tokio::test]
async fn delegation_to_unknown_agent_fails_honestly() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let processor = make_processor(seen.clone()).await;

    let mut signal = Signal::new(
        SignalSource::Cli,
        "user",
        CALLER,
        "@ghost do something impossible",
    );
    signal.principal = Some(principal());

    let body = text(processor.process(signal).await.unwrap());

    assert!(
        seen.lock().unwrap().is_empty(),
        "no real agent should have been invoked for an unknown target"
    );
    assert!(
        body.contains("ghost"),
        "response should name the missing agent; got: {body}"
    );
    // The honest path lists the real roster instead of pretending success.
    assert!(
        body.contains("recorder"),
        "response should hint the registered roster; got: {body}"
    );
}
