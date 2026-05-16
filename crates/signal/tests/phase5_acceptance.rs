//! v1.0.0 Phase 5 acceptance test (PR-5g).
//!
//! Demonstrates the cardinal rule end-to-end: a reflex firing produces
//! a Signal that travels the same identity → confirmation → dispatch
//! pipeline as user-typed input. The standing-approval store bypasses
//! the prompt when the (agent, verb) is pre-granted; without that
//! grant, the gate blocks until timeout.
//!
//! Fixture choice: `/mcp-mount foo stdio bar` classifies to
//! `Intent::MountMcpServer`, which `intent_to_auth` maps to
//! `mcp.mount @ Tier::External`. External `requires_confirmation`, so
//! the gate fires. No mcp_host is wired so the dispatch returns a
//! placeholder string after the gate — that's exactly what we want:
//! the test asserts the gate's behavior, not the handler's.

use std::sync::Arc;

use brain_core::BrainConfig;
use brainos_signal::{
    reflex_runner, ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource,
};
use confirm::{GrantKey, SqliteConfirmationEngine, SqliteStandingApprovals, StandingApprovalStore};
use identity::{AgentId, Principal, Tier, UserId};
use reflex::{NoopReflex, ReflexSource};

const AGENT: &str = "reflex-agent";

async fn make_processor() -> (Arc<SignalProcessor>, Arc<dyn StandingApprovalStore>) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    let pool = processor.episodic().pool().clone();

    let store_concrete = SqliteStandingApprovals::new(pool.clone());
    let store: Arc<dyn StandingApprovalStore> = Arc::new(store_concrete);

    let engine = SqliteConfirmationEngine::new(pool.clone()).with_standing_approvals(store.clone());
    engine.ensure_tables().unwrap();
    let engine: Arc<dyn confirm::ConfirmationEngine> = Arc::new(engine);

    let processor = processor
        .with_confirmation_engine(engine)
        .with_standing_approvals(store.clone())
        // 200ms cap so the no-bypass test isn't 60s
        .with_confirmation_timeout(std::time::Duration::from_millis(200));

    std::mem::forget(temp);
    (Arc::new(processor), store)
}

fn principal_for(agent: &str) -> Principal {
    Principal {
        user_id: UserId("test-user".into()),
        agent_id: AgentId(agent.into()),
        scopes: vec!["*".into()],
        tier: Tier::External,
    }
}

fn text(resp: SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        ResponseContent::Error(t) => t,
        other => panic!("expected text/error, got {other:?}"),
    }
}

fn signal_from_reflex(_event: reflex::ReflexEvent) -> Signal {
    // Mount a (non-existent) MCP server. The slash classifies into
    // Intent::MountMcpServer (Tier::External) — exactly the path that
    // exercises the confirmation gate. The mcp_host isn't wired in
    // this test, so once the gate passes the handler returns a
    // "not configured" placeholder. That's the success signal: the
    // gate let it through.
    let mut s = Signal::new(
        SignalSource::Cli,
        "reflex",
        AGENT,
        "/mcp-mount foo stdio dummy",
    );
    s.principal = Some(principal_for(AGENT));
    s
}

#[tokio::test]
async fn reflex_with_standing_approval_bypasses_confirmation_gate() {
    let (processor, store) = make_processor().await;
    store
        .grant(
            &GrantKey::new(AGENT, "mcp", "mount"),
            Some("phase5 acceptance"),
        )
        .await
        .unwrap();

    let source: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("acceptance", "phase5:smoke"));

    // Drive the reflex synchronously: subscribe + read one event, then
    // process the corresponding Signal directly so we can capture the
    // response. `spawn_reflex` does the same loop in production; the
    // synchronous form here gives us the response value for assertion.
    let mut stream = source.subscribe().await.unwrap();
    use futures::StreamExt;
    let event = stream.next().await.expect("noop event");
    let signal = signal_from_reflex(event);

    let started = std::time::Instant::now();
    let resp = processor.process(signal).await.unwrap();
    let elapsed = started.elapsed();

    // The gate must have bypassed (no 200ms timeout) and dispatch
    // reached the handler placeholder.
    assert!(
        elapsed < std::time::Duration::from_millis(150),
        "with bypass, process should return promptly — took {elapsed:?}"
    );
    let body = text(resp);
    assert!(
        !body.contains("timed out") && !body.contains("Approval rejected"),
        "bypass should suppress the gate's rejection paths; got: {body}"
    );
}

#[tokio::test]
async fn reflex_without_standing_approval_times_out_at_gate() {
    let (processor, _store) = make_processor().await;
    // Deliberately no grant for (AGENT, mcp, mount).

    let source: Arc<dyn ReflexSource> =
        Arc::new(NoopReflex::simple("acceptance", "phase5:nogrant"));
    let mut stream = source.subscribe().await.unwrap();
    use futures::StreamExt;
    let event = stream.next().await.expect("noop event");
    let signal = signal_from_reflex(event);

    let resp = processor.process(signal).await.unwrap();
    let body = text(resp);
    assert!(
        body.contains("timed out"),
        "without a bypass, the 200ms confirmation gate must time out; got: {body}"
    );
    assert!(
        body.contains("mcp.mount"),
        "timeout message should name the verb that needed approval; got: {body}"
    );
}

#[tokio::test]
async fn spawn_reflex_drives_pipeline_end_to_end() {
    let (processor, store) = make_processor().await;
    store
        .grant(&GrantKey::new(AGENT, "mcp", "mount"), None)
        .await
        .unwrap();

    let source: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("spawn", "phase5:spawn"));
    let handle = reflex_runner::spawn_reflex("acceptance", source, processor, signal_from_reflex)
        .await
        .expect("spawn");

    // NoopReflex ends after one event; the runner exits when the
    // stream completes. Bounded wait — the test is verifying the
    // task terminates cleanly, not the pipeline content (the
    // synchronous-form tests above already cover that).
    let res = tokio::time::timeout(std::time::Duration::from_secs(2), handle).await;
    assert!(res.is_ok(), "runner should exit after stream ends");
}
