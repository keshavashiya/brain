//! Phase 5 / PR-5f.2 integration: drives `/approval-list` and
//! `/approval-revoke <id>` through the public `SignalProcessor::process`
//! surface so the classifier → authz → pipeline-handler path is
//! exercised end-to-end against a real `SqliteStandingApprovals`.

use std::sync::Arc;

use brain_core::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use confirm::{GrantKey, SqliteStandingApprovals, StandingApprovalStore};

async fn make_processor() -> (SignalProcessor, Arc<dyn StandingApprovalStore>) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    let pool = processor.episodic().pool().clone();
    let store: Arc<dyn StandingApprovalStore> = Arc::new(SqliteStandingApprovals::new(pool));
    let processor = processor.with_standing_approvals(store.clone());
    std::mem::forget(temp);
    (processor, store)
}

fn text(resp: SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn approval_list_slash_renders_active_grants() {
    let (processor, store) = make_processor().await;
    let id = store
        .grant(&GrantKey::new("agent-a", "fs", "write"), Some("nightly"))
        .await
        .unwrap();

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/approval-list",
        ))
        .await
        .unwrap();
    let body = text(resp);
    assert!(
        body.contains(&id),
        "list output must include the grant id, got: {body}"
    );
    assert!(body.contains("agent-a"));
    assert!(body.contains("fs.write"));
    assert!(body.contains("nightly"));
}

#[tokio::test]
async fn approval_list_slash_handles_empty_store() {
    let (processor, _store) = make_processor().await;
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/approval-list",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("No active standing approvals"));
}

#[tokio::test]
async fn approval_revoke_slash_removes_grant() {
    let (processor, store) = make_processor().await;
    let id = store
        .grant(&GrantKey::new("agent-a", "fs", "write"), None)
        .await
        .unwrap();
    assert!(store
        .is_granted(&GrantKey::new("agent-a", "fs", "write"))
        .await
        .unwrap());

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            format!("/approval-revoke {id}"),
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("Revoked"));
    assert!(!store
        .is_granted(&GrantKey::new("agent-a", "fs", "write"))
        .await
        .unwrap());
}

#[tokio::test]
async fn approval_revoke_slash_reports_unknown_id_friendly() {
    let (processor, _store) = make_processor().await;
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/approval-revoke does-not-exist",
        ))
        .await
        .unwrap();
    let body = text(resp);
    assert!(body.contains("not found") || body.contains("already revoked"));
}

#[tokio::test]
async fn slash_handlers_report_not_wired_when_store_missing() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    std::mem::forget(temp);

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/approval-list",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("not wired"));

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/approval-revoke any-id",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("not wired"));
}
