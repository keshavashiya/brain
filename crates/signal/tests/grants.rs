//! End-to-end acceptance for the unified grants ledger: `/grants` must
//! answer "what can Brain see and do, on whose authority?" in one
//! screen — runtime standing approvals, the shell exec allowlist,
//! file-read roots, API keys (scopes only, never the key material),
//! configured LLM providers, and local-only namespaces, each with its
//! provenance and revoke path.

use std::sync::Arc;

use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use confirm::{GrantKey, SqliteStandingApprovals, StandingApprovalStore};

const RAW_API_KEY: &str = "sk-super-secret-raw-key-material";

async fn make_processor() -> (SignalProcessor, Arc<dyn StandingApprovalStore>) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();

    config.security.exec_allowlist = vec!["ls".into(), "git".into()];
    config.security.allowed_paths = vec!["~/projects".into()];
    config.access.api_keys = vec![brain::config::ApiKeyConfig {
        key: RAW_API_KEY.into(),
        name: "ci-bot".into(),
        permissions: vec!["read".into(), "write".into()],
        agent_id: Some("agent-ci".into()),
    }];
    config.memory.namespaces.insert(
        "private".to_string(),
        brain::NamespaceConfig {
            residency: brain::Residency::LocalOnly,
        },
    );

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
async fn grants_slash_unions_every_authority_surface_in_one_screen() {
    let (processor, store) = make_processor().await;
    let grant_id = store
        .grant(&GrantKey::new("agent-a", "fs", "write"), Some("nightly"))
        .await
        .unwrap();

    let resp = processor
        .process(Signal::new(SignalSource::Cli, "cli", "user", "/grants"))
        .await
        .unwrap();
    let body = text(resp);

    // Runtime grant with id + revoke path.
    assert!(
        body.contains(&grant_id),
        "missing standing-approval id:\n{body}"
    );
    assert!(body.contains("agent-a"), "missing grantee agent:\n{body}");
    assert!(body.contains("fs.write"), "missing granted verb:\n{body}");
    assert!(
        body.contains("/approval-revoke"),
        "missing standing-approval revoke path:\n{body}"
    );

    // Config-declared authority, each tagged with its config provenance.
    assert!(
        body.contains("security.exec_allowlist") && body.contains("`ls`"),
        "missing exec allowlist:\n{body}"
    );
    assert!(
        body.contains("security.allowed_paths") && body.contains("~/projects"),
        "missing file-access roots:\n{body}"
    );
    assert!(
        body.contains("ci-bot") && body.contains("read, write") && body.contains("agent-ci"),
        "missing API key scopes:\n{body}"
    );

    // Key *material* must never be rendered — scopes only.
    assert!(
        !body.contains(RAW_API_KEY),
        "raw API key leaked into the grants screen:\n{body}"
    );

    // Provider chain locality (the residency rail) is part of the answer.
    assert!(
        body.contains("Active chat chain:"),
        "missing chain-locality line:\n{body}"
    );

    // Egress limits round out "what can Brain share".
    assert!(
        body.contains("`private`") && body.contains("never sent"),
        "missing local-only namespace line:\n{body}"
    );
}

#[tokio::test]
async fn grants_screen_renders_empty_states_rather_than_omitting_sections() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    config.security.exec_allowlist.clear();
    let processor = SignalProcessor::new(config).await.unwrap();
    std::mem::forget(temp);

    let resp = processor
        .process(Signal::new(SignalSource::Cli, "cli", "user", "/grants"))
        .await
        .unwrap();
    let body = text(resp);

    // Every authority surface appears even when empty — an absent section
    // would read as "nothing to disclose", which is the wrong default.
    assert!(
        body.contains("Standing approvals"),
        "missing section:\n{body}"
    );
    assert!(body.contains("MCP servers"), "missing section:\n{body}");
    assert!(body.contains("exec_allowlist"), "missing section:\n{body}");
    assert!(body.contains("allowed_paths"), "missing section:\n{body}");
    assert!(body.contains("api_keys"), "missing section:\n{body}");
    assert!(body.contains("llm.providers"), "missing section:\n{body}");
    assert!(
        body.contains("no commands allowlisted"),
        "empty allowlist must say so explicitly:\n{body}"
    );
}
