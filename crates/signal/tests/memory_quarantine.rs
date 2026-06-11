//! End-to-end acceptance for memory-writer quarantine: a write from an
//! agent nobody vouched for is stored but excluded from recall, shows up
//! in the `/grants` review queue, and `/memory-approve <agent>` releases
//! it (recording a standing `memory.write` approval so future writes
//! land live). Writers the user already vouched for — an API key bound
//! to the agent id, or a `memory.trust.agents` entry — skip quarantine.

use std::sync::Arc;

use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use confirm::{GrantKey, SqliteStandingApprovals, StandingApprovalStore};

const SECRET: &str = "the vault passphrase is ultramarine-zebra-7";

fn base_config() -> (BrainConfig, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    (config, temp)
}

async fn with_standing_approvals(
    config: BrainConfig,
) -> (SignalProcessor, Arc<dyn StandingApprovalStore>) {
    let processor = SignalProcessor::new(config).await.unwrap();
    let pool = processor.episodic().pool().clone();
    let store: Arc<dyn StandingApprovalStore> = Arc::new(SqliteStandingApprovals::new(pool));
    let processor = processor.with_standing_approvals(store.clone());
    (processor, store)
}

fn text(resp: SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected text, got {other:?}"),
    }
}

async fn send(processor: &SignalProcessor, content: &str, agent: Option<&str>) -> String {
    let mut signal = Signal::new(SignalSource::Cli, "cli", "user", content);
    if let Some(a) = agent {
        signal = signal.with_agent(a);
    }
    text(processor.process(signal).await.unwrap())
}

#[tokio::test]
async fn unvouched_writer_is_quarantined_reviewable_and_releasable() {
    let (config, temp) = base_config();
    let (processor, approvals) = with_standing_approvals(config).await;
    std::mem::forget(temp);

    // An unvouched agent stores a fact and chats (the chat turn lands as
    // an episode) — both write paths run through the attestation gate.
    processor
        .store_fact_direct(
            "personal",
            "test",
            "vault passphrase",
            "is",
            SECRET,
            Some("stranger"),
        )
        .await
        .unwrap();
    send(&processor, SECRET, Some("stranger")).await;

    // Quarantined: recall must not surface the content.
    let recalled = send(&processor, "recall vault passphrase", Some("reader")).await;
    assert!(
        !recalled.contains("ultramarine-zebra-7"),
        "quarantined memory leaked into recall:\n{recalled}"
    );

    // Reviewable: /grants names the writer and the held counts.
    let grants = send(&processor, "/grants", None).await;
    assert!(
        grants.contains("stranger") && grants.contains("Unreviewed memory writers"),
        "quarantine must be visible in /grants:\n{grants}"
    );

    // Releasable: approving the writer frees its memories…
    let approved = send(&processor, "/memory-approve stranger", None).await;
    assert!(
        approved.contains("Approved") && approved.contains("stranger"),
        "unexpected approve response:\n{approved}"
    );
    let recalled = send(&processor, "recall vault passphrase", Some("reader")).await;
    assert!(
        recalled.contains("ultramarine-zebra-7"),
        "released memory must surface in recall:\n{recalled}"
    );

    // …records the grant on the standing-approval rail (revocable via
    // the existing /approval-revoke path)…
    assert!(
        approvals
            .is_granted(&GrantKey::new("stranger", "memory", "write"))
            .await
            .unwrap(),
        "approval must land as a standing memory.write grant"
    );

    // …and future writes from the approved writer land live.
    processor
        .store_fact_direct(
            "personal",
            "test",
            "backup phrase",
            "is",
            "tangerine-falcon-9",
            Some("stranger"),
        )
        .await
        .unwrap();
    let recalled = send(&processor, "recall backup phrase", Some("reader")).await;
    assert!(
        recalled.contains("tangerine-falcon-9"),
        "post-approval write must land live:\n{recalled}"
    );
}

#[tokio::test]
async fn vouched_writers_skip_quarantine() {
    // Vouched via API-key binding.
    let (mut config, temp) = base_config();
    config.access.api_keys = vec![brain::config::ApiKeyConfig {
        key: "k".into(),
        name: "ci".into(),
        permissions: vec!["write".into()],
        agent_id: Some("ci-bot".into()),
    }];
    // Vouched via a trust entry (any configured weight counts as vouched).
    config.memory.trust.agents.insert("vetted".to_string(), 0.8);
    let processor = SignalProcessor::new(config).await.unwrap();
    std::mem::forget(temp);

    processor
        .store_fact_direct(
            "personal",
            "test",
            "wifi code",
            "is",
            "kiwi-llama-3",
            Some("ci-bot"),
        )
        .await
        .unwrap();
    processor
        .store_fact_direct(
            "personal",
            "test",
            "door code",
            "is",
            "mango-otter-4",
            Some("vetted"),
        )
        .await
        .unwrap();

    let recalled = send(&processor, "recall wifi code", Some("reader")).await;
    assert!(
        recalled.contains("kiwi-llama-3"),
        "key-bound writer must not be quarantined:\n{recalled}"
    );
    let recalled = send(&processor, "recall door code", Some("reader")).await;
    assert!(
        recalled.contains("mango-otter-4"),
        "trust-entry writer must not be quarantined:\n{recalled}"
    );
}
