//! Wire-level acceptance for task-tier routing (`llm.tiers`).
//!
//! DoD: with a local fast-tier provider and a cloud deep-tier provider
//! configured, a classification turn provably never reaches the cloud
//! endpoint. Two real HTTP capture servers stand in for the two
//! providers; real `OpenAiProvider`s talk to them over the wire, so the
//! assertion is on actual outbound request bodies, not on routing
//! intent. A control case with `tiers` unset proves the same turn *does*
//! ride the default (cloud-primary) chain — i.e. the fast case is
//! enforced by the tier config, not by accident.

use std::sync::{Arc, Mutex};

use brainos_signal::{Signal, SignalProcessor, SignalSource};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Distinctive token carried in the classified content; warmup pings and
/// other chatter never contain it, so capture assertions key on it.
const MARKER: &str = "quokka-parade-42";

/// Minimal HTTP/1.1 capture server: answers `GET …/models` with a model
/// list (so startup probing works) and any POST with a canned OpenAI
/// chat completion. Records every POST body.
async fn spawn_capture_server(post_bodies: Arc<Mutex<Vec<String>>>) -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let post_bodies = post_bodies.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let header_end = loop {
                    let Ok(n) = stream.read(&mut tmp).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let is_get = headers.starts_with("GET");
                let content_length: usize = headers
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buf.len() < header_end + content_length {
                    let Ok(n) = stream.read(&mut tmp).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                if !is_get {
                    let body = String::from_utf8_lossy(
                        &buf[header_end..header_end + content_length.min(buf.len() - header_end)],
                    )
                    .to_string();
                    post_bodies.lock().unwrap().push(body);
                }

                let reply = if is_get {
                    r#"{"object":"list","data":[{"id":"mock"}]}"#
                } else {
                    r#"{"id":"cap","object":"chat.completion","model":"mock","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}"#
                };
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    reply.len(),
                    reply
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    port
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn provider_entry(name: &str, port: u16) -> brain::ProviderEntry {
    brain::ProviderEntry {
        name: name.to_string(),
        kind: "openai_compat".to_string(),
        base_url: format!("http://127.0.0.1:{port}/v1"),
        api_key: String::new(),
        api_key_file: None,
        model: "mock".to_string(),
        preferred_models: Vec::new(),
    }
}

/// Two providers — "cloud" first (default-chain primary), "local"
/// second; `fast_to_local` opts the fast tier onto the local one.
async fn processor(
    temp_dir: &tempfile::TempDir,
    cloud_port: u16,
    local_port: u16,
    fast_to_local: bool,
) -> SignalProcessor {
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    config.llm.providers = vec![
        provider_entry("cloud", cloud_port),
        provider_entry("local", local_port),
    ];
    if fast_to_local {
        config.llm.tiers.fast = vec!["local".to_string()];
    }
    SignalProcessor::new(config).await.unwrap()
}

/// Ambiguous content (no slash/explicit/regex match) that forces the
/// classifier onto its LLM fallback — the fast-tier call under test.
fn ambiguous_signal() -> Signal {
    Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        format!("hmm, could you weigh in on that {MARKER} situation from earlier?"),
    )
}

/// The DoD case: with `tiers.fast = ["local"]`, the classification turn
/// reaches only the local endpoint; the cloud endpoint never sees its
/// content. Per-tier spend lands in `BudgetStatus` under `tier:fast`.
#[tokio::test]
async fn classification_rides_the_fast_tier_and_never_reaches_the_cloud() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cloud_bodies = Arc::new(Mutex::new(Vec::new()));
    let local_bodies = Arc::new(Mutex::new(Vec::new()));
    let cloud_port = spawn_capture_server(cloud_bodies.clone()).await;
    let local_port = spawn_capture_server(local_bodies.clone()).await;

    let budget: Arc<dyn budget::CostBudget> = {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let b = budget::SqliteBudget::new(pool, budget::BudgetPolicy::default());
        b.ensure_tables().unwrap();
        Arc::new(b)
    };
    let processor = processor(&temp_dir, cloud_port, local_port, true)
        .await
        .with_cost_budget(budget.clone());

    // prepare() runs classification (the fast-tier LLM fallback) but not
    // the deep-tier chat generation — exactly "a classification turn".
    processor
        .prepare(&ambiguous_signal(), None, None)
        .await
        .unwrap();

    let local = local_bodies.lock().unwrap().clone();
    let cloud = cloud_bodies.lock().unwrap().clone();
    assert!(
        local.iter().any(|b| b.contains(MARKER)),
        "control-within-the-test: the classification call must reach the \
         fast (local) endpoint; if it doesn't, the LLM fallback never fired \
         and this test proves nothing. local POSTs: {local:?}"
    );
    assert!(
        !cloud.iter().any(|b| b.contains(MARKER)),
        "a fast-tier classification turn reached the cloud endpoint: {cloud:?}"
    );

    // Per-tier usage is visible in BudgetStatus under the tier key.
    let status = budget.status().await.unwrap();
    assert!(
        status
            .hourly_consumption
            .get("tier:fast:llm_input_tokens")
            .copied()
            .unwrap_or(0)
            > 0,
        "fast-tier spend must be recorded; consumption: {:?}",
        status.hourly_consumption
    );
}

/// Control: with `tiers` unset the same turn rides the default chain,
/// whose primary is the cloud endpoint — proving the routing in the DoD
/// case comes from the tier config.
#[tokio::test]
async fn without_tiers_classification_rides_the_default_chain() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cloud_bodies = Arc::new(Mutex::new(Vec::new()));
    let local_bodies = Arc::new(Mutex::new(Vec::new()));
    let cloud_port = spawn_capture_server(cloud_bodies.clone()).await;
    let local_port = spawn_capture_server(local_bodies.clone()).await;

    let processor = processor(&temp_dir, cloud_port, local_port, false).await;
    processor
        .prepare(&ambiguous_signal(), None, None)
        .await
        .unwrap();

    let cloud = cloud_bodies.lock().unwrap().clone();
    assert!(
        cloud.iter().any(|b| b.contains(MARKER)),
        "with no tiers configured the classification turn must ride the \
         default (cloud-primary) chain. cloud POSTs: {cloud:?}"
    );
}
