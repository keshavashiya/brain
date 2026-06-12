//! Wire-level acceptance for the namespace data-residency policy.
//!
//! Acceptance: with only a "cloud" provider configured, a fact stored in a
//! `local_only` namespace provably never appears in any outbound request
//! body. The test runs a real HTTP capture server and points a real
//! `OpenAiProvider` at it; a thin wrapper reports `is_local() == false`
//! to simulate a hosted endpoint, so every byte the provider would send
//! to a cloud API is captured and inspected on the wire.

use std::sync::{Arc, Mutex};

use brainos_signal::{Signal, SignalProcessor, SignalSource};
use futures::Stream;
use std::pin::Pin;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const SECRET: &str = "ultramarine-zebra-7";
const PUBLIC_CODE: &str = "tangerine-falcon-9";

/// Minimal HTTP/1.1 capture server: records every request body and
/// answers with a canned OpenAI chat completion. One request per
/// connection (`Connection: close`).
async fn spawn_capture_server(bodies: Arc<Mutex<Vec<String>>>) -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let bodies = bodies.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                // Read until end of headers.
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
                let body = String::from_utf8_lossy(
                    &buf[header_end..header_end + content_length.min(buf.len() - header_end)],
                )
                .to_string();
                bodies.lock().unwrap().push(body);

                let reply = r#"{"id":"cap","object":"chat.completion","model":"mock","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
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

/// Delegates every call to a real `OpenAiProvider` pointed at the local
/// capture server, but reports itself as non-local — the locality a
/// hosted endpoint would have. The wire bytes are real; only the
/// locality flag is simulated.
struct CloudSim(cortex::llm::OpenAiProvider);

#[async_trait::async_trait]
impl cortex::LlmProvider for CloudSim {
    async fn generate(
        &self,
        messages: &[cortex::llm::Message],
    ) -> Result<cortex::llm::Response, cortex::llm::LlmError> {
        self.0.generate(messages).await
    }

    async fn generate_with_tools(
        &self,
        messages: &[cortex::llm::Message],
        tools: &[cortex::llm::ToolDef],
    ) -> Result<cortex::llm::Response, cortex::llm::LlmError> {
        self.0.generate_with_tools(messages, tools).await
    }

    async fn generate_stream(
        &self,
        messages: &[cortex::llm::Message],
    ) -> Result<
        Pin<
            Box<
                dyn Stream<Item = Result<cortex::llm::ResponseChunk, cortex::llm::LlmError>> + Send,
            >,
        >,
        cortex::llm::LlmError,
    > {
        self.0.generate_stream(messages).await
    }

    async fn health_check(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "cloudsim"
    }

    fn model(&self) -> &str {
        "mock"
    }

    async fn list_models(&self) -> Result<Vec<String>, cortex::llm::LlmError> {
        Ok(vec!["mock".into()])
    }

    fn is_local(&self) -> bool {
        false
    }
}

async fn processor_with_cloud_chain(temp_dir: &tempfile::TempDir, port: u16) -> SignalProcessor {
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    // Point the construction-time provider pool at the capture server too:
    // the intent classifier keeps the chain it was built with (with_llm
    // below only swaps the generation tiers), and the default-config
    // entry is live local Ollama on dev machines — whose 0.7-temperature
    // classification of the chat turns is nondeterministic and can land
    // on an action intent that errors without a dispatcher. The capture
    // server's canned non-JSON reply makes the LLM fallback parse-fail
    // deterministically, so classification always degrades to Chat.
    config.llm.providers = vec![brain::ProviderEntry {
        name: "capture".to_string(),
        kind: "openai_compat".to_string(),
        base_url: format!("http://127.0.0.1:{port}/v1"),
        api_key: String::new(),
        api_key_file: None,
        model: "mock".to_string(),
        preferred_models: Vec::new(),
    }];
    config.memory.namespaces.insert(
        "private".to_string(),
        brain::NamespaceConfig {
            residency: brain::Residency::LocalOnly,
        },
    );

    let cloud = CloudSim(
        cortex::llm::OpenAiProvider::new(
            &format!("http://127.0.0.1:{port}/v1"),
            None,
            "mock",
            0.0,
            None,
        )
        .unwrap(),
    );
    SignalProcessor::new(config)
        .await
        .unwrap()
        .with_llm(Arc::new(cloud))
}

/// The headline DoD: a fact + episode living in a `local_only` namespace
/// never reach the wire, even when the conversation that recalls them is
/// itself in that namespace; the same rail demonstrably carries memories
/// from an unrestricted namespace.
#[tokio::test]
async fn local_only_memories_never_reach_a_remote_provider() {
    let temp_dir = tempfile::tempdir().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_capture_server(bodies.clone()).await;
    let processor = processor_with_cloud_chain(&temp_dir, port).await;

    // Seed the local-only namespace without involving any provider:
    // a semantic fact (direct store) and an episode carrying the secret.
    processor
        .store_fact_direct("private", "test", "vault passphrase", "is", SECRET, None)
        .await
        .unwrap();
    let sid = processor.episodic().create_session("test").unwrap();
    processor
        .episodic()
        .store_episode(
            &sid,
            "user",
            &format!("my vault passphrase is {SECRET}"),
            0.9,
            Some("private"),
            None,
        )
        .unwrap();

    // Seed an unrestricted namespace the same way (control).
    processor
        .episodic()
        .store_episode(
            &sid,
            "user",
            &format!("the office wifi code is {PUBLIC_CODE}"),
            0.9,
            Some("personal"),
            None,
        )
        .unwrap();

    // Chat in the local-only namespace: recall finds the secret episode
    // (BM25 over "vault passphrase"), the residency gate must withhold it.
    let resp = processor
        .process(
            Signal::new(SignalSource::Cli, "cli", "user", "vault passphrase")
                .with_namespace("private"),
        )
        .await
        .unwrap();
    assert_eq!(resp.status, brainos_signal::ResponseStatus::Ok);

    // Control chat in the unrestricted namespace: its memory must flow
    // to the wire, proving recall→prompt actually carries memories and
    // the private case above was withheld by policy, not by accident.
    let resp = processor
        .process(
            Signal::new(SignalSource::Cli, "cli", "user", "office wifi code")
                .with_namespace("personal"),
        )
        .await
        .unwrap();
    assert_eq!(resp.status, brainos_signal::ResponseStatus::Ok);

    let captured = bodies.lock().unwrap().clone();
    eprintln!("captured {} request bodies:", captured.len());
    for (i, b) in captured.iter().enumerate() {
        eprintln!("--- body {i}: {b}");
    }
    assert!(
        !captured.is_empty(),
        "the capture server must have seen the chat turns"
    );
    assert!(
        captured.iter().any(|b| b.contains(PUBLIC_CODE)),
        "control: an unrestricted-namespace memory must reach the wire; \
         if it doesn't, recall isn't feeding prompts and this test proves nothing"
    );
    assert!(
        !captured.iter().any(|b| b.contains(SECRET)),
        "a local_only fact appeared in an outbound request body"
    );
}

/// Sub-namespaces inherit the parent's residency policy.
#[tokio::test]
async fn sub_namespaces_inherit_local_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_capture_server(bodies.clone()).await;
    let processor = processor_with_cloud_chain(&temp_dir, port).await;

    let sid = processor.episodic().create_session("test").unwrap();
    processor
        .episodic()
        .store_episode(
            &sid,
            "user",
            &format!("my blood type note {SECRET}"),
            0.9,
            Some("private/health"),
            None,
        )
        .unwrap();

    let resp = processor
        .process(
            Signal::new(SignalSource::Cli, "cli", "user", "blood type note")
                .with_namespace("private/health"),
        )
        .await
        .unwrap();
    assert_eq!(resp.status, brainos_signal::ResponseStatus::Ok);

    let captured = bodies.lock().unwrap().clone();
    assert!(
        !captured.iter().any(|b| b.contains(SECRET)),
        "a local_only sub-namespace memory appeared in an outbound request body"
    );
}
