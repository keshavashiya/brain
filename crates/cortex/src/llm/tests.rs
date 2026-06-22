use super::*;

#[test]
fn test_provider_config_default() {
    let config = ProviderConfig::default();
    assert_eq!(config.provider, "ollama");
    assert_eq!(config.model, "qwen2.5-coder:7b");
}

#[test]
fn test_ollama_provider_creation() {
    let provider = OllamaProvider::new("http://localhost:11434", "llama3:8b", 0.5, 2048)
        .expect("OllamaProvider::new should not fail in test");
    assert_eq!(provider.name(), "ollama");
}

#[test]
fn test_openai_provider_creation() {
    let provider = OpenAiProvider::openai("test-key", "gpt-4").unwrap();
    assert_eq!(provider.name(), "openai");
}

#[test]
fn test_openrouter_provider_creation() {
    let provider = OpenAiProvider::openrouter("test-key", "anthropic/claude-3-opus").unwrap();
    assert_eq!(provider.name(), "openrouter");
}

#[test]
fn test_extract_json_from_response() {
    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Payload {
        value: i32,
    }

    assert_eq!(
        extract_json_from_response::<Payload>(r#"{"value": 42}"#)
            .unwrap()
            .value,
        42
    );
    assert_eq!(
        extract_json_from_response::<Payload>("Here is the result: {\"value\": 7} done")
            .unwrap()
            .value,
        7
    );
    assert!(extract_json_from_response::<Payload>("no json here").is_none());
}

fn user_message(text: &str) -> Vec<Message> {
    vec![Message::user(text)]
}

#[tokio::test]
async fn test_ollama_generate_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                    "message": {"role": "assistant", "content": "Hello from mock!"},
                    "done": true,
                    "prompt_eval_count": 5,
                    "eval_count": 4
                }"#,
        )
        .create_async()
        .await;

    let provider = OllamaProvider::new(&server.url(), "test-model", 0.7, 1024).unwrap();
    let resp = provider.generate(&user_message("hi")).await.unwrap();

    assert_eq!(resp.content, "Hello from mock!");
    assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 5);
    assert_eq!(resp.usage.as_ref().unwrap().completion_tokens, 4);
    mock.assert_async().await;
}

#[tokio::test]
async fn test_ollama_generate_500_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_status(500)
        .with_body("internal server error")
        .create_async()
        .await;

    let provider = OllamaProvider::new(&server.url(), "test-model", 0.7, 1024).unwrap();
    let err = provider.generate(&user_message("hi")).await.unwrap_err();

    assert!(
        matches!(err, LlmError::Api { status: 500, .. }),
        "expected Api(500), got {err:?}"
    );
}

#[tokio::test]
async fn test_ollama_generate_rate_limited_as_api_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_status(429)
        .with_body("rate limited")
        .create_async()
        .await;

    let provider = OllamaProvider::new(&server.url(), "test-model", 0.7, 1024).unwrap();
    let err = provider.generate(&user_message("hi")).await.unwrap_err();

    assert!(
        matches!(err, LlmError::Api { status: 429, .. }),
        "expected Api(429), got {err:?}"
    );
}

#[tokio::test]
async fn test_ollama_generate_malformed_json() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("this is not json")
        .create_async()
        .await;

    let provider = OllamaProvider::new(&server.url(), "test-model", 0.7, 1024).unwrap();
    let err = provider.generate(&user_message("hi")).await.unwrap_err();

    assert!(
        matches!(err, LlmError::Http(_)),
        "expected Http(..), got {err:?}"
    );
}

#[tokio::test]
async fn test_openai_generate_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                    "choices": [{
                        "message": {"role": "assistant", "content": "OpenAI mock response"},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15
                    }
                }"#,
        )
        .create_async()
        .await;

    let provider =
        OpenAiProvider::new(&server.url(), Some("test-key"), "gpt-4", 0.7, Some(1024)).unwrap();
    let resp = provider.generate(&user_message("hi")).await.unwrap();

    assert_eq!(resp.content, "OpenAI mock response");
    assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 15);
    mock.assert_async().await;
}

#[tokio::test]
async fn test_openai_generate_500_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/chat/completions")
        .with_status(500)
        .with_body("service unavailable")
        .create_async()
        .await;

    let provider =
        OpenAiProvider::new(&server.url(), Some("test-key"), "gpt-4", 0.7, Some(1024)).unwrap();
    let err = provider.generate(&user_message("hi")).await.unwrap_err();

    assert!(
        matches!(err, LlmError::Api { status: 500, .. }),
        "expected Api(500), got {err:?}"
    );
}

#[tokio::test]
async fn test_openai_sends_bearer_token() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .match_header("authorization", "Bearer my-secret-key")
        .with_status(200)
        .with_body(
            r#"{
                    "choices": [{
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"
                    }],
                    "usage": null
                }"#,
        )
        .create_async()
        .await;

    let provider = OpenAiProvider::new(
        &server.url(),
        Some("my-secret-key"),
        "gpt-4",
        0.7,
        Some(1024),
    )
    .unwrap();
    provider.generate(&user_message("hi")).await.unwrap();
    mock.assert_async().await;
}

fn sample_tool() -> ToolDef {
    ToolDef {
        name: "web_search".to_string(),
        description: "Search the web".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"],
        }),
    }
}

#[tokio::test]
async fn test_openai_generate_with_tools_sends_tools_and_tool_choice() {
    let mut server = mockito::Server::new_async().await;
    // The request body must advertise the tools array and `tool_choice`
    // when a non-empty tools slice is supplied.
    let mock = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(r#""tools""#.to_string()),
            mockito::Matcher::Regex(r#""tool_choice":"auto""#.to_string()),
            mockito::Matcher::Regex(r#""web_search""#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                    "choices": [{
                        "message": {"role": "assistant", "content": "let me search"},
                        "finish_reason": "stop"
                    }],
                    "usage": null
                }"#,
        )
        .create_async()
        .await;

    let provider = OpenAiProvider::new(&server.url(), Some("k"), "gpt-4", 0.7, Some(1024)).unwrap();
    let resp = provider
        .generate_with_tools(&user_message("what's the weather"), &[sample_tool()])
        .await
        .unwrap();

    assert_eq!(resp.content, "let me search");
    assert!(resp.tool_calls.is_empty());
    mock.assert_async().await;
}

#[tokio::test]
async fn test_openai_generate_with_tools_parses_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": null,
                            "tool_calls": [{
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": "{\"query\": \"rust async\"}"
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": null
                }"#,
        )
        .create_async()
        .await;

    let provider = OpenAiProvider::new(&server.url(), Some("k"), "gpt-4", 0.7, Some(1024)).unwrap();
    let resp = provider
        .generate_with_tools(&user_message("search rust async"), &[sample_tool()])
        .await
        .unwrap();

    assert_eq!(resp.content, "");
    assert_eq!(resp.tool_calls.len(), 1);
    let call = &resp.tool_calls[0];
    assert_eq!(call.id.as_deref(), Some("call_abc"));
    assert_eq!(call.name, "web_search");
    assert_eq!(call.arguments, serde_json::json!({ "query": "rust async" }));
    mock.assert_async().await;
}

#[tokio::test]
async fn test_openai_replays_assistant_tool_calls_and_tool_results() {
    // A multi-round conversation must serialize the assistant tool-call
    // turn (content null, tool_calls with type=function and a string-encoded
    // arguments blob) and the role:"tool" result turn carrying tool_call_id.
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(r#""role":"assistant""#.to_string()),
            mockito::Matcher::Regex(r#""type":"function""#.to_string()),
            // arguments are re-encoded as a JSON string, not an object.
            mockito::Matcher::Regex(r#""arguments":"\{\\"query\\":\\"rust\\"\}""#.to_string()),
            mockito::Matcher::Regex(r#""role":"tool""#.to_string()),
            mockito::Matcher::Regex(r#""tool_call_id":"call_1""#.to_string()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                    "choices": [{
                        "message": {"role": "assistant", "content": "found 3 results"},
                        "finish_reason": "stop"
                    }],
                    "usage": null
                }"#,
        )
        .create_async()
        .await;

    let messages = vec![
        Message::user("search rust"),
        Message::assistant_with_tool_calls(
            "",
            vec![ProposedToolCall {
                id: Some("call_1".to_string()),
                name: "web_search".to_string(),
                arguments: serde_json::json!({ "query": "rust" }),
            }],
        ),
        Message::tool_result("call_1", r#"{"results": 3}"#),
    ];

    let provider = OpenAiProvider::new(&server.url(), Some("k"), "gpt-4", 0.7, Some(1024)).unwrap();
    let resp = provider.generate(&messages).await.unwrap();

    assert_eq!(resp.content, "found 3 results");
    mock.assert_async().await;
}

#[tokio::test]
async fn test_default_generate_with_tools_ignores_tools() {
    // A provider that doesn't override `generate_with_tools` must fall
    // back to plain `generate`, ignoring the advertised tools entirely.
    struct CannedTextLlm;

    #[async_trait::async_trait]
    impl LlmProvider for CannedTextLlm {
        async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
            Ok(Response::text("plain answer", None))
        }
        async fn generate_stream(
            &self,
            _messages: &[Message],
        ) -> Result<
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<ResponseChunk, LlmError>> + Send>>,
            LlmError,
        > {
            unimplemented!()
        }
        async fn health_check(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "canned"
        }
        fn model(&self) -> &str {
            "canned"
        }
        async fn list_models(&self) -> Result<Vec<String>, LlmError> {
            Ok(vec![])
        }
    }

    let provider = CannedTextLlm;
    let resp = provider
        .generate_with_tools(&user_message("anything"), &[sample_tool()])
        .await
        .unwrap();

    assert_eq!(resp.content, "plain answer");
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn known_context_window_recognises_commercial_models() {
    // Claude family — 200K.
    assert_eq!(known_context_window("claude-opus-4-8"), Some(200_000));
    assert_eq!(known_context_window("claude-3-5-sonnet"), Some(200_000));
    assert_eq!(
        known_context_window("anthropic/claude-haiku-4-5"),
        Some(200_000)
    );

    // Gemini — 1M.
    assert_eq!(known_context_window("gemini-2.5-pro"), Some(1_000_000));

    // GPT family — turbo and 4o are 128K, base gpt-4 is 32K, 3.5 is 16K.
    assert_eq!(known_context_window("gpt-4o-mini"), Some(128_000));
    assert_eq!(known_context_window("gpt-4-turbo"), Some(128_000));
    assert_eq!(known_context_window("gpt-4"), Some(32_000));
    assert_eq!(known_context_window("gpt-3.5-turbo"), Some(16_000));

    // Reasoning models.
    assert_eq!(known_context_window("o1-preview"), Some(200_000));
    assert_eq!(known_context_window("o3-mini"), Some(200_000));
}

#[test]
fn known_context_window_recognises_open_models() {
    assert_eq!(known_context_window("deepseek-r1"), Some(128_000));
    assert_eq!(known_context_window("qwen2.5-coder:7b"), Some(128_000));
    assert_eq!(known_context_window("llama3.1:8b"), Some(128_000));
    assert_eq!(known_context_window("llama2:7b"), Some(8_192));
    assert_eq!(known_context_window("mistral-large"), Some(128_000));
    assert_eq!(known_context_window("mistral:7b"), Some(32_000));
    // Size-suffix heuristic for unfamiliar OpenRouter community models.
    assert_eq!(
        known_context_window("openai/gpt-oss-120b:free"),
        Some(131_072)
    );
}

#[test]
fn known_context_window_returns_none_for_unknown() {
    assert_eq!(known_context_window("some-bespoke-tiny-model"), None);
}

#[test]
fn window_rule_match_semantics() {
    // foo AND (bar OR baz), rejected by `nope`.
    let yaml =
        "rules:\n  - contains_all_of: [[foo], [bar, baz]]\n    excludes: [nope]\n    window: 999\n";
    let set: super::WindowRuleSet = serde_yaml::from_str(yaml).unwrap();
    let r = &set.rules[0];
    assert!(r.matches("foo-bar"));
    assert!(r.matches("foo-baz-x"));
    assert!(!r.matches("foo"), "missing bar/baz group");
    assert!(!r.matches("bar"), "missing foo group");
    assert!(!r.matches("foo-bar-nope"), "excluded substring");
}

#[test]
fn user_rules_take_precedence_over_embedded() {
    // Mirror the chain `known_context_window` builds: user rules first,
    // then embedded. A user rule for qwen should beat the embedded 128K.
    let user = serde_yaml::from_str::<super::WindowRuleSet>(
        "rules:\n  - contains_all_of: [[qwen]]\n    window: 42\n",
    )
    .unwrap()
    .rules;
    let lower = "qwen2.5-coder:7b".to_string();
    let hit = user
        .iter()
        .chain(super::EMBEDDED_RULES.iter())
        .find(|r| r.matches(&lower))
        .map(|r| r.window);
    assert_eq!(hit, Some(42));
}

#[test]
fn provider_locality_follows_endpoint_host() {
    let local_ollama = OllamaProvider::new("http://localhost:11434", "m", 0.7, 64).unwrap();
    assert!(local_ollama.is_local());

    // Ollama on a LAN host still takes content off this machine.
    let lan_ollama = OllamaProvider::new("http://192.168.1.20:11434", "m", 0.7, 64).unwrap();
    assert!(!lan_ollama.is_local());

    // llama.cpp / LM Studio style loopback OpenAI-compatible server.
    let local_oai = OpenAiProvider::new("http://127.0.0.1:8080/v1", None, "m", 0.7, None).unwrap();
    assert!(local_oai.is_local());

    let hosted =
        OpenAiProvider::new("https://api.openai.com/v1", Some("k"), "m", 0.7, None).unwrap();
    assert!(!hosted.is_local());
}
