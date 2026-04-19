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
    assert_eq!(provider.name(), "openai");
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
    vec![Message {
        role: Role::User,
        content: text.to_string(),
    }]
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
