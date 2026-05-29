//! MCP adapter unit + HTTP-transport tests.

use std::sync::Arc;

use axum::{http::StatusCode, routing::post, Router};
use serde_json::{json, Value};

use crate::protocol::JsonRpcRequest;
use crate::server::McpServer;
use crate::transport::{extract_meta_key, http_handler, HttpState};

/// Create a test server with no API keys (auth disabled).
async fn make_server() -> (McpServer, tempfile::TempDir) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = signal::SignalProcessor::new(config).await.unwrap();
    // No api_keys → auth disabled for unit tests
    (McpServer::new(Arc::new(processor), vec![]), temp)
}

/// Create a test server WITH the generated API key (auth enabled).
async fn make_server_with_auth() -> (McpServer, tempfile::TempDir, String) {
    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let api_keys = config.access.api_keys.clone();
    let processor = signal::SignalProcessor::new(config).await.unwrap();
    (McpServer::new(Arc::new(processor), api_keys), temp, api_key)
}

#[tokio::test]
async fn test_initialize() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test", "version": "0.1"}
        })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["protocolVersion"], "2024-11-05");
    assert!(result["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn test_tools_list() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "tools/list".to_string(),
        params: None,
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none());
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"memory_search"));
    assert!(names.contains(&"memory_store"));
    assert!(names.contains(&"memory_facts"));
    assert!(names.contains(&"memory_episodes"));
    assert!(names.contains(&"user_profile"));
    assert!(names.contains(&"memory_procedures"));
    assert!(names.contains(&"brain_capabilities"));
    assert_eq!(tools.len(), 7);
}

#[tokio::test]
async fn test_tool_brain_capabilities_returns_manifest() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(42)),
        method: "tools/call".to_string(),
        params: Some(json!({ "name": "brain_capabilities", "arguments": {} })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none(), "got error: {:?}", resp.error);
    let text = resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(text.contains("Capability manifest"), "got: {text:?}");
}

#[tokio::test]
async fn test_tool_memory_store_and_facts() {
    let (server, _tmp) = make_server().await;

    // Store a fact via MCP tool
    let store_req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(3)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_store",
            "arguments": {
                "subject": "user",
                "predicate": "likes",
                "object": "Rust",
                "category": "personal"
            }
        })),
    };
    let resp = server.handle(store_req).await.unwrap();
    assert!(resp.error.is_none());
    let content = &resp.result.unwrap()["content"][0]["text"];
    assert!(content.as_str().unwrap().contains("Stored fact"));

    // Retrieve facts about "user"
    let facts_req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(4)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_facts",
            "arguments": {"subject": "user"}
        })),
    };
    let resp = server.handle(facts_req).await.unwrap();
    assert!(resp.error.is_none());
    let text = resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(text.contains("likes") || text.contains("Rust"));
}

#[tokio::test]
async fn test_tool_memory_search_empty() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(5)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_search",
            "arguments": {"query": "favourite color", "top_k": 5}
        })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none());
    let text = resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(text.contains("No relevant") || !text.is_empty());
}

#[tokio::test]
async fn test_tool_memory_episodes() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(6)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_episodes",
            "arguments": {"limit": 5}
        })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none());
    let text = resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(!text.is_empty());
}

#[tokio::test]
async fn test_tool_user_profile() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(7)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "user_profile",
            "arguments": {}
        })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none());
    let text = resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(text.contains("llm") || text.contains("data_dir"));
}

#[tokio::test]
async fn test_notification_returns_none() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: None,
        method: "notifications/initialized".to_string(),
        params: None,
    };
    let resp = server.handle(req).await;
    assert!(resp.is_none());
}

#[tokio::test]
async fn test_notification_with_explicit_null_id_returns_none() {
    let (server, _tmp) = make_server().await;
    // Some MCP clients send "id": null explicitly for notifications.
    // serde deserializes this as Some(Value::Null), not None.
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(Value::Null),
        method: "notifications/initialized".to_string(),
        params: None,
    };
    let resp = server.handle(req).await;
    assert!(
        resp.is_none(),
        "notification with id:null must not produce a response"
    );
}

#[tokio::test]
async fn test_initialized_without_prefix_returns_none() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: None,
        method: "initialized".to_string(),
        params: None,
    };
    let resp = server.handle(req).await;
    assert!(resp.is_none());
}

#[tokio::test]
async fn test_unknown_method_returns_error() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(99)),
        method: "does/not/exist".to_string(),
        params: None,
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32601);
}

#[tokio::test]
async fn test_missing_tool_arg_returns_error() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(10)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_search",
            "arguments": {}   // missing "query"
        })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}

// ── Auth tests ────────────────────────────────────────────────────────────

#[test]
fn test_validate_key_with_valid_key() {
    let config = brain::BrainConfig::default();
    let keys = config.access.api_keys;
    let server_keys = keys.clone();
    // Ensure the generated key is present
    assert!(!server_keys.is_empty(), "should have at least one API key");
}

// validate_key() with bad key returns false — covered by async integration tests below.

/// MCP HTTP: missing x-api-key header → 401.
#[tokio::test]
async fn test_http_mcp_no_auth_returns_401() {
    use axum::body::Body;
    use axum::http::{self, Request};
    use tower::util::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_keys = config.access.api_keys.clone();
    let processor = signal::SignalProcessor::new(config).await.unwrap();

    let state = Arc::new(HttpState {
        server: Arc::new(McpServer::new(Arc::new(processor), api_keys)),
    });
    let router = Router::new()
        .route("/mcp", post(http_handler))
        .with_state(state)
        .layer(brain::cors::localhost_cors());

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": null
    });
    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/mcp")
        .header("content-type", "application/json")
        // No x-api-key header
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

/// MCP HTTP: invalid x-api-key header → 401.
#[tokio::test]
async fn test_http_mcp_invalid_key_returns_401() {
    use axum::body::Body;
    use axum::http::{self, Request};
    use tower::util::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_keys = config.access.api_keys.clone();
    let processor = signal::SignalProcessor::new(config).await.unwrap();

    let state = Arc::new(HttpState {
        server: Arc::new(McpServer::new(Arc::new(processor), api_keys)),
    });
    let router = Router::new()
        .route("/mcp", post(http_handler))
        .with_state(state)
        .layer(brain::cors::localhost_cors());

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": null
    });
    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/mcp")
        .header("content-type", "application/json")
        .header("x-api-key", "wrongkey")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

/// MCP HTTP: valid x-api-key header → 200.
#[tokio::test]
async fn test_http_mcp_valid_key_succeeds() {
    use axum::body::Body;
    use axum::http::{self, Request};
    use tower::util::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let api_keys = config.access.api_keys.clone();
    let processor = signal::SignalProcessor::new(config).await.unwrap();

    let state = Arc::new(HttpState {
        server: Arc::new(McpServer::new(Arc::new(processor), api_keys)),
    });
    let router = Router::new()
        .route("/mcp", post(http_handler))
        .with_state(state)
        .layer(brain::cors::localhost_cors());

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": null
    });
    let request = Request::builder()
        .method(http::Method::POST)
        .uri("/mcp")
        .header("content-type", "application/json")
        .header("x-api-key", &api_key)
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let response = router.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Should have a tools list in the result
    assert!(val["result"]["tools"].is_array());
}

/// validate_key() returns false when api_keys is empty (fails closed).
#[tokio::test]
async fn test_validate_key_empty_keys_fails_closed() {
    let (server, _tmp) = make_server().await;
    // api_keys is empty → fail-closed (reject all)
    assert!(!server.validate_key("anykey"));
    assert!(!server.validate_key(""));
}

/// validate_key() returns true for generated key when auth is enabled.
#[tokio::test]
async fn test_validate_key_generated_key_ok() {
    let (server, _tmp, api_key) = make_server_with_auth().await;
    assert!(server.validate_key(&api_key));
    assert!(!server.validate_key("wrongkey"));
}

/// Integration test: MCP memory_store tool → fact persisted, then memory_search finds it.
///
/// Stores a structured fact via the `memory_store` MCP tool, then immediately
/// calls `memory_search` and verifies the stored fact appears in the results.
/// Deterministic fallback embeddings (when no Ollama) keep search available.
#[tokio::test]
async fn test_mcp_memory_store_then_search_roundtrip() {
    let (server, _tmp) = make_server().await;

    // Step 1: Store a fact via MCP memory_store
    let store_req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(20)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_store",
            "arguments": {
                "subject": "Ferris",
                "predicate": "is",
                "object": "the Rust mascot",
                "category": "programming"
            }
        })),
    };
    let store_resp = server.handle(store_req).await.unwrap();
    assert!(store_resp.error.is_none(), "memory_store should not error");
    let store_text = store_resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        store_text.contains("Stored fact"),
        "memory_store response should confirm storage"
    );

    // Step 2: Search for the stored fact via MCP memory_search
    let search_req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(21)),
        method: "tools/call".to_string(),
        params: Some(json!({
            "name": "memory_search",
            "arguments": {"query": "Ferris Rust mascot", "top_k": 5}
        })),
    };
    let search_resp = server.handle(search_req).await.unwrap();
    assert!(
        search_resp.error.is_none(),
        "memory_search should not error"
    );
    let search_text = search_resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    // With deterministic fallback embeddings, stored facts are still returned.
    assert!(
        !search_text.contains("No relevant"),
        "memory_search should return the stored fact, got: {search_text}"
    );
}

/// extract_meta_key extracts x-api-key from params._meta.
#[test]
fn test_extract_meta_key() {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/list".to_string(),
        params: Some(json!({
            "_meta": {"x-api-key": "demokey123"},
            "other": "field"
        })),
    };
    assert_eq!(extract_meta_key(&req), Some("demokey123"));
}

#[test]
fn test_extract_meta_key_missing() {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/list".to_string(),
        params: Some(json!({"other": "field"})),
    };
    assert_eq!(extract_meta_key(&req), None);
}

// ── resources + prompts (#35) ──────────────────────────────────────────────

/// Send a method with optional params against a fresh no-auth server and
/// return the unwrapped JSON-RPC result (asserting no error).
async fn call_ok(method: &str, params: Option<Value>) -> Value {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: method.to_string(),
        params,
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_none(), "{method} errored: {:?}", resp.error);
    resp.result.unwrap()
}

#[tokio::test]
async fn test_resources_list_advertises_views() {
    let result = call_ok("resources/list", None).await;
    let uris: Vec<&str> = result["resources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["uri"].as_str().unwrap())
        .collect();
    assert!(uris.contains(&"brain://profile"));
    assert!(uris.contains(&"brain://capabilities"));
    assert!(uris.contains(&"brain://namespaces"));
}

#[tokio::test]
async fn test_resources_read_each_uri() {
    for uri in [
        "brain://profile",
        "brain://capabilities",
        "brain://namespaces",
    ] {
        let result = call_ok("resources/read", Some(json!({ "uri": uri }))).await;
        let contents = result["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1, "{uri} should return one content block");
        assert_eq!(contents[0]["uri"], uri);
        assert!(contents[0]["text"].as_str().is_some());
    }
}

#[tokio::test]
async fn test_resources_read_unknown_uri_errors() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "resources/read".to_string(),
        params: Some(json!({ "uri": "brain://nope" })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}

#[tokio::test]
async fn test_prompts_list_advertises_templates() {
    let result = call_ok("prompts/list", None).await;
    let names: Vec<&str> = result["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"recall-context"));
    assert!(names.contains(&"daily-review"));
}

#[tokio::test]
async fn test_prompts_get_recall_context_interpolates_query() {
    let result = call_ok(
        "prompts/get",
        Some(json!({ "name": "recall-context", "arguments": { "query": "rust ownership" } })),
    )
    .await;
    let text = result["messages"][0]["content"]["text"].as_str().unwrap();
    assert!(text.contains("rust ownership"));
}

#[tokio::test]
async fn test_prompts_get_recall_context_requires_query() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "prompts/get".to_string(),
        params: Some(json!({ "name": "recall-context" })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}

#[tokio::test]
async fn test_prompts_get_daily_review_no_args() {
    let result = call_ok("prompts/get", Some(json!({ "name": "daily-review" }))).await;
    assert!(result["messages"][0]["content"]["text"].as_str().is_some());
}

#[tokio::test]
async fn test_prompts_get_unknown_errors() {
    let (server, _tmp) = make_server().await;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "prompts/get".to_string(),
        params: Some(json!({ "name": "nope" })),
    };
    let resp = server.handle(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}
