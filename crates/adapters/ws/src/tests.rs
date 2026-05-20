//! WS adapter unit + integration tests.

use std::{net::SocketAddr, sync::Arc};

use brain::ApiKeyConfig;
use signal::SignalSource;
use tokio_tungstenite::connect_async;
use uuid::Uuid;

use crate::auth::validate_key;
use crate::protocol::{AuthMessage, AuthResponse, ClientMessage, ConnectionInfo};
use crate::serve;
use crate::streaming::process_text_frame;

fn demo_keys() -> Vec<ApiKeyConfig> {
    brain::BrainConfig::default().access.api_keys
}

fn random_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn test_parse_source_defaults_to_websocket() {
    assert_eq!(
        SignalSource::parse(None, SignalSource::WebSocket),
        SignalSource::WebSocket
    );
    assert_eq!(
        SignalSource::parse(Some("ws"), SignalSource::WebSocket),
        SignalSource::WebSocket
    );
    assert_eq!(
        SignalSource::parse(Some("websocket"), SignalSource::WebSocket),
        SignalSource::WebSocket
    );
}

#[test]
fn test_parse_source_all_variants() {
    assert_eq!(
        SignalSource::parse(Some("cli"), SignalSource::WebSocket),
        SignalSource::Cli
    );
    assert_eq!(
        SignalSource::parse(Some("http"), SignalSource::WebSocket),
        SignalSource::Http
    );
    assert_eq!(
        SignalSource::parse(Some("mcp"), SignalSource::WebSocket),
        SignalSource::Mcp
    );
    assert_eq!(
        SignalSource::parse(Some("grpc"), SignalSource::WebSocket),
        SignalSource::Grpc
    );
}

#[test]
fn test_client_message_deserialize_minimal() {
    let json = r#"{"content":"hello world"}"#;
    let msg: ClientMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.content, "hello world");
    assert!(msg.source.is_none());
    assert!(msg.sender.is_none());
    assert!(msg.metadata.is_none());
}

#[test]
fn test_client_message_deserialize_full() {
    let json = r#"{"source":"ws","content":"Remember coffee","sender":"user-1"}"#;
    let msg: ClientMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.content, "Remember coffee");
    assert_eq!(msg.source.as_deref(), Some("ws"));
    assert_eq!(msg.sender.as_deref(), Some("user-1"));
}

#[test]
fn test_connection_info_clone() {
    use std::net::{IpAddr, Ipv4Addr};
    let info = ConnectionInfo {
        id: Uuid::new_v4(),
        peer: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9001),
    };
    let cloned = info.clone();
    assert_eq!(info.id, cloned.id);
}

#[test]
fn test_validate_key_valid() {
    let keys = demo_keys();
    let api_key = &keys.first().unwrap().key;
    assert!(validate_key(&keys, api_key));
}

#[test]
fn test_validate_key_invalid() {
    let keys = demo_keys();
    assert!(!validate_key(&keys, "bad-key"));
    assert!(!validate_key(&keys, ""));
}

#[test]
fn test_validate_key_empty_list_fails_closed() {
    // With empty key list, auth is fail-closed — all keys rejected
    assert!(!validate_key(&[], "anykey"));
}

#[test]
fn test_auth_message_deserialize() {
    let json = r#"{"api_key":"demokey123"}"#;
    let msg: AuthMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.api_key, "demokey123");
}

#[test]
fn test_auth_response_serializes_ok() {
    let resp = AuthResponse {
        status: "authenticated",
        conn_id: Some("some-uuid".to_string()),
        message: None,
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"status\":\"authenticated\""));
    assert!(json.contains("\"conn_id\""));
    // `message` should be skipped when None
    assert!(!json.contains("message"));
}

#[test]
fn test_auth_response_serializes_error() {
    let resp = AuthResponse {
        status: "error",
        conn_id: None,
        message: Some("Invalid API key".to_string()),
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("\"status\":\"error\""));
    assert!(json.contains("\"message\":\"Invalid API key\""));
    // `conn_id` should be skipped when None
    assert!(!json.contains("conn_id"));
}

/// Integration test: process_text_frame with invalid JSON returns error response.
#[tokio::test]
async fn test_process_text_frame_invalid_json() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = signal::SignalProcessor::new(config).await.unwrap();

    let conn_id = Uuid::new_v4();
    let response = process_text_frame("not json at all", conn_id, &processor, None).await;
    assert!(response.is_ok());
    let resp = response.unwrap().unwrap();
    assert_eq!(resp.status, signal::ResponseStatus::Error);
}

/// Integration test: process_text_frame with a StoreFact signal returns Ok.
#[tokio::test]
async fn test_process_text_frame_store_fact() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = signal::SignalProcessor::new(config).await.unwrap();

    let conn_id = Uuid::new_v4();
    let text = r#"{"source":"ws","content":"Remember that Rust is fast","sender":"user-1"}"#;
    let response = process_text_frame(text, conn_id, &processor, None).await;
    assert!(response.is_ok());
    let resp = response.unwrap().unwrap();
    assert_eq!(resp.status, signal::ResponseStatus::Ok);
}

#[tokio::test]
#[ignore = "Requires local TCP listener permissions in the runtime environment"]
async fn test_ws_server_auth_success_and_failure() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
    let port = random_port();

    let server_task = tokio::spawn(serve(processor.clone(), "127.0.0.1", port));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Valid auth
    let (mut ws_ok, _) = connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    ws_ok
        .send(Message::Text(
            serde_json::json!({"api_key": api_key}).to_string().into(),
        ))
        .await
        .unwrap();
    let auth_ok = ws_ok.next().await.unwrap().unwrap();
    let auth_ok_text = auth_ok.into_text().unwrap().to_string();
    assert!(auth_ok_text.contains("\"status\":\"authenticated\""));
    ws_ok.close(None).await.unwrap();

    // Invalid auth
    let (mut ws_bad, _) = connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    ws_bad
        .send(Message::Text(r#"{"api_key":"wrong"}"#.into()))
        .await
        .unwrap();
    let auth_bad = ws_bad.next().await.unwrap().unwrap();
    let auth_bad_text = auth_bad.into_text().unwrap().to_string();
    assert!(auth_bad_text.contains("\"status\":\"error\""));
    assert!(auth_bad_text.contains("Invalid or missing API key"));

    server_task.abort();
}

#[tokio::test]
#[ignore = "Requires local TCP listener permissions in the runtime environment"]
async fn test_ws_server_multi_client_writes() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
    let port = random_port();

    let server_task = tokio::spawn(serve(processor.clone(), "127.0.0.1", port));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut handles = Vec::new();
    // Use semantically distinct content so embeddings don't trigger the
    // dedup threshold (distance < 0.1). Using different subjects,
    // predicates, and domains ensures independent fact storage.
    let payloads = [
        r#"{"source":"ws","sender":"client-0","namespace":"work","content":"Remember the project uses Rust 2021 edition"}"#,
        r#"{"source":"ws","sender":"client-1","namespace":"work","content":"Remember the deployment target is Kubernetes"}"#,
        r#"{"source":"ws","sender":"client-2","namespace":"work","content":"Remember the CI pipeline takes 5 minutes"}"#,
    ];
    for payload in payloads.iter() {
        let url = format!("ws://127.0.0.1:{port}");
        let key = api_key.clone();
        let payload = (*payload).to_string();
        handles.push(tokio::spawn(async move {
            let (mut ws, _) = connect_async(url).await.unwrap();
            ws.send(Message::Text(
                serde_json::json!({"api_key": key}).to_string().into(),
            ))
            .await
            .unwrap();
            let _ = ws.next().await;
            ws.send(Message::Text(payload.into())).await.unwrap();
            let resp = ws
                .next()
                .await
                .unwrap()
                .unwrap()
                .into_text()
                .unwrap()
                .to_string();
            assert!(resp.contains("\"status\":\"Ok\""));
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all writes landed in the shared memory.
    let facts = processor.list_facts(Some("work"));
    assert!(
        facts.len() >= 3,
        "expected at least 3 facts, got {}",
        facts.len()
    );

    server_task.abort();
}

// ─── Streaming tests ──────────────────────────────────────────────────────

/// Integration test: streaming request with `stream: true` returns chunk
/// frames followed by a complete frame. Uses a StoreFact intent which
/// returns `PipelineResult::Complete` — so it should return a single
/// `complete` frame (no chunks) since it's not an LLM-driven intent.
#[tokio::test]
#[ignore = "Requires local TCP listener permissions in the runtime environment"]
async fn test_streaming_store_fact_returns_complete_frame() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
    let port = random_port();

    let server_task = tokio::spawn(serve(processor.clone(), "127.0.0.1", port));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let (mut ws, _) = connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    // Auth
    ws.send(Message::Text(
        serde_json::json!({"api_key": api_key}).to_string().into(),
    ))
    .await
    .unwrap();
    let _ = ws.next().await; // auth response

    // Send streaming request
    ws.send(Message::Text(
        serde_json::json!({
            "content": "Remember that streaming works",
            "stream": true,
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    // Should receive a single `complete` frame (StoreFact is non-LLM)
    let resp = ws.next().await.unwrap().unwrap();
    let resp_text = resp.into_text().unwrap().to_string();
    assert!(resp_text.contains("\"type\":\"complete\""));
    assert!(resp_text.contains("\"status\":\"Ok\""));

    server_task.abort();
}

/// Integration test: `stream: true` chat request returns chunk frames
/// followed by a complete frame, and the accumulated content matches.
#[tokio::test]
#[ignore = "Requires local TCP listener, running LLM provider, and DB access"]
async fn test_streaming_chat_returns_chunks_then_complete() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
    let port = random_port();

    let server_task = tokio::spawn(serve(processor.clone(), "127.0.0.1", port));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let (mut ws, _) = connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    // Auth
    ws.send(Message::Text(
        serde_json::json!({"api_key": api_key}).to_string().into(),
    ))
    .await
    .unwrap();
    let _ = ws.next().await; // auth response

    // Send streaming chat request
    ws.send(Message::Text(
        serde_json::json!({
            "content": "Say hello briefly",
            "stream": true,
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    let mut chunks = Vec::new();

    loop {
        let resp = ws.next().await.unwrap().unwrap();
        let resp_text = resp.into_text().unwrap().to_string();
        let json: serde_json::Value = serde_json::from_str(&resp_text).unwrap();

        match json.get("type").and_then(|v| v.as_str()) {
            Some("chunk") => {
                if let Some(content) = json.get("content").and_then(|v| v.as_str()) {
                    chunks.push(content.to_string());
                }
            }
            Some("complete") => {
                break;
            }
            Some("proactive") => continue,
            other => panic!("Unexpected frame type: {:?}", other),
        }
    }

    assert!(
        !chunks.is_empty(),
        "Should have received at least one chunk"
    );

    let full_text: String = chunks.join("");
    assert!(
        !full_text.is_empty(),
        "Accumulated content should not be empty"
    );

    server_task.abort();
}

/// Integration test: client disconnect mid-stream → drop-guard fires without
/// panicking. The `StreamFinalizer::drop` should call `finalize_streaming`
/// even on early termination.
#[tokio::test]
#[ignore = "Requires local TCP listener, running LLM provider, and DB access"]
async fn test_streaming_cancellation_drop_guard() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let api_key = config.access.api_keys.first().unwrap().key.clone();
    let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
    let port = random_port();

    let server_task = tokio::spawn(serve(processor.clone(), "127.0.0.1", port));
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let (mut ws, _) = connect_async(format!("ws://127.0.0.1:{port}"))
        .await
        .unwrap();
    // Auth
    ws.send(Message::Text(
        serde_json::json!({"api_key": api_key}).to_string().into(),
    ))
    .await
    .unwrap();
    let _ = ws.next().await; // auth response

    // Send streaming chat request
    ws.send(Message::Text(
        serde_json::json!({
            "content": "Tell me a long story",
            "stream": true,
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    // Read at least one frame, then drop the connection
    let mut _received_any = false;
    for _ in 0..3 {
        if let Some(Ok(resp)) = ws.next().await {
            if let Ok(text) = resp.into_text() {
                if text.contains("\"type\":\"chunk\"") || text.contains("\"type\":\"complete\"") {
                    _received_any = true;
                }
            }
        }
    }

    // Drop the WS connection without completing
    drop(ws);

    // Give the server a moment to process the disconnect and run the drop-guard
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // If we got here without panicking, the drop-guard worked correctly.
    // The server task should still be alive (not panicked).
    assert!(
        !server_task.is_finished(),
        "Server should still be running after client disconnect"
    );

    server_task.abort();
}
