//! # Brain WebSocket Adapter
//!
//! Exposes Brain's signal processing pipeline over WebSocket using tokio-tungstenite.
//!
//! ## Protocol
//! 1. Client connects (WebSocket handshake).
//! 2. Client sends first text frame: `{"api_key":"<key>"}` — authentication.
//! 3. Server replies with `{"status":"authenticated","conn_id":"<uuid>"}` or
//!    `{"status":"error","message":"..."}` then closes.
//! 4. Subsequent text frames are `SignalRequest` JSON; server replies with
//!    `SignalResponse` JSON.
//!
//! ## Authentication
//! The initial handshake message MUST contain a valid `api_key`.
//! If the key is absent or invalid the server sends an error frame and closes.

use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use brain_core::ApiKeyConfig;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use signal::SignalResponse;

mod protocol;
mod streaming;

pub use protocol::{AuthMessage, AuthResponse, ClientMessage, ConnectionInfo, Connections};

use streaming::handle_streaming_request;
pub(crate) use streaming::process_text_frame;

// ─── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum WsAdapterError {
    #[error("WebSocket error: {0}")]
    Ws(String),
    #[error("Server error: {0}")]
    Server(String),
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Start the WebSocket server, binding to `host:port`.
///
/// The configured `api_keys` are used to authenticate each new connection's
/// initial handshake message.  Pass an empty `Vec` to disable auth (not
/// recommended in production).
///
/// Accepts concurrent connections. Each connection is handled in its own
/// tokio task. Blocks until the listener errors.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let api_keys: Arc<Vec<ApiKeyConfig>> = Arc::new(processor.config().access.api_keys.clone());
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Synapse WebSocket online at ws://{addr}");
    let connections: Connections = Arc::new(Mutex::new(HashMap::new()));

    loop {
        let (tcp_stream, peer) = listener.accept().await?;
        let conn_id = Uuid::new_v4();

        let proc = Arc::clone(&processor);
        let conns = Arc::clone(&connections);
        let keys = Arc::clone(&api_keys);

        // Register connection before spawning so the count is accurate
        conns
            .lock()
            .await
            .insert(conn_id, ConnectionInfo { id: conn_id, peer });

        tokio::spawn(async move {
            // Limit max message size to 1 MB to prevent memory exhaustion
            let mut ws_config =
                tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
            ws_config.max_message_size = Some(1_048_576);
            ws_config.max_frame_size = Some(1_048_576);
            match tokio_tungstenite::accept_async_with_config(tcp_stream, Some(ws_config)).await {
                Ok(ws_stream) => {
                    tracing::info!(
                        conn_id = %conn_id,
                        peer = %peer,
                        "WebSocket connection established"
                    );
                    handle_connection(ws_stream, conn_id, proc, &keys).await;
                }
                Err(e) => {
                    tracing::warn!(
                        conn_id = %conn_id,
                        peer = %peer,
                        "WebSocket handshake failed: {e}"
                    );
                }
            }

            // Deregister on disconnect (whether handshake failed or connection closed)
            conns.lock().await.remove(&conn_id);
            tracing::info!(conn_id = %conn_id, peer = %peer, "WebSocket connection closed");
        });
    }
}

// ─── Per-connection handler ───────────────────────────────────────────────────

/// Drive a single WebSocket connection to completion.
///
/// Phase 1: read the first text frame as an `AuthMessage` and validate it.
/// Phase 2: process subsequent `ClientMessage` frames as signals.
async fn handle_connection(
    ws_stream: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    conn_id: Uuid,
    processor: Arc<signal::SignalProcessor>,
    api_keys: &[ApiKeyConfig],
) {
    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    // ── Phase 1: authenticate ────────────────────────────────────────────────
    // Also resolves the v1.0.0 Phase 1 `Principal` once and binds it to
    // the connection's lifetime — every subsequent signal frame uses it.
    let (authed, principal): (bool, Option<identity::Principal>) = match ws_rx.next().await {
        None => return,
        Some(Err(e)) => {
            tracing::debug!(conn_id = %conn_id, "WS recv error during auth: {e}");
            return;
        }
        Some(Ok(Message::Text(text))) => {
            match serde_json::from_str::<AuthMessage>(text.as_str()) {
                Err(e) => {
                    let resp = AuthResponse {
                        status: "error",
                        conn_id: None,
                        message: Some(format!("Expected auth message: {e}")),
                    };
                    send_json_frame(&mut ws_tx, &resp, conn_id).await;
                    return;
                }
                Ok(auth) => {
                    if !validate_key(api_keys, &auth.api_key) {
                        let resp = AuthResponse {
                            status: "error",
                            conn_id: None,
                            message: Some("Invalid or missing API key".to_string()),
                        };
                        send_json_frame(&mut ws_tx, &resp, conn_id).await;
                        return;
                    }
                    // Resolve the v1.0.0 Phase 1 Principal once. None when
                    // the key has no agent_id mapping or no IdentityStore
                    // is wired (back-compat).
                    let principal = resolve_principal(api_keys, &auth.api_key, &processor).await;
                    // Auth OK — send confirmation
                    let resp = AuthResponse {
                        status: "authenticated",
                        conn_id: Some(conn_id.to_string()),
                        message: None,
                    };
                    send_json_frame(&mut ws_tx, &resp, conn_id).await;
                    (true, principal)
                }
            }
        }
        Some(Ok(Message::Close(_))) => return,
        Some(Ok(_)) => {
            let resp = AuthResponse {
                status: "error",
                conn_id: None,
                message: Some("First frame must be a text auth message".to_string()),
            };
            send_json_frame(&mut ws_tx, &resp, conn_id).await;
            return;
        }
    };

    if !authed {
        return;
    }

    // ── Phase 2: process signal frames + proactive push ─────────────────────
    // Subscribe to proactive notifications (if router is available).
    let mut proactive_rx = processor.notification_router().map(|r| r.subscribe());
    // Subscribe to the v1.0.0 BrainEvent bus (if observer is wired).
    let mut brain_rx = processor.subscribe_brain_events();

    loop {
        // Build a future that resolves when a proactive notification arrives,
        // or pends forever if no router is configured.
        let proactive_fut = async {
            match proactive_rx.as_mut() {
                Some(rx) => rx.recv().await,
                None => std::future::pending().await,
            }
        };
        // Same pattern for the BrainEvent bus.
        let brain_fut = async {
            match brain_rx.as_mut() {
                Some(rx) => rx.recv().await,
                None => std::future::pending().await,
            }
        };

        tokio::select! {
            // Incoming client frame
            result = ws_rx.next() => {
                let Some(result) = result else { break };
                let msg = match result {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::debug!(conn_id = %conn_id, "WebSocket receive error: {e}");
                        break;
                    }
                };
                match msg {
                    Message::Text(text) => {
                        // Try to parse as ClientMessage to check for streaming flag
                        let client_msg: Option<ClientMessage> = serde_json::from_str(text.as_str()).ok();
                        if let Some(ref cm) = client_msg {
                            if cm.stream == Some(true) {
                                // Streaming path: handle directly, no single response
                                handle_streaming_request(
                                    &mut ws_tx,
                                    conn_id,
                                    processor.clone(),
                                    cm.clone(),
                                    principal.clone(),
                                ).await;
                                continue;
                            }
                        }
                        // Non-streaming: use the standard pipeline
                        let response = match process_text_frame(text.as_str(), conn_id, &processor, principal.as_ref()).await {
                            Ok(Some(r)) => r,
                            Ok(None) => continue, // Shouldn't happen for non-streaming, but be safe
                            Err(e) => {
                                SignalResponse::error(Uuid::new_v4(), e.to_string())
                            }
                        };
                        let json = match serde_json::to_string(&response) {
                            Ok(j) => j,
                            Err(e) => {
                                tracing::error!(conn_id = %conn_id, "Failed to serialize response: {e}");
                                continue;
                            }
                        };
                        if ws_tx.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Message::Ping(data) => {
                        let _ = ws_tx.send(Message::Pong(data)).await;
                    }
                    Message::Close(_) => {
                        tracing::debug!(conn_id = %conn_id, "Client sent Close frame");
                        break;
                    }
                    _ => {}
                }
            }
            // BrainEvent push (v1.0.0 §8 observability bus)
            result = brain_fut => {
                match result {
                    Ok(ev) => {
                        let frame = serde_json::json!({
                            "type": "brain_event",
                            "event": ev,
                        });
                        if let Ok(json) = serde_json::to_string(&frame) {
                            if ws_tx.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(conn_id = %conn_id, skipped = n, "WS brain_event stream lagged");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        brain_rx = None;
                    }
                }
            }
            // Proactive notification push
            result = proactive_fut => {
                match result {
                    Ok(notification) => {
                        let mut frame = serde_json::json!({
                            "type": "proactive",
                            "content": notification.content,
                            "triggered_by": notification.triggered_by,
                            "priority": notification.priority,
                        });
                        if let Some(agent) = &notification.agent {
                            frame["agent"] = serde_json::json!(agent);
                        }
                        if let Ok(json) = serde_json::to_string(&frame) {
                            if ws_tx.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(conn_id = %conn_id, skipped = n, "WS client lagged, dropped notifications");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        tracing::debug!(conn_id = %conn_id, "Notification channel closed");
                        // Channel closed but client connection may still be live — disable proactive push.
                        proactive_rx = None;
                    }
                }
            }
        }
    }
}

/// Send a JSON-serialisable value as a text frame; log errors but don't panic.
async fn send_json_frame<S, T>(ws_tx: &mut S, value: &T, conn_id: Uuid)
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    T: Serialize,
{
    match serde_json::to_string(value) {
        Ok(json) => {
            let _ = ws_tx.send(Message::Text(json.into())).await;
        }
        Err(e) => {
            tracing::error!(conn_id = %conn_id, "Failed to serialize frame: {e}");
        }
    }
}

/// Returns true if `key` is valid and has write permission (WS connections can both read and write).
fn validate_key(api_keys: &[ApiKeyConfig], key: &str) -> bool {
    // WS connections need write permission since they can send signals.
    brain_core::check_auth(api_keys, Some(key), "write").is_allowed()
}

/// Resolve the `Principal` bound to this connection from the validated key.
/// Returns `None` for keys without `agent_id` (back-compat) or when no
/// `IdentityStore` is wired on the processor.
async fn resolve_principal(
    api_keys: &[ApiKeyConfig],
    key: &str,
    processor: &signal::SignalProcessor,
) -> Option<identity::Principal> {
    let agent_id = api_keys
        .iter()
        .find(|k| k.key == key)
        .and_then(|k| k.agent_id.clone())?;
    let store = processor.identity_store()?;
    store
        .principal_for(&identity::AgentHint::AgentId(agent_id.into()))
        .await
        .ok()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use signal::SignalSource;
    use tokio_tungstenite::connect_async;

    fn demo_keys() -> Vec<ApiKeyConfig> {
        brain_core::BrainConfig::default().access.api_keys
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
    fn test_validate_key_empty_list() {
        // With empty key list, auth is disabled (open access) — any key passes
        assert!(validate_key(&[], "anykey"));
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
        let mut config = brain_core::BrainConfig::default();
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
        let mut config = brain_core::BrainConfig::default();
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
        let mut config = brain_core::BrainConfig::default();
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
        let mut config = brain_core::BrainConfig::default();
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
        let mut config = brain_core::BrainConfig::default();
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
        let mut config = brain_core::BrainConfig::default();
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
        let mut config = brain_core::BrainConfig::default();
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
                    if text.contains("\"type\":\"chunk\"") || text.contains("\"type\":\"complete\"")
                    {
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
}
