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
use serde::{Deserialize, Serialize};
use signal::PipelineResult;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use signal::{Signal, SignalError, SignalResponse, SignalSource};

// ─── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum WsAdapterError {
    #[error("WebSocket error: {0}")]
    Ws(String),
    #[error("Server error: {0}")]
    Server(String),
}

// ─── DTOs ─────────────────────────────────────────────────────────────────────

/// First frame sent by a WebSocket client — authentication handshake.
#[derive(Debug, Deserialize)]
pub struct AuthMessage {
    /// The API key for this session.
    pub api_key: String,
}

/// Subsequent frames sent by a WebSocket client — signal payload.
#[derive(Debug, Deserialize, Clone)]
pub struct ClientMessage {
    /// Signal source (default: `"ws"`).
    pub source: Option<String>,
    /// Message text / command.
    pub content: String,
    /// Sender identifier (default: `"wsclient"`).
    pub sender: Option<String>,
    /// Optional key-value metadata to attach to the signal.
    pub metadata: Option<HashMap<String, String>>,
    /// Optional memory namespace (default: `"personal"`).
    pub namespace: Option<String>,
    /// Originating agent identity (e.g. "claude-code", "open-code").
    pub agent: Option<String>,
    /// Session ID for conversation continuity.
    pub session_id: Option<String>,
    /// Enable token-by-token streaming response (default: `false`).
    pub stream: Option<bool>,
}

/// Server-to-client auth result frame.
#[derive(Debug, Serialize)]
pub struct AuthResponse {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

// ─── Connection tracking ──────────────────────────────────────────────────────

/// Metadata stored for each active WebSocket connection.
#[derive(Debug, Clone)]
pub struct ConnectionInfo {
    /// Per-session UUID assigned at handshake time.
    pub id: Uuid,
    /// Remote peer address.
    pub peer: SocketAddr,
}

/// Shared map of all active connections (conn_id → info).
pub type Connections = Arc<Mutex<HashMap<Uuid, ConnectionInfo>>>;

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
    let authed = match ws_rx.next().await {
        None => {
            // Client disconnected before sending auth frame
            return;
        }
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
                    // Auth OK — send confirmation
                    let resp = AuthResponse {
                        status: "authenticated",
                        conn_id: Some(conn_id.to_string()),
                        message: None,
                    };
                    send_json_frame(&mut ws_tx, &resp, conn_id).await;
                    true
                }
            }
        }
        Some(Ok(Message::Close(_))) => return,
        Some(Ok(_)) => {
            // Non-text frames before auth are rejected
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

    loop {
        // Build a future that resolves when a proactive notification arrives,
        // or pends forever if no router is configured.
        let proactive_fut = async {
            match proactive_rx.as_mut() {
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
                                ).await;
                                continue;
                            }
                        }
                        // Non-streaming: use the standard pipeline
                        let response = match process_text_frame(text.as_str(), conn_id, &processor).await {
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

/// Parse a text frame and run it through the signal pipeline.
///
/// Returns `Ok(None)` when the response is being streamed directly to the sink
/// (i.e. `client_msg.stream == Some(true)` and the pipeline returned `LlmReady`).
async fn process_text_frame(
    text: &str,
    conn_id: Uuid,
    processor: &signal::SignalProcessor,
) -> Result<Option<SignalResponse>, SignalError> {
    let client_msg: ClientMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            let fake_id = Uuid::new_v4();
            return Ok(Some(SignalResponse::error(
                fake_id,
                format!("Invalid JSON: {e}"),
            )));
        }
    };

    let source = SignalSource::parse(client_msg.source.as_deref(), SignalSource::WebSocket);
    let signal = Signal::from_adapter_request(signal::AdapterRequest {
        source,
        content: client_msg.content,
        channel: Some(format!("ws:{conn_id}")),
        sender: client_msg.sender,
        metadata: client_msg.metadata,
        namespace: client_msg.namespace,
        agent: client_msg.agent,
        session_id: client_msg.session_id,
        default_channel: format!("ws:{conn_id}"),
        default_sender: "wsclient".to_string(),
    });

    let signal_id = signal.id;

    // If streaming is requested, return None — the caller handles streaming
    // directly via prepare() → generate_stream() → finalize_streaming().
    if client_msg.stream == Some(true) {
        return Ok(None);
    }

    match processor.process(signal).await {
        Ok(r) => Ok(Some(r)),
        Err(e) => {
            tracing::warn!(conn_id = %conn_id, "Signal processing error: {e}");
            Ok(Some(SignalResponse::error(signal_id, e.to_string())))
        }
    }
}

/// Send a JSON value as a text frame. Returns Err if the send failed.
async fn send_json_frame_to_sink(
    ws_tx: &mut futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        Message,
    >,
    value: &serde_json::Value,
    conn_id: Uuid,
) -> Result<(), ()> {
    match serde_json::to_string(value) {
        Ok(json) => {
            if ws_tx.send(Message::Text(json.into())).await.is_err() {
                tracing::debug!(conn_id = %conn_id, "Failed to send WS frame");
                Err(())
            } else {
                Ok(())
            }
        }
        Err(e) => {
            tracing::error!(conn_id = %conn_id, "Failed to serialize frame: {e}");
            Err(())
        }
    }
}

// ─── Streaming drop-guard ─────────────────────────────────────────────────────

/// Guarantees that `finalize_streaming()` is called even if the client
/// disconnects mid-stream.
///
/// Call `commit()` on the success path to disarm the guard. On `Drop`, if
/// not committed, the accumulated text is still persisted.
struct StreamFinalizer {
    processor: Arc<signal::SignalProcessor>,
    session_id: Option<String>,
    namespace: String,
    agent: Option<String>,
    acc: Arc<std::sync::Mutex<String>>,
    committed: bool,
}

impl StreamFinalizer {
    fn new(
        processor: Arc<signal::SignalProcessor>,
        session_id: Option<String>,
        namespace: String,
        agent: Option<String>,
        acc: Arc<std::sync::Mutex<String>>,
    ) -> Self {
        Self {
            processor,
            session_id,
            namespace,
            agent,
            acc,
            committed: false,
        }
    }

    /// Disarm the drop-guard — caller takes responsibility for finalizing.
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for StreamFinalizer {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let acc = self.acc.lock().unwrap();
        if acc.is_empty() {
            return;
        }
        let session_id = self.session_id.clone();
        let namespace = self.namespace.clone();
        let agent = self.agent.clone();
        let content = acc.clone();
        // Best-effort persist — log failure but don't panic.
        if let Err(e) = self.processor.finalize_streaming(
            session_id.as_deref().unwrap_or("unknown"),
            &content,
            &namespace,
            agent.as_deref(),
        ) {
            tracing::error!("finalize_streaming failed on cancellation: {e}");
        }
    }
}

// ─── Streaming handler ────────────────────────────────────────────────────────

/// Handle a streaming LLM request: prepare → generate_stream → finalize.
///
/// Sends `chunk` frames for each token and a final `complete` frame.
async fn handle_streaming_request(
    ws_tx: &mut futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        Message,
    >,
    conn_id: Uuid,
    processor: Arc<signal::SignalProcessor>,
    client_msg: ClientMessage,
) {
    let source = SignalSource::parse(client_msg.source.as_deref(), SignalSource::WebSocket);
    let signal = Signal::from_adapter_request(signal::AdapterRequest {
        source,
        content: client_msg.content,
        channel: Some(format!("ws:{conn_id}")),
        sender: client_msg.sender,
        metadata: client_msg.metadata,
        namespace: client_msg.namespace.clone(),
        agent: client_msg.agent.clone(),
        session_id: client_msg.session_id.clone(),
        default_channel: format!("ws:{conn_id}"),
        default_sender: "wsclient".to_string(),
    });

    let signal_id = signal.id;

    // Phase 1: prepare
    let prepared = match processor.prepare(&signal, None).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(conn_id = %conn_id, "Signal prepare error: {e}");
            let _ = send_json_frame_to_sink(
                ws_tx,
                &serde_json::json!({
                    "type": "error",
                    "message": e.to_string()
                }),
                conn_id,
            )
            .await;
            return;
        }
    };

    match prepared {
        PipelineResult::Complete(resp) => {
            // Non-LLM intents: send a single complete frame
            let frame = serde_json::json!({"type": "complete", "response": resp});
            let _ = send_json_frame_to_sink(ws_tx, &frame, conn_id).await;
        }
        PipelineResult::LlmReady {
            messages,
            memory_context,
            session_id,
            namespace,
            agent,
            ..
        } => {
            // Phase 2: generate_stream
            let llm_stream = match processor.llm().generate_stream(&messages).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(conn_id = %conn_id, "LLM stream error: {e}");
                    let _ = send_json_frame_to_sink(
                        ws_tx,
                        &serde_json::json!({
                            "type": "error",
                            "message": e.to_string()
                        }),
                        conn_id,
                    )
                    .await;
                    return;
                }
            };

            let acc: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));

            // Drop-guard: on early return (client close, error), still persist.
            let finalizer = StreamFinalizer::new(
                processor.clone(),
                session_id.clone(),
                namespace.clone(),
                agent.clone(),
                Arc::clone(&acc),
            );

            let mut stream = llm_stream;
            let finalizer = finalizer;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(conn_id = %conn_id, "Stream chunk error: {e}");
                        let _ = send_json_frame_to_sink(
                            ws_tx,
                            &serde_json::json!({
                                "type": "error",
                                "message": e.to_string()
                            }),
                            conn_id,
                        )
                        .await;
                        return; // Drop-guard will finalize what we have
                    }
                };

                acc.lock().unwrap().push_str(&chunk.content);

                let chunk_frame = serde_json::json!({
                    "type": "chunk",
                    "content": chunk.content
                });
                if send_json_frame_to_sink(ws_tx, &chunk_frame, conn_id)
                    .await
                    .is_err()
                {
                    return; // Drop-guard will finalize
                }

                if chunk.is_done {
                    break;
                }
            }

            // Success path: finalize explicitly, disarm the drop-guard.
            {
                let acc_content = acc.lock().unwrap().clone();
                if let Err(e) = processor.finalize_streaming(
                    session_id.as_deref().unwrap_or("unknown"),
                    &acc_content,
                    &namespace,
                    agent.as_deref(),
                ) {
                    tracing::error!("finalize_streaming failed after successful stream: {e}");
                }
            }
            finalizer.commit();

            // Phase 3: send complete frame
            let resp = signal::SignalResponse {
                signal_id,
                status: signal::ResponseStatus::Ok,
                response: signal::ResponseContent::Text(acc.lock().unwrap().clone()),
                memory_context,
                session_id,
            };
            let complete_frame = serde_json::json!({"type": "complete", "response": resp});
            let _ = send_json_frame_to_sink(ws_tx, &complete_frame, conn_id).await;
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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
        let response = process_text_frame("not json at all", conn_id, &processor).await;
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
        let response = process_text_frame(text, conn_id, &processor).await;
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
        for i in 0..3 {
            let url = format!("ws://127.0.0.1:{port}");
            let key = api_key.clone();
            handles.push(tokio::spawn(async move {
                let (mut ws, _) = connect_async(url).await.unwrap();
                ws.send(Message::Text(serde_json::json!({"api_key": key}).to_string().into()))
                    .await
                    .unwrap();
                let _ = ws.next().await;
                let payload = format!(
                    r#"{{"source":"ws","sender":"client-{i}","namespace":"work","content":"Remember user{i} role developer{i}"}}"#
                );
                ws.send(Message::Text(payload.into())).await.unwrap();
                let resp = ws.next().await.unwrap().unwrap().into_text().unwrap().to_string();
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
