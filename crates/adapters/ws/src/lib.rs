//! # Brain WebSocket Adapter
//!
//! Exposes Brain's signal processing pipeline over WebSocket using tokio-tungstenite.
//!
//! ## Protocol
//! 1. Client connects (WebSocket handshake).
//! 2. Client sends first text frame: `{"api_key":"demo-key-123"}` — authentication.
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
use tokio::sync::Mutex;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use uuid::Uuid;

use signal::{Signal, SignalResponse, SignalSource};

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
#[derive(Debug, Deserialize)]
pub struct ClientMessage {
    /// Signal source (default: `"ws"`).
    pub source: Option<String>,
    /// Message text / command.
    pub content: String,
    /// Sender identifier (default: `"ws-client"`).
    pub sender: Option<String>,
    /// Optional key-value metadata to attach to the signal.
    pub metadata: Option<HashMap<String, String>>,
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
    processor: signal::SignalProcessor,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let api_keys: Arc<Vec<ApiKeyConfig>> = Arc::new(processor.config().access.api_keys.clone());
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Brain WebSocket server listening on ws://{addr}");

    let processor: Arc<signal::SignalProcessor> = Arc::new(processor);
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
            match accept_async(tcp_stream).await {
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

    // ── Phase 2: process signal frames ───────────────────────────────────────
    while let Some(result) = ws_rx.next().await {
        let msg = match result {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(conn_id = %conn_id, "WebSocket receive error: {e}");
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                let response = process_text_frame(text.as_str(), conn_id, &processor).await;
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

/// Returns true if `key` is present in `api_keys` (any permission).
fn validate_key(api_keys: &[ApiKeyConfig], key: &str) -> bool {
    api_keys.iter().any(|k| k.key == key)
}

/// Parse a text frame and run it through the signal pipeline.
async fn process_text_frame(
    text: &str,
    conn_id: Uuid,
    processor: &signal::SignalProcessor,
) -> SignalResponse {
    let client_msg: ClientMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            let fake_id = Uuid::new_v4();
            return SignalResponse::error(fake_id, format!("Invalid JSON: {e}"));
        }
    };

    let source = parse_source(client_msg.source.as_deref());
    let mut signal = Signal::new(
        source,
        format!("ws:{conn_id}"),
        client_msg.sender.unwrap_or_else(|| "ws-client".to_string()),
        client_msg.content,
    );
    if let Some(meta) = client_msg.metadata {
        signal.metadata = meta;
    }

    let signal_id = signal.id;
    match processor.process(signal).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(conn_id = %conn_id, "Signal processing error: {e}");
            SignalResponse::error(signal_id, e.to_string())
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn parse_source(s: Option<&str>) -> SignalSource {
    match s {
        Some("ws") | Some("websocket") | None => SignalSource::WebSocket,
        Some("cli") => SignalSource::Cli,
        Some("http") => SignalSource::Http,
        Some("mcp") => SignalSource::Mcp,
        Some("grpc") => SignalSource::Grpc,
        _ => SignalSource::WebSocket,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_keys() -> Vec<ApiKeyConfig> {
        brain_core::BrainConfig::default().access.api_keys
    }

    #[test]
    fn test_parse_source_defaults_to_websocket() {
        assert_eq!(parse_source(None), SignalSource::WebSocket);
        assert_eq!(parse_source(Some("ws")), SignalSource::WebSocket);
        assert_eq!(parse_source(Some("websocket")), SignalSource::WebSocket);
    }

    #[test]
    fn test_parse_source_all_variants() {
        assert_eq!(parse_source(Some("cli")), SignalSource::Cli);
        assert_eq!(parse_source(Some("http")), SignalSource::Http);
        assert_eq!(parse_source(Some("mcp")), SignalSource::Mcp);
        assert_eq!(parse_source(Some("grpc")), SignalSource::Grpc);
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
        assert!(validate_key(&keys, "demo-key-123"));
    }

    #[test]
    fn test_validate_key_invalid() {
        let keys = demo_keys();
        assert!(!validate_key(&keys, "bad-key"));
        assert!(!validate_key(&keys, ""));
    }

    #[test]
    fn test_validate_key_empty_list() {
        // With empty key list, no key is valid
        assert!(!validate_key(&[], "demo-key-123"));
    }

    #[test]
    fn test_auth_message_deserialize() {
        let json = r#"{"api_key":"demo-key-123"}"#;
        let msg: AuthMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.api_key, "demo-key-123");
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
        assert_eq!(response.status, signal::ResponseStatus::Error);
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
        assert_eq!(response.status, signal::ResponseStatus::Ok);
    }
}
