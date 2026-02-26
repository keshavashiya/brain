//! # Brain WebSocket Adapter
//!
//! Exposes Brain's signal processing pipeline over WebSocket using tokio-tungstenite.
//!
//! ## Protocol
//! - Client sends a JSON text frame: `{"source":"ws","content":"...","sender":"user-1"}`
//! - Server responds with a `SignalResponse` JSON text frame
//!
//! ## Connection lifecycle
//! 1. Client connects → assigned a UUID session ID
//! 2. Client sends Signal JSON frames
//! 3. Server processes each frame and replies with a `SignalResponse` JSON frame
//! 4. On disconnect (Close frame or TCP error), connection is removed gracefully

use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
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

/// JSON frame sent by a WebSocket client.
///
/// The `source`, `sender`, and `metadata` fields are optional — the adapter
/// fills in sensible defaults.
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
/// Accepts concurrent connections. Each connection is handled in its own
/// tokio task. Blocks until the listener errors.
pub async fn serve(
    processor: signal::SignalProcessor,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
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

        // Register connection before spawning so the count is accurate
        conns.lock().await.insert(
            conn_id,
            ConnectionInfo {
                id: conn_id,
                peer,
            },
        );

        tokio::spawn(async move {
            match accept_async(tcp_stream).await {
                Ok(ws_stream) => {
                    tracing::info!(
                        conn_id = %conn_id,
                        peer = %peer,
                        "WebSocket connection established"
                    );
                    handle_connection(ws_stream, conn_id, proc).await;
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
/// Reads text frames, converts them to `Signal`s, processes them through the
/// `SignalProcessor`, and sends `SignalResponse` JSON frames back. Non-text
/// frames (ping, binary, close) are handled gracefully without panicking.
async fn handle_connection(
    ws_stream: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    conn_id: Uuid,
    processor: Arc<signal::SignalProcessor>,
) {
    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    while let Some(result) = ws_rx.next().await {
        let msg = match result {
            Ok(m) => m,
            Err(e) => {
                // Network-level error — close gracefully
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
                // Respond to ping to keep connection alive
                let _ = ws_tx.send(Message::Pong(data)).await;
            }
            Message::Close(_) => {
                tracing::debug!(conn_id = %conn_id, "Client sent Close frame");
                break;
            }
            _ => {
                // Binary, pong, and other frames are silently ignored
            }
        }
    }
}

/// Parse a text frame and run it through the signal pipeline.
///
/// Returns an error `SignalResponse` if parsing fails so the client
/// can see the error without the server panicking.
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
