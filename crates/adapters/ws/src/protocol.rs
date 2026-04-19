use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

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
