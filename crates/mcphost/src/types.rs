//! Public types for the MCP host: server configs, tool descriptors, and outcomes.

use std::{collections::BTreeMap, path::PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Transport-specific configuration for a mounted MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum ServerConfig {
    /// Local child process speaking MCP JSON-RPC on stdin/stdout.
    Stdio {
        command: String,
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
    },

    /// MCP spec 2025-11-25 Streamable HTTP transport.
    StreamableHttp {
        url: String,
        #[serde(default)]
        oauth: Option<OAuthConfig>,
    },

    /// Legacy HTTP+SSE transport. Still spec-required for compatibility.
    HttpSse {
        url: String,
        #[serde(default)]
        oauth: Option<OAuthConfig>,
    },
}

/// OAuth 2.1 + PKCE configuration for HTTP transports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthConfig {
    /// Resource indicator per RFC 8707; reject tokens with mismatched `aud`.
    pub resource: String,
    /// Optional pre-registered client id (else fall back to DCR / Client ID
    /// Metadata Document per MCP spec 2025-11-25).
    #[serde(default)]
    pub client_id: Option<String>,
    /// Optional explicit authorization-server URL (else discovered via PRM).
    #[serde(default)]
    pub authorization_server: Option<String>,
}

/// Server metadata returned from the MCP `initialize` handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub protocol_version: String,
}

/// A tool exposed by a mounted server. Mirrors the MCP `Tool` shape plus the
/// originating server name for routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub server: String,
    pub name: String,
    /// **UNTRUSTED.** Attacker-controllable text shipped by the remote
    /// MCP server. The hash-pin layer in [`crate::rmcp_host`] detects
    /// rug-pull *changes* to this field; callers that surface it to
    /// the LLM must additionally route it through
    /// [`intent::sanitization::render_tool_description_for_prompt`]
    /// before inlining.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for `arguments`. Rendered as **untrusted** content
    /// when shown to the model (CVE-2025-54136 / MCPoison mitigation).
    pub input_schema: serde_json::Value,
}

/// Outcome of a `tools/call`. Structured so audit/observer can render it
/// without re-parsing the raw MCP response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallOutcome {
    pub server: String,
    pub tool: String,
    pub is_error: bool,
    pub content: serde_json::Value,
    pub elapsed_ms: u64,
}

/// In-memory record of a mounted server. The transport-bound `MCPClient`
/// is attached when a real transport is configured; the bare record
/// tracks config + handshake data.
#[derive(Debug, Clone)]
pub struct MountedServer {
    pub name: String,
    pub config: ServerConfig,
    pub mounted_at: DateTime<Utc>,
    pub info: Option<ServerInfo>,
    pub tools: Vec<ToolDescriptor>,
}

/// Snapshot for `list_servers` / `Intent::List { resource: McpServers }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStatus {
    pub name: String,
    pub mounted_at: DateTime<Utc>,
    pub tool_count: usize,
    pub info: Option<ServerInfo>,
    /// True when the server's tool catalog changed after mount-time approval
    /// and the change has not been re-approved: its tools are deregistered
    /// from routing and `call` fails closed until re-consent (or unmount).
    #[serde(default)]
    pub quarantined: bool,
}
