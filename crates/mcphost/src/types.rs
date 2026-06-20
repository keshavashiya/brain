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

/// Per-server egress scopes — the consented allowance a mounted server runs
/// under. Declared at mount, shown at the consent prompt, surfaced in
/// `/grants`, and enforced by the host: out-of-scope tool calls fail closed
/// with an audit row, and stdio children are spawned with network access only
/// when `network` is granted.
///
/// The default is fail-closed for the process axis (no network) and
/// permissive-but-pinned for the tool axis (empty `allowed_tools` = every tool
/// the server advertised at mount, which the catalog hash-pin already guards
/// against later rug-pulls). A caller that wants a narrower surface sets
/// `allowed_tools` explicitly.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ServerScopes {
    /// Tool-name globs (`*` wildcard) the server may expose and have called.
    /// Empty = every tool advertised at mount is in scope (back-compat). When
    /// non-empty, the catalog is filtered at mount and out-of-scope calls are
    /// denied with [`crate::McpHostError::ScopeDenied`].
    #[serde(default)]
    pub allowed_tools: Vec<String>,

    /// stdio only: may the spawned child reach the network? Default `false` —
    /// the sandbox denies outbound network unless this is granted.
    #[serde(default)]
    pub network: bool,

    /// stdio only: filesystem paths the child may access (sandbox workdir
    /// allowlist). Empty = the sandbox default (the child's working directory
    /// only).
    #[serde(default)]
    pub allowed_paths: Vec<PathBuf>,
}

impl ServerScopes {
    /// True when `tool` is within the declared tool scope. An empty
    /// `allowed_tools` means "every advertised tool"; otherwise the tool name
    /// must match one of the globs.
    pub fn allows_tool(&self, tool: &str) -> bool {
        if self.allowed_tools.is_empty() {
            return true;
        }
        self.allowed_tools.iter().any(|g| glob_match(g, tool))
    }

    /// Human-readable one-line summary for the consent prompt and `/grants`.
    pub fn summary(&self) -> String {
        let tools = if self.allowed_tools.is_empty() {
            "all advertised tools".to_string()
        } else {
            self.allowed_tools.join(", ")
        };
        let net = if self.network {
            "network allowed"
        } else {
            "no network"
        };
        if self.allowed_paths.is_empty() {
            format!("tools: {tools}; {net}")
        } else {
            let paths = self
                .allowed_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("tools: {tools}; {net}; paths: {paths}")
        }
    }
}

/// Minimal glob matcher supporting the `*` wildcard (matches any run,
/// including empty). No `?` or character classes — tool names are simple
/// identifiers, so `*` alone covers prefix/suffix/contains patterns. Anchored
/// at both ends.
fn glob_match(pattern: &str, text: &str) -> bool {
    // Split on '*' and require each literal segment to appear in order, with
    // the first anchored to the start and the last to the end.
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == text; // no wildcard → exact match
    }
    let mut pos = 0usize;
    for (i, seg) in segments.iter().enumerate() {
        if seg.is_empty() {
            continue;
        }
        if i == 0 {
            // Leading segment must be a prefix.
            if !text[pos..].starts_with(seg) {
                return false;
            }
            pos += seg.len();
        } else if i == segments.len() - 1 {
            // Trailing segment must be a suffix of the remainder.
            return text[pos..].ends_with(seg);
        } else if let Some(found) = text[pos..].find(seg) {
            pos += found + seg.len();
        } else {
            return false;
        }
    }
    true
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
    /// The egress scopes the user consented to for this server. Enforced by
    /// the host on every `call`.
    pub scopes: ServerScopes,
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
    /// The egress scopes this server runs under (consented at mount).
    #[serde(default)]
    pub scopes: ServerScopes,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tool_scope_allows_everything() {
        let s = ServerScopes::default();
        assert!(s.allows_tool("read_file"));
        assert!(s.allows_tool("anything"));
        // Process axis defaults fail-closed.
        assert!(!s.network);
    }

    #[test]
    fn explicit_tool_scope_is_exact_and_fail_closed() {
        let s = ServerScopes {
            allowed_tools: vec!["read_file".into(), "list_dir".into()],
            ..Default::default()
        };
        assert!(s.allows_tool("read_file"));
        assert!(s.allows_tool("list_dir"));
        assert!(!s.allows_tool("write_file"));
        assert!(!s.allows_tool("read_file_extra"));
    }

    #[test]
    fn glob_matches_prefix_suffix_and_contains() {
        assert!(glob_match("read_*", "read_file"));
        assert!(glob_match("read_*", "read_"));
        assert!(!glob_match("read_*", "write_file"));
        assert!(glob_match("*_file", "read_file"));
        assert!(!glob_match("*_file", "read_dir"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a*c", "abc"));
        assert!(glob_match("a*c", "ac"));
        assert!(!glob_match("a*c", "abd"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "exacto"));
    }

    #[test]
    fn glob_tool_scope_enforced() {
        let s = ServerScopes {
            allowed_tools: vec!["fs_*".into()],
            ..Default::default()
        };
        assert!(s.allows_tool("fs_read"));
        assert!(s.allows_tool("fs_write"));
        assert!(!s.allows_tool("net_fetch"));
    }

    #[test]
    fn summary_is_human_readable() {
        assert_eq!(
            ServerScopes::default().summary(),
            "tools: all advertised tools; no network"
        );
        let s = ServerScopes {
            allowed_tools: vec!["read_*".into()],
            network: true,
            allowed_paths: vec![PathBuf::from("/tmp/work")],
        };
        assert_eq!(
            s.summary(),
            "tools: read_*; network allowed; paths: /tmp/work"
        );
    }
}
