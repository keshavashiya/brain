//! # Brain MCP Host
//!
//! Host-side integration for **external** Model Context Protocol servers.
//! The sibling `brainos-mcp` crate is a *server* (Brain exposes its own
//! tools); this crate is the *host* side — Brain mounts and routes
//! through other people's tool servers.
//!
//! Supported transports (per MCP spec 2025-11-25):
//! - **stdio** — child process speaking JSON-RPC on stdin/stdout
//! - **Streamable HTTP** — current spec transport
//! - **HTTP+SSE** — legacy transport, still spec-required for compatibility
//!
//! This crate currently provides the trait surfaces ([`MCPHost`],
//! [`MCPClient`]), the [`ServerConfig`] / [`OAuthConfig`] / [`ToolDescriptor`]
//! / [`CallOutcome`] types, the [`McpHostError`] taxonomy, and an
//! [`InMemoryMcpHost`] no-transport stub so downstream wiring can be built
//! against the trait before transports are implemented.

use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    sync::Arc,
};

use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::RwLock;

pub mod aud_check;
pub mod capability_index;
pub mod error;
pub mod oauth;
pub mod resilient;
pub mod rmcp_host;
pub mod types;

pub use aud_check::{validate_token_aud, AudCheckOutcome};
pub use capability_index::{InMemoryToolCapabilityIndex, ToolCapabilityIndex};
pub use error::McpHostError;
pub use oauth::{manager_from_vault, VaultCredentialStore};
pub use resilient::{ResilienceConfig, ResilientMcpHost};
pub use rmcp_host::RmcpHost;
pub use types::{
    CallOutcome, MountedServer, OAuthConfig, ServerConfig, ServerInfo, ServerScopes, ServerStatus,
    ToolDescriptor,
};

/// MCP protocol version Brain negotiates against. Per spec 2025-11-25.
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

/// The host: manages the lifecycle of mounted servers and routes tool calls.
#[async_trait]
pub trait MCPHost: Send + Sync {
    /// Mount a new server under `name`. Idempotent: a name collision returns
    /// [`McpHostError::AlreadyMounted`]. Mounts with the default (fail-closed)
    /// egress scopes — see [`mount_with_scopes`](MCPHost::mount_with_scopes).
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError>;

    /// Mount a server under `name` with explicit egress [`ServerScopes`]. The
    /// host enforces these: out-of-scope tool calls fail closed with
    /// [`McpHostError::ScopeDenied`], and stdio children honour the process
    /// axis (network / paths). The default implementation drops `scopes` and
    /// delegates to [`mount`](MCPHost::mount) — hosts that enforce scopes
    /// (the real [`RmcpHost`]) override it.
    async fn mount_with_scopes(
        &self,
        name: String,
        cfg: ServerConfig,
        scopes: ServerScopes,
    ) -> Result<(), McpHostError> {
        let _ = scopes;
        self.mount(name, cfg).await
    }

    /// Gracefully unmount a server (stdin EOF → SIGTERM ladder for stdio,
    /// DELETE `Mcp-Session-Id` for HTTP transports).
    async fn unmount(&self, name: &str) -> Result<(), McpHostError>;

    /// Snapshot of currently-mounted servers.
    async fn list_servers(&self) -> Vec<ServerStatus>;

    /// Flattened tool catalog across all mounts. Raw enumeration only — a
    /// scored capability index is the responsibility of the intent router.
    async fn list_all_tools(&self) -> Vec<ToolDescriptor>;

    /// Invoke `tool` on `server` with `args`. Returns a structured outcome
    /// the caller (typically `SignalProcessor`) renders into an audit event.
    async fn call(
        &self,
        server: &str,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError>;

    /// Re-approve `server`'s *current* tool catalog as the trusted shape,
    /// lifting a catalog-change quarantine if one is active. Returns the
    /// number of tools adopted. The default implementation reports the
    /// host as quarantine-unaware; hosts that pin catalogs override it.
    async fn reconsent(&self, server: &str) -> Result<usize, McpHostError> {
        Err(McpHostError::Transport(format!(
            "this MCP host does not track catalog consent (server '{server}')"
        )))
    }
}

/// A single transport-bound MCP client (one per mounted server).
#[async_trait]
pub trait MCPClient: Send + Sync {
    async fn initialize(&self) -> Result<ServerInfo, McpHostError>;
    async fn list_tools(&self) -> Result<Vec<ToolDescriptor>, McpHostError>;
    async fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError>;
    async fn shutdown(&self) -> Result<(), McpHostError>;
    fn server_info(&self) -> Option<ServerInfo>;
}

/// In-memory `MCPHost` with no transport — records mounts so downstream
/// wiring (Signal, Thalamus intents, tests) can be built against the trait
/// before the real stdio / HTTP clients are wired in.
#[derive(Default)]
pub struct InMemoryMcpHost {
    mounted: RwLock<HashMap<String, MountedServer>>,
}

impl InMemoryMcpHost {
    pub fn new() -> Self {
        Self {
            mounted: RwLock::new(HashMap::new()),
        }
    }

    pub fn shared() -> Arc<dyn MCPHost> {
        Arc::new(Self::new())
    }
}

#[async_trait]
impl MCPHost for InMemoryMcpHost {
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        self.mount_with_scopes(name, cfg, ServerScopes::default())
            .await
    }

    async fn mount_with_scopes(
        &self,
        name: String,
        cfg: ServerConfig,
        scopes: ServerScopes,
    ) -> Result<(), McpHostError> {
        let mut guard = self.mounted.write().await;
        if guard.contains_key(&name) {
            return Err(McpHostError::AlreadyMounted(name));
        }
        guard.insert(
            name.clone(),
            MountedServer {
                name,
                config: cfg,
                mounted_at: Utc::now(),
                info: None,
                tools: Vec::new(),
                scopes,
            },
        );
        Ok(())
    }

    async fn unmount(&self, name: &str) -> Result<(), McpHostError> {
        self.mounted
            .write()
            .await
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| McpHostError::NotMounted(name.to_string()))
    }

    async fn list_servers(&self) -> Vec<ServerStatus> {
        self.mounted
            .read()
            .await
            .values()
            .map(|m| ServerStatus {
                name: m.name.clone(),
                mounted_at: m.mounted_at,
                tool_count: m.tools.len(),
                info: m.info.clone(),
                quarantined: false,
                scopes: m.scopes.clone(),
            })
            .collect()
    }

    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        self.mounted
            .read()
            .await
            .values()
            .flat_map(|m| m.tools.clone())
            .collect()
    }

    async fn call(
        &self,
        server: &str,
        tool: &str,
        _args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        let guard = self.mounted.read().await;
        let mounted = guard
            .get(server)
            .ok_or_else(|| McpHostError::NotMounted(server.to_string()))?;
        // Scope enforcement runs before the transport stub so callers (and
        // tests) can exercise the fail-closed path without a live server.
        if !mounted.scopes.allows_tool(tool) {
            return Err(McpHostError::ScopeDenied {
                server: server.to_string(),
                tool: tool.to_string(),
            });
        }
        // The in-memory host has no real transport — `call` is a stub so
        // callers can detect the no-transport state and downstream wiring
        // can be built against the trait surface.
        Err(McpHostError::Transport(
            "no transport configured for in-memory host".to_string(),
        ))
    }
}

/// Helper used by [`ServerConfig::Stdio`] to keep env maps deterministic.
pub fn empty_env() -> BTreeMap<String, String> {
    BTreeMap::new()
}

/// Helper for callers building stdio configs from path+args.
pub fn stdio_cfg(command: impl Into<String>, args: Vec<String>) -> ServerConfig {
    ServerConfig::Stdio {
        command: command.into(),
        args,
        env: empty_env(),
        cwd: None::<PathBuf>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mount_and_list() {
        let host = InMemoryMcpHost::new();
        host.mount("fs".into(), stdio_cfg("mcp-fs", vec![]))
            .await
            .unwrap();
        let servers = host.list_servers().await;
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "fs");
        assert_eq!(servers[0].tool_count, 0);
    }

    #[tokio::test]
    async fn double_mount_rejected() {
        let host = InMemoryMcpHost::new();
        host.mount("fs".into(), stdio_cfg("mcp-fs", vec![]))
            .await
            .unwrap();
        let err = host
            .mount("fs".into(), stdio_cfg("mcp-fs", vec![]))
            .await
            .unwrap_err();
        assert!(matches!(err, McpHostError::AlreadyMounted(_)));
    }

    #[tokio::test]
    async fn unmount_missing_errors() {
        let host = InMemoryMcpHost::new();
        let err = host.unmount("nope").await.unwrap_err();
        assert!(matches!(err, McpHostError::NotMounted(_)));
    }

    #[tokio::test]
    async fn call_without_transport_errors() {
        let host = InMemoryMcpHost::new();
        host.mount("fs".into(), stdio_cfg("mcp-fs", vec![]))
            .await
            .unwrap();
        let err = host
            .call("fs", "read_text_file", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, McpHostError::Transport(_)));
    }

    #[test]
    fn protocol_version_matches_spec() {
        assert_eq!(MCP_PROTOCOL_VERSION, "2025-11-25");
    }

    #[tokio::test]
    async fn out_of_scope_tool_call_fails_closed() {
        // A server mounted with an explicit tool scope blocks any tool outside
        // it *before* reaching the transport — the fail-closed DoD.
        let host = InMemoryMcpHost::new();
        host.mount_with_scopes(
            "fs".into(),
            stdio_cfg("mcp-fs", vec![]),
            ServerScopes {
                allowed_tools: vec!["read_*".into()],
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // Out of scope → ScopeDenied (never reaches the transport stub).
        let err = host
            .call("fs", "write_text_file", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, McpHostError::ScopeDenied { server, tool }
                if server == "fs" && tool == "write_text_file"),
            "expected ScopeDenied, got {err:?}"
        );

        // In scope → passes the gate, hits the (no-)transport stub.
        let err = host
            .call("fs", "read_text_file", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, McpHostError::Transport(_)),
            "in-scope tool should pass the scope gate"
        );
    }

    #[tokio::test]
    async fn scopeless_mount_allows_all_tools() {
        // Back-compat: a plain `mount` (default scopes) lets every tool through
        // the scope gate — only the network axis defaults fail-closed.
        let host = InMemoryMcpHost::new();
        host.mount("fs".into(), stdio_cfg("mcp-fs", vec![]))
            .await
            .unwrap();
        let err = host
            .call("fs", "anything_at_all", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, McpHostError::Transport(_)),
            "scope-less mount must not block tools, got {err:?}"
        );
        let servers = host.list_servers().await;
        assert!(!servers[0].scopes.network, "network defaults to denied");
    }
}
