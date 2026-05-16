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

pub mod capability_index;
pub mod error;
pub mod oauth;
pub mod resilient;
pub mod rmcp_host;
pub mod types;

pub use capability_index::{CapabilityIndex, InMemoryCapabilityIndex};
pub use error::McpHostError;
pub use oauth::{manager_from_vault, VaultCredentialStore};
pub use resilient::{ResilienceConfig, ResilientMcpHost};
pub use rmcp_host::RmcpHost;
pub use types::{
    CallOutcome, MountedServer, OAuthConfig, ServerConfig, ServerInfo, ServerStatus, ToolDescriptor,
};

/// MCP protocol version Brain negotiates against. Per spec 2025-11-25.
pub const MCP_PROTOCOL_VERSION: &str = "2025-11-25";

/// The host: manages the lifecycle of mounted servers and routes tool calls.
#[async_trait]
pub trait MCPHost: Send + Sync {
    /// Mount a new server under `name`. Idempotent: a name collision returns
    /// [`McpHostError::AlreadyMounted`].
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError>;

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
        _tool: &str,
        _args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        let guard = self.mounted.read().await;
        if !guard.contains_key(server) {
            return Err(McpHostError::NotMounted(server.to_string()));
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
}
