//! `MCPHost` implementation backed by the `rmcp` Rust SDK.
//!
//! Currently supports the **stdio** transport — child processes speaking
//! MCP JSON-RPC on stdin/stdout. The HTTP transports (Streamable HTTP +
//! legacy HTTP+SSE) hook in here once their rmcp feature flags are
//! enabled.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use chrono::Utc;
use rmcp::{
    model::CallToolRequestParams,
    service::{RoleClient, RunningService, ServiceExt},
    transport::TokioChildProcess,
};
use tokio::sync::RwLock;
use tracing::warn;

use crate::{
    error::McpHostError,
    types::{CallOutcome, MountedServer, ServerConfig, ServerInfo, ServerStatus, ToolDescriptor},
    MCPHost,
};

/// Real `MCPHost` backed by `rmcp`. Each mounted server gets a
/// `RunningService<RoleClient, ()>` peer plus a cached metadata snapshot
/// for `list_servers` / `list_all_tools`.
pub struct RmcpHost {
    mounted: RwLock<HashMap<String, Mounted>>,
}

struct Mounted {
    record: MountedServer,
    /// The live rmcp peer. `None` is impossible for fully-initialized
    /// stdio mounts; the option lets us pull the service out during
    /// `unmount` to call `.cancel().await` (which consumes `self`).
    service: Option<RunningService<RoleClient, ()>>,
}

impl Default for RmcpHost {
    fn default() -> Self {
        Self::new()
    }
}

impl RmcpHost {
    pub fn new() -> Self {
        Self {
            mounted: RwLock::new(HashMap::new()),
        }
    }

    pub fn shared() -> Arc<dyn MCPHost> {
        Arc::new(Self::new())
    }

    async fn mount_stdio(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        let ServerConfig::Stdio {
            command,
            args,
            env,
            cwd,
        } = &cfg
        else {
            return Err(McpHostError::Transport(
                "RmcpHost::mount_stdio called with non-stdio config".into(),
            ));
        };

        // Build the child Command. `rmcp::transport::ConfigureCommandExt` is
        // the ergonomic configure-by-closure pattern the rmcp examples use,
        // but the bare `tokio::process::Command` already does everything we
        // need here.
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        for (k, v) in env {
            cmd.env(k, v);
        }
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        let transport = TokioChildProcess::new(cmd)
            .map_err(|e| McpHostError::Transport(format!("spawn '{command}': {e}")))?;

        // `().serve(transport)` runs the MCP `initialize` handshake under
        // the hood — that's why we get a `RunningService` back rather than
        // having to call `initialize` ourselves.
        let svc: RunningService<RoleClient, ()> = ()
            .serve(transport)
            .await
            .map_err(|e| McpHostError::Initialize(e.to_string()))?;

        // Snapshot server info + tools immediately. The peer is cheap to
        // dereference; `list_all_tools` paginates internally so we get the
        // complete catalog without follow-up calls.
        let info = svc.peer_info().map(|init| ServerInfo {
            name: init.server_info.name.to_string(),
            version: init.server_info.version.to_string(),
            protocol_version: init.protocol_version.to_string(),
        });
        let tools_raw = svc
            .list_all_tools()
            .await
            .map_err(|e| McpHostError::Initialize(format!("list_tools after initialize: {e}")))?;
        let tools: Vec<ToolDescriptor> = tools_raw
            .into_iter()
            .map(|t| ToolDescriptor {
                server: name.clone(),
                name: t.name.to_string(),
                description: t.description.map(|d| d.to_string()),
                input_schema: serde_json::Value::Object((*t.input_schema).clone()),
            })
            .collect();

        let record = MountedServer {
            name: name.clone(),
            config: cfg,
            mounted_at: Utc::now(),
            info,
            tools,
        };
        let mut guard = self.mounted.write().await;
        if guard.contains_key(&name) {
            return Err(McpHostError::AlreadyMounted(name));
        }
        guard.insert(
            name,
            Mounted {
                record,
                service: Some(svc),
            },
        );
        Ok(())
    }
}

#[async_trait]
impl MCPHost for RmcpHost {
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        match &cfg {
            ServerConfig::Stdio { .. } => self.mount_stdio(name, cfg).await,
            ServerConfig::StreamableHttp { .. } | ServerConfig::HttpSse { .. } => Err(
                McpHostError::Transport("HTTP transports not yet implemented".into()),
            ),
        }
    }

    async fn unmount(&self, name: &str) -> Result<(), McpHostError> {
        let mut entry = {
            let mut guard = self.mounted.write().await;
            guard
                .remove(name)
                .ok_or_else(|| McpHostError::NotMounted(name.to_string()))?
        };
        if let Some(svc) = entry.service.take() {
            // Graceful shutdown: cancellation completes after the server
            // acks the cancel request or the transport drops, whichever
            // comes first.
            match svc.cancel().await {
                Ok(_) => {}
                Err(e) => {
                    warn!(server = name, error = %e, "rmcp cancel failed");
                }
            }
        }
        Ok(())
    }

    async fn list_servers(&self) -> Vec<ServerStatus> {
        self.mounted
            .read()
            .await
            .values()
            .map(|m| ServerStatus {
                name: m.record.name.clone(),
                mounted_at: m.record.mounted_at,
                tool_count: m.record.tools.len(),
                info: m.record.info.clone(),
            })
            .collect()
    }

    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        self.mounted
            .read()
            .await
            .values()
            .flat_map(|m| m.record.tools.clone())
            .collect()
    }

    async fn call(
        &self,
        server: &str,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        let started = std::time::Instant::now();
        let guard = self.mounted.read().await;
        let mounted = guard
            .get(server)
            .ok_or_else(|| McpHostError::NotMounted(server.to_string()))?;
        let svc = mounted.service.as_ref().ok_or_else(|| {
            McpHostError::Transport(format!("server '{server}' has no live service"))
        })?;

        // `arguments` must be a JSON object per the MCP schema. Anything
        // else gets rejected here rather than at the server side.
        let arguments = match args {
            serde_json::Value::Object(o) => Some(o),
            serde_json::Value::Null => None,
            other => {
                return Err(McpHostError::Transport(format!(
                    "tools/call arguments must be a JSON object or null, got {}",
                    match other {
                        serde_json::Value::Bool(_) => "bool",
                        serde_json::Value::Number(_) => "number",
                        serde_json::Value::String(_) => "string",
                        serde_json::Value::Array(_) => "array",
                        _ => "unknown",
                    }
                )));
            }
        };
        let mut params = CallToolRequestParams::new(tool.to_string());
        params.arguments = arguments;
        let result = svc
            .call_tool(params)
            .await
            .map_err(|e| McpHostError::Rmcp(e.to_string()))?;

        let content =
            serde_json::to_value(&result.content).unwrap_or(serde_json::Value::Array(Vec::new()));
        Ok(CallOutcome {
            server: server.to_string(),
            tool: tool.to_string(),
            is_error: result.is_error.unwrap_or(false),
            content,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}
