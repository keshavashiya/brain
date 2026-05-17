//! `MCPHost` implementation backed by the `rmcp` Rust SDK.
//!
//! Transports supported:
//! - **stdio** — child process speaking MCP JSON-RPC on stdin/stdout.
//! - **Streamable HTTP** — current spec transport. With or without OAuth 2.1.
//! - **HTTP+SSE** — legacy transport, routes through the same streamable HTTP
//!   client (the rmcp transport speaks both shapes against the same endpoint).
//!
//! Per-server tool catalogs are hash-pinned: the SHA-256 of the canonicalized
//! `tools/list` response is captured at mount time. On a refresh (driven by
//! `notifications/tools/list_changed` in a later PR) a hash change emits a
//! `BrainEvent::Error { source: "mcphost", message: "tools/list hash changed …" }`
//! — that's the "rug-pull" signal the consent UI surfaces to the user.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use chrono::Utc;
use http::{HeaderName, HeaderValue};
use observe::{BrainEvent, Observer};
use rmcp::{
    model::CallToolRequestParams,
    service::{RoleClient, RunningService, ServiceExt},
    transport::{
        auth::AuthClient,
        streamable_http_client::{
            StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
        },
        TokioChildProcess,
    },
};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use tracing::warn;
use uuid::Uuid;
use vault::CredentialVault;

use crate::{
    capability_index::CapabilityIndex,
    error::McpHostError,
    oauth,
    types::{CallOutcome, MountedServer, ServerConfig, ServerInfo, ServerStatus, ToolDescriptor},
    MCPHost, MCP_PROTOCOL_VERSION,
};

/// Real `MCPHost` backed by `rmcp`. Each mounted server gets a
/// `RunningService<RoleClient, ()>` peer plus a cached metadata snapshot
/// and a hash-pin of the initial `tools/list`.
pub struct RmcpHost {
    mounted: RwLock<HashMap<String, Mounted>>,
    observer: Option<Arc<dyn Observer>>,
    vault: Option<Arc<dyn CredentialVault>>,
    capability_index: Option<Arc<dyn CapabilityIndex>>,
    tool_registry: Option<Arc<dyn intent::ToolRegistry>>,
}

struct Mounted {
    record: MountedServer,
    /// SHA-256 of the canonicalized initial `tools/list` response. A later
    /// `notifications/tools/list_changed` refresh that produces a different
    /// hash is the "rug-pull" signal (CVE-2025-54136 class).
    tools_hash: String,
    /// The live rmcp peer. `None` is impossible for fully-initialized
    /// mounts; the option lets us pull the service out during `unmount`
    /// to call `.cancel().await` (which consumes `self`).
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
            observer: None,
            vault: None,
            capability_index: None,
            tool_registry: None,
        }
    }

    pub fn shared() -> Arc<dyn MCPHost> {
        Arc::new(Self::new())
    }

    /// Wire an [`Observer`] so rug-pull / refresh-failure events reach the
    /// in-process event bus.
    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Wire a [`CredentialVault`] so HTTP mounts with `oauth: Some` can load
    /// persisted OAuth tokens. Without a vault, an `oauth: Some` config will
    /// fail at mount with [`McpHostError::Auth`].
    pub fn with_vault(mut self, vault: Arc<dyn CredentialVault>) -> Self {
        self.vault = Some(vault);
        self
    }

    /// Wire a [`CapabilityIndex`] so every successful mount auto-registers
    /// the server's tool catalog and every unmount drops it. The intent
    /// router queries the same index when resolving a tool route.
    pub fn with_capability_index(mut self, index: Arc<dyn CapabilityIndex>) -> Self {
        self.capability_index = Some(index);
        self
    }

    /// Wire an [`intent::ToolRegistry`] so every mounted server's tools
    /// land in the workspace-wide capability registry the
    /// [`intent::IntentRouter`] resolves against. Tool ids follow the
    /// `mcp:{server}:{tool_name}` pattern; unmount deregisters every tool
    /// whose source matches the unmounted server, and a rug-pull refresh
    /// re-syncs the registry with the latest catalog.
    pub fn with_tool_registry(mut self, registry: Arc<dyn intent::ToolRegistry>) -> Self {
        self.tool_registry = Some(registry);
        self
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

        let svc: RunningService<RoleClient, ()> = ()
            .serve(transport)
            .await
            .map_err(|e| McpHostError::Initialize(e.to_string()))?;

        self.finalize_mount(name, cfg, svc).await
    }

    async fn mount_http(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        let (url, oauth_cfg) = match &cfg {
            ServerConfig::StreamableHttp { url, oauth } | ServerConfig::HttpSse { url, oauth } => {
                (url.clone(), oauth.clone())
            }
            ServerConfig::Stdio { .. } => {
                return Err(McpHostError::Transport(
                    "RmcpHost::mount_http called with non-HTTP config".into(),
                ))
            }
        };

        // Local HTTP servers must bind 127.0.0.1 and we enforce that by
        // rejecting anything else as part of mounting. Per MCP spec, this
        // mitigates DNS-rebinding against local-only servers.
        validate_local_origin(&url)?;

        let mut transport_cfg = StreamableHttpClientTransportConfig::with_uri(url.clone());
        transport_cfg.custom_headers = protocol_version_headers();

        let svc: RunningService<RoleClient, ()> = if let Some(oauth) = oauth_cfg.as_ref() {
            let vault = self.vault.clone().ok_or_else(|| {
                McpHostError::Auth(
                    "OAuth configured but RmcpHost has no vault — wire one via with_vault()".into(),
                )
            })?;
            // RFC 8707 resource indicator. Defaults to the server URL
            // when the operator hasn't customised it — that's the
            // canonical mapping for vanilla MCP deployments where the
            // server IS the protected resource.
            let expected_resource = if oauth.resource.trim().is_empty() {
                url.as_str()
            } else {
                oauth.resource.as_str()
            };
            let manager = oauth::manager_from_vault(
                &url,
                &name,
                expected_resource,
                vault,
                self.observer.clone(),
            )
            .await?;
            let auth_client = AuthClient::new(reqwest::Client::new(), manager);
            let transport = StreamableHttpClientTransport::with_client(auth_client, transport_cfg);
            ().serve(transport)
                .await
                .map_err(|e| McpHostError::Initialize(e.to_string()))?
        } else {
            let transport = StreamableHttpClientTransport::from_config(transport_cfg);
            ().serve(transport)
                .await
                .map_err(|e| McpHostError::Initialize(e.to_string()))?
        };

        self.finalize_mount(name, cfg, svc).await
    }

    async fn finalize_mount(
        &self,
        name: String,
        cfg: ServerConfig,
        svc: RunningService<RoleClient, ()>,
    ) -> Result<(), McpHostError> {
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
        let tools_hash = hash_tools(&tools);

        let record = MountedServer {
            name: name.clone(),
            config: cfg,
            mounted_at: Utc::now(),
            info,
            tools: tools.clone(),
        };
        let mut guard = self.mounted.write().await;
        if guard.contains_key(&name) {
            return Err(McpHostError::AlreadyMounted(name));
        }
        guard.insert(
            name.clone(),
            Mounted {
                record,
                tools_hash,
                service: Some(svc),
            },
        );
        drop(guard);
        if let Some(index) = &self.capability_index {
            index.upsert(&name, tools.clone());
        }
        if let Some(registry) = &self.tool_registry {
            for t in &tools {
                let _ = registry.register(tool_to_intent_descriptor(&name, t)).await;
            }
        }
        Ok(())
    }

    /// Re-fetch `tools/list` for a mounted server and compare against the
    /// pinned hash. A mismatch emits `BrainEvent::Error` (rug-pull signal)
    /// and updates the pinned record with the new shape so subsequent
    /// refreshes are scored against the latest.
    pub async fn refresh_tools(&self, server: &str) -> Result<bool, McpHostError> {
        let (tools, old_hash) = {
            let guard = self.mounted.read().await;
            let mounted = guard
                .get(server)
                .ok_or_else(|| McpHostError::NotMounted(server.to_string()))?;
            let svc = mounted.service.as_ref().ok_or_else(|| {
                McpHostError::Transport(format!("server '{server}' has no live service"))
            })?;
            let tools_raw = svc
                .list_all_tools()
                .await
                .map_err(|e| McpHostError::Rmcp(format!("tools/list refresh: {e}")))?;
            let tools: Vec<ToolDescriptor> = tools_raw
                .into_iter()
                .map(|t| ToolDescriptor {
                    server: server.to_string(),
                    name: t.name.to_string(),
                    description: t.description.map(|d| d.to_string()),
                    input_schema: serde_json::Value::Object((*t.input_schema).clone()),
                })
                .collect();
            (tools, mounted.tools_hash.clone())
        };
        let new_hash = hash_tools(&tools);

        let changed = new_hash != old_hash;
        if changed {
            if let Some(observer) = &self.observer {
                let _ = observer
                    .publish(BrainEvent::Error {
                        id: Uuid::new_v4(),
                        source: "mcphost".into(),
                        message: format!(
                            "tools/list hash changed for server '{server}' (old={old_hash}, new={new_hash})"
                        ),
                        ts: Utc::now(),
                    })
                    .await;
            }
            let mut guard = self.mounted.write().await;
            if let Some(m) = guard.get_mut(server) {
                m.record.tools = tools.clone();
                m.tools_hash = new_hash;
            }
            drop(guard);
            if let Some(index) = &self.capability_index {
                index.upsert(server, tools.clone());
            }
            if let Some(registry) = &self.tool_registry {
                // Deregister every tool whose source matches this server,
                // then register the fresh shape. The registry overwrites by
                // tool_id so renamed tools land cleanly and disappeared
                // tools are pruned.
                for existing in registry.list().await {
                    if let intent::ToolSource::McpServer { server: s } = &existing.source {
                        if s == server {
                            let _ = registry.deregister(&existing.tool_id).await;
                        }
                    }
                }
                for t in &tools {
                    let _ = registry
                        .register(tool_to_intent_descriptor(server, t))
                        .await;
                }
            }
        }
        Ok(changed)
    }
}

/// Convert an [`mcphost::types::ToolDescriptor`] (the MCP wire shape) into
/// the workspace-wide [`intent::ToolDescriptor`] the capability router
/// resolves against. Tool ids follow `mcp:{server}:{tool_name}` so the same
/// id is stable across refreshes and reachable from a `ToolRoute::Mcp`.
fn tool_to_intent_descriptor(server: &str, t: &ToolDescriptor) -> intent::ToolDescriptor {
    intent::ToolDescriptor {
        tool_id: format!("mcp:{server}:{}", t.name),
        source: intent::ToolSource::McpServer {
            server: server.to_string(),
        },
        // The MCP wire format doesn't carry a verb the way SIT does; until
        // a server-side annotation lands the host stamps a coarse
        // `mcp.<tool_name>` pair so router scoring has something to match.
        verb: intent::Verb::new("mcp", t.name.clone()),
        description: t.description.clone().unwrap_or_default(),
        input_schema: t.input_schema.clone(),
        output_schema: None,
        capabilities: Vec::new(),
        annotations: intent::ToolAnnotations::default(),
        embedding: None,
    }
}

#[async_trait]
impl MCPHost for RmcpHost {
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        match &cfg {
            ServerConfig::Stdio { .. } => self.mount_stdio(name, cfg).await,
            ServerConfig::StreamableHttp { .. } | ServerConfig::HttpSse { .. } => {
                self.mount_http(name, cfg).await
            }
        }
    }

    async fn unmount(&self, name: &str) -> Result<(), McpHostError> {
        let mut entry = {
            let mut guard = self.mounted.write().await;
            guard
                .remove(name)
                .ok_or_else(|| McpHostError::NotMounted(name.to_string()))?
        };
        if let Some(index) = &self.capability_index {
            index.remove(name);
        }
        if let Some(registry) = &self.tool_registry {
            for existing in registry.list().await {
                if let intent::ToolSource::McpServer { server: s } = &existing.source {
                    if s == name {
                        let _ = registry.deregister(&existing.tool_id).await;
                    }
                }
            }
        }
        if let Some(svc) = entry.service.take() {
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

/// Build the static custom-headers map for every HTTP request. The MCP spec
/// requires every client request to advertise the protocol version it
/// negotiated against (the negotiated version itself comes back from the
/// server during `initialize`).
fn protocol_version_headers() -> HashMap<HeaderName, HeaderValue> {
    let mut headers = HashMap::new();
    let name = HeaderName::from_static("mcp-protocol-version");
    if let Ok(value) = HeaderValue::from_str(MCP_PROTOCOL_VERSION) {
        headers.insert(name, value);
    }
    headers
}

/// SHA-256 of the canonicalized `(name, description, input_schema)` tuples in
/// stable order. Stable serialization comes from `serde_json::to_vec` over a
/// pre-sorted vector, since `serde_json::Value::Object` preserves insertion
/// order but JSON object semantics are unordered.
fn hash_tools(tools: &[ToolDescriptor]) -> String {
    let mut canonical: Vec<(String, Option<String>, String)> = tools
        .iter()
        .map(|t| {
            let schema = canonical_json(&t.input_schema);
            (t.name.clone(), t.description.clone(), schema)
        })
        .collect();
    canonical.sort_by(|a, b| a.0.cmp(&b.0));
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!("{:x}", hasher.finalize())
}

/// Canonicalize JSON: sort object keys recursively so insertion order doesn't
/// affect the hash.
fn canonical_json(v: &serde_json::Value) -> String {
    fn sort(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(m) => {
                let mut sorted: Vec<(String, serde_json::Value)> =
                    m.iter().map(|(k, v)| (k.clone(), sort(v))).collect();
                sorted.sort_by(|a, b| a.0.cmp(&b.0));
                serde_json::Value::Object(sorted.into_iter().collect())
            }
            serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(sort).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort(v)).unwrap_or_default()
}

/// Reject HTTP MCP URLs that point at a non-loopback address bound to a
/// local-only port. Public hosts are allowed unconditionally; only the
/// `localhost` / `127.0.0.1` / `::1` cases are checked.
///
/// This is the cheap-but-real mitigation for DNS rebinding against local-
/// only servers (MCP spec security guidance, item 3 in the PR plan).
fn validate_local_origin(url: &str) -> Result<(), McpHostError> {
    let parsed = url::Url::parse(url)
        .map_err(|e| McpHostError::Transport(format!("invalid URL '{url}': {e}")))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(McpHostError::Transport(format!(
                "unsupported URL scheme '{other}' (expected http or https)"
            )))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn td(name: &str, desc: Option<&str>, schema: serde_json::Value) -> ToolDescriptor {
        ToolDescriptor {
            server: "s".into(),
            name: name.into(),
            description: desc.map(|s| s.to_string()),
            input_schema: schema,
        }
    }

    #[test]
    fn hash_tools_is_order_independent() {
        let a = vec![
            td("z", None, json!({"type": "object"})),
            td("a", None, json!({"type": "object"})),
        ];
        let b = vec![
            td("a", None, json!({"type": "object"})),
            td("z", None, json!({"type": "object"})),
        ];
        assert_eq!(hash_tools(&a), hash_tools(&b));
    }

    #[test]
    fn hash_tools_detects_description_change() {
        let a = vec![td("read", Some("safe"), json!({"type": "object"}))];
        let b = vec![td("read", Some("MALICIOUS"), json!({"type": "object"}))];
        assert_ne!(hash_tools(&a), hash_tools(&b));
    }

    #[test]
    fn hash_tools_detects_schema_change() {
        let a = vec![td(
            "fs.read",
            None,
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )];
        let b = vec![td(
            "fs.read",
            None,
            json!({"type": "object", "properties": {"path": {"type": "string"}, "secret": {"type": "string"}}}),
        )];
        assert_ne!(hash_tools(&a), hash_tools(&b));
    }

    #[test]
    fn canonical_json_sorts_keys() {
        let a = json!({"b": 1, "a": 2});
        let b = json!({"a": 2, "b": 1});
        assert_eq!(canonical_json(&a), canonical_json(&b));
    }

    #[test]
    fn protocol_version_header_is_set() {
        let headers = protocol_version_headers();
        let key = HeaderName::from_static("mcp-protocol-version");
        let value = headers.get(&key).expect("header must be present");
        assert_eq!(value.to_str().unwrap(), MCP_PROTOCOL_VERSION);
    }

    #[test]
    fn validate_local_origin_rejects_non_http() {
        assert!(validate_local_origin("ftp://example.com").is_err());
        assert!(validate_local_origin("not a url").is_err());
        assert!(validate_local_origin("http://example.com/mcp").is_ok());
        assert!(validate_local_origin("https://example.com/mcp").is_ok());
        assert!(validate_local_origin("http://127.0.0.1:8080/mcp").is_ok());
    }
}
