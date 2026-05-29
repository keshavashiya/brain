//! `McpServer` — the JSON-RPC dispatcher that maps MCP methods to Brain
//! memory tools. The actual tool implementations live in [`crate::tools`].

use std::sync::Arc;

use brain::ApiKeyConfig;
use serde_json::{json, Value};

use crate::protocol::{JsonRpcRequest, JsonRpcResponse};

/// The MCP server — handles JSON-RPC requests against Brain memory tools.
pub struct McpServer {
    pub(crate) processor: Arc<signal::SignalProcessor>,
    /// Configured API keys; empty means auth is disabled.
    pub api_keys: Vec<ApiKeyConfig>,
}

impl McpServer {
    pub fn new(processor: Arc<signal::SignalProcessor>, api_keys: Vec<ApiKeyConfig>) -> Self {
        Self {
            processor,
            api_keys,
        }
    }

    /// Returns true if the given key is valid with write permission (or if auth is disabled).
    /// MCP clients can both read and write, so we require write permission.
    pub fn validate_key(&self, key: &str) -> bool {
        brain::check_auth(&self.api_keys, Some(key), "write").is_allowed()
    }

    /// Handle a single JSON-RPC request and return a response.
    ///
    /// Returns `None` for notifications (requests without an `id`).
    pub async fn handle(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id.clone().unwrap_or(Value::Null);

        // Notifications have no response.
        // Check both `id: None` (field absent) and `id: Some(Null)` (explicit null)
        // because serde deserializes `"id": null` as Some(Value::Null), not None.
        if req.is_notification() {
            return None;
        }

        let result = match req.method.as_str() {
            "initialize" => self.handle_initialize(&req),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&req).await,
            "resources/list" => Ok(json!({ "resources": [] })),
            "prompts/list" => Ok(json!({ "prompts": [] })),
            "ping" => Ok(json!({})),
            _ => Err((-32601, format!("Method not found: {}", req.method))),
        };

        Some(match result {
            Ok(value) => JsonRpcResponse::ok(id, value),
            Err((code, msg)) => JsonRpcResponse::err(id, code, msg),
        })
    }

    // ── Method handlers ──────────────────────────────────────────────────────

    fn handle_initialize(&self, _req: &JsonRpcRequest) -> Result<Value, (i32, String)> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": "brain",
                "version": env!("CARGO_PKG_VERSION")
            }
        }))
    }

    fn handle_tools_list(&self) -> Result<Value, (i32, String)> {
        static TOOLS_JSON: &str = include_str!("../assets/tools.json");
        serde_json::from_str(TOOLS_JSON)
            .map_err(|e| (-32603, format!("Failed to parse tools.json: {e}")))
    }

    async fn handle_tools_call(&self, req: &JsonRpcRequest) -> Result<Value, (i32, String)> {
        let params = req
            .params
            .as_ref()
            .ok_or((-32602, "Missing params".to_string()))?;

        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing tool name".to_string()))?;

        let args = params.get("arguments").cloned().unwrap_or(json!({}));

        match name {
            "memory_search" => self.tool_memory_search(&args).await,
            "memory_store" => self.tool_memory_store(&args).await,
            "memory_facts" => self.tool_memory_facts(&args),
            "memory_episodes" => self.tool_memory_episodes(&args),
            "user_profile" => self.tool_user_profile(),
            "memory_procedures" => self.tool_memory_procedures(&args),
            "brain_capabilities" => self.tool_capabilities().await,
            other => Err((-32602, format!("Unknown tool: {other}"))),
        }
    }
}
