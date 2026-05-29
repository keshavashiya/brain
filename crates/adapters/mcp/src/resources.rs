//! MCP `resources/list` + `resources/read` — read-only views over Brain's
//! memory, exposed as `brain://…` resources. Bodies reuse the same processor
//! accessors as the memory tools so resources and tools never drift.

use serde_json::{json, Value};

use crate::protocol::JsonRpcRequest;
use crate::server::McpServer;

/// Stable resource URIs.
const URI_PROFILE: &str = "brain://profile";
const URI_CAPABILITIES: &str = "brain://capabilities";
const URI_NAMESPACES: &str = "brain://namespaces";

impl McpServer {
    /// `resources/list` — advertise the read-only memory views.
    pub(crate) fn handle_resources_list(&self) -> Result<Value, (i32, String)> {
        Ok(json!({
            "resources": [
                {
                    "uri": URI_PROFILE,
                    "name": "User profile",
                    "description": "Active LLM/embedding config, data dir, and encryption status.",
                    "mimeType": "application/json"
                },
                {
                    "uri": URI_CAPABILITIES,
                    "name": "Capabilities",
                    "description": "Brain's live capability manifest (tools, agents, backends).",
                    "mimeType": "text/plain"
                },
                {
                    "uri": URI_NAMESPACES,
                    "name": "Memory namespaces",
                    "description": "Configured memory namespaces with fact and episode counts.",
                    "mimeType": "application/json"
                }
            ]
        }))
    }

    /// `resources/read` — return one resource's contents by `uri`.
    pub(crate) async fn handle_resources_read(
        &self,
        req: &JsonRpcRequest,
    ) -> Result<Value, (i32, String)> {
        let uri = req
            .params
            .as_ref()
            .and_then(|p| p.get("uri"))
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing resource uri".to_string()))?;

        let (mime, text) = match uri {
            URI_PROFILE => ("application/json", self.profile_json()),
            URI_CAPABILITIES => ("text/plain", self.processor.capability_manifest().await),
            URI_NAMESPACES => ("application/json", self.namespaces_json()),
            other => return Err((-32602, format!("Unknown resource uri: {other}"))),
        };

        Ok(json!({
            "contents": [
                { "uri": uri, "mimeType": mime, "text": text }
            ]
        }))
    }

    /// Namespaces with counts as a JSON array. `NamespaceStats` isn't
    /// `Serialize`, so build the objects by field.
    fn namespaces_json(&self) -> String {
        let namespaces: Vec<Value> = self
            .processor
            .list_namespaces()
            .into_iter()
            .map(|ns| {
                json!({
                    "namespace": ns.namespace,
                    "fact_count": ns.fact_count,
                    "episode_count": ns.episode_count,
                })
            })
            .collect();
        serde_json::to_string_pretty(&json!({ "namespaces": namespaces })).unwrap_or_default()
    }
}
