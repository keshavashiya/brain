use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Incoming JSON-RPC request (or notification when `id` is absent).
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// A JSON-RPC message is a notification (no response expected).
    ///
    /// True when:
    /// - Method starts with `notifications/` (MCP convention, always a notification)
    /// - Method is `initialized` (MCP lifecycle notification)
    /// - `id` is absent or explicitly `null` (JSON-RPC spec: no id → notification)
    pub fn is_notification(&self) -> bool {
        if self.method.starts_with("notifications/") || self.method == "initialized" {
            return true;
        }
        matches!(&self.id, None | Some(Value::Null))
    }
}

/// Outgoing JSON-RPC response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
        }
    }
}
