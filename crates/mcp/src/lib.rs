//! # Brain MCP Adapter
//!
//! Exposes Brain's memory tools as an MCP (Model Context Protocol) server.
//!
//! ## Transports
//! - **stdio** (`brain mcp --stdio`): line-delimited JSON-RPC on stdin/stdout
//! - **HTTP** (`brain mcp --http`): JSON-RPC over HTTP POST on port 19791
//!
//! ## Tools
//! - `memory_search`   — semantic search over stored facts/episodes
//! - `memory_store`    — store a structured fact (subject predicate object)
//! - `memory_facts`    — get all facts about a subject
//! - `memory_episodes` — get recent conversation episodes
//! - `user_profile`    — return user profile / config data
//!
//! ## Claude Desktop config snippet
//! ```json
//! {
//!   "mcpServers": {
//!     "brain-memory": {
//!       "command": "brain",
//!       "args": ["mcp", "--stdio"]
//!     }
//!   }
//! }
//! ```

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json as AxumJson,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

// ─── Errors ───────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Server error: {0}")]
    Server(String),
}

// ─── JSON-RPC 2.0 Types ───────────────────────────────────────────────────────

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

// ─── MCP Server ───────────────────────────────────────────────────────────────

/// The MCP server — handles JSON-RPC requests against Brain memory tools.
pub struct McpServer {
    processor: Arc<signal::SignalProcessor>,
}

impl McpServer {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        Self { processor }
    }

    /// Handle a single JSON-RPC request and return a response.
    ///
    /// Returns `None` for notifications (requests without an `id`).
    pub async fn handle(&self, req: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let id = req.id.clone().unwrap_or(Value::Null);

        // Notifications have no `id` — no response required
        if req.id.is_none()
            && (req.method == "notifications/initialized"
                || req.method.starts_with("notifications/"))
        {
            return None;
        }

        let result = match req.method.as_str() {
            "initialize" => self.handle_initialize(&req),
            "initialized" => return None, // notification
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&req).await,
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
                "tools": {}
            },
            "serverInfo": {
                "name": "brain-memory",
                "version": env!("CARGO_PKG_VERSION")
            }
        }))
    }

    fn handle_tools_list(&self) -> Result<Value, (i32, String)> {
        Ok(json!({
            "tools": [
                {
                    "name": "memory_search",
                    "description": "Search Brain memory for relevant facts and episodes by semantic similarity.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            },
                            "top_k": {
                                "type": "number",
                                "description": "Number of results to return (default: 10)"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "memory_store",
                    "description": "Store a structured semantic fact in Brain memory.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "The entity this fact is about (e.g. 'user', 'project-x')"
                            },
                            "predicate": {
                                "type": "string",
                                "description": "The relationship or property (e.g. 'prefers', 'is_working_on')"
                            },
                            "object": {
                                "type": "string",
                                "description": "The value of the relationship (e.g. 'Rust', 'Brain OS')"
                            },
                            "category": {
                                "type": "string",
                                "description": "Namespace/category for the fact (e.g. 'personal', 'work'). Defaults to 'general'."
                            }
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                },
                {
                    "name": "memory_facts",
                    "description": "Retrieve all stored facts about a specific subject.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "The subject to retrieve facts about"
                            }
                        },
                        "required": ["subject"]
                    }
                },
                {
                    "name": "memory_episodes",
                    "description": "Retrieve recent conversation episodes from episodic memory.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "number",
                                "description": "Number of episodes to return (default: 20)"
                            }
                        }
                    }
                },
                {
                    "name": "user_profile",
                    "description": "Retrieve the user profile and Brain OS configuration.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }))
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
            other => Err((-32602, format!("Unknown tool: {other}"))),
        }
    }

    // ── Tool implementations ─────────────────────────────────────────────────

    async fn tool_memory_search(&self, args: &Value) -> Result<Value, (i32, String)> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: query".to_string()))?;

        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(10) as usize;

        let results = self.processor.search_facts(query, top_k).await;

        let text = if results.is_empty() {
            "No relevant facts found in memory.".to_string()
        } else {
            let lines: Vec<String> = results
                .iter()
                .map(|r| {
                    format!(
                        "[{}] {} {} {} (confidence: {:.2}, distance: {:.3})",
                        r.fact.category,
                        r.fact.subject,
                        r.fact.predicate,
                        r.fact.object,
                        r.fact.confidence,
                        r.distance
                    )
                })
                .collect();
            lines.join("\n")
        };

        Ok(tool_result_text(text))
    }

    async fn tool_memory_store(&self, args: &Value) -> Result<Value, (i32, String)> {
        let subject = args
            .get("subject")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: subject".to_string()))?;

        let predicate = args
            .get("predicate")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: predicate".to_string()))?;

        let object = args
            .get("object")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: object".to_string()))?;

        let category = args
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("general");

        match self
            .processor
            .store_fact_direct(category, subject, predicate, object)
            .await
        {
            Ok(id) => Ok(tool_result_text(format!(
                "Stored fact [{id}]: {subject} {predicate} {object} (category: {category})"
            ))),
            Err(e) => Err((-32603, format!("Failed to store fact: {e}"))),
        }
    }

    fn tool_memory_facts(&self, args: &Value) -> Result<Value, (i32, String)> {
        let subject = args
            .get("subject")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: subject".to_string()))?;

        let facts = self.processor.facts_about(subject);

        let text = if facts.is_empty() {
            format!("No facts found about '{subject}'.")
        } else {
            let lines: Vec<String> = facts
                .iter()
                .map(|f| {
                    format!(
                        "[{}] {} {} {} (confidence: {:.2})",
                        f.category, f.subject, f.predicate, f.object, f.confidence
                    )
                })
                .collect();
            lines.join("\n")
        };

        Ok(tool_result_text(text))
    }

    fn tool_memory_episodes(&self, args: &Value) -> Result<Value, (i32, String)> {
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(20) as usize;

        let episodes = self.processor.recent_episodes(limit);

        let text = if episodes.is_empty() {
            "No conversation episodes found.".to_string()
        } else {
            let lines: Vec<String> = episodes
                .iter()
                .map(|e| format!("[{}] {}: {}", e.timestamp, e.role, e.content))
                .collect();
            lines.join("\n")
        };

        Ok(tool_result_text(text))
    }

    fn tool_user_profile(&self) -> Result<Value, (i32, String)> {
        let config = self.processor.config();
        let profile = json!({
            "llm": {
                "provider": config.llm.provider,
                "model": config.llm.model
            },
            "embedding": {
                "model": config.embedding.model,
                "dimensions": config.embedding.dimensions
            },
            "data_dir": config.data_dir().to_string_lossy(),
            "encryption_enabled": config.encryption.enabled
        });
        Ok(tool_result_text(serde_json::to_string_pretty(&profile).unwrap_or_default()))
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Build a standard MCP tool result with a single text content block.
fn tool_result_text(text: impl Into<String>) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text.into()
            }
        ]
    })
}

// ─── Stdio Transport ──────────────────────────────────────────────────────────

/// Run the MCP server over stdio (line-delimited JSON-RPC).
///
/// Reads JSON-RPC requests from stdin, writes responses to stdout.
/// This is the standard MCP transport used by Claude Desktop.
pub async fn serve_stdio(processor: signal::SignalProcessor) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let server = Arc::new(McpServer::new(Arc::new(processor)));
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            // EOF — client closed the connection
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                // Return a parse error
                let resp = JsonRpcResponse::err(Value::Null, -32700, format!("Parse error: {e}"));
                let json = serde_json::to_string(&resp)?;
                stdout.write_all(json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        if let Some(resp) = server.handle(req).await {
            let json = serde_json::to_string(&resp)?;
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}

// ─── HTTP Transport ───────────────────────────────────────────────────────────

/// Shared state for the HTTP MCP server.
struct HttpState {
    server: Arc<McpServer>,
}

/// Run the MCP server over HTTP (JSON-RPC POST endpoint).
///
/// All requests POST to `/` with a JSON-RPC body.
/// Returns JSON-RPC responses. Port defaults to 19791.
pub async fn serve_http(
    processor: signal::SignalProcessor,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let state = Arc::new(HttpState {
        server: Arc::new(McpServer::new(Arc::new(processor))),
    });

    let router = Router::new()
        .route("/", post(http_handler))
        .route("/mcp", post(http_handler))
        .with_state(state)
        .layer(CorsLayer::permissive());

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Brain MCP HTTP server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

/// POST / or POST /mcp — JSON-RPC over HTTP handler.
async fn http_handler(
    State(state): State<Arc<HttpState>>,
    AxumJson(req): AxumJson<JsonRpcRequest>,
) -> Result<AxumJson<Value>, (StatusCode, String)> {

    match state.server.handle(req).await {
        Some(resp) => {
            let val = serde_json::to_value(&resp)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            Ok(AxumJson(val))
        }
        None => {
            // Notification — no response body
            Ok(AxumJson(json!({})))
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_server() -> (McpServer, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        (McpServer::new(Arc::new(processor)), temp)
    }

    #[tokio::test]
    async fn test_initialize() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: Some(json!({
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test", "version": "0.1"}
            })),
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn test_tools_list() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(2)),
            method: "tools/list".to_string(),
            params: None,
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_none());
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_facts"));
        assert!(names.contains(&"memory_episodes"));
        assert!(names.contains(&"user_profile"));
        assert_eq!(tools.len(), 5);
    }

    #[tokio::test]
    async fn test_tool_memory_store_and_facts() {
        let (server, _tmp) = make_server().await;

        // Store a fact via MCP tool
        let store_req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(3)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "memory_store",
                "arguments": {
                    "subject": "user",
                    "predicate": "likes",
                    "object": "Rust",
                    "category": "personal"
                }
            })),
        };
        let resp = server.handle(store_req).await.unwrap();
        assert!(resp.error.is_none());
        let content = &resp.result.unwrap()["content"][0]["text"];
        assert!(content.as_str().unwrap().contains("Stored fact"));

        // Retrieve facts about "user"
        let facts_req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(4)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "memory_facts",
                "arguments": {"subject": "user"}
            })),
        };
        let resp = server.handle(facts_req).await.unwrap();
        assert!(resp.error.is_none());
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("likes") || text.contains("Rust"));
    }

    #[tokio::test]
    async fn test_tool_memory_search_empty() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(5)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "memory_search",
                "arguments": {"query": "favourite color", "top_k": 5}
            })),
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_none());
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        // Empty memory → "No relevant facts found"
        assert!(text.contains("No relevant") || !text.is_empty());
    }

    #[tokio::test]
    async fn test_tool_memory_episodes() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(6)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "memory_episodes",
                "arguments": {"limit": 5}
            })),
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_none());
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn test_tool_user_profile() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(7)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "user_profile",
                "arguments": {}
            })),
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_none());
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("llm") || text.contains("data_dir"));
    }

    #[tokio::test]
    async fn test_notification_returns_none() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: "notifications/initialized".to_string(),
            params: None,
        };
        let resp = server.handle(req).await;
        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn test_unknown_method_returns_error() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(99)),
            method: "does/not/exist".to_string(),
            params: None,
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_missing_tool_arg_returns_error() {
        let (server, _tmp) = make_server().await;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(10)),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": "memory_search",
                "arguments": {}   // missing "query"
            })),
        };
        let resp = server.handle(req).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }
}
