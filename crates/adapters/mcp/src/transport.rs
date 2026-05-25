use std::sync::Arc;

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json as AxumJson},
    routing::post,
    Router,
};
use serde_json::Value;

use super::{JsonRpcRequest, JsonRpcResponse, McpServer};

/// Extract `x-api-key` from `params._meta` of a JSON-RPC request.
pub(crate) fn extract_meta_key(req: &JsonRpcRequest) -> Option<&str> {
    req.params
        .as_ref()
        .and_then(|p| p.get("_meta"))
        .and_then(|m| m.get("x-api-key"))
        .and_then(Value::as_str)
}

/// Run the MCP server over stdio (line-delimited JSON-RPC).
///
/// ## Authentication
///
/// Stdio clients authenticate in one of two ways:
///
/// 1. **Per-request** — include `params._meta["x-api-key"]` in each JSON-RPC
///    request (same as HTTP header auth).
/// 2. **Session-level** — set the `BRAIN_API_KEY` env var to a valid API key.
///    The entire stdio session is then pre-authenticated; per-request `_meta`
///    is not required. This is the recommended approach for MCP clients
///    (e.g. Claude Code) that cannot inject custom `_meta` fields.
pub async fn serve_stdio(processor: signal::SignalProcessor) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let api_keys = processor.config().access.api_keys.clone();

    let session_authed = match std::env::var("BRAIN_API_KEY") {
        Ok(env_key) if !env_key.is_empty() => {
            // Issue 62: constant-time lookup; empty configured-list still
            // accepts (back-compat with no-auth dev mode).
            let valid =
                api_keys.is_empty() || brain::auth::find_key_ct(&api_keys, &env_key).is_some();
            if !valid {
                anyhow::bail!(
                    "BRAIN_API_KEY does not match any configured API key. \
                     Check your ~/.brain/config.yaml access.api_keys."
                );
            }
            true
        }
        _ => false,
    };

    let server = Arc::new(McpServer::new(Arc::new(processor), api_keys));
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        tracing::debug!(line = %trimmed.get(..120).unwrap_or(trimmed), "MCP stdio ← request");

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::err(Value::Null, -32700, format!("Parse error: {e}"));
                let json = serde_json::to_string(&resp)?;
                stdout.write_all(json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        };

        if !session_authed && !server.api_keys.is_empty() && !req.is_notification() {
            let meta_key = extract_meta_key(&req);
            let key_ok = meta_key.is_some_and(|k| server.validate_key(k));
            if !key_ok {
                let id = req.id.clone().unwrap_or(Value::Null);
                let resp = JsonRpcResponse::err(
                    id,
                    -32600,
                    "Unauthorized: provide valid x-api-key in params._meta \
                     or set BRAIN_API_KEY env var",
                );
                let json = serde_json::to_string(&resp)?;
                stdout.write_all(json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
                continue;
            }
        }

        if let Some(resp) = server.handle(req).await {
            let json = serde_json::to_string(&resp)?;
            tracing::debug!(len = json.len(), "MCP stdio → response");
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        } else {
            tracing::debug!(method = %trimmed.get(..80).unwrap_or(trimmed), "MCP stdio → notification (no response)");
        }
    }

    Ok(())
}

/// Shared state for the HTTP MCP server.
pub(crate) struct HttpState {
    pub(crate) server: Arc<McpServer>,
}

/// Run the MCP server over HTTP (JSON-RPC POST endpoint).
pub async fn serve_http(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let api_keys = processor.config().access.api_keys.clone();
    let state = Arc::new(HttpState {
        server: Arc::new(McpServer::new(processor, api_keys)),
    });

    let router = Router::new()
        .route("/", post(http_handler))
        .route("/mcp", post(http_handler))
        .with_state(state)
        .layer(brain::cors::localhost_cors())
        .layer(axum::extract::DefaultBodyLimit::max(1_048_576))
        .layer(tower::limit::ConcurrencyLimitLayer::new(100));

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Synapse MCP online at http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

/// POST / or POST /mcp — JSON-RPC over HTTP handler.
pub(crate) async fn http_handler(
    State(state): State<Arc<HttpState>>,
    headers: HeaderMap,
    AxumJson(req): AxumJson<JsonRpcRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    if !state.server.api_keys.is_empty() {
        let key = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                (
                    StatusCode::UNAUTHORIZED,
                    "Missing x-api-key header".to_string(),
                )
            })?;
        if !state.server.validate_key(key) {
            return Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string()));
        }
    }

    match state.server.handle(req).await {
        Some(resp) => {
            let val = serde_json::to_value(&resp)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
            Ok(AxumJson(val).into_response())
        }
        None => Ok(StatusCode::NO_CONTENT.into_response()),
    }
}
