//! Shared bootstrap — single source of truth for building a fully-wired SignalProcessor.
//!
//! Used by `brain serve`, `brain chat`, and `brain mcp` to eliminate
//! bootstrap duplication and ensure consistent backend wiring.

use std::sync::Arc;

use crate::encryption::{resolve_encryptor, resolve_llm_api_key};
use backends::*;

/// Build a fully-wired `SignalProcessor` from config.
///
/// This is the canonical bootstrap path. All CLI commands that need a processor
/// should call this instead of wiring backends individually.
///
/// What gets wired:
/// - Encryption (if enabled)
/// - Action dispatcher with memory backend
/// - Web search backend (searxng / tavily / custom)
/// - Scheduling backend
/// - Messaging backend (webhooks)
///
/// What is NOT wired here (caller-specific):
/// - Notification router (only needed for `brain serve`)
/// - Background tasks (consolidation, proactivity — only `brain serve`)
pub async fn build_processor(
    config: &brain_core::BrainConfig,
) -> anyhow::Result<signal::SignalProcessor> {
    let encryptor = resolve_encryptor(config)?;
    let mut processor =
        signal::SignalProcessor::new_with_encryptor(config.clone(), encryptor).await?;

    let action_dispatcher = build_action_dispatcher(config, &processor)?;
    processor = processor.with_action_dispatcher(action_dispatcher);

    Ok(processor)
}

/// Build the action dispatcher with all configured backends.
fn build_action_dispatcher(
    config: &brain_core::BrainConfig,
    processor: &signal::SignalProcessor,
) -> anyhow::Result<cortex::actions::ActionDispatcher> {
    let embedding_dim = processor.embedding_dim();
    let llm_api_key = resolve_llm_api_key(config);
    let embedder = Arc::new(tokio::sync::Mutex::new(hippocampus::Embedder::from_config(
        &config.llm.provider,
        &config.llm.base_url,
        &config.embedding.model,
        &llm_api_key,
    )));

    let action_backend = Arc::new(DefaultMemoryBackend {
        semantic: processor.semantic().cloned(),
        embedder,
        embedding_dim,
    });

    let action_config = cortex::actions::ActionConfig {
        command_allowlist: config.security.exec_allowlist.clone(),
        command_timeout_secs: config.security.exec_timeout_seconds as u64,
        enable_web_search: config.actions.web_search.enabled,
        enable_scheduling: config.actions.scheduling.enabled,
        enable_channel_send: config.actions.messaging.enabled,
        web_search_top_k: config.actions.web_search.default_top_k,
    };

    let mut dispatcher =
        cortex::actions::ActionDispatcher::with_memory_backend(action_config, action_backend);
    dispatcher.set_namespace("personal");

    // ── Web search backend ───────────────────────────────────────────────
    if config.actions.web_search.enabled {
        let ws = &config.actions.web_search;
        let timeout = ws.timeout_ms;
        let endpoint = ws.endpoint.trim();
        let res = &config.actions.resilience;

        let backend_result: Result<
            Option<Arc<dyn cortex::actions::WebSearchBackend>>,
            anyhow::Error,
        > = match ws.provider {
            brain_core::config::WebSearchProvider::Searxng => {
                let ep = if endpoint.is_empty() {
                    "http://localhost:8888"
                } else {
                    endpoint
                };
                SearxngSearchBackend::new(ep, timeout, res).map(|b| Some(Arc::new(b) as _))
            }
            brain_core::config::WebSearchProvider::Tavily => {
                let api_key = ws.api_key.trim();
                if api_key.is_empty() {
                    tracing::warn!(
                            "actions.web_search.provider=tavily but api_key is empty; backend not configured"
                        );
                    Ok(None)
                } else {
                    let ep = if endpoint.is_empty() {
                        "https://api.tavily.com"
                    } else {
                        endpoint
                    };
                    TavilySearchBackend::new(ep, api_key, timeout, res)
                        .map(|b| Some(Arc::new(b) as _))
                }
            }
            brain_core::config::WebSearchProvider::Custom => {
                if endpoint.is_empty() {
                    tracing::warn!(
                            "actions.web_search.provider=custom but endpoint is empty; backend not configured"
                        );
                    Ok(None)
                } else {
                    CustomSearchBackend::new(endpoint, timeout, res).map(|b| Some(Arc::new(b) as _))
                }
            }
        };

        match backend_result {
            Ok(Some(backend)) => {
                tracing::info!(
                    provider = %serde_json::to_string(&ws.provider).unwrap_or_default().trim_matches('"'),
                    "Web search backend configured"
                );
                dispatcher = dispatcher.with_web_search_backend(backend);
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("Web search backend init failed: {e}"),
        }
    }

    // ── Scheduling backend ───────────────────────────────────────────────
    if config.actions.scheduling.enabled {
        let backend = DefaultSchedulingBackend {
            db: processor.episodic().pool().clone(),
            mode: config.actions.scheduling.mode.clone(),
        };
        dispatcher = dispatcher.with_scheduling_backend(Arc::new(backend));
    }

    // ── Messaging backend ────────────────────────────────────────────────
    if config.actions.messaging.enabled {
        if config.actions.messaging.channels.is_empty() {
            tracing::warn!(
                "actions.messaging.enabled=true but no channel webhook mappings are configured"
            );
        } else {
            let res = &config.actions.resilience;
            match WebhookMessageBackend::new(
                &config.actions.messaging.channels,
                config.actions.messaging.timeout_ms,
                res,
            ) {
                Ok(backend) => {
                    tracing::info!("Message backend configured");
                    dispatcher = dispatcher.with_message_backend(Arc::new(backend));
                }
                Err(e) => tracing::warn!("Message backend init failed: {e}"),
            }
        }
    }

    Ok(dispatcher)
}

/// Check if a Brain daemon is already running by probing its health endpoint.
///
/// Returns the base URL (e.g. `http://127.0.0.1:19789`) if the daemon is alive.
pub async fn detect_running_daemon(config: &brain_core::BrainConfig) -> Option<String> {
    let host = &config.adapters.http.host;
    let port = config.adapters.http.port;
    let base_url = format!("http://{host}:{port}");
    let health_url = format!("{base_url}/health");

    let client = reqwest::Client::builder()
        .timeout(brain_core::timeouts::HEALTH_CHECK)
        .build()
        .ok()?;

    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!(url = %base_url, "Detected running Brain daemon");
            Some(base_url)
        }
        Ok(resp) => {
            tracing::debug!(status = %resp.status(), "Daemon health check returned non-success");
            None
        }
        Err(e) => {
            tracing::debug!(error = %e, "Daemon health check failed");
            None
        }
    }
}

/// Proxy MCP stdio through a running daemon's MCP HTTP transport.
///
/// Reads JSON-RPC lines from stdin, forwards each as an HTTP POST to the
/// daemon's MCP endpoint, and writes the response to stdout. This ensures
/// that the daemon's single SignalProcessor handles all requests — no
/// ruvector lock contention, no memory isolation.
pub async fn proxy_mcp_stdio(
    mcp_url: &str,
    config: &brain_core::BrainConfig,
) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let client = reqwest::Client::builder()
        .timeout(brain_core::timeouts::DAEMON_SETUP)
        .build()?;

    // Resolve API key for the x-api-key header.
    let api_key = std::env::var("BRAIN_API_KEY").unwrap_or_default();
    let api_key = if api_key.is_empty() {
        config
            .access
            .api_keys
            .first()
            .map(|k| k.key.clone())
            .unwrap_or_default()
    } else {
        api_key
    };

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // EOF
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Forward the raw JSON-RPC request to the daemon's MCP HTTP endpoint.
        let resp = client
            .post(mcp_url)
            .header("Content-Type", "application/json")
            .header("x-api-key", &api_key)
            .body(trimmed.to_string())
            .send()
            .await;

        match resp {
            Ok(r) => {
                // 204 No Content = notification ack — nothing to forward to the
                // stdio client (JSON-RPC spec: no response for notifications).
                if r.status() == reqwest::StatusCode::NO_CONTENT {
                    continue;
                }
                let body = r.text().await.unwrap_or_default();
                if !body.is_empty() {
                    stdout.write_all(body.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
            }
            Err(e) => {
                // Connection error — daemon may have stopped. Return a JSON-RPC error.
                let err_resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32603,
                        "message": format!("Daemon proxy error: {e}")
                    }
                });
                let json = serde_json::to_string(&err_resp)?;
                stdout.write_all(json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }
    }

    Ok(())
}
