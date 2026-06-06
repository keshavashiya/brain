//! Daemon-client helpers: a process-wide HTTP client, health-probe based
//! daemon detection, the `require_daemon` retry wrapper, and the MCP stdio
//! proxy that forwards JSON-RPC through a running daemon.

/// Shared `reqwest::Client` for CLI control-plane calls.
///
/// `detect_running_daemon` runs up to four times per `require_daemon`
/// invocation, and `proxy_mcp_stdio` previously built a second client
/// for the entire MCP proxy session. Each `Client::builder().build()`
/// constructs a fresh connection pool — wasteful when we're hitting
/// the same loopback origin every time.
///
/// One process-wide client, no `.timeout()` baked in (per-call
/// `RequestBuilder::timeout` overrides differ between the
/// health-check path and the long-lived MCP proxy path).
fn shared_client() -> Option<&'static reqwest::Client> {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    if let Some(c) = CLIENT.get() {
        return Some(c);
    }
    match reqwest::Client::builder().build() {
        Ok(c) => Some(CLIENT.get_or_init(|| c)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to build shared reqwest client");
            None
        }
    }
}

/// Check if a Brain daemon is already running by probing its health endpoint.
///
/// Returns the base URL (e.g. `http://127.0.0.1:19789`) if the daemon is alive.
pub async fn detect_running_daemon(config: &brain::BrainConfig) -> Option<String> {
    let host = &config.adapters.http.host;
    let port = config.adapters.http.port;
    let base_url = format!("http://{host}:{port}");
    let health_url = format!("{base_url}/health");

    let client = shared_client()?;

    match client
        .get(&health_url)
        .timeout(brain::timeouts::HEALTH_CHECK)
        .send()
        .await
    {
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

/// Require a running Brain daemon, returning its base URL or a clear error.
///
/// Retries a few times to handle the case where the daemon is still booting.
/// This is the canonical way for CLI commands to ensure they don't create
/// their own SignalProcessor (which would cause RuVector lock contention
/// and memory isolation).
pub async fn require_daemon(config: &brain::BrainConfig) -> anyhow::Result<String> {
    let max_attempts = 4;
    for attempt in 0..max_attempts {
        if let Some(url) = detect_running_daemon(config).await {
            return Ok(url);
        }
        if attempt < max_attempts - 1 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    let port = config.adapters.http.port;
    anyhow::bail!(
        "No running Brain daemon detected (expected at http://127.0.0.1:{port}).\n\
         Run `brain start` to wake the daemon first.\n\
         All CLI commands require a running daemon to ensure a single shared SignalProcessor."
    )
}

/// Proxy MCP stdio through a running daemon's MCP HTTP transport.
///
/// Reads JSON-RPC lines from stdin, forwards each as an HTTP POST to the
/// daemon's MCP endpoint, and writes the response to stdout. This ensures
/// that the daemon's single SignalProcessor handles all requests — no
/// ruvector lock contention, no memory isolation.
pub async fn proxy_mcp_stdio(mcp_url: &str, config: &brain::BrainConfig) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let client = shared_client()
        .ok_or_else(|| anyhow::anyhow!("failed to build shared HTTP client"))?
        .clone();
    // Per-request timeout applied at .send() time below — the shared
    // client itself stays timeout-free so health-check (short) and
    // MCP-proxy (long) sites can each set their own without
    // conflicting at the client level.

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
            .timeout(brain::timeouts::DAEMON_SETUP)
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
