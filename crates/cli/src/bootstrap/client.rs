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

/// Probe the daemon's health endpoint up to `attempts` times, returning the
/// base URL on the first success.
///
/// `/health` is a trivial, stateless handler, so a failed probe against a
/// daemon whose process is alive almost always means the HTTP listener hasn't
/// finished binding yet (the window right after `brain start`/restart, where
/// connections are *refused* and fail fast). Retrying rides out that bind
/// window so a transient race isn't misreported as a down/zombie daemon.
/// Callers that already know the daemon should be down should use the
/// single-shot [`detect_running_daemon`] to stay fast.
pub async fn probe_daemon_with_retries(
    config: &brain::BrainConfig,
    attempts: u32,
) -> Option<String> {
    for attempt in 0..attempts {
        if let Some(url) = detect_running_daemon(config).await {
            return Some(url);
        }
        if attempt + 1 < attempts {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A config whose HTTP adapter points at `port` on loopback.
    fn config_for(port: u16) -> brain::BrainConfig {
        let mut config = brain::BrainConfig::default();
        config.adapters.http.host = "127.0.0.1".to_string();
        config.adapters.http.port = port;
        config
    }

    /// Accept one connection on `listener`, read the request, and reply with a
    /// minimal HTTP/1.1 200 — enough for reqwest to see `/health` succeed.
    async fn serve_one_health(listener: TcpListener) {
        if let Ok((mut sock, _)) = listener.accept().await {
            let mut buf = [0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let body = br#"{"status":"ok","version":"test"}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.write_all(body).await;
            let _ = sock.flush().await;
        }
    }

    /// Bind a loopback listener on an OS-assigned port and return both.
    async fn bound_listener() -> (TcpListener, u16) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        (listener, port)
    }

    #[tokio::test]
    async fn probe_succeeds_against_live_health() {
        let (listener, port) = bound_listener().await;
        tokio::spawn(serve_one_health(listener));
        let config = config_for(port);

        let url = probe_daemon_with_retries(&config, 4).await;
        assert_eq!(url, Some(format!("http://127.0.0.1:{port}")));
    }

    #[tokio::test]
    async fn probe_returns_none_when_port_is_closed() {
        // Reserve a port, then drop the listener so nothing is bound there:
        // connections are refused and every attempt fails fast.
        let (listener, port) = bound_listener().await;
        drop(listener);
        let config = config_for(port);

        assert!(probe_daemon_with_retries(&config, 4).await.is_none());
    }

    /// The core fix: a daemon still binding its listener fails the first
    /// (single-shot) probe but is caught by the retrying probe — so status no
    /// longer misreports a soon-to-be-serving daemon as a zombie / stale PID.
    #[tokio::test]
    async fn retrying_probe_rides_out_a_delayed_bind() {
        let (reservation, port) = bound_listener().await;
        let config = config_for(port);

        // At t0 nothing is serving — the single-shot probe must fail.
        drop(reservation);
        assert!(
            detect_running_daemon(&config).await.is_none(),
            "single-shot probe should miss a not-yet-bound listener"
        );

        // Bind + serve `/health` shortly after, within the retry window.
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(350)).await;
            if let Ok(listener) = TcpListener::bind(format!("127.0.0.1:{port}")).await {
                serve_one_health(listener).await;
            }
        });

        let url = probe_daemon_with_retries(&config, 6).await;
        assert_eq!(url, Some(format!("http://127.0.0.1:{port}")));
    }
}
