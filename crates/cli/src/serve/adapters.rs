//! Per-adapter bind+spawn helpers for `cmd_serve`.
//!
//! Each `try_bind_*` helper pre-binds the TCP port to surface
//! `EADDRINUSE` before the adapter takes ownership, then spawns the
//! adapter's serve future on the shared join set. On success it prints
//! the operator-facing "Synapse X → URL" line and returns `Ok(())`.
//! cmd_serve appends "X" to `bound_adapters` on success and the
//! returned error string to `failed_adapters` on failure.
//!
//! HTTP is the only adapter cmd_serve treats as critical (it serves the
//! health-check endpoint other CLI subcommands rely on). The caller
//! decides whether to `bail!` on its error — these helpers only report.

use std::sync::Arc;

pub(super) type WebhookHandlers =
    std::collections::HashMap<String, Arc<channel::transport::inbound::WebhookInboundTransport>>;

pub(super) async fn try_bind_http(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
    handlers: WebhookHandlers,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> Result<(), String> {
    match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
        Ok(_listener) => {
            println!("  Synapse HTTP  → http://{}:{}", host, port);
            let h = host.to_string();
            set.spawn(async move { httpadapter::serve(processor, handlers, &h, port).await });
            Ok(())
        }
        Err(e) => Err(format!("HTTP ({host}:{port}): {e}")),
    }
}

pub(super) async fn try_bind_ws(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> Result<(), String> {
    match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
        Ok(_listener) => {
            println!("  Synapse WS    → ws://{}:{}", host, port);
            let h = host.to_string();
            set.spawn(async move { wsadapter::serve(processor, &h, port).await });
            Ok(())
        }
        Err(e) => Err(format!("WS ({host}:{port}): {e}")),
    }
}

#[cfg(feature = "grpc")]
pub(super) async fn try_bind_grpc(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> Result<(), String> {
    match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
        Ok(_listener) => {
            println!("  Synapse gRPC  → {}:{}", host, port);
            let h = host.to_string();
            set.spawn(async move { grpcadapter::serve(processor, &h, port).await });
            Ok(())
        }
        Err(e) => Err(format!("gRPC ({host}:{port}): {e}")),
    }
}

pub(super) async fn try_bind_mcp(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> Result<(), String> {
    match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
        Ok(_listener) => {
            println!("  Synapse MCP   → http://{}:{}", host, port);
            let h = host.to_string();
            set.spawn(async move { mcp::serve_http(processor, &h, port).await });
            Ok(())
        }
        Err(e) => Err(format!("MCP ({host}:{port}): {e}")),
    }
}

/// Terminal Bridge gRPC server — exposes the wired
/// `processor.terminal_bridge()` over the wire. Caller has already
/// verified `adapters.terminal.enabled` and a wired bridge + identity
/// store before invoking; this helper just binds, wraps the auth, and
/// spawns the tonic server.
pub(super) async fn try_bind_terminal(
    bridge: Arc<terminal::TerminalBridge>,
    identity_store: Arc<dyn identity::IdentityStore>,
    api_keys: Vec<brain::ApiKeyConfig>,
    host: &str,
    port: u16,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> Result<(), String> {
    match tokio::net::TcpListener::bind(format!("{host}:{port}")).await {
        Ok(_listener) => {
            println!("  Synapse Term  → {}:{}", host, port);
            // Pair the wired identity store with this adapter's
            // api-key clone at spawn time — same shape as the
            // HTTP/WS/gRPC/MCP serve sites.
            let auth = terminal::TerminalAuth::new(identity_store, api_keys);
            let bridge_for_spawn = bridge.as_ref().clone().with_auth(auth);
            let host = host.to_string();
            set.spawn(async move {
                let addr = format!("{host}:{port}").parse().map_err(|e| {
                    anyhow::anyhow!("Terminal Bridge: invalid bind addr {host}:{port}: {e}")
                })?;
                tonic::transport::Server::builder()
                    .add_service(bridge_for_spawn.into_server())
                    .serve(addr)
                    .await
                    .map_err(|e| anyhow::anyhow!("Terminal Bridge serve: {e}"))
            });
            Ok(())
        }
        Err(e) => Err(format!("Terminal ({host}:{port}): {e}")),
    }
}
