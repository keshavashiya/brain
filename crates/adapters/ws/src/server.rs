//! `serve()` entrypoint — accepts TCP connections and spawns one
//! `connection::handle_connection` task per accepted socket.

use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use brain::ApiKeyConfig;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::connection::handle_connection;
use crate::protocol::{ConnectionInfo, Connections};

/// Start the WebSocket server, binding to `host:port`.
///
/// The configured `api_keys` are used to authenticate each new connection's
/// initial handshake message.  Pass an empty `Vec` to disable auth (not
/// recommended in production).
///
/// Accepts concurrent connections. Each connection is handled in its own
/// tokio task. Blocks until the listener errors.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let api_keys: Arc<Vec<ApiKeyConfig>> = Arc::new(processor.config().access.api_keys.clone());
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Synapse WebSocket online at ws://{addr}");
    let connections: Connections = Arc::new(Mutex::new(HashMap::new()));

    loop {
        let (tcp_stream, peer) = listener.accept().await?;
        let conn_id = Uuid::new_v4();

        let proc = Arc::clone(&processor);
        let conns = Arc::clone(&connections);
        let keys = Arc::clone(&api_keys);

        // Register connection before spawning so the count is accurate
        conns
            .lock()
            .await
            .insert(conn_id, ConnectionInfo { id: conn_id, peer });

        tokio::spawn(async move {
            // Limit max message size to 1 MB to prevent memory exhaustion
            let mut ws_config =
                tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
            ws_config.max_message_size = Some(1_048_576);
            ws_config.max_frame_size = Some(1_048_576);
            match tokio_tungstenite::accept_async_with_config(tcp_stream, Some(ws_config)).await {
                Ok(ws_stream) => {
                    tracing::info!(
                        conn_id = %conn_id,
                        peer = %peer,
                        "WebSocket connection established"
                    );
                    handle_connection(ws_stream, conn_id, proc, &keys).await;
                }
                Err(e) => {
                    tracing::warn!(
                        conn_id = %conn_id,
                        peer = %peer,
                        "WebSocket handshake failed: {e}"
                    );
                }
            }

            // Deregister on disconnect (whether handshake failed or connection closed)
            conns.lock().await.remove(&conn_id);
            tracing::info!(conn_id = %conn_id, peer = %peer, "WebSocket connection closed");
        });
    }
}
