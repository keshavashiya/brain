//! `serve()` entrypoint — binds the listener, wires the auth interceptor onto
//! each service, and runs the tonic server.

use std::{net::SocketAddr, sync::Arc};

use tonic::{transport::Server, Request};

use crate::agent_proto::agent_service_server::AgentServiceServer;
use crate::auth::auth_interceptor;
use crate::memory_proto::memory_service_server::MemoryServiceServer;
use crate::state::{AgentServiceImpl, MemoryServiceImpl};

/// Start the gRPC server, binding to `host:port`.
///
/// Registers both `MemoryService` and `AgentService`.
/// All requests are authenticated via `x-api-key` or `authorization` metadata.
/// Blocks until the server shuts down.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{host}:{port}").parse()?;

    let auth_keys = Arc::new(processor.config().access.api_keys.clone());
    let rate_limits = processor.client_rate_limits().cloned();

    let memory_svc =
        MemoryServiceServer::with_interceptor(MemoryServiceImpl::new(processor.clone()), {
            let keys = Arc::clone(&auth_keys);
            let rl = rate_limits.clone();
            move |req: Request<()>| auth_interceptor(req, &keys, rl.as_ref())
        });
    let agent_svc = AgentServiceServer::with_interceptor(AgentServiceImpl::new(processor), {
        let keys = Arc::clone(&auth_keys);
        let rl = rate_limits.clone();
        move |req: Request<()>| auth_interceptor(req, &keys, rl.as_ref())
    });

    tracing::info!("Synapse gRPC online at {addr}");

    Server::builder()
        .add_service(memory_svc)
        .add_service(agent_svc)
        .serve(addr)
        .await?;

    Ok(())
}
