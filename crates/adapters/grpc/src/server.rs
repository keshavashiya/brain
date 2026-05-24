//! `serve()` entrypoint — binds the listener, wires the auth interceptor onto
//! each service, and runs the tonic server.

use std::{net::SocketAddr, sync::Arc};

use tonic::service::interceptor::InterceptedService;
use tonic::{transport::Server, Request};

use crate::agent_proto::agent_service_server::AgentServiceServer;
use crate::auth::auth_interceptor;
use crate::memory_proto::memory_service_server::MemoryServiceServer;
use crate::state::{AgentServiceImpl, MemoryServiceImpl};

/// Per-RPC payload cap (decoded + encoded). Tonic's default is 4 MiB on the
/// decoder and unlimited on the encoder; pinning both bounds an unauthenticated
/// peer from forcing the server to buffer arbitrarily large frames before the
/// auth interceptor runs, and bounds reply allocation for tools like
/// `GetFacts` whose response size scales with namespace count.
const MAX_GRPC_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

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

    // Set message-size caps on the inner *ServiceServer (the interceptor
    // wrapper doesn't expose the knobs) and then wrap. Two-step rather than
    // the `with_interceptor` shorthand for that reason.
    let memory_inner = MemoryServiceServer::new(MemoryServiceImpl::new(processor.clone()))
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
    let memory_svc = InterceptedService::new(memory_inner, {
        let keys = Arc::clone(&auth_keys);
        let rl = rate_limits.clone();
        move |req: Request<()>| auth_interceptor(req, &keys, rl.as_ref())
    });

    let agent_inner = AgentServiceServer::new(AgentServiceImpl::new(processor))
        .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
    let agent_svc = InterceptedService::new(agent_inner, {
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
