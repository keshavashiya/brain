//! # Brain gRPC Adapter
//!
//! Exposes Brain's signal processing pipeline over gRPC using tonic.
//!
//! ## Services
//! - `MemoryService` — semantic memory search, store, list, and signal streaming
//! - `AgentService`  — agent connect, send signal, receive streaming updates
//!
//! ## Ports
//! - Default gRPC port: **19792**

mod auth;
mod errors;
mod events;
mod handlers;
mod helpers;
mod server;
mod state;

/// Types and server/client stubs generated from `proto/memory.proto`.
pub mod memory_proto {
    tonic::include_proto!("brain.memory");
}

/// Types and server/client stubs generated from `proto/agent.proto`.
pub mod agent_proto {
    tonic::include_proto!("brain.agent");
}

pub use errors::GrpcAdapterError;
pub use server::serve;
pub use state::{AgentServiceImpl, MemoryServiceImpl};

#[cfg(test)]
mod tests;
