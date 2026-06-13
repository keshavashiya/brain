//! Shared bootstrap — single source of truth for building a fully-wired SignalProcessor.
//!
//! Used by `brain serve`, `brain chat`, and `brain mcp` to eliminate
//! bootstrap duplication and ensure consistent backend wiring.
//!
//! Split by concern: [`processor`] (the `build_processor` entry point),
//! [`dispatcher`] (action backends), [`safety`] (audit/confirm/budget/
//! orchestrator), [`agents`] (delegation registry), and [`client`] (daemon
//! detection + MCP stdio proxy).

mod agents;
mod client;
mod dispatcher;
mod processor;
mod safety;

#[cfg(test)]
mod tests;

pub use client::{
    detect_running_daemon, probe_daemon_with_retries, proxy_mcp_stdio, require_daemon,
};
pub use processor::build_processor;
