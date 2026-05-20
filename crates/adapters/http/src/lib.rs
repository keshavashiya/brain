//! # Brain HTTP REST API Adapter
//!
//! Exposes Brain's signal processing pipeline over HTTP using axum.
//!
//! ## Routes
//! - `GET  /health`             — health check (no auth required)
//! - `GET  /metrics`            — Prometheus-format counters (no auth required)
//! - `GET  /ui`                 — embedded memory explorer web UI (no auth required)
//! - `GET  /openapi.json`       — OpenAPI 3.0 specification (no auth required)
//! - `GET  /api`                 — Swagger UI (no auth required)
//! - `POST /v1/signals`         — submit a signal (requires write)
//! - `GET  /v1/signals/:id`     — retrieve cached signal response (requires read)
//! - `POST /v1/memory/search`   — semantic search over stored facts (requires read)
//! - `GET  /v1/memory/facts`    — list all semantic facts (requires read)
//! - `GET  /v1/events`          — SSE stream of signal events + proactive notifications (requires read)
//!
//! ## Authentication
//! All `/v1/*` routes require `Authorization: Bearer <api-key>` header.
//! A random key is generated on `brain init` and printed to stdout.

pub mod auth;
pub mod handlers;
pub mod metrics;
pub mod middleware;
pub mod server;
pub mod state;
pub mod types;

// Re-export primary public types for convenience.
pub use server::{create_router, serve};
pub use state::AppState;
pub use types::HttpAdapterError;

#[cfg(test)]
mod tests;
