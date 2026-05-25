//! # Brain HTTP REST API Adapter
//!
//! Exposes Brain's signal processing pipeline over HTTP using axum.
//!
//! ## Routes
//!
//! Unauthenticated:
//! - `GET    /`                         — redirect to `/ui`
//! - `GET    /health`                   — liveness probe
//! - `GET    /metrics`                  — Prometheus-format counters
//! - `GET    /ui`                       — embedded memory explorer web UI
//! - `GET    /openapi.json`             — OpenAPI 3.0 specification
//! - `GET    /api`                      — Swagger UI
//!
//! Authenticated (`Authorization: Bearer <api-key>`):
//! - `POST   /v1/signals`               — submit a signal *(scope: write)*
//! - `GET    /v1/signals/:id`           — retrieve cached signal response *(scope: read)*
//! - `POST   /v1/memory/search`         — semantic search over stored facts *(scope: read)*
//! - `GET    /v1/memory/facts`          — paginated fact list *(scope: read)*
//! - `GET    /v1/memory/namespaces`     — list namespaces with counts *(scope: read)*
//! - `GET    /v1/memory/export`         — full memory export *(scope: export)*
//! - `POST   /v1/memory/import`         — import facts/episodes *(scope: write)*
//! - `GET    /v1/schedules`             — list scheduled intents *(scope: read)*
//! - `DELETE /v1/schedules/:id`         — cancel a scheduled intent *(scope: write)*
//! - `GET    /v1/events`                — SSE stream of signal/notification/brain events *(scope: read)*
//! - `POST   /v1/webhooks/:id`          — inbound webhook delivery *(scope: write — or a configured signature verifier in lieu of Bearer auth)*
//!
//! ## Authentication
//!
//! A random API key is generated on `brain init` and printed to stdout.
//! Configure additional keys (and their permission scopes) under
//! `access.api_keys[]` in the config; `admin` is treated as an implicit
//! superset of every other scope. Webhook endpoints accept a verifier
//! (HMAC or Ed25519) as an alternative to Bearer auth — see
//! `adapters.http.webhook_verifiers` in the config.

pub mod auth;
pub mod handlers;
pub mod metrics;
pub mod middleware;
pub mod server;
pub mod state;
pub mod types;
pub mod validate;

// Re-export primary public types for convenience.
pub use server::{create_router, serve};
pub use state::AppState;
pub use types::HttpAdapterError;

#[cfg(test)]
mod tests;
