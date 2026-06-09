//! Action backends and resilience primitives for Brain OS.
//!
//! Extracted from the CLI crate so they can be reused by adapters and tests.

pub mod error;
pub mod fetch;
pub mod memory;
pub mod messaging;
pub mod net;
pub mod resilience;
pub mod scheduling;
pub mod search;
pub mod security;

pub use error::BackendInitError;
pub use fetch::BasicUrlFetcher;
pub use memory::DefaultMemoryBackend;
pub use messaging::{
    json_escape, render_message_template, WebhookMessageBackend, DEFAULT_MESSAGE_BODY,
};
pub use net::NetDiagnostics;
pub use resilience::{resilient_send, Breaker, CircuitBreaker, ResilientSendError};
pub use scheduling::DefaultSchedulingBackend;
pub use search::{
    CustomSearchBackend, DuckDuckGoSearchBackend, SearxngSearchBackend, TavilySearchBackend,
};
pub use security::ConfigSecurityAuditor;
