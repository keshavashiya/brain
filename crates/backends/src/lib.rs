//! Action backends and resilience primitives for Brain OS.
//!
//! Extracted from the CLI crate so they can be reused by adapters and tests.

pub mod memory;
pub mod messaging;
pub mod resilience;
pub mod scheduling;
pub mod search;

pub use memory::DefaultMemoryBackend;
pub use messaging::{
    json_escape, render_message_template, WebhookMessageBackend, DEFAULT_MESSAGE_BODY,
};
pub use resilience::{resilient_send, CircuitBreaker};
pub use scheduling::DefaultSchedulingBackend;
pub use search::{CustomSearchBackend, SearxngSearchBackend, TavilySearchBackend};
