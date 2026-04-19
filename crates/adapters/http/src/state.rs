//! Shared application state for HTTP handlers.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State as AxumState;
use channel::transport::inbound::WebhookInboundTransport;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::metrics::Metrics;

/// Maximum number of cached signal responses before eviction.
pub const CACHE_CAPACITY: usize = 1000;

/// Shared state for all HTTP handlers.
pub struct AppState {
    pub processor: Arc<signal::SignalProcessor>,
    /// Registry of active webhook transports (transport_id → transport).
    pub webhook_handlers: HashMap<String, Arc<WebhookInboundTransport>>,
    /// LRU cache: signal_id → SignalResponse. Bounded to `CACHE_CAPACITY` entries.
    pub cache: Mutex<lru::LruCache<Uuid, signal::SignalResponse>>,
    /// Configured API keys (loaded from BrainConfig).
    pub api_keys: Vec<brain_core::ApiKeyConfig>,
    /// Request counters and latency.
    pub metrics: Arc<Metrics>,
}

/// Axum extractor alias — use `AppStateRef` in handler signatures.
pub type AppStateRef = AxumState<Arc<AppState>>;
