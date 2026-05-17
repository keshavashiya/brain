//! HTTP server construction and lifecycle.

use std::collections::HashMap;
use std::{net::SocketAddr, sync::Arc};

use axum::{routing::delete, Router};
use channel::transport::inbound::WebhookInboundTransport;
use tokio::sync::Mutex;

use crate::handlers::*;
use crate::metrics::Metrics;
use crate::state::{AppState, CACHE_CAPACITY};

/// Build the axum router with all routes.
///
/// `api_keys` is taken from `BrainConfig.access.api_keys` by the caller.
/// When `cors_enabled` is true, CORS is restricted to localhost origins (Brain is a local daemon).
/// When false, no CORS layer is applied (useful for reverse-proxy setups that handle CORS externally).
pub fn create_router(
    processor: Arc<signal::SignalProcessor>,
    webhook_handlers: HashMap<String, Arc<WebhookInboundTransport>>,
    api_keys: Vec<brain_core::ApiKeyConfig>,
    cors_enabled: bool,
) -> Router {
    let rate_limits = processor.client_rate_limits().cloned();
    let state = Arc::new(AppState {
        processor,
        webhook_handlers,
        cache: Mutex::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(CACHE_CAPACITY).unwrap(),
        )),
        api_keys,
        metrics: Arc::new(Metrics::default()),
    });

    // Limit concurrent /v1/* requests (100 in-flight per instance)
    // Limit request body size to 1 MB to prevent memory exhaustion
    let mut v1_routes = Router::new()
        .route("/v1/signals", post(post_signal_handler))
        .route("/v1/signals/:id", get(get_signal_handler))
        .route("/v1/memory/search", post(search_memory_handler))
        .route("/v1/memory/facts", get(get_facts_handler))
        .route("/v1/memory/namespaces", get(get_namespaces_handler))
        .route("/v1/memory/export", get(export_memory_handler))
        .route("/v1/memory/import", post(import_memory_handler))
        .route("/v1/schedules", get(list_schedules_handler))
        .route("/v1/schedules/:id", delete(cancel_schedule_handler))
        .route("/v1/events", get(sse_events_handler))
        .route("/v1/webhooks/:id", post(post_webhook_handler))
        .layer(axum::extract::DefaultBodyLimit::max(1_048_576))
        .layer(tower::limit::ConcurrencyLimitLayer::new(100));
    if let Some(registry) = rate_limits {
        v1_routes = v1_routes.layer(axum::middleware::from_fn(move |req, next| {
            let registry = registry.clone();
            async move { crate::middleware::rate_limit(registry, req, next).await }
        }));
    }

    let router = Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/ui", get(ui_handler))
        .route("/openapi.json", get(openapi_handler))
        .route("/api", get(swagger_ui_handler))
        .merge(v1_routes)
        .with_state(state);

    if cors_enabled {
        router.layer(brain_core::cors::localhost_cors())
    } else {
        router
    }
}

use axum::routing::{get, post};

/// Start the HTTP server, binding to `host:port`.
///
/// Blocks until the server shuts down.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    webhook_handlers: HashMap<String, Arc<WebhookInboundTransport>>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let cors_enabled = processor.config().adapters.http.cors;
    let api_keys = processor.config().access.api_keys.clone();
    let router = create_router(processor, webhook_handlers, api_keys, cors_enabled);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Synapse HTTP online at http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
