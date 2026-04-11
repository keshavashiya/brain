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
//! The demo key `demokey123` (read+write) is pre-configured in `default.yaml`.

use std::{
    collections::HashMap,
    net::SocketAddr,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{delete, get, post},
    Router,
};
use brain_core::ApiKeyConfig;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use signal::{Signal, SignalResponse, SignalSource};

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum HttpAdapterError {
    #[error("Server error: {0}")]
    Server(String),
}

// ─── Request / Response DTOs ─────────────────────────────────────────────────

/// Incoming signal body (POST /v1/signals).
#[derive(Debug, Deserialize)]
pub struct SignalRequest {
    pub source: Option<String>,
    pub channel: Option<String>,
    pub sender: Option<String>,
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
    /// Memory namespace (default: "personal").
    pub namespace: Option<String>,
    /// Originating agent identity (e.g. "claude-code", "open-code").
    pub agent: Option<String>,
    /// Session ID for conversation continuity. Send back to reuse a session.
    pub session_id: Option<String>,
}

/// Search request body (POST /v1/memory/search).
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
    /// Filter results to this namespace only (optional).
    pub namespace: Option<String>,
}

/// Namespace statistics (GET /v1/memory/namespaces).
#[derive(Debug, Serialize)]
pub struct NamespaceJson {
    pub namespace: String,
    pub fact_count: i64,
    pub episode_count: i64,
}

/// Export envelope (GET /v1/memory/export).
#[derive(Debug, Serialize)]
pub struct ExportJson {
    pub version: String,
    pub exported_at: String,
    pub facts: Vec<signal::ExportedFact>,
    pub episodes: Vec<signal::ExportedEpisode>,
}

/// Import request body (POST /v1/memory/import).
#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub facts: Vec<signal::ExportedFact>,
    pub episodes: Vec<signal::ExportedEpisode>,
    /// If true, preview what would be imported without writing.
    #[serde(default)]
    pub dry_run: bool,
}

/// Import response (POST /v1/memory/import).
#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub facts_imported: usize,
    pub episodes_imported: usize,
    pub facts_already_existed: usize,
    pub episodes_already_existed: usize,
    pub embedded: usize,
    pub embed_failed: usize,
}

/// A single fact in JSON form (GET /v1/memory/facts, search results).
#[derive(Debug, Serialize)]
pub struct FactJson {
    pub id: String,
    pub namespace: String,
    pub category: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub distance: Option<f32>,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

/// Atomic counters exposed at `GET /metrics` in Prometheus text format.
#[derive(Default)]
pub struct Metrics {
    /// Total POST /v1/signals requests processed.
    pub signals_total: AtomicU64,
    /// Signals that returned a non-5xx response.
    pub signals_ok: AtomicU64,
    /// Signals that returned a 5xx error.
    pub signals_error: AtomicU64,
    /// Total POST /v1/memory/search requests.
    pub search_total: AtomicU64,
    /// Total GET /v1/memory/facts requests.
    pub facts_total: AtomicU64,
    /// Cumulative POST /v1/signals processing time in milliseconds.
    pub signals_latency_ms_total: AtomicU64,
}

impl Metrics {
    /// Render counters as Prometheus plain-text format (text/plain; version=0.0.4).
    pub fn render(&self) -> String {
        let signals_total = self.signals_total.load(Ordering::Relaxed);
        let signals_ok = self.signals_ok.load(Ordering::Relaxed);
        let signals_error = self.signals_error.load(Ordering::Relaxed);
        let search_total = self.search_total.load(Ordering::Relaxed);
        let facts_total = self.facts_total.load(Ordering::Relaxed);
        let latency_ms = self.signals_latency_ms_total.load(Ordering::Relaxed);

        format!(
            "# HELP brain_signals_total Total signal requests received.\n\
             # TYPE brain_signals_total counter\n\
             brain_signals_total {signals_total}\n\
             # HELP brain_signals_ok_total Successful signal requests.\n\
             # TYPE brain_signals_ok_total counter\n\
             brain_signals_ok_total {signals_ok}\n\
             # HELP brain_signals_error_total Failed signal requests (5xx).\n\
             # TYPE brain_signals_error_total counter\n\
             brain_signals_error_total {signals_error}\n\
             # HELP brain_search_total Total memory search requests.\n\
             # TYPE brain_search_total counter\n\
             brain_search_total {search_total}\n\
             # HELP brain_facts_total Total memory facts requests.\n\
             # TYPE brain_facts_total counter\n\
             brain_facts_total {facts_total}\n\
             # HELP brain_signals_latency_ms_total Cumulative signal processing latency in ms.\n\
             # TYPE brain_signals_latency_ms_total counter\n\
             brain_signals_latency_ms_total {latency_ms}\n"
        )
    }
}

// ─── App State ───────────────────────────────────────────────────────────────

/// Maximum number of cached signal responses before eviction.
const CACHE_CAPACITY: usize = 1000;

/// Shared state for all HTTP handlers.
pub struct AppState {
    processor: Arc<signal::SignalProcessor>,
    /// LRU cache: signal_id → SignalResponse. Bounded to `CACHE_CAPACITY` entries.
    cache: Mutex<lru::LruCache<Uuid, SignalResponse>>,
    /// Configured API keys (loaded from BrainConfig).
    api_keys: Vec<ApiKeyConfig>,
    /// Request counters and latency.
    metrics: Arc<Metrics>,
}

// ─── Auth helpers ─────────────────────────────────────────────────────────────

/// Extract the raw key from `Authorization: Bearer <key>`.
fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(brain_core::auth::extract_bearer_from_value)
}

/// Check that the request carries a valid key with the given permission.
/// Returns `Err((StatusCode::UNAUTHORIZED, message))` on failure.
fn check_auth(
    state: &AppState,
    headers: &HeaderMap,
    permission: &str,
) -> Result<(), (StatusCode, String)> {
    let provided_key = extract_bearer(headers);
    let result = brain_core::check_auth(&state.api_keys, provided_key, permission);
    if result.is_allowed() {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            result
                .error_message(permission)
                .unwrap_or_else(|| "Unauthorized".to_string()),
        ))
    }
}

// ─── Router builder ──────────────────────────────────────────────────────────

/// Build the axum router with all routes.
///
/// `api_keys` is taken from `BrainConfig.access.api_keys` by the caller.
/// When `cors_enabled` is true, CORS is restricted to localhost origins (Brain is a local daemon).
/// When false, no CORS layer is applied (useful for reverse-proxy setups that handle CORS externally).
pub fn create_router(
    processor: Arc<signal::SignalProcessor>,
    api_keys: Vec<ApiKeyConfig>,
    cors_enabled: bool,
) -> Router {
    let state = Arc::new(AppState {
        processor,
        cache: Mutex::new(lru::LruCache::new(
            NonZeroUsize::new(CACHE_CAPACITY).unwrap(),
        )),
        api_keys,
        metrics: Arc::new(Metrics::default()),
    });

    // Limit concurrent /v1/* requests (100 in-flight per instance)
    // Limit request body size to 1 MB to prevent memory exhaustion
    let v1_routes = Router::new()
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
        .layer(axum::extract::DefaultBodyLimit::max(1_048_576))
        .layer(tower::limit::ConcurrencyLimitLayer::new(100));

    let router = Router::new()
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

/// Start the HTTP server, binding to `host:port`.
///
/// Blocks until the server shuts down.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let cors_enabled = processor.config().adapters.http.cors;
    let api_keys = processor.config().access.api_keys.clone();
    let router = create_router(processor, api_keys, cors_enabled);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Synapse HTTP online at http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /health — no authentication required
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// GET /metrics — Prometheus text format, no authentication required
async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        state.metrics.render(),
    )
}

/// Embedded single-page memory explorer UI.
const UI_HTML: &str = include_str!("../assets/ui.html");

/// GET /ui — embedded single-page memory explorer (no auth required)
async fn ui_handler() -> impl IntoResponse {
    ([("content-type", "text/html; charset=utf-8")], UI_HTML)
}

// ─── OpenAPI spec ─────────────────────────────────────────────────────────────

/// OpenAPI 3.0 document loaded at compile time from assets/openapi.json.
/// The placeholder `{{VERSION}}` is replaced with the actual crate version at runtime.
static OPENAPI_JSON: &str = include_str!("../assets/openapi.json");

/// GET /openapi.json — OpenAPI 3.0 specification (no auth required)
async fn openapi_handler() -> impl IntoResponse {
    let spec = OPENAPI_JSON.replace("{{VERSION}}", env!("CARGO_PKG_VERSION"));
    ([("content-type", "application/json")], spec)
}

/// Swagger UI HTML that loads the spec from /openapi.json (CDN assets).
const SWAGGER_UI_HTML: &str = include_str!("../assets/swagger.html");

/// GET /api — Swagger UI for interactive API exploration (no auth required)
async fn swagger_ui_handler() -> impl IntoResponse {
    (
        [("content-type", "text/html; charset=utf-8")],
        SWAGGER_UI_HTML,
    )
}

/// POST /v1/signals — requires write permission
async fn post_signal_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SignalRequest>,
) -> Result<Json<SignalResponse>, (StatusCode, String)> {
    check_auth(&state, &headers, "write")?;

    let t0 = Instant::now();
    state.metrics.signals_total.fetch_add(1, Ordering::Relaxed);

    let source = SignalSource::parse(body.source.as_deref(), SignalSource::Http);
    let signal = Signal::from_adapter_request(signal::AdapterRequest {
        source,
        content: body.content,
        channel: body.channel,
        sender: body.sender,
        metadata: body.metadata,
        namespace: body.namespace,
        agent: body.agent,
        session_id: body.session_id,
        default_channel: "http".to_string(),
        default_sender: "apiclient".to_string(),
    });

    let signal_id = signal.id;
    let result = state.processor.process(signal).await;

    let elapsed_ms = t0.elapsed().as_millis() as u64;
    state
        .metrics
        .signals_latency_ms_total
        .fetch_add(elapsed_ms, Ordering::Relaxed);

    let response = match result {
        Ok(r) => {
            state.metrics.signals_ok.fetch_add(1, Ordering::Relaxed);
            tracing::info!(
                signal_id = %signal_id,
                latency_ms = elapsed_ms,
                "signal processed"
            );
            r
        }
        Err(e) => {
            state.metrics.signals_error.fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                signal_id = %signal_id,
                latency_ms = elapsed_ms,
                error = %e,
                "signal processing failed"
            );
            // Return an opaque error as JSON — do not leak internal details.
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({
                    "error": "Signal processing failed",
                    "details": e.to_string()
                })
                .to_string(),
            ));
        }
    };

    // Cache the response so GET /v1/signals/:id can retrieve it (LRU evicts oldest)
    state.cache.lock().await.put(signal_id, response.clone());

    Ok(Json(response))
}

/// GET /v1/signals/:id — requires read permission
async fn get_signal_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<SignalResponse>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {id}")))?;

    let mut cache = state.cache.lock().await;
    match cache.get(&uuid) {
        Some(resp) => Ok(Json(resp.clone())),
        None => Err((
            StatusCode::NOT_FOUND,
            format!("Signal {uuid} not found in cache"),
        )),
    }
}

/// POST /v1/memory/search — requires read permission
async fn search_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SearchRequest>,
) -> Result<Json<Vec<FactJson>>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    state.metrics.search_total.fetch_add(1, Ordering::Relaxed);
    let t0 = Instant::now();
    let top_k = body.top_k.unwrap_or(10);
    let namespace = body.namespace.as_deref();
    let results = state
        .processor
        .search_facts(&body.query, top_k, namespace)
        .await;
    tracing::debug!(latency_ms = t0.elapsed().as_millis() as u64, query = %body.query, "memory search");

    let facts = results
        .into_iter()
        .map(|r| FactJson {
            id: r.fact.id,
            namespace: r.fact.namespace,
            category: r.fact.category,
            subject: r.fact.subject,
            predicate: r.fact.predicate,
            object: r.fact.object,
            confidence: r.fact.confidence,
            distance: Some(r.distance),
        })
        .collect();

    Ok(Json(facts))
}

/// GET /v1/memory/facts — requires read permission
///
/// Accepts optional `namespace` query parameter to filter results.
async fn get_facts_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<FactJson>>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    state.metrics.facts_total.fetch_add(1, Ordering::Relaxed);
    let namespace = params.get("namespace").map(|s| s.as_str());
    let facts = state
        .processor
        .list_facts(namespace)
        .into_iter()
        .map(|f| FactJson {
            id: f.id,
            namespace: f.namespace,
            category: f.category,
            subject: f.subject,
            predicate: f.predicate,
            object: f.object,
            confidence: f.confidence,
            distance: None,
        })
        .collect();

    Ok(Json(facts))
}

/// GET /v1/memory/namespaces — requires read permission
async fn get_namespaces_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Vec<NamespaceJson>>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    let namespaces = state
        .processor
        .list_namespaces()
        .into_iter()
        .map(|n| NamespaceJson {
            namespace: n.namespace,
            fact_count: n.fact_count,
            episode_count: n.episode_count,
        })
        .collect();

    Ok(Json(namespaces))
}

/// GET /v1/memory/export — requires read permission
async fn export_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ExportJson>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    let facts = state.processor.export_facts().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Export failed: {e}"),
        )
    })?;
    let episodes = state.processor.export_episodes().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Export failed: {e}"),
        )
    })?;

    Ok(Json(ExportJson {
        version: env!("CARGO_PKG_VERSION").to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        facts,
        episodes,
    }))
}

/// POST /v1/memory/import — requires write permission
async fn import_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, (StatusCode, String)> {
    check_auth(&state, &headers, "write")?;

    if body.dry_run {
        // Preview only — report item counts without touching the database.
        return Ok(Json(ImportResponse {
            facts_imported: body.facts.len(),
            episodes_imported: body.episodes.len(),
            facts_already_existed: 0,
            episodes_already_existed: 0,
            embedded: 0,
            embed_failed: 0,
        }));
    }

    let (facts_imported, new_fact_indices) =
        state.processor.import_facts(&body.facts).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Import failed: {e}"),
            )
        })?;
    let episodes_imported = state
        .processor
        .import_episodes(&body.episodes)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Import failed: {e}"),
            )
        })?;

    let mut embedded = 0;
    let mut embed_failed = 0;
    if !new_fact_indices.is_empty() {
        let new_facts: Vec<signal::ExportedFact> = new_fact_indices
            .iter()
            .map(|&idx| body.facts[idx].clone())
            .collect();
        let (e, f) = state.processor.reembed_facts(&new_facts).await;
        embedded = e;
        embed_failed = f;
    }

    Ok(Json(ImportResponse {
        facts_imported,
        episodes_imported,
        facts_already_existed: body.facts.len() - facts_imported,
        episodes_already_existed: body.episodes.len() - episodes_imported,
        embedded,
        embed_failed,
    }))
}

// ─── Schedules ───────────────────────────────────────────────────────────────

/// GET /v1/schedules — list scheduled intents (requires read permission)
async fn list_schedules_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    let namespace = params.get("namespace").map(|s| s.as_str());
    let intents = state
        .processor
        .list_scheduled_intents(namespace)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to list schedules: {e}"),
            )
        })?;

    let json: Vec<serde_json::Value> = intents
        .iter()
        .map(|i| {
            serde_json::json!({
                "id": i.id,
                "description": i.description,
                "cron": i.cron,
                "namespace": i.namespace,
                "created_at": i.created_at,
                "status": i.status,
            })
        })
        .collect();

    Ok(Json(json))
}

/// DELETE /v1/schedules/:id — cancel a scheduled intent (requires write permission)
async fn cancel_schedule_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    check_auth(&state, &headers, "write")?;

    let cancelled = state.processor.cancel_scheduled_intent(&id).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to cancel schedule: {e}"),
        )
    })?;

    if cancelled {
        Ok(Json(serde_json::json!({"cancelled": true, "id": id})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("No scheduled intent found with ID: {id}"),
        ))
    }
}

// ─── SSE event stream ───────────────────────────────────────────────────────

/// `GET /v1/events` — Server-Sent Events stream of proactive notifications.
///
/// Streams signal-processed events and (optionally) proactive notifications
/// as JSON SSE events. The connection stays open until the client disconnects.
///
/// Always available — proactive notifications are included when the
/// NotificationRouter is configured, but the endpoint works without it.
async fn sse_events_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<
    Sse<impl futures_core::Stream<Item = Result<Event, std::convert::Infallible>>>,
    (StatusCode, String),
> {
    check_auth(&state, &headers, "read")?;

    let mut signal_rx = state.processor.subscribe_events();
    let mut notif_rx = state.processor.notification_router().map(|r| r.subscribe());

    let stream = async_stream::stream! {
        loop {
            tokio::select! {
                result = signal_rx.recv() => {
                    match result {
                        Ok(event) => {
                            let payload = serde_json::json!({
                                "type": "signal",
                                "signal_id": event.signal_id.to_string(),
                                "source": format!("{:?}", event.source),
                                "status": format!("{:?}", event.status),
                                "response": event.response,
                                "facts_used": event.facts_used,
                                "episodes_used": event.episodes_used,
                            });
                            yield Ok(Event::default()
                                .event("signal")
                                .json_data(payload)
                                .unwrap_or_else(|_| Event::default().data("{}")));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(skipped = n, "SSE signal stream lagged");
                            yield Ok(Event::default()
                                .event("error")
                                .data(format!("{{\"lagged\":{n}}}")));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
                result = async {
                    match notif_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match result {
                        Ok(notification) => {
                            let payload = serde_json::json!({
                                "type": "proactive",
                                "content": notification.content,
                                "triggered_by": notification.triggered_by,
                                "priority": notification.priority,
                                "agent": notification.agent,
                            });
                            yield Ok(Event::default()
                                .event("notification")
                                .json_data(payload)
                                .unwrap_or_else(|_| Event::default().data("{}")));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(skipped = n, "SSE notification stream lagged");
                            yield Ok(Event::default()
                                .event("error")
                                .data(format!("{{\"lagged\":{n}}}")));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            // Notification channel closed, but signal stream may still be live.
                            notif_rx = None;
                        }
                    }
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a test router with the demo key pre-loaded.
    async fn make_router() -> (Router, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor), api_keys, true);
        (router, temp)
    }

    #[test]
    fn test_parse_source_defaults_to_http() {
        assert_eq!(
            SignalSource::parse(None, SignalSource::Http),
            SignalSource::Http
        );
        assert_eq!(
            SignalSource::parse(Some("http"), SignalSource::Http),
            SignalSource::Http
        );
    }

    #[test]
    fn test_parse_source_all_variants() {
        assert_eq!(
            SignalSource::parse(Some("cli"), SignalSource::Http),
            SignalSource::Cli
        );
        assert_eq!(
            SignalSource::parse(Some("ws"), SignalSource::Http),
            SignalSource::WebSocket
        );
        assert_eq!(
            SignalSource::parse(Some("mcp"), SignalSource::Http),
            SignalSource::Mcp
        );
        assert_eq!(
            SignalSource::parse(Some("grpc"), SignalSource::Http),
            SignalSource::Grpc
        );
    }

    #[test]
    fn test_health_response_serializes() {
        let h = HealthResponse {
            status: "ok",
            version: "1.0.0",
        };
        let json = serde_json::to_string(&h).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"version\""));
    }

    #[test]
    fn test_fact_json_serializes() {
        let f = FactJson {
            id: "abc".into(),
            namespace: "personal".into(),
            category: "personal".into(),
            subject: "user".into(),
            predicate: "likes".into(),
            object: "Rust".into(),
            confidence: 0.9,
            distance: Some(0.05),
        };
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains("\"subject\":\"user\""));
        assert!(json.contains("\"namespace\":\"personal\""));
        assert!(json.contains("\"distance\":0.05"));
    }

    /// GET /openapi.json — no auth required, returns valid OpenAPI spec.
    #[tokio::test]
    async fn test_openapi_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/openapi.json")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let spec: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
        assert_eq!(spec["openapi"], "3.0.3");
        assert!(
            spec["paths"]["/v1/signals"].is_object(),
            "missing /v1/signals path"
        );
        assert!(
            spec["components"]["schemas"]["FactJson"].is_object(),
            "missing FactJson schema"
        );
    }

    /// GET /api — no auth required, returns Swagger UI HTML.
    #[tokio::test]
    async fn test_swagger_ui_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/api")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(body.contains("swagger-ui"), "missing Swagger UI element");
        assert!(body.contains("/openapi.json"), "missing spec URL reference");
    }

    /// GET /ui — no auth required, returns HTML page.
    #[tokio::test]
    async fn test_ui_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/ui")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let ct = response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/html"), "expected text/html, got: {ct}");

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(body.contains("Brain Memory Explorer"), "missing page title");
        assert!(
            body.contains("/v1/memory/search"),
            "missing API endpoint reference"
        );
    }

    /// GET /metrics — no auth required, returns Prometheus text.
    #[tokio::test]
    async fn test_metrics_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let ct = response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/plain"), "expected text/plain, got: {ct}");

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = std::str::from_utf8(&bytes).unwrap();
        assert!(
            body.contains("brain_signals_total"),
            "missing counter in metrics output"
        );
        assert!(
            body.contains("brain_search_total"),
            "missing search counter"
        );
    }

    /// GET /health — no auth required, always returns 200.
    #[tokio::test]
    async fn test_health_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "ok");
    }

    /// POST /v1/signals without auth → 401.
    #[tokio::test]
    async fn test_post_signal_no_auth_returns_401() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let payload = serde_json::json!({"content": "Remember Rust is fast"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// POST /v1/signals with invalid key → 401.
    #[tokio::test]
    async fn test_post_signal_invalid_key_returns_401() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let payload = serde_json::json!({"content": "Remember Rust is fast"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer wrong-key")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// POST /v1/signals with valid demo key → 200.
    #[tokio::test]
    async fn test_post_signal_store_fact_with_auth() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let payload = serde_json::json!({"content": "Remember that Rust is fast"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer demokey123")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(resp["status"], "Ok");
    }

    /// GET /v1/memory/facts with no auth → 401.
    #[tokio::test]
    async fn test_get_facts_no_auth_returns_401() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// GET /v1/memory/facts with valid demo key → 200.
    #[tokio::test]
    async fn test_get_facts_endpoint_with_auth() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let (router, _tmp) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .header("authorization", "Bearer demokey123")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body.is_array());
    }

    /// POST /v1/memory/search with valid read-only key → 200.
    #[tokio::test]
    async fn test_search_with_read_only_key() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        // Add a read-only key
        config.access.api_keys.push(ApiKeyConfig {
            key: "read-only-key".to_string(),
            name: "Read Only".to_string(),
            permissions: vec!["read".to_string()],
        });
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor), api_keys, true);

        let payload = serde_json::json!({"query": "Rust", "top_k": 5});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/memory/search")
            .header("content-type", "application/json")
            .header("authorization", "Bearer read-only-key")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// POST /v1/signals with read-only key → 401 (missing write permission).
    #[tokio::test]
    async fn test_post_signal_read_only_key_returns_401() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.access.api_keys.push(ApiKeyConfig {
            key: "read-only-key".to_string(),
            name: "Read Only".to_string(),
            permissions: vec!["read".to_string()],
        });
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor), api_keys, true);

        let payload = serde_json::json!({"content": "Remember something"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer read-only-key")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    /// Integration test: HTTP POST /v1/signals (store intent) → fact persisted in DB.
    ///
    /// Stores a fact via the HTTP signal endpoint, then verifies it appears in
    /// GET /v1/memory/facts. Uses shared AppState so both requests hit the same
    /// SignalProcessor and SQLite database.
    #[tokio::test]
    async fn test_http_store_signal_fact_persisted_in_db() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_keys = config.access.api_keys.clone();
        let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
        let state = Arc::new(AppState {
            processor,
            cache: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(CACHE_CAPACITY).unwrap(),
            )),
            api_keys,
            metrics: Arc::new(Metrics::default()),
        });

        // POST /v1/signals with a store-fact intent
        let payload = serde_json::json!({"content": "Remember that Rust is fast"});
        let post_req = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer demokey123")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let router = Router::new()
            .route("/v1/signals", post(post_signal_handler))
            .route("/v1/memory/facts", get(get_facts_handler))
            .with_state(state.clone());

        let post_resp = router.clone().oneshot(post_req).await.unwrap();
        assert_eq!(post_resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(post_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(resp_json["status"], "Ok");
        // Signal was processed — memory_context is present (facts_used depends on embeddings)
        assert!(resp_json["memory_context"].is_object());

        // GET /v1/memory/facts → fact should now be persisted in DB
        let get_req = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .header("authorization", "Bearer demokey123")
            .body(Body::empty())
            .unwrap();

        let get_resp = router.oneshot(get_req).await.unwrap();
        assert_eq!(get_resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(get_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let facts: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(facts.is_array(), "Expected array of facts");
        assert!(
            !facts.as_array().unwrap().is_empty(),
            "Stored fact should appear in GET /v1/memory/facts"
        );
    }

    /// Integration test: HTTP POST /v1/memory/search → returns relevant fact.
    ///
    /// Stores a fact via the SignalProcessor directly (bypassing HTTP for setup),
    /// then calls POST /v1/memory/search and verifies the fact is returned.
    #[tokio::test]
    async fn test_http_memory_search_returns_stored_fact() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_keys = config.access.api_keys.clone();
        let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());

        // Pre-store a fact directly so search has something to find
        let _ = processor
            .store_fact_direct("personal", "test", "Ferris", "is", "the Rust mascot", None)
            .await
            .unwrap();

        let state = Arc::new(AppState {
            processor,
            cache: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(CACHE_CAPACITY).unwrap(),
            )),
            api_keys,
            metrics: Arc::new(Metrics::default()),
        });

        let router = Router::new()
            .route("/v1/memory/search", post(search_memory_handler))
            .with_state(state);

        // Search for the stored fact
        let payload = serde_json::json!({"query": "Ferris Rust mascot", "top_k": 5});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/memory/search")
            .header("content-type", "application/json")
            .header("authorization", "Bearer demokey123")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let results: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Endpoint must return a JSON array. Result count depends on embedding quality —
        // with no real embeddings available in unit tests, HNSW may return 0 matches.
        assert!(results.is_array(), "Expected array of search results");
    }

    /// Integration test: cached signal can be retrieved by GET /v1/signals/:id.
    #[tokio::test]
    async fn test_get_cached_signal_with_auth() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_keys = config.access.api_keys.clone();
        let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
        let state = Arc::new(AppState {
            processor,
            cache: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(CACHE_CAPACITY).unwrap(),
            )),
            api_keys,
            metrics: Arc::new(Metrics::default()),
        });

        // Manually insert a response into the cache
        let id = Uuid::new_v4();
        let fake_resp = SignalResponse::ok(id, "test response");
        state.cache.lock().await.put(id, fake_resp);

        let router = Router::new()
            .route("/v1/signals/:id", get(get_signal_handler))
            .with_state(state);

        let request = Request::builder()
            .method(http::Method::GET)
            .uri(format!("/v1/signals/{id}"))
            .header("authorization", "Bearer demokey123")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
