//! # Brain HTTP REST API Adapter
//!
//! Exposes Brain's signal processing pipeline over HTTP using axum.
//!
//! ## Routes
//! - `GET  /health`             — health check
//! - `POST /v1/signals`         — submit a signal, get a response
//! - `GET  /v1/signals/:id`     — retrieve cached signal response by UUID
//! - `POST /v1/memory/search`   — semantic search over stored facts
//! - `GET  /v1/memory/facts`    — list all semantic facts

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
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
}

/// Search request body (POST /v1/memory/search).
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
}

/// A single fact in JSON form (GET /v1/memory/facts, search results).
#[derive(Debug, Serialize)]
pub struct FactJson {
    pub id: String,
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

// ─── App State ───────────────────────────────────────────────────────────────

/// Shared state for all HTTP handlers.
pub struct AppState {
    processor: Arc<signal::SignalProcessor>,
    /// In-memory cache: signal_id → SignalResponse.
    cache: Mutex<HashMap<Uuid, SignalResponse>>,
}

// ─── Router builder ──────────────────────────────────────────────────────────

/// Build the axum router with all routes and CORS enabled.
pub fn create_router(processor: Arc<signal::SignalProcessor>) -> Router {
    let state = Arc::new(AppState {
        processor,
        cache: Mutex::new(HashMap::new()),
    });

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/signals", post(post_signal_handler))
        .route("/v1/signals/:id", get(get_signal_handler))
        .route("/v1/memory/search", post(search_memory_handler))
        .route("/v1/memory/facts", get(get_facts_handler))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// Start the HTTP server, binding to `host:port`.
///
/// Blocks until the server shuts down.
pub async fn serve(
    processor: signal::SignalProcessor,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let router = create_router(Arc::new(processor));
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Brain HTTP API listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// GET /health
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// POST /v1/signals
///
/// Accepts a JSON `SignalRequest`, processes it through `SignalProcessor`,
/// caches the response by signal UUID, and returns it.
async fn post_signal_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SignalRequest>,
) -> Result<Json<SignalResponse>, (StatusCode, String)> {
    let source = parse_source(body.source.as_deref());
    let mut signal = Signal::new(
        source,
        body.channel.unwrap_or_else(|| "http".to_string()),
        body.sender.unwrap_or_else(|| "api-client".to_string()),
        body.content,
    );
    if let Some(meta) = body.metadata {
        signal.metadata = meta;
    }

    let signal_id = signal.id;
    let response = state
        .processor
        .process(signal)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Cache the response so GET /v1/signals/:id can retrieve it
    state.cache.lock().await.insert(signal_id, response.clone());

    Ok(Json(response))
}

/// GET /v1/signals/:id
///
/// Returns a previously-cached `SignalResponse` by UUID.
async fn get_signal_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SignalResponse>, (StatusCode, String)> {
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, format!("Invalid UUID: {id}")))?;

    let cache = state.cache.lock().await;
    match cache.get(&uuid) {
        Some(resp) => Ok(Json(resp.clone())),
        None => Err((
            StatusCode::NOT_FOUND,
            format!("Signal {uuid} not found in cache"),
        )),
    }
}

/// POST /v1/memory/search
///
/// Accepts `{query, top_k}` and returns ranked semantic facts.
async fn search_memory_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SearchRequest>,
) -> Json<Vec<FactJson>> {
    let top_k = body.top_k.unwrap_or(10);
    let results = state.processor.search_facts(&body.query, top_k).await;

    let facts = results
        .into_iter()
        .map(|r| FactJson {
            id: r.fact.id,
            category: r.fact.category,
            subject: r.fact.subject,
            predicate: r.fact.predicate,
            object: r.fact.object,
            confidence: r.fact.confidence,
            distance: Some(r.distance),
        })
        .collect();

    Json(facts)
}

/// GET /v1/memory/facts
///
/// Returns all active (non-superseded) semantic facts.
async fn get_facts_handler(State(state): State<Arc<AppState>>) -> Json<Vec<FactJson>> {
    let facts = state
        .processor
        .list_facts()
        .into_iter()
        .map(|f| FactJson {
            id: f.id,
            category: f.category,
            subject: f.subject,
            predicate: f.predicate,
            object: f.object,
            confidence: f.confidence,
            distance: None,
        })
        .collect();

    Json(facts)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn parse_source(s: Option<&str>) -> SignalSource {
    match s {
        Some("http") | None => SignalSource::Http,
        Some("cli") => SignalSource::Cli,
        Some("ws") | Some("websocket") => SignalSource::WebSocket,
        Some("mcp") => SignalSource::Mcp,
        Some("grpc") => SignalSource::Grpc,
        _ => SignalSource::Http,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_source_defaults_to_http() {
        assert_eq!(parse_source(None), SignalSource::Http);
        assert_eq!(parse_source(Some("http")), SignalSource::Http);
    }

    #[test]
    fn test_parse_source_all_variants() {
        assert_eq!(parse_source(Some("cli")), SignalSource::Cli);
        assert_eq!(parse_source(Some("ws")), SignalSource::WebSocket);
        assert_eq!(parse_source(Some("mcp")), SignalSource::Mcp);
        assert_eq!(parse_source(Some("grpc")), SignalSource::Grpc);
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
            category: "personal".into(),
            subject: "user".into(),
            predicate: "likes".into(),
            object: "Rust".into(),
            confidence: 0.9,
            distance: Some(0.05),
        };
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains("\"subject\":\"user\""));
        assert!(json.contains("\"distance\":0.05"));
    }

    /// Integration test: POST /v1/signals → StoreFact intent → cached response.
    ///
    /// Spins up an in-process router against a temp SignalProcessor.
    #[tokio::test]
    async fn test_post_signal_store_fact() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor));

        let payload = serde_json::json!({
            "content": "Remember that Rust is fast"
        });

        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
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

    /// Integration test: GET /health returns {"status":"ok"}.
    #[tokio::test]
    async fn test_health_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor));

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

    /// Integration test: GET /v1/memory/facts returns an array.
    #[tokio::test]
    async fn test_get_facts_endpoint() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor));

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
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

    /// Integration test: cached signal can be retrieved by GET /v1/signals/:id.
    #[tokio::test]
    async fn test_get_cached_signal() {
        use axum::body::Body;
        use axum::http::{self, Request};
        use tower::util::ServiceExt;

        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());
        let state = Arc::new(AppState {
            processor,
            cache: Mutex::new(HashMap::new()),
        });

        // Manually insert a response into the cache
        let id = Uuid::new_v4();
        let fake_resp = SignalResponse::ok(id, "test response");
        state.cache.lock().await.insert(id, fake_resp);

        let router = Router::new()
            .route("/v1/signals/:id", get(get_signal_handler))
            .with_state(state);

        let request = Request::builder()
            .method(http::Method::GET)
            .uri(format!("/v1/signals/{id}"))
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
