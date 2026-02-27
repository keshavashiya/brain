//! # Brain HTTP REST API Adapter
//!
//! Exposes Brain's signal processing pipeline over HTTP using axum.
//!
//! ## Routes
//! - `GET  /health`             — health check (no auth required)
//! - `POST /v1/signals`         — submit a signal (requires write)
//! - `GET  /v1/signals/:id`     — retrieve cached signal response (requires read)
//! - `POST /v1/memory/search`   — semantic search over stored facts (requires read)
//! - `GET  /v1/memory/facts`    — list all semantic facts (requires read)
//!
//! ## Authentication
//! All `/v1/*` routes require `Authorization: Bearer <api-key>` header.
//! The demo key `demo-key-123` (read+write) is pre-configured in `default.yaml`.

use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use brain_core::ApiKeyConfig;
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
    /// Memory namespace (default: "personal").
    pub namespace: Option<String>,
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

// ─── App State ───────────────────────────────────────────────────────────────

/// Shared state for all HTTP handlers.
pub struct AppState {
    processor: Arc<signal::SignalProcessor>,
    /// In-memory cache: signal_id → SignalResponse.
    cache: Mutex<HashMap<Uuid, SignalResponse>>,
    /// Configured API keys (loaded from BrainConfig).
    api_keys: Vec<ApiKeyConfig>,
}

// ─── Auth helpers ─────────────────────────────────────────────────────────────

/// Extract the raw key from `Authorization: Bearer <key>`.
fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
}

/// Check that the request carries a valid key with the given permission.
/// Returns `Err((StatusCode::UNAUTHORIZED, message))` on failure.
fn check_auth(
    state: &AppState,
    headers: &HeaderMap,
    permission: &str,
) -> Result<(), (StatusCode, String)> {
    let raw_key = extract_bearer(headers).ok_or_else(|| {
        (
            StatusCode::UNAUTHORIZED,
            "Missing Authorization: Bearer <key> header".to_string(),
        )
    })?;

    match state.api_keys.iter().find(|k| k.key == raw_key) {
        None => Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string())),
        Some(k) if !k.has_permission(permission) => Err((
            StatusCode::UNAUTHORIZED,
            format!("API key does not have '{}' permission", permission),
        )),
        Some(_) => Ok(()),
    }
}

// ─── Router builder ──────────────────────────────────────────────────────────

/// Build the axum router with all routes and CORS enabled.
///
/// `api_keys` is taken from `BrainConfig.access.api_keys` by the caller.
pub fn create_router(
    processor: Arc<signal::SignalProcessor>,
    api_keys: Vec<ApiKeyConfig>,
) -> Router {
    let state = Arc::new(AppState {
        processor,
        cache: Mutex::new(HashMap::new()),
        api_keys,
    });

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/signals", post(post_signal_handler))
        .route("/v1/signals/:id", get(get_signal_handler))
        .route("/v1/memory/search", post(search_memory_handler))
        .route("/v1/memory/facts", get(get_facts_handler))
        .route("/v1/memory/namespaces", get(get_namespaces_handler))
        .with_state(state)
        .layer(CorsLayer::permissive())
}

/// Start the HTTP server, binding to `host:port`.
///
/// Blocks until the server shuts down.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let api_keys = processor.config().access.api_keys.clone();
    let router = create_router(processor, api_keys);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    tracing::info!("Brain HTTP API listening on http://{addr}");
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

/// POST /v1/signals — requires write permission
async fn post_signal_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SignalRequest>,
) -> Result<Json<SignalResponse>, (StatusCode, String)> {
    check_auth(&state, &headers, "write")?;

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
    if let Some(ns) = body.namespace {
        signal.namespace = ns;
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

/// GET /v1/signals/:id — requires read permission
async fn get_signal_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<SignalResponse>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

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

/// POST /v1/memory/search — requires read permission
async fn search_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SearchRequest>,
) -> Result<Json<Vec<FactJson>>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

    let top_k = body.top_k.unwrap_or(10);
    let namespace = body.namespace.as_deref();
    let results = state
        .processor
        .search_facts(&body.query, top_k, namespace)
        .await;

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
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Vec<FactJson>>, (StatusCode, String)> {
    check_auth(&state, &headers, "read")?;

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

    /// Build a test router with the demo key pre-loaded.
    async fn make_router() -> (Router, tempfile::TempDir) {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = create_router(Arc::new(processor), api_keys);
        (router, temp)
    }

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
            .header("authorization", "Bearer demo-key-123")
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
            .header("authorization", "Bearer demo-key-123")
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
        let router = create_router(Arc::new(processor), api_keys);

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
        let router = create_router(Arc::new(processor), api_keys);

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
            cache: Mutex::new(HashMap::new()),
            api_keys,
        });

        // POST /v1/signals with a store-fact intent
        let payload = serde_json::json!({"content": "Remember that Rust is fast"});
        let post_req = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer demo-key-123")
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
        // StoreFact response includes facts_used = 1
        assert_eq!(resp_json["memory_context"]["facts_used"], 1);

        // GET /v1/memory/facts → fact should now be persisted in DB
        let get_req = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .header("authorization", "Bearer demo-key-123")
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
            .store_fact_direct("personal", "test", "Ferris", "is", "the Rust mascot")
            .await
            .unwrap();

        let state = Arc::new(AppState {
            processor,
            cache: Mutex::new(HashMap::new()),
            api_keys,
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
            .header("authorization", "Bearer demo-key-123")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let results: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(results.is_array(), "Expected array of search results");
        // With zero-vector fallback, all stored facts are returned at distance 1.0
        assert!(
            !results.as_array().unwrap().is_empty(),
            "POST /v1/memory/search should return the stored fact"
        );
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
            cache: Mutex::new(HashMap::new()),
            api_keys,
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
            .header("authorization", "Bearer demo-key-123")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
