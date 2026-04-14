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
pub mod server;
pub mod state;
pub mod types;

// Re-export primary public types for convenience.
pub use server::{create_router, serve};
pub use state::AppState;
pub use types::HttpAdapterError;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, num::NonZeroUsize, sync::Arc};

    use axum::{
        body::Body,
        http::{self, HeaderMap, Request},
        response::IntoResponse,
        routing::{delete, get, post},
        Router,
    };
    use tokio::sync::Mutex;
    use tower::util::ServiceExt;
    use uuid::Uuid;

    use super::handlers::*;
    use super::metrics::Metrics;
    use super::state::{AppState, CACHE_CAPACITY};
    use super::types::*;

    /// Build a test router with the API key pre-loaded.
    async fn make_router() -> (Router, tempfile::TempDir, String) {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_key = config.access.api_keys.first().unwrap().key.clone();
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = crate::create_router(Arc::new(processor), api_keys, true);
        (router, temp, api_key)
    }

    #[test]
    fn test_parse_source_defaults_to_http() {
        assert_eq!(
            signal::SignalSource::parse(None, signal::SignalSource::Http),
            signal::SignalSource::Http
        );
        assert_eq!(
            signal::SignalSource::parse(Some("http"), signal::SignalSource::Http),
            signal::SignalSource::Http
        );
    }

    #[test]
    fn test_parse_source_all_variants() {
        assert_eq!(
            signal::SignalSource::parse(Some("cli"), signal::SignalSource::Http),
            signal::SignalSource::Cli
        );
        assert_eq!(
            signal::SignalSource::parse(Some("ws"), signal::SignalSource::Http),
            signal::SignalSource::WebSocket
        );
        assert_eq!(
            signal::SignalSource::parse(Some("mcp"), signal::SignalSource::Http),
            signal::SignalSource::Mcp
        );
        assert_eq!(
            signal::SignalSource::parse(Some("grpc"), signal::SignalSource::Http),
            signal::SignalSource::Grpc
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
        let (router, _tmp, _api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/openapi.json")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

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
        let (router, _tmp, _api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/api")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

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
        let (router, _tmp, _api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/ui")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

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
        let (router, _tmp, _api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

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
        let (router, _tmp, _api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "ok");
    }

    /// POST /v1/signals without auth → 401.
    #[tokio::test]
    async fn test_post_signal_no_auth_returns_401() {
        let (router, _tmp, _api_key) = make_router().await;

        let payload = serde_json::json!({"content": "Remember Rust is fast"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::UNAUTHORIZED);
    }

    /// POST /v1/signals with invalid key → 401.
    #[tokio::test]
    async fn test_post_signal_invalid_key_returns_401() {
        let (router, _tmp, _api_key) = make_router().await;

        let payload = serde_json::json!({"content": "Remember Rust is fast"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer wrong-key")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::UNAUTHORIZED);
    }

    /// POST /v1/signals with valid API key → 200.
    #[tokio::test]
    async fn test_post_signal_store_fact_with_auth() {
        let (router, _tmp, api_key) = make_router().await;

        let payload = serde_json::json!({"content": "Remember that Rust is fast"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {api_key}"))
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(resp["status"], "Ok");
    }

    /// GET /v1/memory/facts with no auth → 401.
    #[tokio::test]
    async fn test_get_facts_no_auth_returns_401() {
        let (router, _tmp, _api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::UNAUTHORIZED);
    }

    /// GET /v1/memory/facts with valid API key → 200.
    #[tokio::test]
    async fn test_get_facts_endpoint_with_auth() {
        let (router, _tmp, api_key) = make_router().await;

        let request = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .header("authorization", format!("Bearer {api_key}"))
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body.is_array());
    }

    /// POST /v1/memory/search with valid read-only key → 200.
    #[tokio::test]
    async fn test_search_with_read_only_key() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.access.api_keys.push(brain_core::ApiKeyConfig {
            key: "read-only-key".to_string(),
            name: "Read Only".to_string(),
            permissions: vec!["read".to_string()],
        });
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = crate::create_router(Arc::new(processor), api_keys, true);

        let payload = serde_json::json!({"query": "Rust", "top_k": 5});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/memory/search")
            .header("content-type", "application/json")
            .header("authorization", "Bearer read-only-key")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
    }

    /// POST /v1/signals with read-only key → 401 (missing write permission).
    #[tokio::test]
    async fn test_post_signal_read_only_key_returns_401() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.access.api_keys.push(brain_core::ApiKeyConfig {
            key: "read-only-key".to_string(),
            name: "Read Only".to_string(),
            permissions: vec!["read".to_string()],
        });
        let api_keys = config.access.api_keys.clone();
        let processor = signal::SignalProcessor::new(config).await.unwrap();
        let router = crate::create_router(Arc::new(processor), api_keys, true);

        let payload = serde_json::json!({"content": "Remember something"});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", "Bearer read-only-key")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::UNAUTHORIZED);
    }

    /// Integration test: HTTP POST /v1/signals (store intent) → fact persisted in DB.
    #[tokio::test]
    async fn test_http_store_signal_fact_persisted_in_db() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_key = config.access.api_keys.first().unwrap().key.clone();
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

        let payload = serde_json::json!({"content": "Remember that Rust is fast"});
        let post_req = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/signals")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {api_key}"))
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let router = Router::new()
            .route("/v1/signals", post(post_signal_handler))
            .route("/v1/memory/facts", get(get_facts_handler))
            .with_state(state.clone());

        let post_resp = router.clone().oneshot(post_req).await.unwrap();
        assert_eq!(post_resp.status(), http::StatusCode::OK);

        let bytes = axum::body::to_bytes(post_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let resp_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(resp_json["status"], "Ok");
        assert!(resp_json["memory_context"].is_object());

        let get_req = Request::builder()
            .method(http::Method::GET)
            .uri("/v1/memory/facts")
            .header("authorization", format!("Bearer {api_key}"))
            .body(Body::empty())
            .unwrap();

        let get_resp = router.oneshot(get_req).await.unwrap();
        assert_eq!(get_resp.status(), http::StatusCode::OK);

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
    #[tokio::test]
    async fn test_http_memory_search_returns_stored_fact() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_key = config.access.api_keys.first().unwrap().key.clone();
        let api_keys = config.access.api_keys.clone();
        let processor = Arc::new(signal::SignalProcessor::new(config).await.unwrap());

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

        let payload = serde_json::json!({"query": "Ferris Rust mascot", "top_k": 5});
        let request = Request::builder()
            .method(http::Method::POST)
            .uri("/v1/memory/search")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {api_key}"))
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let results: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(results.is_array(), "Expected array of search results");
    }

    /// Integration test: cached signal can be retrieved by GET /v1/signals/:id.
    #[tokio::test]
    async fn test_get_cached_signal_with_auth() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let api_key = config.access.api_keys.first().unwrap().key.clone();
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

        let id = Uuid::new_v4();
        let fake_resp = signal::SignalResponse::ok(id, "test response");
        state.cache.lock().await.put(id, fake_resp);

        let router = Router::new()
            .route("/v1/signals/:id", get(get_signal_handler))
            .with_state(state);

        let request = Request::builder()
            .method(http::Method::GET)
            .uri(format!("/v1/signals/{id}"))
            .header("authorization", format!("Bearer {api_key}"))
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), http::StatusCode::OK);
    }
}
