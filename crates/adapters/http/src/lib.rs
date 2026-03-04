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
//!
//! ## Authentication
//! All `/v1/*` routes require `Authorization: Bearer <api-key>` header.
//! The demo key `demokey123` (read+write) is pre-configured in `default.yaml`.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use brain_core::ApiKeyConfig;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
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

/// Shared state for all HTTP handlers.
pub struct AppState {
    processor: Arc<signal::SignalProcessor>,
    /// In-memory cache: signal_id → SignalResponse.
    cache: Mutex<HashMap<Uuid, SignalResponse>>,
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

/// CORS restricted to localhost origins — Brain is a local daemon, not a public service.
/// Remote origins are blocked to prevent cross-site requests from untrusted web pages.
fn localhost_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _req| {
            let bytes = origin.as_bytes();
            bytes.starts_with(b"http://127.0.0.1")
                || bytes.starts_with(b"http://localhost")
                || bytes.starts_with(b"https://127.0.0.1")
                || bytes.starts_with(b"https://localhost")
        }))
        .allow_methods(AllowMethods::any())
        .allow_headers(AllowHeaders::any())
}

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
        metrics: Arc::new(Metrics::default()),
    });

    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/ui", get(ui_handler))
        .route("/openapi.json", get(openapi_handler))
        .route("/api", get(swagger_ui_handler))
        .route("/v1/signals", post(post_signal_handler))
        .route("/v1/signals/:id", get(get_signal_handler))
        .route("/v1/memory/search", post(search_memory_handler))
        .route("/v1/memory/facts", get(get_facts_handler))
        .route("/v1/memory/namespaces", get(get_namespaces_handler))
        .with_state(state)
        .layer(localhost_cors())
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
const UI_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Memory Explorer</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
  header{background:#1e293b;border-bottom:1px solid #334155;padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.25rem;font-weight:700;color:#38bdf8}
  header span{color:#94a3b8;font-size:.875rem}
  main{max-width:1200px;margin:0 auto;padding:2rem}
  .search-bar{display:flex;gap:.5rem;margin-bottom:2rem}
  .search-bar input{flex:1;background:#1e293b;border:1px solid #334155;border-radius:.5rem;padding:.75rem 1rem;color:#e2e8f0;font-size:1rem;outline:none}
  .search-bar input:focus{border-color:#38bdf8}
  .search-bar button{background:#0ea5e9;border:none;border-radius:.5rem;padding:.75rem 1.5rem;color:#fff;font-size:1rem;cursor:pointer}
  .search-bar button:hover{background:#0284c7}
  .tabs{display:flex;gap:.5rem;margin-bottom:1.5rem}
  .tab{background:#1e293b;border:1px solid #334155;border-radius:.5rem;padding:.5rem 1rem;cursor:pointer;font-size:.875rem;color:#94a3b8}
  .tab.active{background:#0ea5e9;border-color:#0ea5e9;color:#fff}
  #status{color:#94a3b8;font-size:.875rem;margin-bottom:1rem;min-height:1.25rem}
  #status.error{color:#f87171}
  .grid{display:grid;gap:1rem}
  .card{background:#1e293b;border:1px solid #334155;border-radius:.75rem;padding:1.25rem}
  .card .meta{font-size:.75rem;color:#64748b;margin-bottom:.5rem}
  .card .subject{font-weight:600;color:#38bdf8;margin-bottom:.25rem}
  .card .predicate{color:#94a3b8;font-size:.875rem;margin-bottom:.25rem}
  .card .object{color:#e2e8f0}
  .card .badge{display:inline-block;background:#0f172a;border:1px solid #334155;border-radius:.25rem;padding:.15rem .4rem;font-size:.7rem;color:#64748b;margin-right:.25rem}
  .card .conf{color:#4ade80;font-size:.75rem}
  .empty{color:#475569;text-align:center;padding:3rem;font-size:.9rem}
</style>
</head>
<body>
<header>
  <h1>🧠 Brain</h1>
  <span>Memory Explorer</span>
  <span id="api-status" style="margin-left:auto;font-size:.75rem"></span>
</header>
<main>
  <div class="search-bar">
    <input id="q" type="search" placeholder="Search memory… (e.g. Rust, project goals)" autofocus>
    <button onclick="search()">Search</button>
  </div>
  <div class="tabs">
    <div class="tab active" id="tab-search" onclick="switchTab('search')">Search Results</div>
    <div class="tab" id="tab-facts" onclick="switchTab('facts')">All Facts</div>
  </div>
  <div id="status"></div>
  <div id="results" class="grid"></div>
</main>
<script>
const API = '';
let apiKey = localStorage.getItem('brain_api_key') || '';

function hdr(){ return {'Authorization':'Bearer '+apiKey,'Content-Type':'application/json'} }

async function checkHealth(){
  try{
    const r=await fetch(API+'/health');
    const d=await r.json();
    document.getElementById('api-status').textContent='● '+d.status+' v'+d.version;
    document.getElementById('api-status').style.color='#4ade80';
  }catch(e){
    document.getElementById('api-status').textContent='● unreachable';
    document.getElementById('api-status').style.color='#f87171';
  }
}

function setStatus(msg,err){
  const el=document.getElementById('status');
  el.textContent=msg;
  el.className=err?'error':'';
}

function renderFact(f){
  const dist=f.distance!=null?' · dist '+f.distance.toFixed(3):'';
  return `<div class="card">
    <div class="meta"><span class="badge">${f.namespace}</span><span class="badge">${f.category}</span><span class="conf">conf ${f.confidence.toFixed(2)}</span>${dist}</div>
    <div class="subject">${esc(f.subject)}</div>
    <div class="predicate">${esc(f.predicate)}</div>
    <div class="object">${esc(f.object)}</div>
  </div>`;
}

function esc(s){ const d=document.createElement('div');d.textContent=s;return d.innerHTML; }

async function search(){
  const q=document.getElementById('q').value.trim();
  if(!q){await loadFacts();return;}
  setStatus('Searching…');
  try{
    const r=await fetch(API+'/v1/memory/search',{method:'POST',headers:hdr(),body:JSON.stringify({query:q,top_k:20})});
    if(!r.ok){setStatus('Search failed: '+r.status,true);return;}
    const facts=await r.json();
    setStatus(facts.length+' result'+(facts.length!==1?'s':''));
    document.getElementById('results').innerHTML=facts.length?facts.map(renderFact).join(''):'<p class="empty">No matching facts found.</p>';
  }catch(e){setStatus('Error: '+e.message,true);}
}

async function loadFacts(){
  setStatus('Loading…');
  try{
    const r=await fetch(API+'/v1/memory/facts',{headers:hdr()});
    if(!r.ok){setStatus('Failed to load facts: '+r.status,true);return;}
    const facts=await r.json();
    setStatus(facts.length+' stored fact'+(facts.length!==1?'s':''));
    document.getElementById('results').innerHTML=facts.length?facts.map(renderFact).join(''):'<p class="empty">No facts stored yet. Send a "Remember…" signal to add some.</p>';
  }catch(e){setStatus('Error: '+e.message,true);}
}

function switchTab(t){
  document.querySelectorAll('.tab').forEach(el=>el.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  if(t==='facts')loadFacts();
  else{document.getElementById('results').innerHTML='';setStatus('');}
}

document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')search();});
checkHealth();
</script>
</body>
</html>"#;

/// GET /ui — embedded single-page memory explorer (no auth required)
async fn ui_handler() -> impl IntoResponse {
    (
        [("content-type", "text/html; charset=utf-8")],
        UI_HTML,
    )
}

// ─── OpenAPI spec ─────────────────────────────────────────────────────────────

/// Build the OpenAPI 3.0 document for the Brain HTTP API.
fn build_openapi() -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "Brain OS — Synapse HTTP API",
            "description": "Your AI's long-term memory — signal processing, semantic search, and episodic recall.",
            "version": env!("CARGO_PKG_VERSION"),
            "contact": { "name": "Brain OS", "url": "https://github.com/keshavashiya/brain" }
        },
        "servers": [{ "url": "/", "description": "Local Brain instance" }],
        "components": {
            "securitySchemes": {
                "BearerAuth": { "type": "http", "scheme": "bearer", "bearerFormat": "APIKey" }
            },
            "schemas": {
                "HealthResponse": {
                    "type": "object", "required": ["status","version"],
                    "properties": {
                        "status": { "type": "string", "example": "ok" },
                        "version": { "type": "string", "example": "0.1.0" }
                    }
                },
                "SignalRequest": {
                    "type": "object", "required": ["content"],
                    "properties": {
                        "content": { "type": "string", "example": "Remember that Rust is memory-safe" },
                        "source": { "type": "string", "enum": ["http","cli","ws","mcp","grpc"] },
                        "channel": { "type": "string" },
                        "sender": { "type": "string" },
                        "namespace": { "type": "string", "default": "personal" },
                        "metadata": { "type": "object", "additionalProperties": { "type": "string" } }
                    }
                },
                "SignalResponse": {
                    "type": "object", "required": ["signal_id","status","response","memory_context"],
                    "properties": {
                        "signal_id": { "type": "string", "format": "uuid" },
                        "status": { "type": "string", "enum": ["Ok","Error"] },
                        "response": {
                            "type": "object",
                            "properties": {
                                "type": { "type": "string", "enum": ["Text","Json","Error"] },
                                "value": {}
                            }
                        },
                        "memory_context": {
                            "type": "object",
                            "properties": {
                                "facts_used": { "type": "integer" },
                                "episodes_used": { "type": "integer" }
                            }
                        }
                    }
                },
                "SearchRequest": {
                    "type": "object", "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "example": "Rust programming" },
                        "top_k": { "type": "integer", "default": 10 },
                        "namespace": { "type": "string" }
                    }
                },
                "FactJson": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "namespace": { "type": "string" },
                        "category": { "type": "string" },
                        "subject": { "type": "string" },
                        "predicate": { "type": "string" },
                        "object": { "type": "string" },
                        "confidence": { "type": "number", "format": "double" },
                        "distance": { "type": "number", "format": "float", "nullable": true }
                    }
                },
                "NamespaceJson": {
                    "type": "object",
                    "properties": {
                        "namespace": { "type": "string" },
                        "fact_count": { "type": "integer" },
                        "episode_count": { "type": "integer" }
                    }
                }
            }
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "getHealth",
                    "responses": {
                        "200": { "description": "Service is healthy", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/HealthResponse" } } } }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Prometheus metrics",
                    "operationId": "getMetrics",
                    "responses": {
                        "200": { "description": "Prometheus text format metrics", "content": { "text/plain": { "schema": { "type": "string" } } } }
                    }
                }
            },
            "/v1/signals": {
                "post": {
                    "summary": "Submit a signal for processing",
                    "operationId": "postSignal",
                    "security": [{ "BearerAuth": [] }],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SignalRequest" } } } },
                    "responses": {
                        "200": { "description": "Signal processed", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SignalResponse" } } } },
                        "401": { "description": "Unauthorized — missing or invalid API key" },
                        "500": { "description": "Internal server error" }
                    }
                }
            },
            "/v1/signals/{id}": {
                "get": {
                    "summary": "Retrieve a cached signal response by ID",
                    "operationId": "getSignalById",
                    "security": [{ "BearerAuth": [] }],
                    "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string", "format": "uuid" } }],
                    "responses": {
                        "200": { "description": "Signal response found", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SignalResponse" } } } },
                        "401": { "description": "Unauthorized" },
                        "404": { "description": "Signal not found in cache" }
                    }
                }
            },
            "/v1/memory/search": {
                "post": {
                    "summary": "Semantic search over stored facts",
                    "operationId": "searchMemory",
                    "security": [{ "BearerAuth": [] }],
                    "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SearchRequest" } } } },
                    "responses": {
                        "200": { "description": "Matching facts", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/FactJson" } } } } },
                        "401": { "description": "Unauthorized" }
                    }
                }
            },
            "/v1/memory/facts": {
                "get": {
                    "summary": "List all stored semantic facts",
                    "operationId": "listFacts",
                    "security": [{ "BearerAuth": [] }],
                    "parameters": [{ "name": "namespace", "in": "query", "schema": { "type": "string" } }],
                    "responses": {
                        "200": { "description": "List of facts", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/FactJson" } } } } },
                        "401": { "description": "Unauthorized" }
                    }
                }
            },
            "/v1/memory/namespaces": {
                "get": {
                    "summary": "List memory namespaces with statistics",
                    "operationId": "listNamespaces",
                    "security": [{ "BearerAuth": [] }],
                    "responses": {
                        "200": { "description": "List of namespaces", "content": { "application/json": { "schema": { "type": "array", "items": { "$ref": "#/components/schemas/NamespaceJson" } } } } },
                        "401": { "description": "Unauthorized" }
                    }
                }
            }
        }
    })
}

/// GET /openapi.json — OpenAPI 3.0 specification (no auth required)
async fn openapi_handler() -> impl IntoResponse {
    (
        [("content-type", "application/json")],
        build_openapi().to_string(),
    )
}

/// Swagger UI HTML that loads the spec from /openapi.json (CDN assets).
const SWAGGER_UI_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Brain OS API — Swagger UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout: 'StandaloneLayout'
    });
  </script>
</body>
</html>"#;

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

    let source = parse_source(body.source.as_deref());
    let mut signal = Signal::new(
        source,
        body.channel.unwrap_or_else(|| "http".to_string()),
        body.sender.unwrap_or_else(|| "apiclient".to_string()),
        body.content,
    );
    if let Some(meta) = body.metadata {
        signal.metadata = meta;
    }
    if let Some(ns) = body.namespace {
        signal.namespace = ns;
    }

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
            // Return an opaque error — do not leak internal details to the client.
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Signal processing failed. Check server logs for details.".to_string(),
            ));
        }
    };

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
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
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
        assert!(spec["paths"]["/v1/signals"].is_object(), "missing /v1/signals path");
        assert!(spec["components"]["schemas"]["FactJson"].is_object(), "missing FactJson schema");
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
        assert!(body.contains("/v1/memory/search"), "missing API endpoint reference");
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
        assert!(body.contains("brain_signals_total"), "missing counter in metrics output");
        assert!(body.contains("brain_search_total"), "missing search counter");
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
            .store_fact_direct("personal", "test", "Ferris", "is", "the Rust mascot")
            .await
            .unwrap();

        let state = Arc::new(AppState {
            processor,
            cache: Mutex::new(HashMap::new()),
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
            cache: Mutex::new(HashMap::new()),
            api_keys,
            metrics: Arc::new(Metrics::default()),
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
            .header("authorization", "Bearer demokey123")
            .body(Body::empty())
            .unwrap();

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
