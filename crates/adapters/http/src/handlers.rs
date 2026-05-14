//! HTTP route handlers for the Brain REST API.

use std::{collections::HashMap, sync::atomic::Ordering, sync::Arc, time::Instant};

use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::{HeaderMap, Response, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
};
use uuid::Uuid;

use crate::auth;
use crate::state::AppState;
use crate::types::*;

// ─── Embedded assets ─────────────────────────────────────────────────────────

const UI_HTML: &str = include_str!("../assets/ui.html");
static OPENAPI_JSON: &str = include_str!("../assets/openapi.json");
const SWAGGER_UI_HTML: &str = include_str!("../assets/swagger.html");

// ─── Public endpoints (no auth) ──────────────────────────────────────────────

/// GET /health
pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// GET /metrics — Prometheus text format
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.metrics.refresh_gauges(&state.processor);
    let body = state.metrics.render(state.processor.metrics());
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

/// GET /ui — embedded single-page memory explorer
pub async fn ui_handler() -> impl IntoResponse {
    ([("content-type", "text/html; charset=utf-8")], UI_HTML)
}

/// GET / — soft redirect to the diagnostic UI so the bare host:port lands somewhere.
pub async fn root_handler() -> axum::response::Redirect {
    axum::response::Redirect::temporary("/ui")
}

/// GET /openapi.json — OpenAPI 3.0 specification
pub async fn openapi_handler() -> impl IntoResponse {
    let spec = OPENAPI_JSON.replace("{{VERSION}}", env!("CARGO_PKG_VERSION"));
    ([("content-type", "application/json")], spec)
}

/// GET /api — Swagger UI for interactive API exploration
pub async fn swagger_ui_handler() -> impl IntoResponse {
    (
        [("content-type", "text/html; charset=utf-8")],
        SWAGGER_UI_HTML,
    )
}

// ─── Protected /v1/* endpoints ───────────────────────────────────────────────

/// POST /v1/signals — requires write permission
pub async fn post_signal_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SignalRequest>,
) -> Result<Json<signal::SignalResponse>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "write")?;

    let t0 = Instant::now();
    state.metrics.signals_total.fetch_add(1, Ordering::Relaxed);

    // v1.0.0 Phase 1: resolve Principal from the API-key → agent_id mapping
    // before constructing the Signal. None when the key has no agent_id
    // configured (pre-Phase-1 back-compat) or no IdentityStore is wired.
    let principal = auth::resolve_principal(&state, &headers).await;

    let source = signal::SignalSource::parse(body.source.as_deref(), signal::SignalSource::Http);
    let sig = signal::Signal::from_adapter_request(signal::AdapterRequest {
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
    })
    .with_principal_opt(principal);

    let signal_id = sig.id;
    let result = state.processor.process(sig).await;

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

    state.cache.lock().await.put(signal_id, response.clone());
    Ok(Json(response))
}

/// GET /v1/signals/:id — requires read permission
pub async fn get_signal_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<signal::SignalResponse>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "read")?;

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
pub async fn search_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<SearchRequest>,
) -> Result<Json<Vec<FactJson>>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "read")?;

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
pub async fn get_facts_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<FactJson>>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "read")?;

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
pub async fn get_namespaces_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Vec<NamespaceJson>>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "read")?;

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
pub async fn export_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ExportJson>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "read")?;

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
pub async fn import_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<ImportRequest>,
) -> Result<Json<ImportResponse>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "write")?;

    if body.dry_run {
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

/// GET /v1/schedules — list scheduled intents (requires read)
pub async fn list_schedules_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "read")?;

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

/// DELETE /v1/schedules/:id — cancel a scheduled intent (requires write)
pub async fn cancel_schedule_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "write")?;

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

/// Filter parameters for `GET /v1/events`. All fields optional.
/// Matches the spec in `docs/v1.0.0.md` §8.3.
#[derive(Debug, Default, serde::Deserialize)]
pub struct EventQuery {
    /// BrainEvent variant discriminant, e.g. `signal_received`, `tool_call_started`.
    pub kind: Option<String>,
    /// Filter to a specific tool_id (only applies to tool-bound BrainEvents).
    pub tool_id: Option<String>,
    /// Principal filter — accepted for forward compatibility; Phase 0 events
    /// do not yet carry a principal so this filter currently matches nothing
    /// when set. Implemented in Phase 1 (`docs/v1.0.0.md` §7).
    pub principal: Option<String>,
    /// RFC3339 timestamp; only events with `ts >= since` are forwarded.
    pub since: Option<chrono::DateTime<chrono::Utc>>,
}

impl EventQuery {
    /// Returns `true` if the event should be forwarded to the client.
    pub fn matches(&self, ev: &observe::BrainEvent) -> bool {
        if let Some(k) = &self.kind {
            if ev.kind() != k.as_str() {
                return false;
            }
        }
        if let Some(t) = &self.tool_id {
            if ev.tool_id() != Some(t.as_str()) {
                return false;
            }
        }
        if let Some(since) = self.since {
            let ts = brain_event_ts(ev);
            if ts < since {
                return false;
            }
        }
        // Principal filter: Phase 0 events don't carry a principal yet.
        if self.principal.is_some() {
            return false;
        }
        true
    }
}

fn brain_event_ts(ev: &observe::BrainEvent) -> chrono::DateTime<chrono::Utc> {
    use observe::BrainEvent::*;
    match ev {
        SignalReceived { ts, .. }
        | IntentClassified { ts, .. }
        | ReasoningStep { ts, .. }
        | ToolRouteResolved { ts, .. }
        | ConfirmationRequested { ts, .. }
        | ConfirmationResolved { ts, .. }
        | ToolCallStarted { ts, .. }
        | ToolCallFinished { ts, .. }
        | ReflexFired { ts, .. }
        | AuditAppended { ts, .. }
        | BudgetCrossed { ts, .. }
        | BreakerStateChange { ts, .. }
        | Error { ts, .. } => *ts,
    }
}

/// `GET /v1/events` — Server-Sent Events stream.
///
/// Surfaces three classes of events on a single connection:
/// - `brain_event` — structured `BrainEvent`s from the v1.0.0 Observer bus
///   (filterable via `?kind=`, `?tool_id=`, `?principal=`, `?since=`).
/// - `signal` — legacy `SignalProcessedEvent` (kept for existing consumers).
/// - `notification` — proactive notifications.
pub async fn sse_events_handler(
    State(state): State<Arc<AppState>>,
    Query(filter): Query<EventQuery>,
    headers: HeaderMap,
) -> Result<
    Sse<impl futures_core::Stream<Item = Result<Event, std::convert::Infallible>>>,
    (StatusCode, String),
> {
    auth::check_auth(&state, &headers, "read")?;

    let mut signal_rx = state.processor.subscribe_events();
    let mut notif_rx = state.processor.notification_router().map(|r| r.subscribe());
    let mut brain_rx = state.processor.subscribe_brain_events();

    let stream = async_stream::stream! {
        loop {
            tokio::select! {
                result = async {
                    match brain_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    match result {
                        Ok(ev) => {
                            if !filter.matches(&ev) { continue; }
                            yield Ok(Event::default()
                                .event("brain_event")
                                .json_data(&ev)
                                .unwrap_or_else(|_| Event::default().data("{}")));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(skipped = n, "SSE brain_event stream lagged");
                            yield Ok(Event::default()
                                .event("error")
                                .data(format!("{{\"lagged\":{n}}}")));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            brain_rx = None;
                        }
                    }
                }
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
                            notif_rx = None;
                        }
                    }
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

// ─── Webhook handlers ────────────────────────────────────────────────────────

/// POST /v1/webhooks/:id — ingest inbound messages from external platforms.
///
/// This endpoint finds the registered `WebhookInboundTransport` for the given
/// ID and delegates signature verification and message extraction to it.
pub async fn post_webhook_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let transport = match state.webhook_handlers.get(&id) {
        Some(t) => t,
        None => {
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(format!("Unknown transport ID: {id}"))
                .unwrap()
                .into_response();
        }
    };

    let resp = transport.handle_request(&headers, &body).await;

    Response::builder()
        .status(StatusCode::from_u16(resp.status).unwrap_or(StatusCode::OK))
        .header("content-type", resp.content_type)
        .body(resp.body)
        .unwrap()
        .into_response()
}
