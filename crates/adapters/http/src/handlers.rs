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
use crate::validate;

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

    validate::check_content(&body.content)?;
    if let Some(ns) = body.namespace.as_deref() {
        validate::check_short_ident("namespace", ns)?;
    }
    if let Some(agent) = body.agent.as_deref() {
        validate::check_short_ident("agent", agent)?;
    }
    if let Some(sid) = body.session_id.as_deref() {
        validate::check_short_ident("session_id", sid)?;
    }
    if let Some(sender) = body.sender.as_deref() {
        validate::check_short_ident("sender", sender)?;
    }
    if let Some(channel) = body.channel.as_deref() {
        validate::check_short_ident("channel", channel)?;
    }
    if let Some(source) = body.source.as_deref() {
        validate::check_short_ident("source", source)?;
    }

    let t0 = Instant::now();
    state.metrics.signals_total.fetch_add(1, Ordering::Relaxed);

    // Resolve Principal from the API-key → agent_id mapping before
    // constructing the Signal. None when the key has no agent_id configured
    // (back-compat) or no IdentityStore is wired.
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
            let public = e.to_public();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({
                    "error": public.code,
                    "message": public.message,
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

    validate::check_query(&body.query)?;
    validate::check_top_k(body.top_k)?;
    if let Some(ns) = body.namespace.as_deref() {
        validate::check_short_ident("namespace", ns)?;
    }

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
    // Default page size protects an unscoped GET from streaming the
    // whole fact table on a multi-thousand-row store. Callers can pass
    // `?limit=N&offset=M` (limit capped at 1000) to walk pages, or
    // `?limit=0` to opt back into the unbounded list.
    const DEFAULT_LIMIT: usize = 100;
    const MAX_LIMIT: usize = 1000;
    let limit_param = params.get("limit").and_then(|s| s.parse::<usize>().ok());
    let offset = params
        .get("offset")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let limit = match limit_param {
        Some(0) => None,
        Some(n) => Some(n.min(MAX_LIMIT)),
        None => Some(DEFAULT_LIMIT),
    };
    let facts = state
        .processor
        .list_facts_paginated(namespace, limit, offset)
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

/// GET /v1/memory/export — requires `export` permission (Issue 123).
///
/// Bulk export is gated behind its own scope so a daily `read` key can't
/// be used to exfil the entire memory store. Mint an API key with
/// `permissions: ["export"]` (or `["admin"]`) to call this endpoint.
pub async fn export_memory_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ExportJson>, (StatusCode, String)> {
    auth::check_auth(&state, &headers, "export")?;

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
#[derive(Debug, Default, serde::Deserialize)]
pub struct EventQuery {
    /// BrainEvent variant discriminant, e.g. `signal_received`, `tool_call_started`.
    pub kind: Option<String>,
    /// Filter to a specific tool_id (only applies to tool-bound BrainEvents).
    pub tool_id: Option<String>,
    /// Correlation-id filter. Every event in one signal flow shares the
    /// originating signal's id via [`observe::BrainEvent::id`], so
    /// `?correlation=<uuid>` reconstructs a single turn end-to-end. Matched as
    /// a string so a malformed value simply matches nothing rather than 400.
    pub correlation: Option<String>,
    /// Principal filter — accepted for forward compatibility; current events
    /// do not yet carry a principal on the bus, so this filter matches
    /// nothing when set.
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
        if let Some(c) = &self.correlation {
            if ev.id().to_string() != *c {
                return false;
            }
        }
        if let Some(since) = self.since {
            let ts = brain_event_ts(ev);
            if ts < since {
                return false;
            }
        }
        // Principal filter: bus events don't carry a principal yet.
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
        | ToolRouteResolved { ts, .. }
        | ConfirmationRequested { ts, .. }
        | ConfirmationResolved { ts, .. }
        | ToolCallStarted { ts, .. }
        | ToolCallFinished { ts, .. }
        | ReflexFired { ts, .. }
        | AuditAppended { ts, .. }
        | BudgetCrossed { ts, .. }
        | ResourcePressure { ts, .. }
        | BreakerStateChange { ts, .. }
        | Error { ts, .. }
        | TerminalSessionOpened { ts, .. }
        | TerminalSessionClosed { ts, .. }
        | TaskStateChange { ts, .. }
        | ConnectivityChanged { ts, .. }
        | ServiceHealthChanged { ts, .. } => *ts,
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
    // Issue 131: when on, content previews are scrubbed before serialization
    // so a read-scoped observer only sees the event shape.
    let redact = state.processor.config().adapters.http.sse_redact_previews;

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
                            let response_field = if redact {
                                serde_json::json!({ "redacted": true, "len": event.response.len() })
                            } else {
                                serde_json::Value::String(event.response.clone())
                            };
                            let payload = serde_json::json!({
                                "type": "signal",
                                "signal_id": event.signal_id.to_string(),
                                "source": format!("{:?}", event.source),
                                "status": format!("{:?}", event.status),
                                "response": response_field,
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
                            let content_field = if redact {
                                serde_json::json!({ "redacted": true, "len": notification.content.len() })
                            } else {
                                serde_json::Value::String(notification.content.clone())
                            };
                            let payload = serde_json::json!({
                                "type": "proactive",
                                "content": content_field,
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
///
/// Per Issue 52: when the transport has no signature verifier configured
/// (`PreparedVerifier::None`), we additionally require a Brain
/// `Authorization: Bearer <api-key>` with `write` permission so the
/// endpoint can't be abused as an open inbound message queue. Verified
/// transports (HMAC, Ed25519) keep their existing flow — the verifier
/// itself is the auth gate.
pub async fn post_webhook_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let transport = match state.webhook_handlers.get(&id) {
        Some(t) => t,
        None => {
            return build_webhook_response(
                StatusCode::NOT_FOUND,
                None,
                format!("Unknown transport ID: {id}"),
            )
            .into_response();
        }
    };

    if !transport.has_verifier() {
        if let Err((status, msg)) = auth::check_auth(&state, &headers, "write") {
            tracing::warn!(
                transport = %id,
                status = %status,
                "webhook rejected — verifier-less transport requires Bearer auth"
            );
            return build_webhook_response(status, None, msg).into_response();
        }
    }

    let resp = transport.handle_request(&headers, &body).await;
    let status = StatusCode::from_u16(resp.status).unwrap_or(StatusCode::OK);
    build_webhook_response(status, Some(resp.content_type), resp.body).into_response()
}

/// Build a webhook response, falling back to a 500 if the supplied
/// status/headers/body are unrepresentable (e.g. invalid `Content-Type` value).
fn build_webhook_response(
    status: StatusCode,
    content_type: Option<String>,
    body: String,
) -> Response<String> {
    let mut builder = Response::builder().status(status);
    if let Some(ct) = content_type {
        builder = builder.header("content-type", ct);
    }
    builder.body(body).unwrap_or_else(|err| {
        tracing::error!(error = %err, "failed to build webhook response");
        let mut fallback = Response::new("Internal Server Error".to_string());
        *fallback.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        fallback
    })
}

#[cfg(test)]
mod event_query_tests {
    use super::EventQuery;
    use chrono::Utc;
    use uuid::Uuid;

    fn signal_received(id: Uuid) -> observe::BrainEvent {
        observe::BrainEvent::SignalReceived {
            id,
            signal: observe::SignalSummary {
                source: "cli".into(),
                channel: "c".into(),
                sender: "s".into(),
                namespace: "personal".into(),
                content_preview: "hi".into(),
            },
            ts: Utc::now(),
        }
    }

    #[test]
    fn correlation_filter_matches_only_its_flow() {
        let mine = Uuid::new_v4();
        let other = Uuid::new_v4();
        let filter = EventQuery {
            correlation: Some(mine.to_string()),
            ..Default::default()
        };
        assert!(filter.matches(&signal_received(mine)));
        assert!(!filter.matches(&signal_received(other)));
    }

    #[test]
    fn malformed_correlation_matches_nothing_rather_than_erroring() {
        let filter = EventQuery {
            correlation: Some("not-a-uuid".into()),
            ..Default::default()
        };
        assert!(!filter.matches(&signal_received(Uuid::new_v4())));
    }

    #[test]
    fn absent_correlation_does_not_constrain() {
        let filter = EventQuery::default();
        assert!(filter.matches(&signal_received(Uuid::new_v4())));
    }
}
