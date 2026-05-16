//! # Brain gRPC Adapter
//!
//! Exposes Brain's signal processing pipeline over gRPC using tonic.
//!
//! ## Services
//! - `MemoryService` — semantic memory search, store, list, and signal streaming
//! - `AgentService`  — agent connect, send signal, receive streaming updates
//!
//! ## Ports
//! - Default gRPC port: **19792**

use std::{net::SocketAddr, pin::Pin, sync::Arc};

use tokio_stream::Stream;
use tonic::{transport::Server, Request, Response, Status};
use uuid::Uuid;

use signal::{Signal, SignalSource};

/// Convert protobuf empty string to Option (proto3 defaults strings to "").
fn non_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

// ── Generated protobuf types ──────────────────────────────────────────────────

/// Types and server/client stubs generated from `proto/memory.proto`.
pub mod memory_proto {
    tonic::include_proto!("brain.memory");
}

/// Types and server/client stubs generated from `proto/agent.proto`.
pub mod agent_proto {
    tonic::include_proto!("brain.agent");
}

use agent_proto::{
    agent_service_server::{AgentService, AgentServiceServer},
    BrainEventMessage, BrainEventsRequest, ConnectRequest, ConnectResponse, ReceiveRequest,
    SignalRequest as AgentSignalRequest, SignalResponse as AgentSignalResponse, SignalUpdate,
};
use memory_proto::{
    memory_service_server::{MemoryService, MemoryServiceServer},
    Fact, GetFactsRequest, GetFactsResponse, SearchRequest, SearchResponse, SignalEvent,
    SignalRequest as MemorySignalRequest, StoreRequest, StoreResponse,
};

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum GrpcAdapterError {
    #[error("Server error: {0}")]
    Server(String),
}

// ── MemoryService implementation ──────────────────────────────────────────────

/// gRPC implementation of `MemoryService`.
pub struct MemoryServiceImpl {
    processor: Arc<signal::SignalProcessor>,
    api_keys: Arc<Vec<brain_core::ApiKeyConfig>>,
}

impl MemoryServiceImpl {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        let api_keys = Arc::new(processor.config().access.api_keys.clone());
        Self {
            processor,
            api_keys,
        }
    }

    /// Resolve the `Principal` from the request metadata. Returns `None`
    /// for keys without `agent_id` (back-compat).
    async fn resolve_principal<T>(&self, req: &Request<T>) -> Option<identity::Principal> {
        resolve_principal_from_metadata(req, &self.api_keys, self.processor.as_ref()).await
    }
}

/// Stream type alias for the server-streaming `StreamSignals` RPC.
type SignalEventStream = Pin<Box<dyn Stream<Item = Result<SignalEvent, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl MemoryService for MemoryServiceImpl {
    /// Search semantic memory using a text query.
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let top_k = if req.top_k == 0 {
            10
        } else {
            req.top_k as usize
        };

        let namespace = non_empty(req.namespace);

        let results = self
            .processor
            .search_facts(&req.query, top_k, namespace.as_deref())
            .await;

        let facts = results
            .into_iter()
            .map(|r| Fact {
                id: r.fact.id,
                category: r.fact.category,
                subject: r.fact.subject,
                predicate: r.fact.predicate,
                object: r.fact.object,
                confidence: r.fact.confidence,
                distance: r.distance,
            })
            .collect();

        Ok(Response::new(SearchResponse { facts }))
    }

    /// Store a structured fact in semantic memory.
    async fn store(
        &self,
        request: Request<StoreRequest>,
    ) -> Result<Response<StoreResponse>, Status> {
        let req = request.into_inner();
        let category = non_empty(req.category).unwrap_or_else(|| "general".to_string());
        let namespace = non_empty(req.namespace).unwrap_or_else(|| "personal".to_string());

        match self
            .processor
            .store_fact_direct(
                &namespace,
                &category,
                &req.subject,
                &req.predicate,
                &req.object,
                None,
            )
            .await
        {
            Ok(fact_id) => Ok(Response::new(StoreResponse {
                fact_id,
                success: true,
                message: "Fact stored successfully".to_string(),
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    /// List all active facts, optionally filtered by subject and/or namespace.
    async fn get_facts(
        &self,
        request: Request<GetFactsRequest>,
    ) -> Result<Response<GetFactsResponse>, Status> {
        let req = request.into_inner();

        let namespace = non_empty(req.namespace);

        let raw_facts = if req.subject.is_empty() {
            self.processor.list_facts(namespace.as_deref())
        } else {
            self.processor
                .facts_about(&req.subject, namespace.as_deref())
        };

        let facts = raw_facts
            .into_iter()
            .map(|f| Fact {
                id: f.id,
                category: f.category,
                subject: f.subject,
                predicate: f.predicate,
                object: f.object,
                confidence: f.confidence,
                distance: 0.0,
            })
            .collect();

        Ok(Response::new(GetFactsResponse { facts }))
    }

    type StreamSignalsStream = SignalEventStream;

    /// Process a signal and stream the response event(s).
    async fn stream_signals(
        &self,
        request: Request<MemorySignalRequest>,
    ) -> Result<Response<Self::StreamSignalsStream>, Status> {
        let principal = self.resolve_principal(&request).await;
        let req = request.into_inner();
        let source = SignalSource::parse(Some(&req.source), SignalSource::Grpc);

        let sig = Signal::from_adapter_request(signal::AdapterRequest {
            source,
            content: req.content,
            channel: non_empty(req.channel),
            sender: non_empty(req.sender),
            metadata: Some(req.metadata),
            namespace: non_empty(req.namespace),
            agent: non_empty(req.agent),
            session_id: non_empty(req.session_id),
            default_channel: "grpc".to_string(),
            default_sender: "grpcclient".to_string(),
        })
        .with_principal_opt(principal);

        let processor = self.processor.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        tokio::spawn(async move {
            match processor.process(sig).await {
                Ok(resp) => {
                    let event = SignalEvent {
                        signal_id: resp.signal_id.to_string(),
                        status: format!("{:?}", resp.status),
                        response: response_to_string(resp.response),
                        facts_used: resp.memory_context.facts_used as u32,
                        episodes_used: resp.memory_context.episodes_used as u32,
                        session_id: resp.session_id.unwrap_or_default(),
                    };
                    let _ = tx.send(Ok(event)).await;
                }
                Err(e) => {
                    let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                }
            }
        });

        let stream: SignalEventStream = Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }
}

// ── AgentService implementation ───────────────────────────────────────────────

/// gRPC implementation of `AgentService`.
pub struct AgentServiceImpl {
    processor: Arc<signal::SignalProcessor>,
    api_keys: Arc<Vec<brain_core::ApiKeyConfig>>,
}

impl AgentServiceImpl {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        let api_keys = Arc::new(processor.config().access.api_keys.clone());
        Self {
            processor,
            api_keys,
        }
    }

    async fn resolve_principal<T>(&self, req: &Request<T>) -> Option<identity::Principal> {
        resolve_principal_from_metadata(req, &self.api_keys, self.processor.as_ref()).await
    }
}

/// Stream type alias for the server-streaming `ReceiveSignals` RPC.
type SignalUpdateStream =
    Pin<Box<dyn Stream<Item = Result<SignalUpdate, Status>> + Send + 'static>>;

/// Stream type alias for the server-streaming `BrainEvents` RPC.
type BrainEventStream =
    Pin<Box<dyn Stream<Item = Result<BrainEventMessage, Status>> + Send + 'static>>;

/// Predicate matching a `BrainEvent` against the gRPC filter fields.
fn brain_event_matches(ev: &observe::BrainEvent, filter: &BrainEventsRequest) -> bool {
    if !filter.kind.is_empty() && ev.kind() != filter.kind {
        return false;
    }
    if !filter.tool_id.is_empty() && ev.tool_id() != Some(filter.tool_id.as_str()) {
        return false;
    }
    if !filter.principal.is_empty() {
        // Bus events do not yet carry a principal; the filter rejects
        // everything when set.
        return false;
    }
    if !filter.since.is_empty() {
        let Ok(since) = chrono::DateTime::parse_from_rfc3339(&filter.since) else {
            return false;
        };
        let since = since.with_timezone(&chrono::Utc);
        if brain_event_ts(ev) < since {
            return false;
        }
    }
    true
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
        | Error { ts, .. }
        | TerminalSessionOpened { ts, .. }
        | TerminalSessionClosed { ts, .. }
        | TaskStateChange { ts, .. } => *ts,
    }
}

#[tonic::async_trait]
impl AgentService for AgentServiceImpl {
    /// Establish a session and return a session ID.
    async fn connect(
        &self,
        request: Request<ConnectRequest>,
    ) -> Result<Response<ConnectResponse>, Status> {
        let req = request.into_inner();
        let session_id = Uuid::new_v4().to_string();

        tracing::info!(
            agent_id = %req.agent_id,
            agent_type = %req.agent_type,
            session_id = %session_id,
            "gRPC agent connected"
        );

        Ok(Response::new(ConnectResponse {
            session_id,
            accepted: true,
            message: format!(
                "Synapse established — welcome, {} ({}).",
                req.agent_id, req.agent_type
            ),
        }))
    }

    /// Send a signal and receive a single response.
    async fn send_signal(
        &self,
        request: Request<AgentSignalRequest>,
    ) -> Result<Response<AgentSignalResponse>, Status> {
        let principal = self.resolve_principal(&request).await;
        let req = request.into_inner();
        let source = SignalSource::parse(Some(&req.source), SignalSource::Grpc);

        let sig = Signal::from_adapter_request(signal::AdapterRequest {
            source,
            content: req.content,
            channel: non_empty(req.channel),
            sender: non_empty(req.sender),
            metadata: Some(req.metadata),
            namespace: non_empty(req.namespace),
            agent: non_empty(req.agent),
            session_id: non_empty(req.session_id),
            default_channel: "grpc".to_string(),
            default_sender: "agent".to_string(),
        })
        .with_principal_opt(principal);

        match self.processor.process(sig).await {
            Ok(resp) => Ok(Response::new(AgentSignalResponse {
                signal_id: resp.signal_id.to_string(),
                status: format!("{:?}", resp.status),
                response: response_to_string(resp.response),
                facts_used: resp.memory_context.facts_used as u32,
                episodes_used: resp.memory_context.episodes_used as u32,
                session_id: resp.session_id.unwrap_or_default(),
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    type ReceiveSignalsStream = SignalUpdateStream;

    /// Subscribe to a stream of updates for a session.
    async fn receive_signals(
        &self,
        request: Request<ReceiveRequest>,
    ) -> Result<Response<Self::ReceiveSignalsStream>, Status> {
        let req = request.into_inner();
        let session_id = req.session_id.clone();
        let mut events = self.processor.subscribe_events();

        tracing::debug!(session_id = %session_id, "ReceiveSignals stream opened");

        let (tx, rx) = tokio::sync::mpsc::channel(32);
        let now = chrono::Utc::now().to_rfc3339();

        tokio::spawn(async move {
            // Send an initial "connected" event
            if tx
                .send(Ok(SignalUpdate {
                    event_type: "connected".to_string(),
                    content: format!("Session {session_id} active"),
                    timestamp: now,
                }))
                .await
                .is_err()
            {
                return;
            }

            loop {
                match events.recv().await {
                    Ok(event) => {
                        let content = format!(
                            "[{}:{}] {}",
                            event.namespace, event.signal_id, event.response
                        );
                        if tx
                            .send(Ok(SignalUpdate {
                                event_type: "processed".to_string(),
                                content,
                                timestamp: event.timestamp.to_rfc3339(),
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        if tx
                            .send(Ok(SignalUpdate {
                                event_type: "lagged".to_string(),
                                content: format!("Dropped {skipped} events"),
                                timestamp: chrono::Utc::now().to_rfc3339(),
                            }))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        let stream: SignalUpdateStream = Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }

    type BrainEventsStream = BrainEventStream;

    /// Subscribe to the v1.0.0 BrainEvent bus, mirroring the SSE/WS surfaces.
    /// Each emitted message carries the BrainEvent as JSON in `event_json`.
    async fn brain_events(
        &self,
        request: Request<BrainEventsRequest>,
    ) -> Result<Response<Self::BrainEventsStream>, Status> {
        let filter = request.into_inner();
        let Some(mut rx) = self.processor.subscribe_brain_events() else {
            return Err(Status::failed_precondition(
                "observability bus not wired on this SignalProcessor",
            ));
        };

        tracing::debug!(?filter.kind, ?filter.tool_id, "BrainEvents stream opened");

        let (tx, out) = tokio::sync::mpsc::channel::<Result<BrainEventMessage, Status>>(64);
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        if !brain_event_matches(&ev, &filter) {
                            continue;
                        }
                        let event_json = match serde_json::to_string(&ev) {
                            Ok(s) => s,
                            Err(e) => {
                                tracing::warn!("BrainEvents serialise failed: {e}");
                                continue;
                            }
                        };
                        if tx.send(Ok(BrainEventMessage { event_json })).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_n)) => {
                        // Client is slow; skip silently. (HTTP SSE / WS surfaces
                        // emit a Lagged marker; gRPC's strongly-typed stream has
                        // no idle frame — silent drop keeps the protocol clean.)
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        let stream: BrainEventStream = Box::pin(tokio_stream::wrappers::ReceiverStream::new(out));
        Ok(Response::new(stream))
    }
}

// ── Server ────────────────────────────────────────────────────────────────────

/// Start the gRPC server, binding to `host:port`.
///
/// Registers both `MemoryService` and `AgentService`.
/// All requests are authenticated via `x-api-key` or `authorization` metadata.
/// Blocks until the server shuts down.
pub async fn serve(
    processor: Arc<signal::SignalProcessor>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{host}:{port}").parse()?;

    let auth_keys = Arc::new(processor.config().access.api_keys.clone());

    let memory_svc =
        MemoryServiceServer::with_interceptor(MemoryServiceImpl::new(processor.clone()), {
            let keys = Arc::clone(&auth_keys);
            move |req: Request<()>| auth_interceptor(req, &keys)
        });
    let agent_svc = AgentServiceServer::with_interceptor(AgentServiceImpl::new(processor), {
        let keys = Arc::clone(&auth_keys);
        move |req: Request<()>| auth_interceptor(req, &keys)
    });

    tracing::info!("Synapse gRPC online at {addr}");

    Server::builder()
        .add_service(memory_svc)
        .add_service(agent_svc)
        .serve(addr)
        .await?;

    Ok(())
}

/// Tonic interceptor that validates API key authentication with permission checking.
///
/// Accepts the key from either `x-api-key` or `authorization` (Bearer) metadata.
/// Returns `UNAUTHENTICATED` / `PERMISSION_DENIED` if no valid key is found.
/// If no keys are configured, all requests are allowed (open mode).
fn auth_interceptor(
    req: Request<()>,
    api_keys: &[brain_core::ApiKeyConfig],
) -> Result<Request<()>, Status> {
    let metadata = req.metadata();

    // Extract key from x-api-key or authorization header
    let provided_key = metadata
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            metadata.get("authorization").and_then(|v| {
                v.to_str()
                    .ok()
                    .and_then(|s| brain_core::auth::extract_bearer_from_value(s).or(Some(s)))
            })
        });

    // gRPC requests can both read and write, so require write permission.
    let result = brain_core::check_auth(api_keys, provided_key, "write");
    match result {
        brain_core::AuthResult::Open | brain_core::AuthResult::Allowed => Ok(req),
        brain_core::AuthResult::InsufficientPermission => Err(Status::permission_denied(
            result.error_message("write").unwrap_or_default(),
        )),
        _ => Err(Status::unauthenticated(
            result.error_message("write").unwrap_or_default(),
        )),
    }
}

/// Resolve the `Principal` for a gRPC request by reading the key from
/// `x-api-key` / `authorization` metadata, looking up its configured
/// `agent_id`, and asking the `IdentityStore` for the principal.
/// Returns `None` when any step is missing — Signal.principal then stays
/// `None` and the pipeline's identity gate is skipped (back-compat).
async fn resolve_principal_from_metadata<T>(
    req: &Request<T>,
    api_keys: &[brain_core::ApiKeyConfig],
    processor: &signal::SignalProcessor,
) -> Option<identity::Principal> {
    let metadata = req.metadata();
    let key = metadata
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            metadata.get("authorization").and_then(|v| {
                v.to_str()
                    .ok()
                    .and_then(|s| brain_core::auth::extract_bearer_from_value(s).or(Some(s)))
            })
        })?;
    let agent_id = api_keys
        .iter()
        .find(|k| k.key == key)
        .and_then(|k| k.agent_id.clone())?;
    let store = processor.identity_store()?;
    store
        .principal_for(&identity::AgentHint::AgentId(agent_id.into()))
        .await
        .ok()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn response_to_string(content: signal::ResponseContent) -> String {
    match content {
        signal::ResponseContent::Text(t) => t,
        signal::ResponseContent::Json(v) => v.to_string(),
        signal::ResponseContent::Error(e) => e,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a SignalProcessor backed by a temp directory.
    async fn make_processor() -> Arc<signal::SignalProcessor> {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let proc = signal::SignalProcessor::new(config).await.unwrap();
        // Keep temp alive by leaking (fine for tests)
        std::mem::forget(temp);
        Arc::new(proc)
    }

    #[tokio::test]
    async fn test_memory_service_get_facts_empty() {
        let processor = make_processor().await;
        let svc = MemoryServiceImpl::new(processor);

        let req = Request::new(GetFactsRequest {
            subject: String::new(),
            namespace: String::new(),
        });
        let resp = svc.get_facts(req).await.unwrap();
        assert!(resp.into_inner().facts.is_empty());
    }

    #[tokio::test]
    async fn test_memory_service_get_facts_with_subject_filter() {
        let processor = make_processor().await;
        let svc = MemoryServiceImpl::new(processor);

        let req = Request::new(GetFactsRequest {
            subject: "rust".to_string(),
            namespace: String::new(),
        });
        let resp = svc.get_facts(req).await.unwrap();
        // No facts stored yet — result should be empty
        assert!(resp.into_inner().facts.is_empty());
    }

    #[tokio::test]
    async fn test_memory_service_search_empty() {
        let processor = make_processor().await;
        let svc = MemoryServiceImpl::new(processor);

        let req = Request::new(SearchRequest {
            query: "what is Rust".to_string(),
            top_k: 5,
            namespace: String::new(),
        });
        let resp = svc.search(req).await.unwrap();
        // No facts in empty store
        assert!(resp.into_inner().facts.is_empty());
    }

    #[tokio::test]
    async fn test_memory_service_search_default_top_k() {
        let processor = make_processor().await;
        let svc = MemoryServiceImpl::new(processor);

        // top_k = 0 should default to 10 internally without panicking
        let req = Request::new(SearchRequest {
            query: "test".to_string(),
            top_k: 0,
            namespace: String::new(),
        });
        let resp = svc.search(req).await.unwrap();
        assert!(resp.into_inner().facts.is_empty());
    }

    #[tokio::test]
    async fn test_agent_service_connect() {
        let processor = make_processor().await;
        let svc = AgentServiceImpl::new(processor);

        let req = Request::new(ConnectRequest {
            agent_id: "testagent".to_string(),
            agent_type: "assistant".to_string(),
        });
        let resp = svc.connect(req).await.unwrap();
        let inner = resp.into_inner();
        assert!(inner.accepted);
        assert!(!inner.session_id.is_empty());
        // session_id should be a valid UUID
        assert!(Uuid::parse_str(&inner.session_id).is_ok());
    }

    #[tokio::test]
    async fn test_agent_service_send_signal() {
        let processor = make_processor().await;
        let svc = AgentServiceImpl::new(processor);

        let req = Request::new(AgentSignalRequest {
            source: "grpc".to_string(),
            channel: "test".to_string(),
            sender: "testagent".to_string(),
            content: "Remember that Rust is fast".to_string(),
            metadata: std::collections::HashMap::new(),
            namespace: String::new(),
            agent: String::new(),
            session_id: String::new(),
        });
        let resp = svc.send_signal(req).await.unwrap();
        let inner = resp.into_inner();
        // Signal was processed — should have a valid UUID and an "Ok" status
        assert!(!inner.signal_id.is_empty());
        assert!(Uuid::parse_str(&inner.signal_id).is_ok());
        assert_eq!(inner.status, "Ok");
    }

    #[tokio::test]
    async fn test_memory_stream_signals() {
        let processor = make_processor().await;
        let svc = MemoryServiceImpl::new(processor);

        let req = Request::new(MemorySignalRequest {
            source: "grpc".to_string(),
            channel: "test".to_string(),
            sender: "testclient".to_string(),
            content: "Remember that Brain is the central AI OS".to_string(),
            metadata: std::collections::HashMap::new(),
            namespace: String::new(),
            agent: String::new(),
            session_id: String::new(),
        });
        let resp = svc.stream_signals(req).await.unwrap();
        let mut stream = resp.into_inner();

        use tokio_stream::StreamExt;
        let first = stream.next().await;
        // Should receive exactly one event
        assert!(first.is_some());
        let event = first.unwrap().unwrap();
        assert!(!event.signal_id.is_empty());
        assert_eq!(event.status, "Ok");
    }

    #[tokio::test]
    async fn test_agent_receive_signals_sends_connected_event() {
        let processor = make_processor().await;
        let svc = AgentServiceImpl::new(processor);

        let session_id = Uuid::new_v4().to_string();
        let req = Request::new(ReceiveRequest {
            session_id: session_id.clone(),
        });
        let resp = svc.receive_signals(req).await.unwrap();
        let mut stream = resp.into_inner();

        use tokio_stream::StreamExt;
        let first = stream.next().await;
        assert!(first.is_some());
        let update = first.unwrap().unwrap();
        assert_eq!(update.event_type, "connected");
        assert!(update.content.contains(&session_id));
    }

    #[tokio::test]
    async fn test_agent_receive_signals_fanout_after_send_signal() {
        use tokio::time::{timeout, Duration};
        use tokio_stream::StreamExt;

        let processor = make_processor().await;
        let agent_svc = AgentServiceImpl::new(processor.clone());

        let recv_req = Request::new(ReceiveRequest {
            session_id: Uuid::new_v4().to_string(),
        });
        let recv_resp = agent_svc.receive_signals(recv_req).await.unwrap();
        let mut stream = recv_resp.into_inner();

        // Initial connected event
        let connected = stream.next().await.unwrap().unwrap();
        assert_eq!(connected.event_type, "connected");

        let send_req = Request::new(AgentSignalRequest {
            source: "grpc".to_string(),
            channel: "test".to_string(),
            sender: "testagent".to_string(),
            content: "Remember that fanout works".to_string(),
            metadata: std::collections::HashMap::new(),
            namespace: "personal".to_string(),
            agent: String::new(),
            session_id: String::new(),
        });
        let send_resp = agent_svc.send_signal(send_req).await.unwrap().into_inner();
        assert_eq!(send_resp.status, "Ok");

        let next = timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("expected fanout event within timeout");
        let update = next.expect("stream closed").expect("stream error");
        assert_eq!(update.event_type, "processed");
        assert!(update.content.contains("fanout works"));
    }

    #[test]
    fn test_parse_source() {
        assert_eq!(
            SignalSource::parse(Some("grpc"), SignalSource::Grpc),
            SignalSource::Grpc
        );
        assert_eq!(
            SignalSource::parse(Some(""), SignalSource::Grpc),
            SignalSource::Grpc
        );
        assert_eq!(
            SignalSource::parse(Some("http"), SignalSource::Grpc),
            SignalSource::Http
        );
        assert_eq!(
            SignalSource::parse(Some("cli"), SignalSource::Grpc),
            SignalSource::Cli
        );
        assert_eq!(
            SignalSource::parse(Some("ws"), SignalSource::Grpc),
            SignalSource::WebSocket
        );
        assert_eq!(
            SignalSource::parse(Some("mcp"), SignalSource::Grpc),
            SignalSource::Mcp
        );
        assert_eq!(
            SignalSource::parse(Some("unknown"), SignalSource::Grpc),
            SignalSource::Grpc
        );
    }

    #[test]
    fn test_non_empty() {
        assert_eq!(non_empty(String::new()), None);
        assert_eq!(non_empty("hello".to_string()), Some("hello".to_string()));
    }

    #[test]
    fn test_response_to_string() {
        assert_eq!(
            response_to_string(signal::ResponseContent::Text("hello".to_string())),
            "hello"
        );
        assert_eq!(
            response_to_string(signal::ResponseContent::Error("err".to_string())),
            "err"
        );
        let json = serde_json::json!({"key": "val"});
        let s = response_to_string(signal::ResponseContent::Json(json));
        assert!(s.contains("key"));
    }
}
