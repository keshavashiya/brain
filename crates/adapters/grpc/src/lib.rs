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
    ConnectRequest, ConnectResponse, ReceiveRequest, SignalRequest as AgentSignalRequest,
    SignalResponse as AgentSignalResponse, SignalUpdate,
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
}

impl MemoryServiceImpl {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        Self { processor }
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

        let namespace = if req.namespace.is_empty() {
            None
        } else {
            Some(req.namespace.as_str())
        };

        let results = self.processor.search_facts(&req.query, top_k, namespace).await;

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
        let category = if req.category.is_empty() {
            "general"
        } else {
            &req.category
        };
        let namespace = if req.namespace.is_empty() {
            "personal"
        } else {
            &req.namespace
        };

        match self
            .processor
            .store_fact_direct(
                namespace,
                category,
                &req.subject,
                &req.predicate,
                &req.object,
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

        let namespace = if req.namespace.is_empty() {
            None
        } else {
            Some(req.namespace.as_str())
        };

        let raw_facts = if req.subject.is_empty() {
            self.processor.list_facts(namespace)
        } else {
            self.processor.facts_about(&req.subject)
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
        let req = request.into_inner();
        let source = parse_source(&req.source);

        let mut sig = Signal::new(
            source,
            if req.channel.is_empty() {
                "grpc"
            } else {
                &req.channel
            },
            if req.sender.is_empty() {
                "grpcclient"
            } else {
                &req.sender
            },
            req.content.clone(),
        );
        sig.metadata = req.metadata;

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
}

impl AgentServiceImpl {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        Self { processor }
    }
}

/// Stream type alias for the server-streaming `ReceiveSignals` RPC.
type SignalUpdateStream =
    Pin<Box<dyn Stream<Item = Result<SignalUpdate, Status>> + Send + 'static>>;

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
                "Welcome, {} ({})! Brain gRPC session established.",
                req.agent_id, req.agent_type
            ),
        }))
    }

    /// Send a signal and receive a single response.
    async fn send_signal(
        &self,
        request: Request<AgentSignalRequest>,
    ) -> Result<Response<AgentSignalResponse>, Status> {
        let req = request.into_inner();
        let source = parse_source(&req.source);

        let mut sig = Signal::new(
            source,
            if req.channel.is_empty() {
                "grpc"
            } else {
                &req.channel
            },
            if req.sender.is_empty() {
                "agent"
            } else {
                &req.sender
            },
            req.content.clone(),
        );
        sig.metadata = req.metadata;

        match self.processor.process(sig).await {
            Ok(resp) => Ok(Response::new(AgentSignalResponse {
                signal_id: resp.signal_id.to_string(),
                status: format!("{:?}", resp.status),
                response: response_to_string(resp.response),
                facts_used: resp.memory_context.facts_used as u32,
                episodes_used: resp.memory_context.episodes_used as u32,
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    type ReceiveSignalsStream = SignalUpdateStream;

    /// Subscribe to a stream of updates for a session.
    ///
    /// Currently sends a single "connected" event and then the stream ends.
    /// In a full implementation this would fan-out events to subscribers.
    async fn receive_signals(
        &self,
        request: Request<ReceiveRequest>,
    ) -> Result<Response<Self::ReceiveSignalsStream>, Status> {
        let req = request.into_inner();
        let session_id = req.session_id.clone();

        tracing::debug!(session_id = %session_id, "ReceiveSignals stream opened");

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let now = chrono::Utc::now().to_rfc3339();

        tokio::spawn(async move {
            // Send an initial "connected" event
            let _ = tx
                .send(Ok(SignalUpdate {
                    event_type: "connected".to_string(),
                    content: format!("Session {session_id} active"),
                    timestamp: now,
                }))
                .await;
            // Stream ends naturally when tx is dropped
        });

        let stream: SignalUpdateStream = Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
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

    let api_keys: Vec<String> = processor
        .config()
        .access
        .api_keys
        .iter()
        .map(|k| k.key.clone())
        .collect();

    let auth_keys = Arc::new(api_keys);

    let memory_svc = MemoryServiceServer::with_interceptor(
        MemoryServiceImpl::new(processor.clone()),
        {
            let keys = Arc::clone(&auth_keys);
            move |req: Request<()>| auth_interceptor(req, &keys)
        },
    );
    let agent_svc = AgentServiceServer::with_interceptor(
        AgentServiceImpl::new(processor),
        {
            let keys = Arc::clone(&auth_keys);
            move |req: Request<()>| auth_interceptor(req, &keys)
        },
    );

    tracing::info!("Brain gRPC server listening on {addr}");

    Server::builder()
        .add_service(memory_svc)
        .add_service(agent_svc)
        .serve(addr)
        .await?;

    Ok(())
}

/// Tonic interceptor that validates API key authentication.
///
/// Accepts the key from either `x-api-key` or `authorization` (Bearer) metadata.
/// Returns `UNAUTHENTICATED` if no valid key is found. If no keys are configured,
/// all requests are allowed (open mode).
fn auth_interceptor(req: Request<()>, api_keys: &[String]) -> Result<Request<()>, Status> {
    // If no keys configured, allow all (open mode)
    if api_keys.is_empty() {
        return Ok(req);
    }

    let metadata = req.metadata();

    // Check x-api-key header
    if let Some(key) = metadata.get("x-api-key") {
        if let Ok(key_str) = key.to_str() {
            if api_keys.iter().any(|k| k == key_str) {
                return Ok(req);
            }
        }
    }

    // Check authorization header (Bearer token)
    if let Some(auth) = metadata.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let token = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
            if api_keys.iter().any(|k| k == token) {
                return Ok(req);
            }
        }
    }

    Err(Status::unauthenticated("Missing or invalid API key"))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_source(s: &str) -> SignalSource {
    match s {
        "grpc" | "" => SignalSource::Grpc,
        "http" => SignalSource::Http,
        "cli" => SignalSource::Cli,
        "ws" | "websocket" => SignalSource::WebSocket,
        "mcp" => SignalSource::Mcp,
        _ => SignalSource::Grpc,
    }
}

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

    #[test]
    fn test_parse_source() {
        assert_eq!(parse_source("grpc"), SignalSource::Grpc);
        assert_eq!(parse_source(""), SignalSource::Grpc);
        assert_eq!(parse_source("http"), SignalSource::Http);
        assert_eq!(parse_source("cli"), SignalSource::Cli);
        assert_eq!(parse_source("ws"), SignalSource::WebSocket);
        assert_eq!(parse_source("mcp"), SignalSource::Mcp);
        assert_eq!(parse_source("unknown"), SignalSource::Grpc);
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
