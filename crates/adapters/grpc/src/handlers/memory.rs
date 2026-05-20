//! `MemoryService` RPC handlers — search, store, list, stream_signals.

use std::pin::Pin;

use tokio_stream::Stream;
use tonic::{Request, Response, Status};

use signal::{Signal, SignalSource};

use crate::errors::public_status;
use crate::helpers::{non_empty, response_to_string};
use crate::memory_proto::{
    memory_service_server::MemoryService, Fact, GetFactsRequest, GetFactsResponse, SearchRequest,
    SearchResponse, SignalEvent, SignalRequest as MemorySignalRequest, StoreRequest, StoreResponse,
};
use crate::state::MemoryServiceImpl;

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
            Err(e) => {
                tracing::error!(error = %e, "gRPC store_fact failed");
                Err(public_status(&e))
            }
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
                    tracing::error!(error = %e, "gRPC stream_signals processing failed");
                    let _ = tx.send(Err(public_status(&e))).await;
                }
            }
        });

        let stream: SignalEventStream = Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }
}
