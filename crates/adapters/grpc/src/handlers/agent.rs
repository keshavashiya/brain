//! `AgentService` RPC handlers — connect, send_signal, receive_signals,
//! brain_events.

use std::pin::Pin;

use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use signal::{Signal, SignalSource};

use crate::agent_proto::{
    agent_service_server::AgentService, BrainEventMessage, BrainEventsRequest, ConnectRequest,
    ConnectResponse, ReceiveRequest, SignalRequest as AgentSignalRequest,
    SignalResponse as AgentSignalResponse, SignalUpdate,
};
use crate::errors::public_status;
use crate::events::brain_event_matches;
use crate::helpers::{non_empty, response_to_string};
use crate::state::AgentServiceImpl;

/// Stream type alias for the server-streaming `ReceiveSignals` RPC.
type SignalUpdateStream =
    Pin<Box<dyn Stream<Item = Result<SignalUpdate, Status>> + Send + 'static>>;

/// Stream type alias for the server-streaming `BrainEvents` RPC.
type BrainEventStream =
    Pin<Box<dyn Stream<Item = Result<BrainEventMessage, Status>> + Send + 'static>>;

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
            Err(e) => {
                tracing::error!(error = %e, "gRPC send_signal processing failed");
                Err(public_status(&e))
            }
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
