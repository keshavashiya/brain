use std::sync::Arc;

use futures_util::StreamExt;
use signal::{PipelineResult, Signal, SignalError, SignalResponse, SignalSource};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use super::ClientMessage;

/// Parse a text frame and run it through the signal pipeline.
///
/// Returns `Ok(None)` when the response is being streamed directly to the sink
/// (i.e. `client_msg.stream == Some(true)` and the pipeline returned `LlmReady`).
pub(crate) async fn process_text_frame(
    text: &str,
    conn_id: Uuid,
    processor: &signal::SignalProcessor,
    principal: Option<&identity::Principal>,
) -> Result<Option<SignalResponse>, SignalError> {
    let client_msg: ClientMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            let fake_id = Uuid::new_v4();
            return Ok(Some(SignalResponse::error(
                fake_id,
                format!("Invalid JSON: {e}"),
            )));
        }
    };

    let source = SignalSource::parse(client_msg.source.as_deref(), SignalSource::WebSocket);
    let signal = Signal::from_adapter_request(signal::AdapterRequest {
        source,
        content: client_msg.content,
        channel: Some(format!("ws:{conn_id}")),
        sender: client_msg.sender,
        metadata: client_msg.metadata,
        namespace: client_msg.namespace,
        agent: client_msg.agent,
        session_id: client_msg.session_id,
        default_channel: format!("ws:{conn_id}"),
        default_sender: "wsclient".to_string(),
    })
    .with_principal_opt(principal.cloned());

    let signal_id = signal.id;

    if client_msg.stream == Some(true) {
        return Ok(None);
    }

    match processor.process(signal).await {
        Ok(r) => Ok(Some(r)),
        Err(e) => {
            tracing::warn!(conn_id = %conn_id, "Signal processing error: {e}");
            Ok(Some(SignalResponse::error(
                signal_id,
                e.to_public().message.to_string(),
            )))
        }
    }
}

/// Push a JSON value as a text frame through the per-connection fan-in
/// mpsc. Returns Err if the writer task is gone (client disconnected) so
/// callers can short-circuit and stop generating more chunks.
async fn send_json_frame_to_sink(
    out_tx: &mpsc::Sender<Message>,
    value: &serde_json::Value,
    conn_id: Uuid,
) -> Result<(), ()> {
    match serde_json::to_string(value) {
        Ok(json) => {
            if out_tx.send(Message::Text(json.into())).await.is_err() {
                tracing::debug!(conn_id = %conn_id, "Failed to send WS frame (writer closed)");
                Err(())
            } else {
                Ok(())
            }
        }
        Err(e) => {
            tracing::error!(conn_id = %conn_id, "Failed to serialize frame: {e}");
            Err(())
        }
    }
}

/// RAII drop guard that removes the per-signal cancel registry entry when
/// the streaming handler returns — whether normally, via early-return, or
/// via panic. Mirrors `signal::pipeline::CancelGuard` for adapters that
/// bypass `SignalProcessor::process()` and drive the pipeline themselves.
struct WsCancelGuard {
    processor: Arc<signal::SignalProcessor>,
    signal_id: Uuid,
}

impl Drop for WsCancelGuard {
    fn drop(&mut self) {
        self.processor.unregister_cancel(self.signal_id);
    }
}

/// Guarantees that `finalize_streaming()` is called even if the client
/// disconnects mid-stream.
struct StreamFinalizer {
    processor: Arc<signal::SignalProcessor>,
    session_id: Option<String>,
    namespace: String,
    agent: Option<String>,
    acc: Arc<std::sync::Mutex<String>>,
    committed: bool,
}

impl StreamFinalizer {
    fn new(
        processor: Arc<signal::SignalProcessor>,
        session_id: Option<String>,
        namespace: String,
        agent: Option<String>,
        acc: Arc<std::sync::Mutex<String>>,
    ) -> Self {
        Self {
            processor,
            session_id,
            namespace,
            agent,
            acc,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for StreamFinalizer {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Drop runs on cancellation paths; a poisoned mutex must not panic
        // a second time. Recover the inner String either way.
        let acc = self
            .acc
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if acc.is_empty() {
            return;
        }
        let session_id = self.session_id.clone();
        let namespace = self.namespace.clone();
        let agent = self.agent.clone();
        let content = acc.clone();
        let processor = self.processor.clone();
        // `finalize_streaming` is async (attestation gate); Drop can't
        // await, so hand the persist off to the runtime. Drop runs on a
        // tokio worker on every cancellation path that reaches here.
        tokio::spawn(async move {
            if let Err(e) = processor
                .finalize_streaming(
                    session_id.as_deref().unwrap_or("unknown"),
                    &content,
                    &namespace,
                    agent.as_deref(),
                )
                .await
            {
                tracing::error!("finalize_streaming failed on cancellation: {e}");
            }
        });
    }
}

/// Handle a streaming LLM request: prepare → generate_stream → finalize.
///
/// Sends `chunk` frames for each token and a final `complete` frame.
pub(crate) async fn handle_streaming_request(
    out_tx: mpsc::Sender<Message>,
    conn_id: Uuid,
    processor: Arc<signal::SignalProcessor>,
    client_msg: ClientMessage,
    principal: Option<identity::Principal>,
) {
    let ws_tx = &out_tx;
    let source = SignalSource::parse(client_msg.source.as_deref(), SignalSource::WebSocket);
    let signal = Signal::from_adapter_request(signal::AdapterRequest {
        source,
        content: client_msg.content,
        channel: Some(format!("ws:{conn_id}")),
        sender: client_msg.sender,
        metadata: client_msg.metadata,
        namespace: client_msg.namespace.clone(),
        agent: client_msg.agent.clone(),
        session_id: client_msg.session_id.clone(),
        default_channel: format!("ws:{conn_id}"),
        default_sender: "wsclient".to_string(),
    })
    .with_principal_opt(principal);

    let signal_id = signal.id;

    // Mirror what `SignalProcessor::process` does at its entry so the
    // streaming path is observable too — without this, a `brain chat`
    // round-trip emits zero BrainEvents and `brain tail` looks broken.
    processor.publish_signal_received(&signal).await;

    // Register a cancellation notify so a concurrent `Intent::Cancel(Signal)`
    // for this id reaches the prepare/chunk loops below. The standard
    // pipeline installs this via `CancelGuard` inside `process()`, but
    // streaming bypasses `process()`, so we own the lifecycle here — the
    // `WsCancelGuard` mirrors that RAII removal on every return path.
    let cancel_notify = processor.register_cancel(signal_id).await;
    let _cancel_guard = WsCancelGuard {
        processor: processor.clone(),
        signal_id,
    };

    // Surface progress while the pipeline runs — otherwise the client just
    // sees nothing until the first LLM token. These frames are advisory.
    let _ = send_json_frame_to_sink(
        ws_tx,
        &serde_json::json!({"type": "status", "stage": "routing", "message": "routing…"}),
        conn_id,
    )
    .await;

    // Channel for pipeline → streaming handler progress updates ("searching…" etc).
    // The pipeline sends stage names; we forward them as status frames while prepare() runs.
    let (prog_tx, mut prog_rx) = tokio::sync::mpsc::channel::<&'static str>(8);
    let mut prepare_fut = Box::pin(processor.prepare(&signal, None, Some(prog_tx)));

    let prepared = loop {
        tokio::select! {
            biased;
            _ = cancel_notify.notified() => {
                let _ = send_json_frame_to_sink(
                    ws_tx,
                    &serde_json::json!({
                        "type": "error",
                        "code": "cancelled",
                        "message": format!("signal {signal_id} cancelled before LLM dispatch"),
                    }),
                    conn_id,
                )
                .await;
                return;
            }
            result = &mut prepare_fut => {
                match result {
                    Ok(p) => break p,
                    Err(e) => {
                        tracing::warn!(conn_id = %conn_id, "Signal prepare error: {e}");
                        let public = e.to_public();
                        let _ = send_json_frame_to_sink(
                            ws_tx,
                            &serde_json::json!({
                                "type": "error",
                                "code": public.code,
                                "message": public.message,
                            }),
                            conn_id,
                        )
                        .await;
                        return;
                    }
                }
            }
            Some(stage) = prog_rx.recv() => {
                let _ = send_json_frame_to_sink(
                    ws_tx,
                    &serde_json::json!({"type": "status", "stage": stage, "message": stage}),
                    conn_id,
                )
                .await;
            }
        }
    };

    match prepared {
        PipelineResult::Complete(resp) => {
            let frame = serde_json::json!({"type": "complete", "response": resp});
            let _ = send_json_frame_to_sink(ws_tx, &frame, conn_id).await;
        }
        PipelineResult::LlmReady {
            messages,
            memory_context,
            session_id,
            namespace,
            agent,
            ..
        } => {
            let status_msg = if memory_context.facts_used == 0 && memory_context.episodes_used == 0
            {
                "thinking…"
            } else {
                "recalling memories…"
            };
            let _ = send_json_frame_to_sink(
                ws_tx,
                &serde_json::json!({
                    "type": "status",
                    "stage": "thinking",
                    "message": status_msg,
                    "facts_used": memory_context.facts_used,
                    "episodes_used": memory_context.episodes_used,
                }),
                conn_id,
            )
            .await;

            let llm_stream = match processor.llm().generate_stream(&messages).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(conn_id = %conn_id, "LLM stream error: {e}");
                    let _ = send_json_frame_to_sink(
                        ws_tx,
                        &serde_json::json!({
                            "type": "error",
                            "message": e.to_string()
                        }),
                        conn_id,
                    )
                    .await;
                    return;
                }
            };

            let acc: Arc<std::sync::Mutex<String>> = Arc::new(std::sync::Mutex::new(String::new()));
            let finalizer = StreamFinalizer::new(
                processor.clone(),
                session_id.clone(),
                namespace.clone(),
                agent.clone(),
                Arc::clone(&acc),
            );

            let mut stream = llm_stream;
            let finalizer = finalizer;

            loop {
                let chunk_result = tokio::select! {
                    biased;
                    _ = cancel_notify.notified() => {
                        let _ = send_json_frame_to_sink(
                            ws_tx,
                            &serde_json::json!({
                                "type": "error",
                                "code": "cancelled",
                                "message": format!("signal {signal_id} cancelled mid-stream"),
                            }),
                            conn_id,
                        )
                        .await;
                        return;
                    }
                    next = stream.next() => match next {
                        Some(r) => r,
                        None => break,
                    },
                };
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!(conn_id = %conn_id, "Stream chunk error: {e}");
                        let _ = send_json_frame_to_sink(
                            ws_tx,
                            &serde_json::json!({
                                "type": "error",
                                "message": e.to_string()
                            }),
                            conn_id,
                        )
                        .await;
                        return;
                    }
                };

                acc.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push_str(&chunk.content);

                let chunk_frame = serde_json::json!({
                    "type": "chunk",
                    "content": chunk.content
                });
                if send_json_frame_to_sink(ws_tx, &chunk_frame, conn_id)
                    .await
                    .is_err()
                {
                    return;
                }

                if chunk.is_done {
                    break;
                }
            }

            {
                let acc_content = acc
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone();
                if let Err(e) = processor
                    .finalize_streaming(
                        session_id.as_deref().unwrap_or("unknown"),
                        &acc_content,
                        &namespace,
                        agent.as_deref(),
                    )
                    .await
                {
                    tracing::error!("finalize_streaming failed after successful stream: {e}");
                }
            }
            finalizer.commit();

            let resp = signal::SignalResponse {
                signal_id,
                status: signal::ResponseStatus::Ok,
                response: signal::ResponseContent::Text(
                    acc.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
                ),
                memory_context,
                session_id,
            };
            let complete_frame = serde_json::json!({"type": "complete", "response": resp});
            let _ = send_json_frame_to_sink(ws_tx, &complete_frame, conn_id).await;
        }
    }
}
