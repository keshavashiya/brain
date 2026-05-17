use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use signal::{PipelineResult, Signal, SignalError, SignalResponse, SignalSource};
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

/// Send a JSON value as a text frame. Returns Err if the send failed.
async fn send_json_frame_to_sink(
    ws_tx: &mut futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        Message,
    >,
    value: &serde_json::Value,
    conn_id: Uuid,
) -> Result<(), ()> {
    match serde_json::to_string(value) {
        Ok(json) => {
            if ws_tx.send(Message::Text(json.into())).await.is_err() {
                tracing::debug!(conn_id = %conn_id, "Failed to send WS frame");
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
        let acc = self.acc.lock().unwrap();
        if acc.is_empty() {
            return;
        }
        let session_id = self.session_id.clone();
        let namespace = self.namespace.clone();
        let agent = self.agent.clone();
        let content = acc.clone();
        if let Err(e) = self.processor.finalize_streaming(
            session_id.as_deref().unwrap_or("unknown"),
            &content,
            &namespace,
            agent.as_deref(),
        ) {
            tracing::error!("finalize_streaming failed on cancellation: {e}");
        }
    }
}

/// Handle a streaming LLM request: prepare → generate_stream → finalize.
///
/// Sends `chunk` frames for each token and a final `complete` frame.
pub(crate) async fn handle_streaming_request(
    ws_tx: &mut futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        Message,
    >,
    conn_id: Uuid,
    processor: Arc<signal::SignalProcessor>,
    client_msg: ClientMessage,
    principal: Option<identity::Principal>,
) {
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

            while let Some(chunk_result) = stream.next().await {
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

                acc.lock().unwrap().push_str(&chunk.content);

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
                let acc_content = acc.lock().unwrap().clone();
                if let Err(e) = processor.finalize_streaming(
                    session_id.as_deref().unwrap_or("unknown"),
                    &acc_content,
                    &namespace,
                    agent.as_deref(),
                ) {
                    tracing::error!("finalize_streaming failed after successful stream: {e}");
                }
            }
            finalizer.commit();

            let resp = signal::SignalResponse {
                signal_id,
                status: signal::ResponseStatus::Ok,
                response: signal::ResponseContent::Text(acc.lock().unwrap().clone()),
                memory_context,
                session_id,
            };
            let complete_frame = serde_json::json!({"type": "complete", "response": resp});
            let _ = send_json_frame_to_sink(ws_tx, &complete_frame, conn_id).await;
        }
    }
}
