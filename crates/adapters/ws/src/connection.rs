//! Per-connection driver: authenticates the handshake, wires the fan-in writer
//! task + chat-transport relay, then dispatches incoming frames concurrently
//! with brain-event and proactive push streams.

use std::sync::Arc;

use brain::ApiKeyConfig;
use channel::DeliveryIntent;
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use signal::SignalResponse;

use crate::auth::{resolve_principal, validate_key};
use crate::chat_transport::{ws_channel_id, WsChatTransport};
use crate::protocol::{AuthMessage, AuthResponse, ClientMessage};
use crate::streaming::{handle_streaming_request, process_text_frame};

/// Bounded fan-in capacity for the per-connection writer mpsc. Sized so the
/// writer task can absorb a burst of brain_event/proactive/approval frames
/// without forcing producers to wait, but small enough that a stalled
/// client's queue can't grow unbounded.
const OUTGOING_QUEUE_CAPACITY: usize = 64;
/// Bounded queue for `DeliveryIntent`s arriving on `WsChatTransport.send`.
/// The relay task drains this into the outgoing queue.
const INTENT_QUEUE_CAPACITY: usize = 16;

/// Drive a single WebSocket connection to completion.
///
/// Step 1: read the first text frame as an `AuthMessage` and validate it.
/// Step 2: process subsequent `ClientMessage` frames as signals.
pub(crate) async fn handle_connection(
    ws_stream: tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    conn_id: Uuid,
    processor: Arc<signal::SignalProcessor>,
    api_keys: &[ApiKeyConfig],
) {
    let (mut ws_tx, mut ws_rx) = ws_stream.split();

    // ── Step 1: authenticate ─────────────────────────────────────────────────
    // Auth runs before we spawn the writer task so failure paths can close
    // the socket directly without leaving a writer task draining a dead
    // channel. Once auth succeeds we hand `ws_tx` to the writer task and
    // every subsequent producer (frame handler, brain_event, approval
    // prompt) writes via the fan-in `out_tx` mpsc.
    let (authed, principal, client_key): (bool, Option<identity::Principal>, Option<String>) =
        match ws_rx.next().await {
            None => return,
            Some(Err(e)) => {
                tracing::debug!(conn_id = %conn_id, "WS recv error during auth: {e}");
                return;
            }
            Some(Ok(Message::Text(text))) => {
                match serde_json::from_str::<AuthMessage>(text.as_str()) {
                    Err(e) => {
                        let resp = AuthResponse {
                            status: "error",
                            conn_id: None,
                            message: Some(format!("Expected auth message: {e}")),
                        };
                        send_json_frame(&mut ws_tx, &resp, conn_id).await;
                        return;
                    }
                    Ok(auth) => {
                        if !validate_key(api_keys, &auth.api_key) {
                            let resp = AuthResponse {
                                status: "error",
                                conn_id: None,
                                message: Some("Invalid or missing API key".to_string()),
                            };
                            send_json_frame(&mut ws_tx, &resp, conn_id).await;
                            return;
                        }
                        let principal =
                            resolve_principal(api_keys, &auth.api_key, &processor).await;
                        let resp = AuthResponse {
                            status: "authenticated",
                            conn_id: Some(conn_id.to_string()),
                            message: None,
                        };
                        send_json_frame(&mut ws_tx, &resp, conn_id).await;
                        (true, principal, Some(auth.api_key))
                    }
                }
            }
            Some(Ok(Message::Close(_))) => return,
            Some(Ok(_)) => {
                let resp = AuthResponse {
                    status: "error",
                    conn_id: None,
                    message: Some("First frame must be a text auth message".to_string()),
                };
                send_json_frame(&mut ws_tx, &resp, conn_id).await;
                return;
            }
        };

    if !authed {
        return;
    }
    let rate_limits = processor.client_rate_limits().cloned();

    // ── Fan-in writer ───────────────────────────────────────────────────────
    // One task owns `ws_tx`; everyone else (frame handlers, push streams,
    // approval relay) sends `Message`s through `out_tx`. This is what makes
    // concurrent frame processing safe — multiple in-flight signal pipelines
    // can stream chunks back without contending for the sink.
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(OUTGOING_QUEUE_CAPACITY);
    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            if ws_tx.send(msg).await.is_err() {
                break;
            }
        }
        // Best-effort: politely close once producers are done.
        let _ = ws_tx.send(Message::Close(None)).await;
        let _ = ws_tx.close().await;
    });

    // ── Register this connection as a Channel transport ─────────────────────
    // So `ChannelDispatcher::dispatch(Confirm-category intent)` can reach
    // the active chat session. The transport pushes intents through an
    // mpsc which the relay task below converts into JSON frames.
    let dispatcher = processor.channel_dispatcher().cloned();
    let channel_id = ws_channel_id(conn_id);
    let (intent_tx, mut intent_rx) = mpsc::channel::<DeliveryIntent>(INTENT_QUEUE_CAPACITY);
    if let Some(d) = &dispatcher {
        let transport = Arc::new(WsChatTransport::new(conn_id, intent_tx.clone()));
        if let Err(e) = d.register_transport(transport).await {
            tracing::warn!(
                conn_id = %conn_id, error = %e,
                "WS chat transport register failed; approval prompts won't reach this session"
            );
        } else {
            tracing::debug!(conn_id = %conn_id, channel_id = %channel_id, "WS chat transport registered");
        }
    }
    // Nonces of approval prompts relayed to this connection. On disconnect we
    // withdraw any that are still pending so the parked pipeline returns at
    // once instead of holding a ghost gate to the tier timeout (W1).
    let pending_nonces: Arc<std::sync::Mutex<std::collections::HashSet<String>>> =
        Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
    let relay_out = out_tx.clone();
    let relay_conn = conn_id;
    let relay_nonces = pending_nonces.clone();
    let relay_handle = tokio::spawn(async move {
        while let Some(intent) = intent_rx.recv().await {
            if let Some(nonce) = &intent.nonce {
                if let Ok(mut set) = relay_nonces.lock() {
                    set.insert(nonce.clone());
                }
            }
            let frame = serde_json::json!({
                "type": "approval_request",
                "intent_id": intent.id,
                "category": format!("{:?}", intent.category).to_lowercase(),
                "urgency": format!("{:?}", intent.urgency).to_lowercase(),
                "nonce": intent.nonce,
                "content": intent.content,
            });
            let Ok(json) = serde_json::to_string(&frame) else {
                tracing::error!(conn_id = %relay_conn, "approval frame serialize failed");
                continue;
            };
            if relay_out.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    // ── Step 2: process signal frames + proactive push ──────────────────────
    let mut proactive_rx = processor.notification_router().map(|r| r.subscribe());
    let mut brain_rx = processor.subscribe_brain_events();

    loop {
        let proactive_fut = async {
            match proactive_rx.as_mut() {
                Some(rx) => rx.recv().await,
                None => std::future::pending().await,
            }
        };
        let brain_fut = async {
            match brain_rx.as_mut() {
                Some(rx) => rx.recv().await,
                None => std::future::pending().await,
            }
        };

        tokio::select! {
            // Incoming client frame — spawn the handler so the outer loop
            // keeps draining ws_rx (and approval pushes keep flowing) even
            // while a prior signal is awaiting confirmation.
            result = ws_rx.next() => {
                let Some(result) = result else { break };
                let msg = match result {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::debug!(conn_id = %conn_id, "WebSocket receive error: {e}");
                        break;
                    }
                };
                match msg {
                    Message::Text(text) => {
                        if let (Some(reg), Some(key)) = (rate_limits.as_ref(), client_key.as_ref()) {
                            let limiter = reg.get_or_create(key);
                            if !limiter.try_acquire() {
                                tracing::warn!(conn_id = %conn_id, "WS rate limit exceeded");
                                let frame = serde_json::json!({
                                    "type": "error",
                                    "code": "rate_limited",
                                    "message": "Too many requests",
                                });
                                if let Ok(json) = serde_json::to_string(&frame) {
                                    let _ = out_tx.send(Message::Text(json.into())).await;
                                }
                                continue;
                            }
                        }
                        let text_string = text.to_string();
                        let client_msg: Option<ClientMessage> =
                            serde_json::from_str(text_string.as_str()).ok();
                        let out_tx_clone = out_tx.clone();
                        let proc_clone = processor.clone();
                        let principal_clone = principal.clone();
                        tokio::spawn(async move {
                            if let Some(cm) = client_msg.as_ref() {
                                if cm.stream == Some(true) {
                                    handle_streaming_request(
                                        out_tx_clone,
                                        conn_id,
                                        proc_clone,
                                        cm.clone(),
                                        principal_clone,
                                    )
                                    .await;
                                    return;
                                }
                            }
                            let response = match process_text_frame(
                                text_string.as_str(),
                                conn_id,
                                &proc_clone,
                                principal_clone.as_ref(),
                            )
                            .await
                            {
                                Ok(Some(r)) => r,
                                Ok(None) => return,
                                Err(e) => {
                                    tracing::warn!(
                                        conn_id = %conn_id,
                                        "process_text_frame error: {e}"
                                    );
                                    SignalResponse::error(
                                        Uuid::new_v4(),
                                        e.to_public().message.to_string(),
                                    )
                                }
                            };
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = out_tx_clone.send(Message::Text(json.into())).await;
                            }
                        });
                    }
                    Message::Ping(data) => {
                        let _ = out_tx.send(Message::Pong(data)).await;
                    }
                    Message::Close(_) => {
                        tracing::debug!(conn_id = %conn_id, "Client sent Close frame");
                        break;
                    }
                    _ => {}
                }
            }
            result = brain_fut => {
                match result {
                    Ok(ev) => {
                        let frame = serde_json::json!({
                            "type": "brain_event",
                            "event": ev,
                        });
                        if let Ok(json) = serde_json::to_string(&frame) {
                            if out_tx.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(conn_id = %conn_id, skipped = n, "WS brain_event stream lagged");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        brain_rx = None;
                    }
                }
            }
            result = proactive_fut => {
                match result {
                    Ok(notification) => {
                        let mut frame = serde_json::json!({
                            "type": "proactive",
                            "content": notification.content,
                            "triggered_by": notification.triggered_by,
                            "priority": notification.priority,
                        });
                        if let Some(agent) = &notification.agent {
                            frame["agent"] = serde_json::json!(agent);
                        }
                        if let Ok(json) = serde_json::to_string(&frame) {
                            if out_tx.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(conn_id = %conn_id, skipped = n, "WS client lagged, dropped notifications");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        tracing::debug!(conn_id = %conn_id, "Notification channel closed");
                        proactive_rx = None;
                    }
                }
            }
        }
    }

    // ── Teardown ────────────────────────────────────────────────────────────
    // Withdraw any approval gates this connection raised but never answered —
    // the client is gone, so blocking the parked pipeline to the tier timeout
    // would only hold a ghost gate (W1). `withdraw` is idempotent and a no-op
    // on nonces the user already resolved.
    let to_withdraw: Vec<String> = pending_nonces
        .lock()
        .map(|set| set.iter().cloned().collect())
        .unwrap_or_default();
    if !to_withdraw.is_empty() {
        if let Some(engine) = processor.confirmation_engine() {
            for nonce in to_withdraw {
                if let Err(e) = engine
                    .withdraw(&nonce, "originating client disconnected")
                    .await
                {
                    tracing::debug!(conn_id = %conn_id, nonce = %nonce, error = %e, "approval withdraw failed");
                }
            }
        }
    }

    // Order matters: unregister first (so no new intents arrive), drop
    // `intent_tx` (so the relay task exits), then drop `out_tx` (so the
    // writer task exits). The join handles guarantee both background tasks
    // are fully torn down before this future returns.
    if let Some(d) = &dispatcher {
        if let Err(e) = d.unregister_transport(&channel_id).await {
            tracing::debug!(conn_id = %conn_id, error = %e, "WS transport unregister failed");
        }
    }
    drop(intent_tx);
    drop(out_tx);
    let _ = relay_handle.await;
    let _ = writer_handle.await;
}

/// Send a JSON-serialisable value as a text frame; log errors but don't panic.
async fn send_json_frame<S, T>(ws_tx: &mut S, value: &T, conn_id: Uuid)
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
    T: Serialize,
{
    match serde_json::to_string(value) {
        Ok(json) => {
            let _ = ws_tx.send(Message::Text(json.into())).await;
        }
        Err(e) => {
            tracing::error!(conn_id = %conn_id, "Failed to serialize frame: {e}");
        }
    }
}
