//! WebSocket transport for the one-shot chat paths: connect + authenticate,
//! send a signal, and drive inbound frames to completion. The split
//! sink/stream pair returned by [`connect_ws_session`] feeds the interactive
//! loop in [`super::reader`].

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use super::frames::{
    apply_frame, FrameOutcome, RenderStyle, ResponseAccumulator, ONE_SHOT_APPROVAL_HINT,
};
use super::render::{render_plain_direct, render_status_line};

pub(super) type WsSink = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;
pub(super) type WsStream = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// One-shot path: drive frames until the response settles or the connection
/// closes. Used by `chat_non_interactive` and during the brief auth+send
/// for the legacy single-message flow.
async fn drive_frames(
    sink: &mut WsSink,
    stream: &mut WsStream,
    style: RenderStyle,
) -> anyhow::Result<Option<String>> {
    let mut acc = ResponseAccumulator::with_style(style);
    // Live elapsed spinner: long turns (a slow LLM, an 80s task decompose)
    // otherwise show a single static `routing…` and feel hung. We tick a
    // dim status line with the current stage + elapsed seconds while waiting
    // on the next frame, so the session always looks alive. Suppressed for
    // Plain (deterministic-subcommand) output.
    let start = std::time::Instant::now();
    let mut ticker = tokio::time::interval(std::time::Duration::from_millis(250));
    ticker.tick().await; // discard the immediate first tick (no 0s flash)
    loop {
        let frame = tokio::select! {
            biased;
            frame = stream.next() => frame,
            _ = ticker.tick() => {
                if acc.style != RenderStyle::Plain {
                    let elapsed = start.elapsed().as_secs();
                    let stage = acc.status.as_deref().unwrap_or("working…");
                    let _ = render_status_line(&format!("{stage} ({elapsed}s)"));
                }
                continue;
            }
        };
        let frame = match frame {
            Some(Ok(frame)) => frame,
            Some(Err(e)) => return Err(anyhow::anyhow!("WebSocket error: {e}")),
            None => return Err(anyhow::anyhow!("Connection closed by server")),
        };

        let text = match frame {
            Message::Text(t) => t.to_string(),
            Message::Ping(data) => {
                let _ = sink.send(Message::Pong(data)).await;
                continue;
            }
            Message::Close(_) => {
                return Err(anyhow::anyhow!("Server closed the connection"));
            }
            _ => continue,
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();

        match apply_frame(&mut acc, &parsed) {
            FrameOutcome::Continue => continue,
            FrameOutcome::Complete => return Ok(acc.finalize()),
            FrameOutcome::Approval(body) => {
                // Render the gate prompt + actionable guidance and return
                // immediately. Closing the stream here (the caller drops the
                // socket) signals the daemon to withdraw the pending nonce
                // instead of leaving a ghost gate until it times out.
                ResponseAccumulator::render_approval_prompt(&body);
                render_plain_direct(ONE_SHOT_APPROVAL_HINT);
                return Ok(None);
            }
            FrameOutcome::Error(msg) => {
                ResponseAccumulator::render_error(&msg);
                return Err(anyhow::anyhow!(msg));
            }
        }
    }
}

/// One-shot: connect, auth, send, drain, return.
async fn send_ws_message(
    ws_url: &str,
    api_key: &str,
    message: &str,
    session_id: &str,
    style: RenderStyle,
) -> anyhow::Result<Option<String>> {
    let (ws, _) = tokio_tungstenite::connect_async(ws_url).await?;
    let (mut sink, mut stream) = ws.split();

    let auth_frame = serde_json::json!({ "api_key": api_key });
    sink.send(Message::Text(auth_frame.to_string().into()))
        .await?;

    let auth_response = match stream.next().await {
        Some(Ok(Message::Text(t))) => t.to_string(),
        Some(Ok(_)) => return Err(anyhow::anyhow!("Unexpected non-text auth response")),
        Some(Err(e)) => return Err(anyhow::anyhow!("WebSocket error during auth: {e}")),
        None => return Err(anyhow::anyhow!("No auth response from Brain")),
    };

    let auth_json: serde_json::Value = serde_json::from_str(&auth_response).unwrap_or_default();
    if auth_json.get("status").and_then(|s| s.as_str()) != Some("authenticated") {
        let error = auth_json
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Authentication failed");
        return Err(anyhow::anyhow!("Auth error: {error}"));
    }

    let signal = serde_json::json!({
        "content": message,
        "session_id": session_id,
        "stream": true,
    });
    sink.send(Message::Text(signal.to_string().into())).await?;

    drive_frames(&mut sink, &mut stream, style).await
}

/// Connect + authenticate, returning the split sink/stream pair the
/// interactive loop drives concurrently.
pub(super) async fn connect_ws_session(
    ws_url: &str,
    api_key: &str,
) -> anyhow::Result<(WsSink, WsStream)> {
    let (ws, _) = tokio_tungstenite::connect_async(ws_url).await?;
    let (mut sink, mut stream) = ws.split();

    let auth_frame = serde_json::json!({ "api_key": api_key });
    sink.send(Message::Text(auth_frame.to_string().into()))
        .await?;

    let auth_response = match stream.next().await {
        Some(Ok(Message::Text(t))) => t.to_string(),
        Some(Ok(_)) => return Err(anyhow::anyhow!("Unexpected non-text auth response")),
        Some(Err(e)) => return Err(anyhow::anyhow!("WebSocket error during auth: {e}")),
        None => return Err(anyhow::anyhow!("No auth response from Brain")),
    };

    let auth_json: serde_json::Value = serde_json::from_str(&auth_response).unwrap_or_default();
    if auth_json.get("status").and_then(|s| s.as_str()) != Some("authenticated") {
        let error = auth_json
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Authentication failed");
        return Err(anyhow::anyhow!("Auth error: {error}"));
    }

    Ok((sink, stream))
}

pub(super) fn ws_url(config: &brain::BrainConfig) -> String {
    format!(
        "ws://{}:{}",
        config.adapters.http.host, config.adapters.ws.port
    )
}

pub(super) fn first_api_key(config: &brain::BrainConfig) -> anyhow::Result<String> {
    config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .ok_or_else(|| anyhow::anyhow!("No API key configured. Run `brain init`."))
}

/// One-shot conversational turn (`brain chat "…"`): chat-styled output.
pub(crate) async fn chat_non_interactive(
    config: &brain::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    chat_non_interactive_styled(config, message, RenderStyle::Chat).await
}

/// One-shot output of a deterministic subcommand that rides the chat WS path
/// (e.g. `brain capabilities`): plain output, no `routing…` line or `Brain:`
/// label.
pub(crate) async fn command_over_chat(
    config: &brain::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    chat_non_interactive_styled(config, message, RenderStyle::Plain).await
}

async fn chat_non_interactive_styled(
    config: &brain::BrainConfig,
    message: &str,
    style: RenderStyle,
) -> anyhow::Result<()> {
    let _ = crate::bootstrap::require_daemon(config).await?;
    let ws_url = ws_url(config);
    let api_key = first_api_key(config)?;
    let session_id = uuid::Uuid::new_v4().to_string();

    let response = send_ws_message(&ws_url, &api_key, message, &session_id, style).await?;
    drop(response);
    Ok(())
}
