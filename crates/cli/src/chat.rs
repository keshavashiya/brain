//! Chat commands — interactive and non-interactive conversation modes.
//!
//! Uses WebSocket for communication with the daemon, enabling lower latency
//! and a consistent protocol across all adapters.

use std::io::stdout;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::ExecutableCommand;
use futures_util::{SinkExt, StreamExt};
use rustyline::DefaultEditor;
use tokio_tungstenite::tungstenite::Message;

use crate::status::show_status;

type WsSink = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;
type WsStream = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// Send a chat message via a running daemon's WebSocket API.
///
/// Connects, authenticates, sends the signal, and returns the response text.
/// `session_id` is included so the daemon groups episodes into the same
/// conversation.
///
/// Returns `Ok(Some(response_text))` if the server responds,
/// or `Err` on failures.
async fn send_ws_message(
    ws_url: &str,
    api_key: &str,
    message: &str,
    session_id: &str,
) -> anyhow::Result<Option<String>> {
    let (ws, _) = tokio_tungstenite::connect_async(ws_url).await?;
    let (mut sink, mut stream) = ws.split();

    // Phase 1: Auth handshake
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

    // Phase 2: Send signal
    let signal = serde_json::json!({
        "content": message,
        "session_id": session_id,
    });
    sink.send(Message::Text(signal.to_string().into())).await?;

    // Phase 3: Read response — skip proactive notifications, which the server
    // may push unsolicited on the same stream before the real reply.
    loop {
        let frame = match stream.next().await {
            Some(Ok(frame)) => frame,
            Some(Err(e)) => return Err(anyhow::anyhow!("WebSocket error: {e}")),
            None => return Err(anyhow::anyhow!("No response from Brain")),
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

        let response_json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();

        if response_json.get("type").and_then(|v| v.as_str()) == Some("proactive") {
            continue; // ignore unsolicited pushes, keep waiting for the reply
        }

        let reply = response_json
            .get("response")
            .and_then(|r| r.get("value").or_else(|| r.get("Error")))
            .and_then(|v| v.as_str())
            .map(String::from);

        return Ok(reply.filter(|t| !t.is_empty()));
    }
}

/// Persistent WS connection for interactive mode.
struct WsSession {
    sink: WsSink,
    stream: WsStream,
}

impl WsSession {
    async fn send_message(
        &mut self,
        message: &str,
        session_id: &str,
    ) -> anyhow::Result<Option<String>> {
        let signal = serde_json::json!({
            "content": message,
            "session_id": session_id,
        });
        self.sink
            .send(Message::Text(signal.to_string().into()))
            .await?;

        loop {
            let frame = match self.stream.next().await {
                Some(Ok(frame)) => frame,
                Some(Err(e)) => return Err(anyhow::anyhow!("WebSocket error: {e}")),
                None => return Err(anyhow::anyhow!("Connection closed by server")),
            };

            match frame {
                Message::Text(t) => {
                    let response_json: serde_json::Value =
                        serde_json::from_str(&t).unwrap_or_default();

                    // Skip proactive notifications — they're pushed unsolicited
                    if response_json.get("type").and_then(|v| v.as_str()) == Some("proactive") {
                        // Print proactive notifications inline so the user sees them
                        if let Some(content) = response_json["content"].as_str() {
                            let mut out = stdout();
                            out.execute(SetForegroundColor(Color::Yellow))?;
                            out.execute(Print("[proactive] "))?;
                            out.execute(ResetColor)?;
                            println!("{content}");
                        }
                        continue; // keep waiting for the actual response
                    }

                    let text = response_json
                        .get("response")
                        .and_then(|r| r.get("value").or_else(|| r.get("Error")))
                        .and_then(|v| v.as_str())
                        .map(String::from);

                    return Ok(text.filter(|t| !t.is_empty()));
                }
                Message::Ping(data) => {
                    let _ = self.sink.send(Message::Pong(data)).await;
                    continue; // wait for next frame
                }
                Message::Close(_) => {
                    return Err(anyhow::anyhow!("Server closed the connection"));
                }
                _ => continue, // ignore Pong, Binary, etc.
            }
        }
    }
}

async fn connect_ws_session(ws_url: &str, api_key: &str) -> anyhow::Result<WsSession> {
    let (ws, _) = tokio_tungstenite::connect_async(ws_url).await?;
    let (mut sink, mut stream) = ws.split();

    // Auth handshake
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

    Ok(WsSession { sink, stream })
}

fn ws_url(config: &brain_core::BrainConfig) -> String {
    format!(
        "ws://{}:{}",
        config.adapters.http.host, config.adapters.ws.port
    )
}

fn first_api_key(config: &brain_core::BrainConfig) -> anyhow::Result<String> {
    config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .ok_or_else(|| anyhow::anyhow!("No API key configured. Run `brain init`."))
}

pub(crate) async fn chat_non_interactive(
    config: &brain_core::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    // Ensure the daemon is up — gives a clear error instead of a raw WS connect failure.
    let _ = crate::bootstrap::require_daemon(config).await?;
    let ws_url = ws_url(config);
    let api_key = first_api_key(config)?;
    let session_id = uuid::Uuid::new_v4().to_string();

    let response = send_ws_message(&ws_url, &api_key, message, &session_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Daemon returned empty response"))?;

    println!("{response}");
    Ok(())
}

pub(crate) async fn chat_interactive(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
    let _ = crate::bootstrap::require_daemon(config).await?;
    let ws_url = ws_url(config);
    let api_key = first_api_key(config)?;

    let mut session_id = uuid::Uuid::new_v4().to_string();

    let ver = env!("CARGO_PKG_VERSION");
    let title = format!("Brain v{ver}");
    let tagline = "Your AI's long-term memory";
    let width = 37;
    println!("╔═{}═╗", "═".repeat(width));
    println!("║ {:^w$} ║", title, w = width);
    println!("║ {:^w$} ║", tagline, w = width);
    println!("╚═{}═╝", "═".repeat(width));
    println!();
    println!("  Cortex:  {}", config.llm.model);
    println!("  Memory:  {}", config.data_dir().display());
    println!("  Synapse: connected to daemon (WebSocket)");
    println!();
    println!("Signals: /status  /clear  /quit");
    println!();

    // Establish persistent WS connection
    let mut ws = connect_ws_session(&ws_url, &api_key).await?;

    let mut rl = DefaultEditor::new()?;
    let history_path = config.data_dir().join("history.txt");
    let _ = rl.load_history(&history_path);

    loop {
        match rl.readline("You: ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(input);

                match input {
                    "/quit" | "/exit" | "/q" => {
                        println!("Going dormant...");
                        break;
                    }
                    "/status" => {
                        show_status(config).await?;
                        continue;
                    }
                    "/clear" => {
                        session_id = uuid::Uuid::new_v4().to_string();
                        println!("Session cleared — starting fresh conversation.");
                        continue;
                    }
                    s if s.starts_with('/') => {
                        println!("Unknown signal: {s}");
                        println!("Available: /status  /clear  /quit");
                        continue;
                    }
                    _ => {}
                }

                let response = ws.send_message(input, &session_id).await;

                match response {
                    Ok(Some(response)) => {
                        let mut out = stdout();
                        out.execute(SetForegroundColor(Color::Green))?;
                        out.execute(Print("Brain: "))?;
                        out.execute(ResetColor)?;
                        println!("{response}");
                    }
                    Ok(None) => {
                        eprintln!("Daemon returned empty response.");
                    }
                    Err(e) => {
                        eprintln!("{}", crate::errors::friendly_error(&e));
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                println!("Going dormant...");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(&history_path);
    Ok(())
}
