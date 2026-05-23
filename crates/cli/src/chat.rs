//! Chat commands — interactive and non-interactive conversation modes.
//!
//! Uses WebSocket for communication with the daemon, enabling lower latency
//! and a consistent protocol across all adapters.
//!
//! Interactive mode runs two halves concurrently so the user can reply to
//! an approval prompt (`approve <nonce>` / `reject <nonce>`) while a prior
//! signal is still in-flight. Without this, a 60s confirmation window
//! always expires before rustyline gets the next line. The reader half
//! receives frames and emits rendered text via rustyline's external
//! printer (so prints land cleanly above the prompt); the sender half
//! reads lines and pushes them on the socket.

use std::io::{stdout, Write};
use std::sync::Arc;

use crossterm::cursor::MoveToColumn;
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};
use crossterm::ExecutableCommand;
use futures_util::{SinkExt, StreamExt};
use rustyline::{DefaultEditor, ExternalPrinter};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use crate::encryption::resolve_llm_api_key;
use crate::status::show_status;

/// Render a transient progress line ("routing…", "thinking…") that will be
/// overwritten when the real response is rendered. Only used in the
/// non-interactive (one-shot) path; the interactive loop drops status
/// frames because rustyline's external printer can't overwrite lines.
fn render_status_line(message: &str) -> std::io::Result<()> {
    let mut out = stdout();
    out.execute(MoveToColumn(0))?;
    out.execute(Clear(ClearType::CurrentLine))?;
    out.execute(SetAttribute(Attribute::Dim))?;
    out.execute(SetForegroundColor(Color::DarkGrey))?;
    out.execute(Print(format!("  {message}")))?;
    out.execute(ResetColor)?;
    out.execute(SetAttribute(Attribute::Reset))?;
    out.flush()?;
    Ok(())
}

/// Clear the transient status line before rendering real output.
fn clear_status_line() -> std::io::Result<()> {
    let mut out = stdout();
    out.execute(MoveToColumn(0))?;
    out.execute(Clear(ClearType::CurrentLine))?;
    out.flush()?;
    Ok(())
}

/// A reasonable terminal width for markdown rendering. Falls back to 80
/// when stdout isn't a TTY (e.g. piped output) so wrapped lines stay
/// sane. Capped at 100 because lines longer than that hurt readability
/// even on wide monitors — the eye loses track of the wrap target.
fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(c, _)| c as usize)
        .unwrap_or(80)
        .clamp(40, 100)
}

/// Markdown skin tuned for Brain's dark-terminal aesthetic.
fn brain_skin() -> termimad::MadSkin {
    let mut skin = termimad::MadSkin::default_dark();
    use termimad::crossterm::style::Color::*;
    skin.bold.set_fg(White);
    skin.italic.set_fg(AnsiValue(244));
    skin.inline_code.set_fg(Yellow);
    skin.code_block.set_fg(AnsiValue(252));
    skin.headers.iter_mut().for_each(|h| h.set_fg(Green));
    skin.bullet.set_fg(Green);
    skin.quote_mark.set_fg(AnsiValue(244));
    skin
}

/// Replace HTML break tags (commonly emitted inside GFM table cells) with
/// real newlines so termimad can wrap them properly. UTF-8 safe.
fn preprocess_markdown(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find('<') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if let Some(consumed) = match_br_tag(tail) {
            out.push('\n');
            rest = &tail[consumed..];
        } else {
            out.push('<');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// If `s` starts with a `<br>`, `<br/>`, or `<br />` tag (case-insensitive),
/// returns the byte length of the tag. Tries longest variants first so a
/// `<br />` input doesn't match the shorter `<br>` prefix.
fn match_br_tag(s: &str) -> Option<usize> {
    for variant in ["<br />", "<br/>", "<br>"] {
        let len = variant.len();
        if s.len() >= len && s.as_bytes()[..len].eq_ignore_ascii_case(variant.as_bytes()) {
            return Some(len);
        }
    }
    None
}

/// Label printed before a rendered response body.
#[derive(Clone, Copy)]
enum ResponseLabel {
    Brain,
    Proactive,
    Error,
}

impl ResponseLabel {
    /// Hand-rolled ANSI prefix so the rendered output can be assembled as
    /// a `String` and shipped through rustyline's external printer. We
    /// don't go via crossterm here because crossterm wants a writer.
    fn ansi_prefix(self) -> &'static str {
        match self {
            // bold + green / yellow / red, reset at end
            ResponseLabel::Brain => "\x1b[1;32mBrain:\x1b[0m\n",
            ResponseLabel::Proactive => "\x1b[1;33m[proactive]\x1b[0m\n",
            ResponseLabel::Error => "\x1b[1;31mError:\x1b[0m\n",
        }
    }

    fn write_prefix_direct(self) -> std::io::Result<()> {
        let mut out = stdout();
        match self {
            ResponseLabel::Brain => {
                out.execute(SetForegroundColor(Color::Green))?;
                out.execute(SetAttribute(Attribute::Bold))?;
                out.execute(Print("Brain:"))?;
            }
            ResponseLabel::Proactive => {
                out.execute(SetForegroundColor(Color::Yellow))?;
                out.execute(SetAttribute(Attribute::Bold))?;
                out.execute(Print("[proactive]"))?;
            }
            ResponseLabel::Error => {
                out.execute(SetForegroundColor(Color::Red))?;
                out.execute(SetAttribute(Attribute::Bold))?;
                out.execute(Print("Error:"))?;
            }
        }
        out.execute(SetAttribute(Attribute::Reset))?;
        out.execute(ResetColor)?;
        println!();
        out.flush()?;
        Ok(())
    }
}

/// Build the full rendered string (label + markdown body + trailing blank
/// line) for a response. Empty bodies render as empty so the caller can
/// skip them.
fn render_to_string(label: ResponseLabel, body: &str) -> String {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    let processed = preprocess_markdown(trimmed);
    let skin = brain_skin();
    let formatted = skin.text(&processed, Some(terminal_width()));
    let rendered = formatted.to_string();
    let rendered = rendered.trim_end_matches('\n');
    let mut out = String::with_capacity(rendered.len() + 32);
    out.push_str(label.ansi_prefix());
    out.push_str(rendered);
    out.push('\n');
    out
}

/// Direct-render path used by the one-shot (non-interactive) chat —
/// writes straight to stdout via crossterm.
fn render_response_direct(label: ResponseLabel, body: &str) {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return;
    }
    let _ = label.write_prefix_direct();
    let processed = preprocess_markdown(trimmed);
    let skin = brain_skin();
    let formatted = skin.text(&processed, Some(terminal_width()));
    let rendered = formatted.to_string();
    let rendered = rendered.trim_end_matches('\n');
    print!("{rendered}\n\n");
    let _ = stdout().flush();
}

/// Aggregates incoming WS frames into a single buffered response for the
/// one-shot path. The interactive path uses [`InteractiveAccumulator`]
/// which prints through an external printer instead.
struct ResponseAccumulator {
    body: String,
    label: Option<ResponseLabel>,
}

impl ResponseAccumulator {
    fn new() -> Self {
        Self {
            body: String::new(),
            label: None,
        }
    }

    fn push_chunk(&mut self, content: &str) {
        if content.is_empty() {
            return;
        }
        self.label.get_or_insert(ResponseLabel::Brain);
        self.body.push_str(content);
    }

    fn set_complete_body(&mut self, content: &str) {
        if self.body.is_empty() {
            self.body.push_str(content);
        }
        self.label.get_or_insert(ResponseLabel::Brain);
    }

    fn render_proactive(content: &str) {
        let _ = clear_status_line();
        render_response_direct(ResponseLabel::Proactive, content);
    }

    fn render_approval_prompt(content: &str) {
        let _ = clear_status_line();
        render_response_direct(ResponseLabel::Proactive, content);
    }

    fn render_error(message: &str) {
        let _ = clear_status_line();
        render_response_direct(ResponseLabel::Error, message);
    }

    fn finalize(self) -> Option<String> {
        if let Some(label) = self.label {
            let _ = clear_status_line();
            render_response_direct(label, &self.body);
        }
        Some(self.body).filter(|t| !t.is_empty())
    }
}

/// Extract the body from a `complete` frame. The daemon may shape this as
/// either `{response: {response: {value: "..."}}}` (SignalResponse with a
/// Text variant) or `{response: {value: "...", Error: "..."}}` (legacy).
/// We try both and fall back to empty string.
fn extract_complete_body(frame: &serde_json::Value) -> &str {
    let response = match frame.get("response") {
        Some(r) => r,
        None => return "",
    };
    if let Some(s) = response
        .get("response")
        .and_then(|c| c.get("value"))
        .and_then(|v| v.as_str())
    {
        return s;
    }
    if let Some(s) = response.get("value").and_then(|v| v.as_str()) {
        return s;
    }
    if let Some(s) = response.get("Error").and_then(|v| v.as_str()) {
        return s;
    }
    ""
}

/// Result of routing a single inbound text frame into the one-shot
/// accumulator. The interactive loop has its own routing.
enum FrameOutcome {
    Continue,
    Complete,
    Error(String),
}

fn apply_frame(acc: &mut ResponseAccumulator, frame: &serde_json::Value) -> FrameOutcome {
    match frame.get("type").and_then(|v| v.as_str()) {
        Some("status") => {
            if acc.body.is_empty() {
                if let Some(msg) = frame.get("message").and_then(|m| m.as_str()) {
                    let _ = render_status_line(msg);
                }
            }
            FrameOutcome::Continue
        }
        Some("proactive") => {
            if let Some(content) = frame.get("content").and_then(|c| c.as_str()) {
                ResponseAccumulator::render_proactive(content);
            }
            FrameOutcome::Continue
        }
        Some("approval_request") => {
            let body = frame
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("Approval required.");
            ResponseAccumulator::render_approval_prompt(body);
            FrameOutcome::Continue
        }
        Some("chunk") => {
            if let Some(content) = frame.get("content").and_then(|c| c.as_str()) {
                acc.push_chunk(content);
            }
            FrameOutcome::Continue
        }
        Some("complete") => {
            let body = extract_complete_body(frame);
            acc.set_complete_body(body);
            FrameOutcome::Complete
        }
        Some("error") => {
            let msg = frame
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error")
                .to_string();
            FrameOutcome::Error(msg)
        }
        _ => FrameOutcome::Continue,
    }
}

type WsSink = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;
type WsStream = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

/// One-shot path: drive frames until the response settles or the connection
/// closes. Used by `chat_non_interactive` and during the brief auth+send
/// for the legacy single-message flow.
async fn drive_frames(sink: &mut WsSink, stream: &mut WsStream) -> anyhow::Result<Option<String>> {
    let mut acc = ResponseAccumulator::new();
    loop {
        let frame = match stream.next().await {
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

    drive_frames(&mut sink, &mut stream).await
}

/// Connect + authenticate, returning the split sink/stream pair the
/// interactive loop drives concurrently.
async fn connect_ws_session(ws_url: &str, api_key: &str) -> anyhow::Result<(WsSink, WsStream)> {
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

fn ws_url(config: &brain::BrainConfig) -> String {
    format!(
        "ws://{}:{}",
        config.adapters.http.host, config.adapters.ws.port
    )
}

fn first_api_key(config: &brain::BrainConfig) -> anyhow::Result<String> {
    config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .ok_or_else(|| anyhow::anyhow!("No API key configured. Run `brain init`."))
}

pub(crate) async fn chat_non_interactive(
    config: &brain::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    let _ = crate::bootstrap::require_daemon(config).await?;
    let ws_url = ws_url(config);
    let api_key = first_api_key(config)?;
    let session_id = uuid::Uuid::new_v4().to_string();

    let response = send_ws_message(&ws_url, &api_key, message, &session_id).await?;
    drop(response);
    Ok(())
}

/// Inbound frames from the reader task to the input-loop driver, used so
/// the input loop can run `/status`-style work after the reader briefly
/// pauses and so it knows when the socket has gone away.
enum ReaderEvent {
    /// The reader observed a clean or error close. The input loop should
    /// stop after the user's next keystroke.
    Closed(Option<String>),
}

/// Format a status message as a dim grey line for the external printer.
/// If `overwrite` is true, prepends an ANSI sequence that moves the
/// cursor up one line and clears it, so the new line overdraws the
/// previously printed status (or response label) instead of stacking.
/// Rustyline redraws its prompt on the same line afterward.
fn render_status_for_printer(stage: &str, message: &str, overwrite: bool) -> String {
    let body = if message.trim().is_empty() {
        stage
    } else {
        message
    };
    let prefix = if overwrite { "\x1b[1A\x1b[2K" } else { "" };
    format!("{prefix}\x1b[2;90m  \u{2026} {body}\x1b[0m\n")
}

/// Wrap a rendered response (label + body) with a leading "clear previous
/// line" escape when there's a transient status line above it. This lets
/// the final response replace the last `… thinking…` line in-place.
fn with_overwrite_prefix(rendered: String, overwrite: bool) -> String {
    if !overwrite || rendered.is_empty() {
        return rendered;
    }
    format!("\x1b[1A\x1b[2K{rendered}")
}

/// Run the WS reader task: pull frames, accumulate chunks, render
/// completed/proactive/approval/error bodies via the external printer.
/// Status frames render as a single dim line per stage transition so the
/// user has visible feedback while the pipeline runs; duplicates of the
/// same stage are suppressed to avoid stacking.
async fn run_reader<P: ExternalPrinter + Send + 'static>(
    mut stream: WsStream,
    mut printer: P,
    notify: mpsc::Sender<ReaderEvent>,
) {
    let mut acc = ResponseAccumulator::new();
    let mut last_status_stage: Option<String> = None;
    // Tracks whether the most recent printed line was a transient status.
    // When true, the next print (status or response) prepends an ANSI
    // sequence that overdraws that line in place.
    let mut status_line_pending = false;
    loop {
        let frame = match stream.next().await {
            Some(Ok(frame)) => frame,
            Some(Err(e)) => {
                let _ = notify
                    .send(ReaderEvent::Closed(Some(format!("WebSocket error: {e}"))))
                    .await;
                return;
            }
            None => {
                let _ = notify.send(ReaderEvent::Closed(None)).await;
                return;
            }
        };

        let text = match frame {
            Message::Text(t) => t.to_string(),
            Message::Ping(_) => continue,
            Message::Close(_) => {
                let _ = notify.send(ReaderEvent::Closed(None)).await;
                return;
            }
            _ => continue,
        };

        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();

        match parsed.get("type").and_then(|v| v.as_str()) {
            Some("status") => {
                let stage = parsed
                    .get("stage")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let message = parsed
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if last_status_stage.as_deref() != Some(stage.as_str()) {
                    let _ = printer.print(render_status_for_printer(
                        &stage,
                        message,
                        status_line_pending,
                    ));
                    last_status_stage = Some(stage);
                    status_line_pending = true;
                }
            }
            Some("proactive") => {
                if let Some(content) = parsed.get("content").and_then(|c| c.as_str()) {
                    let rendered = render_to_string(ResponseLabel::Proactive, content);
                    let rendered = with_overwrite_prefix(rendered, status_line_pending);
                    if !rendered.is_empty() {
                        let _ = printer.print(rendered);
                        status_line_pending = false;
                    }
                }
            }
            Some("approval_request") => {
                let body = parsed
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("Approval required.");
                let rendered = render_to_string(ResponseLabel::Proactive, body);
                let rendered = with_overwrite_prefix(rendered, status_line_pending);
                if !rendered.is_empty() {
                    let _ = printer.print(rendered);
                    status_line_pending = false;
                }
            }
            Some("chunk") => {
                if let Some(content) = parsed.get("content").and_then(|c| c.as_str()) {
                    acc.push_chunk(content);
                }
            }
            Some("complete") => {
                let body = extract_complete_body(&parsed);
                acc.set_complete_body(body);
                let finished = std::mem::replace(&mut acc, ResponseAccumulator::new());
                if let Some(label) = finished.label {
                    let rendered = render_to_string(label, &finished.body);
                    let rendered = with_overwrite_prefix(rendered, status_line_pending);
                    if !rendered.is_empty() {
                        let _ = printer.print(rendered);
                    }
                }
                last_status_stage = None;
                status_line_pending = false;
            }
            Some("error") => {
                let msg = parsed
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error");
                let rendered = render_to_string(ResponseLabel::Error, msg);
                let rendered = with_overwrite_prefix(rendered, status_line_pending);
                if !rendered.is_empty() {
                    let _ = printer.print(rendered);
                }
                last_status_stage = None;
                status_line_pending = false;
            }
            _ => {}
        }
    }
}

pub(crate) async fn chat_interactive(config: &brain::BrainConfig) -> anyhow::Result<()> {
    let _ = crate::bootstrap::require_daemon(config).await?;
    let ws_url = ws_url(config);
    let api_key = first_api_key(config)?;

    let session_id = Arc::new(std::sync::Mutex::new(uuid::Uuid::new_v4().to_string()));

    let ver = env!("CARGO_PKG_VERSION");
    let title = format!("Brain v{ver}");
    let tagline = "Your AI's long-term memory";
    let width = 37;
    println!("╔═{}═╗", "═".repeat(width));
    println!("║ {:^w$} ║", title, w = width);
    println!("║ {:^w$} ║", tagline, w = width);
    println!("╚═{}═╝", "═".repeat(width));
    println!();
    let cortex_model = {
        let llm_key = resolve_llm_api_key(config);
        let mut llm_cfg = config.llm.clone();
        if llm_cfg.providers.is_empty() {
            llm_cfg.api_key = llm_key;
        }
        cortex::llm::select_provider(&llm_cfg)
            .await
            .map(|p| format!("{} ({})", p.model(), p.name()))
            .unwrap_or_else(|_| config.llm.model.clone())
    };
    println!("  Cortex:  {}", cortex_model);
    println!("  Memory:  {}", config.data_dir().display());
    println!("  Synapse: connected to daemon (WebSocket)");
    println!();
    println!("Signals: /status  /clear  /quit");
    println!();

    let (sink, stream) = connect_ws_session(&ws_url, &api_key).await?;

    let mut rl = DefaultEditor::new()?;
    let history_path = config.data_dir().join("history.txt");
    let _ = rl.load_history(&history_path);
    let printer = rl
        .create_external_printer()
        .map_err(|e| anyhow::anyhow!("Failed to attach external printer: {e}"))?;

    let (reader_tx, mut reader_rx) = mpsc::channel::<ReaderEvent>(4);
    let reader_handle = tokio::spawn(run_reader(stream, printer, reader_tx));

    // Bridge blocking rustyline into async land. The blocking thread owns
    // the editor; it sends each non-empty line over `line_tx`. Closing
    // `line_tx` (by dropping the thread) tells the async loop to wind
    // down.
    let (line_tx, mut line_rx) = mpsc::channel::<LineEvent>(1);
    let history_path_for_thread = history_path.clone();
    let input_thread = std::thread::spawn(move || {
        let mut rl = rl;
        loop {
            match rl.readline("You: ") {
                Ok(line) => {
                    let trimmed = line.trim().to_string();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let _ = rl.add_history_entry(&trimmed);
                    if line_tx.blocking_send(LineEvent::Line(trimmed)).is_err() {
                        break;
                    }
                }
                Err(rustyline::error::ReadlineError::Interrupted)
                | Err(rustyline::error::ReadlineError::Eof) => {
                    let _ = line_tx.blocking_send(LineEvent::Quit);
                    break;
                }
                Err(e) => {
                    let _ = line_tx.blocking_send(LineEvent::Error(e.to_string()));
                    break;
                }
            }
        }
        let _ = rl.save_history(&history_path_for_thread);
    });

    let mut sink = sink;
    let result: anyhow::Result<()> = loop {
        tokio::select! {
            line = line_rx.recv() => {
                match line {
                    Some(LineEvent::Line(line)) => {
                        match line.as_str() {
                            "/quit" | "/exit" | "/q" => {
                                println!("Going dormant...");
                                break Ok(());
                            }
                            "/status" => {
                                if let Err(e) = show_status(config).await {
                                    eprintln!("status: {e}");
                                }
                                continue;
                            }
                            "/clear" => {
                                *session_id.lock().unwrap() = uuid::Uuid::new_v4().to_string();
                                println!("Session cleared — starting fresh conversation.");
                                continue;
                            }
                            s if looks_like_slash_command(s) => {
                                println!("Unknown signal: {s}");
                                println!("Available: /status  /clear  /quit");
                                continue;
                            }
                            _ => {}
                        }

                        let sid = session_id.lock().unwrap().clone();
                        let signal = serde_json::json!({
                            "content": line,
                            "session_id": sid,
                            "stream": true,
                        });
                        if let Err(e) = sink.send(Message::Text(signal.to_string().into())).await {
                            eprintln!("send failed: {e}");
                            break Err(anyhow::anyhow!(e));
                        }
                    }
                    Some(LineEvent::Quit) => {
                        println!("Going dormant...");
                        break Ok(());
                    }
                    Some(LineEvent::Error(msg)) => {
                        eprintln!("Error: {msg}");
                        break Err(anyhow::anyhow!(msg));
                    }
                    None => {
                        // Input thread exited unexpectedly.
                        break Ok(());
                    }
                }
            }
            event = reader_rx.recv() => {
                match event {
                    Some(ReaderEvent::Closed(reason)) => {
                        if let Some(reason) = reason {
                            eprintln!("Connection lost: {reason}");
                        } else {
                            eprintln!("Connection closed by server.");
                        }
                        break Ok(());
                    }
                    None => break Ok(()),
                }
            }
        }
    };

    // Close the WS politely, drop the input thread (rustyline still
    // blocked on readline will be killed when the process exits or the
    // join completes after the user hits Enter).
    let _ = sink.close().await;
    drop(line_rx);
    let _ = reader_handle.await;
    // We deliberately do not join `input_thread`: rustyline is still
    // blocked on a read, and there's no clean cross-platform way to
    // wake it. Returning here lets the runtime tear down on its own.
    drop(input_thread);

    result
}

/// True when the input should be treated as a slash command rather than a
/// chat message. An absolute file path like `/Users/foo/bar.docx ...` is
/// not a slash command — the first token contains inner `/` separators.
/// Short tokens like `/staus` are still flagged so typos surface instead
/// of getting silently sent as messages.
fn looks_like_slash_command(s: &str) -> bool {
    let Some(rest) = s.strip_prefix('/') else {
        return false;
    };
    let first_token = rest.split_whitespace().next().unwrap_or(rest);
    !first_token.contains('/')
}

/// Events from the blocking rustyline thread to the async loop.
enum LineEvent {
    Line(String),
    Quit,
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_replaces_br_variants() {
        let input = "a<br>b<br/>c<br />d<BR>e";
        assert_eq!(preprocess_markdown(input), "a\nb\nc\nd\ne");
    }

    #[test]
    fn preprocess_leaves_other_html_alone() {
        let input = "see <code>foo</code> and <em>x</em>";
        assert_eq!(preprocess_markdown(input), input);
    }

    #[test]
    fn preprocess_passes_plain_text_through() {
        let input = "Hello, world!\n\nNo HTML here.";
        assert_eq!(preprocess_markdown(input), input);
    }

    #[test]
    fn extract_complete_body_signal_response_shape() {
        let frame = serde_json::json!({
            "type": "complete",
            "response": {
                "response": {"type": "Text", "value": "hello"}
            }
        });
        assert_eq!(extract_complete_body(&frame), "hello");
    }

    #[test]
    fn extract_complete_body_legacy_shape() {
        let frame = serde_json::json!({
            "type": "complete",
            "response": {"value": "legacy body"}
        });
        assert_eq!(extract_complete_body(&frame), "legacy body");
    }

    #[test]
    fn extract_complete_body_missing_returns_empty() {
        let frame = serde_json::json!({"type": "complete"});
        assert_eq!(extract_complete_body(&frame), "");
    }

    #[test]
    fn accumulator_collects_chunks() {
        let mut acc = ResponseAccumulator::new();
        acc.push_chunk("hello ");
        acc.push_chunk("world");
        assert_eq!(acc.body, "hello world");
    }

    #[test]
    fn accumulator_complete_does_not_overwrite_streamed_chunks() {
        let mut acc = ResponseAccumulator::new();
        acc.push_chunk("streamed");
        acc.set_complete_body("ignored");
        assert_eq!(acc.body, "streamed");
    }

    #[test]
    fn accumulator_complete_fills_when_empty() {
        let mut acc = ResponseAccumulator::new();
        acc.set_complete_body("batch body");
        assert_eq!(acc.body, "batch body");
    }

    #[test]
    fn apply_frame_routes_chunk_into_accumulator() {
        let mut acc = ResponseAccumulator::new();
        let frame = serde_json::json!({"type": "chunk", "content": "tok"});
        assert!(matches!(
            apply_frame(&mut acc, &frame),
            FrameOutcome::Continue
        ));
        assert_eq!(acc.body, "tok");
    }

    #[test]
    fn apply_frame_returns_complete_on_terminal_frame() {
        let mut acc = ResponseAccumulator::new();
        let frame = serde_json::json!({
            "type": "complete",
            "response": {"response": {"value": "done"}}
        });
        let outcome = apply_frame(&mut acc, &frame);
        assert!(matches!(outcome, FrameOutcome::Complete));
    }

    #[test]
    fn apply_frame_returns_error_with_message() {
        let mut acc = ResponseAccumulator::new();
        let frame = serde_json::json!({"type": "error", "message": "boom"});
        match apply_frame(&mut acc, &frame) {
            FrameOutcome::Error(msg) => assert_eq!(msg, "boom"),
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn render_to_string_includes_label_prefix() {
        let s = render_to_string(ResponseLabel::Brain, "hello");
        assert!(s.contains("Brain:"));
        assert!(s.contains("hello"));
    }

    #[test]
    fn status_for_printer_skips_overwrite_on_first_print() {
        let s = render_status_for_printer("routing", "routing…", false);
        assert!(!s.starts_with("\x1b[1A"));
        assert!(s.contains("routing…"));
    }

    #[test]
    fn status_for_printer_overwrites_subsequent_lines() {
        let s = render_status_for_printer("thinking", "thinking…", true);
        assert!(s.starts_with("\x1b[1A\x1b[2K"));
        assert!(s.contains("thinking…"));
    }

    #[test]
    fn overwrite_prefix_noop_on_empty_or_disabled() {
        assert_eq!(with_overwrite_prefix(String::new(), true), "");
        assert_eq!(
            with_overwrite_prefix("Brain:\nhi\n".to_string(), false),
            "Brain:\nhi\n"
        );
    }

    #[test]
    fn overwrite_prefix_prepends_when_enabled() {
        let out = with_overwrite_prefix("Brain:\nhi\n".to_string(), true);
        assert!(out.starts_with("\x1b[1A\x1b[2K"));
        assert!(out.contains("Brain:"));
    }

    #[test]
    fn slash_command_detected_for_known_form() {
        assert!(looks_like_slash_command("/quit"));
        assert!(looks_like_slash_command("/status"));
        assert!(looks_like_slash_command("/staus")); // typo still flagged
        assert!(looks_like_slash_command("/foo bar baz"));
    }

    #[test]
    fn slash_command_rejected_for_paths_and_messages() {
        assert!(!looks_like_slash_command("/Users/me/file.docx"));
        assert!(!looks_like_slash_command(
            "/Users/me/file.docx what is this?"
        ));
        assert!(!looks_like_slash_command("hello"));
        assert!(!looks_like_slash_command(""));
    }

    #[test]
    fn render_to_string_empty_body_yields_empty() {
        assert!(render_to_string(ResponseLabel::Brain, "").is_empty());
        assert!(render_to_string(ResponseLabel::Brain, "   \n  ").is_empty());
    }
}
