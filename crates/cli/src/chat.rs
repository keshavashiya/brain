//! Chat commands — interactive and non-interactive conversation modes.
//!
//! Uses WebSocket for communication with the daemon, enabling lower latency
//! and a consistent protocol across all adapters.
//!
//! Rendering is unified: every server frame (chunk, complete, proactive,
//! error) is buffered into an in-memory accumulator and rendered exactly
//! once via [`render_response`] when the response settles. This avoids
//! cursor-rewind hacks that broke whenever streamed content scrolled past
//! the terminal bottom.

use std::io::{stdout, Write};

use crossterm::cursor::MoveToColumn;
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};
use crossterm::ExecutableCommand;
use futures_util::{SinkExt, StreamExt};
use rustyline::DefaultEditor;
use tokio_tungstenite::tungstenite::Message;

use crate::encryption::resolve_llm_api_key;
use crate::status::show_status;

/// Render a transient progress line ("routing…", "thinking…") that will be
/// overwritten when the real response is rendered.
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
/// when stdout isn't a TTY (e.g. piped output) so wrapped lines stay sane.
fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(c, _)| c as usize)
        .unwrap_or(80)
        .clamp(40, 120)
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
    fn write_prefix(self) -> std::io::Result<()> {
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

/// The single rendering path for any response body. Prints the label header
/// on its own line, then the markdown-rendered body. Always called *after*
/// the transient status line has been cleared.
fn render_response(label: ResponseLabel, body: &str) {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return;
    }
    let _ = label.write_prefix();
    let processed = preprocess_markdown(trimmed);
    let skin = brain_skin();
    let formatted = skin.text(&processed, Some(terminal_width()));
    print!("{formatted}");
    println!();
    let _ = stdout().flush();
}

/// Aggregates incoming WS frames into a single buffered response, then
/// renders it exactly once. The same struct is used for both the one-shot
/// (`send_ws_message`) and persistent (`WsSession::send_message`) paths so
/// behaviour cannot drift between them.
struct ResponseAccumulator {
    body: String,
    /// `Some` once we know which label to render with. The first content-
    /// bearing frame fixes this; subsequent frames just append.
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
        render_response(ResponseLabel::Proactive, content);
    }

    fn render_error(message: &str) {
        let _ = clear_status_line();
        render_response(ResponseLabel::Error, message);
    }

    /// Render whatever was buffered. Caller is responsible for clearing the
    /// status line first if appropriate.
    fn finalize(self) -> Option<String> {
        if let Some(label) = self.label {
            let _ = clear_status_line();
            render_response(label, &self.body);
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

/// Result of routing a single inbound text frame into the accumulator.
enum FrameOutcome {
    /// Keep reading more frames.
    Continue,
    /// Response settled — render and return the body.
    Complete,
    /// Server reported an error.
    Error(String),
}

/// Apply one parsed text frame to the accumulator. Pure routing — no IO
/// beyond the transient status-line and inline proactive renders.
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

/// Drive the frame loop on `stream`, applying frames into `acc` until the
/// response settles or the connection ends. Returns the final rendered body
/// (if any) on success, or an error.
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

/// Send a chat message via a running daemon's WebSocket API. One-shot
/// connection: connects, authenticates, sends, drains frames, returns.
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

/// Persistent WS connection for interactive mode. Auth happens once at
/// connect time; each `send_message` reuses the same socket.
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
            "stream": true,
        });
        self.sink
            .send(Message::Text(signal.to_string().into()))
            .await?;

        drive_frames(&mut self.sink, &mut self.stream).await
    }
}

async fn connect_ws_session(ws_url: &str, api_key: &str) -> anyhow::Result<WsSession> {
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
    let _ = crate::bootstrap::require_daemon(config).await?;
    let ws_url = ws_url(config);
    let api_key = first_api_key(config)?;
    let session_id = uuid::Uuid::new_v4().to_string();

    let response = send_ws_message(&ws_url, &api_key, message, &session_id).await?;
    drop(response);
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
                    Ok(Some(_)) => {}
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
}
