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

/// A REPL signal (slash-command) recognized inside the interactive chat.
///
/// This is the single source of truth: the banner, the `/help` listing, and
/// the unknown-signal hint are all rendered from [`SIGNALS`], so adding a
/// signal here surfaces it everywhere automatically.
struct Signal {
    /// Canonical name, e.g. `/status`.
    name: &'static str,
    /// Alternate spellings that invoke the same action.
    aliases: &'static [&'static str],
    /// One-line description shown by `/help`.
    summary: &'static str,
}

const SIGNALS: &[Signal] = &[
    Signal {
        name: "/help",
        aliases: &["/?"],
        summary: "list available signals",
    },
    Signal {
        name: "/status",
        aliases: &[],
        summary: "show cortex, memory, and synapse status",
    },
    Signal {
        name: "/clear",
        aliases: &[],
        summary: "start a fresh conversation",
    },
    Signal {
        name: "/quit",
        aliases: &["/exit", "/q"],
        summary: "go dormant and exit chat",
    },
];

/// The REPL signals rendered as product-self-model docs, so the SOUL's
/// grounding lists the real in-chat commands (`/help`, `/status`, …) instead of
/// inventing plausible ones like `/msg`. [`SIGNALS`] stays the single source of
/// truth — the clap-walked [`crate::command_catalog::build`] is its CLI
/// counterpart.
pub(crate) fn signal_catalog() -> Vec<selfmodel::SignalDoc> {
    SIGNALS
        .iter()
        .map(|s| selfmodel::SignalDoc {
            name: s.name.to_string(),
            summary: s.summary.to_string(),
        })
        .collect()
}

/// Space-separated list of canonical signal names, for the banner and the
/// unknown-signal hint.
fn signals_line() -> String {
    SIGNALS
        .iter()
        .map(|s| s.name)
        .collect::<Vec<_>>()
        .join("  ")
}

/// Multi-line `/help` body: each signal with its aliases and summary.
fn signals_help() -> String {
    let width = SIGNALS.iter().map(|s| s.name.len()).max().unwrap_or(0);
    SIGNALS
        .iter()
        .map(|s| {
            let aliases = if s.aliases.is_empty() {
                String::new()
            } else {
                format!(" ({})", s.aliases.join(", "))
            };
            format!(
                "  {:<width$}  {}{}",
                s.name,
                s.summary,
                aliases,
                width = width
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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

/// Strip the language info-string from opening code fences
/// (```` ```bash ```` → ```` ``` ````). termimad renders the info-string as a
/// literal first line *inside* the code block (so `bash` shows on its own line
/// above the command), which reads as a rendering bug. Removing the info-string
/// at the source leaves a clean fenced block.
///
/// Only the fence line itself is rewritten: a bare fence (```` ``` ````, used
/// for both closing fences and language-less openings) and longer fences
/// (```` ```` ````, which carry no info-string) are left untouched, as is all
/// fenced content.
fn strip_code_fence_langs(input: &str) -> String {
    input
        .split('\n')
        .map(|line| {
            let trimmed = line.trim_start();
            let Some(info) = trimmed.strip_prefix("```") else {
                return line.to_string();
            };
            // A bare fence has nothing after the backticks; a longer fence
            // starts with another backtick. Neither carries a language token.
            if info.trim().is_empty() || info.starts_with('`') {
                return line.to_string();
            }
            // Opening fence with a language/info token → keep just the fence,
            // preserving the (≤3 space) indentation CommonMark allows.
            let indent = &line[..line.len() - trimmed.len()];
            format!("{indent}```")
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Reflow GFM tables that are too wide for the terminal into definition-style
/// bullet lists, one bullet per cell (`- **<header>:** <cell>`), with a blank
/// line between rows. termimad otherwise char-wraps each cell into a vertical
/// stack of single characters once the table's natural width exceeds the
/// terminal, which is unreadable. Tables that fit within `width` are left
/// untouched so termimad renders them as real tables.
fn reflow_wide_tables(input: &str, width: usize) -> String {
    let lines: Vec<&str> = input.split('\n').collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        // A table block is a header row immediately followed by a delimiter row.
        if i + 1 < lines.len() && is_table_row(lines[i]) && is_table_delimiter(lines[i + 1]) {
            let header = split_table_row(lines[i]);
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut j = i + 2;
            while j < lines.len() && is_table_row(lines[j]) {
                rows.push(split_table_row(lines[j]));
                j += 1;
            }
            if table_natural_width(&header, &rows) > width {
                out.push(reflow_table(&header, &rows));
            } else {
                // Fits — keep the original lines verbatim for termimad.
                out.extend(lines[i..j].iter().map(|s| s.to_string()));
            }
            i = j;
        } else {
            out.push(lines[i].to_string());
            i += 1;
        }
    }
    out.join("\n")
}

/// A line that could be a table row: contains a `|` and at least one
/// non-pipe character.
fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.contains('|') && t.chars().any(|c| c != '|' && !c.is_whitespace())
}

/// The GFM delimiter row: every cell is dashes with optional leading/trailing
/// colons (alignment markers), and there is at least one dash overall.
fn is_table_delimiter(line: &str) -> bool {
    let cells = split_table_row(line);
    if cells.is_empty() {
        return false;
    }
    let mut saw_dash = false;
    for cell in &cells {
        let bytes = cell.trim().trim_start_matches(':').trim_end_matches(':');
        if bytes.is_empty() || !bytes.chars().all(|c| c == '-') {
            return false;
        }
        saw_dash = true;
    }
    saw_dash
}

/// Split a table row into trimmed cells, dropping the empty cells produced by
/// the conventional leading/trailing `|`.
fn split_table_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| c.trim().to_string()).collect()
}

/// Estimate the rendered width of a table: summed max column widths plus the
/// `| ` / ` | ` / ` |` border padding termimad draws.
fn table_natural_width(header: &[String], rows: &[Vec<String>]) -> usize {
    let cols = header
        .len()
        .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if cols == 0 {
        return 0;
    }
    let mut widths = vec![0usize; cols];
    for (idx, cell) in header.iter().enumerate() {
        widths[idx] = widths[idx].max(cell.chars().count());
    }
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            if idx < cols {
                widths[idx] = widths[idx].max(cell.chars().count());
            }
        }
    }
    // Each column contributes its content plus `| ` + trailing space, and the
    // table closes with a final `|`: 3 chars of border per column + 1.
    widths.iter().sum::<usize>() + cols * 3 + 1
}

/// Render a table as a bullet list: each row becomes a group of
/// `- **<header>:** <cell>` bullets, groups separated by a blank line.
fn reflow_table(header: &[String], rows: &[Vec<String>]) -> String {
    let mut blocks: Vec<String> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut lines: Vec<String> = Vec::with_capacity(row.len());
        for (idx, cell) in row.iter().enumerate() {
            if cell.is_empty() {
                continue;
            }
            match header.get(idx).filter(|h| !h.is_empty()) {
                Some(h) => lines.push(format!("- **{h}:** {cell}")),
                None => lines.push(format!("- {cell}")),
            }
        }
        if !lines.is_empty() {
            blocks.push(lines.join("\n"));
        }
    }
    blocks.join("\n\n")
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

/// Run the markdown body through the preprocessing passes (HTML `<br>`
/// normalization + wide-table reflow) and termimad, returning the rendered
/// string with trailing newlines trimmed. Shared by both render paths.
fn render_markdown_body(body: &str, width: usize) -> String {
    let processed = preprocess_markdown(body);
    let processed = strip_code_fence_langs(&processed);
    let processed = reflow_wide_tables(&processed, width);
    let skin = brain_skin();
    let formatted = skin.text(&processed, Some(width));
    formatted.to_string().trim_end_matches('\n').to_string()
}

/// Build the full rendered string (label + markdown body + trailing blank
/// line) for a response. Empty bodies render as empty so the caller can
/// skip them.
fn render_to_string(label: ResponseLabel, body: &str) -> String {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    let rendered = render_markdown_body(trimmed, terminal_width());
    let mut out = String::with_capacity(rendered.len() + 32);
    out.push_str(label.ansi_prefix());
    out.push_str(&rendered);
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
    let rendered = render_markdown_body(trimmed, terminal_width());
    print!("{rendered}\n\n");
    // best-effort: a failed stdout flush means the pipe is closed and the
    // process is exiting anyway — nothing actionable to recover.
    let _ = stdout().flush();
}

/// Unlabeled one-shot render: the markdown body, no `Brain:`/`[proactive]`
/// label. Used for the approval-gate prompt + guidance, which are direct
/// responses to the user's own request and shouldn't carry a chat or nudge
/// label. (Deterministic subcommands like `brain capabilities` print their
/// pre-formatted body verbatim instead — see the `Plain` finalize branch.)
fn render_plain_direct(body: &str) {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return;
    }
    let rendered = render_markdown_body(trimmed, terminal_width());
    print!("{rendered}\n\n");
    // best-effort: a failed stdout flush means the pipe is closed and the
    // process is exiting anyway — nothing actionable to recover.
    let _ = stdout().flush();
}

/// Aggregates incoming WS frames into a single buffered response for the
/// one-shot path. The interactive path uses [`InteractiveAccumulator`]
/// which prints through an external printer instead.
/// How a one-shot response is rendered. `Chat` is the conversational
/// `brain chat "…"` look (transient `routing…` status line + `Brain:`
/// label). `Plain` is for deterministic subcommands like `brain capabilities`
/// that ride the same WS path but should print only their body — no status
/// line, no chat label.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum RenderStyle {
    #[default]
    Chat,
    Plain,
}

struct ResponseAccumulator {
    body: String,
    label: Option<ResponseLabel>,
    style: RenderStyle,
    /// Latest pipeline stage message ("routing…", "thinking…") for the
    /// one-shot elapsed spinner to display while frames are awaited.
    status: Option<String>,
}

impl ResponseAccumulator {
    fn new() -> Self {
        Self::with_style(RenderStyle::Chat)
    }

    fn with_style(style: RenderStyle) -> Self {
        Self {
            body: String::new(),
            label: None,
            style,
            status: None,
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
        // The gate body is self-describing ("Approval needed (external): …").
        // A `[proactive]` label on top misframes a direct response to the
        // user's own request as an unsolicited nudge — render it unlabeled.
        render_plain_direct(content);
    }

    fn render_error(message: &str) {
        let _ = clear_status_line();
        render_response_direct(ResponseLabel::Error, message);
    }

    fn finalize(self) -> Option<String> {
        match self.style {
            RenderStyle::Plain => {
                let trimmed = self.body.trim_end();
                if !trimmed.is_empty() {
                    // Deterministic subcommands (`brain capabilities`) send
                    // pre-formatted plain text. Print it verbatim rather than
                    // through the markdown renderer, which de-indents the
                    // manifest's `when:` lines and right-pads wrapped lines
                    // with trailing spaces. Plain never draws a status line
                    // (status frames + the spinner are suppressed upstream),
                    // so there's nothing to clear first.
                    print!("{trimmed}\n\n");
                    // best-effort: a closed stdout pipe means we're exiting
                    // anyway — nothing actionable to recover.
                    let _ = stdout().flush();
                }
            }
            RenderStyle::Chat => {
                if let Some(label) = self.label {
                    let _ = clear_status_line();
                    render_response_direct(label, &self.body);
                }
            }
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
    /// The daemon parked the signal on a confirmation gate. A one-shot
    /// client has no stdin loop to answer it, so this is terminal here:
    /// render the prompt + guidance and return rather than blocking to the
    /// server-side nonce timeout. Carries the prompt body (which includes
    /// the nonce the daemon minted).
    Approval(String),
    Error(String),
}

/// Guidance appended after a one-shot approval prompt. Without this the CLI
/// would block until the daemon timed the nonce out (60s External / 300s
/// Destructive); instead we explain how to actually grant the action.
const ONE_SHOT_APPROVAL_HINT: &str = "\
This action needs your approval, which can't be answered in one-shot mode.
- Interactive: run `brain chat`, then reply `approve <nonce>` (or `reject <nonce>`).
- Standing grant: pre-authorize it once via the `[confirm] standing_approvals` \
config so future runs skip the gate.";

fn apply_frame(acc: &mut ResponseAccumulator, frame: &serde_json::Value) -> FrameOutcome {
    match frame.get("type").and_then(|v| v.as_str()) {
        Some("status") => {
            // Plain (deterministic-subcommand) output suppresses the
            // transient `routing…` line; it's chat-render chrome. Otherwise
            // record the stage for the elapsed spinner (driven by the frame
            // loop) rather than rendering here, so the line keeps ticking.
            if acc.style != RenderStyle::Plain {
                if let Some(msg) = frame.get("message").and_then(|m| m.as_str()) {
                    acc.status = Some(msg.to_string());
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
                .unwrap_or("Approval required.")
                .to_string();
            FrameOutcome::Approval(body)
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
                let message = parsed.get("message").and_then(|v| v.as_str()).unwrap_or("");
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
        let llm_key = resolve_llm_api_key(config)?;
        let mut llm_cfg = config.llm.clone();
        if llm_cfg.providers.is_empty() {
            #[allow(deprecated)]
            {
                llm_cfg.api_key = llm_key;
            }
        }
        #[allow(deprecated)]
        let fallback_model = config.llm.model.clone();
        cortex::llm::select_provider(&llm_cfg)
            .await
            .map(|p| format!("{} ({})", p.model(), p.name()))
            .unwrap_or(fallback_model)
    };
    println!("  Cortex:  {}", cortex_model);
    println!("  Memory:  {}", config.data_dir().display());
    println!("  Synapse: connected to daemon (WebSocket)");
    println!();
    println!("Signals: {}", signals_line());
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
    // Back-channel that gates the next prompt redraw. After sending a line the
    // input thread blocks here until the async loop has finished handling it
    // (printed a command's output / dispatched a chat). Without this, the
    // thread re-prints "You: " the instant it hands off the line, racing a
    // synchronous `/status` block that's still printing — the two interleave.
    let (ack_tx, mut ack_rx) = mpsc::channel::<()>(1);
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
                    // Wait for the async loop to finish handling this line
                    // before redrawing the prompt. `None` means the loop has
                    // exited (ack_tx dropped) — wind down.
                    if ack_rx.blocking_recv().is_none() {
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
                        // Handle local commands (which print synchronously) and
                        // chat dispatch, then release the input thread to redraw
                        // its prompt. Quitting breaks *without* acking — the
                        // thread unblocks when `ack_tx` drops on loop exit, so it
                        // doesn't reprint "You: " after "Going dormant...".
                        let handled_locally = match line.as_str() {
                            "/quit" | "/exit" | "/q" => {
                                println!("Going dormant...");
                                break Ok(());
                            }
                            "/help" | "/?" => {
                                println!("Signals:\n{}", signals_help());
                                true
                            }
                            "/status" => {
                                if let Err(e) = show_status(config).await {
                                    eprintln!("status: {e}");
                                }
                                true
                            }
                            "/clear" => {
                                *session_id.lock().unwrap() = uuid::Uuid::new_v4().to_string();
                                println!("Session cleared — starting fresh conversation.");
                                true
                            }
                            s if looks_like_slash_command(s) => {
                                println!("Unknown signal: {s}");
                                println!("Available: {}", signals_line());
                                true
                            }
                            _ => false,
                        };

                        if !handled_locally {
                            let sid = session_id.lock().unwrap().clone();
                            let signal = serde_json::json!({
                                "content": line,
                                "session_id": sid,
                                "stream": true,
                            });
                            if let Err(e) =
                                sink.send(Message::Text(signal.to_string().into())).await
                            {
                                eprintln!("send failed: {e}");
                                break Err(anyhow::anyhow!(e));
                            }
                        }

                        // Line fully handled — let the input thread reprompt.
                        let _ = ack_tx.send(()).await;
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
    // Release the input thread if it's parked waiting for an ack (e.g. after
    // `/quit`): dropping the sender makes its `blocking_recv` return `None`.
    drop(ack_tx);
    let _ = reader_handle.await;
    // We deliberately do not join `input_thread`: rustyline may still be
    // blocked on a read, and there's no clean cross-platform way to wake it.
    // Returning here lets the runtime tear down on its own.
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
    fn signals_line_lists_every_canonical_name() {
        let line = signals_line();
        for sig in SIGNALS {
            assert!(line.contains(sig.name), "banner missing {}", sig.name);
        }
    }

    #[test]
    fn signal_catalog_mirrors_the_signals_table() {
        let catalog = signal_catalog();
        assert_eq!(catalog.len(), SIGNALS.len());
        for (doc, sig) in catalog.iter().zip(SIGNALS) {
            assert_eq!(doc.name, sig.name);
            assert_eq!(doc.summary, sig.summary);
        }
        // The real in-chat signals are present so the self-model can ground
        // them; the phantom the SOUL once fabricated is not.
        let names: Vec<&str> = catalog.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"/status"));
        assert!(!names.contains(&"/msg"));
    }

    #[test]
    fn strip_code_fence_langs_drops_info_string_only() {
        let input = "before\n```bash\nls -la\n```\nafter";
        let out = strip_code_fence_langs(input);
        // The language token is gone from the opening fence…
        assert!(!out.contains("```bash"), "info string survived: {out}");
        // …but the fence, content, and closing fence remain intact.
        assert_eq!(out, "before\n```\nls -la\n```\nafter");
    }

    #[test]
    fn strip_code_fence_langs_preserves_indent_and_other_fences() {
        // Indented opening fence keeps its indentation.
        assert_eq!(strip_code_fence_langs("  ```rust"), "  ```");
        // Bare fences (closing / language-less) are untouched.
        assert_eq!(strip_code_fence_langs("```"), "```");
        // A longer fence carries no info string and is left alone.
        assert_eq!(strip_code_fence_langs("````"), "````");
        // Non-fence lines pass through verbatim, even with inline backticks.
        assert_eq!(strip_code_fence_langs("use `cargo` now"), "use `cargo` now");
    }

    #[test]
    fn signals_help_covers_names_aliases_and_summaries() {
        let help = signals_help();
        for sig in SIGNALS {
            assert!(help.contains(sig.name), "help missing {}", sig.name);
            assert!(
                help.contains(sig.summary),
                "help missing summary for {}",
                sig.name
            );
            for alias in sig.aliases {
                assert!(help.contains(alias), "help missing alias {alias}");
            }
        }
    }

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

    const FOUR_COL_TABLE: &str = "\
| Tool | Tier | Network | Description |
| --- | --- | --- | --- |
| web_search | External | yes | Search the public web for fresh information |
| shell_exec | Execute | no | Run an allowlisted command in the sandbox |";

    #[test]
    fn wide_table_reflows_to_bullets() {
        // At 80 cols the 4-column table is far too wide and must reflow.
        let out = reflow_wide_tables(FOUR_COL_TABLE, 80);
        assert!(!out.contains("---"), "delimiter row should be gone: {out}");
        assert!(out.contains("- **Tool:** web_search"));
        assert!(out.contains("- **Network:** yes"));
        assert!(out.contains("- **Tool:** shell_exec"));
        // Rows are separated by a blank line.
        assert!(out.contains("\n\n- **Tool:** shell_exec"));
    }

    #[test]
    fn narrow_table_is_left_untouched() {
        // A table that comfortably fits is handed to termimad verbatim.
        let table = "\
| A | B |
| --- | --- |
| 1 | 2 |";
        assert_eq!(reflow_wide_tables(table, 100), table);
    }

    #[test]
    fn reflow_preserves_surrounding_prose() {
        let input = format!("Here are my tools:\n\n{FOUR_COL_TABLE}\n\nThat's all.");
        let out = reflow_wide_tables(&input, 80);
        assert!(out.starts_with("Here are my tools:"));
        assert!(out.trim_end().ends_with("That's all."));
        assert!(out.contains("- **Description:** Search the public web for fresh information"));
    }

    #[test]
    fn text_without_tables_is_unchanged() {
        let input = "Just a line with a | pipe but no table.\nAnd another.";
        assert_eq!(reflow_wide_tables(input, 80), input);
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
    fn default_style_is_chat() {
        assert_eq!(RenderStyle::default(), RenderStyle::Chat);
        assert_eq!(ResponseAccumulator::new().style, RenderStyle::Chat);
    }

    #[test]
    fn status_frame_is_recorded_for_spinner() {
        // Chat mode records the stage so the elapsed spinner can display it.
        let mut acc = ResponseAccumulator::new();
        let status =
            serde_json::json!({"type": "status", "stage": "thinking", "message": "thinking…"});
        assert!(matches!(
            apply_frame(&mut acc, &status),
            FrameOutcome::Continue
        ));
        assert_eq!(acc.status.as_deref(), Some("thinking…"));
    }

    #[test]
    fn plain_style_does_not_record_status() {
        let mut acc = ResponseAccumulator::with_style(RenderStyle::Plain);
        let status = serde_json::json!({"type": "status", "message": "routing…"});
        apply_frame(&mut acc, &status);
        assert!(acc.status.is_none());
    }

    #[test]
    fn plain_style_suppresses_status_line() {
        // In Plain mode a `status` frame must not render the `routing…` line.
        // We can't capture stdout here, but the gating reads `acc.style`, so
        // assert the style is carried and the frame is still consumed cleanly.
        let mut acc = ResponseAccumulator::with_style(RenderStyle::Plain);
        assert_eq!(acc.style, RenderStyle::Plain);
        let status = serde_json::json!({"type": "status", "message": "routing…"});
        assert!(matches!(
            apply_frame(&mut acc, &status),
            FrameOutcome::Continue
        ));
        // Status frames never contribute to the body or set a chat label.
        assert!(acc.body.is_empty());
        assert!(acc.label.is_none());
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
    fn apply_frame_returns_approval_with_prompt() {
        // One-shot: an approval gate must be terminal (carry the prompt so the
        // caller can render guidance and return) rather than Continue —
        // otherwise the loop blocks to the server-side nonce timeout (W1).
        let mut acc = ResponseAccumulator::new();
        let frame = serde_json::json!({"type": "approval_request", "content": "approve abc123?"});
        match apply_frame(&mut acc, &frame) {
            FrameOutcome::Approval(body) => assert_eq!(body, "approve abc123?"),
            _ => panic!("expected Approval"),
        }
    }

    #[test]
    fn apply_frame_approval_falls_back_when_content_missing() {
        let mut acc = ResponseAccumulator::new();
        let frame = serde_json::json!({"type": "approval_request"});
        match apply_frame(&mut acc, &frame) {
            FrameOutcome::Approval(body) => assert_eq!(body, "Approval required."),
            _ => panic!("expected Approval"),
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
