//! Interactive chat loop. Runs two halves concurrently so the user can reply
//! to an approval prompt (`approve <nonce>` / `reject <nonce>`) while a prior
//! signal is still in-flight. Without this, a 60s confirmation window always
//! expires before rustyline gets the next line. The reader half receives
//! frames and emits rendered text via rustyline's external printer (so prints
//! land cleanly above the prompt); the sender half reads lines and pushes them
//! on the socket.

use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use rustyline::{DefaultEditor, ExternalPrinter};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

use crate::encryption::resolve_llm_api_key;
use crate::status::show_status;

use super::frames::{extract_complete_body, ResponseAccumulator};
use super::render::{render_to_string, ResponseLabel};
use super::signals::{signals_help, signals_line};
use super::transport::{connect_ws_session, first_api_key, ws_url, WsStream};

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
pub(super) fn render_status_for_printer(stage: &str, message: &str, overwrite: bool) -> String {
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
pub(super) fn with_overwrite_prefix(rendered: String, overwrite: bool) -> String {
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
pub(super) fn looks_like_slash_command(s: &str) -> bool {
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
