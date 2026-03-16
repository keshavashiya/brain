//! Chat commands — interactive and non-interactive conversation modes.

use std::io::{stdout, Write};
use std::sync::Arc;

use crossterm::cursor;
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::terminal;
use crossterm::ExecutableCommand;
use rustyline::DefaultEditor;

use crate::session::{BrainSession, PrepareResult};
use crate::status::show_status;

/// Display proactive nudges at session start: pending outbox items and open loops.
fn show_proactive_nudges(brain: &BrainSession, config: &brain_core::BrainConfig) {
    let mut nudges: Vec<String> = Vec::new();

    if let Ok(pending) = brain.db().pending_notifications(5) {
        for n in &pending {
            nudges.push(n.content.clone());
            let _ = brain.db().mark_notification_delivered(&n.id);
        }
    }

    if config.proactivity.enabled && config.proactivity.open_loop.enabled {
        let detector = ganglia::OpenLoopDetector::new(
            brain.db().clone(),
            ganglia::OpenLoopConfig {
                scan_window_hours: config.proactivity.open_loop.scan_window_hours,
                resolution_window_hours: config.proactivity.open_loop.resolution_window_hours,
                max_reminders: 3,
            },
        );
        if let Ok(reminders) = detector.generate_reminders() {
            for r in reminders {
                nudges.push(r.content);
            }
        }
    }

    if !nudges.is_empty() {
        println!("\x1b[33m📌 Nudges:\x1b[0m");
        for nudge in &nudges {
            println!("  \x1b[33m• {nudge}\x1b[0m");
        }
        println!();
    }
}

pub(crate) async fn chat_non_interactive(
    config: &brain_core::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    use futures::StreamExt;

    let mut brain = BrainSession::new(config).await?;

    show_proactive_nudges(&brain, config);

    match brain.prepare_context(message).await? {
        PrepareResult::ActionResult(text) => {
            println!("{text}");
        }
        PrepareResult::LlmReady(messages) => {
            match brain.llm.generate_stream(&messages).await {
                Ok(mut stream) => {
                    let mut full_response = String::new();
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(c) => {
                                print!("{}", c.content);
                                let _ = stdout().flush();
                                full_response.push_str(&c.content);
                                if c.is_done {
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("\nStream error: {e}");
                                break;
                            }
                        }
                    }
                    println!();
                    brain.finalize_response(message, &full_response)?;
                }
                Err(_) => {
                    let response = brain.llm.generate(&messages).await?;
                    println!("{}", response.content);
                    brain.finalize_response(message, &response.content)?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) async fn chat_interactive(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
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

    let mut brain = BrainSession::new(config).await?;

    if brain.processor.semantic().is_some() {
        println!("  Synapse: standalone (full memory)");
    } else {
        println!("  Synapse: standalone (episodic only)");
    }

    println!();
    println!("Signals: /status  /clear  /quit");
    println!();
    let mut rl = DefaultEditor::new()?;
    let history_path = config.data_dir().join("history.txt");
    let _ = rl.load_history(&history_path);

    show_proactive_nudges(&brain, config);

    if brain.semantic_fact_count() == 0 && brain.episode_count() == 0 {
        let mut out = stdout();
        out.execute(SetForegroundColor(Color::Green))?;
        out.execute(Print("Brain: "))?;
        out.execute(ResetColor)?;
        println!("{}", cortex::context::ONBOARDING_GREETING);
        println!();
        brain.record_onboarding_greeting();
    }

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
                        brain.clear_history();
                        println!("Short-term memory cleared.");
                        continue;
                    }
                    s if s.starts_with('/') => {
                        println!("Unknown signal: {s}");
                        println!("Available: /status  /clear  /quit");
                        continue;
                    }
                    _ => {}
                }

                let phase = Arc::new(std::sync::atomic::AtomicU8::new(0));
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let phase_c = Arc::clone(&phase);
                let stop_c = Arc::clone(&stop);
                let spinner_handle = tokio::spawn(async move {
                    let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
                    let mut i = 0;
                    while !stop_c.load(std::sync::atomic::Ordering::Relaxed) {
                        let label = match phase_c.load(std::sync::atomic::Ordering::Relaxed) {
                            0 => "Recalling memories",
                            _ => "Thinking",
                        };
                        {
                            let mut out = stdout();
                            let _ = out.execute(cursor::MoveToColumn(0));
                            let _ = out.execute(terminal::Clear(terminal::ClearType::CurrentLine));
                            let _ = write!(
                                out,
                                "\x1b[90m  {} {}\x1b[0m",
                                frames[i % frames.len()],
                                label
                            );
                            let _ = out.flush();
                        }
                        i += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                    }
                    let mut out = stdout();
                    let _ = out.execute(cursor::MoveToColumn(0));
                    let _ = out.execute(terminal::Clear(terminal::ClearType::CurrentLine));
                    let _ = out.flush();
                });

                let prepare_result = brain.prepare_context(input).await;

                let mut spinner_handle = Some(spinner_handle);
                let dismiss_spinner =
                    |stop: &Arc<std::sync::atomic::AtomicBool>,
                     handle: &mut Option<tokio::task::JoinHandle<()>>| {
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        handle.take()
                    };

                match prepare_result {
                    Ok(PrepareResult::ActionResult(text)) => {
                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle) {
                            let _ = h.await;
                        }
                        let mut out = stdout();
                        out.execute(SetForegroundColor(Color::Green))?;
                        out.execute(Print("Brain: "))?;
                        out.execute(ResetColor)?;
                        println!("{text}");
                    }
                    Ok(PrepareResult::LlmReady(messages)) => {
                        phase.store(1, std::sync::atomic::Ordering::Relaxed);

                        let stream_result = brain.llm.generate_stream(&messages).await;
                        match stream_result {
                            Ok(mut stream) => {
                                use futures::StreamExt;
                                let mut full_response = String::new();
                                while let Some(chunk) = stream.next().await {
                                    match chunk {
                                        Ok(c) => {
                                            if let Some(h) =
                                                dismiss_spinner(&stop, &mut spinner_handle)
                                            {
                                                let _ = h.await;
                                                let mut out = stdout();
                                                out.execute(SetForegroundColor(Color::Green))?;
                                                out.execute(Print("Brain: "))?;
                                                out.execute(ResetColor)?;
                                                let _ = out.flush();
                                            }
                                            print!("{}", c.content);
                                            let _ = stdout().flush();
                                            full_response.push_str(&c.content);
                                            if c.is_done {
                                                break;
                                            }
                                        }
                                        Err(e) => {
                                            if let Some(h) =
                                                dismiss_spinner(&stop, &mut spinner_handle)
                                            {
                                                let _ = h.await;
                                            }
                                            eprintln!("\nStream error: {e}");
                                            break;
                                        }
                                    }
                                }
                                if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle) {
                                    let _ = h.await;
                                }
                                println!();
                                if let Err(e) = brain.finalize_response(input, &full_response) {
                                    tracing::warn!("Failed to store response: {e}");
                                }
                            }
                            Err(_) => {
                                match brain.llm.generate(&messages).await {
                                    Ok(response) => {
                                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle)
                                        {
                                            let _ = h.await;
                                        }
                                        let mut out = stdout();
                                        out.execute(SetForegroundColor(Color::Green))?;
                                        out.execute(Print("Brain: "))?;
                                        out.execute(ResetColor)?;
                                        println!("{}", response.content);
                                        if let Err(e) =
                                            brain.finalize_response(input, &response.content)
                                        {
                                            tracing::warn!("Failed to store response: {e}");
                                        }
                                    }
                                    Err(e) => {
                                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle)
                                        {
                                            let _ = h.await;
                                        }
                                        let msg = e.to_string();
                                        if msg.contains("timed out") || msg.contains("Timeout") {
                                            eprintln!("LLM timed out — model may still be loading. Try again.");
                                        } else if msg.contains("error sending request")
                                            || msg.contains("connection refused")
                                            || msg.contains("Connection refused")
                                        {
                                            eprintln!("LLM unreachable — is Ollama running? (`ollama serve`)");
                                        } else {
                                            eprintln!("Error: {msg}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle) {
                            let _ = h.await;
                        }
                        let msg = e.to_string();
                        if msg.contains("timed out") || msg.contains("Timeout") {
                            eprintln!("LLM timed out — model may still be loading. Try again.");
                        } else if msg.contains("error sending request")
                            || msg.contains("connection refused")
                            || msg.contains("Connection refused")
                        {
                            eprintln!("LLM unreachable — is Ollama running? (`ollama serve`)");
                        } else {
                            eprintln!("Error: {msg}");
                        }
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
