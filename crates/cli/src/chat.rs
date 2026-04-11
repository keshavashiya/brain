//! Chat commands — interactive and non-interactive conversation modes.

use std::io::stdout;
use std::time::Duration;

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::ExecutableCommand;
use rustyline::DefaultEditor;

use crate::status::show_status;

/// Send a chat message via a running daemon's HTTP API.
///
/// `session_id` is included so the daemon groups episodes into the same
/// conversation. When the user types `/clear`, the caller rotates the
/// session_id to start a fresh conversation.
///
/// Returns `Ok(Some(response_text))` if the server responds,
/// or `Err` on failures.
async fn try_server_chat_via_url(
    daemon_url: &str,
    config: &brain_core::BrainConfig,
    message: &str,
    session_id: &str,
) -> anyhow::Result<Option<String>> {
    let signal_url = format!("{daemon_url}/v1/signals");
    let api_key = config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default();

    let client = reqwest::Client::new();
    let resp = client
        .post(&signal_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&serde_json::json!({
            "content": message,
            "session_id": session_id,
        }))
        .timeout(Duration::from_secs(120))
        .send()
        .await?
        .json::<serde_json::Value>()
        .await?;

    let text = if let Some(t) = resp["response"]["value"].as_str() {
        t.to_string()
    } else if let Some(e) = resp["response"]["Error"].as_str() {
        e.to_string()
    } else {
        String::new()
    };
    if text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(text))
    }
}

pub(crate) async fn chat_non_interactive(
    config: &brain_core::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    let daemon_url = crate::bootstrap::require_daemon(config).await?;
    let session_id = uuid::Uuid::new_v4().to_string();

    let response = try_server_chat_via_url(&daemon_url, config, message, &session_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Daemon returned empty response"))?;

    println!("{response}");
    Ok(())
}

pub(crate) async fn chat_interactive(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
    let daemon_url = crate::bootstrap::require_daemon(config).await?;
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
    println!("  Synapse: connected to daemon (HTTP)");
    println!();
    println!("Signals: /status  /clear  /quit");
    println!();
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

                let response =
                    try_server_chat_via_url(&daemon_url, config, input, &session_id).await;

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
