//! Scheduled intent management commands.
//!
//! All operations go through the running daemon's HTTP API to ensure
//! a single shared SignalProcessor (no RuVector lock contention).

use std::time::Duration;

use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum SchedulesAction {
    /// List all scheduled intents.
    List {
        /// Filter by namespace
        #[arg(long, short)]
        namespace: Option<String>,
    },
    /// Cancel a scheduled intent.
    Cancel {
        /// The ID of the intent to cancel.
        id: String,
    },
}

pub(crate) async fn cmd_schedules(
    config: &brain_core::BrainConfig,
    action: SchedulesAction,
) -> anyhow::Result<()> {
    let daemon_url = crate::bootstrap::require_daemon(config).await?;
    let api_key = config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    match action {
        SchedulesAction::List { namespace } => {
            let mut url = format!("{daemon_url}/v1/schedules");
            if let Some(ns) = &namespace {
                url.push_str(&format!("?namespace={ns}"));
            }

            let intents: Vec<serde_json::Value> = client
                .get(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;

            if intents.is_empty() {
                println!("No scheduled intents found.");
            } else {
                println!(
                    "{:<38} {:<30} {:<15} {:<10} {:<15}",
                    "ID", "Description", "Cron", "Status", "Namespace"
                );
                println!("{:-<110}", "");
                for intent in &intents {
                    let desc = intent["description"].as_str().unwrap_or("");
                    let desc_display = if desc.len() > 27 {
                        format!("{}...", &desc[..27])
                    } else {
                        desc.to_string()
                    };
                    println!(
                        "{:<38} {:<30} {:<15} {:<10} {:<15}",
                        intent["id"].as_str().unwrap_or(""),
                        desc_display,
                        intent["cron"].as_str().unwrap_or("-"),
                        intent["status"].as_str().unwrap_or(""),
                        intent["namespace"].as_str().unwrap_or(""),
                    );
                }
            }
        }
        SchedulesAction::Cancel { id } => {
            let resp = client
                .delete(format!("{daemon_url}/v1/schedules/{id}"))
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await?;

            if resp.status().is_success() {
                println!("Successfully cancelled scheduled intent: {id}");
            } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
                println!("No scheduled intent found with ID: {id}");
            } else {
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("Failed to cancel schedule: {body}");
            }
        }
    }
    Ok(())
}
