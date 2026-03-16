//! Scheduled intent management commands.

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
    let pool = storage::SqlitePool::open(&config.sqlite_path())?;

    match action {
        SchedulesAction::List { namespace } => {
            let intents = pool.list_scheduled_intents(namespace.as_deref())?;
            if intents.is_empty() {
                println!("No scheduled intents found.");
            } else {
                println!("{:<38} {:<30} {:<15} {:<10} {:<15}", "ID", "Description", "Cron", "Status", "Namespace");
                println!("{:-<110}", "");
                for intent in intents {
                    println!(
                        "{:<38} {:<30} {:<15} {:<10} {:<15}",
                        intent.id,
                        if intent.description.len() > 27 { format!("{}...", &intent.description[..27]) } else { intent.description.clone() },
                        intent.cron.as_deref().unwrap_or("-"),
                        intent.status,
                        intent.namespace
                    );
                }
            }
        }
        SchedulesAction::Cancel { id } => {
            if pool.cancel_scheduled_intent(&id)? {
                println!("Successfully cancelled scheduled intent: {}", id);
            } else {
                println!("No scheduled intent found with ID: {}", id);
            }
        }
    }
    Ok(())
}
