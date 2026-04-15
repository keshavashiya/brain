//! CLI commands for confirmation engine management.

use anyhow::Result;
use clap::Subcommand;
use confirm::{ApprovalDecision, ConfirmationEngine, SqliteConfirmationEngine};
use storage::SqlitePool;

#[derive(Subcommand)]
pub(crate) enum ConfirmAction {
    /// List pending approval requests
    Pending,

    /// Approve a request
    Approve {
        /// Nonce of the approval request
        nonce: String,
    },

    /// Reject a request
    Reject {
        /// Nonce of the approval request
        nonce: String,
        /// Optional reason for rejection
        #[arg(long)]
        reason: Option<String>,
    },

    /// Check status of a request
    Status {
        /// Nonce of the approval request
        nonce: String,
    },
}

pub(crate) async fn cmd_confirm(
    config: &brain_core::BrainConfig,
    action: ConfirmAction,
) -> Result<()> {
    let db_path = config.data_dir().join("db/brain.db");
    if !db_path.exists() {
        anyhow::bail!("Database not found. Run `brain init` first.");
    }

    let pool = SqlitePool::open(&db_path)?;
    let engine = SqliteConfirmationEngine::new(pool);
    engine.ensure_tables()?;

    match action {
        ConfirmAction::Pending => {
            let pending = engine.pending().await?;

            if pending.is_empty() {
                println!("No pending approval requests.");
                return Ok(());
            }

            println!(
                "{:<8} {:<12} {:<40} {:<10}",
                "Nonce", "Tier", "Description", "Timeout"
            );
            println!("{}", "-".repeat(80));

            for spec in &pending {
                let nonce_short = &spec.nonce[..8];
                let timeout_str = format!("{}s", spec.timeout.as_secs());
                println!(
                    "{:<8} {:<12} {:<40} {:<10}",
                    nonce_short,
                    spec.tier.to_string(),
                    &spec.action_description[..spec.action_description.len().min(38)],
                    timeout_str,
                );
            }
        }

        ConfirmAction::Approve { nonce } => {
            engine.respond(&nonce, ApprovalDecision::Approve).await?;
            println!("Approval request {nonce} approved.");
        }

        ConfirmAction::Reject { nonce, reason } => {
            let decision = match reason {
                Some(r) => ApprovalDecision::RejectWithReason(r),
                None => ApprovalDecision::Reject,
            };
            engine.respond(&nonce, decision).await?;
            println!("Approval request {nonce} rejected.");
        }

        ConfirmAction::Status { nonce } => {
            let status = engine.status(&nonce).await?;
            match status {
                confirm::ApprovalStatus::Pending { since } => {
                    println!("Status: PENDING (since {since})");
                }
                confirm::ApprovalStatus::Resolved {
                    outcome,
                    resolved_at,
                } => {
                    println!("Status: RESOLVED ({resolved_at})");
                    match outcome {
                        confirm::ApprovalOutcome::Approved => {
                            println!("  Outcome: Approved");
                        }
                        confirm::ApprovalOutcome::Rejected { reason } => {
                            println!("  Outcome: Rejected ({reason})");
                        }
                        confirm::ApprovalOutcome::TimedOut => {
                            println!("  Outcome: Timed out");
                        }
                        confirm::ApprovalOutcome::Aborted { reason } => {
                            println!("  Outcome: Aborted ({reason})");
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
