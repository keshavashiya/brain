//! CLI commands for audit trail management.

use anyhow::Result;
use audit::{ActionTier, AuditOutcome, AuditQuerySpec, AuditTrail, SqliteAuditTrail};
use chrono::Duration;
use clap::Subcommand;
use storage::SqlitePool;

#[derive(Subcommand)]
pub(crate) enum AuditAction {
    /// List recent audit entries
    List {
        /// Show entries from today
        #[arg(long)]
        today: bool,
        /// Number of entries to show (default: 20)
        #[arg(long, short, default_value = "20")]
        limit: usize,
        /// Filter by tier
        #[arg(long)]
        tier: Option<String>,
        /// Filter by outcome
        #[arg(long)]
        outcome: Option<String>,
    },

    /// Show activity summary
    Summary {
        /// Time window in hours (default: 24)
        #[arg(long, default_value = "24")]
        hours: i64,
    },

    /// Prune entries older than a threshold
    Prune {
        /// Prune entries older than N days
        #[arg(long, default_value = "30")]
        older_than_days: i64,
    },
}

pub(crate) async fn cmd_audit(config: &brain_core::BrainConfig, action: AuditAction) -> Result<()> {
    // Phase 1a: Direct SQLite access since daemon doesn't expose audit endpoints yet

    match action {
        AuditAction::List {
            today,
            limit,
            tier,
            outcome,
        } => {
            let mut spec = AuditQuerySpec::default().limit(limit);
            if today {
                spec = AuditQuerySpec::today().limit(limit);
            }
            if let Some(t) = tier {
                spec = spec.tier(match t.to_lowercase().as_str() {
                    "read" => ActionTier::Read,
                    "write" => ActionTier::Write,
                    "execute" => ActionTier::Execute,
                    "destructive" => ActionTier::Destructive,
                    "external" => ActionTier::External,
                    _ => anyhow::bail!(
                        "Invalid tier: {t}. Must be: read, write, execute, destructive, external"
                    ),
                });
            }
            if let Some(o) = outcome {
                spec = spec.outcome(match o.to_lowercase().as_str() {
                    "success" => AuditOutcome::Success,
                    "failure" => AuditOutcome::Failure,
                    "cancelled" => AuditOutcome::Cancelled,
                    "timeout" => AuditOutcome::Timeout,
                    _ => anyhow::bail!(
                        "Invalid outcome: {o}. Must be: success, failure, cancelled, timeout"
                    ),
                });
            }

            // For Phase 1a, we use direct SQLite access since the daemon
            // doesn't yet expose audit endpoints via HTTP.
            let db_path = config.data_dir().join("db/brain.db");
            if !db_path.exists() {
                anyhow::bail!("Database not found. Run `brain init` first.");
            }

            let pool = SqlitePool::open(&db_path)?;
            let trail = SqliteAuditTrail::new(pool);
            trail.ensure_tables()?;

            let entries = trail.query(spec).await?;

            if entries.is_empty() {
                println!("No audit entries found.");
                return Ok(());
            }

            println!(
                "{:<8} {:<20} {:<12} {:<12} {:<10} Action",
                "ID", "Timestamp", "Tier", "Outcome", "Exit"
            );
            println!("{}", "-".repeat(120));

            for entry in &entries {
                let id_short = &entry.id[..8];
                let exit_str = entry
                    .exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "{:<8} {:<20} {:<12} {:<12} {:<10} {}",
                    id_short,
                    &entry.timestamp[..19],
                    entry.tier.to_string(),
                    entry.outcome.to_string(),
                    exit_str,
                    &entry.action[..entry.action.len().min(40)],
                );
            }
        }

        AuditAction::Summary { hours } => {
            let db_path = config.data_dir().join("db/brain.db");
            if !db_path.exists() {
                anyhow::bail!("Database not found. Run `brain init` first.");
            }

            let pool = SqlitePool::open(&db_path)?;
            let trail = SqliteAuditTrail::new(pool);
            trail.ensure_tables()?;

            let window = Duration::hours(hours);
            let summary = trail.summarize(window).await?;

            println!("Audit Summary (last {hours} hours):");
            println!("  Total entries: {}", summary.total_entries);
            println!();
            println!("  By outcome:");
            for (outcome, count) in &summary.by_outcome {
                println!("    {outcome}: {count}");
            }
            println!();
            println!("  By tier:");
            for (tier, count) in &summary.by_tier {
                println!("    {tier}: {count}");
            }
            println!();
            println!("  By source:");
            for (source, count) in &summary.by_source {
                println!("    {source}: {count}");
            }
            if let Some(avg) = summary.avg_duration_ms {
                println!();
                println!("  Avg duration: {:.0}ms", avg);
            }
        }

        AuditAction::Prune { older_than_days } => {
            let db_path = config.data_dir().join("db/brain.db");
            if !db_path.exists() {
                anyhow::bail!("Database not found. Run `brain init` first.");
            }

            let pool = SqlitePool::open(&db_path)?;
            let trail = SqliteAuditTrail::new(pool);
            trail.ensure_tables()?;

            let window = Duration::days(older_than_days);
            let pruned = trail.prune(window).await?;

            println!("Pruned {pruned} audit entries older than {older_than_days} days.");
            println!("Note: Phase 1a prune is not yet fully implemented (immutable triggers).");
        }
    }

    Ok(())
}
