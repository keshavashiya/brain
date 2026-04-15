//! CLI commands for budget management.

use anyhow::Result;
use budget::{BudgetPolicy, CostBudget, ResourceKind, SqliteBudget};
use clap::Subcommand;
use storage::SqlitePool;

#[derive(Subcommand)]
pub(crate) enum BudgetAction {
    /// Show current consumption status
    Status,

    /// Record consumption (for testing/manual tracking)
    Record {
        /// Provider name (e.g., openai, claude-code)
        provider: String,
        /// Resource type: llm_input_tokens, llm_output_tokens, sandbox_wall_clock_ms
        resource: String,
        /// Units consumed
        units: u64,
    },

    /// Check if a consumption would be within budget
    Check {
        /// Provider name (e.g., openai, claude-code)
        provider: String,
        /// Resource type
        resource: String,
        /// Units to check
        units: u64,
    },
}

pub(crate) async fn cmd_budget(
    config: &brain_core::BrainConfig,
    action: BudgetAction,
) -> Result<()> {
    let db_path = config.data_dir().join("db/brain.db");
    if !db_path.exists() {
        anyhow::bail!("Database not found. Run `brain init` first.");
    }

    let pool = SqlitePool::open(&db_path)?;
    let policy = BudgetPolicy::default();
    let budget = SqliteBudget::new(pool, policy);
    budget.ensure_tables()?;

    match action {
        BudgetAction::Status => {
            let status = budget.status().await?;

            if status.hourly_consumption.is_empty() && status.daily_consumption.is_empty() {
                println!("No budget consumption recorded.");
                return Ok(());
            }

            if !status.hourly_consumption.is_empty() {
                println!("Hourly consumption:");
                let mut items: Vec<_> = status.hourly_consumption.iter().collect();
                items.sort_by(|a, b| a.0.cmp(b.0));
                for (key, units) in items {
                    println!("  {key}: {units}");
                }
            }

            if !status.daily_consumption.is_empty() {
                println!();
                println!("Daily consumption:");
                let mut items: Vec<_> = status.daily_consumption.iter().collect();
                items.sort_by(|a, b| a.0.cmp(b.0));
                for (key, units) in items {
                    println!("  {key}: {units}");
                }
            }

            if !status.warnings.is_empty() {
                println!();
                println!("Warnings:");
                for warning in &status.warnings {
                    println!("  {warning}");
                }
            }
        }

        BudgetAction::Record {
            provider,
            resource,
            units,
        } => {
            let resource_kind = parse_resource_kind(&resource)?;
            budget.record(&provider, &resource_kind, units).await?;
            println!("Recorded {units} {resource} for {provider}.");
        }

        BudgetAction::Check {
            provider,
            resource,
            units,
        } => {
            let resource_kind = parse_resource_kind(&resource)?;
            let decision = budget.check(&provider, &resource_kind, units).await?;

            match decision {
                budget::BudgetDecision::Allowed => {
                    println!("Budget check: ALLOWED ({units} {resource} for {provider})");
                }
                budget::BudgetDecision::Warn { consumed_pct } => {
                    println!("Budget check: WARNING ({consumed_pct:.0}% consumed)");
                }
                budget::BudgetDecision::Exceeded { ceiling, consumed } => {
                    println!("Budget check: EXCEEDED (ceiling: {ceiling}, consumed: {consumed})");
                }
            }
        }
    }

    Ok(())
}

fn parse_resource_kind(resource: &str) -> Result<ResourceKind> {
    match resource.to_lowercase().as_str() {
        "llm_input_tokens" => Ok(ResourceKind::LlmInputTokens),
        "llm_output_tokens" => Ok(ResourceKind::LlmOutputTokens),
        "sandbox_wall_clock_ms" => Ok(ResourceKind::SandboxWallClockMs),
        other if other.starts_with("api_call:") => Ok(ResourceKind::ApiCall {
            endpoint: other["api_call:".len()..].to_string(),
        }),
        other if other.starts_with("agent_delegation:") => Ok(ResourceKind::AgentDelegation {
            agent: other["agent_delegation:".len()..].to_string(),
        }),
        other => anyhow::bail!(
            "Invalid resource: {other}. Must be one of: llm_input_tokens, llm_output_tokens, sandbox_wall_clock_ms, api_call:<endpoint>, agent_delegation:<agent>"
        ),
    }
}
