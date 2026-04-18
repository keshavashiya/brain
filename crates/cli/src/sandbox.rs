//! CLI commands for sandbox execution.

use anyhow::Result;
use clap::Subcommand;
use sandbox::{IsolatedSandbox, SandboxCommand, SandboxExecutor};

#[derive(Subcommand)]
pub(crate) enum SandboxAction {
    /// Execute a command in the sandbox
    Run {
        /// Command to execute (quoted string)
        #[arg(last = true, required = true)]
        command: Vec<String>,
        /// Action tier (read, write, execute, destructive, external)
        #[arg(long, default_value = "execute")]
        tier: String,
        /// Working directory
        #[arg(long)]
        workdir: Option<String>,
        /// Timeout in seconds
        #[arg(long)]
        timeout: Option<u64>,
    },
}

pub(crate) async fn cmd_sandbox(
    config: &brain_core::BrainConfig,
    action: SandboxAction,
) -> Result<()> {
    match action {
        SandboxAction::Run {
            command,
            tier,
            workdir,
            timeout,
        } => {
            if command.is_empty() {
                anyhow::bail!("No command provided. Use: brain sandbox run -- \"echo hello\"");
            }

            let binary = command[0].clone();
            let args = command[1..].to_vec();

            let tier_kind = match tier.to_lowercase().as_str() {
                "read" => sandbox::ActionTier::Read,
                "write" => sandbox::ActionTier::Write,
                "execute" => sandbox::ActionTier::Execute,
                "destructive" => sandbox::ActionTier::Destructive,
                "external" => sandbox::ActionTier::External,
                other => anyhow::bail!(
                    "Invalid tier: {other}. Must be: read, write, execute, destructive, external"
                ),
            };

            let mut cmd = SandboxCommand::new(binary, args).with_tier(tier_kind);

            if let Some(dir) = workdir {
                cmd = cmd.with_workdir(std::path::PathBuf::from(dir));
            }

            if let Some(secs) = timeout {
                cmd = cmd.with_timeout(std::time::Duration::from_secs(secs));
            }

            let default_timeout =
                std::time::Duration::from_secs(config.security.exec_timeout_seconds as u64);
            let sandbox =
                IsolatedSandbox::new(config.security.exec_allowlist.clone(), default_timeout)
                    .with_allowed_paths(vec![
                        std::path::PathBuf::from(&config.brain.data_dir),
                        std::env::current_dir().unwrap_or_default(),
                    ]);
            let outcome = sandbox.run(cmd).await?;

            if !outcome.stdout.is_empty() {
                println!("{}", outcome.stdout);
            }
            if !outcome.stderr.is_empty() {
                eprintln!("{}", outcome.stderr);
            }

            println!(
                "Exit code: {}, Duration: {:.2}s",
                outcome.exit_code,
                outcome.duration.as_secs_f64()
            );
        }
    }

    Ok(())
}
