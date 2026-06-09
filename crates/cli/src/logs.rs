//! `brain logs analyze` — log pattern detection.
//!
//! Thin CLI surface over [`backends::logs`]: the deterministic pattern pass and
//! the optional, strictly-grounded LLM narration both live in the backend; this
//! module only parses args, maps `--source` onto [`backends::logs::LogSource`],
//! and prints. The same `analyze` digest is what the `logs.analyze` chat
//! tool-loop capability returns.

use anyhow::Result;
use brain::BrainConfig;

/// Where to read logs from. The CLI keeps its own `clap`-deriving enum and maps
/// it onto the dependency-free [`backends::logs::LogSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LogSource {
    /// The daemon's own rotated logs in `~/.brain/logs/`.
    Brain,
    /// OS logs: `log show` (macOS) / sandboxed `journalctl` (Linux).
    System,
}

impl From<LogSource> for backends::logs::LogSource {
    fn from(s: LogSource) -> Self {
        match s {
            LogSource::Brain => backends::logs::LogSource::Brain,
            LogSource::System => backends::logs::LogSource::System,
        }
    }
}

#[derive(clap::Subcommand)]
pub enum LogsAction {
    /// Scan recent logs for recurring error/warning patterns, with an optional
    /// plain-language summary.
    Analyze {
        /// Where to read logs from: the daemon's own log, or the OS log.
        #[arg(long, value_enum, default_value = "brain")]
        source: LogSource,
        /// How far back to look, as `<n>{m|h|d}` (e.g. `30m`, `1h`, `2d`).
        /// For `--source system` this bounds the OS query; for `--source brain`
        /// it drops lines older than the window (lines without a parseable
        /// timestamp are kept).
        #[arg(long, default_value = "1h")]
        since: String,
        /// Cap on how many of the most recent lines to analyse.
        #[arg(long, default_value_t = 2000)]
        lines: usize,
        /// Skip the LLM summary; print the deterministic digest only.
        #[arg(long)]
        no_llm: bool,
    },
}

/// Entry point for `brain logs <action>`.
pub async fn cmd_logs(config: &BrainConfig, action: LogsAction) -> Result<()> {
    match action {
        LogsAction::Analyze {
            source,
            since,
            lines,
            no_llm,
        } => {
            // The deterministic digest is the real deliverable; narration is a
            // bonus that degrades gracefully when no provider is reachable.
            let digest = backends::logs::analyze(config, source.into(), &since, lines).await?;
            println!("{digest}");

            if no_llm {
                return Ok(());
            }
            match backends::logs::narrate(config, &digest).await {
                Ok(summary) => {
                    println!("\nSummary");
                    println!("{summary}");
                }
                Err(e) => println!("\n(LLM summary unavailable: {e})"),
            }
            Ok(())
        }
    }
}
