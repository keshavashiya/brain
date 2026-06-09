//! `brain security audit` — offline security-posture review (Issue 140).
//!
//! Thin CLI surface over [`backends::security`]: the operator runs the audit
//! directly (no consent gate — they're invoking the binary themselves), and we
//! print the same `render()` the chat tool-loop relays. The auditor, the
//! findings, and the rationale all live in `backends::security`; this module
//! only parses args and prints (or serialises) the report.

use anyhow::Result;
use brain::BrainConfig;

#[derive(clap::Subcommand)]
pub enum SecurityAction {
    /// Audit the security posture of the current configuration and report
    /// severity-ranked findings.
    Audit {
        /// Emit findings as JSON instead of the human-readable report.
        #[arg(long)]
        json: bool,
    },
}

/// Entry point for `brain security <action>`.
pub async fn cmd_security(config: &BrainConfig, action: SecurityAction) -> Result<()> {
    match action {
        SecurityAction::Audit { json } => {
            let findings = backends::security::audit(config);
            if json {
                println!("{}", serde_json::to_string_pretty(&findings)?);
            } else {
                print!("{}", backends::security::render(&findings));
            }
            Ok(())
        }
    }
}
