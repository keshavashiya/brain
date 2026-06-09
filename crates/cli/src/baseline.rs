//! `brain baseline capture/diff/list` — system baseline + drift detection.
//!
//! Thin CLI surface over [`backends::baseline`]: the snapshot format, the
//! capture/diff/list logic, and the storage layout all live in the backend.
//! This module only parses args, assembles the live capability inventory from
//! the native descriptors (so the backend stays a pure function of its inputs —
//! it never reaches back into the binary crate), and prints the rendered
//! report. The same functions back the `baseline.*` chat tool-loop capability.

use anyhow::Result;
use brain::BrainConfig;

use crate::capabilities::capability_inventory;

#[derive(clap::Subcommand)]
pub enum BaselineAction {
    /// Capture a new baseline snapshot of the current system state and store it
    /// as the next version.
    Capture {
        /// Optional human-readable label recorded alongside the snapshot.
        #[arg(long)]
        label: Option<String>,
    },
    /// Compare two snapshots and report drift (added / removed / changed facts).
    ///
    /// With no flags, compares the latest stored baseline against the *current*
    /// live system state. `--from <v>` picks the stored baseline to compare from;
    /// `--to <v>` compares against another stored baseline instead of live state.
    Diff {
        /// Stored baseline version to compare *from* (default: the latest).
        #[arg(long)]
        from: Option<u32>,
        /// Stored baseline version to compare *to* (default: current live state).
        #[arg(long)]
        to: Option<u32>,
    },
    /// List stored baseline snapshots, newest first.
    List,
}

/// Entry point for `brain baseline <action>`.
pub async fn cmd_baseline(config: &BrainConfig, action: BaselineAction) -> Result<()> {
    let report = match action {
        BaselineAction::Capture { label } => {
            backends::baseline::capture(config, &capability_inventory(config), label.as_deref())?
        }
        BaselineAction::Diff { from, to } => {
            backends::baseline::diff(config, &capability_inventory(config), from, to)?
        }
        BaselineAction::List => backends::baseline::list(config)?,
    };
    print!("{report}");
    if !report.ends_with('\n') {
        println!();
    }
    Ok(())
}
