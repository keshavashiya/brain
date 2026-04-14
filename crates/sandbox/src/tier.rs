//! Action tier enum for sandbox executor.

use serde::{Deserialize, Serialize};

/// Action tier determines confirmation requirement.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionTier {
    /// Read-only — never requires confirmation
    Read,
    /// Write — implicit confirmation, user can undo
    Write,
    /// Execute — sandboxed, reversible
    Execute,
    /// Destructive — explicit approval required
    Destructive,
    /// External — explicit approval + credential audit
    External,
}

impl std::fmt::Display for ActionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionTier::Read => write!(f, "read"),
            ActionTier::Write => write!(f, "write"),
            ActionTier::Execute => write!(f, "execute"),
            ActionTier::Destructive => write!(f, "destructive"),
            ActionTier::External => write!(f, "external"),
        }
    }
}
