//! Action tier enum — shared with audit crate.

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

impl ActionTier {
    /// Whether this tier requires explicit confirmation.
    pub fn requires_confirmation(self) -> bool {
        matches!(self, ActionTier::Destructive | ActionTier::External)
    }

    /// Default timeout for this tier.
    pub fn default_timeout(self) -> std::time::Duration {
        match self {
            ActionTier::Read => std::time::Duration::from_secs(30),
            ActionTier::Write => std::time::Duration::from_secs(60),
            ActionTier::Execute => std::time::Duration::from_secs(120),
            ActionTier::Destructive => std::time::Duration::from_secs(300),
            ActionTier::External => std::time::Duration::from_secs(300),
        }
    }
}
