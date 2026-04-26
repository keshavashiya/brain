//! Cross-cutting security primitives shared by audit, confirm, sandbox,
//! and orchestrator. Lives here to keep the type single-sourced — every
//! consumer crate already depends on `brain_core`, so promoting these
//! to the leaf avoids the previous three-way duplication and the manual
//! `convert_tier()` shims that came with it.

use serde::{Deserialize, Serialize};

/// Action tier determines confirmation requirement and timeout policy.
///
/// Mirrors VISION §6 (human in the loop): each tier maps to a distinct
/// approval ceremony. `Read`/`Write`/`Execute` are auto-approved (with
/// audit recording); `Destructive` and `External` block on explicit
/// human approval.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ActionTier {
    /// Read-only — never requires confirmation.
    Read,
    /// Write — implicit confirmation, user can undo.
    Write,
    /// Execute — sandboxed, reversible.
    Execute,
    /// Destructive — explicit approval required.
    Destructive,
    /// External — explicit approval + credential audit.
    External,
}

impl ActionTier {
    /// Whether this tier requires explicit confirmation before execution.
    pub fn requires_confirmation(self) -> bool {
        matches!(self, ActionTier::Destructive | ActionTier::External)
    }

    /// Default approval timeout for this tier.
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

impl std::fmt::Display for ActionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ActionTier::Read => "read",
            ActionTier::Write => "write",
            ActionTier::Execute => "execute",
            ActionTier::Destructive => "destructive",
            ActionTier::External => "external",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_confirmation_only_destructive_external() {
        assert!(!ActionTier::Read.requires_confirmation());
        assert!(!ActionTier::Write.requires_confirmation());
        assert!(!ActionTier::Execute.requires_confirmation());
        assert!(ActionTier::Destructive.requires_confirmation());
        assert!(ActionTier::External.requires_confirmation());
    }

    #[test]
    fn default_timeout_increases_with_tier() {
        assert!(ActionTier::Read.default_timeout() < ActionTier::Execute.default_timeout());
        assert!(ActionTier::Execute.default_timeout() < ActionTier::Destructive.default_timeout());
    }

    #[test]
    fn display_matches_serde_repr() {
        assert_eq!(ActionTier::Destructive.to_string(), "destructive");
        let json = serde_json::to_string(&ActionTier::Destructive).unwrap();
        assert_eq!(json, "\"destructive\"");
    }
}
