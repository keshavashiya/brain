//! Timeout and escalation policy.

use serde::{Deserialize, Serialize};

/// What to do when an approval request times out.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EscalationPolicy {
    /// Cancel the action, log the timeout.
    Abort,
    /// Alert user, then abort.
    NotifyAndAbort,
    /// Put the action in pending queue, retry later.
    Defer,
    /// Auto-approve — only for tier-lowered actions the user routinely approves.
    AutoApprove,
}

impl std::fmt::Display for EscalationPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EscalationPolicy::Abort => write!(f, "abort"),
            EscalationPolicy::NotifyAndAbort => write!(f, "notify_and_abort"),
            EscalationPolicy::Defer => write!(f, "defer"),
            EscalationPolicy::AutoApprove => write!(f, "auto_approve"),
        }
    }
}
