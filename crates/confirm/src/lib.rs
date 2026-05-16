//! # Brain Confirmation Engine
//!
//! Human approval gates with nonce-based workflows, timeout escalation,
//! and tiered confirmation (read/write/execute/destructive/external).

pub mod nonce;
pub mod notifier;
pub mod standing;
pub mod tier;
pub mod timeout;

pub use nonce::{
    ApprovalDecision, ApprovalOutcome, ApprovalSpec, ApprovalStatus, ConfirmError,
    ConfirmationEngine, SqliteConfirmationEngine,
};
pub use notifier::ApprovalNotifier;
pub use standing::{GrantKey, SqliteStandingApprovals, StandingApproval, StandingApprovalStore};
pub use tier::ActionTier;
pub use timeout::EscalationPolicy;
