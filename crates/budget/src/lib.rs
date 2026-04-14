//! # Brain Cost Budget
//!
//! Per-action and rolling caps on LLM tokens, paid-API calls, sandbox wall-clock.
//! Soft-warn at 50% / 80%, hard-stop at 100% requiring explicit re-approval.
//! Budget breach is an auditable event delivered to preferred channel.

pub mod ledger;
pub mod policy;
pub mod warn;

pub use ledger::{
    BudgetDecision, BudgetError, BudgetStatus, CostBudget, ResourceKind, SqliteBudget,
};
pub use policy::BudgetPolicy;
