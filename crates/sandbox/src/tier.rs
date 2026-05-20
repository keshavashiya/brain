//! Action tier — re-export of the canonical type.
//!
//! Lives in `brain::security` so audit, confirm, sandbox, and
//! orchestrator share one definition. This module exists only to keep
//! existing call sites that import via `sandbox::tier::ActionTier`
//! working without churn.

pub use brain::security::ActionTier;
