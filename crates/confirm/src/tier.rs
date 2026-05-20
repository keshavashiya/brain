//! Action tier — re-export of the canonical type.
//!
//! Lives in `brain::security` so audit, confirm, sandbox, and
//! orchestrator share one definition. This module is a thin
//! compatibility shim so callers using `confirm::ActionTier` /
//! `confirm::tier::ActionTier` continue to work.

pub use brain::security::ActionTier;
