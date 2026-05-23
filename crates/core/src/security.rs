//! Cross-cutting security primitives shared by audit, confirm, sandbox,
//! and orchestrator.
//!
//! `ActionTier` is an alias for [`identity::Tier`]: the same enum names
//! both "the tier a principal holds" and "the tier an action requires",
//! and the authorization check is a single `principal.tier >= action.tier`
//! comparison. Keeping them as one type removes the manual `convert_tier`
//! shims the codebase used to carry. The `ActionTier` name is preserved
//! at this re-export site so action-side call sites read naturally.

pub use identity::Tier as ActionTier;
