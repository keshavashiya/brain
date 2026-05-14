//! # Brain Identity & Principal
//!
//! Phase 1 of the v1.0.0 plan (`docs/v1.0.0.md` §7). Implements:
//!
//! - [`Principal`] — who is asking (`user_id` + `agent_id` + scopes + tier).
//! - [`Tier`] — ordered authorization level: `Read < Write < Execute < Destructive < External`.
//! - [`IdentityStore`] — async trait that resolves principals and authorizes
//!   actions. Receives an [`AuthorizationRequest`] (not a verb string) so
//!   path-scope checks can read `modifiers["path"]` / `modifiers["cwd"]`
//!   without depending on later-phase types (`IntentToken` is Phase 3).
//! - [`ConfigIdentityStore`] — default in-memory implementation backed by
//!   the `identity:` section of `~/.brain/config.yaml`.
//!
//! Threading into `Signal` and adapter resolution happens in Tier B/C of the
//! Phase 1 PR sequence.

pub mod store;
pub mod types;

pub use store::{ConfigIdentityStore, IdentityConfig, PrincipalConfig};
pub use types::{
    AgentHint, AgentId, AuthorizationRequest, CheckOutcome, IdentityError, IdentityStore,
    Principal, Tier, UserId,
};
