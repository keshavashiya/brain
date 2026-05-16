//! # Brain Identity & Principal
//!
//! - [`Principal`] — who is asking (`user_id` + `agent_id` + scopes + tier).
//! - [`Tier`] — ordered authorization level: `Read < Write < Execute < Destructive < External`.
//! - [`IdentityStore`] — async trait that resolves principals and authorizes
//!   actions. Receives an [`AuthorizationRequest`] (not a verb string) so
//!   path-scope checks can read `modifiers["path"]` / `modifiers["cwd"]`
//!   without depending on the higher-level `IntentToken` type.
//! - [`ConfigIdentityStore`] — default in-memory implementation backed by
//!   the `identity:` section of `~/.brain/config.yaml`.

pub mod store;
pub mod types;

pub use store::{ConfigIdentityStore, IdentityConfig, PrincipalConfig};
pub use types::{
    AgentHint, AgentId, AuthorizationRequest, CheckOutcome, IdentityError, IdentityStore,
    Principal, Tier, UserId,
};
