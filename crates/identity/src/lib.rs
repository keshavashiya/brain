//! # Brain Identity & Principal
//!
//! - [`Principal`] — who is asking (`user_id` + `agent_id` + scopes + tier).
//! - [`Tier`] — ordered authorization level: `Read < Write < Execute < Destructive < External`.
//! - [`IdentityStore`] — async trait that resolves principals and authorizes
//!   actions. Receives an [`AuthorizationRequest`] (not a verb string) so
//!   path-scope checks can read `modifiers["path"]` / `modifiers["cwd"]`
//!   without depending on the higher-level `IntentToken` type.
//! - [`ModifierConstraint`] — per-principal, per-`(verb, modifier)` allowlist;
//!   the general form of the built-in `path_allowlist` and the enforcement
//!   substrate for capability-scoped Skill Packs.
//! - [`ConfigIdentityStore`] — default in-memory implementation backed by
//!   the `identity:` section of `~/.brain/config.yaml`.
//! - [`IdentityKey`] — the install's persistent root secret, from which other
//!   at-rest keys (e.g. the audit-trail cipher) are derived.

pub mod key;
pub mod store;
pub mod types;

pub use key::{IdentityKey, IdentityKeyError, KEY_LEN};
pub use store::{ConfigIdentityStore, IdentityConfig, PrincipalConfig};
pub use types::{
    AgentHint, AgentId, AuthorizationRequest, CheckOutcome, IdentityError, IdentityStore,
    MatchKind, ModifierConstraint, Principal, Tier, UserId,
};
