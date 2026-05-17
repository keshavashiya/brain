//! # Brain Channel Intelligence
//!
//! Routes confirmations, nudges, and reports to the right user-facing channel,
//! learns preferences from behavior, and correlates user responses back to
//! pending approvals across channels.
//!
//! ## Layers
//! - [`types`]   — `ChannelKind`, `UrgencyLevel`, `DeliveryIntent`, `DeliveryOutcome`,
//!   `ChannelDescriptor`, `DeliveryCategory`.
//! - [`error`]   — `ChannelError`.
//! - [`preference`] — `ChannelPreferenceStore` (persisted learned preferences).
//! - [`router`]  — `ChannelRouter` trait + `DefaultChannelRouter` (selection policy).
//! - [`correlate`] — `ConfirmationCorrelator` (inbound nonce + approve/reject parsing).
//! - [`relay`]   — `RelayAdapter` wrapping `bridge::BridgeClient` for outbound WS gateways.
//!
//! All components are opt-in. They wire into `SignalProcessor` via builder
//! methods and compose with existing `NotificationRouter` webhook tiers.

pub mod correlate;
pub mod dispatch;
pub mod error;
pub mod preference;
pub mod relay;
pub mod router;
pub mod transport;
pub mod types;

// `CorrelatedCommand` is intentionally not re-exported — it's an
// internal step inside `ConfirmationCorrelator::process`; callers see
// `CorrelationOutcome` only (audit Issue 33).
pub use correlate::{ConfirmationCorrelator, CorrelationOutcome};
pub use dispatch::{ChannelDispatcher, DeliveryReceipt};
pub use error::ChannelError;
pub use preference::{
    ChannelPreference, ChannelPreferenceStore, RecordedInteraction, SqlitePreferenceStore,
};
pub use relay::{RelayAdapter, RelayConfig};
// `RoutingDecision` is the return type of the public `ChannelRouter`
// trait so it must stay `pub` in `router.rs`, but no external caller
// destructures it — drop the top-level re-export so the surface
// `channel::*` advertises (audit Issue 34) shrinks. Inside the crate
// it's still reachable via `channel::router::RoutingDecision`.
pub use router::{ChannelRouter, DefaultChannelRouter, RoutingContext};
pub use transport::{ChannelTransport, InboundMessage, MessageHandle, TransportHealth};
pub use types::{
    ChannelDescriptor, ChannelKind, DeliveryCategory, DeliveryIntent, DeliveryOutcome, UrgencyLevel,
};
