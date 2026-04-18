//! # Brain Channel Intelligence
//!
//! Phase 4 foundation — routes confirmations, nudges, and reports to the
//! right user-facing channel, learns preferences from behavior, and
//! correlates user responses back to pending approvals across channels.
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
pub mod error;
pub mod preference;
pub mod relay;
pub mod router;
pub mod transport;
pub mod types;

pub use correlate::{ConfirmationCorrelator, CorrelatedCommand, CorrelationOutcome};
pub use error::ChannelError;
pub use preference::{
    ChannelPreference, ChannelPreferenceStore, RecordedInteraction, SqlitePreferenceStore,
};
pub use relay::{RelayAdapter, RelayConfig};
pub use router::{ChannelRouter, DefaultChannelRouter, RoutingContext, RoutingDecision};
pub use transport::{ChannelTransport, InboundMessage, MessageHandle, TransportHealth};
pub use types::{
    ChannelDescriptor, ChannelKind, DeliveryCategory, DeliveryIntent, DeliveryOutcome, UrgencyLevel,
};
