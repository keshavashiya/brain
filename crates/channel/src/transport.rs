//! Channel transport trait — the common abstraction for anything that can
//! deliver a [`DeliveryIntent`] out and surface user replies back in.
//!
//! `RelayAdapter` (BridgeClient gateway), `HttpPolledTransport` (Telegram-
//! style long polling), `WebhookInboundTransport` (Discord/Slack/GitHub
//! HTTP callbacks), and `WebhookOutboundTransport` (one-way push) all
//! implement this trait so bootstrap code can treat them uniformly.

pub mod http_polled;
pub mod jsonpath;
pub mod outbound;
pub mod preset;
pub mod preset_loader;
pub mod webhook_inbound;
pub mod webhook_outbound;

pub use http_polled::{HttpPolledConfig, HttpPolledTransport};
pub use preset::{
    CursorTransform, FieldExtractors, HttpMethod, PollSpec, PresetDefinition, PresetKind, SendSpec,
    VerifierSpec, WebhookInboundSpec,
};
pub use webhook_inbound::{WebhookInboundConfig, WebhookInboundTransport, WebhookResponse};
pub use webhook_outbound::{WebhookOutboundConfig, WebhookOutboundTransport};

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::error::ChannelError;
use crate::types::{ChannelDescriptor, DeliveryIntent};

/// A transport-agnostic inbound message — what a user replied with on any
/// channel. Unlike `bridge::BridgeMessage` (WS-frame specific) this is the
/// normalised shape the correlator and signal pipeline consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    /// Unique id for this inbound — used for correlation if the transport
    /// expects a reply (HTTP webhook returns, relay BridgeMessage reply).
    pub id: String,
    /// Raw text content.
    pub content: String,
    /// Channel id this arrived on (matches a registered
    /// [`ChannelDescriptor::id`]).
    pub channel_id: String,
    /// Optional platform-side user identifier (Telegram user id, Discord
    /// member id, Slack user id). Preference learning uses it to namespace
    /// weights per-user when multiple users share a channel.
    pub user_ref: Option<String>,
    /// Optional platform-side thread/chat identifier — used by outbound
    /// replies to target the same conversation (Telegram chat id, Discord
    /// channel id, Slack thread_ts).
    pub reply_to: Option<String>,
    /// Arbitrary extra fields extracted by the transport (JSONPath matches
    /// the preset couldn't map onto a first-class field).
    pub extra: HashMap<String, String>,
    /// When the transport observed this message.
    pub received_at: DateTime<Utc>,
}

impl InboundMessage {
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        channel_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            channel_id: channel_id.into(),
            user_ref: None,
            reply_to: None,
            extra: HashMap::new(),
            received_at: Utc::now(),
        }
    }
}

/// Opaque handle returned on successful outbound delivery — lets the caller
/// correlate delivery receipts, edit sent messages, etc. Transports that
/// can't provide a platform-side id fall back to the intent id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageHandle {
    pub delivery_id: String,
    pub platform_id: Option<String>,
}

impl MessageHandle {
    pub fn new(delivery_id: impl Into<String>) -> Self {
        Self {
            delivery_id: delivery_id.into(),
            platform_id: None,
        }
    }

    pub fn with_platform_id(mut self, id: impl Into<String>) -> Self {
        self.platform_id = Some(id.into());
        self
    }
}

/// Health signal a transport reports back so the router can skip it when
/// the remote side is misbehaving.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum TransportHealth {
    /// Last operation succeeded; transport is reachable.
    Healthy,
    /// Transient failures observed but retries still in progress.
    Degraded { reason: String },
    /// Hard failure — auth broken, 401/403, repeated 5xx past budget.
    Down { reason: String },
}

impl TransportHealth {
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }
}

/// Common interface every concrete channel transport implements.
///
/// `descriptor()` is cheap and sync so the router can register without
/// waiting on a connection. `send()` pushes outbound; `inbound()` hands
/// back a fresh `broadcast::Receiver` — the caller (usually the bootstrap
/// pipeline handler) runs the correlator + signal fallback on its own
/// task. `health()` is advisory.
#[async_trait]
pub trait ChannelTransport: Send + Sync {
    /// Channel descriptor for router registration. Must match the `id`
    /// the bootstrap code expects to address this transport by.
    fn descriptor(&self) -> ChannelDescriptor;

    /// Push an outbound intent. Implementations resolve the target (chat
    /// id, webhook URL, etc.) from `intent.metadata`/`reply_to` /
    /// configured defaults and perform the send.
    async fn send(&self, intent: &DeliveryIntent) -> Result<MessageHandle, ChannelError>;

    /// Subscribe to inbound messages. Transports that are outbound-only
    /// (pure webhook push) may return a receiver that never yields.
    fn inbound(&self) -> broadcast::Receiver<InboundMessage>;

    /// Current transport health — used by router to deprioritise or skip.
    async fn health(&self) -> TransportHealth;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inbound_builder_defaults_received_at() {
        let m = InboundMessage::new("id-1", "hello", "telegram-personal");
        assert_eq!(m.id, "id-1");
        assert_eq!(m.channel_id, "telegram-personal");
        assert!(m.user_ref.is_none());
        assert!(m.reply_to.is_none());
        assert!(m.extra.is_empty());
    }

    #[test]
    fn message_handle_platform_id_optional() {
        let h = MessageHandle::new("d-1");
        assert_eq!(h.delivery_id, "d-1");
        assert!(h.platform_id.is_none());
        let h2 = MessageHandle::new("d-2").with_platform_id("tg-42");
        assert_eq!(h2.platform_id.as_deref(), Some("tg-42"));
    }

    #[test]
    fn health_helpers() {
        assert!(TransportHealth::Healthy.is_healthy());
        assert!(!TransportHealth::Degraded {
            reason: "retry".into()
        }
        .is_healthy());
        assert!(!TransportHealth::Down {
            reason: "401".into()
        }
        .is_healthy());
    }
}
