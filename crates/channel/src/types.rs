//! Core channel types — independent of any specific transport.
//!
//! These types describe *what* a delivery is (confirmation request, nudge,
//! report) and *where* it could be delivered (a registered channel handle),
//! without committing to a single router or preference policy.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Kind of user-facing channel a message can be delivered through.
///
/// `Local` covers CLI / native desktop notifications. `Http` covers the
/// outbox polled by web clients. `WebSocket` covers live sessions bound
/// directly to the daemon. `Relay` covers outbound gateways (Slack/Telegram/
/// Discord bridges, custom HTTP webhooks) reached via `bridge::BridgeClient`.
/// `Webhook` covers non-interactive push (one-way notification to a URL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    Local,
    Http,
    WebSocket,
    Relay,
    Webhook,
}

impl ChannelKind {
    /// Whether this transport can carry a user response back to Brain.
    /// Webhook-only channels are one-way and cannot surface an approval reply.
    pub fn supports_response(&self) -> bool {
        matches!(
            self,
            Self::Local | Self::Http | Self::WebSocket | Self::Relay
        )
    }
}

impl std::fmt::Display for ChannelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Local => "local",
            Self::Http => "http",
            Self::WebSocket => "websocket",
            Self::Relay => "relay",
            Self::Webhook => "webhook",
        };
        write!(f, "{s}")
    }
}

/// Urgency level that influences channel selection + timeout semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UrgencyLevel {
    /// Background — nudges, retrospectives, end-of-day summaries.
    Low,
    /// Normal — routine confirmations, reports.
    Normal,
    /// Elevated — time-sensitive approvals (budget breach, destructive).
    High,
    /// Emergency — bypass quiet hours and preference weighting.
    Critical,
}

impl UrgencyLevel {
    /// Whether this urgency should bypass quiet-hours policy.
    pub fn overrides_quiet_hours(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

/// Category of delivery — used to look up channel preferences.
///
/// Preferences are stored per category because users usually want
/// different channels for different content: `Confirm` on Telegram for
/// instant approval, `Nudge` as a desktop toast, `Report` to email.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryCategory {
    /// Approval / confirmation request.
    Confirm,
    /// Proactive nudge (habit reminder, open-loop surfacing).
    Nudge,
    /// Periodic report / summary.
    Report,
    /// Direct user-request response (reply to what they asked).
    Response,
    /// Audit / system alert (budget breach, failed task).
    Alert,
}

impl DeliveryCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Confirm => "confirm",
            Self::Nudge => "nudge",
            Self::Report => "report",
            Self::Response => "response",
            Self::Alert => "alert",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "confirm" => Some(Self::Confirm),
            "nudge" => Some(Self::Nudge),
            "report" => Some(Self::Report),
            "response" => Some(Self::Response),
            "alert" => Some(Self::Alert),
            _ => None,
        }
    }
}

impl std::fmt::Display for DeliveryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A concrete channel registered with the router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelDescriptor {
    /// Stable identifier used for routing (e.g. `"telegram"`, `"slack-alerts"`).
    pub id: String,
    /// Transport kind.
    pub kind: ChannelKind,
    /// Human-readable label shown to the user.
    pub label: String,
    /// Whether the channel is currently reachable (set to false on sustained
    /// delivery failures; the router skips unhealthy channels).
    pub healthy: bool,
    /// Optional tags — e.g. `["work", "desktop"]` for filtering.
    pub tags: Vec<String>,
}

impl ChannelDescriptor {
    pub fn new(id: impl Into<String>, kind: ChannelKind, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            label: label.into(),
            healthy: true,
            tags: Vec::new(),
        }
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// A single outbound delivery request handed to the router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryIntent {
    /// Unique delivery ID (also used for audit correlation).
    pub id: String,
    /// Content body. Approval prompts already include nonce and alternatives.
    pub content: String,
    /// Delivery category (drives preference lookup).
    pub category: DeliveryCategory,
    /// Urgency (drives selection + timeout).
    pub urgency: UrgencyLevel,
    /// Namespace the delivery belongs to (default `"personal"`).
    pub namespace: String,
    /// Optional nonce — populated for `Confirm` intents so the correlator
    /// can match user responses.
    pub nonce: Option<String>,
    /// Optional preferred channel ID from the caller (overrides learned prefs).
    pub preferred_channel: Option<String>,
    /// Channel the user initiated the conversation on — used as fallback.
    pub initiation_channel: Option<String>,
    /// Arbitrary metadata carried through to the transport (e.g. thread IDs).
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (for time-of-day aware routing).
    pub created_at: DateTime<Utc>,
}

impl DeliveryIntent {
    pub fn new(
        content: impl Into<String>,
        category: DeliveryCategory,
        urgency: UrgencyLevel,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            category,
            urgency,
            namespace: "personal".to_string(),
            nonce: None,
            preferred_channel: None,
            initiation_channel: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_nonce(mut self, nonce: impl Into<String>) -> Self {
        self.nonce = Some(nonce.into());
        self
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn with_preferred(mut self, channel_id: impl Into<String>) -> Self {
        self.preferred_channel = Some(channel_id.into());
        self
    }

    pub fn with_initiation(mut self, channel_id: impl Into<String>) -> Self {
        self.initiation_channel = Some(channel_id.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

/// Result of attempting delivery on a single channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryOutcome {
    pub delivery_id: String,
    pub channel_id: String,
    pub success: bool,
    pub attempted_at: DateTime<Utc>,
    pub error: Option<String>,
}

impl DeliveryOutcome {
    pub fn success(delivery_id: impl Into<String>, channel_id: impl Into<String>) -> Self {
        Self {
            delivery_id: delivery_id.into(),
            channel_id: channel_id.into(),
            success: true,
            attempted_at: Utc::now(),
            error: None,
        }
    }

    pub fn failure(
        delivery_id: impl Into<String>,
        channel_id: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            delivery_id: delivery_id.into(),
            channel_id: channel_id.into(),
            success: false,
            attempted_at: Utc::now(),
            error: Some(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_kind_response_support() {
        assert!(ChannelKind::Local.supports_response());
        assert!(ChannelKind::Relay.supports_response());
        assert!(!ChannelKind::Webhook.supports_response());
    }

    #[test]
    fn urgency_quiet_hours_override() {
        assert!(!UrgencyLevel::Low.overrides_quiet_hours());
        assert!(!UrgencyLevel::Normal.overrides_quiet_hours());
        assert!(UrgencyLevel::High.overrides_quiet_hours());
        assert!(UrgencyLevel::Critical.overrides_quiet_hours());
    }

    #[test]
    fn category_roundtrip() {
        for cat in [
            DeliveryCategory::Confirm,
            DeliveryCategory::Nudge,
            DeliveryCategory::Report,
            DeliveryCategory::Response,
            DeliveryCategory::Alert,
        ] {
            let s = cat.as_str();
            assert_eq!(DeliveryCategory::parse(s), Some(cat));
        }
    }

    #[test]
    fn intent_builders() {
        let intent = DeliveryIntent::new(
            "Deploy to prod?",
            DeliveryCategory::Confirm,
            UrgencyLevel::High,
        )
        .with_nonce("abc123")
        .with_preferred("telegram")
        .with_initiation("cli")
        .with_metadata("thread", "42");

        assert_eq!(intent.category, DeliveryCategory::Confirm);
        assert_eq!(intent.nonce.as_deref(), Some("abc123"));
        assert_eq!(intent.preferred_channel.as_deref(), Some("telegram"));
        assert_eq!(intent.initiation_channel.as_deref(), Some("cli"));
        assert_eq!(
            intent.metadata.get("thread").map(String::as_str),
            Some("42")
        );
        assert!(!intent.id.is_empty());
    }
}
