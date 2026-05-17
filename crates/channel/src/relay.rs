//! Outbound relay — glue between `bridge::BridgeClient` and channel semantics.
//!
//! A `RelayAdapter` wraps a single remote gateway (Slack bot, Telegram bridge,
//! custom HTTP agent) and makes it look like a first-class Brain channel:
//!
//! - **Inbound** messages pass through [`ConfirmationCorrelator`] first so
//!   cross-channel approvals (`approve <nonce>` typed on Telegram resolving
//!   a prompt issued via CLI) work without per-platform glue. Non-correlation
//!   messages are handed to a caller-provided [`SignalHandler`] fallback —
//!   typically the signal pipeline's request handler — so relays double as
//!   plain chat frontends.
//! - **Outbound** [`DeliveryIntent`]s are translated to [`BridgeMessage`]
//!   and pushed as proactive frames over the same WebSocket via the bridge
//!   client's proactive channel.
//! - Every observation (response in, delivery-failure out) updates
//!   [`ChannelPreferenceStore`] so the router learns which gateway the user
//!   actually reads.
//!
//! ## Lifecycle
//! 1. Construct with config + router + correlator + prefs + fallback handler.
//! 2. Call [`RelayAdapter::register_channel`] once to publish the descriptor.
//! 3. Spawn [`RelayAdapter::run`] on a tokio task — it connects and blocks.
//! 4. Deliver intents via [`RelayAdapter::deliver`]; the inbound path handles
//!    user replies independently.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bridge::{BridgeClient, BridgeConfig, BridgeMessage};
use chrono::Utc;
use tokio::sync::broadcast;

use crate::correlate::{ConfirmationCorrelator, CorrelationOutcome};
use crate::error::ChannelError;
use crate::preference::{ChannelPreferenceStore, RecordedInteraction};
use crate::router::ChannelRouter;
use crate::transport::{ChannelTransport, InboundMessage, MessageHandle, TransportHealth};
use crate::types::{
    ChannelDescriptor, ChannelKind, DeliveryCategory, DeliveryIntent, DeliveryOutcome,
};

/// Buffer size for outbound proactive frames. A slow bridge receiver will
/// lag past this boundary and the `broadcast` channel drops oldest — fine
/// for relay load where the router tries alternatives on failure anyway.
const DEFAULT_OUTBOUND_CAPACITY: usize = 128;

/// Configuration for a single relay gateway.
#[derive(Debug, Clone)]
pub struct RelayConfig {
    /// Stable channel id registered with the router (e.g. `"chat-main"`).
    pub channel_id: String,
    /// Human-readable label used for CLI display and audit entries.
    pub label: String,
    /// WebSocket URL of the gateway.
    pub url: String,
    /// Namespace used when recording interactions (default `"personal"`).
    pub namespace: String,
    /// Reconnection tuning for the underlying `BridgeClient`.
    pub bridge: BridgeConfig,
}

impl RelayConfig {
    pub fn new(
        channel_id: impl Into<String>,
        label: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            channel_id: channel_id.into(),
            label: label.into(),
            url: url.into(),
            namespace: "personal".to_string(),
            bridge: BridgeConfig::default(),
        }
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn with_bridge(mut self, cfg: BridgeConfig) -> Self {
        self.bridge = cfg;
        self
    }
}

/// Handler for non-correlation inbound messages.
///
/// When a relay message isn't an approve/reject command with a known nonce,
/// the relay hands it off here for normal signal processing. The returned
/// string is sent back to the gateway as the reply body.
#[async_trait]
pub trait SignalHandler: Send + Sync {
    async fn handle(&self, msg: &BridgeMessage) -> String;
}

/// Default `SignalHandler` that acknowledges receipt — useful when no
/// higher-level pipeline is wired yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct AckSignalHandler;

#[async_trait]
impl SignalHandler for AckSignalHandler {
    async fn handle(&self, _msg: &BridgeMessage) -> String {
        "Received.".to_string()
    }
}

/// Relay adapter — one gateway, one WebSocket, bidirectional.
pub struct RelayAdapter {
    config: RelayConfig,
    router: Arc<dyn ChannelRouter>,
    correlator: Arc<ConfirmationCorrelator>,
    preferences: Arc<dyn ChannelPreferenceStore>,
    fallback: Arc<dyn SignalHandler>,
    outbound_tx: broadcast::Sender<BridgeMessage>,
    /// Raw inbound fan-out for `ChannelTransport::inbound()` subscribers.
    /// Published to before the built-in correlator/fallback path runs, so
    /// new code using the generic transport trait can observe the same
    /// stream without disturbing legacy behaviour.
    inbound_tx: broadcast::Sender<InboundMessage>,
}

impl RelayAdapter {
    /// Create a new adapter. Doesn't open a connection — call [`run`] for
    /// that. Doesn't register with the router either — call
    /// [`register_channel`] or register externally.
    pub fn new(
        config: RelayConfig,
        router: Arc<dyn ChannelRouter>,
        correlator: Arc<ConfirmationCorrelator>,
        preferences: Arc<dyn ChannelPreferenceStore>,
        fallback: Arc<dyn SignalHandler>,
    ) -> Self {
        let (tx, _rx) = broadcast::channel(DEFAULT_OUTBOUND_CAPACITY);
        let (inbound_tx, _) = broadcast::channel(DEFAULT_OUTBOUND_CAPACITY);
        Self {
            config,
            router,
            correlator,
            preferences,
            fallback,
            outbound_tx: tx,
            inbound_tx,
        }
    }

    /// The channel id this adapter publishes under (matches the
    /// [`ChannelDescriptor`] registered with the router).
    pub fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    /// Register this adapter's channel descriptor with the router so
    /// [`DeliveryIntent`]s can be routed to it. Safe to call repeatedly —
    /// re-registration refreshes the descriptor.
    pub async fn register_channel(&self) -> Result<(), ChannelError> {
        let desc = ChannelDescriptor::new(
            &self.config.channel_id,
            ChannelKind::Relay,
            &self.config.label,
        );
        self.router.register(desc).await
    }

    /// Clone of the outbound broadcast sender — useful for tests and admin
    /// tools that need to inject frames independently of `deliver`.
    pub fn outbound_sender(&self) -> broadcast::Sender<BridgeMessage> {
        self.outbound_tx.clone()
    }

    /// Start the bridge loop. Blocks until the bridge exhausts its retry
    /// budget — callers typically run this on a dedicated task.
    pub async fn run(self: Arc<Self>) -> Result<(), ChannelError> {
        let client = BridgeClient::new(&self.config.url, self.config.bridge.clone());
        let proactive_rx = self.outbound_tx.subscribe();
        let me = self.clone();
        client
            .connect_and_relay_bidirectional(
                move |msg| {
                    let me = me.clone();
                    async move { me.handle_inbound(msg).await }
                },
                Some(proactive_rx),
            )
            .await
            .map_err(|e| ChannelError::Relay(e.to_string()))
    }

    /// Process one inbound message and build the reply. Exposed for testing
    /// without a WebSocket and for direct invocation by non-bridge callers.
    pub async fn handle_inbound(&self, msg: BridgeMessage) -> BridgeMessage {
        let now = Utc::now();

        // Fan out the raw inbound to any `ChannelTransport::inbound()`
        // subscriber before we touch correlator/fallback. Send-errors here
        // mean no subscriber — fine, legacy path still runs below.
        let _ = self
            .inbound_tx
            .send(bridge_to_inbound(&msg, &self.config.channel_id));

        let outcome = self.correlator.process(&msg.content).await;

        let reply_text = match &outcome {
            Ok(CorrelationOutcome::NoMatch) => self.fallback.handle(&msg).await,
            Ok(CorrelationOutcome::Applied {
                approved, reason, ..
            }) => match (*approved, reason) {
                (true, _) => "Approved.".to_string(),
                (false, Some(r)) => format!("Rejected: {r}"),
                (false, None) => "Rejected.".to_string(),
            },
            Ok(CorrelationOutcome::AlreadyResolved { .. }) => {
                "This approval has already been resolved.".to_string()
            }
            Ok(CorrelationOutcome::UnknownNonce { nonce }) => {
                format!("Unknown approval nonce: {nonce}")
            }
            Ok(CorrelationOutcome::EngineError { nonce, error }) => {
                format!("Failed to apply approval {nonce}: {error}")
            }
            Err(e) => format!("Correlation error: {e}"),
        };

        // Confirm category for anything that looked like an approval command
        // (even unknown/already-resolved) so the router learns the user replies
        // on this channel; plain chat counts as Response.
        let category = match &outcome {
            Ok(CorrelationOutcome::Applied { .. })
            | Ok(CorrelationOutcome::AlreadyResolved { .. })
            | Ok(CorrelationOutcome::UnknownNonce { .. })
            | Ok(CorrelationOutcome::EngineError { .. }) => DeliveryCategory::Confirm,
            _ => DeliveryCategory::Response,
        };

        if let Err(e) = self
            .preferences
            .record_interaction(RecordedInteraction {
                namespace: self.config.namespace.clone(),
                category,
                channel_id: self.config.channel_id.clone(),
                delivered_at: now,
                responded_at: Some(now),
                delivered_ok: true,
            })
            .await
        {
            tracing::warn!(
                error = %e,
                channel = %self.config.channel_id,
                "Failed to record inbound interaction"
            );
        }

        BridgeMessage::reply(&msg, reply_text)
    }

    /// Push a [`DeliveryIntent`] out through the relay as a [`BridgeMessage`].
    ///
    /// Returns a [`DeliveryOutcome`] tagged with this adapter's channel id.
    /// A failure here means the outbound broadcast had no live subscriber —
    /// i.e. [`run`] isn't connected. The preference store is updated with
    /// the failure signal so the router decays this channel.
    pub async fn deliver(&self, intent: &DeliveryIntent) -> DeliveryOutcome {
        let now = Utc::now();

        // Build metadata: adapter defaults first, caller-provided keys last
        // so the caller can override anything (including `nonce`).
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("category".to_string(), intent.category.to_string());
        meta.insert(
            "urgency".to_string(),
            format!("{:?}", intent.urgency).to_lowercase(),
        );
        if let Some(n) = &intent.nonce {
            meta.insert("nonce".to_string(), n.clone());
        }
        for (k, v) in &intent.metadata {
            meta.insert(k.clone(), v.clone());
        }

        let bridge_msg = BridgeMessage {
            id: intent.id.clone(),
            content: intent.content.clone(),
            source: Some(self.config.channel_id.clone()),
            metadata: Some(meta),
        };

        match self.outbound_tx.send(bridge_msg) {
            Ok(_) => DeliveryOutcome::success(intent.id.clone(), self.config.channel_id.clone()),
            Err(e) => {
                if let Err(err) = self
                    .preferences
                    .record_interaction(RecordedInteraction {
                        namespace: intent.namespace.clone(),
                        category: intent.category,
                        channel_id: self.config.channel_id.clone(),
                        delivered_at: now,
                        responded_at: None,
                        delivered_ok: false,
                    })
                    .await
                {
                    tracing::warn!(
                        error = %err,
                        "Failed to record delivery failure interaction"
                    );
                }
                DeliveryOutcome::failure(
                    intent.id.clone(),
                    self.config.channel_id.clone(),
                    format!("no active bridge subscriber: {e}"),
                )
            }
        }
    }
}

fn bridge_to_inbound(msg: &BridgeMessage, channel_id: &str) -> InboundMessage {
    let mut inbound = InboundMessage::new(msg.id.clone(), msg.content.clone(), channel_id);
    if let Some(meta) = &msg.metadata {
        inbound.user_ref = meta.get("user_id").cloned();
        inbound.reply_to = meta.get("reply_to").cloned();
        for (k, v) in meta {
            if k != "user_id" && k != "reply_to" {
                inbound.extra.insert(k.clone(), v.clone());
            }
        }
    }
    inbound
}

#[async_trait]
impl ChannelTransport for RelayAdapter {
    fn descriptor(&self) -> ChannelDescriptor {
        ChannelDescriptor::new(
            &self.config.channel_id,
            ChannelKind::Relay,
            &self.config.label,
        )
    }

    async fn send(&self, intent: &DeliveryIntent) -> Result<MessageHandle, ChannelError> {
        let outcome = self.deliver(intent).await;
        if outcome.success {
            Ok(MessageHandle::new(outcome.delivery_id))
        } else {
            Err(ChannelError::Relay(
                outcome
                    .error
                    .unwrap_or_else(|| "relay deliver failed".into()),
            ))
        }
    }

    fn inbound(&self) -> broadcast::Receiver<InboundMessage> {
        self.inbound_tx.subscribe()
    }

    async fn health(&self) -> TransportHealth {
        // Best-effort: if the outbound broadcast has no subscribers the
        // bridge loop isn't connected.
        if self.outbound_tx.receiver_count() == 0 {
            TransportHealth::Down {
                reason: "bridge not connected".into(),
            }
        } else {
            TransportHealth::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preference::SqlitePreferenceStore;
    use crate::router::DefaultChannelRouter;
    use crate::types::UrgencyLevel;
    use async_trait::async_trait;
    use confirm::{
        ApprovalDecision, ApprovalOutcome, ApprovalSpec, ApprovalStatus, ConfirmError,
        ConfirmationEngine,
    };
    use std::sync::Mutex;

    const NONCE: &str = "550e8400-e29b-41d4-a716-446655440000";

    #[derive(Default)]
    struct MockEngine {
        responded: Mutex<Vec<(String, ApprovalDecision)>>,
    }

    #[async_trait]
    impl ConfirmationEngine for MockEngine {
        async fn request(&self, _spec: ApprovalSpec) -> Result<ApprovalOutcome, ConfirmError> {
            unimplemented!()
        }
        async fn respond(
            &self,
            nonce: &str,
            decision: ApprovalDecision,
        ) -> Result<(), ConfirmError> {
            self.responded
                .lock()
                .unwrap()
                .push((nonce.into(), decision));
            Ok(())
        }
        async fn status(&self, _nonce: &str) -> Result<ApprovalStatus, ConfirmError> {
            Ok(ApprovalStatus::Pending { since: Utc::now() })
        }
        async fn pending(&self) -> Result<Vec<ApprovalSpec>, ConfirmError> {
            Ok(vec![])
        }
    }

    struct RecordingFallback {
        calls: Mutex<Vec<String>>,
    }

    impl RecordingFallback {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
            })
        }
        fn count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl SignalHandler for RecordingFallback {
        async fn handle(&self, msg: &BridgeMessage) -> String {
            self.calls.lock().unwrap().push(msg.content.clone());
            format!("fallback:{}", msg.content)
        }
    }

    async fn mk_adapter() -> (
        Arc<RelayAdapter>,
        Arc<DefaultChannelRouter>,
        Arc<SqlitePreferenceStore>,
        Arc<RecordingFallback>,
    ) {
        let db = storage::SqlitePool::open_memory().unwrap();
        let prefs = Arc::new(SqlitePreferenceStore::new(db));
        prefs.ensure_tables().unwrap();

        let router = Arc::new(DefaultChannelRouter::new(
            prefs.clone() as Arc<dyn ChannelPreferenceStore>
        ));

        let engine: Arc<dyn ConfirmationEngine> = Arc::new(MockEngine::default());
        let correlator = Arc::new(ConfirmationCorrelator::new(engine));

        let fallback = RecordingFallback::new();

        let cfg = RelayConfig::new("test-relay", "Test Relay", "ws://127.0.0.1:0/unused");
        let adapter = Arc::new(RelayAdapter::new(
            cfg,
            router.clone() as Arc<dyn ChannelRouter>,
            correlator,
            prefs.clone() as Arc<dyn ChannelPreferenceStore>,
            fallback.clone() as Arc<dyn SignalHandler>,
        ));

        (adapter, router, prefs, fallback)
    }

    #[tokio::test]
    async fn register_channel_adds_to_router() {
        let (adapter, router, _, _) = mk_adapter().await;
        adapter.register_channel().await.unwrap();
        let channels = router.list_channels().await.unwrap();
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].id, "test-relay");
        assert_eq!(channels[0].kind, ChannelKind::Relay);
        assert!(channels[0].healthy);
    }

    #[tokio::test]
    async fn inbound_no_match_invokes_fallback() {
        let (adapter, _, prefs, fallback) = mk_adapter().await;
        let msg = BridgeMessage::new("hey what's up");
        let reply = adapter.handle_inbound(msg).await;
        assert_eq!(reply.content, "fallback:hey what's up");
        assert_eq!(fallback.count(), 1);

        let recorded = prefs.list_all("personal").await.unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].category, DeliveryCategory::Response);
        assert_eq!(recorded[0].channel_id, "test-relay");
        assert_eq!(recorded[0].response_count, 1);
        assert_eq!(recorded[0].success_count, 1);
    }

    #[tokio::test]
    async fn inbound_approval_applies_and_records_confirm() {
        let (adapter, _, prefs, fallback) = mk_adapter().await;
        let msg = BridgeMessage::new(format!("approve {NONCE}"));
        let original_id = msg.id.clone();
        let reply = adapter.handle_inbound(msg).await;
        assert_eq!(reply.content, "Approved.");
        assert_eq!(reply.id, original_id, "reply should reuse message id");
        assert_eq!(fallback.count(), 0);

        let recorded = prefs.list_all("personal").await.unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].category, DeliveryCategory::Confirm);
    }

    #[tokio::test]
    async fn inbound_rejection_with_reason_formatted() {
        let (adapter, _, _, _) = mk_adapter().await;
        let msg = BridgeMessage::new(format!("reject {NONCE} budget exceeded"));
        let reply = adapter.handle_inbound(msg).await;
        assert!(reply.content.starts_with("Rejected: "));
        assert!(reply.content.contains("budget exceeded"));
    }

    #[tokio::test]
    async fn deliver_without_subscriber_records_failure() {
        let (adapter, _, prefs, _) = mk_adapter().await;
        let intent = DeliveryIntent::new("wake up", DeliveryCategory::Nudge, UrgencyLevel::Normal);
        let outcome = adapter.deliver(&intent).await;
        assert!(!outcome.success);
        assert!(outcome
            .error
            .as_deref()
            .unwrap_or("")
            .contains("no active bridge subscriber"));

        let recorded = prefs.list_all("personal").await.unwrap();
        assert_eq!(recorded.len(), 1);
        // Delivery failure seeds weight at 0.0 (SIGNAL_DELIVERY_FAIL).
        assert!(recorded[0].weight.abs() < 1e-6);
    }

    #[tokio::test]
    async fn deliver_with_subscriber_pushes_frame() {
        let (adapter, _, prefs, _) = mk_adapter().await;
        let mut rx = adapter.outbound_sender().subscribe();

        let intent = DeliveryIntent::new("approve?", DeliveryCategory::Confirm, UrgencyLevel::High)
            .with_nonce(NONCE)
            .with_metadata("thread", "chan-42");

        let outcome = adapter.deliver(&intent).await;
        assert!(outcome.success);

        let pushed = rx.try_recv().unwrap();
        assert_eq!(pushed.content, "approve?");
        assert_eq!(pushed.source.as_deref(), Some("test-relay"));
        let meta = pushed.metadata.expect("metadata present");
        assert_eq!(meta.get("nonce").map(String::as_str), Some(NONCE));
        assert_eq!(meta.get("category").map(String::as_str), Some("confirm"));
        assert_eq!(meta.get("urgency").map(String::as_str), Some("high"));
        assert_eq!(meta.get("thread").map(String::as_str), Some("chan-42"));

        // No interaction rows should be recorded for a successful outbound
        // dispatch — we only record when we observe a response or a failure.
        let recorded = prefs.list_all("personal").await.unwrap();
        assert!(recorded.is_empty());
    }

    #[tokio::test]
    async fn ack_signal_handler_returns_received() {
        let h = AckSignalHandler;
        let reply = h.handle(&BridgeMessage::new("hi")).await;
        assert_eq!(reply, "Received.");
    }

    #[test]
    fn relay_config_builders() {
        let cfg = RelayConfig::new("ch", "Channel", "ws://example/").with_namespace("work");
        assert_eq!(cfg.namespace, "work");
    }
}
