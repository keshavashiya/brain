//! Channel transport backed by an active WebSocket chat session.
//!
//! Every authenticated WS connection registers one of these with the
//! `ChannelDispatcher` so confirmation prompts (and any other outbound
//! `DeliveryIntent`) can reach the user on the same socket they're
//! chatting on. The transport doesn't own the socket directly — it pushes
//! intents into a bounded mpsc that the connection's writer task drains
//! into JSON frames.
//!
//! Inbound is intentionally a no-op: chat content already flows through
//! `process_text_frame` into the signal pipeline, where the existing
//! `RespondToApproval` regex handles `approve <nonce>` / `reject <nonce>`
//! responses. Surfacing a duplicate inbound stream here would double-route
//! every chat message.
//!
//! On disconnect the connection handler calls
//! `ChannelDispatcher::unregister_transport(&channel_id)` so the router
//! stops considering this id as a candidate.

use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};

use channel::{
    ChannelDescriptor, ChannelError, ChannelKind, ChannelTransport, DeliveryIntent, InboundMessage,
    MessageHandle, TransportHealth,
};

// `ChannelError::DeliveryFailed` is the right variant for transport-side
// send failures; we surface the underlying mpsc closed error as its body.

/// Stable channel-id prefix used for WS chat sessions. The full id is
/// `ws:<conn-uuid>` so each connection registers a unique descriptor.
pub const WS_CHAT_CHANNEL_PREFIX: &str = "ws:";

pub fn ws_channel_id(conn_id: uuid::Uuid) -> String {
    format!("{WS_CHAT_CHANNEL_PREFIX}{conn_id}")
}

pub struct WsChatTransport {
    descriptor: ChannelDescriptor,
    outbound: mpsc::Sender<DeliveryIntent>,
}

impl WsChatTransport {
    pub fn new(conn_id: uuid::Uuid, outbound: mpsc::Sender<DeliveryIntent>) -> Self {
        let id = ws_channel_id(conn_id);
        let descriptor = ChannelDescriptor::new(id, ChannelKind::Local, "Brain Chat (WS)");
        Self {
            descriptor,
            outbound,
        }
    }

    #[allow(dead_code)]
    pub fn channel_id(&self) -> &str {
        &self.descriptor.id
    }
}

#[async_trait]
impl ChannelTransport for WsChatTransport {
    fn descriptor(&self) -> ChannelDescriptor {
        self.descriptor.clone()
    }

    async fn send(&self, intent: &DeliveryIntent) -> Result<MessageHandle, ChannelError> {
        self.outbound
            .send(intent.clone())
            .await
            .map_err(|e| ChannelError::DeliveryFailed(format!("WS chat outbound dropped: {e}")))?;
        Ok(MessageHandle::new(&intent.id))
    }

    fn inbound(&self) -> broadcast::Receiver<InboundMessage> {
        // The chat session never surfaces inbound through this trait —
        // signal-pipeline routing handles user replies. Hand back a closed
        // receiver so polling never yields and the channel stays inert.
        let (_tx, rx) = broadcast::channel(1);
        rx
    }

    async fn health(&self) -> TransportHealth {
        if self.outbound.is_closed() {
            TransportHealth::Down {
                reason: "WS connection closed".into(),
            }
        } else {
            TransportHealth::Healthy
        }
    }
}
