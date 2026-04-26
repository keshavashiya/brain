//! Channel-backed approval notifier.
//!
//! Bridges the `confirm` crate's [`ApprovalNotifier`] hook to the
//! `channel` crate's [`ChannelDispatcher`]. Lives in `signal` (rather
//! than `confirm` or `channel`) because this is the only crate that
//! depends on both — keeping it here avoids inverting the
//! `confirm` → `channel` dependency direction.

use std::sync::Arc;

use async_trait::async_trait;

use channel::{ChannelDispatcher, DeliveryCategory, DeliveryIntent, UrgencyLevel};
use confirm::notifier::{ApprovalNotifier, NotifyError};
use confirm::ApprovalSpec;

/// Sends approval prompts through the channel dispatcher. Maps the
/// confirm engine's tier → urgency, formats a prompt body that includes
/// the nonce so the user knows how to reply, and delegates routing to
/// the dispatcher (which respects learned channel preferences).
pub struct ChannelApprovalNotifier {
    dispatcher: Arc<ChannelDispatcher>,
}

impl ChannelApprovalNotifier {
    pub fn new(dispatcher: Arc<ChannelDispatcher>) -> Self {
        Self { dispatcher }
    }
}

#[async_trait]
impl ApprovalNotifier for ChannelApprovalNotifier {
    async fn notify(&self, spec: &ApprovalSpec) -> Result<(), NotifyError> {
        let urgency = match spec.tier {
            confirm::ActionTier::Destructive => UrgencyLevel::High,
            confirm::ActionTier::External => UrgencyLevel::Normal,
            // The other tiers are auto-approved upstream so should never
            // reach the notifier; default to Normal for forward-compat.
            _ => UrgencyLevel::Normal,
        };

        let body = format!(
            "Approval needed ({tier}):\n  {desc}\n\nReply `approve {nonce}` or `reject {nonce}`.",
            tier = spec.tier,
            desc = spec.action_description,
            nonce = spec.nonce,
        );

        let mut intent = DeliveryIntent::new(body, DeliveryCategory::Confirm, urgency)
            .with_nonce(spec.nonce.clone());
        if let Some(ch) = &spec.preferred_channel {
            intent = intent.with_preferred(ch);
        }

        self.dispatcher
            .dispatch(intent)
            .await
            .map(|_receipt| ())
            .map_err(|e| NotifyError::Delivery(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use channel::{
        ChannelDescriptor, ChannelDispatcher, ChannelKind, ChannelRouter, ChannelTransport,
        DefaultChannelRouter, MessageHandle, SqlitePreferenceStore, TransportHealth,
    };
    use confirm::ActionTier;
    use tokio::sync::broadcast;

    struct CapturingTransport {
        descriptor: ChannelDescriptor,
        captured: tokio::sync::Mutex<Vec<DeliveryIntent>>,
    }

    #[async_trait]
    impl ChannelTransport for CapturingTransport {
        fn descriptor(&self) -> ChannelDescriptor {
            self.descriptor.clone()
        }
        async fn send(
            &self,
            intent: &DeliveryIntent,
        ) -> Result<MessageHandle, channel::ChannelError> {
            self.captured.lock().await.push(intent.clone());
            Ok(MessageHandle::new(&intent.id))
        }
        fn inbound(&self) -> broadcast::Receiver<channel::InboundMessage> {
            let (_, rx) = broadcast::channel(1);
            rx
        }
        async fn health(&self) -> TransportHealth {
            TransportHealth::Healthy
        }
    }

    #[tokio::test]
    async fn notifier_dispatches_with_nonce_in_body() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqlitePreferenceStore::new(db));
        store.ensure_tables().unwrap();
        let router: Arc<dyn ChannelRouter> = Arc::new(DefaultChannelRouter::new(store));
        let dispatcher = Arc::new(ChannelDispatcher::new(router));
        let transport = Arc::new(CapturingTransport {
            descriptor: ChannelDescriptor::new("cli", ChannelKind::Local, "CLI"),
            captured: tokio::sync::Mutex::new(Vec::new()),
        });
        dispatcher
            .register_transport(transport.clone())
            .await
            .unwrap();

        let notifier = ChannelApprovalNotifier::new(dispatcher);
        let spec = ApprovalSpec::new("force-push to main", ActionTier::Destructive);
        notifier.notify(&spec).await.unwrap();

        let captured = transport.captured.lock().await;
        assert_eq!(captured.len(), 1);
        assert!(captured[0].content.contains(&spec.nonce));
        assert!(captured[0].content.contains("force-push"));
        assert_eq!(captured[0].nonce.as_deref(), Some(spec.nonce.as_str()));
    }
}
