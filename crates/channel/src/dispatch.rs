//! Channel dispatcher — the production glue that actually delivers a
//! [`DeliveryIntent`] through a registered [`ChannelTransport`].
//!
//! [`ChannelRouter`] returns a ranked list of [`ChannelDescriptor`]s for an
//! intent. The dispatcher owns the inverse map (`descriptor.id` →
//! `Arc<dyn ChannelTransport>`) and walks the candidate list, calling
//! `transport.send()` on the first transport that succeeds. Transports
//! that fail are reported via tracing; the dispatcher falls through to
//! the next candidate so a flaky channel never blocks delivery on its own.
//!
//! Callers register a transport once at boot via [`ChannelDispatcher::register_transport`]
//! — that registers the descriptor with the router AND keeps the transport
//! handle for `dispatch()` to use. After that, anything that needs to push
//! a message out (orchestrator's `Notify` step, confirmation engine's
//! approval prompts) goes through `dispatch()` and forgets about transports
//! entirely.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::error::ChannelError;
use crate::router::{ChannelRouter, RoutingContext};
use crate::transport::{ChannelTransport, MessageHandle};
use crate::types::DeliveryIntent;

/// Receipt returned on successful delivery — captures which channel the
/// intent ended up on and the routing reason, so callers can log or audit
/// the decision trail.
#[derive(Debug, Clone)]
pub struct DeliveryReceipt {
    pub channel_id: String,
    pub reason: String,
    pub handle: MessageHandle,
}

/// Routes via [`ChannelRouter`] and sends via the matching
/// [`ChannelTransport`]. Falls through to the next candidate on any
/// per-transport failure.
pub struct ChannelDispatcher {
    router: Arc<dyn ChannelRouter>,
    transports: RwLock<HashMap<String, Arc<dyn ChannelTransport>>>,
}

impl ChannelDispatcher {
    pub fn new(router: Arc<dyn ChannelRouter>) -> Self {
        Self {
            router,
            transports: RwLock::new(HashMap::new()),
        }
    }

    /// Register a transport — also publishes its descriptor to the router
    /// so it's eligible for selection.
    pub async fn register_transport(
        &self,
        transport: Arc<dyn ChannelTransport>,
    ) -> Result<(), ChannelError> {
        let descriptor = transport.descriptor();
        let id = descriptor.id.clone();
        self.router.register(descriptor).await?;
        self.transports.write().await.insert(id, transport);
        Ok(())
    }

    pub fn router(&self) -> &Arc<dyn ChannelRouter> {
        &self.router
    }

    /// Resolve, send. Returns the receipt of the first successful send.
    /// Each per-channel error is logged but not propagated until the
    /// candidate list is exhausted.
    pub async fn dispatch(&self, intent: DeliveryIntent) -> Result<DeliveryReceipt, ChannelError> {
        self.dispatch_with_context(intent, RoutingContext::default())
            .await
    }

    pub async fn dispatch_with_context(
        &self,
        intent: DeliveryIntent,
        ctx: RoutingContext,
    ) -> Result<DeliveryReceipt, ChannelError> {
        let decision = self.router.route(&intent, &ctx).await?;
        let transports = self.transports.read().await;

        let mut last_err: Option<ChannelError> = None;
        for (descriptor, reason) in decision.candidates.iter().zip(decision.reasons.iter()) {
            let Some(transport) = transports.get(&descriptor.id) else {
                tracing::debug!(
                    channel = %descriptor.id,
                    "router selected channel but no transport is registered for it"
                );
                continue;
            };
            match transport.send(&intent).await {
                Ok(handle) => {
                    return Ok(DeliveryReceipt {
                        channel_id: descriptor.id.clone(),
                        reason: reason.clone(),
                        handle,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        channel = %descriptor.id,
                        error = %e,
                        "delivery failed; trying next candidate"
                    );
                    last_err = Some(e);
                }
            }
        }

        Err(last_err
            .unwrap_or_else(|| ChannelError::NoChannelAvailable(intent.category, intent.urgency)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preference::SqlitePreferenceStore;
    use crate::router::DefaultChannelRouter;
    use crate::transport::TransportHealth;
    use crate::types::{ChannelDescriptor, ChannelKind, DeliveryCategory, UrgencyLevel};
    use async_trait::async_trait;
    use tokio::sync::broadcast;

    struct StubTransport {
        descriptor: ChannelDescriptor,
        fail_send: bool,
        sent: tokio::sync::Mutex<Vec<DeliveryIntent>>,
    }

    impl StubTransport {
        fn new(id: &str, fail: bool) -> Self {
            Self {
                descriptor: ChannelDescriptor::new(id, ChannelKind::Local, id),
                fail_send: fail,
                sent: tokio::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ChannelTransport for StubTransport {
        fn descriptor(&self) -> ChannelDescriptor {
            self.descriptor.clone()
        }
        async fn send(&self, intent: &DeliveryIntent) -> Result<MessageHandle, ChannelError> {
            if self.fail_send {
                return Err(ChannelError::Relay("forced failure".into()));
            }
            self.sent.lock().await.push(intent.clone());
            Ok(MessageHandle::new(&intent.id))
        }
        fn inbound(&self) -> broadcast::Receiver<crate::transport::InboundMessage> {
            let (_, rx) = broadcast::channel(1);
            rx
        }
        async fn health(&self) -> TransportHealth {
            TransportHealth::Healthy
        }
    }

    async fn mk_dispatcher() -> ChannelDispatcher {
        let db = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqlitePreferenceStore::new(db));
        store.ensure_tables().unwrap();
        let router: Arc<dyn ChannelRouter> = Arc::new(DefaultChannelRouter::new(store));
        ChannelDispatcher::new(router)
    }

    #[tokio::test]
    async fn dispatch_succeeds_on_healthy_transport() {
        let dispatcher = mk_dispatcher().await;
        let transport = Arc::new(StubTransport::new("cli", false));
        dispatcher
            .register_transport(transport.clone())
            .await
            .unwrap();

        let intent = DeliveryIntent::new("hello", DeliveryCategory::Response, UrgencyLevel::Normal);
        let receipt = dispatcher.dispatch(intent).await.unwrap();

        assert_eq!(receipt.channel_id, "cli");
        assert_eq!(transport.sent.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn dispatch_falls_through_failed_transport() {
        let dispatcher = mk_dispatcher().await;
        let bad = Arc::new(StubTransport::new("bad", true));
        let good = Arc::new(StubTransport::new("good", false));
        dispatcher.register_transport(bad).await.unwrap();
        dispatcher.register_transport(good.clone()).await.unwrap();

        let intent = DeliveryIntent::new("ping", DeliveryCategory::Nudge, UrgencyLevel::Normal)
            .with_preferred("bad");
        let receipt = dispatcher.dispatch(intent).await.unwrap();

        // Preferred "bad" failed -> should have ended on "good".
        assert_eq!(receipt.channel_id, "good");
        assert_eq!(good.sent.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn dispatch_returns_error_when_all_fail() {
        let dispatcher = mk_dispatcher().await;
        let bad = Arc::new(StubTransport::new("bad", true));
        dispatcher.register_transport(bad).await.unwrap();

        let intent = DeliveryIntent::new("ping", DeliveryCategory::Nudge, UrgencyLevel::Normal);
        let err = dispatcher.dispatch(intent).await.unwrap_err();
        assert!(matches!(err, ChannelError::Relay(_)));
    }
}
