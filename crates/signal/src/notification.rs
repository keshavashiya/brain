//! Notification Router — delivers proactive messages through multiple tiers.
//!
//! Delivery tiers (all run on every message):
//! 1. **Outbox** — SQLite write, guaranteed. Drained on next user interaction.
//! 2. **Broadcast** — tokio broadcast channel for live WS/SSE sessions.
//! 3. **Webhooks** — push to configured messaging channels (Slack, Discord, etc.).

use storage::SqlitePool;
use tokio::sync::broadcast;
use tracing;

/// A proactive notification ready for delivery.
#[derive(Debug, Clone)]
pub struct ProactiveNotification {
    pub content: String,
    pub triggered_by: String,
    pub priority: i32,
    /// Originating agent (if the notification is agent-attributed).
    pub agent: Option<String>,
}

impl From<ganglia::ProactiveMessage> for ProactiveNotification {
    fn from(msg: ganglia::ProactiveMessage) -> Self {
        Self {
            content: msg.content,
            triggered_by: msg.triggered_by,
            priority: 1,
            agent: msg.agent,
        }
    }
}

/// Routes proactive messages through outbox, broadcast, and webhook tiers.
pub struct NotificationRouter {
    db: SqlitePool,
    proactive_tx: broadcast::Sender<ProactiveNotification>,
    delivery_config: brain_core::DeliveryConfig,
    /// Webhook sender — when set, proactive messages are pushed to configured channels.
    webhook_sender: Option<Box<dyn WebhookSender>>,
}

/// Trait for sending webhook notifications (implemented by WebhookMessageBackend in cli).
#[async_trait::async_trait]
pub trait WebhookSender: Send + Sync {
    async fn send_notification(
        &self,
        channel: &str,
        content: &str,
        namespace: &str,
    ) -> Result<(), String>;
}

impl NotificationRouter {
    /// Create a new notification router.
    pub fn new(db: SqlitePool, delivery_config: brain_core::DeliveryConfig) -> Self {
        let (proactive_tx, _) = broadcast::channel(256);
        Self {
            db,
            proactive_tx,
            delivery_config,
            webhook_sender: None,
        }
    }

    /// Attach a webhook sender for push delivery.
    pub fn with_webhook_sender(mut self, sender: Box<dyn WebhookSender>) -> Self {
        self.webhook_sender = Some(sender);
        self
    }

    /// Subscribe to the proactive notification broadcast channel.
    pub fn subscribe(&self) -> broadcast::Receiver<ProactiveNotification> {
        self.proactive_tx.subscribe()
    }

    /// Deliver a proactive message through all configured tiers.
    pub async fn deliver(&self, notification: ProactiveNotification) {
        // Tier 1: Outbox (always, if enabled)
        if self.delivery_config.outbox {
            if let Err(e) = self.db.insert_notification(
                &notification.content,
                notification.priority,
                &notification.triggered_by,
                None,
            ) {
                tracing::warn!("Failed to write notification to outbox: {e}");
            }
        }

        // Tier 2: Broadcast to live sessions (if enabled)
        if self.delivery_config.broadcast {
            // send() only fails if there are no active receivers — that's fine.
            let _ = self.proactive_tx.send(notification.clone());
        }

        // Tier 3: Webhook push to configured channels
        if let Some(sender) = &self.webhook_sender {
            for channel_key in &self.delivery_config.webhook_channels {
                if let Err(e) = sender
                    .send_notification(channel_key, &notification.content, "personal")
                    .await
                {
                    tracing::warn!(
                        channel = %channel_key,
                        "Webhook notification delivery failed: {e}"
                    );
                }
            }
        }
    }

    /// Drain pending outbox items, returning them and marking as delivered.
    pub fn drain_pending(&self, limit: usize) -> Vec<storage::sqlite::Notification> {
        match self.db.pending_notifications(limit) {
            Ok(notifications) => {
                for n in &notifications {
                    if let Err(e) = self.db.mark_notification_delivered(&n.id) {
                        tracing::warn!(id = %n.id, "Failed to mark notification delivered: {e}");
                    }
                }
                notifications
            }
            Err(e) => {
                tracing::warn!("Failed to read pending notifications: {e}");
                Vec::new()
            }
        }
    }

    /// Prune old outbox entries (called during consolidation).
    pub fn prune(&self) {
        let max_age = self.delivery_config.max_outbox_age_days;
        match self.db.prune_notifications(max_age) {
            Ok(n) if n > 0 => tracing::info!(pruned = n, "Pruned old outbox notifications"),
            Ok(_) => {}
            Err(e) => tracing::warn!("Failed to prune outbox: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_router() -> NotificationRouter {
        let db = SqlitePool::open_memory().unwrap();
        let config = brain_core::DeliveryConfig {
            outbox: true,
            broadcast: true,
            webhook_channels: Vec::new(),
            max_outbox_age_days: 7,
        };
        NotificationRouter::new(db, config)
    }

    #[tokio::test]
    async fn test_deliver_writes_to_outbox() {
        let router = test_router();
        let notification = ProactiveNotification {
            content: "Time to review your PRs".into(),
            triggered_by: "habit:pr_review".into(),
            priority: 2,
            agent: None,
        };
        router.deliver(notification).await;

        let pending = router.drain_pending(10);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].content, "Time to review your PRs");
        assert_eq!(pending[0].priority, 2);
        assert_eq!(pending[0].triggered_by, "habit:pr_review");
    }

    #[tokio::test]
    async fn test_deliver_broadcasts_to_subscribers() {
        let router = test_router();
        let mut rx = router.subscribe();

        let notification = ProactiveNotification {
            content: "Don't forget your standup".into(),
            triggered_by: "habit:standup".into(),
            priority: 1,
            agent: None,
        };
        router.deliver(notification).await;

        let received = rx.try_recv().unwrap();
        assert_eq!(received.content, "Don't forget your standup");
    }

    #[tokio::test]
    async fn test_drain_marks_delivered_idempotent() {
        let router = test_router();
        let notification = ProactiveNotification {
            content: "Test".into(),
            triggered_by: "test".into(),
            priority: 1,
            agent: None,
        };
        router.deliver(notification).await;

        // First drain returns the notification
        let first = router.drain_pending(10);
        assert_eq!(first.len(), 1);

        // Second drain returns empty (already delivered)
        let second = router.drain_pending(10);
        assert_eq!(second.len(), 0);
    }

    #[tokio::test]
    async fn test_prune_removes_old_delivered() {
        let router = test_router();

        // Deliver and drain (marks as delivered)
        let notification = ProactiveNotification {
            content: "Old nudge".into(),
            triggered_by: "test".into(),
            priority: 1,
            agent: None,
        };
        router.deliver(notification).await;
        let drained = router.drain_pending(10);
        assert_eq!(drained.len(), 1);

        // Prune should not panic even with no old entries
        router.prune();
    }

    /// Mock WebhookSender that records calls for testing.
    struct MockWebhookSender {
        calls: std::sync::Mutex<Vec<(String, String, String)>>,
    }

    impl MockWebhookSender {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl WebhookSender for MockWebhookSender {
        async fn send_notification(
            &self,
            channel: &str,
            content: &str,
            namespace: &str,
        ) -> Result<(), String> {
            self.calls.lock().unwrap().push((
                channel.to_string(),
                content.to_string(),
                namespace.to_string(),
            ));
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_webhook_sender_called_for_configured_channels() {
        let db = SqlitePool::open_memory().unwrap();
        let config = brain_core::DeliveryConfig {
            outbox: false,
            broadcast: false,
            webhook_channels: vec!["slack-general".into(), "discord-alerts".into()],
            max_outbox_age_days: 7,
        };
        let sender = Arc::new(MockWebhookSender::new());
        let router = NotificationRouter::new(db, config)
            .with_webhook_sender(Box::new(MockWebhookSenderWrapper(sender.clone())));

        let notification = ProactiveNotification {
            content: "Time to hydrate".into(),
            triggered_by: "habit:water".into(),
            priority: 1,
            agent: None,
        };
        router.deliver(notification).await;

        let calls = sender.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "slack-general");
        assert_eq!(calls[0].1, "Time to hydrate");
        assert_eq!(calls[0].2, "personal");
        assert_eq!(calls[1].0, "discord-alerts");
    }

    /// Wrapper to bridge Arc<MockWebhookSender> into Box<dyn WebhookSender>.
    struct MockWebhookSenderWrapper(Arc<MockWebhookSender>);

    #[async_trait::async_trait]
    impl WebhookSender for MockWebhookSenderWrapper {
        async fn send_notification(
            &self,
            channel: &str,
            content: &str,
            namespace: &str,
        ) -> Result<(), String> {
            self.0.send_notification(channel, content, namespace).await
        }
    }

    #[tokio::test]
    async fn test_webhook_not_called_when_no_channels() {
        let db = SqlitePool::open_memory().unwrap();
        let config = brain_core::DeliveryConfig {
            outbox: false,
            broadcast: false,
            webhook_channels: Vec::new(),
            max_outbox_age_days: 7,
        };
        let sender = Arc::new(MockWebhookSender::new());
        let router = NotificationRouter::new(db, config)
            .with_webhook_sender(Box::new(MockWebhookSenderWrapper(sender.clone())));

        let notification = ProactiveNotification {
            content: "Should not trigger webhook".into(),
            triggered_by: "test".into(),
            priority: 1,
            agent: None,
        };
        router.deliver(notification).await;

        let calls = sender.calls.lock().unwrap();
        assert_eq!(calls.len(), 0);
    }

    #[tokio::test]
    async fn test_from_proactive_message() {
        let msg = ganglia::ProactiveMessage {
            content: "You usually check email around now".into(),
            triggered_by: "email".into(),
            created_at: chrono::Utc::now(),
            agent: None,
        };
        let notification: ProactiveNotification = msg.into();
        assert_eq!(notification.content, "You usually check email around now");
        assert_eq!(notification.triggered_by, "email");
        assert_eq!(notification.priority, 1);
    }
}
