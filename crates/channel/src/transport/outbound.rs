//! Generic push-only outbound webhook transport.
//!
//! For platforms that accept a one-way HTTP POST (Slack incoming
//! webhooks, Discord webhook URLs, generic "fire-and-forget" endpoints).
//! No inbound path — subscribers get a receiver that never yields.

use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{broadcast, RwLock};

use crate::error::ChannelError;
use crate::transport::preset::PresetDefinition;
use crate::transport::send::http_send;
use crate::transport::{ChannelTransport, InboundMessage, MessageHandle, TransportHealth};
use crate::types::{ChannelDescriptor, ChannelKind, DeliveryIntent};

const INBOUND_CAPACITY: usize = 1;

#[derive(Debug, Clone)]
pub struct WebhookOutboundConfig {
    pub channel_id: String,
    pub label: String,
    pub preset: PresetDefinition,
    /// Credential substituted into `{credential}` — typically empty
    /// because webhook URLs already carry the secret in-path.
    pub credential: String,
}

impl WebhookOutboundConfig {
    pub fn new(
        channel_id: impl Into<String>,
        label: impl Into<String>,
        preset: PresetDefinition,
    ) -> Self {
        Self {
            channel_id: channel_id.into(),
            label: label.into(),
            preset,
            credential: String::new(),
        }
    }

    pub fn with_credential(mut self, credential: impl Into<String>) -> Self {
        self.credential = credential.into();
        self
    }
}

pub struct WebhookOutboundTransport {
    config: WebhookOutboundConfig,
    client: reqwest::Client,
    inbound_tx: broadcast::Sender<InboundMessage>,
    health: RwLock<TransportHealth>,
}

impl WebhookOutboundTransport {
    pub fn new(config: WebhookOutboundConfig) -> Result<Self, ChannelError> {
        if config.preset.send.is_none() {
            return Err(ChannelError::Relay(format!(
                "preset '{}' missing send block for WebhookOutboundTransport",
                config.preset.id
            )));
        }
        let client = reqwest::Client::builder()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .build()
            .map_err(|e| ChannelError::Relay(format!("reqwest client: {e}")))?;
        let (inbound_tx, _) = broadcast::channel(INBOUND_CAPACITY);
        Ok(Self {
            config,
            client,
            inbound_tx,
            health: RwLock::new(TransportHealth::Healthy),
        })
    }

    pub fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    async fn set_health(&self, h: TransportHealth) {
        *self.health.write().await = h;
    }
}

#[async_trait]
impl ChannelTransport for WebhookOutboundTransport {
    fn descriptor(&self) -> ChannelDescriptor {
        ChannelDescriptor::new(
            &self.config.channel_id,
            ChannelKind::Webhook,
            &self.config.label,
        )
    }

    async fn send(&self, intent: &DeliveryIntent) -> Result<MessageHandle, ChannelError> {
        let send = self
            .config
            .preset
            .send
            .as_ref()
            .expect("validated in new()");
        match http_send(&self.client, send, &self.config.credential, intent).await {
            Ok(handle) => {
                self.set_health(TransportHealth::Healthy).await;
                Ok(handle)
            }
            Err(e) => {
                self.set_health(TransportHealth::Degraded {
                    reason: e.to_string(),
                })
                .await;
                Err(e)
            }
        }
    }

    fn inbound(&self) -> broadcast::Receiver<InboundMessage> {
        self.inbound_tx.subscribe()
    }

    async fn health(&self) -> TransportHealth {
        self.health.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::preset::{HttpMethod, PresetKind, SendSpec};
    use crate::types::{DeliveryCategory, UrgencyLevel};
    use std::collections::HashMap;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn slack_like_preset(url: &str) -> PresetDefinition {
        PresetDefinition {
            id: "slack".into(),
            kind: PresetKind::WebhookOutbound,
            label: Some("Slack".into()),
            poll: None,
            send: Some(SendSpec {
                url_template: url.to_string(),
                method: HttpMethod::Post,
                body_template: r#"{"text":"{content}"}"#.into(),
                content_type: "application/json".into(),
                headers: HashMap::new(),
            }),
            webhook: None,
            verifier: None,
        }
    }

    #[tokio::test]
    async fn outbound_posts_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .and(body_string_contains("hello slack"))
            .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
            .expect(1)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let preset = slack_like_preset(&url);
        let cfg = WebhookOutboundConfig::new("slack", "slack", preset);
        let t = WebhookOutboundTransport::new(cfg).unwrap();

        let intent =
            DeliveryIntent::new("hello slack", DeliveryCategory::Nudge, UrgencyLevel::Normal);
        t.send(&intent).await.unwrap();

        assert!(matches!(t.health().await, TransportHealth::Healthy));
    }

    #[tokio::test]
    async fn failure_marks_degraded() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/hook"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .expect(1)
            .mount(&server)
            .await;

        let url = format!("{}/hook", server.uri());
        let preset = slack_like_preset(&url);
        let cfg = WebhookOutboundConfig::new("slack", "slack", preset);
        let t = WebhookOutboundTransport::new(cfg).unwrap();

        let intent = DeliveryIntent::new("x", DeliveryCategory::Nudge, UrgencyLevel::Normal);
        assert!(t.send(&intent).await.is_err());
        assert!(matches!(t.health().await, TransportHealth::Degraded { .. }));
    }

    #[tokio::test]
    async fn missing_send_block_rejected() {
        let preset = PresetDefinition {
            id: "bad".into(),
            kind: PresetKind::WebhookOutbound,
            label: None,
            poll: None,
            send: None,
            webhook: None,
            verifier: None,
        };
        let cfg = WebhookOutboundConfig::new("x", "x", preset);
        assert!(WebhookOutboundTransport::new(cfg).is_err());
    }
}
