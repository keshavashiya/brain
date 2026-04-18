//! Generic HTTP long-polling transport.
//!
//! Drives inbound via a preset's `PollSpec`: issues templated requests,
//! extracts messages via JSONPath, advances a cursor, publishes each
//! message to a broadcast. Outbound uses the preset's `SendSpec` to
//! POST a templated body.
//!
//! Platform-specific behavior lives in preset YAML, not here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::sync::{broadcast, RwLock};

use crate::error::ChannelError;
use crate::transport::jsonpath::JsonPath;
use crate::transport::preset::{render_template, FieldExtractors, PollSpec, PresetDefinition};
use crate::transport::send::http_send;
use crate::transport::{ChannelTransport, InboundMessage, MessageHandle, TransportHealth};
use crate::types::{ChannelDescriptor, ChannelKind, DeliveryIntent};

const INBOUND_CAPACITY: usize = 256;

/// Configuration for a single HTTP-polled transport instance.
#[derive(Debug, Clone)]
pub struct HttpPolledConfig {
    /// Stable channel id registered with the router.
    pub channel_id: String,
    /// Human-readable label.
    pub label: String,
    /// Preset driving poll + send behaviour.
    pub preset: PresetDefinition,
    /// Credential substituted into `{credential}` in url/body templates —
    /// bot token, API key, etc. Resolved from vault at bootstrap.
    pub credential: String,
}

impl HttpPolledConfig {
    pub fn new(
        channel_id: impl Into<String>,
        label: impl Into<String>,
        preset: PresetDefinition,
        credential: impl Into<String>,
    ) -> Self {
        Self {
            channel_id: channel_id.into(),
            label: label.into(),
            preset,
            credential: credential.into(),
        }
    }
}

/// Generic HTTP-polled transport. Spawn [`run`] on a dedicated task and
/// subscribe via [`ChannelTransport::inbound`] to consume messages.
pub struct HttpPolledTransport {
    config: HttpPolledConfig,
    client: reqwest::Client,
    inbound_tx: broadcast::Sender<InboundMessage>,
    health: RwLock<TransportHealth>,
}

impl HttpPolledTransport {
    pub fn new(config: HttpPolledConfig) -> Result<Self, ChannelError> {
        // Validate preset has a poll block — the engine can't do anything
        // without it. Send is optional (a transport can be inbound-only).
        if config.preset.poll.is_none() {
            return Err(ChannelError::Relay(format!(
                "preset '{}' missing poll block for HttpPolledTransport",
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
        let mut guard = self.health.write().await;
        // Sticky Down — only a healthy poll clears it. Prevents the
        // generic Retry path from masking an auth-rejection signal.
        if matches!(*guard, TransportHealth::Down { .. })
            && !matches!(h, TransportHealth::Healthy | TransportHealth::Down { .. })
        {
            return;
        }
        *guard = h;
    }

    /// Run the polling loop. Returns when the `shutdown` future resolves.
    pub async fn run(self: Arc<Self>, mut shutdown: tokio::sync::oneshot::Receiver<()>) {
        let poll = self.config.preset.poll.clone().expect("validated in new()");
        let mut cursor = poll.cursor_initial.clone();
        let extract_parsers = match PreparedExtractors::build(&poll.extract) {
            Ok(e) => e,
            Err(err) => {
                tracing::error!(
                    channel = %self.config.channel_id,
                    error = %err,
                    "Preset extractors failed to parse — transport disabled"
                );
                self.set_health(TransportHealth::Down {
                    reason: format!("extractor parse: {err}"),
                })
                .await;
                return;
            }
        };
        let cursor_path = match JsonPath::parse(&poll.cursor_field) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    channel = %self.config.channel_id,
                    error = %e,
                    "cursor_field parse failed — transport disabled"
                );
                self.set_health(TransportHealth::Down {
                    reason: format!("cursor_field parse: {e}"),
                })
                .await;
                return;
            }
        };
        let messages_path = match JsonPath::parse(&poll.messages_path) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    channel = %self.config.channel_id,
                    error = %e,
                    "messages_path parse failed — transport disabled"
                );
                self.set_health(TransportHealth::Down {
                    reason: format!("messages_path parse: {e}"),
                })
                .await;
                return;
            }
        };

        loop {
            tokio::select! {
                _ = &mut shutdown => {
                    tracing::info!(
                        channel = %self.config.channel_id,
                        "HttpPolledTransport shutdown signal received"
                    );
                    return;
                }
                step = self.poll_once(&poll, &cursor, &cursor_path, &messages_path, &extract_parsers) => {
                    match step {
                        PollStep::Advance(new_cursor, had_messages) => {
                            cursor = new_cursor;
                            if !had_messages {
                                tokio::time::sleep(Duration::from_millis(poll.idle_ms)).await;
                            }
                        }
                        PollStep::Retry(reason) => {
                            tracing::warn!(
                                channel = %self.config.channel_id,
                                reason = %reason,
                                "Poll failed — backing off"
                            );
                            self.set_health(TransportHealth::Degraded { reason }).await;
                            tokio::time::sleep(Duration::from_secs(2)).await;
                        }
                    }
                }
            }
        }
    }

    async fn poll_once(
        &self,
        poll: &PollSpec,
        cursor: &str,
        cursor_path: &JsonPath,
        messages_path: &JsonPath,
        extractors: &PreparedExtractors,
    ) -> PollStep {
        let mut vars = HashMap::new();
        vars.insert("credential", self.config.credential.as_str());
        vars.insert("cursor", cursor);
        let url = render_template(&poll.url_template, &vars);

        let mut headers = HeaderMap::new();
        for (k, v) in &poll.headers {
            if let (Ok(name), Ok(val)) = (
                HeaderName::try_from(k.as_str()),
                HeaderValue::from_str(&render_template(v, &vars)),
            ) {
                headers.insert(name, val);
            }
        }

        let req = self
            .client
            .request(poll.method.as_reqwest(), &url)
            .headers(headers)
            .timeout(Duration::from_secs(poll.timeout_secs));

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return PollStep::Retry(format!("request send: {e}")),
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            if status.as_u16() == 401 || status.as_u16() == 403 {
                self.set_health(TransportHealth::Down {
                    reason: format!("auth rejected ({status})"),
                })
                .await;
            }
            return PollStep::Retry(format!("status {status}: {body}"));
        }

        let body: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return PollStep::Retry(format!("response json: {e}")),
        };

        // Reset health on a successful fetch.
        self.set_health(TransportHealth::Healthy).await;

        let messages = messages_path.eval(&body);
        let had_messages = !messages.is_empty();
        for msg in &messages {
            match extractors.extract(&self.config.channel_id, msg) {
                Ok(inbound) => {
                    let _ = self.inbound_tx.send(inbound);
                }
                Err(err) => {
                    tracing::debug!(
                        channel = %self.config.channel_id,
                        error = %err,
                        "Skipping message — required extractor missing"
                    );
                }
            }
        }

        // Cursor advances off the response root, not per-message — lets
        // presets like Telegram use `$.result[-1].update_id`.
        let new_cursor = match cursor_path.eval_string(&body) {
            Some(extracted) => poll.cursor_transform.apply(&extracted),
            None => cursor.to_string(),
        };

        PollStep::Advance(new_cursor, had_messages)
    }
}

enum PollStep {
    /// (new cursor, had_messages)
    Advance(String, bool),
    Retry(String),
}

struct PreparedExtractors {
    id: Option<JsonPath>,
    text: JsonPath,
    user_ref: Option<JsonPath>,
    reply_to: Option<JsonPath>,
    extra: Vec<(String, JsonPath)>,
}

impl PreparedExtractors {
    fn build(ex: &FieldExtractors) -> Result<Self, String> {
        Ok(Self {
            id: ex
                .id
                .as_ref()
                .map(|s| JsonPath::parse(s))
                .transpose()
                .map_err(|e| format!("id: {e}"))?,
            text: JsonPath::parse(&ex.text).map_err(|e| format!("text: {e}"))?,
            user_ref: ex
                .user_ref
                .as_ref()
                .map(|s| JsonPath::parse(s))
                .transpose()
                .map_err(|e| format!("user_ref: {e}"))?,
            reply_to: ex
                .reply_to
                .as_ref()
                .map(|s| JsonPath::parse(s))
                .transpose()
                .map_err(|e| format!("reply_to: {e}"))?,
            extra: ex
                .extra
                .iter()
                .map(|(k, v)| {
                    JsonPath::parse(v)
                        .map(|p| (k.clone(), p))
                        .map_err(|e| format!("extra[{k}]: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn extract(&self, channel_id: &str, msg: &Value) -> Result<InboundMessage, &'static str> {
        let text = self.text.eval_string(msg).ok_or("text not found")?;
        let id = self
            .id
            .as_ref()
            .and_then(|p| p.eval_string(msg))
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut inbound = InboundMessage::new(id, text, channel_id);
        inbound.user_ref = self.user_ref.as_ref().and_then(|p| p.eval_string(msg));
        inbound.reply_to = self.reply_to.as_ref().and_then(|p| p.eval_string(msg));
        for (k, path) in &self.extra {
            if let Some(v) = path.eval_string(msg) {
                inbound.extra.insert(k.clone(), v);
            }
        }
        inbound.received_at = Utc::now();
        Ok(inbound)
    }
}

#[async_trait]
impl ChannelTransport for HttpPolledTransport {
    fn descriptor(&self) -> ChannelDescriptor {
        ChannelDescriptor::new(
            &self.config.channel_id,
            ChannelKind::HttpPolled,
            &self.config.label,
        )
    }

    async fn send(&self, intent: &DeliveryIntent) -> Result<MessageHandle, ChannelError> {
        let send = self.config.preset.send.as_ref().ok_or_else(|| {
            ChannelError::Relay(format!(
                "preset '{}' has no send block — transport is inbound-only",
                self.config.preset.id
            ))
        })?;
        http_send(&self.client, send, &self.config.credential, intent).await
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
    use crate::transport::preset::{
        CursorTransform, FieldExtractors, HttpMethod, PollSpec, PresetKind, SendSpec,
    };
    use crate::types::{DeliveryCategory, UrgencyLevel};
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn telegram_like_preset(base: &str) -> PresetDefinition {
        let poll_url = format!("{base}/bot{{credential}}/getUpdates?offset={{cursor}}&timeout=1");
        let send_url = format!("{base}/bot{{credential}}/sendMessage");
        PresetDefinition {
            id: "telegram-test".into(),
            kind: PresetKind::HttpPolled,
            label: Some("Telegram test".into()),
            poll: Some(PollSpec {
                url_template: poll_url,
                method: HttpMethod::Get,
                cursor_initial: "0".into(),
                cursor_field: "$.result[-1].update_id".into(),
                cursor_transform: CursorTransform::PlusOne,
                messages_path: "$.result[*]".into(),
                extract: FieldExtractors {
                    id: Some("$.update_id".into()),
                    text: "$.message.text".into(),
                    user_ref: Some("$.message.from.id".into()),
                    reply_to: Some("$.message.chat.id".into()),
                    extra: HashMap::new(),
                },
                timeout_secs: 5,
                idle_ms: 20,
                headers: HashMap::new(),
            }),
            send: Some(SendSpec {
                url_template: send_url,
                method: HttpMethod::Post,
                body_template: r#"{"chat_id":"{reply_to}","text":"{content}"}"#.into(),
                content_type: "application/json".into(),
                headers: HashMap::new(),
            }),
            webhook: None,
            verifier: None,
        }
    }

    #[tokio::test]
    async fn extracts_messages_and_advances_cursor() {
        let server = MockServer::start().await;

        // First call: cursor=0 returns two messages with update_ids 100,101.
        // Second call: cursor=102 returns empty — test then shuts down.
        Mock::given(method("GET"))
            .and(path("/botTOKEN/getUpdates"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": [
                    {"update_id": 100, "message": {"text": "hello", "from": {"id": 9}, "chat": {"id": 5}}},
                    {"update_id": 101, "message": {"text": "world", "from": {"id": 9}, "chat": {"id": 5}}}
                ]
            })))
            .expect(1..)
            .mount(&server)
            .await;

        let preset = telegram_like_preset(&server.uri());
        let cfg = HttpPolledConfig::new("telegram-test", "tg", preset, "TOKEN");
        let transport = Arc::new(HttpPolledTransport::new(cfg).unwrap());
        let mut rx = transport.inbound();

        let (tx, rx_shutdown) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(transport.clone().run(rx_shutdown));

        // We only need to see the first two messages.
        let m1 = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("recv timeout")
            .expect("channel closed");
        let m2 = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("recv timeout 2")
            .expect("channel closed 2");

        let _ = tx.send(());
        let _ = handle.await;

        assert_eq!(m1.content, "hello");
        assert_eq!(m1.user_ref.as_deref(), Some("9"));
        assert_eq!(m1.reply_to.as_deref(), Some("5"));
        assert_eq!(m2.content, "world");
    }

    #[tokio::test]
    async fn send_posts_templated_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/botTOKEN/sendMessage"))
            .and(body_string_contains("hello from brain"))
            .and(body_string_contains("\"chat_id\":\"42\""))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": {"message_id": 777}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let preset = telegram_like_preset(&server.uri());
        let cfg = HttpPolledConfig::new("telegram-test", "tg", preset, "TOKEN");
        let transport = HttpPolledTransport::new(cfg).unwrap();

        let intent = DeliveryIntent::new(
            "hello from brain",
            DeliveryCategory::Response,
            UrgencyLevel::Normal,
        )
        .with_metadata("reply_to", "42");

        let handle = transport.send(&intent).await.unwrap();
        assert_eq!(handle.platform_id.as_deref(), Some("777"));
    }

    #[tokio::test]
    async fn missing_send_block_errors() {
        let server = MockServer::start().await;
        let mut preset = telegram_like_preset(&server.uri());
        preset.send = None;
        let cfg = HttpPolledConfig::new("tg", "tg", preset, "TOK");
        let transport = HttpPolledTransport::new(cfg).unwrap();
        let intent = DeliveryIntent::new("x", DeliveryCategory::Response, UrgencyLevel::Normal);
        let err = transport.send(&intent).await.unwrap_err();
        matches!(err, ChannelError::Relay(_));
    }

    #[tokio::test]
    async fn auth_rejection_marks_down() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/botTOKEN/getUpdates"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        let preset = telegram_like_preset(&server.uri());
        let cfg = HttpPolledConfig::new("tg", "tg", preset, "TOKEN");
        let transport = Arc::new(HttpPolledTransport::new(cfg).unwrap());

        let (tx, rx_shutdown) = tokio::sync::oneshot::channel();
        let run_handle = tokio::spawn(transport.clone().run(rx_shutdown));

        // Poll runs, gets 401, sets health to Down. Give it a moment.
        tokio::time::sleep(Duration::from_millis(150)).await;

        let h = transport.health().await;
        assert!(matches!(h, TransportHealth::Down { .. }));

        let _ = tx.send(());
        let _ = run_handle.await;
    }
}
