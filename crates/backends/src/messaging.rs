//! Message backend — webhook delivery with resilience.

use std::collections::HashMap;
use std::sync::Arc;

use brain::metrics::SubsystemMetrics;

use crate::resilience::{http_breaker, resilient_send, CircuitBreaker};

pub const DEFAULT_MESSAGE_BODY: &str = r#"{"channel":"{{channel}}","recipient":"{{recipient}}","content":"{{content}}","namespace":"{{namespace}}","timestamp":"{{timestamp}}"}"#;

/// JSON-escape a string value (without surrounding quotes).
pub fn json_escape(s: &str) -> String {
    let escaped = serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s));
    escaped[1..escaped.len() - 1].to_string()
}

/// Render a message template by replacing `{{placeholder}}` tokens.
pub fn render_message_template(
    template: &str,
    channel: &str,
    recipient: &str,
    content: &str,
    namespace: &str,
    timestamp: &str,
) -> String {
    template
        .replace("{{channel}}", &json_escape(channel))
        .replace("{{recipient}}", &json_escape(recipient))
        .replace("{{content}}", &json_escape(content))
        .replace("{{namespace}}", &json_escape(namespace))
        .replace("{{timestamp}}", &json_escape(timestamp))
}

pub struct WebhookMessageBackend {
    channels: HashMap<String, brain::config::ChannelConfig>,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl WebhookMessageBackend {
    pub fn new(
        channels: &HashMap<String, brain::config::ChannelConfig>,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
    ) -> Result<Self, crate::error::BackendInitError> {
        Self::new_with_metrics(channels, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        channels: &HashMap<String, brain::config::ChannelConfig>,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> Result<Self, crate::error::BackendInitError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
            .build()
            .map_err(|e| crate::error::BackendInitError::HttpClient("message client", e))?;
        let cb = http_breaker(
            "webhook-message",
            resilience.circuit_breaker_threshold,
            resilience.circuit_breaker_cooldown_secs,
            metrics,
        );
        Ok(Self {
            channels: channels
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect(),
            client,
            circuit_breaker: Arc::new(cb),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl signal::notification::WebhookSender for WebhookMessageBackend {
    async fn send_notification(
        &self,
        channel: &str,
        content: &str,
        namespace: &str,
    ) -> Result<(), String> {
        let channel_cfg = self
            .channels
            .get(&channel.to_ascii_lowercase())
            .ok_or_else(|| format!("No webhook mapping for channel '{channel}'"))?
            .clone();

        let client = self.client.clone();
        let template = if channel_cfg.body.is_empty() {
            DEFAULT_MESSAGE_BODY.to_string()
        } else {
            channel_cfg.body.clone()
        };
        let url = channel_cfg.url.clone();
        let headers = channel_cfg.headers.clone();
        let channel_owned = channel.to_string();
        let content_owned = content.to_string();
        let namespace_owned = namespace.to_string();

        let response = resilient_send(
            || {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let rendered = render_message_template(
                    &template,
                    &channel_owned,
                    "",
                    &content_owned,
                    &namespace_owned,
                    &timestamp,
                );
                let mut req = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(rendered);
                for (key, value) in &headers {
                    req = req.header(key.as_str(), value.as_str());
                }
                req
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| format!("webhook send failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!(
                "webhook for channel '{}' returned HTTP {}",
                channel,
                response.status()
            ));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl cortex::actions::MessageBackend for WebhookMessageBackend {
    async fn send(
        &self,
        channel: &str,
        recipient: &str,
        content: &str,
        namespace: &str,
    ) -> Result<cortex::actions::MessageOutcome, cortex::actions::ActionError> {
        let channel_cfg = self
            .channels
            .get(&channel.to_ascii_lowercase())
            .ok_or_else(|| {
                cortex::actions::ActionError::InvalidArguments(format!(
                    "No webhook mapping for channel '{}'",
                    channel
                ))
            })?
            .clone();

        let client = self.client.clone();
        let template = if channel_cfg.body.is_empty() {
            DEFAULT_MESSAGE_BODY.to_string()
        } else {
            channel_cfg.body.clone()
        };
        let url = channel_cfg.url.clone();
        let headers = channel_cfg.headers.clone();
        let channel_owned = channel.to_string();
        let recipient_owned = recipient.to_string();
        let content_owned = content.to_string();
        let namespace_owned = namespace.to_string();

        let response = resilient_send(
            || {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let rendered = render_message_template(
                    &template,
                    &channel_owned,
                    &recipient_owned,
                    &content_owned,
                    &namespace_owned,
                    &timestamp,
                );
                let mut req = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(rendered);
                for (key, value) in &headers {
                    req = req.header(key.as_str(), value.as_str());
                }
                req
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "webhook for channel '{}' returned HTTP {}",
                channel,
                response.status()
            )));
        }

        let body = response.text().await.unwrap_or_default();
        let mut delivery_id = format!("msg-{}", chrono::Utc::now().timestamp_micros());
        let mut status = "accepted".to_string();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(id) = value
                .get("id")
                .or_else(|| value.get("delivery_id"))
                .and_then(serde_json::Value::as_str)
            {
                delivery_id = id.to_string();
            }
            if let Some(s) = value.get("status").and_then(serde_json::Value::as_str) {
                status = s.to_string();
            }
        }

        Ok(cortex::actions::MessageOutcome {
            delivery_id,
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_message_template_default() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            "deploy done",
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("default template should produce valid JSON");
        assert_eq!(parsed["channel"], "alerts");
        assert_eq!(parsed["recipient"], "alice");
        assert_eq!(parsed["content"], "deploy done");
        assert_eq!(parsed["namespace"], "work");
        assert_eq!(parsed["timestamp"], "2026-03-08T12:00:00Z");
    }

    #[test]
    fn test_render_message_template_custom_template() {
        let template = r#"{"text": "[{{channel}}] {{content}}"}"#;
        let rendered = render_message_template(
            template,
            "ops",
            "bob",
            "server is down",
            "personal",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("custom template should produce valid JSON");
        assert_eq!(parsed["text"], "[ops] server is down");
    }

    #[test]
    fn test_render_message_template_escapes_quotes() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            r#"He said "hello""#,
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("escaped content should produce valid JSON");
        assert_eq!(parsed["content"], r#"He said "hello""#);
    }

    #[test]
    fn test_render_message_template_escapes_newlines() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            "line1\nline2",
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("newline content should produce valid JSON");
        assert_eq!(parsed["content"], "line1\nline2");
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape(r#"say "hi""#), r#"say \"hi\""#);
        assert_eq!(json_escape("a\nb"), r#"a\nb"#);
        assert_eq!(json_escape("back\\slash"), r#"back\\slash"#);
    }

    // ─── Mock HTTP tests ────────────────────────────────────────────────────

    fn fast_resilience() -> brain::config::ResilienceConfig {
        brain::config::ResilienceConfig {
            max_retries: 0,
            retry_base_ms: 10,
            circuit_breaker_threshold: 5,
            circuit_breaker_cooldown_secs: 60,
        }
    }

    fn build_channels(name: &str, url: &str) -> HashMap<String, brain::config::ChannelConfig> {
        let mut map = HashMap::new();
        map.insert(
            name.to_string(),
            brain::config::ChannelConfig {
                url: url.to_string(),
                body: String::new(),
                headers: HashMap::new(),
            },
        );
        map
    }

    #[tokio::test]
    async fn test_webhook_send_success() {
        use cortex::actions::MessageBackend;

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .match_header("content-type", "application/json")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"id": "delivery-123", "status": "sent"}"#)
            .create_async()
            .await;

        let webhook_url = format!("{}/hook", server.url());
        let channels = build_channels("alerts", &webhook_url);
        let backend = WebhookMessageBackend::new(&channels, 5000, &fast_resilience()).unwrap();
        let result = backend
            .send("alerts", "alice", "server is down", "work")
            .await
            .unwrap();
        assert_eq!(result.delivery_id, "delivery-123");
        assert_eq!(result.status, "sent");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_webhook_send_5xx_fails() {
        use cortex::actions::MessageBackend;

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/hook")
            .with_status(500)
            .with_body("server error")
            .expect_at_least(1)
            .create_async()
            .await;

        let webhook_url = format!("{}/hook", server.url());
        let channels = build_channels("alerts", &webhook_url);
        let backend = WebhookMessageBackend::new(&channels, 5000, &fast_resilience()).unwrap();
        let result = backend.send("alerts", "alice", "boom", "work").await;
        assert!(result.is_err(), "expected 5xx to fail");
    }

    #[tokio::test]
    async fn test_webhook_unknown_channel_returns_invalid_args() {
        use cortex::actions::MessageBackend;

        let channels = build_channels("alerts", "http://unused");
        let backend = WebhookMessageBackend::new(&channels, 5000, &fast_resilience()).unwrap();
        let err = backend
            .send("unknown-channel", "alice", "hi", "work")
            .await
            .unwrap_err();
        assert!(
            matches!(err, cortex::actions::ActionError::InvalidArguments(_)),
            "expected InvalidArguments error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_webhook_fallback_delivery_id_when_no_body_id() {
        use cortex::actions::MessageBackend;

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_body(r#"{"other": "field"}"#)
            .create_async()
            .await;

        let webhook_url = format!("{}/hook", server.url());
        let channels = build_channels("alerts", &webhook_url);
        let backend = WebhookMessageBackend::new(&channels, 5000, &fast_resilience()).unwrap();
        let result = backend.send("alerts", "alice", "hi", "work").await.unwrap();
        // Fallback ID is a generated timestamp-based token starting with "msg-"
        assert!(result.delivery_id.starts_with("msg-"));
        assert_eq!(result.status, "accepted");
    }
}
