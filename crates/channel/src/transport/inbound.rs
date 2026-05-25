//! Generic inbound webhook transport.
//!
//! Accepts an HTTP POST (raw headers + body) from an outer HTTP server,
//! verifies the signature per the preset's `VerifierSpec`, optionally
//! short-circuits a handshake response (Discord PING, Slack
//! `url_verification`), extracts messages via JSONPath, publishes them
//! to a broadcast, and returns a templated ack body.
//!
//! Outbound replies use the preset's `SendSpec` via the shared
//! `http_send` helper.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use reqwest::header::HeaderMap;
use serde_json::Value;
use sha2::Sha256;
use tokio::sync::{broadcast, RwLock};

use crate::error::ChannelError;
use crate::transport::jsonpath::JsonPath;
use crate::transport::preset::{
    render_template, FieldExtractors, PresetDefinition, VerifierSpec, WebhookInboundSpec,
};
use crate::transport::send::http_send;
use crate::transport::{ChannelTransport, InboundMessage, MessageHandle, TransportHealth};
use crate::types::{ChannelDescriptor, ChannelKind, DeliveryIntent};

type HmacSha256 = Hmac<Sha256>;

const INBOUND_CAPACITY: usize = 256;

/// Config for a [`WebhookInboundTransport`] instance.
#[derive(Debug, Clone)]
pub struct WebhookInboundConfig {
    pub channel_id: String,
    pub label: String,
    pub preset: PresetDefinition,
    /// Credential substituted into outbound `{credential}` templates —
    /// bot token, API key. May be empty if the preset has no send block.
    pub credential: String,
    /// HMAC shared secret (raw bytes) or Ed25519 pubkey (hex). Required
    /// when the preset's verifier is anything other than `None`.
    pub signing_secret: Option<String>,
}

impl WebhookInboundConfig {
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
            signing_secret: None,
        }
    }

    pub fn with_credential(mut self, credential: impl Into<String>) -> Self {
        self.credential = credential.into();
        self
    }

    pub fn with_signing_secret(mut self, secret: impl Into<String>) -> Self {
        self.signing_secret = Some(secret.into());
        self
    }
}

/// Response an outer HTTP server should write back to the webhook caller.
#[derive(Debug, Clone)]
pub struct WebhookResponse {
    pub status: u16,
    pub body: String,
    pub content_type: String,
}

impl WebhookResponse {
    pub fn ok() -> Self {
        Self {
            status: 200,
            body: String::new(),
            content_type: "application/json".to_string(),
        }
    }

    pub fn ok_with(body: String, content_type: String) -> Self {
        Self {
            status: 200,
            body,
            content_type,
        }
    }

    pub fn unauthorized(reason: &str) -> Self {
        Self {
            status: 401,
            body: format!("{{\"error\":\"{}\"}}", reason.replace('"', "'")),
            content_type: "application/json".to_string(),
        }
    }

    pub fn bad_request(reason: &str) -> Self {
        Self {
            status: 400,
            body: format!("{{\"error\":\"{}\"}}", reason.replace('"', "'")),
            content_type: "application/json".to_string(),
        }
    }
}

pub struct WebhookInboundTransport {
    config: WebhookInboundConfig,
    client: reqwest::Client,
    inbound_tx: broadcast::Sender<InboundMessage>,
    health: RwLock<TransportHealth>,
    extractors: PreparedExtractors,
    messages_path: JsonPath,
    verifier: PreparedVerifier,
    ack_only_when: Option<(JsonPath, Value)>,
    ack_extractors: Vec<(String, JsonPath)>,
}

enum PreparedVerifier {
    None,
    Hmac {
        key: Vec<u8>,
        header: String,
        prefix: Option<String>,
        timestamp_header: Option<String>,
        max_skew_secs: u64,
    },
    Ed25519 {
        pubkey: VerifyingKey,
        sig_header: String,
        ts_header: String,
        max_skew_secs: u64,
    },
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

impl WebhookInboundTransport {
    pub fn new(config: WebhookInboundConfig) -> Result<Self, ChannelError> {
        let webhook = config.preset.webhook.as_ref().ok_or_else(|| {
            ChannelError::Relay(format!(
                "preset '{}' missing webhook block for WebhookInboundTransport",
                config.preset.id
            ))
        })?;

        let extractors = PreparedExtractors::build(&webhook.extract)
            .map_err(|e| ChannelError::Relay(format!("extractors: {e}")))?;
        let messages_path = JsonPath::parse(&webhook.messages_path)
            .map_err(|e| ChannelError::Relay(format!("messages_path: {e}")))?;

        let verifier = prepare_verifier(
            config.preset.verifier.as_ref(),
            config.signing_secret.as_deref(),
        )?;

        let ack_only_when = webhook
            .ack_only_when
            .as_ref()
            .map(|expr| parse_condition(expr))
            .transpose()
            .map_err(|e| ChannelError::Relay(format!("ack_only_when: {e}")))?;

        let ack_extractors = webhook
            .ack_extract
            .iter()
            .map(|(k, v)| {
                JsonPath::parse(v)
                    .map(|p| (k.clone(), p))
                    .map_err(|e| format!("ack_extract[{k}]: {e}"))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ChannelError::Relay)?;

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
            extractors,
            messages_path,
            verifier,
            ack_only_when,
            ack_extractors,
        })
    }

    pub fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    /// True when this transport ships its own request authentication
    /// (HMAC signature, Ed25519, etc.). The HTTP server uses this to
    /// decide whether `POST /v1/webhooks/:id` may run anonymously or
    /// must additionally require a Brain API key — see Issue 52.
    pub fn has_verifier(&self) -> bool {
        !matches!(self.verifier, PreparedVerifier::None)
    }

    async fn set_health(&self, h: TransportHealth) {
        *self.health.write().await = h;
    }

    /// Handle a single inbound HTTP request. Verifies signature, checks
    /// for handshake shortcut, extracts messages, publishes to inbound,
    /// and returns what the outer HTTP server should write back.
    pub async fn handle_request(&self, headers: &HeaderMap, raw_body: &[u8]) -> WebhookResponse {
        if let Err(reason) = self.verify(headers, raw_body) {
            self.set_health(TransportHealth::Degraded {
                reason: format!("signature rejected: {reason}"),
            })
            .await;
            return WebhookResponse::unauthorized(&reason);
        }

        let body_val: Value = match serde_json::from_slice(raw_body) {
            Ok(v) => v,
            Err(e) => {
                return WebhookResponse::bad_request(&format!("invalid json: {e}"));
            }
        };

        let webhook = self
            .config
            .preset
            .webhook
            .as_ref()
            .expect("validated in new()");

        if let Some((path, expected)) = &self.ack_only_when {
            let matched = path.eval(&body_val).first().map(|v| *v == expected) == Some(true);
            if matched {
                self.set_health(TransportHealth::Healthy).await;
                return self.build_ack(webhook, &body_val);
            }
        }

        let messages = self.messages_path.eval(&body_val);
        for msg in &messages {
            match self.extractors.extract(&self.config.channel_id, msg) {
                Ok(inbound) => {
                    let _ = self.inbound_tx.send(inbound);
                }
                Err(err) => {
                    tracing::debug!(
                        channel = %self.config.channel_id,
                        error = %err,
                        "Skipping webhook message — required extractor missing"
                    );
                }
            }
        }

        self.set_health(TransportHealth::Healthy).await;
        self.build_ack(webhook, &body_val)
    }

    fn build_ack(&self, webhook: &WebhookInboundSpec, body_val: &Value) -> WebhookResponse {
        let Some(tpl) = webhook.ack_body.as_ref() else {
            return WebhookResponse::ok();
        };
        let mut captured: HashMap<String, String> = HashMap::new();
        for (k, path) in &self.ack_extractors {
            if let Some(v) = path.eval_string(body_val) {
                captured.insert(k.clone(), v);
            }
        }
        let vars: HashMap<&str, &str> = captured
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let body = render_template(tpl, &vars);
        WebhookResponse::ok_with(body, webhook.ack_content_type.clone())
    }

    fn verify(&self, headers: &HeaderMap, body: &[u8]) -> Result<(), String> {
        self.verify_at(headers, body, Utc::now().timestamp())
    }

    /// Internal variant accepting an injected "now" so the skew window
    /// is testable without time-traveling the clock.
    fn verify_at(&self, headers: &HeaderMap, body: &[u8], now_secs: i64) -> Result<(), String> {
        match &self.verifier {
            PreparedVerifier::None => Ok(()),
            PreparedVerifier::Hmac {
                key,
                header,
                prefix,
                timestamp_header,
                max_skew_secs,
            } => {
                let sig_header = header_str(headers, header)
                    .ok_or_else(|| format!("missing header {header}"))?;
                let provided = prefix
                    .as_deref()
                    .and_then(|p| sig_header.strip_prefix(p))
                    .unwrap_or(sig_header);
                let mut mac =
                    <HmacSha256 as Mac>::new_from_slice(key).map_err(|e| format!("hmac: {e}"))?;
                mac.update(body);
                let expected = hex::encode(mac.finalize().into_bytes());
                if !constant_time_eq(provided.as_bytes(), expected.as_bytes()) {
                    return Err("hmac mismatch".into());
                }
                if let Some(ts_header) = timestamp_header.as_deref() {
                    let ts = header_str(headers, ts_header)
                        .ok_or_else(|| format!("missing header {ts_header}"))?;
                    check_timestamp_skew(ts, now_secs, *max_skew_secs)?;
                }
                Ok(())
            }
            PreparedVerifier::Ed25519 {
                pubkey,
                sig_header,
                ts_header,
                max_skew_secs,
            } => {
                let sig_hex = header_str(headers, sig_header)
                    .ok_or_else(|| format!("missing header {sig_header}"))?;
                let ts = header_str(headers, ts_header)
                    .ok_or_else(|| format!("missing header {ts_header}"))?;
                let sig_bytes = hex::decode(sig_hex).map_err(|e| format!("signature hex: {e}"))?;
                let sig_arr: [u8; 64] = sig_bytes
                    .try_into()
                    .map_err(|_| "signature must be 64 bytes".to_string())?;
                let signature = Signature::from_bytes(&sig_arr);
                let mut signed = Vec::with_capacity(ts.len() + body.len());
                signed.extend_from_slice(ts.as_bytes());
                signed.extend_from_slice(body);
                pubkey
                    .verify(&signed, &signature)
                    .map_err(|e| format!("ed25519: {e}"))?;
                check_timestamp_skew(ts, now_secs, *max_skew_secs)?;
                Ok(())
            }
        }
    }
}

/// Reject timestamps further than `max_skew_secs` from `now_secs`. Accepts
/// integer seconds; Discord and Slack both use that wire shape. Replay
/// protection on signed-but-stale messages.
fn check_timestamp_skew(ts: &str, now_secs: i64, max_skew_secs: u64) -> Result<(), String> {
    let ts_secs: i64 = ts
        .trim()
        .parse()
        .map_err(|_| format!("timestamp '{ts}' is not an integer"))?;
    let skew = (now_secs - ts_secs).unsigned_abs();
    if skew > max_skew_secs {
        return Err(format!(
            "timestamp skew {skew}s exceeds {max_skew_secs}s window"
        ));
    }
    Ok(())
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn prepare_verifier(
    spec: Option<&VerifierSpec>,
    secret: Option<&str>,
) -> Result<PreparedVerifier, ChannelError> {
    match spec {
        None | Some(VerifierSpec::None) => Ok(PreparedVerifier::None),
        Some(VerifierSpec::HmacSha256 {
            header,
            prefix,
            timestamp_header,
            max_skew_secs,
        }) => {
            let key_str = secret.ok_or_else(|| {
                ChannelError::Relay("HmacSha256 verifier requires signing_secret".into())
            })?;
            Ok(PreparedVerifier::Hmac {
                key: key_str.as_bytes().to_vec(),
                header: header.clone(),
                prefix: prefix.clone(),
                timestamp_header: timestamp_header.clone(),
                max_skew_secs: *max_skew_secs,
            })
        }
        Some(VerifierSpec::DiscordEd25519 {
            signature_header,
            timestamp_header,
            max_skew_secs,
        }) => {
            let pub_hex = secret.ok_or_else(|| {
                ChannelError::Relay(
                    "DiscordEd25519 verifier requires signing_secret (pubkey hex)".into(),
                )
            })?;
            let bytes = hex::decode(pub_hex)
                .map_err(|e| ChannelError::Relay(format!("pubkey hex: {e}")))?;
            let arr: [u8; 32] = bytes
                .try_into()
                .map_err(|_| ChannelError::Relay("pubkey must be 32 bytes".into()))?;
            let pubkey = VerifyingKey::from_bytes(&arr)
                .map_err(|e| ChannelError::Relay(format!("pubkey: {e}")))?;
            Ok(PreparedVerifier::Ed25519 {
                pubkey,
                sig_header: signature_header.clone(),
                ts_header: timestamp_header.clone(),
                max_skew_secs: *max_skew_secs,
            })
        }
    }
}

fn parse_condition(expr: &str) -> Result<(JsonPath, Value), String> {
    let (lhs, rhs) = expr
        .split_once(" == ")
        .ok_or_else(|| format!("expected '<path> == <json>', got '{expr}'"))?;
    let path = JsonPath::parse(lhs.trim()).map_err(|e| format!("{e}"))?;
    let value: Value =
        serde_json::from_str(rhs.trim()).map_err(|e| format!("rhs not valid JSON: {e}"))?;
    Ok((path, value))
}

#[async_trait]
impl ChannelTransport for WebhookInboundTransport {
    fn descriptor(&self) -> ChannelDescriptor {
        ChannelDescriptor::new(
            &self.config.channel_id,
            ChannelKind::WebhookInbound,
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

// `Arc<WebhookInboundTransport>` is the expected usage so outer HTTP
// handlers can share the transport across requests.
impl WebhookInboundTransport {
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::preset::{
        FieldExtractors, HttpMethod, PresetKind, SendSpec, VerifierSpec, WebhookInboundSpec,
    };
    use ed25519_dalek::{Signer, SigningKey};
    use rand::RngCore;
    use reqwest::header::{HeaderName, HeaderValue};

    fn make_preset(verifier: Option<VerifierSpec>, ack: Option<&str>) -> PresetDefinition {
        let mut webhook = WebhookInboundSpec {
            messages_path: "$".into(),
            extract: FieldExtractors {
                id: Some("$.id".into()),
                text: "$.text".into(),
                user_ref: Some("$.user".into()),
                reply_to: None,
                extra: HashMap::new(),
            },
            ack_body: ack.map(String::from),
            ack_content_type: "application/json".into(),
            ack_extract: HashMap::new(),
            ack_only_when: None,
        };
        if ack.is_some() {
            webhook.ack_only_when = Some("$.type == 1".into());
            webhook.ack_body = ack.map(String::from);
        }
        PresetDefinition {
            id: "webhook-test".into(),
            kind: PresetKind::WebhookInbound,
            label: Some("test".into()),
            poll: None,
            send: Some(SendSpec {
                url_template: "http://example.invalid/{reply_to}".into(),
                method: HttpMethod::Post,
                body_template: "{}".into(),
                content_type: "application/json".into(),
                headers: HashMap::new(),
            }),
            webhook: Some(webhook),
            verifier,
        }
    }

    #[tokio::test]
    async fn unverified_request_with_no_verifier_is_accepted() {
        let preset = make_preset(None, None);
        let cfg = WebhookInboundConfig::new("w", "w", preset);
        let t = WebhookInboundTransport::new(cfg).unwrap();
        let mut rx = t.inbound();

        let body = br#"{"id":"m1","text":"hi","user":"u1"}"#;
        let resp = t.handle_request(&HeaderMap::new(), body).await;
        assert_eq!(resp.status, 200);

        let m = tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(m.content, "hi");
        assert_eq!(m.id, "m1");
        assert_eq!(m.user_ref.as_deref(), Some("u1"));
    }

    #[tokio::test]
    async fn hmac_mismatch_rejected() {
        let preset = make_preset(
            Some(VerifierSpec::HmacSha256 {
                header: "X-Sig".into(),
                prefix: Some("sha256=".into()),
                timestamp_header: None,
                max_skew_secs: 300,
            }),
            None,
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret("topsecret");
        let t = WebhookInboundTransport::new(cfg).unwrap();

        let body = br#"{"id":"m1","text":"hi","user":"u1"}"#;
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-sig"),
            HeaderValue::from_static("sha256=deadbeef"),
        );
        let resp = t.handle_request(&headers, body).await;
        assert_eq!(resp.status, 401);
    }

    #[tokio::test]
    async fn hmac_valid_signature_accepted() {
        let preset = make_preset(
            Some(VerifierSpec::HmacSha256 {
                header: "X-Sig".into(),
                prefix: Some("sha256=".into()),
                timestamp_header: None,
                max_skew_secs: 300,
            }),
            None,
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret("topsecret");
        let t = WebhookInboundTransport::new(cfg).unwrap();
        let mut rx = t.inbound();

        let body = br#"{"id":"m1","text":"hi","user":"u1"}"#;
        let mut mac = <HmacSha256 as Mac>::new_from_slice(b"topsecret").unwrap();
        mac.update(body);
        let sig = hex::encode(mac.finalize().into_bytes());

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-sig"),
            HeaderValue::from_str(&format!("sha256={sig}")).unwrap(),
        );
        let resp = t.handle_request(&headers, body).await;
        assert_eq!(resp.status, 200);

        let m = tokio::time::timeout(Duration::from_millis(100), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(m.content, "hi");
    }

    #[tokio::test]
    async fn ed25519_handshake_short_circuits() {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        let signing = SigningKey::from_bytes(&seed);
        let verify: VerifyingKey = signing.verifying_key();
        let pub_hex = hex::encode(verify.to_bytes());

        let preset = make_preset(
            Some(VerifierSpec::DiscordEd25519 {
                signature_header: "X-Signature-Ed25519".into(),
                timestamp_header: "X-Signature-Timestamp".into(),
                max_skew_secs: 1_000_000_000,
            }),
            Some(r#"{"type": 1}"#),
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret(pub_hex);
        let t = WebhookInboundTransport::new(cfg).unwrap();
        let mut rx = t.inbound();

        let body = br#"{"type": 1}"#;
        let ts = "1234567890";
        let mut signed = Vec::new();
        signed.extend_from_slice(ts.as_bytes());
        signed.extend_from_slice(body);
        let sig = signing.sign(&signed);

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-signature-ed25519"),
            HeaderValue::from_str(&hex::encode(sig.to_bytes())).unwrap(),
        );
        headers.insert(
            HeaderName::from_static("x-signature-timestamp"),
            HeaderValue::from_static("1234567890"),
        );

        let resp = t.handle_request(&headers, body).await;
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("\"type\""));
        // Handshake must NOT produce an inbound message.
        assert!(tokio::time::timeout(Duration::from_millis(50), rx.recv())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn ed25519_bad_signature_rejected() {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        let signing = SigningKey::from_bytes(&seed);
        let pub_hex = hex::encode(signing.verifying_key().to_bytes());

        let preset = make_preset(
            Some(VerifierSpec::DiscordEd25519 {
                signature_header: "X-Signature-Ed25519".into(),
                timestamp_header: "X-Signature-Timestamp".into(),
                max_skew_secs: 1_000_000_000,
            }),
            Some(r#"{"type": 1}"#),
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret(pub_hex);
        let t = WebhookInboundTransport::new(cfg).unwrap();

        let body = br#"{"type": 1}"#;
        // Sign wrong bytes.
        let sig = signing.sign(b"different");

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-signature-ed25519"),
            HeaderValue::from_str(&hex::encode(sig.to_bytes())).unwrap(),
        );
        headers.insert(
            HeaderName::from_static("x-signature-timestamp"),
            HeaderValue::from_static("1"),
        );

        let resp = t.handle_request(&headers, body).await;
        assert_eq!(resp.status, 401);
    }

    // ─── Replay-protection (Issue 58) ────────────────────────────────────

    impl WebhookInboundTransport {
        /// Test-only accessor so the replay-protection unit tests can read
        /// the raw verifier verdict without round-tripping through
        /// `handle_request` (which translates errors into HTTP responses).
        fn verifier_verify(&self, headers: &HeaderMap, body: &[u8]) -> Result<(), String> {
            self.verify(headers, body)
        }
    }

    #[tokio::test]
    async fn hmac_with_timestamp_rejects_stale_signature() {
        let preset = make_preset(
            Some(VerifierSpec::HmacSha256 {
                header: "X-Sig".into(),
                prefix: None,
                timestamp_header: Some("X-Ts".into()),
                max_skew_secs: 60,
            }),
            None,
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret("topsecret");
        let t = WebhookInboundTransport::new(cfg).unwrap();

        let body = br#"{"id":"m1","text":"hi","user":"u1"}"#;
        let mut mac = <HmacSha256 as Mac>::new_from_slice(b"topsecret").unwrap();
        mac.update(body);
        let sig = hex::encode(mac.finalize().into_bytes());

        let now = Utc::now().timestamp();
        let stale_ts = (now - 3600).to_string();
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-sig"),
            HeaderValue::from_str(&sig).unwrap(),
        );
        headers.insert(
            HeaderName::from_static("x-ts"),
            HeaderValue::from_str(&stale_ts).unwrap(),
        );

        let err = t.verifier_verify(&headers, body).unwrap_err();
        assert!(err.contains("skew"), "{err}");
    }

    #[tokio::test]
    async fn hmac_with_timestamp_accepts_fresh_signature() {
        let preset = make_preset(
            Some(VerifierSpec::HmacSha256 {
                header: "X-Sig".into(),
                prefix: None,
                timestamp_header: Some("X-Ts".into()),
                max_skew_secs: 60,
            }),
            None,
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret("topsecret");
        let t = WebhookInboundTransport::new(cfg).unwrap();

        let body = br#"{"id":"m1","text":"hi","user":"u1"}"#;
        let mut mac = <HmacSha256 as Mac>::new_from_slice(b"topsecret").unwrap();
        mac.update(body);
        let sig = hex::encode(mac.finalize().into_bytes());

        let now = Utc::now().timestamp();
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-sig"),
            HeaderValue::from_str(&sig).unwrap(),
        );
        headers.insert(
            HeaderName::from_static("x-ts"),
            HeaderValue::from_str(&now.to_string()).unwrap(),
        );

        assert!(t.verifier_verify(&headers, body).is_ok());
    }

    #[tokio::test]
    async fn ed25519_rejects_stale_timestamp() {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        let signing = SigningKey::from_bytes(&seed);
        let pub_hex = hex::encode(signing.verifying_key().to_bytes());

        let preset = make_preset(
            Some(VerifierSpec::DiscordEd25519 {
                signature_header: "X-Signature-Ed25519".into(),
                timestamp_header: "X-Signature-Timestamp".into(),
                max_skew_secs: 60,
            }),
            None,
        );
        let cfg = WebhookInboundConfig::new("w", "w", preset).with_signing_secret(pub_hex);
        let t = WebhookInboundTransport::new(cfg).unwrap();

        let body = br#"{"id":"m1","text":"hi","user":"u1"}"#;
        let stale_ts = (Utc::now().timestamp() - 3600).to_string();
        let mut signed = Vec::new();
        signed.extend_from_slice(stale_ts.as_bytes());
        signed.extend_from_slice(body);
        let sig = signing.sign(&signed);

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-signature-ed25519"),
            HeaderValue::from_str(&hex::encode(sig.to_bytes())).unwrap(),
        );
        headers.insert(
            HeaderName::from_static("x-signature-timestamp"),
            HeaderValue::from_str(&stale_ts).unwrap(),
        );

        let err = t.verifier_verify(&headers, body).unwrap_err();
        assert!(err.contains("skew"), "{err}");
    }
}
