//! Declarative preset definitions — the data that drives the generic
//! transport engines.
//!
//! A preset describes a platform's HTTP surface without any platform-
//! specific code: URL templates, JSONPath extractors, signature
//! verification. The generic engines (`HttpPolledTransport`,
//! `WebhookInboundTransport`, `WebhookOutboundTransport`) interpret a
//! preset at runtime.
//!
//! Presets ship embedded under `crates/channel/presets/*.yaml` and are
//! loaded lazily. Users can drop overrides into the `presets/` override
//! directory (`config.override_dir("presets")`, i.e.
//! `<data_dir>/presets/<id>.yaml`) which take precedence over the embedded
//! copy. The override directory is passed in by the caller so it honors
//! `brain.data_dir` rather than re-deriving `$HOME/.brain` here.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ChannelError;

const EMBEDDED_TELEGRAM: &str = include_str!("../../presets/telegram.yaml");
const EMBEDDED_DISCORD: &str = include_str!("../../presets/discord.yaml");
const EMBEDDED_SLACK: &str = include_str!("../../presets/slack.yaml");

/// Supported preset shapes. Each maps to one transport engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresetKind {
    /// Long-poll an HTTP endpoint (Telegram `getUpdates`, Mastodon streams).
    HttpPolled,
    /// Receive inbound via HTTP callback (Discord Interactions, Slack Events,
    /// GitHub webhooks).
    WebhookInbound,
    /// Push outbound via HTTP POST only — no inbound path (Slack incoming,
    /// Discord webhook URL, generic).
    WebhookOutbound,
}

/// HTTP method for templated requests. `Post` is the common case; `Get`
/// is used for the Telegram-style long-poll GET with query string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

impl HttpMethod {
    pub fn as_reqwest(&self) -> reqwest::Method {
        match self {
            Self::Get => reqwest::Method::GET,
            Self::Post => reqwest::Method::POST,
            Self::Put => reqwest::Method::PUT,
            Self::Delete => reqwest::Method::DELETE,
            Self::Patch => reqwest::Method::PATCH,
        }
    }
}

/// How to advance the cursor after processing a batch of messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CursorTransform {
    /// Use the extracted value as-is.
    Identity,
    /// Parse as i64 and add 1 (Telegram update_id semantics).
    PlusOne,
    /// Replace with a literal value.
    Literal(String),
}

impl CursorTransform {
    pub fn apply(&self, extracted: &str) -> String {
        match self {
            Self::Identity => extracted.to_string(),
            Self::PlusOne => extracted
                .parse::<i64>()
                .map(|n| (n + 1).to_string())
                .unwrap_or_else(|_| extracted.to_string()),
            Self::Literal(v) => v.clone(),
        }
    }
}

/// JSONPath expressions the polling engine uses to pull fields out of each
/// message payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldExtractors {
    /// Path to the platform-side message id. Optional — falls back to
    /// a generated UUID per message.
    #[serde(default)]
    pub id: Option<String>,
    /// Path to the message text content (required).
    pub text: String,
    /// Path to the user identifier (optional).
    #[serde(default)]
    pub user_ref: Option<String>,
    /// Path to the chat/thread identifier used by outbound replies.
    #[serde(default)]
    pub reply_to: Option<String>,
    /// Additional named extractors whose values end up in
    /// [`InboundMessage::extra`](super::InboundMessage::extra).
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

/// Polling loop configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollSpec {
    pub url_template: String,
    pub method: HttpMethod,
    /// Value used for `{cursor}` on the very first request.
    #[serde(default)]
    pub cursor_initial: String,
    /// JSONPath to the next-cursor value in a response.
    pub cursor_field: String,
    /// Transform applied to the extracted cursor before the next request.
    #[serde(default = "default_identity")]
    pub cursor_transform: CursorTransform,
    /// JSONPath yielding each message in the response (wildcard expected).
    pub messages_path: String,
    pub extract: FieldExtractors,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Sleep between polls when the response has no messages.
    #[serde(default = "default_idle_ms")]
    pub idle_ms: u64,
    /// Optional static headers added to every poll request.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

fn default_identity() -> CursorTransform {
    CursorTransform::Identity
}
fn default_timeout_secs() -> u64 {
    45
}
fn default_idle_ms() -> u64 {
    500
}

/// Outbound send configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendSpec {
    pub url_template: String,
    pub method: HttpMethod,
    /// Body template. Variables: `{content}`, `{reply_to}`, `{id}`,
    /// `{credential}`, plus any key from the intent's metadata.
    pub body_template: String,
    #[serde(default = "default_content_type_json")]
    pub content_type: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

fn default_content_type_json() -> String {
    "application/json".to_string()
}

/// Inbound webhook configuration — how to extract messages from an
/// incoming HTTP POST and what (optional) body to return as the HTTP
/// response. Used by [`PresetKind::WebhookInbound`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookInboundSpec {
    /// JSONPath yielding each message in the request body. Use `$` when
    /// the body itself is a single message (Discord interactions, Slack
    /// slash commands) or `$.event` / `$.result[*]` for event wrappers.
    pub messages_path: String,
    pub extract: FieldExtractors,
    /// Optional body sent back as the HTTP response after verification.
    /// Rendered with `{var}` substitution — available vars are `challenge`
    /// (Slack url_verification) and any captured from the request body
    /// via `ack_extract`.
    #[serde(default)]
    pub ack_body: Option<String>,
    /// Content type for `ack_body`.
    #[serde(default = "default_content_type_json")]
    pub ack_content_type: String,
    /// Optional named extractors captured from the request body and
    /// exposed as template vars for `ack_body`. For Slack URL
    /// verification the preset maps `challenge: "$.challenge"`.
    #[serde(default)]
    pub ack_extract: HashMap<String, String>,
    /// If the request body matches this JSON-path-evaluated value, the
    /// transport returns `ack_body` *without* publishing to inbound.
    /// Used for handshake echoes (Discord `type: 1` PING, Slack
    /// `url_verification`). Format: `$.path == "literal"` or
    /// `$.path == 1`.
    #[serde(default)]
    pub ack_only_when: Option<String>,
}

/// Signature verifier used by [`PresetKind::WebhookInbound`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum VerifierSpec {
    /// No verification — for trusted internal webhooks only.
    None,
    /// HMAC-SHA256 of the raw body, hex-encoded in a header.
    HmacSha256 {
        /// Header carrying the signature (e.g. `X-Hub-Signature-256`).
        header: String,
        /// Optional prefix to strip (`sha256=`).
        #[serde(default)]
        prefix: Option<String>,
        /// Optional header carrying a unix-seconds timestamp the signature
        /// was generated at. When configured the receiver enforces
        /// `max_skew_secs` against the current clock — this is the replay
        /// window. Presets without a signed timestamp leave this `None`
        /// and remain vulnerable to replay on the signature alone.
        #[serde(default)]
        timestamp_header: Option<String>,
        /// Replay window in seconds — only consulted when `timestamp_header`
        /// is set. Default 300 (industry standard for HMAC webhooks).
        #[serde(default = "default_max_skew_secs")]
        max_skew_secs: u64,
    },
    /// Discord Interactions Ed25519 signature — pubkey-verified.
    DiscordEd25519 {
        /// Header carrying the signature.
        #[serde(default = "default_discord_sig_header")]
        signature_header: String,
        /// Header carrying the timestamp.
        #[serde(default = "default_discord_ts_header")]
        timestamp_header: String,
        /// Replay window in seconds. The Ed25519 signature already binds
        /// the timestamp into the signed payload, so a stale timestamp
        /// cannot be tampered with — but it CAN be re-sent. Reject anything
        /// older than this window.
        #[serde(default = "default_max_skew_secs")]
        max_skew_secs: u64,
    },
}

fn default_max_skew_secs() -> u64 {
    300
}

fn default_discord_sig_header() -> String {
    "X-Signature-Ed25519".to_string()
}
fn default_discord_ts_header() -> String {
    "X-Signature-Timestamp".to_string()
}

/// Full preset definition loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetDefinition {
    /// Preset id (file stem: `telegram`, `discord`, `slack`, …).
    pub id: String,
    pub kind: PresetKind,
    /// Human-readable label the UI can show.
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub poll: Option<PollSpec>,
    #[serde(default)]
    pub send: Option<SendSpec>,
    #[serde(default)]
    pub webhook: Option<WebhookInboundSpec>,
    #[serde(default)]
    pub verifier: Option<VerifierSpec>,
}

impl PresetDefinition {
    pub fn from_yaml(raw: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(raw)
    }
}

/// Look up the YAML source for a preset id — user override at
/// `<override_dir>/<id>.yaml` first (when `override_dir` is supplied), then
/// embedded fallback. Returns `None` if no preset is known by that id.
pub fn load_yaml(id: &str, override_dir: Option<&Path>) -> Option<String> {
    if let Some(dir) = override_dir {
        let path = dir.join(format!("{id}.yaml"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            tracing::debug!(preset = %id, path = %path.display(), "loaded user preset override");
            return Some(text);
        }
    }
    embedded_yaml(id).map(String::from)
}

/// Parse a preset by id — same lookup order as [`load_yaml`]. Pass the
/// resolved presets override directory (`config.override_dir("presets")`)
/// or `None` to use embedded presets only.
pub fn load(id: &str, override_dir: Option<&Path>) -> Result<PresetDefinition, ChannelError> {
    let yaml = load_yaml(id, override_dir)
        .ok_or_else(|| ChannelError::Relay(format!("unknown preset id: {id}")))?;
    PresetDefinition::from_yaml(&yaml)
        .map_err(|e| ChannelError::Relay(format!("preset '{id}' parse: {e}")))
}

/// Embedded preset source (no filesystem access) — useful for tests and
/// when the user override path is unavailable.
pub fn embedded_yaml(id: &str) -> Option<&'static str> {
    match id {
        "telegram" => Some(EMBEDDED_TELEGRAM),
        "discord" => Some(EMBEDDED_DISCORD),
        "slack" => Some(EMBEDDED_SLACK),
        _ => None,
    }
}

/// Substitute `{var}` tokens in a template. Unknown variables are left in
/// place — this is intentional so missing-var bugs surface visibly at
/// the HTTP layer instead of silently collapsing to empty strings.
pub fn render_template(template: &str, vars: &HashMap<&str, &str>) -> String {
    let mut out = template.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{k}}}"), v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_transforms() {
        assert_eq!(CursorTransform::Identity.apply("7"), "7");
        assert_eq!(CursorTransform::PlusOne.apply("7"), "8");
        assert_eq!(CursorTransform::PlusOne.apply("not-an-int"), "not-an-int");
        assert_eq!(
            CursorTransform::Literal("fixed".into()).apply("ignored"),
            "fixed"
        );
    }

    #[test]
    fn render_simple() {
        let mut vars = HashMap::new();
        vars.insert("name", "world");
        vars.insert("greeting", "hi");
        assert_eq!(render_template("{greeting}, {name}!", &vars), "hi, world!");
    }

    #[test]
    fn render_leaves_unknown_alone() {
        let vars = HashMap::new();
        assert_eq!(render_template("{unknown}", &vars), "{unknown}");
    }

    #[test]
    fn telegram_preset_parses() {
        let p = load("telegram", None).unwrap();
        assert_eq!(p.id, "telegram");
        assert_eq!(p.kind, PresetKind::HttpPolled);
        assert!(p.poll.is_some());
        assert!(p.send.is_some());
    }

    #[test]
    fn discord_preset_parses() {
        let p = load("discord", None).unwrap();
        assert_eq!(p.id, "discord");
        assert_eq!(p.kind, PresetKind::WebhookInbound);
        assert!(p.webhook.is_some());
        assert!(p.verifier.is_some());
        assert!(p.send.is_some());
    }

    #[test]
    fn slack_preset_parses() {
        let p = load("slack", None).unwrap();
        assert_eq!(p.id, "slack");
        assert_eq!(p.kind, PresetKind::WebhookOutbound);
        assert!(p.send.is_some());
    }

    #[test]
    fn unknown_preset_errors() {
        assert!(load("nope-no-preset", None).is_err());
    }

    #[test]
    fn user_override_dir_takes_precedence_over_embedded() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("telegram.yaml"),
            "id: telegram\nkind: webhook_outbound\nsend:\n  url_template: \"https://example.test\"\n  method: POST\n  body_template: '{}'\n",
        )
        .unwrap();
        // Override wins over the embedded http_polled telegram preset.
        let p = load("telegram", Some(dir.path())).unwrap();
        assert_eq!(p.kind, PresetKind::WebhookOutbound);
        // With no override dir, the embedded preset is returned.
        let embedded = load("telegram", None).unwrap();
        assert_eq!(embedded.kind, PresetKind::HttpPolled);
    }

    #[test]
    fn preset_roundtrip_yaml() {
        let yaml = r#"
id: telegram
kind: http_polled
label: "Telegram"
poll:
  url_template: "https://api.telegram.org/bot{credential}/getUpdates?offset={cursor}&timeout=30"
  method: GET
  cursor_initial: "0"
  cursor_field: "$.result[-1].update_id"
  cursor_transform: plus_one
  messages_path: "$.result[*]"
  extract:
    text: "$.message.text"
    user_ref: "$.message.from.id"
    reply_to: "$.message.chat.id"
send:
  url_template: "https://api.telegram.org/bot{credential}/sendMessage"
  method: POST
  body_template: '{"chat_id":"{reply_to}","text":"{content}"}'
"#;
        let preset = PresetDefinition::from_yaml(yaml).unwrap();
        assert_eq!(preset.id, "telegram");
        assert_eq!(preset.kind, PresetKind::HttpPolled);
        let poll = preset.poll.unwrap();
        assert_eq!(poll.cursor_transform, CursorTransform::PlusOne);
        assert_eq!(poll.method, HttpMethod::Get);
        let send = preset.send.unwrap();
        assert_eq!(send.method, HttpMethod::Post);
    }
}
