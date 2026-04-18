//! Preset loader — reads a preset by id, preferring a user override at
//! `~/.brain/presets/<id>.yaml` over the embedded copy shipped with the
//! crate.
//!
//! The embedded presets are compiled in via `include_str!` so no
//! runtime file lookup is required in the common case.

use std::path::PathBuf;

use crate::error::ChannelError;
use crate::transport::preset::PresetDefinition;

const EMBEDDED_TELEGRAM: &str = include_str!("../../presets/telegram.yaml");
const EMBEDDED_DISCORD_INTERACTIONS: &str = include_str!("../../presets/discord-interactions.yaml");
const EMBEDDED_SLACK_WEBHOOK: &str = include_str!("../../presets/slack-webhook.yaml");

/// Look up the YAML source for a preset id — user override first, then
/// embedded fallback. Returns `None` if no preset is known by that id.
pub fn load_yaml(id: &str) -> Option<String> {
    if let Some(path) = user_override_path(id) {
        if let Ok(text) = std::fs::read_to_string(&path) {
            tracing::debug!(preset = %id, path = %path.display(), "loaded user preset override");
            return Some(text);
        }
    }
    embedded_yaml(id).map(String::from)
}

/// Parse a preset by id — same lookup order as [`load_yaml`].
pub fn load(id: &str) -> Result<PresetDefinition, ChannelError> {
    let yaml =
        load_yaml(id).ok_or_else(|| ChannelError::Relay(format!("unknown preset id: {id}")))?;
    PresetDefinition::from_yaml(&yaml)
        .map_err(|e| ChannelError::Relay(format!("preset '{id}' parse: {e}")))
}

/// Embedded preset source (no filesystem access) — useful for tests and
/// when the user override path is unavailable.
pub fn embedded_yaml(id: &str) -> Option<&'static str> {
    match id {
        "telegram" => Some(EMBEDDED_TELEGRAM),
        "discord-interactions" => Some(EMBEDDED_DISCORD_INTERACTIONS),
        "slack-webhook" => Some(EMBEDDED_SLACK_WEBHOOK),
        _ => None,
    }
}

fn user_override_path(id: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut path = PathBuf::from(home);
    path.push(".brain");
    path.push("presets");
    path.push(format!("{id}.yaml"));
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::preset::PresetKind;

    #[test]
    fn telegram_preset_parses() {
        let p = load("telegram").unwrap();
        assert_eq!(p.id, "telegram");
        assert_eq!(p.kind, PresetKind::HttpPolled);
        assert!(p.poll.is_some());
        assert!(p.send.is_some());
    }

    #[test]
    fn discord_interactions_preset_parses() {
        let p = load("discord-interactions").unwrap();
        assert_eq!(p.kind, PresetKind::WebhookInbound);
        assert!(p.webhook.is_some());
        assert!(p.verifier.is_some());
        assert!(p.send.is_some());
    }

    #[test]
    fn slack_webhook_preset_parses() {
        let p = load("slack-webhook").unwrap();
        assert_eq!(p.kind, PresetKind::WebhookOutbound);
        assert!(p.send.is_some());
    }

    #[test]
    fn unknown_preset_errors() {
        assert!(load("nope-no-preset").is_err());
    }
}
