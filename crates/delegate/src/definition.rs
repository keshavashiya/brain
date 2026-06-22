//! The single data shape that describes a delegate agent.
//!
//! One owned, serde-deserializable [`AgentDefinition`] replaces the two
//! shapes that used to drift apart — the old `&'static str`
//! `AgentFingerprint` (auto-discovery) and the owned `CustomAgentSpec`
//! (hand-configured). Built-in agents ship as embedded YAML seeds under
//! `crates/delegate/agents/*.yaml`; users extend or override them by
//! dropping `*.yaml` files into `<data_dir>/agents/`.
//!
//! This mirrors the channel-preset pattern
//! (`crates/channel/src/transport/preset.rs`): embedded seed +
//! user-override, interpreted by one generic engine — never a `vec![]` of
//! definitions baked into the binary.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::traits::AgentCapabilities;

fn default_version_args() -> Vec<String> {
    vec!["--version".to_string()]
}

fn default_true() -> bool {
    true
}

/// A complete, self-describing delegate agent definition.
///
/// Two flavours, distinguished by `binary`:
/// - **Discovered** (`binary` = `None`): the discovery pass scans `$PATH`
///   for the first hit among `binary_names`, then probes its version.
/// - **Explicit** (`binary` = `Some`): a fixed path; registered directly
///   without a `$PATH` scan (the old `CustomAgentSpec` case).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Canonical id used by the registry (e.g. `"claude"`).
    pub id: String,
    /// Optional shorthand/legacy id that resolves to `id`. Lets a renamed
    /// agent keep its old routing name working (e.g. `claude-code`).
    #[serde(default)]
    pub alias: Option<String>,
    /// Candidate binary names tried on `$PATH`, in priority order. Required
    /// for discovered agents; ignored when `binary` is set.
    #[serde(default)]
    pub binary_names: Vec<String>,
    /// Explicit binary path. When set, the agent skips `$PATH` discovery
    /// and is registered directly.
    #[serde(default)]
    pub binary: Option<PathBuf>,
    /// Args to extract a version banner / health-probe the binary.
    #[serde(default = "default_version_args")]
    pub version_args: Vec<String>,
    /// Args passed before the prompt. `{prompt}` and `{task_id}` are
    /// substituted at spawn time; otherwise the prompt goes on stdin.
    #[serde(default)]
    pub args: Vec<String>,
    /// Whether the rendered prompt is written to the child's stdin instead
    /// of being templated into `args`.
    #[serde(default = "default_true")]
    pub prompt_via_stdin: bool,
    /// Declared capabilities used for routing and sandbox policy.
    #[serde(default)]
    pub capabilities: AgentCapabilities,
    /// Working directory the delegate cd's into before spawning. Task-level
    /// workdir wins when present; this is the static default.
    #[serde(default)]
    pub workdir: Option<PathBuf>,
}

impl AgentDefinition {
    /// Reject definitions that can never produce a runnable delegate.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("agent definition has an empty `id`".to_string());
        }
        if self.binary.is_none() && self.binary_names.is_empty() {
            return Err(format!(
                "agent '{}' has neither `binary` nor `binary_names` — nothing to run",
                self.id
            ));
        }
        Ok(())
    }

    /// True for definitions that go through the `$PATH` discovery pass
    /// rather than being registered from a fixed path.
    pub fn is_discoverable(&self) -> bool {
        self.binary.is_none()
    }
}

/// Embedded built-in seeds, shipped with the binary. Extend this list when
/// a new CLI agent becomes common enough to warrant zero-config wiring —
/// but most additions belong in a user `<data_dir>/agents/<id>.yaml` file,
/// which needs no rebuild.
const EMBEDDED_SEEDS: &[(&str, &str)] = &[
    ("claude", include_str!("../agents/claude.yaml")),
    ("codex", include_str!("../agents/codex.yaml")),
    ("aider", include_str!("../agents/aider.yaml")),
    ("gemini", include_str!("../agents/gemini.yaml")),
    ("qwen", include_str!("../agents/qwen.yaml")),
    ("opencode", include_str!("../agents/opencode.yaml")),
];

/// Parse the embedded seed definitions. A malformed seed is logged and
/// skipped rather than panicking — the rest still load.
pub fn embedded_definitions() -> Vec<AgentDefinition> {
    EMBEDDED_SEEDS
        .iter()
        .filter_map(
            |(id, raw)| match serde_yaml::from_str::<AgentDefinition>(raw) {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::error!(agent = %id, error = %e, "embedded agent seed failed to parse");
                    None
                }
            },
        )
        .collect()
}

/// Build the full definition set: embedded seeds merged with user files
/// from `override_dir` (typically `config.override_dir("agents")`).
///
/// Merge is by `id`: a user file with the same id replaces the embedded
/// seed, and a new id adds a brand-new agent family — the zero-rebuild
/// extension path. The result is sorted by id for deterministic ordering.
pub fn load_definitions(override_dir: Option<&Path>) -> Vec<AgentDefinition> {
    let mut by_id: BTreeMap<String, AgentDefinition> = BTreeMap::new();
    for def in embedded_definitions() {
        by_id.insert(def.id.clone(), def);
    }
    if let Some(dir) = override_dir {
        for def in read_user_definitions(dir) {
            by_id.insert(def.id.clone(), def);
        }
    }
    by_id.into_values().collect()
}

/// Read and validate every `*.yaml` under `dir`. Unreadable/invalid files
/// are logged and skipped — one bad file never blocks the rest. A missing
/// directory is simply "no overrides".
fn read_user_definitions(dir: &Path) -> Vec<AgentDefinition> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "could not read user agent definition");
                continue;
            }
        };
        match serde_yaml::from_str::<AgentDefinition>(&raw) {
            Ok(def) => match def.validate() {
                Ok(()) => {
                    tracing::debug!(path = %path.display(), agent = %def.id, "loaded user agent definition");
                    out.push(def);
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "invalid user agent definition — skipped")
                }
            },
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "user agent definition parse failed — skipped")
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_seeds_all_parse_and_validate() {
        let defs = embedded_definitions();
        assert_eq!(defs.len(), EMBEDDED_SEEDS.len(), "every seed must parse");
        for d in &defs {
            d.validate().unwrap_or_else(|e| panic!("{}: {e}", d.id));
        }
    }

    #[test]
    fn ids_use_binary_names_not_package_names() {
        let defs = embedded_definitions();
        let ids: Vec<&str> = defs.iter().map(|d| d.id.as_str()).collect();
        assert!(ids.contains(&"claude"), "claude (not claude-code): {ids:?}");
        assert!(ids.contains(&"gemini"), "gemini (not gemini-cli): {ids:?}");
        assert!(ids.contains(&"qwen"), "qwen (not qwen-code): {ids:?}");
        assert!(!ids.contains(&"claude-code"));
        assert!(!ids.contains(&"gemini-cli"));
        assert!(!ids.contains(&"qwen-code"));
    }

    #[test]
    fn renamed_agents_keep_legacy_alias() {
        let defs = embedded_definitions();
        let claude = defs.iter().find(|d| d.id == "claude").unwrap();
        assert_eq!(claude.alias.as_deref(), Some("claude-code"));
    }

    #[test]
    fn user_dir_overrides_and_extends() {
        let dir = tempfile::tempdir().unwrap();
        // Override an existing seed and add a brand-new family.
        std::fs::write(
            dir.path().join("claude.yaml"),
            "id: claude\nbinary_names: [my-claude]\nargs: [\"--go\"]\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("newcli.yaml"),
            "id: newcli\nbinary_names: [newcli]\n",
        )
        .unwrap();
        let defs = load_definitions(Some(dir.path()));
        let claude = defs.iter().find(|d| d.id == "claude").unwrap();
        assert_eq!(claude.binary_names, vec!["my-claude".to_string()]);
        assert!(defs.iter().any(|d| d.id == "newcli"), "new family added");
    }

    #[test]
    fn invalid_user_file_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("bad.yaml"), "id: \"\"\n").unwrap();
        std::fs::write(
            dir.path().join("good.yaml"),
            "id: good\nbinary_names: [good]\n",
        )
        .unwrap();
        let defs = load_definitions(Some(dir.path()));
        assert!(defs.iter().any(|d| d.id == "good"));
        assert!(!defs.iter().any(|d| d.id.is_empty()));
    }

    #[test]
    fn missing_override_dir_yields_embedded_only() {
        let defs = load_definitions(Some(Path::new("/no/such/dir")));
        assert_eq!(defs.len(), embedded_definitions().len());
    }
}
