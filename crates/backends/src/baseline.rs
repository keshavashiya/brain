//! System baseline + drift detection — the executor behind
//! `brain baseline capture/diff/list` and the `baseline.capture`/`diff`/`list`
//! native capabilities.
//!
//! A *baseline* is a deterministic, offline snapshot of the brain's stable
//! system facts — version, platform, the effective LLM/embedding wiring,
//! adapter exposure, action toggles, the live native-capability inventory,
//! and the security surface. Each snapshot is a flat `key → value` map: every
//! entry is one fact, so a later *diff* can report precisely which facts were
//! added, removed, or changed since the baseline was taken.
//!
//! **Storage.** Snapshots are persisted as versioned JSON files under
//! `~/.brain/baselines/baseline-<NNNN>.json`, where `<NNNN>` is a monotonically
//! increasing version number. This is local-first and inspectable, and — unlike
//! writing into the semantic fact store — it never opens SQLite/RuVector, so it
//! can't contend with a running daemon (the same reason `brain status` avoids
//! opening the DB directly).
//!
//! **Truthful by construction.** Capture reads only the loaded config, the
//! process environment, and the capability inventory it is handed. There is no
//! LLM and no network: the snapshot is exactly what the running config declares,
//! and a diff is a literal set comparison of two such snapshots.
//!
//! **Capability inventory is injected.** The native-capability descriptors live
//! in the binary crate; rather than reach back into it (a layering inversion),
//! the caller passes the inventory in as a [`CapabilitySummary`] slice — the
//! composition root assembles it from the descriptor list. Keeps this core a
//! pure function of its inputs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use brain::BrainConfig;
use serde::{Deserialize, Serialize};

/// One native capability, flattened for the baseline snapshot. The composition
/// root builds these from the registered descriptors and hands them to
/// [`capture`] / [`diff`], so this core never depends on the descriptor source.
#[derive(Debug, Clone)]
pub struct CapabilitySummary {
    pub namespace: String,
    pub action: String,
    /// The consent tier (e.g. `read`, `write`, `external`); `unknown` if unset.
    pub tier: String,
}

/// One persisted baseline snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Baseline {
    /// Monotonic version number; also the `<NNNN>` in the filename.
    version: u32,
    /// When the snapshot was captured (RFC3339).
    captured_at: String,
    /// Optional human-readable label.
    #[serde(default)]
    label: Option<String>,
    /// The flat fact map — every entry is one `key → value` system fact.
    facts: BTreeMap<String, String>,
}

// ─── capture ──────────────────────────────────────────────────────────────────

/// Capture a new baseline snapshot and store it as the next version. Returns the
/// rendered confirmation (version, fact count, and the file path).
pub fn capture(
    config: &BrainConfig,
    inventory: &[CapabilitySummary],
    label: Option<&str>,
) -> Result<String> {
    let dir = baseline_dir(config);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating baseline directory {}", dir.display()))?;

    let version = next_version(&dir);
    let baseline = Baseline {
        version,
        captured_at: chrono::Utc::now().to_rfc3339(),
        label: label.map(str::to_string),
        facts: capture_facts(config, inventory),
    };

    let path = baseline_path(&dir, version);
    let json =
        serde_json::to_string_pretty(&baseline).context("serialising the baseline snapshot")?;
    std::fs::write(&path, json)
        .with_context(|| format!("writing baseline to {}", path.display()))?;

    let mut out = format!(
        "Captured baseline v{} ({} facts){}\n",
        version,
        baseline.facts.len(),
        label.map(|l| format!(" — {l}")).unwrap_or_default(),
    );
    out.push_str(&format!("  {}", path.display()));
    Ok(out)
}

/// Build the snapshot from the loaded config, the process environment, and the
/// injected capability inventory. No I/O beyond reading `config` — so the result
/// is deterministic and stable across runs of the same config + inventory.
fn capture_facts(
    config: &BrainConfig,
    inventory: &[CapabilitySummary],
) -> BTreeMap<String, String> {
    let mut f = BTreeMap::new();
    let mut put = |k: &str, v: String| {
        f.insert(k.to_string(), v);
    };

    // Identity & platform.
    put("brain.version", env!("CARGO_PKG_VERSION").to_string());
    put("os.platform", std::env::consts::OS.to_string());
    put("os.arch", std::env::consts::ARCH.to_string());

    // LLM wiring. Per-provider entries make drift point at the exact provider
    // that changed; the legacy single-provider fields are recorded only when no
    // `providers[]` are configured (their deprecated-but-live fallback role).
    put("llm.context_window", config.llm.context_window.to_string());
    if config.llm.providers.is_empty() {
        #[allow(deprecated)]
        {
            put("llm.provider", config.llm.provider.clone());
            put("llm.model", config.llm.model.clone());
        }
    } else {
        for p in &config.llm.providers {
            put(
                &format!("llm.provider.{}", p.name),
                format!("{} ({})", p.kind, p.model),
            );
        }
    }

    // Embedding + encryption.
    put("embedding.model", config.embedding.model.clone());
    put(
        "embedding.dimensions",
        config.embedding.dimensions.to_string(),
    );
    put("encryption.enabled", config.encryption.enabled.to_string());

    // Adapter exposure — what the daemon listens on.
    let a = &config.adapters;
    put("adapter.http", adapter_fact(a.http.enabled, a.http.port));
    put("adapter.ws", adapter_fact(a.ws.enabled, a.ws.port));
    put("adapter.mcp", adapter_fact(a.mcp.enabled, a.mcp.port));
    put("adapter.grpc", adapter_fact(a.grpc.enabled, a.grpc.port));
    put(
        "adapter.terminal",
        adapter_fact(a.terminal.enabled, a.terminal.port),
    );

    // Action toggles.
    put(
        "actions.web_search",
        config.actions.web_search.enabled.to_string(),
    );
    put(
        "actions.scheduling",
        config.actions.scheduling.enabled.to_string(),
    );
    put(
        "actions.messaging",
        config.actions.messaging.enabled.to_string(),
    );

    // Agents.
    put(
        "agents.auto_discovery",
        config.agents.auto_discovery.to_string(),
    );
    let mut delegates: Vec<&str> = config
        .agents
        .delegates
        .iter()
        .map(|d| d.name.as_str())
        .collect();
    delegates.sort_unstable();
    put("agents.delegates", join_or_none(&delegates));

    // Reflex sources (reactive signal taps). `fs` is a list of watchers, so its
    // fact is the watcher count; `cron`/`sys` are single entries with a toggle.
    put("reflex.fs.watchers", config.reflex.fs.len().to_string());
    put(
        "reflex.cron.enabled",
        config.reflex.cron.enabled.to_string(),
    );
    put("reflex.sys.enabled", config.reflex.sys.enabled.to_string());

    // Monitored services.
    let mut services: Vec<&str> = config
        .monitoring
        .services
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    services.sort_unstable();
    put("monitoring.services", join_or_none(&services));

    // Security surface.
    let mut allow: Vec<&str> = config
        .security
        .exec_allowlist
        .iter()
        .map(String::as_str)
        .collect();
    allow.sort_unstable();
    put("security.exec_allowlist", join_or_none(&allow));
    put(
        "security.allowed_paths",
        config.security.allowed_paths.len().to_string(),
    );

    // Live native-capability inventory — one fact per capability, keyed by verb,
    // valued by its consent tier. A capability appearing/disappearing (e.g. when
    // an `actions.*` toggle flips) shows up as an added/removed fact in the diff.
    for c in inventory {
        put(
            &format!("capability.{}.{}", c.namespace, c.action),
            c.tier.clone(),
        );
    }

    f
}

fn adapter_fact(enabled: bool, port: u16) -> String {
    if enabled {
        format!("enabled port={port}")
    } else {
        "disabled".to_string()
    }
}

/// Render a sorted list as a comma-joined string, or `(none)` when empty, so an
/// empty list is still a stable, comparable fact value rather than a missing key.
fn join_or_none(items: &[&str]) -> String {
    if items.is_empty() {
        "(none)".to_string()
    } else {
        items.join(", ")
    }
}

// ─── diff ───────────────────────────────────────────────────────────────────

/// Compare two baselines and render the drift. With no versions, compares the
/// latest stored baseline against the *current* live state; `from` picks the
/// stored baseline to compare from; `to` compares against another stored
/// baseline instead of live state.
pub fn diff(
    config: &BrainConfig,
    inventory: &[CapabilitySummary],
    from: Option<u32>,
    to: Option<u32>,
) -> Result<String> {
    let dir = baseline_dir(config);
    let stored = list_baselines(&dir);
    if stored.is_empty() {
        anyhow::bail!(
            "no baselines stored in {} — capture one first with `brain baseline capture`",
            dir.display()
        );
    }

    // Resolve the "from" side: an explicit version, or the latest stored.
    let from_version = from.unwrap_or_else(|| stored.last().map(|b| b.version).unwrap());
    let base = load_baseline(&dir, from_version)?;

    // Resolve the "to" side: another stored baseline, or the current live state.
    let (target_facts, target_label) = match to {
        Some(v) => {
            let b = load_baseline(&dir, v)?;
            (
                b.facts,
                format!("baseline v{v} ({})", short_time(&b.captured_at)),
            )
        }
        None => (
            capture_facts(config, inventory),
            "current live state".to_string(),
        ),
    };

    let from_label = format!(
        "baseline v{} ({})",
        base.version,
        short_time(&base.captured_at)
    );
    let mut out = format!("Drift: {from_label}  →  {target_label}\n");
    if let Some(l) = &base.label {
        out.push_str(&format!("  from label: {l}\n"));
    }

    let drift = Drift::between(&base.facts, &target_facts);
    out.push_str(&drift.render());
    Ok(out)
}

/// The set difference between two fact maps.
struct Drift {
    added: Vec<(String, String)>,
    removed: Vec<(String, String)>,
    /// `(key, old, new)`.
    changed: Vec<(String, String, String)>,
}

impl Drift {
    fn between(from: &BTreeMap<String, String>, to: &BTreeMap<String, String>) -> Drift {
        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        for (k, new) in to {
            match from.get(k) {
                None => added.push((k.clone(), new.clone())),
                Some(old) if old != new => changed.push((k.clone(), old.clone(), new.clone())),
                Some(_) => {}
            }
        }
        for (k, old) in from {
            if !to.contains_key(k) {
                removed.push((k.clone(), old.clone()));
            }
        }
        Drift {
            added,
            removed,
            changed,
        }
    }

    fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    fn render(&self) -> String {
        if self.is_empty() {
            return "  No drift — every fact matches.\n".to_string();
        }
        let mut s = String::new();
        for (k, old, new) in &self.changed {
            s.push_str(&format!("  ~ {k}: {old} → {new}\n"));
        }
        for (k, v) in &self.added {
            s.push_str(&format!("  + {k}: {v}\n"));
        }
        for (k, v) in &self.removed {
            s.push_str(&format!("  - {k}: {v}\n"));
        }
        s.push_str(&format!(
            "\n  {} changed, {} added, {} removed\n",
            self.changed.len(),
            self.added.len(),
            self.removed.len(),
        ));
        s
    }
}

// ─── list ─────────────────────────────────────────────────────────────────

/// Render the stored baseline snapshots, newest first.
pub fn list(config: &BrainConfig) -> Result<String> {
    let dir = baseline_dir(config);
    let mut baselines = list_baselines(&dir);
    if baselines.is_empty() {
        return Ok(format!(
            "No baselines stored in {}. Capture one with `brain baseline capture`.",
            dir.display()
        ));
    }
    baselines.reverse(); // newest first
    let mut out = format!("Baselines in {}:\n", dir.display());
    for b in &baselines {
        let label = b
            .label
            .as_deref()
            .map(|l| format!(" — {l}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "  v{:<4} {}  ({} facts){}\n",
            b.version,
            short_time(&b.captured_at),
            b.facts.len(),
            label,
        ));
    }
    Ok(out)
}

// ─── storage helpers ──────────────────────────────────────────────────────────

fn baseline_dir(config: &BrainConfig) -> PathBuf {
    config.data_dir().join("baselines")
}

fn baseline_path(dir: &Path, version: u32) -> PathBuf {
    dir.join(format!("baseline-{version:04}.json"))
}

/// All stored baselines, ascending by version. Unreadable / malformed files are
/// skipped rather than aborting the whole command.
fn list_baselines(dir: &Path) -> Vec<Baseline> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<Baseline> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_baseline_file(p))
        .filter_map(|p| std::fs::read_to_string(&p).ok())
        .filter_map(|s| serde_json::from_str::<Baseline>(&s).ok())
        .collect();
    out.sort_by_key(|b| b.version);
    out
}

fn is_baseline_file(p: &Path) -> bool {
    p.is_file()
        && p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("baseline-") && n.ends_with(".json"))
}

fn load_baseline(dir: &Path, version: u32) -> Result<Baseline> {
    let path = baseline_path(dir, version);
    let content = std::fs::read_to_string(&path).with_context(|| {
        format!(
            "no baseline v{version} at {} — run `brain baseline list` to see stored versions",
            path.display()
        )
    })?;
    serde_json::from_str(&content)
        .with_context(|| format!("parsing baseline v{version} at {}", path.display()))
}

/// The next free version = one past the highest stored version (1 when empty).
fn next_version(dir: &Path) -> u32 {
    list_baselines(dir)
        .last()
        .map(|b| b.version + 1)
        .unwrap_or(1)
}

/// Trim an RFC3339 timestamp to `YYYY-MM-DD HH:MM` for compact display; falls
/// back to the raw string if it isn't the shape we expect.
fn short_time(rfc3339: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(rfc3339)
        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|_| rfc3339.to_string())
}

// ─── cortex backend wiring ──────────────────────────────────────────────────

/// The [`cortex::actions::BaselineBackend`] implementation. Holds a snapshot of
/// the config and the capability inventory taken at boot, and runs the
/// deterministic capture/diff/list on each dispatch. The chat tool-loop
/// dispatches `baseline.*` through this; the CLI calls the free functions
/// directly (assembling the inventory itself).
pub struct BaselineProvider {
    config: BrainConfig,
    inventory: Vec<CapabilitySummary>,
}

impl BaselineProvider {
    pub fn new(config: BrainConfig, inventory: Vec<CapabilitySummary>) -> Self {
        Self { config, inventory }
    }
}

#[async_trait::async_trait]
impl cortex::actions::BaselineBackend for BaselineProvider {
    async fn capture(&self, label: Option<&str>) -> Result<String, cortex::actions::ActionError> {
        capture(&self.config, &self.inventory, label).map_err(to_action_err)
    }

    async fn diff(
        &self,
        from: Option<u32>,
        to: Option<u32>,
    ) -> Result<String, cortex::actions::ActionError> {
        diff(&self.config, &self.inventory, from, to).map_err(to_action_err)
    }

    async fn list(&self) -> Result<String, cortex::actions::ActionError> {
        list(&self.config).map_err(to_action_err)
    }
}

fn to_action_err(e: anyhow::Error) -> cortex::actions::ActionError {
    cortex::actions::ActionError::ExecutionFailed(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BrainConfig {
        BrainConfig::default()
    }

    /// A minimal always-on inventory, mirroring the native set's stable members
    /// so the capability-fact assertions hold without depending on the CLI.
    fn test_inventory() -> Vec<CapabilitySummary> {
        vec![
            CapabilitySummary {
                namespace: "memory".into(),
                action: "store".into(),
                tier: "write".into(),
            },
            CapabilitySummary {
                namespace: "net".into(),
                action: "http".into(),
                tier: "external".into(),
            },
        ]
    }

    #[test]
    fn capture_is_deterministic_for_one_config() {
        let cfg = test_config();
        let inv = test_inventory();
        assert_eq!(capture_facts(&cfg, &inv), capture_facts(&cfg, &inv));
    }

    #[test]
    fn capture_records_core_identity_facts() {
        let facts = capture_facts(&test_config(), &test_inventory());
        assert_eq!(
            facts.get("brain.version").map(String::as_str),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(
            facts.get("os.arch").map(String::as_str),
            Some(std::env::consts::ARCH)
        );
        // Injected inventory shows up as capability facts with their tier.
        assert!(facts.contains_key("capability.memory.store"));
        assert_eq!(
            facts.get("capability.memory.store").map(String::as_str),
            Some("write")
        );
    }

    #[test]
    fn dropping_an_inventory_entry_drops_its_capability_fact() {
        let cfg = test_config();
        let with_web = capture_facts(&cfg, &test_inventory());
        // Inventory without the web capability (mirrors web_search disabled).
        let without_web = capture_facts(
            &cfg,
            &[CapabilitySummary {
                namespace: "memory".into(),
                action: "store".into(),
                tier: "write".into(),
            }],
        );

        assert!(with_web.contains_key("capability.net.http"));
        assert!(!without_web.contains_key("capability.net.http"));

        let drift = Drift::between(&with_web, &without_web);
        assert!(drift
            .removed
            .iter()
            .any(|(k, _)| k == "capability.net.http"));
    }

    #[test]
    fn drift_between_identical_maps_is_empty() {
        let facts = capture_facts(&test_config(), &test_inventory());
        let drift = Drift::between(&facts, &facts);
        assert!(drift.is_empty());
        assert!(drift.render().contains("No drift"));
    }

    #[test]
    fn drift_classifies_added_removed_changed() {
        let mut from = BTreeMap::new();
        from.insert("keep".to_string(), "same".to_string());
        from.insert("change".to_string(), "old".to_string());
        from.insert("gone".to_string(), "x".to_string());
        let mut to = BTreeMap::new();
        to.insert("keep".to_string(), "same".to_string());
        to.insert("change".to_string(), "new".to_string());
        to.insert("fresh".to_string(), "y".to_string());

        let drift = Drift::between(&from, &to);
        assert_eq!(drift.added, vec![("fresh".to_string(), "y".to_string())]);
        assert_eq!(drift.removed, vec![("gone".to_string(), "x".to_string())]);
        assert_eq!(
            drift.changed,
            vec![("change".to_string(), "old".to_string(), "new".to_string())]
        );
        assert!(!drift.is_empty());
    }

    #[test]
    fn next_version_increments_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        assert_eq!(next_version(p), 1);

        let b = Baseline {
            version: 1,
            captured_at: chrono::Utc::now().to_rfc3339(),
            label: Some("first".into()),
            facts: capture_facts(&test_config(), &test_inventory()),
        };
        std::fs::write(
            baseline_path(p, 1),
            serde_json::to_string_pretty(&b).unwrap(),
        )
        .unwrap();

        assert_eq!(next_version(p), 2);
        let loaded = load_baseline(p, 1).unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.label.as_deref(), Some("first"));
        assert_eq!(loaded.facts, b.facts);
    }

    #[test]
    fn list_baselines_ignores_non_baseline_files_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        for v in [3u32, 1, 2] {
            let b = Baseline {
                version: v,
                captured_at: chrono::Utc::now().to_rfc3339(),
                label: None,
                facts: BTreeMap::new(),
            };
            std::fs::write(baseline_path(p, v), serde_json::to_string(&b).unwrap()).unwrap();
        }
        std::fs::write(p.join("notes.txt"), "ignore me").unwrap();
        std::fs::write(p.join("baseline-bad.json"), "{not json").unwrap();

        let versions: Vec<u32> = list_baselines(p).iter().map(|b| b.version).collect();
        assert_eq!(versions, vec![1, 2, 3]);
    }

    #[test]
    fn empty_lists_render_as_stable_none_value() {
        // A fresh default has no manual delegates — the fact is "(none)", a
        // comparable value, not a missing key.
        let facts = capture_facts(&test_config(), &test_inventory());
        assert_eq!(
            facts.get("agents.delegates").map(String::as_str),
            Some("(none)")
        );
    }

    #[test]
    fn short_time_compacts_rfc3339() {
        assert_eq!(short_time("2026-06-09T10:11:12.345Z"), "2026-06-09 10:11");
        // Garbage falls back to the raw string.
        assert_eq!(short_time("not-a-time"), "not-a-time");
    }
}
