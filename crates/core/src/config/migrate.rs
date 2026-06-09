//! Config-file migration across binary versions.
//!
//! The user's `~/.brain/config.yaml` carries a `brain.version` stamp — the app
//! version that last wrote it. When a newer binary loads an older file, renamed
//! or moved keys would be silently lost: figment drops keys the struct no longer
//! has, and back-fills missing keys from the embedded defaults. The user's
//! intent (a value they set under the *old* name) vanishes.
//!
//! This module runs **before** [`BrainConfig::load`](super::BrainConfig::load),
//! operating on the raw YAML so renames survive into the typed struct. It:
//!
//! 1. reads the file's `brain.version` (the *from* version),
//! 2. applies every [`ConfigMigration`] introduced in `(from, binary]`,
//! 3. warns about keys the current schema no longer recognizes,
//! 4. snapshots the prior file to `config.yaml.bak-v<from>` and rewrites it
//!    with the new version stamp.
//!
//! An *older* binary reading a *newer* config needs no migration — figment is
//! already tolerant of unknown keys — so the engine only ever moves forward.

use std::path::Path;

use serde_yaml::Value;

use super::BrainConfig;

/// One forward config transform, keyed by the version that introduced it.
///
/// `apply` mutates the root mapping in place (rename/move/default-fill/remove)
/// and returns a human-readable line per change it made, for the migration log.
pub struct ConfigMigration {
    /// The binary version that first required this transform. Applied when the
    /// file's stamped version is below it and the running binary is at or above.
    pub introduced_in: &'static str,
    /// Short description of what the transform does (for diagnostics).
    pub description: &'static str,
    /// The transform. Returns one log line per concrete change applied.
    pub apply: fn(&mut serde_yaml::Mapping) -> Vec<String>,
}

/// Ordered registry of config migrations.
///
/// Empty today — no config field has been renamed or moved in the v0.x line, so
/// there is nothing to transform yet. The machinery (version-stamp bump,
/// backup, unknown-key warning) still runs. Append an entry here the moment a
/// field changes shape between versions; the engine applies it automatically.
pub(crate) const MIGRATIONS: &[ConfigMigration] = &[];

/// What a migration pass did, for the caller to report.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct MigrationOutcome {
    pub from_version: String,
    pub to_version: String,
    /// One line per applied transform (plus the version bump itself).
    pub changes: Vec<String>,
    /// Dotted paths present in the user file but absent from the current
    /// schema — likely typos or fields removed in a newer version.
    pub unknown_keys: Vec<String>,
    /// The path the snapshot was written to, if the file was rewritten.
    pub backup_path: Option<std::path::PathBuf>,
}

impl BrainConfig {
    /// Migrate the user config file at the default path forward to this binary's
    /// version, if it is stamped older. Returns `Ok(None)` when there is nothing
    /// to do (no file, unparseable, or already current). Errors only on a failed
    /// backup or rewrite — a half-migrated file must never be left on disk.
    pub fn migrate_user_config_if_needed() -> std::io::Result<Option<MigrationOutcome>> {
        Self::migrate_config_at(&Self::user_config_path())
    }

    /// Path-taking core of [`migrate_user_config_if_needed`]. Separated so tests
    /// can drive the full read → backup → rewrite round-trip against a temp file
    /// without mutating the process-global `BRAIN_CONFIG` env var.
    pub(crate) fn migrate_config_at(path: &Path) -> std::io::Result<Option<MigrationOutcome>> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path)?;
        // A malformed file is not this layer's problem to report — the normal
        // load path surfaces the parse error with full context. Skip quietly.
        let Ok(value) = serde_yaml::from_str::<Value>(&raw) else {
            return Ok(None);
        };

        let from = config_version(&value).unwrap_or_else(|| "0.0.0".to_string());
        let to = env!("CARGO_PKG_VERSION").to_string();

        if !is_older(&from, &to) {
            return Ok(None);
        }

        // Parse into an owned mapping for the transforms to mutate.
        let Value::Mapping(mut root) = value else {
            return Ok(None);
        };

        let mut changes = apply_migrations(&mut root, &from, &to, MIGRATIONS);
        changes.push(format!("stamped brain.version {from} → {to}"));

        let unknown_keys = unknown_keys_against_defaults(&Value::Mapping(root.clone()));

        // Snapshot the prior file before overwriting it.
        let backup_path = backup_sibling(path, &from);
        std::fs::copy(path, &backup_path)?;

        let rewritten = serde_yaml::to_string(&Value::Mapping(root))
            .map_err(|e| std::io::Error::other(format!("serialize migrated config: {e}")))?;
        std::fs::write(path, rewritten)?;

        Ok(Some(MigrationOutcome {
            from_version: from,
            to_version: to,
            changes,
            unknown_keys,
            backup_path: Some(backup_path),
        }))
    }
}

/// `config.yaml` → `config.yaml.bak-v<from>` next to it.
fn backup_sibling(path: &Path, from: &str) -> std::path::PathBuf {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("config.yaml");
    path.with_file_name(format!("{name}.bak-v{from}"))
}

/// Read `brain.version` from a parsed config value.
fn config_version(value: &Value) -> Option<String> {
    value
        .get("brain")
        .and_then(|b| b.get("version"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Apply every migration whose `introduced_in` is in `(from, to]`, then stamp
/// `brain.version = to`. Pure over the mapping; no filesystem access. Returns
/// the concatenated change log from each applied transform.
fn apply_migrations(
    root: &mut serde_yaml::Mapping,
    from: &str,
    to: &str,
    migrations: &[ConfigMigration],
) -> Vec<String> {
    let mut log = Vec::new();
    for m in migrations {
        // (from, to]: strictly newer than the file, at or below the binary.
        if is_older(from, m.introduced_in) && !is_older(to, m.introduced_in) {
            log.extend((m.apply)(root));
        }
    }
    stamp_version(root, to);
    log
}

/// Set `brain.version` to `to`, creating the `brain` mapping if absent.
fn stamp_version(root: &mut serde_yaml::Mapping, to: &str) {
    let brain = root
        .entry(Value::String("brain".into()))
        .or_insert_with(|| Value::Mapping(serde_yaml::Mapping::new()));
    if let Value::Mapping(b) = brain {
        b.insert(
            Value::String("version".into()),
            Value::String(to.to_string()),
        );
    }
}

/// Parse a dotted `major.minor.patch` version into a comparable tuple. Missing
/// or non-numeric components read as 0, so `"0.4"` and `"0.4.0"` compare equal
/// and a junk stamp sorts oldest.
fn parse_semver(v: &str) -> (u64, u64, u64) {
    // Drop any pre-release/build suffix (`-rc1`, `+meta`).
    let core = v.split(['-', '+']).next().unwrap_or(v);
    let mut it = core
        .split('.')
        .map(|p| p.trim().parse::<u64>().unwrap_or(0));
    (
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
    )
}

/// `a` is strictly an older version than `b`.
fn is_older(a: &str, b: &str) -> bool {
    parse_semver(a) < parse_semver(b)
}

/// Best-effort unknown-key detection: dotted paths present in `user` whose key
/// is absent from the embedded default config tree. Only recurses through
/// mappings — sequence elements (e.g. provider entries) are not walked, since
/// their inner keys are list-item shapes, not top-level schema keys.
fn unknown_keys_against_defaults(user: &Value) -> Vec<String> {
    let reference: Value = serde_yaml::from_str(super::DEFAULT_CONFIG)
        .expect("embedded default.yaml must parse as YAML");
    let mut out = Vec::new();
    diff_keys(user, &reference, String::new(), &mut out);
    out
}

fn diff_keys(user: &Value, reference: &Value, prefix: String, out: &mut Vec<String>) {
    let (Value::Mapping(u), Value::Mapping(r)) = (user, reference) else {
        return;
    };
    for (k, uv) in u {
        let Some(key) = k.as_str() else { continue };
        let path = if prefix.is_empty() {
            key.to_string()
        } else {
            format!("{prefix}.{key}")
        };
        match r.get(k) {
            None => out.push(path),
            Some(rv) => diff_keys(uv, rv, path, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_from(yaml: &str) -> serde_yaml::Mapping {
        match serde_yaml::from_str::<Value>(yaml).unwrap() {
            Value::Mapping(m) => m,
            _ => panic!("expected a mapping"),
        }
    }

    #[test]
    fn semver_ordering_is_numeric_not_lexical() {
        assert!(is_older("0.9.0", "0.10.0"), "0.9 < 0.10 numerically");
        assert!(is_older("0.4.0", "0.4.1"));
        assert!(!is_older("0.4.0", "0.4.0"));
        assert!(!is_older("1.0.0", "0.9.9"));
        // Short and suffixed forms normalise.
        assert!(!is_older("0.4", "0.4.0"));
        assert_eq!(parse_semver("0.5.0-rc1"), (0, 5, 0));
        // Junk stamp sorts oldest.
        assert!(is_older("garbage", "0.0.1"));
    }

    #[test]
    fn apply_migrations_runs_only_the_in_range_window() {
        // Three synthetic migrations; the file is at 0.4.0, binary at 0.6.0, so
        // only 0.5.0 and 0.6.0 should fire — not the already-applied 0.4.0.
        fn tag(root: &mut serde_yaml::Mapping, key: &str) -> Vec<String> {
            root.insert(Value::String(key.into()), Value::Bool(true));
            vec![format!("set {key}")]
        }
        const MS: &[ConfigMigration] = &[
            ConfigMigration {
                introduced_in: "0.4.0",
                description: "already applied",
                apply: |r| tag(r, "m040"),
            },
            ConfigMigration {
                introduced_in: "0.5.0",
                description: "in range",
                apply: |r| tag(r, "m050"),
            },
            ConfigMigration {
                introduced_in: "0.6.0",
                description: "in range (== binary)",
                apply: |r| tag(r, "m060"),
            },
        ];

        let mut root = map_from("brain:\n  version: \"0.4.0\"\n");
        let log = apply_migrations(&mut root, "0.4.0", "0.6.0", MS);

        assert!(
            !root.contains_key(Value::String("m040".into())),
            "0.4.0 already in the file"
        );
        assert!(
            root.contains_key(Value::String("m050".into())),
            "0.5.0 in (0.4.0, 0.6.0]"
        );
        assert!(
            root.contains_key(Value::String("m060".into())),
            "0.6.0 in (0.4.0, 0.6.0]"
        );
        assert_eq!(log.len(), 2, "two transforms fired");
        // Version stamp advanced to the binary version.
        assert_eq!(
            config_version(&Value::Mapping(root)).as_deref(),
            Some("0.6.0")
        );
    }

    #[test]
    fn migration_renames_a_field_without_losing_intent() {
        // A representative rename transform: old.key -> new.key.
        const RENAME: &[ConfigMigration] = &[ConfigMigration {
            introduced_in: "0.5.0",
            description: "rename legacy.timeout -> network.timeout",
            apply: |root| {
                let legacy = root.remove(Value::String("legacy".into()));
                if let Some(Value::Mapping(mut l)) = legacy {
                    if let Some(t) = l.remove(Value::String("timeout".into())) {
                        let net = root
                            .entry(Value::String("network".into()))
                            .or_insert_with(|| Value::Mapping(serde_yaml::Mapping::new()));
                        if let Value::Mapping(n) = net {
                            n.insert(Value::String("timeout".into()), t);
                        }
                        return vec!["moved legacy.timeout → network.timeout".into()];
                    }
                }
                vec![]
            },
        }];

        let mut root = map_from("brain:\n  version: \"0.4.0\"\nlegacy:\n  timeout: 42\n");
        apply_migrations(&mut root, "0.4.0", "0.5.0", RENAME);

        // The user's value survived under the new path; the old path is gone.
        assert!(!root.contains_key(Value::String("legacy".into())));
        let net = root.get(Value::String("network".into())).unwrap();
        assert_eq!(net.get("timeout").and_then(Value::as_i64), Some(42));
    }

    /// A unique scratch path under the system temp dir (no tempfile dev-dep).
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "brain-migrate-{tag}-{}-{nanos}.yaml",
            std::process::id()
        ))
    }

    #[test]
    fn migrate_config_at_is_noop_when_file_absent() {
        let path = scratch_path("absent");
        assert!(BrainConfig::migrate_config_at(&path).unwrap().is_none());
    }

    #[test]
    fn migrate_config_at_round_trips_file_and_snapshots() {
        // A real-shaped config stamped at an ancient version, with a user value
        // we must not lose and an unrecognized key we should flag.
        let path = scratch_path("roundtrip");
        std::fs::write(
            &path,
            "brain:\n  version: \"0.0.1\"\n  data_dir: \"~/.brain\"\n  bogus_key: 7\n",
        )
        .unwrap();

        let outcome = BrainConfig::migrate_config_at(&path)
            .unwrap()
            .expect("an older file must migrate");

        // Version advanced to this binary; a snapshot of the old file exists.
        assert_eq!(outcome.from_version, "0.0.1");
        assert_eq!(outcome.to_version, env!("CARGO_PKG_VERSION"));
        let backup = outcome.backup_path.clone().unwrap();
        assert!(backup.exists(), "expected snapshot at {}", backup.display());
        assert!(backup.to_string_lossy().contains(".bak-v0.0.1"));

        // The rewritten file is stamped current and the user value survived.
        let rewritten = std::fs::read_to_string(&path).unwrap();
        let value: Value = serde_yaml::from_str(&rewritten).unwrap();
        assert_eq!(
            config_version(&value).as_deref(),
            Some(env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(
            value
                .get("brain")
                .and_then(|b| b.get("data_dir"))
                .and_then(Value::as_str),
            Some("~/.brain")
        );
        assert!(outcome
            .unknown_keys
            .contains(&"brain.bogus_key".to_string()));

        // Re-running is now a no-op (already at current version).
        assert!(BrainConfig::migrate_config_at(&path).unwrap().is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&backup);
    }

    #[test]
    fn unknown_keys_flags_only_unrecognized_paths() {
        // `brain.data_dir` is real; `brain.bogus` and top-level `nonsense` are not.
        let user = serde_yaml::from_str::<Value>(
            "brain:\n  data_dir: \"~/.brain\"\n  bogus: 1\nnonsense:\n  x: 2\n",
        )
        .unwrap();
        let unknown = unknown_keys_against_defaults(&user);
        assert!(
            unknown.contains(&"brain.bogus".to_string()),
            "got {unknown:?}"
        );
        assert!(unknown.contains(&"nonsense".to_string()), "got {unknown:?}");
        assert!(
            !unknown.contains(&"brain.data_dir".to_string()),
            "data_dir is valid"
        );
    }
}
