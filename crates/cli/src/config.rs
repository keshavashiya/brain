//! `brain config` — inspect, validate, and discover the runtime config.
//!
//! Three actions wrap [`brain::BrainConfig`] surface that's otherwise only
//! reachable through the daemon's startup path:
//!
//! - `validate [--file PATH]` runs the same semantic checks the daemon runs
//!   at boot, so operators can dry-run a config change before installing it.
//! - `show [--defaults]` prints the resolved effective config (or the
//!   embedded defaults), useful for debugging the four-layer Figment merge.
//! - `path` prints the resolved user config path (`BRAIN_CONFIG` aware).

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum ConfigAction {
    /// Validate the configuration without starting the daemon.
    ///
    /// Runs the same checks `brain start` / `brain serve` perform at boot:
    /// port conflicts, LLM base_url scheme, data-dir writability, API key
    /// presence, web-search / messaging wiring, and resilience bounds. With
    /// no `--file`, validates the resolved layered config (defaults + user
    /// file + env). With `--file PATH`, validates that file in isolation
    /// over the embedded defaults — useful before `mv newconfig.yaml ~/.brain/config.yaml`.
    ///
    /// Exits non-zero on hard errors so it's safe to use in scripts. Warnings
    /// print to stderr but do not change the exit code.
    Validate {
        /// Validate a specific YAML file instead of the resolved user config.
        #[arg(long, short)]
        file: Option<PathBuf>,
    },
    /// Print the resolved effective configuration as YAML.
    ///
    /// Defaults to the layered config the daemon would see (embedded defaults
    /// → `~/.brain/config.yaml` → `BRAIN_*` env vars). With `--defaults`,
    /// prints the embedded default config — the schema-by-example reference.
    Show {
        /// Print the embedded default config instead of the resolved layered one.
        #[arg(long)]
        defaults: bool,
    },
    /// Print the user config path (`BRAIN_CONFIG` env-var aware).
    Path,
}

pub(crate) fn cmd_config(config: &brain::BrainConfig, action: ConfigAction) -> anyhow::Result<()> {
    match action {
        ConfigAction::Validate { file } => cmd_validate(config, file.as_deref()),
        ConfigAction::Show { defaults } => cmd_show(config, defaults),
        ConfigAction::Path => {
            println!("{}", brain::BrainConfig::user_config_path().display());
            Ok(())
        }
    }
}

fn cmd_validate(config: &brain::BrainConfig, file: Option<&std::path::Path>) -> anyhow::Result<()> {
    let (target, source_label) = match file {
        Some(path) => {
            if !path.exists() {
                anyhow::bail!("Config file not found: {}", path.display());
            }
            let cfg = brain::BrainConfig::load_from(Some(path))
                .map_err(|e| anyhow::anyhow!("failed to load {}: {e}", path.display()))?;
            (cfg, format!("{}", path.display()))
        }
        None => (config.clone(), "resolved config".to_string()),
    };

    match target.validate() {
        Err(hard_err) => {
            eprintln!("ERROR: {source_label}: {hard_err}");
            anyhow::bail!("Configuration is invalid");
        }
        Ok(warnings) => {
            for w in &warnings {
                eprintln!("WARNING: {w}");
            }
            if warnings.is_empty() {
                println!("OK: {source_label} is valid (no warnings)");
            } else {
                println!(
                    "OK: {source_label} is valid ({} warning{})",
                    warnings.len(),
                    if warnings.len() == 1 { "" } else { "s" }
                );
            }
            Ok(())
        }
    }
}

fn cmd_show(config: &brain::BrainConfig, defaults: bool) -> anyhow::Result<()> {
    if defaults {
        print!("{}", brain::BrainConfig::default_config_content());
        return Ok(());
    }

    let yaml = serde_yaml::to_string(config)
        .map_err(|e| anyhow::anyhow!("failed to serialise config: {e}"))?;
    print!("{yaml}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn validate_default_config_succeeds() {
        let config = brain::BrainConfig::default();
        // Default has an auto-generated API key, so validate succeeds.
        let warnings = config.validate().expect("default must validate");
        // proactivity is enabled by default; web_search DDG is the zero-config
        // built-in; nothing should warn on the default shape.
        for w in &warnings {
            // surface unexpected default-shape warnings for debugging.
            eprintln!("unexpected default warning: {w}");
        }
    }

    #[test]
    fn validate_rejects_missing_file() {
        let config = brain::BrainConfig::default();
        let result = cmd_validate(
            &config,
            Some(std::path::Path::new("/nonexistent/brain.yaml")),
        );
        let err = result.expect_err("should fail on missing file");
        assert!(
            err.to_string().contains("not found"),
            "error should mention the file: {err}",
        );
    }

    #[test]
    fn validate_accepts_default_yaml_file() {
        let mut tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        tmp.write_all(brain::BrainConfig::default_config_content().as_bytes())
            .expect("write default config");
        // default_config_content() ships `api_keys: []`, which makes
        // validate() fail (no API key configured). Write_default_config
        // injects a generated key — we mimic that here for the test.
        let with_key = brain::BrainConfig::default_config_content()
            .replace(
                "api_keys: []",
                "api_keys:\n    - key: \"brk_test_validate_yaml\"\n      name: \"test\"\n      permissions: [read, write]",
            );
        let mut tmp2 = tempfile::NamedTempFile::new().expect("create tempfile");
        tmp2.write_all(with_key.as_bytes()).expect("write");

        let config = brain::BrainConfig::default();
        cmd_validate(&config, Some(tmp2.path())).expect("should validate");
    }

    #[test]
    fn show_defaults_round_trips() {
        let yaml = brain::BrainConfig::default_config_content();
        // The schema-by-example must remain parseable into a BrainConfig.
        let _: brain::BrainConfig =
            serde_yaml::from_str(yaml).expect("default config must round-trip");
    }

    #[test]
    fn show_resolved_serialises() {
        let config = brain::BrainConfig::default();
        let yaml = serde_yaml::to_string(&config).expect("serialise");
        assert!(yaml.contains("brain:"));
        assert!(yaml.contains("adapters:"));
    }
}
