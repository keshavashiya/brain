//! `brain init` — write default config, create data subdirs, optionally seal
//! the blood-brain barrier (encryption-at-rest).

use crate::doctor::check_ollama_models;
#[cfg(feature = "encryption")]
use crate::encryption;
use brain::BrainConfig;

pub(crate) async fn cmd_init(
    config: &BrainConfig,
    force: bool,
    #[cfg(feature = "encryption")] encrypt: bool,
) -> anyhow::Result<()> {
    let data_dir = config.data_dir();
    println!("Forming neural pathways...");
    println!("  Cortex (data dir):  {}", data_dir.display());

    let generated_key = match BrainConfig::write_default_config(force)? {
        Some((path, key)) => {
            println!("  Genome (config):    {} (written)", path.display());
            // The embedded default ships `quiet_hours.timezone: "UTC"`, which is
            // wrong for almost everyone. Detect the host zone and bake it in so
            // proactive quiet-hours line up with the wall clock out of the box.
            match apply_detected_timezone(&path) {
                Some(tz) => println!("  Timezone:           {tz} (detected)"),
                None => println!(
                    "  Timezone:           UTC (could not detect host zone — edit if wrong)"
                ),
            }
            Some(key)
        }
        None => {
            println!(
                "  Genome (config):    {} (exists, --force to overwrite)",
                BrainConfig::user_config_path().display()
            );
            None
        }
    };

    let subdirs = ["db", "ruvector", "models", "logs", "exports"];
    for sub in &subdirs {
        println!("  Region:             {}", data_dir.join(sub).display());
    }

    // Hardware-aware model recommendation: probe the host once and say what
    // local model size actually fits, so the first config edit is informed.
    let host = selfmodel::HostModel::probe(Some(&data_dir));
    println!("  Hardware:           {}", host.summary_line());
    if let Some(rec) = host.local_model_recommendation() {
        println!(
            "                      class: {} — local models {} recommended",
            host.machine_class(),
            rec
        );
    }

    // Probe Ollama and only warn about the embedding model when it's
    // actually missing.
    check_ollama_models(config).await;

    #[cfg(feature = "encryption")]
    if encrypt {
        let salt = storage::Encryptor::generate_salt();
        encryption::write_salt(config, &salt)?;

        // Silently failing to flip `enabled: false` → `enabled: true` would
        // leave encryption off after we told the user the barrier is sealed.
        let config_path = BrainConfig::user_config_path();
        let yaml = std::fs::read_to_string(&config_path).map_err(|e| {
            anyhow::anyhow!(
                "Salt written, but failed to read {} to enable encryption: {e}",
                config_path.display()
            )
        })?;
        // Scope the flip to the `encryption:` block. `enabled: false` appears
        // in several sections, so match the section header too.
        let patched = yaml.replace(
            "encryption:\n  enabled: false",
            "encryption:\n  enabled: true                  # Activated by `brain init --encrypt`",
        );
        if patched == yaml {
            anyhow::bail!(
                "Salt written, but `encryption.enabled: false` line not found in {}. \
                 Set `encryption.enabled: true` manually before starting Brain.",
                config_path.display()
            );
        }
        std::fs::write(&config_path, &patched).map_err(|e| {
            anyhow::anyhow!(
                "Salt written, but failed to write {} to enable encryption: {e}",
                config_path.display()
            )
        })?;

        println!(
            "\n  Blood-brain barrier: sealed (salt → {})",
            encryption::salt_path(config).display()
        );
        println!("  Set BRAIN_PASSPHRASE env var for the daemon, or");
        println!("  Brain will prompt you for a passphrase on startup.");
    }

    if let Some(key) = generated_key {
        println!("\n  API key:   {}", key);
        println!("  Use this key for HTTP/WS/MCP authentication.");
    }

    println!(
        "\nNeural pathways formed. Edit {} to customize your genome.",
        BrainConfig::user_config_path().display()
    );

    Ok(())
}

/// Patch the freshly-written config's `quiet_hours.timezone` from the default
/// `"UTC"` to the host's IANA zone, when we can detect one. Returns the zone we
/// wrote, or `None` if detection failed or the file couldn't be patched (in
/// which case the safe `"UTC"` default is left untouched).
fn apply_detected_timezone(config_path: &std::path::Path) -> Option<String> {
    let tz = detect_system_timezone()?;
    let yaml = std::fs::read_to_string(config_path).ok()?;
    // Single occurrence in the embedded default; replace just the value and
    // leave the trailing comment in place.
    let patched = yaml.replacen("timezone: \"UTC\"", &format!("timezone: \"{tz}\""), 1);
    if patched == yaml {
        return None;
    }
    std::fs::write(config_path, patched).ok()?;
    Some(tz)
}

/// Best-effort host IANA timezone via the `/etc/localtime` symlink (the
/// portable trick on macOS and Linux). The target looks like
/// `…/zoneinfo/Asia/Kolkata`; we take everything after `zoneinfo/` and accept
/// it only if it parses as a real `chrono_tz::Tz`, so we never write garbage.
fn detect_system_timezone() -> Option<String> {
    let target = std::fs::read_link("/etc/localtime").ok()?;
    let target = target.to_str()?;
    let zone = target.rsplit_once("/zoneinfo/").map(|(_, z)| z)?;
    if zone.is_empty() || zone == "UTC" {
        return None;
    }
    zone.parse::<chrono_tz::Tz>().ok().map(|_| zone.to_string())
}
