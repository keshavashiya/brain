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

    // Probe Ollama and only warn about the embedding model when it's
    // actually missing.
    check_ollama_models(config).await;

    #[cfg(feature = "encryption")]
    if encrypt {
        let salt = storage::Encryptor::generate_salt();
        encryption::write_salt(config, &salt)?;

        let config_path = BrainConfig::user_config_path();
        if let Ok(yaml) = std::fs::read_to_string(&config_path) {
            let patched = yaml.replace(
                "enabled: false               # Run `brain init --encrypt` to generate a salt and enable",
                "enabled: true                # Activated by `brain init --encrypt`",
            );
            let _ = std::fs::write(&config_path, patched);
        }

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
