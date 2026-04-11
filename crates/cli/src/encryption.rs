//! Encryption helpers — salt management, encryptor resolution, API key lookup.

#[cfg(feature = "encryption")]
use std::io::IsTerminal;

use brain_core::BrainConfig;

#[cfg(feature = "encryption")]
pub(crate) fn salt_path(config: &BrainConfig) -> std::path::PathBuf {
    config.data_dir().join("db/salt")
}

#[cfg(feature = "encryption")]
pub(crate) fn load_salt(config: &BrainConfig) -> Option<[u8; 16]> {
    let bytes = std::fs::read(salt_path(config)).ok()?;
    if bytes.len() == 16 {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes);
        Some(arr)
    } else {
        None
    }
}

#[cfg(feature = "encryption")]
pub(crate) fn write_salt(config: &BrainConfig, salt: &[u8; 16]) -> anyhow::Result<()> {
    let path = salt_path(config);
    std::fs::write(&path, salt.as_slice())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Resolve the LLM API key from config, with env var fallback for backwards compatibility.
pub(crate) fn resolve_llm_api_key(config: &BrainConfig) -> String {
    let from_config = config.llm.api_key.trim().to_string();
    if !from_config.is_empty() {
        return from_config;
    }
    std::env::var("BRAIN_LLM__API_KEY").unwrap_or_default()
}

/// Build an `Encryptor` from config + passphrase, or `None` when encryption is disabled.
#[cfg(feature = "encryption")]
pub(crate) fn resolve_encryptor(
    config: &BrainConfig,
) -> anyhow::Result<Option<storage::Encryptor>> {
    if !config.encryption.enabled {
        return Ok(None);
    }

    let salt = load_salt(config).ok_or_else(|| {
        anyhow::anyhow!(
            "Encryption is enabled but no salt file found at {}.\n\
             Run `brain init --encrypt` to generate one.",
            salt_path(config).display()
        )
    })?;

    let passphrase = if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
        p
    } else if !std::io::stdin().is_terminal() {
        // Running non-interactively (e.g. spawned by an MCP client).
        // Cannot prompt — bail with a clear, single-line message.
        anyhow::bail!(
            "Encryption is enabled but BRAIN_PASSPHRASE is not set. \
             Set it in your MCP client's env config or in your shell profile."
        );
    } else {
        rpassword::prompt_password("Brain passphrase: ").map_err(|e| {
            anyhow::anyhow!(
                "Failed to read passphrase: {e}\n\
                 Hint: set the BRAIN_PASSPHRASE environment variable when running \
                 without a terminal (e.g. as an MCP server)."
            )
        })?
    };

    let encryptor = storage::Encryptor::from_passphrase(&passphrase, &salt)
        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;

    Ok(Some(encryptor))
}
