//! Encryption helpers — salt management, encryptor resolution, API key lookup.

#[cfg(feature = "encryption")]
use std::io::IsTerminal;

use brain::BrainConfig;

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
///
/// An empty/whitespace-only `BRAIN_LLM__API_KEY` is treated as a user error
/// (e.g. unset-but-exported in a shell profile) and reported up the stack
/// rather than silently producing an empty key the LLM call will reject.
pub(crate) fn resolve_llm_api_key(config: &BrainConfig) -> anyhow::Result<String> {
    let from_config = config.llm.api_key.trim().to_string();
    if !from_config.is_empty() {
        return Ok(from_config);
    }
    match std::env::var("BRAIN_LLM__API_KEY") {
        Ok(v) if v.trim().is_empty() => Err(anyhow::anyhow!(
            "BRAIN_LLM__API_KEY is set but empty — unset it or provide a real key"
        )),
        Ok(v) => Ok(v),
        Err(_) => Ok(String::new()),
    }
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

/// Resolve the daemon passphrase before spawning the background daemon.
///
/// `brain start` runs in the foreground, so prompting on the TTY is fine.
/// The returned passphrase is forwarded into the child via `BRAIN_PASSPHRASE`.
/// Returns `Ok(None)` when encryption is disabled at the config level.
#[cfg(feature = "encryption")]
pub(crate) fn resolve_start_passphrase(config: &BrainConfig) -> anyhow::Result<Option<String>> {
    if !config.encryption.enabled {
        return Ok(None);
    }

    if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
        return Ok(Some(p));
    }

    let salt = load_salt(config).ok_or_else(|| {
        anyhow::anyhow!(
            "Encryption is enabled but no salt file found.\n\
             Run `brain init --encrypt` to generate one."
        )
    })?;
    let p = rpassword::prompt_password("Brain passphrase: ")
        .map_err(|e| anyhow::anyhow!("Failed to read passphrase: {e}"))?;
    storage::Encryptor::from_passphrase(&p, &salt)
        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;
    Ok(Some(p))
}
