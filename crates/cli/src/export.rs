//! Export / import commands — backup and restore memory data.
//!
//! All operations go through the running daemon's HTTP API to ensure
//! a single shared SignalProcessor (no RuVector lock contention).
//!
//! Exports can be sealed in a self-contained passphrase envelope
//! (AES-256-GCM, Argon2id-derived key, a fresh per-export salt embedded
//! in the file — so a backup is decryptable on any machine, not just
//! this install). When encryption at rest is enabled, sealing is the
//! *default*: writing plaintext requires the explicit `--plaintext`
//! flag.

use std::time::Duration;

/// JSON envelope written / read by `brain export` / `brain import`.
#[derive(serde::Serialize, serde::Deserialize)]
struct MemoryExport {
    version: String,
    exported_at: String,
    /// Namespaces whose residency policy was `local_only` at export
    /// time. Facts/episodes in them (or their sub-namespaces) have
    /// never been sent off-machine — handle this file accordingly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    local_only_namespaces: Vec<String>,
    facts: Vec<signal::ExportedFact>,
    episodes: Vec<signal::ExportedEpisode>,
}

/// Sealed-export envelope. Self-contained: the KDF salt rides inside,
/// so only the passphrase is needed to open it anywhere.
#[cfg(feature = "encryption")]
#[derive(serde::Serialize, serde::Deserialize)]
struct SealedExport {
    /// Format marker + version — how `brain import` recognizes an
    /// encrypted file.
    brain_export_sealed: u32,
    kdf: String,
    cipher: String,
    /// Hex-encoded Argon2id salt, freshly generated per export (never
    /// the at-rest database salt).
    salt: String,
    /// Base64-encoded nonce-prefixed AES-256-GCM ciphertext of the
    /// plaintext `MemoryExport` JSON.
    ciphertext: String,
}

/// Seal plaintext export JSON under a passphrase.
#[cfg(feature = "encryption")]
fn seal_export(plaintext_json: &str, passphrase: &str) -> anyhow::Result<String> {
    let salt = storage::Encryptor::generate_salt();
    let enc = storage::Encryptor::from_passphrase(passphrase, &salt)
        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;
    let ciphertext = enc
        .encrypt_string(plaintext_json)
        .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))?;
    let sealed = SealedExport {
        brain_export_sealed: 1,
        kdf: "argon2id".into(),
        cipher: "aes-256-gcm".into(),
        salt: salt.iter().map(|b| format!("{b:02x}")).collect(),
        ciphertext,
    };
    Ok(serde_json::to_string_pretty(&sealed)?)
}

/// True when `raw` is a sealed export envelope.
#[cfg(feature = "encryption")]
fn is_sealed_export(raw: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .is_some_and(|v| v.get("brain_export_sealed").is_some())
}

/// Open a sealed export envelope back into plaintext JSON.
#[cfg(feature = "encryption")]
fn open_export(raw: &str, passphrase: &str) -> anyhow::Result<String> {
    let sealed: SealedExport =
        serde_json::from_str(raw).map_err(|e| anyhow::anyhow!("Invalid sealed export: {e}"))?;
    if sealed.brain_export_sealed != 1 {
        anyhow::bail!(
            "Unsupported sealed-export version {} — update brain",
            sealed.brain_export_sealed
        );
    }
    if !sealed.salt.len().is_multiple_of(2) {
        anyhow::bail!("Invalid sealed export: malformed salt");
    }
    let salt: Vec<u8> = (0..sealed.salt.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&sealed.salt[i..i + 2], 16))
        .collect::<Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("Invalid sealed export: malformed salt"))?;
    let enc = storage::Encryptor::from_passphrase(passphrase, &salt)
        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;
    enc.decrypt_string(&sealed.ciphertext)
        .map_err(|_| anyhow::anyhow!("Decryption failed — wrong passphrase or corrupted file"))
}

/// Resolve the passphrase for sealing a new export: `BRAIN_PASSPHRASE`
/// if set, otherwise prompt twice (a typo here means an unrecoverable
/// backup, so new envelopes get a confirmation prompt).
#[cfg(feature = "encryption")]
fn export_passphrase_new() -> anyhow::Result<String> {
    if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
        return Ok(p);
    }
    let p = rpassword::prompt_password("Export passphrase: ")?;
    if p.is_empty() {
        anyhow::bail!("Empty passphrase — aborting (use --plaintext for an unencrypted export)");
    }
    let confirm = rpassword::prompt_password("Confirm passphrase: ")?;
    if p != confirm {
        anyhow::bail!("Passphrases do not match — aborting");
    }
    Ok(p)
}

/// Resolve the passphrase for opening an existing sealed export.
#[cfg(feature = "encryption")]
fn export_passphrase_existing() -> anyhow::Result<String> {
    if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
        return Ok(p);
    }
    Ok(rpassword::prompt_password("Export passphrase: ")?)
}

pub(crate) async fn cmd_export(
    config: &brain::BrainConfig,
    output: Option<&str>,
    encrypt: bool,
    plaintext: bool,
) -> anyhow::Result<()> {
    // Sealing policy: explicit flags win; with neither, encryption at
    // rest makes sealed the default — plaintext only by explicit opt-out.
    let seal = encrypt || (config.encryption.enabled && !plaintext);
    #[cfg(not(feature = "encryption"))]
    if seal {
        anyhow::bail!(
            "Encryption at rest is enabled, so exports are sealed by default — but this \
             build has no encryption support. Pass --plaintext to export unencrypted."
        );
    }
    let daemon_url = crate::bootstrap::require_daemon(config).await?;

    let api_key = config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let resp = client
        .get(format!("{daemon_url}/v1/memory/export"))
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let facts: Vec<signal::ExportedFact> = serde_json::from_value(resp["facts"].clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse facts: {e}"))?;
    let episodes: Vec<signal::ExportedEpisode> =
        serde_json::from_value(resp["episodes"].clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse episodes: {e}"))?;

    let n_facts = facts.len();
    let n_episodes = episodes.len();

    let export = MemoryExport {
        version: resp["version"].as_str().unwrap_or("unknown").to_string(),
        exported_at: resp["exported_at"].as_str().unwrap_or("").to_string(),
        local_only_namespaces: config
            .memory
            .local_only_namespaces()
            .into_iter()
            .map(String::from)
            .collect(),
        facts,
        episodes,
    };

    let json = serde_json::to_string_pretty(&export)?;

    #[cfg(feature = "encryption")]
    let (json, sealed_note) = if seal {
        let passphrase = export_passphrase_new()?;
        (seal_export(&json, &passphrase)?, " (sealed)")
    } else {
        (json, "")
    };
    #[cfg(not(feature = "encryption"))]
    let sealed_note = "";

    match output {
        Some(path) => {
            std::fs::write(path, &json)?;
            println!("Exported {n_facts} facts and {n_episodes} episodes to {path}{sealed_note}");
        }
        None => {
            println!("{}", json);
        }
    }

    Ok(())
}

pub(crate) async fn cmd_import(
    config: &brain::BrainConfig,
    file: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    let daemon_url = crate::bootstrap::require_daemon(config).await?;

    let raw =
        std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("Cannot read {file}: {e}"))?;
    // Sealed envelopes are recognized by their format marker and opened
    // with the export passphrase before the normal parse.
    #[cfg(feature = "encryption")]
    let raw = if is_sealed_export(&raw) {
        let passphrase = export_passphrase_existing()?;
        open_export(&raw, &passphrase)?
    } else {
        raw
    };
    #[cfg(not(feature = "encryption"))]
    if raw.contains("\"brain_export_sealed\"") {
        anyhow::bail!(
            "{file} is a sealed (encrypted) export, but this build has no encryption support."
        );
    }
    let export: MemoryExport =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("Invalid export file: {e}"))?;

    println!(
        "Import preview: {} facts, {} episodes (exported at {})",
        export.facts.len(),
        export.episodes.len(),
        export.exported_at,
    );

    if dry_run {
        println!("Dry-run: no changes written.");
        return Ok(());
    }

    let api_key = config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let import_body = serde_json::json!({
        "facts": export.facts,
        "episodes": export.episodes,
        "dry_run": false,
    });

    let resp = client
        .post(format!("{daemon_url}/v1/memory/import"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&import_body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let facts_imported = resp["facts_imported"].as_u64().unwrap_or(0) as usize;
    let episodes_imported = resp["episodes_imported"].as_u64().unwrap_or(0) as usize;
    let facts_existed = resp["facts_already_existed"].as_u64().unwrap_or(0) as usize;
    let episodes_existed = resp["episodes_already_existed"].as_u64().unwrap_or(0) as usize;
    let embedded = resp["embedded"].as_u64().unwrap_or(0) as usize;
    let embed_failed = resp["embed_failed"].as_u64().unwrap_or(0) as usize;

    println!(
        "Imported: {} new facts, {} new episodes ({} facts and {} episodes already existed).",
        facts_imported, episodes_imported, facts_existed, episodes_existed,
    );

    if embedded > 0 {
        println!("Re-embedded {embedded} facts into vector index.");
    }
    if embed_failed > 0 {
        println!("Warning: {embed_failed} facts failed to embed.");
    }

    Ok(())
}

#[cfg(all(test, feature = "encryption"))]
mod seal_tests {
    use super::*;

    const EXPORT_JSON: &str = r#"{
        "version": "1",
        "exported_at": "2026-06-11T00:00:00Z",
        "facts": [],
        "episodes": []
    }"#;

    #[test]
    fn seal_open_round_trip() {
        let sealed = seal_export(EXPORT_JSON, "correct horse battery staple").unwrap();
        assert!(is_sealed_export(&sealed));
        assert!(
            !sealed.contains("exported_at"),
            "sealed file must not leak plaintext fields"
        );
        let opened = open_export(&sealed, "correct horse battery staple").unwrap();
        assert_eq!(opened, EXPORT_JSON);
        // The opened JSON parses as a real export envelope.
        let parsed: MemoryExport = serde_json::from_str(&opened).unwrap();
        assert_eq!(parsed.version, "1");
    }

    #[test]
    fn wrong_passphrase_fails_closed() {
        let sealed = seal_export(EXPORT_JSON, "right").unwrap();
        let err = open_export(&sealed, "wrong").unwrap_err();
        assert!(
            err.to_string().contains("Decryption failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn each_export_uses_a_fresh_salt() {
        let a = seal_export(EXPORT_JSON, "p").unwrap();
        let b = seal_export(EXPORT_JSON, "p").unwrap();
        let salt = |raw: &str| serde_json::from_str::<SealedExport>(raw).unwrap().salt;
        assert_ne!(salt(&a), salt(&b), "salts must be per-export, not reused");
    }

    #[test]
    fn plaintext_export_is_not_mistaken_for_sealed() {
        assert!(!is_sealed_export(EXPORT_JSON));
        assert!(!is_sealed_export("not json at all"));
    }
}
