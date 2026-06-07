//! Encrypted-file fallback backend.
//!
//! Layout under `<dir>/`:
//! ```text
//! <dir>/
//!   .verifier           — Argon2id hash of passphrase (PHC string)
//!   <tool>/<key>.enc    — one encrypted blob per entry (nonce || ciphertext)
//!   <tool>/<key>.meta   — sidecar JSON: { shape, created_at, last_used_at }
//! ```
//!
//! Encryption: AES-256-GCM with 96-bit random nonce per write. Key derived
//! from passphrase via Argon2id (46 MiB, t=1, p=1) using a per-vault salt
//! stored alongside the verifier.

use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::{Algorithm, Argon2, Params, Version};
use chrono::Utc;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use tokio::fs;
use zeroize::Zeroizing;

use crate::inject::{CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape};
use crate::vault::VaultError;

const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
const VERIFIER_FILE: &str = ".verifier";
const SALT_FILE: &str = ".salt";

/// Argon2id parameters — OWASP 2024 minimum recommendation for password
/// hashing on commodity hardware (Issue 60). 46 MiB memory, t=1
/// iteration, parallelism=1. Bumped from the v0.4.0 value
/// (19 MiB, t=2, p=1) which was still inside OWASP's older range but
/// now sits below the 2024 floor. Higher memory raises GPU/ASIC cost
/// disproportionately to CPU cost on the legitimate single-user path.
fn argon2_params() -> Params {
    Params::new(46 * 1024, 1, 1, Some(KEY_LEN)).expect("valid argon2 params")
}

/// Source of the fallback passphrase.
#[derive(Debug, Clone)]
pub enum PassphraseSource {
    /// Direct passphrase (for tests / programmatic init).
    Direct(String),
    /// Read from environment variable (default: `BRAIN_VAULT_PASSPHRASE`).
    EnvVar(String),
    /// Read from a file path.
    File(PathBuf),
    /// Prompt on the TTY via `rpassword`.
    Prompt,
}

impl PassphraseSource {
    pub fn resolve(&self) -> Result<String, VaultError> {
        match self {
            PassphraseSource::Direct(s) => Ok(s.clone()),
            PassphraseSource::EnvVar(name) => {
                std::env::var(name).map_err(|_| VaultError::PassphraseMissing)
            }
            PassphraseSource::File(path) => {
                let raw = std::fs::read_to_string(path)?;
                Ok(raw.trim_end_matches(['\n', '\r']).to_string())
            }
            PassphraseSource::Prompt => {
                rpassword::prompt_password("Vault passphrase: ").map_err(VaultError::Io)
            }
        }
    }
}

/// Encrypted-file backend.
pub struct FileBackend {
    dir: PathBuf,
    passphrase: PassphraseSource,
}

impl FileBackend {
    pub fn new(dir: PathBuf, passphrase: PassphraseSource) -> Self {
        Self { dir, passphrase }
    }

    /// Initialise the vault directory and write the passphrase verifier.
    /// Idempotent: re-running with the same passphrase is a no-op; a
    /// different passphrase returns `BadPassphrase`.
    pub async fn init(&self) -> Result<(), VaultError> {
        fs::create_dir_all(&self.dir).await?;

        let verifier_path = self.dir.join(VERIFIER_FILE);
        let salt_path = self.dir.join(SALT_FILE);

        // Issue 61: zeroize on drop.
        let passphrase = Zeroizing::new(self.passphrase.resolve()?);

        if verifier_path.exists() {
            // Verify the supplied passphrase matches the stored verifier.
            let hash_str = fs::read_to_string(&verifier_path).await?;
            let parsed = PasswordHash::new(&hash_str)
                .map_err(|e| VaultError::InvalidData(format!("verifier parse: {e}")))?;
            Argon2::default()
                .verify_password(passphrase.as_bytes(), &parsed)
                .map_err(|_| VaultError::BadPassphrase)?;
            return Ok(());
        }

        // First-time init: write verifier + salt.
        let salt = SaltString::generate(&mut OsRng);
        let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon2_params());
        let hash = argon
            .hash_password(passphrase.as_bytes(), &salt)
            .map_err(|e| VaultError::Crypto(format!("hash_password: {e}")))?
            .to_string();
        fs::write(&verifier_path, hash).await?;

        // Persist salt for key derivation (distinct from verifier salt — we
        // use one salt for the derivation key across all entries).
        let mut salt_bytes = [0u8; 16];
        OsRng.fill_bytes(&mut salt_bytes);
        fs::write(&salt_path, salt_bytes).await?;

        Ok(())
    }

    /// Ensure verifier exists; returns `BadPassphrase` if it doesn't match.
    /// Issue 61: the resolved passphrase is wrapped in `Zeroizing` so it
    /// is scrubbed from memory when the binding drops.
    async fn require_verified(&self) -> Result<Zeroizing<String>, VaultError> {
        let verifier_path = self.dir.join(VERIFIER_FILE);
        if !verifier_path.exists() {
            return Err(VaultError::BackendUnavailable(format!(
                "vault not initialised at {} — run `brain vault init`",
                self.dir.display()
            )));
        }
        let passphrase = Zeroizing::new(self.passphrase.resolve()?);
        let hash_str = fs::read_to_string(&verifier_path).await?;
        let parsed = PasswordHash::new(&hash_str)
            .map_err(|e| VaultError::InvalidData(format!("verifier parse: {e}")))?;
        Argon2::default()
            .verify_password(passphrase.as_bytes(), &parsed)
            .map_err(|_| VaultError::BadPassphrase)?;
        Ok(passphrase)
    }

    /// Issue 61: derived key is wrapped in `Zeroizing` so its bytes are
    /// scrubbed from RAM when the value is dropped. Callers should keep
    /// the returned binding scoped tightly around the AES-GCM call.
    async fn derive_key(&self, passphrase: &str) -> Result<Zeroizing<[u8; KEY_LEN]>, VaultError> {
        let salt_path = self.dir.join(SALT_FILE);
        let salt = fs::read(&salt_path).await?;
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon2_params());
        argon
            .hash_password_into(passphrase.as_bytes(), &salt, key.as_mut())
            .map_err(|e| VaultError::Crypto(format!("derive: {e}")))?;
        Ok(key)
    }

    fn entry_paths(&self, tool: &str, key: &str) -> (PathBuf, PathBuf) {
        let base = self.dir.join(sanitize(tool));
        let name = sanitize(key);
        (
            base.join(format!("{name}.enc")),
            base.join(format!("{name}.meta")),
        )
    }

    pub async fn store(
        &self,
        tool: &str,
        key: &str,
        value: CredentialValue,
        shape: InjectionShape,
    ) -> Result<(), VaultError> {
        let passphrase = self.require_verified().await?;
        let derived = self.derive_key(&passphrase).await?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(derived.as_slice()));

        let mut nonce_bytes = [0u8; NONCE_LEN];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, value.as_str().as_bytes())
            .map_err(|e| VaultError::Crypto(format!("encrypt: {e}")))?;

        let (enc_path, meta_path) = self.entry_paths(tool, key);
        if let Some(parent) = enc_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        fs::write(&enc_path, &blob).await?;

        let now = Utc::now().to_rfc3339();
        let meta = StoredMeta {
            shape,
            created_at: now.clone(),
            last_used_at: None,
        };
        fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).await?;
        Ok(())
    }

    pub async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError> {
        let passphrase = self.require_verified().await?;
        let derived = self.derive_key(&passphrase).await?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(derived.as_slice()));

        let (enc_path, meta_path) = self.entry_paths(tool, key);
        if !enc_path.exists() {
            return Err(VaultError::NotFound {
                tool: tool.to_string(),
                key: key.to_string(),
            });
        }
        let blob = fs::read(&enc_path).await?;
        if blob.len() <= NONCE_LEN {
            return Err(VaultError::InvalidData(format!(
                "blob too short: {} bytes",
                blob.len()
            )));
        }
        let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
        let plaintext = cipher
            .decrypt(Nonce::from_slice(nonce_bytes), ciphertext)
            .map_err(|e| VaultError::Crypto(format!("decrypt: {e}")))?;
        let value = String::from_utf8(plaintext)
            .map_err(|e| VaultError::InvalidData(format!("utf8: {e}")))?;

        let meta: StoredMeta = serde_json::from_slice(&fs::read(&meta_path).await?)
            .map_err(|e| VaultError::InvalidData(format!("meta: {e}")))?;

        // Update last_used_at best-effort.
        let updated = StoredMeta {
            last_used_at: Some(Utc::now().to_rfc3339()),
            ..meta.clone()
        };
        if let Err(err) = fs::write(
            &meta_path,
            serde_json::to_vec_pretty(&updated).unwrap_or_default(),
        )
        .await
        {
            tracing::warn!(error = %err, "vault: failed to update last_used_at");
        }

        Ok(InjectedCredential {
            shape: meta.shape,
            value: CredentialValue::new(value),
        })
    }

    pub async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError> {
        let (enc_path, meta_path) = self.entry_paths(tool, key);
        if !enc_path.exists() {
            return Err(VaultError::NotFound {
                tool: tool.to_string(),
                key: key.to_string(),
            });
        }
        fs::remove_file(&enc_path).await?;
        if meta_path.exists() {
            let _ = fs::remove_file(&meta_path).await;
        }
        Ok(())
    }

    pub async fn list(&self, tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError> {
        let mut out = Vec::new();
        if !self.dir.exists() {
            return Ok(out);
        }
        let mut tool_dirs = fs::read_dir(&self.dir).await?;
        while let Some(entry) = tool_dirs.next_entry().await? {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let tool_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            if let Some(filter) = tool {
                if tool_name != sanitize(filter) && tool_name != filter {
                    continue;
                }
            }
            if let Err(err) = collect_entries(&path, &tool_name, &mut out).await {
                tracing::warn!(tool = %tool_name, error = %err, "vault: list failed for tool dir");
            }
        }
        Ok(out)
    }
}

async fn collect_entries(
    tool_dir: &Path,
    tool_name: &str,
    out: &mut Vec<CredentialMetadata>,
) -> Result<(), VaultError> {
    let mut entries = fs::read_dir(tool_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("meta") {
            continue;
        }
        let key_name = match path
            .file_stem()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
        {
            Some(s) => s,
            None => continue,
        };
        let raw = fs::read(&path).await?;
        let meta: StoredMeta = serde_json::from_slice(&raw)
            .map_err(|e| VaultError::InvalidData(format!("meta: {e}")))?;
        out.push(CredentialMetadata {
            tool: tool_name.to_string(),
            key: key_name,
            backend: "file".to_string(),
            created_at: meta.created_at,
            last_used_at: meta.last_used_at,
            shape: meta.shape,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMeta {
    shape: InjectionShape,
    created_at: String,
    last_used_at: Option<String>,
}

/// Sanitize tool/key names for use as filesystem components.
/// Replaces anything outside `[A-Za-z0-9._-]` with `_`.
///
/// The allowlist keeps `.`, which alone would let the traversal components
/// `.` and `..` through — and `entry_paths` joins the sanitized *tool* name
/// onto the vault dir with no suffix, so a tool literally named `..` would
/// escape one level up. An empty name would also collapse onto the vault dir
/// itself. Both cases are neutralized to a safe `Normal` component.
fn sanitize(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();

    // "", ".", and ".." are the only mapped results that aren't a single
    // ordinary path component. Prefix them so they stay inside the vault dir.
    if mapped.is_empty() || mapped == "." || mapped == ".." {
        format!("_{mapped}")
    } else {
        mapped
    }
}

#[cfg(test)]
mod sanitize_tests {
    use super::sanitize;
    use proptest::prelude::*;
    use std::path::{Component, Path};

    /// Fragments weighted toward path metacharacters an attacker-controlled
    /// tool/key name might carry: traversal dots, separators, NUL, normal text.
    fn path_fragment() -> impl Strategy<Value = String> {
        prop_oneof![
            Just(".".to_string()),
            Just("..".to_string()),
            Just("/".to_string()),
            Just("\\".to_string()),
            Just("\0".to_string()),
            Just("~".to_string()),
            "[A-Za-z0-9._-]{0,6}",
            ".*", // arbitrary, including unicode and control bytes
        ]
    }

    fn hostile_name(max: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(path_fragment(), 0..max).prop_map(|f| f.concat())
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// Whatever the input, the sanitized name is always exactly one
        /// ordinary path component — never empty, never `.`/`..`, never a
        /// separator — so joining it onto the vault dir can neither traverse
        /// out of it nor collapse onto the dir itself.
        #[test]
        fn sanitize_yields_one_safe_component(s in hostile_name(12)) {
            let out = sanitize(&s);

            prop_assert!(!out.is_empty(), "empty component for {s:?}");
            prop_assert!(
                out.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
                "illegal char survived for {s:?}: {out:?}"
            );

            let comps: Vec<Component> = Path::new(&out).components().collect();
            prop_assert_eq!(comps.len(), 1, "not a single component for {:?}: {:?}", s, out);
            prop_assert!(
                matches!(comps[0], Component::Normal(_)),
                "non-Normal component (traversal/root) for {s:?}: {out:?}"
            );
        }

        /// The bytes the tool/key actually appears under stay inside the
        /// vault dir: `dir.join(sanitize(name))` is always a direct child.
        #[test]
        fn sanitized_join_stays_under_dir(s in hostile_name(12)) {
            let dir = Path::new("/vault/root");
            let joined = dir.join(sanitize(&s));
            prop_assert_eq!(joined.parent(), Some(dir), "escaped vault dir for {:?}", s);
        }
    }
}
