//! `CredentialVault` trait, error type, default vault impl, and config.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use audit::{ActionTier, AuditEntry, AuditTrail};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend::{BackendKind, VaultBackend};
use crate::file::{FileBackend, PassphraseSource};
use crate::inject::{CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape};

#[derive(Debug, Error)]
pub enum VaultError {
    #[error("credential not found for tool={tool} key={key}")]
    NotFound { tool: String, key: String },

    #[error("vault backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("bad passphrase — verifier check failed")]
    BadPassphrase,

    #[error("passphrase required but no source available (env BRAIN_VAULT_PASSPHRASE, passphrase_file, or TTY)")]
    PassphraseMissing,

    #[error("encryption/decryption failed: {0}")]
    Crypto(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid data: {0}")]
    InvalidData(String),

    #[error("backend error: {0}")]
    Backend(String),
}

/// Vault configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultConfig {
    /// Backend selection. `Auto` picks the best available backend at runtime.
    #[serde(default)]
    pub backend: BackendSelection,

    /// Base directory for the encrypted-file backend and verifier.
    /// Defaults to `$HOME/.brain/vault`.
    #[serde(default)]
    pub dir: Option<PathBuf>,

    /// Path to a file containing the fallback passphrase. Ignored if the env
    /// var `BRAIN_VAULT_PASSPHRASE` is set. If neither is provided and the
    /// file backend is active, the caller prompts on the TTY.
    #[serde(default)]
    pub passphrase_file: Option<PathBuf>,
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            backend: BackendSelection::Auto,
            dir: None,
            passphrase_file: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelection {
    #[default]
    Auto,
    Keychain,
    File,
}

/// Credential vault.
///
/// Per-tool scoping is caller-side: the `(tool, key)` pair forms the
/// namespace. There is no cross-tool ACL — single-user machine per
/// VISION §1.
#[async_trait]
pub trait CredentialVault: Send + Sync {
    /// Store a credential with a given injection shape.
    async fn store(
        &self,
        tool: &str,
        key: &str,
        value: CredentialValue,
        shape: InjectionShape,
    ) -> Result<(), VaultError>;

    /// Retrieve a credential prepared for injection.
    async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError>;

    /// Delete a credential.
    async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError>;

    /// List credential metadata (no values).
    async fn list(&self, tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError>;

    /// Which backend is active.
    fn backend_kind(&self) -> BackendKind;
}

/// Default vault that delegates to a `VaultBackend` and optionally records
/// audit entries for `get` / `store` / `delete`.
pub struct DefaultVault {
    backend: VaultBackend,
    audit: Option<Arc<dyn AuditTrail>>,
}

impl DefaultVault {
    pub fn new(backend: VaultBackend) -> Self {
        Self {
            backend,
            audit: None,
        }
    }

    pub fn with_audit(mut self, audit: Arc<dyn AuditTrail>) -> Self {
        self.audit = Some(audit);
        self
    }

    async fn record(&self, tier: ActionTier, action: &str, tool: &str, key: &str) {
        let Some(audit) = &self.audit else {
            return;
        };
        let metadata = serde_json::json!({
            "tool": tool,
            "key": key,
            "backend": self.backend.kind().to_string(),
        });
        let entry = AuditEntry::new(
            format!("vault.{action} {tool}:{key}"),
            format!("vault.{action}"),
            format!("vault.{action}"),
            tier,
        )
        .with_source("vault")
        .with_metadata(metadata);
        if let Err(err) = audit.record(entry).await {
            tracing::warn!(error = %err, "vault: audit record failed");
        }
    }
}

#[async_trait]
impl CredentialVault for DefaultVault {
    async fn store(
        &self,
        tool: &str,
        key: &str,
        value: CredentialValue,
        shape: InjectionShape,
    ) -> Result<(), VaultError> {
        self.backend.store(tool, key, value, shape).await?;
        self.record(ActionTier::Write, "store", tool, key).await;
        Ok(())
    }

    async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError> {
        let injected = self.backend.get(tool, key).await?;
        self.record(ActionTier::External, "get", tool, key).await;
        Ok(injected)
    }

    async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError> {
        self.backend.delete(tool, key).await?;
        self.record(ActionTier::Write, "delete", tool, key).await;
        Ok(())
    }

    async fn list(&self, tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError> {
        self.backend.list(tool).await
    }

    fn backend_kind(&self) -> BackendKind {
        self.backend.kind()
    }
}

/// Resolve a `VaultBackend` from a `VaultConfig`. On `Auto`, prefers the OS
/// keychain when compiled in, otherwise falls back to the encrypted file
/// backend. Returns `BackendUnavailable` if explicitly requesting a keychain
/// backend on a platform that wasn't compiled in.
pub fn resolve_backend(config: &VaultConfig) -> Result<VaultBackend, VaultError> {
    let vault_dir = resolve_vault_dir(config)?;
    let file_backend = || FileBackend::new(vault_dir.clone(), resolve_passphrase(config));

    match config.backend {
        BackendSelection::File => Ok(VaultBackend::File(file_backend())),
        BackendSelection::Keychain => keychain_backend_or_err(),
        BackendSelection::Auto => {
            // If the file vault has been initialised (verifier present),
            // prefer it over the OS keychain — the user's explicit
            // `vault init --file` choice stays sticky without a separate
            // config file.
            if vault_dir.join(".verifier").exists() {
                return Ok(VaultBackend::File(file_backend()));
            }
            match keychain_backend_or_err() {
                Ok(b) => Ok(b),
                Err(_) => Ok(VaultBackend::File(file_backend())),
            }
        }
    }
}

fn keychain_backend_or_err() -> Result<VaultBackend, VaultError> {
    #[cfg(target_os = "macos")]
    {
        Ok(VaultBackend::Keychain(
            crate::keychain::KeychainBackend::new(),
        ))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(VaultBackend::SecretService(
            crate::keyring::SecretServiceBackend::new(),
        ))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Err(VaultError::BackendUnavailable(
            "no OS keychain compiled in for this platform".into(),
        ))
    }
}

fn resolve_vault_dir(config: &VaultConfig) -> Result<PathBuf, VaultError> {
    if let Some(dir) = &config.dir {
        return Ok(dir.clone());
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| VaultError::BackendUnavailable("HOME not set".into()))?;
    Ok(home.join(".brain").join("vault"))
}

fn resolve_passphrase(config: &VaultConfig) -> PassphraseSource {
    if std::env::var("BRAIN_VAULT_PASSPHRASE").is_ok() {
        return PassphraseSource::EnvVar("BRAIN_VAULT_PASSPHRASE".to_string());
    }
    if let Some(path) = &config.passphrase_file {
        return PassphraseSource::File(path.clone());
    }
    PassphraseSource::Prompt
}
