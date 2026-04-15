//! Backend dispatch — one enum with cfg-gated variants.

use crate::file::FileBackend;
use crate::inject::{CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape};
use crate::vault::VaultError;

#[cfg(target_os = "macos")]
use crate::keychain::KeychainBackend;

#[cfg(target_os = "linux")]
use crate::keyring::SecretServiceBackend;

/// Which backend is active. Exposed to callers for audit metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Keychain,
    SecretService,
    File,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Keychain => write!(f, "keychain"),
            BackendKind::SecretService => write!(f, "secret-service"),
            BackendKind::File => write!(f, "file"),
        }
    }
}

/// A runtime-selected backend. The enum avoids dyn-dispatch for this small
/// closed set and keeps cfg gates localized.
pub enum VaultBackend {
    #[cfg(target_os = "macos")]
    Keychain(KeychainBackend),
    #[cfg(target_os = "linux")]
    SecretService(SecretServiceBackend),
    File(FileBackend),
}

impl VaultBackend {
    pub fn kind(&self) -> BackendKind {
        match self {
            #[cfg(target_os = "macos")]
            VaultBackend::Keychain(_) => BackendKind::Keychain,
            #[cfg(target_os = "linux")]
            VaultBackend::SecretService(_) => BackendKind::SecretService,
            VaultBackend::File(_) => BackendKind::File,
        }
    }

    pub async fn store(
        &self,
        tool: &str,
        key: &str,
        value: CredentialValue,
        shape: InjectionShape,
    ) -> Result<(), VaultError> {
        match self {
            #[cfg(target_os = "macos")]
            VaultBackend::Keychain(b) => b.store(tool, key, value, shape).await,
            #[cfg(target_os = "linux")]
            VaultBackend::SecretService(b) => b.store(tool, key, value, shape).await,
            VaultBackend::File(b) => b.store(tool, key, value, shape).await,
        }
    }

    pub async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError> {
        match self {
            #[cfg(target_os = "macos")]
            VaultBackend::Keychain(b) => b.get(tool, key).await,
            #[cfg(target_os = "linux")]
            VaultBackend::SecretService(b) => b.get(tool, key).await,
            VaultBackend::File(b) => b.get(tool, key).await,
        }
    }

    pub async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError> {
        match self {
            #[cfg(target_os = "macos")]
            VaultBackend::Keychain(b) => b.delete(tool, key).await,
            #[cfg(target_os = "linux")]
            VaultBackend::SecretService(b) => b.delete(tool, key).await,
            VaultBackend::File(b) => b.delete(tool, key).await,
        }
    }

    pub async fn list(&self, tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError> {
        match self {
            #[cfg(target_os = "macos")]
            VaultBackend::Keychain(b) => b.list(tool).await,
            #[cfg(target_os = "linux")]
            VaultBackend::SecretService(b) => b.list(tool).await,
            VaultBackend::File(b) => b.list(tool).await,
        }
    }
}
