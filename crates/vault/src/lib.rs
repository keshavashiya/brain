//! # Brain Credential Vault
//!
//! Secure credential storage with OS keychain preferred and an AES-256-GCM
//! encrypted-file fallback. Raw values are injected at execution time and
//! never passed through BrainOS memory in plaintext form outside the vault
//! call site.

pub mod backend;
pub mod file;
pub mod inject;
pub mod vault;

#[cfg(target_os = "macos")]
pub mod keychain;

#[cfg(target_os = "linux")]
pub mod keyring;

pub use backend::{BackendKind, VaultBackend};
pub use inject::{CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape};
pub use vault::{
    resolve_backend, BackendSelection, CredentialVault, DefaultVault, VaultConfig, VaultError,
};
