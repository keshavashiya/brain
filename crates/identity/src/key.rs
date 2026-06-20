//! `IdentityKey` — the install's persistent root secret.
//!
//! A single 32-byte secret, minted once on first use and stored at
//! `<data_dir>/identity.key` with owner-only permissions. It is the root of
//! Brain's at-rest key material: every other local key (e.g. the audit-trail
//! cipher) is *derived* from it via [`IdentityKey::derive_subkey`] rather than
//! stored separately, so there is exactly one secret to protect and back up.
//!
//! Derivation uses BLAKE3's keyed-KDF mode: distinct context strings yield
//! cryptographically independent sub-keys, and the same `(root, context)` pair
//! always reproduces the same sub-key — which is what lets the audit trail
//! decrypt rows written in an earlier process.

use std::path::Path;

use thiserror::Error;

/// Length of the root secret and every derived sub-key, in bytes (256 bits).
pub const KEY_LEN: usize = 32;

#[derive(Debug, Error)]
pub enum IdentityKeyError {
    #[error("identity key I/O at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("identity key at {path} is malformed: expected {KEY_LEN} bytes, found {found}")]
    Malformed { path: String, found: usize },
    #[error("failed to gather entropy for a new identity key: {0}")]
    Entropy(String),
}

/// The install's persistent root secret. Cheap to clone (32 bytes inline).
#[derive(Clone)]
pub struct IdentityKey {
    secret: [u8; KEY_LEN],
}

impl IdentityKey {
    /// Wrap raw secret bytes. Primarily for tests and callers that source the
    /// secret themselves.
    pub fn from_bytes(secret: [u8; KEY_LEN]) -> Self {
        Self { secret }
    }

    /// Load the root secret from `path`, minting and persisting a fresh random
    /// one if the file is absent. The parent directory is created if needed,
    /// and the file is written with `0600` permissions on Unix.
    ///
    /// Concurrent first-time creation is benign: the create is best-effort
    /// exclusive, and a racing process that loses the race simply reads the
    /// winner's freshly written key.
    pub fn load_or_create(path: &Path) -> Result<Self, IdentityKeyError> {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_loaded_bytes(path, bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::create(path),
            Err(source) => Err(IdentityKeyError::Io {
                path: path.display().to_string(),
                source,
            }),
        }
    }

    fn from_loaded_bytes(path: &Path, bytes: Vec<u8>) -> Result<Self, IdentityKeyError> {
        let secret: [u8; KEY_LEN] =
            bytes
                .as_slice()
                .try_into()
                .map_err(|_| IdentityKeyError::Malformed {
                    path: path.display().to_string(),
                    found: bytes.len(),
                })?;
        Ok(Self { secret })
    }

    fn create(path: &Path) -> Result<Self, IdentityKeyError> {
        let mut secret = [0u8; KEY_LEN];
        getrandom::fill(&mut secret).map_err(|e| IdentityKeyError::Entropy(e.to_string()))?;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| IdentityKeyError::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
        }

        // Create exclusively so two racing daemons don't both write — the loser
        // gets AlreadyExists and falls back to reading the winner's key.
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            let mut f = opts.open(path)?;
            f.write_all(&secret)?;
            f.sync_all()
        };

        match write() {
            Ok(()) => Ok(Self { secret }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // Someone else minted it between our read and write — adopt theirs.
                let bytes = std::fs::read(path).map_err(|source| IdentityKeyError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
                Self::from_loaded_bytes(path, bytes)
            }
            Err(source) => Err(IdentityKeyError::Io {
                path: path.display().to_string(),
                source,
            }),
        }
    }

    /// Derive a 32-byte sub-key bound to `context`. Distinct contexts yield
    /// independent keys; the same context always reproduces the same key.
    ///
    /// Use a stable, unique context string per consumer, e.g.
    /// `"brain-audit-trail-v1"`.
    pub fn derive_subkey(&self, context: &str) -> [u8; KEY_LEN] {
        blake3::derive_key(context, &self.secret)
    }

    /// The raw root secret. Prefer [`derive_subkey`](Self::derive_subkey) —
    /// exposing the root directly is only for callers that must serialize it.
    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.secret
    }
}

impl std::fmt::Debug for IdentityKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the secret.
        f.write_str("IdentityKey(<redacted>)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_or_create_mints_then_reloads_the_same_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");

        let first = IdentityKey::load_or_create(&path).unwrap();
        assert!(path.exists(), "key file should be persisted");

        let second = IdentityKey::load_or_create(&path).unwrap();
        assert_eq!(
            first.as_bytes(),
            second.as_bytes(),
            "reload must return the persisted key, not a new one"
        );
    }

    #[test]
    fn mints_parent_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/sub/identity.key");
        let _ = IdentityKey::load_or_create(&path).unwrap();
        assert!(path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn minted_key_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        let _ = IdentityKey::load_or_create(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "key file must be 0600");
    }

    #[test]
    fn derive_subkey_is_deterministic_and_context_separated() {
        let k = IdentityKey::from_bytes([7u8; KEY_LEN]);
        let a1 = k.derive_subkey("ctx-a");
        let a2 = k.derive_subkey("ctx-a");
        let b = k.derive_subkey("ctx-b");
        assert_eq!(a1, a2, "same context → same sub-key");
        assert_ne!(a1, b, "different context → independent sub-key");
        assert_ne!(a1, *k.as_bytes(), "sub-key must not equal the root secret");
    }

    #[test]
    fn different_roots_yield_different_subkeys() {
        let k1 = IdentityKey::from_bytes([1u8; KEY_LEN]);
        let k2 = IdentityKey::from_bytes([2u8; KEY_LEN]);
        assert_ne!(k1.derive_subkey("ctx"), k2.derive_subkey("ctx"));
    }

    #[test]
    fn malformed_key_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("identity.key");
        std::fs::write(&path, b"too short").unwrap();
        let err = IdentityKey::load_or_create(&path).unwrap_err();
        assert!(matches!(err, IdentityKeyError::Malformed { found, .. } if found == 9));
    }

    #[test]
    fn debug_does_not_leak_secret() {
        let k = IdentityKey::from_bytes([0xABu8; KEY_LEN]);
        let s = format!("{k:?}");
        assert!(!s.contains("ab"), "debug should not print key bytes: {s}");
        assert!(s.contains("redacted"));
    }
}
