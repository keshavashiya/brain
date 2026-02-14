//! Encryption layer for data at rest.
//!
//! - AES-256-GCM for encrypting content columns
//! - Argon2id for deriving encryption keys from user passphrase
//! - Per-record unique nonce generation

use aes_gcm::{
    Aes256Gcm, Key, Nonce,
    aead::{Aead, KeyInit, OsRng},
};
use argon2::Argon2;
use thiserror::Error;

/// Fixed nonce size for AES-256-GCM (96 bits).
const NONCE_SIZE: usize = 12;

/// Salt size for Argon2id key derivation.
const SALT_SIZE: usize = 16;

/// Errors from the encryption layer.
#[derive(Debug, Error)]
pub enum EncryptionError {
    #[error("Encryption failed: {0}")]
    EncryptFailed(String),

    #[error("Decryption failed: {0}")]
    DecryptFailed(String),

    #[error("Key derivation failed: {0}")]
    KeyDerivation(String),

    #[error("Invalid data format")]
    InvalidFormat,
}

/// Encryption key manager.
///
/// Holds the derived AES-256 key in memory for the session duration.
/// The key is derived from the user's passphrase using Argon2id.
#[derive(Clone)]
pub struct Encryptor {
    key: Key<Aes256Gcm>,
}

impl Encryptor {
    /// Create an encryptor from a raw 32-byte key.
    pub fn from_key(key: [u8; 32]) -> Self {
        Self {
            key: Key::<Aes256Gcm>::from(key),
        }
    }

    /// Derive an encryption key from a passphrase and salt.
    ///
    /// Uses Argon2id with default parameters (memory=19 MiB, iterations=2, parallelism=1).
    pub fn from_passphrase(passphrase: &str, salt: &[u8]) -> Result<Self, EncryptionError> {
        let mut key = [0u8; 32];

        Argon2::default()
            .hash_password_into(passphrase.as_bytes(), salt, &mut key)
            .map_err(|e| EncryptionError::KeyDerivation(e.to_string()))?;

        Ok(Self::from_key(key))
    }

    /// Encrypt plaintext. Returns `nonce || ciphertext`.
    ///
    /// Each call generates a fresh 96-bit random nonce, prepended to output.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        use aes_gcm::aead::rand_core::RngCore;

        let cipher = Aes256Gcm::new(&self.key);

        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| EncryptionError::EncryptFailed(e.to_string()))?;

        // Prepend nonce to ciphertext
        let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    /// Decrypt data produced by `encrypt()`. Input format: `nonce || ciphertext`.
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        if data.len() < NONCE_SIZE {
            return Err(EncryptionError::InvalidFormat);
        }

        let (nonce_bytes, ciphertext) = data.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);
        let cipher = Aes256Gcm::new(&self.key);

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| EncryptionError::DecryptFailed(e.to_string()))
    }

    /// Encrypt a string, returning base64-encoded ciphertext.
    pub fn encrypt_string(&self, plaintext: &str) -> Result<String, EncryptionError> {
        use base64::Engine;
        let encrypted = self.encrypt(plaintext.as_bytes())?;
        Ok(base64::engine::general_purpose::STANDARD.encode(encrypted))
    }

    /// Decrypt a base64-encoded ciphertext back to a string.
    pub fn decrypt_string(&self, encoded: &str) -> Result<String, EncryptionError> {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_e| EncryptionError::InvalidFormat)?;
        let decrypted = self.decrypt(&data)?;
        String::from_utf8(decrypted)
            .map_err(|e| EncryptionError::DecryptFailed(format!("Invalid UTF-8: {e}")))
    }

    /// Generate a random salt for key derivation.
    pub fn generate_salt() -> [u8; SALT_SIZE] {
        use aes_gcm::aead::rand_core::RngCore;
        let mut salt = [0u8; SALT_SIZE];
        OsRng.fill_bytes(&mut salt);
        salt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_encryptor() -> Encryptor {
        Encryptor::from_key([42u8; 32])
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let enc = test_encryptor();
        let plaintext = b"Hello, Brain!";
        let ciphertext = enc.encrypt(plaintext).unwrap();
        let decrypted = enc.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_produces_different_nonces() {
        let enc = test_encryptor();
        let a = enc.encrypt(b"same data").unwrap();
        let b = enc.encrypt(b"same data").unwrap();
        // Different nonces → different ciphertexts
        assert_ne!(a, b);
        // But both decrypt to the same value
        assert_eq!(enc.decrypt(&a).unwrap(), enc.decrypt(&b).unwrap());
    }

    #[test]
    fn test_decrypt_wrong_key_fails() {
        let enc1 = Encryptor::from_key([1u8; 32]);
        let enc2 = Encryptor::from_key([2u8; 32]);
        let ciphertext = enc1.encrypt(b"secret").unwrap();
        assert!(enc2.decrypt(&ciphertext).is_err());
    }

    #[test]
    fn test_decrypt_truncated_fails() {
        let enc = test_encryptor();
        assert!(enc.decrypt(&[0u8; 5]).is_err()); // Too short for nonce
    }

    #[test]
    fn test_string_roundtrip() {
        let enc = test_encryptor();
        let original = "Keshav likes Rust";
        let encrypted = enc.encrypt_string(original).unwrap();
        let decrypted = enc.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, original);
    }

    #[test]
    fn test_passphrase_derivation() {
        let salt = Encryptor::generate_salt();
        let enc = Encryptor::from_passphrase("my-strong-passphrase", &salt).unwrap();
        let ciphertext = enc.encrypt(b"test data").unwrap();

        // Same passphrase + salt → same key → decrypts
        let enc2 = Encryptor::from_passphrase("my-strong-passphrase", &salt).unwrap();
        assert_eq!(enc2.decrypt(&ciphertext).unwrap(), b"test data");

        // Different passphrase → different key → fails
        let enc3 = Encryptor::from_passphrase("wrong-passphrase", &salt).unwrap();
        assert!(enc3.decrypt(&ciphertext).is_err());
    }
}
