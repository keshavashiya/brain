//! OAuth provider trait and shared types.
//!
//! Extends `LlmProvider` for providers that require browser-consent flows,
//! short-lived access tokens, and refresh-token rotation. Token storage goes
//! through the credential vault so secrets never sit in plain-text config.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::llm::LlmError;

/// Presented to the user during device-code flow.
#[derive(Debug, Clone, Serialize)]
pub struct AuthChallenge {
    pub verification_uri: String,
    pub verification_uri_complete: String,
    pub user_code: String,
    pub device_code: String,
    pub expires_in: u64,
}

/// Tokens returned after the user completes the consent flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSet {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub expiry_date: Option<DateTime<Utc>>,
    #[serde(default)]
    pub resource_url: Option<String>,
}

impl TokenSet {
    pub fn is_expired(&self) -> bool {
        match self.expiry_date {
            Some(exp) => Utc::now() >= exp,
            None => true,
        }
    }
}

/// Quota state for a provider.
#[derive(Debug, Clone, Serialize)]
pub struct QuotaStatus {
    pub limit: u64,
    pub used: u64,
    pub reset_at: DateTime<Utc>,
}

/// Trait for providers that authenticate via OAuth.
#[async_trait::async_trait]
pub trait OAuthProvider: Send + Sync {
    fn provider_name(&self) -> &str;

    async fn begin_device_auth(&self) -> Result<AuthChallenge, LlmError>;

    async fn poll_device_token(
        &self,
        device_code: &str,
        code_verifier: &str,
    ) -> Result<PollResult, LlmError>;

    async fn refresh_tokens(&self) -> Result<TokenSet, LlmError>;

    async fn load_tokens(&self) -> Result<Option<TokenSet>, LlmError>;

    async fn save_tokens(&self, tokens: &TokenSet) -> Result<(), LlmError>;

    async fn clear_tokens(&self) -> Result<(), LlmError>;
}

/// Result of polling the device token endpoint.
#[derive(Debug)]
pub enum PollResult {
    Pending,
    SlowDown,
    Ready(TokenSet),
}

// ─── PKCE ──────────────────────────────────────────────────────────────────

pub fn generate_code_verifier() -> String {
    use rand::Rng;
    let bytes: [u8; 32] = rand::thread_rng().gen();
    base64url_encode(&bytes)
}

pub fn generate_code_challenge(verifier: &str) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(verifier.as_bytes());
    base64url_encode(&hash)
}

pub fn base64url_encode(data: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(data.len() * 4 / 3 + 4);
    let engine = base64_chars();
    for chunk in data.chunks(3) {
        let n = match chunk.len() {
            3 => (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8) | u32::from(chunk[2]),
            2 => (u32::from(chunk[0]) << 16) | (u32::from(chunk[1]) << 8),
            1 => u32::from(chunk[0]) << 16,
            _ => unreachable!(),
        };
        let indices = [
            (n >> 18) & 0x3F,
            (n >> 12) & 0x3F,
            (n >> 6) & 0x3F,
            n & 0x3F,
        ];
        let _ = out.write_char(engine[indices[0] as usize]);
        let _ = out.write_char(engine[indices[1] as usize]);
        if chunk.len() > 1 {
            let _ = out.write_char(engine[indices[2] as usize]);
        }
        if chunk.len() > 2 {
            let _ = out.write_char(engine[indices[3] as usize]);
        }
    }
    out
}

fn base64_chars() -> [char; 64] {
    let mut table = ['\0'; 64];
    for (i, c) in (b'A'..=b'Z').enumerate() {
        table[i] = c as char;
    }
    for (i, c) in (b'a'..=b'z').enumerate() {
        table[26 + i] = c as char;
    }
    for (i, c) in (b'0'..=b'9').enumerate() {
        table[52 + i] = c as char;
    }
    table[62] = '-';
    table[63] = '_';
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_verifier_is_correct_length() {
        let v = generate_code_verifier();
        assert!(v.len() >= 43, "verifier too short: {}", v.len());
    }

    #[test]
    fn pkce_challenge_deterministic() {
        let c1 = generate_code_challenge("test-verifier");
        let c2 = generate_code_challenge("test-verifier");
        assert_eq!(c1, c2);
        assert_ne!(c1, generate_code_challenge("different"));
    }

    #[test]
    fn base64url_no_padding_or_plus_slash() {
        let data = [0xFF, 0xFE, 0xFD, 0xFC];
        let encoded = base64url_encode(&data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
    }

    #[test]
    fn token_set_expired() {
        let expired = TokenSet {
            access_token: "x".into(),
            refresh_token: None,
            expiry_date: Some(Utc::now() - chrono::Duration::hours(1)),
            resource_url: None,
        };
        assert!(expired.is_expired());

        let valid = TokenSet {
            access_token: "x".into(),
            refresh_token: None,
            expiry_date: Some(Utc::now() + chrono::Duration::hours(1)),
            resource_url: None,
        };
        assert!(!valid.is_expired());
    }
}
