//! OAuth access-token audience-claim (`aud`) validation.
//!
//! Threat: CVE-2025-6514 class "confused-deputy via token passthrough."
//! A legitimate access token issued for MCP server A is replayed
//! against server B. The mitigation is RFC 8707 `resource` indicators
//! at token-request time, plus `aud`-claim validation at token-use
//! time.
//!
//! Brain mcphost is the OAuth client. The complementary risk on the
//! client side is that the vault stores a token whose `aud` doesn't
//! match the configured `resource` for the server name we're keyed
//! by — either because the token was minted with the wrong resource
//! indicator at the time, or because something tampered with the
//! vault rows after the fact. This module's job is to catch that at
//! load time and fail closed before the bad token reaches the wire.
//!
//! ## What we validate
//!
//! - If the access token parses as a JWT (three base64url-encoded
//!   segments, middle segment is JSON) AND the JSON carries an `aud`
//!   claim: the claim must contain the configured resource string.
//! - If the token doesn't parse as a JWT (opaque token — common for
//!   first-party AS / OAuth 2.1 confidential clients): the audience
//!   check is skipped. Opaque tokens carry no in-band audience info;
//!   the resource indicator we passed at request time is the
//!   server-side enforcement.
//! - If the token parses as a JWT but has no `aud` claim: log a
//!   warning. RFC 9068 says JWT access tokens MUST carry `aud`, but
//!   real-world ASes deviate, so we don't fail closed on this.
//!
//! Signature verification is intentionally out of scope. Verifying
//! would require fetching the AS's JWKS, which adds a network round
//! trip and key-rotation complexity that rmcp's transport layer
//! already owns. The vault is the trust boundary here — we got the
//! token from a successful PKCE flow through TLS to the AS, and the
//! vault row is gated by the OS keychain or the encrypted-file
//! backend.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

/// Outcome of an audience check against an access token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudCheckOutcome {
    /// Token parses as JWT, has an `aud` claim, and the configured
    /// resource is present.
    Match,
    /// Token parses as JWT, has an `aud` claim, but the configured
    /// resource is **not** present. This is the confused-deputy
    /// signal — caller should reject the token and emit a
    /// `BrainEvent::Error`.
    Mismatch {
        found: Vec<String>,
        expected: String,
    },
    /// Token parses as JWT but the payload has no `aud` claim. Warn,
    /// don't fail closed — some ASes still don't include it.
    MissingAud,
    /// Token does not parse as a JWT (doesn't have three `.`-separated
    /// base64url segments, or the middle segment isn't valid JSON).
    /// This is the common case for opaque OAuth 2.1 tokens; no
    /// in-band audience info, validation is skipped.
    OpaqueToken,
}

impl AudCheckOutcome {
    /// Is this outcome a hard failure that should reject the token?
    pub fn is_mismatch(&self) -> bool {
        matches!(self, AudCheckOutcome::Mismatch { .. })
    }
}

/// Decode the `aud` claim from `access_token` and compare against
/// `expected_resource`. See module docs for what each outcome means.
///
/// `expected_resource` is the RFC 8707 resource indicator the client
/// configured for the server — typically the server's base URL, or a
/// distinct logical identifier the AS expects.
pub fn validate_token_aud(access_token: &str, expected_resource: &str) -> AudCheckOutcome {
    let Some(payload_b64) = jwt_payload_segment(access_token) else {
        return AudCheckOutcome::OpaqueToken;
    };
    let Ok(payload_bytes) = URL_SAFE_NO_PAD.decode(payload_b64) else {
        return AudCheckOutcome::OpaqueToken;
    };
    let Ok(payload) = serde_json::from_slice::<serde_json::Value>(&payload_bytes) else {
        return AudCheckOutcome::OpaqueToken;
    };

    let Some(aud_field) = payload.get("aud") else {
        return AudCheckOutcome::MissingAud;
    };

    let found = collect_aud(aud_field);
    if found.is_empty() {
        return AudCheckOutcome::MissingAud;
    }

    if found.iter().any(|v| v == expected_resource) {
        AudCheckOutcome::Match
    } else {
        AudCheckOutcome::Mismatch {
            found,
            expected: expected_resource.to_string(),
        }
    }
}

/// Extract the middle (payload) segment of a compact JWS-encoded JWT.
/// Returns `None` if the structure doesn't look like a JWT — exactly
/// three non-empty `.`-separated segments.
fn jwt_payload_segment(token: &str) -> Option<&str> {
    let mut parts = token.split('.');
    let _header = parts.next().filter(|s| !s.is_empty())?;
    let payload = parts.next().filter(|s| !s.is_empty())?;
    let _signature = parts.next().filter(|s| !s.is_empty())?;
    if parts.next().is_some() {
        return None;
    }
    Some(payload)
}

/// RFC 7519 `aud` is either a single string or an array of strings.
/// Other shapes (number, object) are treated as missing.
fn collect_aud(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    /// Build a fake compact-JWS-shaped token: `header.payload.sig`,
    /// where every segment is base64url(no-pad). Header + signature
    /// are placeholders — only the payload is read.
    pub(crate) fn fake_jwt(payload: serde_json::Value) -> String {
        let header_b64 = URL_SAFE_NO_PAD.encode(b"{\"alg\":\"RS256\",\"typ\":\"JWT\"}");
        let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
        let sig_b64 = URL_SAFE_NO_PAD.encode(b"signature-bytes");
        format!("{header_b64}.{payload_b64}.{sig_b64}")
    }

    #[test]
    fn match_for_string_aud() {
        let token = fake_jwt(serde_json::json!({
            "aud": "https://server.example.com",
            "sub": "user-1",
        }));
        let outcome = validate_token_aud(&token, "https://server.example.com");
        assert_eq!(outcome, AudCheckOutcome::Match);
    }

    #[test]
    fn match_for_array_aud_containing_expected() {
        let token = fake_jwt(serde_json::json!({
            "aud": ["https://other.example.com", "https://server.example.com"],
        }));
        let outcome = validate_token_aud(&token, "https://server.example.com");
        assert_eq!(outcome, AudCheckOutcome::Match);
    }

    #[test]
    fn mismatch_for_string_aud_pointing_elsewhere() {
        let token = fake_jwt(serde_json::json!({
            "aud": "https://other.example.com",
        }));
        let outcome = validate_token_aud(&token, "https://server.example.com");
        match outcome {
            AudCheckOutcome::Mismatch { found, expected } => {
                assert_eq!(found, vec!["https://other.example.com".to_string()]);
                assert_eq!(expected, "https://server.example.com");
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn mismatch_for_array_aud_without_expected() {
        let token = fake_jwt(serde_json::json!({
            "aud": ["a", "b", "c"],
        }));
        let outcome = validate_token_aud(&token, "z");
        assert!(outcome.is_mismatch());
    }

    #[test]
    fn missing_aud_when_payload_has_no_field() {
        let token = fake_jwt(serde_json::json!({
            "sub": "user-1",
        }));
        assert_eq!(
            validate_token_aud(&token, "anything"),
            AudCheckOutcome::MissingAud
        );
    }

    #[test]
    fn missing_aud_when_field_is_unexpected_type() {
        // RFC 7519 says aud is string or array-of-strings; numbers /
        // objects / arrays-of-non-strings are treated as missing.
        let token = fake_jwt(serde_json::json!({ "aud": 42 }));
        assert_eq!(
            validate_token_aud(&token, "anything"),
            AudCheckOutcome::MissingAud
        );
        let token = fake_jwt(serde_json::json!({ "aud": [123, 456] }));
        assert_eq!(
            validate_token_aud(&token, "anything"),
            AudCheckOutcome::MissingAud
        );
    }

    #[test]
    fn opaque_token_when_not_three_segments() {
        // Common opaque-token shapes from real-world OAuth deployments.
        assert_eq!(
            validate_token_aud("abc.def", "anything"),
            AudCheckOutcome::OpaqueToken
        );
        assert_eq!(
            validate_token_aud("just-a-random-string", "anything"),
            AudCheckOutcome::OpaqueToken
        );
        assert_eq!(
            validate_token_aud("a.b.c.d", "anything"),
            AudCheckOutcome::OpaqueToken
        );
        assert_eq!(
            validate_token_aud("a..c", "anything"),
            AudCheckOutcome::OpaqueToken
        );
    }

    #[test]
    fn opaque_token_when_middle_segment_isnt_json() {
        let header = URL_SAFE_NO_PAD.encode(b"{}");
        let payload = URL_SAFE_NO_PAD.encode(b"not-actually-json");
        let sig = URL_SAFE_NO_PAD.encode(b"sig");
        let token = format!("{header}.{payload}.{sig}");
        assert_eq!(
            validate_token_aud(&token, "anything"),
            AudCheckOutcome::OpaqueToken
        );
    }

    #[test]
    fn opaque_token_when_middle_segment_isnt_base64url() {
        let token = "abc.@@@not-base64@@@.def";
        assert_eq!(
            validate_token_aud(token, "anything"),
            AudCheckOutcome::OpaqueToken
        );
    }
}
