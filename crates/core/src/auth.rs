//! Shared authentication logic for all protocol adapters.
//!
//! This module is the single source of truth for API key validation and
//! permission checking. Every adapter (HTTP, WebSocket, gRPC, MCP) must
//! use these functions instead of rolling their own validation.

use subtle::ConstantTimeEq;

use crate::config::ApiKeyConfig;

/// Result of an authentication check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Auth is disabled (no API keys configured) — allow all.
    Open,
    /// Key is valid and has the requested permission.
    Allowed,
    /// No key was provided.
    MissingKey,
    /// Key was provided but does not match any configured key.
    InvalidKey,
    /// Key is valid but lacks the requested permission.
    InsufficientPermission,
}

impl AuthResult {
    /// Returns `true` if the request should be allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, AuthResult::Open | AuthResult::Allowed)
    }

    /// Returns a human-readable error message, or `None` if allowed.
    pub fn error_message(&self, permission: &str) -> Option<String> {
        match self {
            AuthResult::Open | AuthResult::Allowed => None,
            AuthResult::MissingKey => Some("Missing API key".to_string()),
            AuthResult::InvalidKey => Some("Invalid API key".to_string()),
            AuthResult::InsufficientPermission => {
                Some(format!("API key does not have '{permission}' permission"))
            }
        }
    }
}

/// Extract a bearer token from an "Authorization" header value string.
///
/// E.g. `extract_bearer_from_value("Bearer abc123")` returns `Some("abc123")`.
pub fn extract_bearer_from_value(value: &str) -> Option<&str> {
    value.strip_prefix("Bearer ")
}

/// Validate an API key against the configured keys with permission checking.
///
/// - If `api_keys` is empty, auth is disabled and all requests are allowed.
/// - If `provided_key` is `None`, returns `MissingKey`.
/// - If the key doesn't match any configured key, returns `InvalidKey`.
/// - If the key matches but lacks the required permission, returns `InsufficientPermission`.
/// - Otherwise returns `Allowed`.
pub fn check_auth(
    api_keys: &[ApiKeyConfig],
    provided_key: Option<&str>,
    permission: &str,
) -> AuthResult {
    if api_keys.is_empty() {
        return AuthResult::Open;
    }

    let key = match provided_key {
        Some(k) if !k.is_empty() => k,
        _ => return AuthResult::MissingKey,
    };

    match api_keys.iter().find(|k| {
        let a = k.key.as_bytes();
        let b = key.as_bytes();
        // NOTE: length precheck may leak key length via timing side-channel.
        // This is accepted because: (1) Brain binds to 127.0.0.1 by default,
        // so remote attackers cannot reach auth endpoints; (2) keys are
        // minted with ~128 bits of entropy, making length knowledge
        // inconsequential; (3) the timing delta (~50 ns) is below practical
        // measurement thresholds even on localhost.
        // See docs/deferred-followups-0.2.x.md for full analysis.
        a.len() == b.len() && a.ct_eq(b).into()
    }) {
        None => AuthResult::InvalidKey,
        Some(k) if !k.has_permission(permission) => AuthResult::InsufficientPermission,
        Some(_) => AuthResult::Allowed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keys() -> Vec<ApiKeyConfig> {
        vec![
            ApiKeyConfig {
                key: "rw-key".to_string(),
                name: "Read-Write".to_string(),
                permissions: vec!["read".to_string(), "write".to_string()],
                agent_id: None,
            },
            ApiKeyConfig {
                key: "ro-key".to_string(),
                name: "Read-Only".to_string(),
                permissions: vec!["read".to_string()],
                agent_id: None,
            },
        ]
    }

    #[test]
    fn test_empty_keys_allows_all() {
        assert_eq!(check_auth(&[], Some("anything"), "write"), AuthResult::Open);
        assert_eq!(check_auth(&[], None, "read"), AuthResult::Open);
    }

    #[test]
    fn test_missing_key() {
        let keys = test_keys();
        assert_eq!(check_auth(&keys, None, "read"), AuthResult::MissingKey);
        assert_eq!(check_auth(&keys, Some(""), "read"), AuthResult::MissingKey);
    }

    #[test]
    fn test_invalid_key() {
        let keys = test_keys();
        assert_eq!(
            check_auth(&keys, Some("bad-key"), "read"),
            AuthResult::InvalidKey
        );
    }

    #[test]
    fn test_insufficient_permission() {
        let keys = test_keys();
        assert_eq!(
            check_auth(&keys, Some("ro-key"), "write"),
            AuthResult::InsufficientPermission
        );
    }

    #[test]
    fn test_valid_read() {
        let keys = test_keys();
        assert_eq!(
            check_auth(&keys, Some("ro-key"), "read"),
            AuthResult::Allowed
        );
        assert_eq!(
            check_auth(&keys, Some("rw-key"), "read"),
            AuthResult::Allowed
        );
    }

    #[test]
    fn test_valid_write() {
        let keys = test_keys();
        assert_eq!(
            check_auth(&keys, Some("rw-key"), "write"),
            AuthResult::Allowed
        );
    }
}
