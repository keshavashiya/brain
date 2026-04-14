//! Authentication middleware helpers for HTTP routes.

use axum::http::{HeaderMap, StatusCode};

use crate::state::AppState;

/// Extract the raw key from `Authorization: Bearer <key>`.
pub fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(brain_core::auth::extract_bearer_from_value)
}

/// Check that the request carries a valid key with the given permission.
/// Returns `Err((StatusCode::UNAUTHORIZED, message))` on failure.
pub fn check_auth(
    state: &AppState,
    headers: &HeaderMap,
    permission: &str,
) -> Result<(), (StatusCode, String)> {
    let provided_key = extract_bearer(headers);
    let result = brain_core::check_auth(&state.api_keys, provided_key, permission);
    if result.is_allowed() {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            result
                .error_message(permission)
                .unwrap_or_else(|| "Unauthorized".to_string()),
        ))
    }
}
