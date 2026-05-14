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

/// Resolve the requesting `Principal` from the request headers.
///
/// Returns `None` when:
/// - no `Authorization: Bearer` header is present, OR
/// - the key has no `agent_id` configured (back-compat — pre-Phase-1 keys), OR
/// - no `IdentityStore` is wired on the `SignalProcessor`, OR
/// - the agent_id is unknown to the identity store.
///
/// `None` propagates as `Signal.principal = None`, which the pipeline's
/// identity gate treats as "skip enforcement". Adapters that want to
/// REQUIRE a principal should refuse the request when this returns `None`.
pub async fn resolve_principal(
    state: &AppState,
    headers: &HeaderMap,
) -> Option<identity::Principal> {
    let key = extract_bearer(headers)?;
    let agent_id = state
        .api_keys
        .iter()
        .find(|k| k.key == key)
        .and_then(|k| k.agent_id.clone())?;
    let store = state.processor.identity_store()?;
    store
        .principal_for(&identity::AgentHint::AgentId(agent_id.into()))
        .await
        .ok()
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
