//! Per-connection API-key validation and `IdentityStore`-backed principal
//! resolution for inbound WebSocket sessions.

use brain::ApiKeyConfig;

/// Returns true if `key` is valid and has write permission (WS connections can both read and write).
pub(crate) fn validate_key(api_keys: &[ApiKeyConfig], key: &str) -> bool {
    // WS connections need write permission since they can send signals.
    brain::check_auth(api_keys, Some(key), "write").is_allowed()
}

/// Resolve the `Principal` bound to this connection from the validated key.
/// Returns `None` for keys without `agent_id` (back-compat) or when no
/// `IdentityStore` is wired on the processor.
pub(crate) async fn resolve_principal(
    api_keys: &[ApiKeyConfig],
    key: &str,
    processor: &signal::SignalProcessor,
) -> Option<identity::Principal> {
    let agent_id = api_keys
        .iter()
        .find(|k| k.key == key)
        .and_then(|k| k.agent_id.clone())?;
    let store = processor.identity_store()?;
    store
        .principal_for(&identity::AgentHint::AgentId(agent_id.into()))
        .await
        .ok()
}
