//! API-key authentication interceptor and identity-store principal resolution
//! for inbound gRPC requests.

use std::sync::Arc;

use tonic::{Request, Status};

/// Tonic interceptor that validates API key authentication with permission checking.
///
/// Accepts the key from either `x-api-key` or `authorization` (Bearer) metadata.
/// Returns `UNAUTHENTICATED` / `PERMISSION_DENIED` if no valid key is found.
/// If no keys are configured, all requests are allowed (open mode).
pub(crate) fn auth_interceptor(
    req: Request<()>,
    api_keys: &[brain::ApiKeyConfig],
    rate_limits: Option<&Arc<::resilience::RateLimitRegistry>>,
) -> Result<Request<()>, Status> {
    let metadata = req.metadata();

    // Extract key from x-api-key or authorization header
    let provided_key = metadata
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            metadata.get("authorization").and_then(|v| {
                v.to_str()
                    .ok()
                    .and_then(|s| brain::auth::extract_bearer_from_value(s).or(Some(s)))
            })
        });

    // gRPC requests can both read and write, so require write permission.
    let result = brain::check_auth(api_keys, provided_key, "write");
    match result {
        brain::AuthResult::Allowed => {}
        brain::AuthResult::InsufficientPermission => {
            return Err(Status::permission_denied(
                result.error_message("write").unwrap_or_default(),
            ));
        }
        _ => {
            return Err(Status::unauthenticated(
                result.error_message("write").unwrap_or_default(),
            ));
        }
    }

    // Per-client rate limit (Issue 51). Only applied when both a
    // registry is wired and the caller presented a key — open mode
    // (no `api_keys` configured) intentionally has no throttling.
    if let (Some(registry), Some(key)) = (rate_limits, provided_key) {
        let limiter = registry.get_or_create(key);
        if !limiter.try_acquire() {
            tracing::warn!("gRPC rate limit exceeded for client");
            return Err(Status::resource_exhausted("Too many requests"));
        }
    }

    Ok(req)
}

/// Resolve the `Principal` for a gRPC request by reading the key from
/// `x-api-key` / `authorization` metadata, looking up its configured
/// `agent_id`, and asking the `IdentityStore` for the principal.
/// Returns `None` when any step is missing — Signal.principal then stays
/// `None` and the pipeline's identity gate is skipped (back-compat).
pub(crate) async fn resolve_principal_from_metadata<T>(
    req: &Request<T>,
    api_keys: &[brain::ApiKeyConfig],
    processor: &signal::SignalProcessor,
) -> Option<identity::Principal> {
    let metadata = req.metadata();
    let key = metadata
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            metadata.get("authorization").and_then(|v| {
                v.to_str()
                    .ok()
                    .and_then(|s| brain::auth::extract_bearer_from_value(s).or(Some(s)))
            })
        })?;
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
