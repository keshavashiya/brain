//! Tower middleware applied to `/v1/*` routes.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use resilience::RateLimitRegistry;

use crate::auth;

/// Per-client (per-API-key) rate-limit middleware (Issue 51).
///
/// Anonymous requests pass through — they hit the auth handler and get a
/// 401 instead. Authenticated requests check the registry's bucket for
/// the caller's key and are rejected with HTTP 429 when drained. Detail
/// about retry-after is intentionally omitted from the public body; the
/// log line carries the key fingerprint for operators.
pub async fn rate_limit(
    registry: Arc<RateLimitRegistry>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(key) = auth::extract_bearer(request.headers()) {
        let limiter = registry.get_or_create(key);
        if !limiter.try_acquire() {
            let fingerprint = key_fingerprint(key);
            tracing::warn!(
                client = %fingerprint,
                "rate limit exceeded for client"
            );
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [("content-type", "application/json")],
                Body::from(
                    serde_json::json!({
                        "error": "rate_limited",
                        "message": "Too many requests",
                    })
                    .to_string(),
                ),
            )
                .into_response();
        }
    }
    next.run(request).await
}

/// Render the first 8 chars of an API key for log correlation without
/// dumping the full secret.
fn key_fingerprint(key: &str) -> String {
    let prefix: String = key.chars().take(8).collect();
    format!("{prefix}…")
}
