//! Shared CORS configuration for all HTTP-based adapters.
//!
//! Brain is a local daemon — remote origins are blocked to prevent cross-site
//! requests from untrusted web pages.

use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

/// CORS layer restricted to localhost origins.
///
/// Allows requests from `http(s)://127.0.0.1:*` and `http(s)://localhost:*`.
/// All methods and headers are permitted for matching origins.
pub fn localhost_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _req| {
            let bytes = origin.as_bytes();
            bytes.starts_with(b"http://127.0.0.1")
                || bytes.starts_with(b"http://localhost")
                || bytes.starts_with(b"https://127.0.0.1")
                || bytes.starts_with(b"https://localhost")
        }))
        .allow_methods(AllowMethods::any())
        .allow_headers(AllowHeaders::any())
}
