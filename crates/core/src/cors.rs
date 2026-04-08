//! Shared CORS configuration for all HTTP-based adapters.
//!
//! Brain is a local daemon — remote origins are blocked to prevent cross-site
//! requests from untrusted web pages.

use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

fn is_localhost_origin<B: AsRef<[u8]>>(origin: &B) -> bool {
    let bytes = origin.as_ref();
    // Exact matches only — reject localhost.attacker.com
    bytes == b"http://localhost"
        || bytes == b"http://localhost:0"
        || bytes == b"https://localhost"
        || bytes == b"https://localhost:0"
        || bytes.starts_with(b"http://localhost:")
        || bytes.starts_with(b"https://localhost:")
        || bytes == b"http://127.0.0.1"
        || bytes == b"https://127.0.0.1"
        || bytes.starts_with(b"http://127.0.0.1:")
        || bytes.starts_with(b"https://127.0.0.1:")
}

/// CORS layer restricted to localhost origins.
///
/// Allows requests from `http(s)://127.0.0.1:*` and `http(s)://localhost:*`.
/// All methods and headers are permitted for matching origins.
pub fn localhost_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _req| {
            is_localhost_origin(origin)
        }))
        .allow_methods(AllowMethods::any())
        .allow_headers(AllowHeaders::any())
}
