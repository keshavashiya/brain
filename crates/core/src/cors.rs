//! Shared CORS configuration for all HTTP-based adapters.
//!
//! Brain is a local daemon — remote origins are blocked to prevent cross-site
//! requests from untrusted web pages.

use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

/// Origin matcher for localhost-only adapters.
///
/// Issue 57: the previous implementation used `starts_with("http://localhost:")`,
/// which would also accept an origin like `http://localhost:evil.example` —
/// browsers will not normally produce such an Origin header, but the prefix
/// match is a footgun the moment something else relays it. This version
/// requires:
///   * an exact `http(s)://localhost` or `http(s)://127.0.0.1` (no port), OR
///   * the same with a `:` followed by **digits only** (1-5 chars, 0-65535).
///
/// No userinfo (`@`), path (`/`), query (`?`), or fragment (`#`) is permitted
/// after the host, because none of those are valid in an Origin per RFC 6454.
fn is_localhost_origin<B: AsRef<[u8]>>(origin: &B) -> bool {
    let bytes = origin.as_ref();
    for prefix in [
        b"http://localhost".as_slice(),
        b"https://localhost".as_slice(),
        b"http://127.0.0.1".as_slice(),
        b"https://127.0.0.1".as_slice(),
    ] {
        if let Some(rest) = bytes.strip_prefix(prefix) {
            if rest.is_empty() {
                return true;
            }
            if rest.first() == Some(&b':') && is_valid_port_suffix(&rest[1..]) {
                return true;
            }
        }
    }
    false
}

fn is_valid_port_suffix(bytes: &[u8]) -> bool {
    if bytes.is_empty() || bytes.len() > 5 {
        return false;
    }
    if !bytes.iter().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let port: u32 = bytes
        .iter()
        .fold(0u32, |acc, b| acc * 10 + (b - b'0') as u32);
    port <= u16::MAX as u32
}

/// CORS layer restricted to localhost origins.
///
/// Allows requests from `http(s)://127.0.0.1[:port]` and
/// `http(s)://localhost[:port]` exactly — no prefix tricks.
/// All methods and headers are permitted for matching origins.
pub fn localhost_cors() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _req| {
            is_localhost_origin(origin)
        }))
        .allow_methods(AllowMethods::any())
        .allow_headers(AllowHeaders::any())
}

#[cfg(test)]
mod tests {
    use super::is_localhost_origin;

    fn ok(s: &str) {
        assert!(is_localhost_origin(&s.as_bytes()), "should accept {s}");
    }
    fn no(s: &str) {
        assert!(!is_localhost_origin(&s.as_bytes()), "should reject {s}");
    }

    #[test]
    fn accepts_bare_localhost() {
        ok("http://localhost");
        ok("https://localhost");
        ok("http://127.0.0.1");
        ok("https://127.0.0.1");
    }

    #[test]
    fn accepts_localhost_with_port() {
        ok("http://localhost:19789");
        ok("http://localhost:0");
        ok("http://localhost:65535");
        ok("https://127.0.0.1:443");
    }

    #[test]
    fn rejects_prefix_lookalikes() {
        no("http://localhost.attacker.com");
        no("http://localhost:8080.attacker.com");
        no("http://localhost:8080@attacker.com");
        no("http://localhost:8080/evil");
        no("http://localhost:8080?x=1");
        no("http://localhost:8080#frag");
        no("http://127.0.0.1.attacker.com");
    }

    #[test]
    fn rejects_remote_origins() {
        no("http://example.com");
        no("https://attacker.com");
        no("file:///etc/passwd");
        no("");
    }

    #[test]
    fn rejects_oversized_or_nonnumeric_port() {
        no("http://localhost:999999");
        no("http://localhost:65536");
        no("http://localhost:abc");
        no("http://localhost:");
    }
}
