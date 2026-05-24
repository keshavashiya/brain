//! Field-level input validation for HTTP request bodies.
//!
//! Issue 56: oversized or malformed fields used to flow all the way to
//! `signal::pipeline` before being rejected. These checks fail fast at the
//! adapter boundary with a 400 so the pipeline only ever sees bounded input.
//!
//! Length budgets are looser than the 1 MiB body cap on `/v1/*` so the body
//! limit stays the outer wall and these are the inner shape checks.

use axum::http::StatusCode;

/// Max byte length of `SignalRequest.content`. Tuned to leave headroom for
/// metadata + transport framing inside the 1 MiB per-request body limit.
pub const MAX_CONTENT_BYTES: usize = 256 * 1024;
/// Max byte length of `SearchRequest.query`. Search queries that exceed
/// this are almost certainly accidental dumps.
pub const MAX_QUERY_BYTES: usize = 16 * 1024;
/// Max byte length of short identifier-ish fields (namespace, agent,
/// session_id, sender, channel).
pub const MAX_IDENT_BYTES: usize = 256;
/// Hard ceiling on `top_k`. The recall layer ignores anything above its own
/// configured max anyway; this just prevents a client from forcing the
/// adapter to round-trip absurd values.
pub const MAX_TOP_K: usize = 1000;

fn bad(reason: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, reason.into())
}

/// Reject empty / oversized free-text payloads.
pub fn check_content(content: &str) -> Result<(), (StatusCode, String)> {
    if content.is_empty() {
        return Err(bad("content must not be empty"));
    }
    if content.len() > MAX_CONTENT_BYTES {
        return Err(bad(format!(
            "content exceeds {MAX_CONTENT_BYTES} bytes ({} given)",
            content.len()
        )));
    }
    Ok(())
}

/// Reject empty / oversized search queries.
pub fn check_query(q: &str) -> Result<(), (StatusCode, String)> {
    if q.is_empty() {
        return Err(bad("query must not be empty"));
    }
    if q.len() > MAX_QUERY_BYTES {
        return Err(bad(format!(
            "query exceeds {MAX_QUERY_BYTES} bytes ({} given)",
            q.len()
        )));
    }
    Ok(())
}

/// Reject empty / oversized short identifier fields. Also rejects ASCII
/// control characters which have no place in any identifier and would
/// otherwise turn into junk in logs.
pub fn check_short_ident(name: &str, value: &str) -> Result<(), (StatusCode, String)> {
    if value.is_empty() {
        return Err(bad(format!("{name} must not be empty")));
    }
    if value.len() > MAX_IDENT_BYTES {
        return Err(bad(format!(
            "{name} exceeds {MAX_IDENT_BYTES} bytes ({} given)",
            value.len()
        )));
    }
    if value.bytes().any(|b| b.is_ascii_control()) {
        return Err(bad(format!("{name} contains control characters")));
    }
    Ok(())
}

/// Bound `top_k` to a sane window. `None` means caller will fall back to a
/// pipeline-internal default and is fine.
pub fn check_top_k(top_k: Option<usize>) -> Result<(), (StatusCode, String)> {
    match top_k {
        None => Ok(()),
        Some(0) => Err(bad("top_k must be ≥ 1")),
        Some(k) if k > MAX_TOP_K => Err(bad(format!("top_k exceeds {MAX_TOP_K} ({k} given)"))),
        Some(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_bounds() {
        assert!(check_content("").is_err());
        assert!(check_content("hi").is_ok());
        let big = "x".repeat(MAX_CONTENT_BYTES + 1);
        assert!(check_content(&big).is_err());
    }

    #[test]
    fn query_bounds() {
        assert!(check_query("").is_err());
        assert!(check_query("rust").is_ok());
        let big = "q".repeat(MAX_QUERY_BYTES + 1);
        assert!(check_query(&big).is_err());
    }

    #[test]
    fn short_ident_rejects_control_chars() {
        assert!(check_short_ident("namespace", "personal").is_ok());
        assert!(check_short_ident("namespace", "").is_err());
        assert!(check_short_ident("namespace", "bad\nname").is_err());
        let big = "n".repeat(MAX_IDENT_BYTES + 1);
        assert!(check_short_ident("namespace", &big).is_err());
    }

    #[test]
    fn top_k_bounds() {
        assert!(check_top_k(None).is_ok());
        assert!(check_top_k(Some(0)).is_err());
        assert!(check_top_k(Some(10)).is_ok());
        assert!(check_top_k(Some(MAX_TOP_K)).is_ok());
        assert!(check_top_k(Some(MAX_TOP_K + 1)).is_err());
    }
}
