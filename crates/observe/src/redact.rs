//! Vault-handle redaction.
//!
//! Per `docs/v1.0.0.md` §8.5: secrets pulled from `CredentialVault` are tagged
//! with a sentinel-wrapped envelope at injection time. The [`Redactor`] walks
//! any `serde_json::Value` and replaces every marked envelope with the opaque
//! handle reference `<vault:HANDLE>` — the raw secret value never reaches the
//! bus, the SQLite audit log, or any user-facing surface.
//!
//! The wire shape of a marked secret is:
//!
//! ```text
//! \x00brain-vault:<handle>:<value>\x00
//! ```
//!
//! NUL bytes are illegal in `String` from external sources (HTTP / WS / gRPC
//! all reject them), so the sentinel is collision-free in practice.

use serde_json::Value;

pub const SENTINEL_PREFIX: &str = "\x00brain-vault:";
pub const SENTINEL_SUFFIX: &str = "\x00";

const SEPARATOR: char = ':';

/// Wrap a secret value with the redaction sentinel. Called at vault-injection time.
pub fn mark(handle: &str, value: &str) -> String {
    format!("{SENTINEL_PREFIX}{handle}{SEPARATOR}{value}{SENTINEL_SUFFIX}")
}

/// Stateless redactor. Walks any `serde_json::Value` and rewrites every marked
/// string (including those embedded in larger strings) to the opaque
/// `<vault:HANDLE>` form.
#[derive(Default, Clone, Debug)]
pub struct Redactor;

impl Redactor {
    pub fn new() -> Self {
        Self
    }

    /// Redact in place. Use this when you own the value.
    pub fn redact(&self, value: &mut Value) {
        match value {
            Value::String(s) => {
                if let Some(replaced) = redact_string(s) {
                    *s = replaced;
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.redact(item);
                }
            }
            Value::Object(map) => {
                for (_k, v) in map.iter_mut() {
                    self.redact(v);
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    /// Convenience: clone, redact, return.
    pub fn redacted(&self, value: &Value) -> Value {
        let mut v = value.clone();
        self.redact(&mut v);
        v
    }
}

/// Returns `Some(replaced)` if `s` contained any marked envelope, else `None`.
fn redact_string(s: &str) -> Option<String> {
    if !s.contains(SENTINEL_PREFIX) {
        return None;
    }

    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find(SENTINEL_PREFIX) {
        out.push_str(&rest[..start]);
        let after_prefix = &rest[start + SENTINEL_PREFIX.len()..];
        match after_prefix.find(SENTINEL_SUFFIX) {
            Some(end) => {
                let body = &after_prefix[..end];
                // `body` is "<handle>:<value>"; pull the handle only.
                let handle = body.split(SEPARATOR).next().unwrap_or("");
                out.push_str("<vault:");
                out.push_str(handle);
                out.push('>');
                rest = &after_prefix[end + SENTINEL_SUFFIX.len()..];
            }
            None => {
                // Unterminated sentinel — drop the rest defensively rather than leak it.
                out.push_str("<vault:malformed>");
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    #[test]
    fn mark_then_redact_string_leaves_only_handle() {
        let r = Redactor::new();
        let mut v = Value::String(mark("api-key", "sk-secret-42"));
        r.redact(&mut v);
        assert_eq!(v.as_str().unwrap(), "<vault:api-key>");
    }

    #[test]
    fn embedded_secret_in_larger_string() {
        let r = Redactor::new();
        let s = format!("prefix {} suffix", mark("tok", "abc123"));
        let mut v = Value::String(s);
        r.redact(&mut v);
        assert_eq!(v.as_str().unwrap(), "prefix <vault:tok> suffix");
    }

    #[test]
    fn multiple_secrets_per_string_all_replaced() {
        let r = Redactor::new();
        let s = format!("{} and {}", mark("a", "v1"), mark("b", "v2"));
        let mut v = Value::String(s);
        r.redact(&mut v);
        assert_eq!(v.as_str().unwrap(), "<vault:a> and <vault:b>");
    }

    #[test]
    fn redacts_nested_object_and_array() {
        let r = Redactor::new();
        let mut v = json!({
            "headers": {
                "authorization": mark("bearer", "TOKEN-VALUE"),
                "x-trace": "no-secret-here",
            },
            "body": ["plain", mark("api", "SECRET-IN-ARRAY")],
            "count": 3,
        });
        r.redact(&mut v);
        assert_eq!(
            v["headers"]["authorization"].as_str().unwrap(),
            "<vault:bearer>"
        );
        assert_eq!(v["headers"]["x-trace"].as_str().unwrap(), "no-secret-here");
        assert_eq!(v["body"][1].as_str().unwrap(), "<vault:api>");
        assert_eq!(v["count"].as_i64().unwrap(), 3);
    }

    #[test]
    fn malformed_sentinel_does_not_leak_tail() {
        let r = Redactor::new();
        // Prefix without suffix — should not leak whatever follows.
        let mut v = Value::String(format!("{SENTINEL_PREFIX}handle:LEAK"));
        r.redact(&mut v);
        let out = v.as_str().unwrap();
        assert!(out.contains("<vault:malformed>"), "got: {out}");
        assert!(!out.contains("LEAK"), "raw value leaked: {out}");
    }

    proptest! {
        /// The core invariant: no raw secret value survives redaction.
        #[test]
        fn no_raw_secret_survives_redaction(
            handle in "[a-zA-Z0-9_-]{1,16}",
            // Disallow NUL — it's the sentinel boundary and proptest can pick it.
            // The handle's colon would also break parsing; the regex above already excludes it.
            value in "[^\\x00:]{4,32}",
            wrap in "[^\\x00]{0,32}",
        ) {
            // Inject the marked secret into surrounding context.
            let raw = format!("{wrap}{}{wrap}", mark(&handle, &value));
            let r = Redactor::new();
            let mut v = json!({ "deep": [{ "field": raw }] });
            r.redact(&mut v);
            let serialized = v.to_string();
            prop_assert!(!serialized.contains(&value),
                "secret leaked through redaction: {serialized}");
            prop_assert!(serialized.contains(&format!("<vault:{handle}>")),
                "handle missing from redaction: {serialized}");
        }
    }
}
