//! Shared HTTP send helper used by transports that implement outbound via
//! a preset's [`SendSpec`]. Keeps request templating in one place.

use std::collections::HashMap;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use serde_json::Value;

use crate::error::ChannelError;
use crate::transport::jsonpath::JsonPath;
use crate::transport::preset::{render_template, SendSpec};
use crate::transport::MessageHandle;
use crate::types::DeliveryIntent;

/// Execute an outbound send using the given [`SendSpec`]. Handles template
/// substitution for credential/content/id/reply_to plus any metadata keys
/// and returns a [`MessageHandle`] with an optional platform-side id.
pub async fn http_send(
    client: &reqwest::Client,
    send: &SendSpec,
    credential: &str,
    intent: &DeliveryIntent,
) -> Result<MessageHandle, ChannelError> {
    let mut vars: HashMap<&str, &str> = HashMap::new();
    vars.insert("credential", credential);
    vars.insert("content", intent.content.as_str());
    vars.insert("id", intent.id.as_str());
    let reply_to = intent.metadata.get("reply_to").map(String::as_str);
    if let Some(r) = reply_to {
        vars.insert("reply_to", r);
    }
    for (k, v) in &intent.metadata {
        if !["credential", "content", "id", "reply_to"].contains(&k.as_str()) {
            vars.insert(k.as_str(), v.as_str());
        }
    }

    // URL template: percent-encoding is the caller's job; raw substitution
    // is fine for `{credential}` and cursor-style values.
    let url = render_template(&send.url_template, &vars);

    // Body template: when the wire format is JSON, the values must be
    // JSON-string-escaped or any quote/newline/backslash in `content` will
    // produce a malformed body and the platform will reject it (Telegram
    // 400 "can't parse entities"). Escape per content type.
    let body = if is_json_content_type(&send.content_type) {
        let escaped: HashMap<&str, String> = vars
            .iter()
            .map(|(k, v)| (*k, json_escape_inner(v)))
            .collect();
        let escaped_refs: HashMap<&str, &str> =
            escaped.iter().map(|(k, v)| (*k, v.as_str())).collect();
        render_template(&send.body_template, &escaped_refs)
    } else {
        render_template(&send.body_template, &vars)
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        CONTENT_TYPE,
        HeaderValue::from_str(&send.content_type)
            .map_err(|e| ChannelError::Relay(format!("invalid content_type: {e}")))?,
    );
    for (k, v) in &send.headers {
        if let (Ok(name), Ok(val)) = (
            HeaderName::try_from(k.as_str()),
            HeaderValue::from_str(&render_template(v, &vars)),
        ) {
            headers.insert(name, val);
        }
    }

    let resp = client
        .request(send.method.as_reqwest(), &url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|e| ChannelError::Relay(format!("send request: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(ChannelError::Relay(format!("send status {status}: {text}")));
    }

    let platform_id = resp.json::<Value>().await.ok().and_then(|v| {
        // Common shapes: Telegram $.result.message_id, Slack $.ts,
        // Discord $.id. Try each in order; first hit wins.
        for path in ["$.result.message_id", "$.ts", "$.id", "$.message_id"] {
            if let Ok(p) = JsonPath::parse(path) {
                if let Some(s) = p.eval_string(&v) {
                    return Some(s);
                }
            }
        }
        None
    });

    let mut handle = MessageHandle::new(&intent.id);
    if let Some(id) = platform_id {
        handle = handle.with_platform_id(id);
    }
    Ok(handle)
}

fn is_json_content_type(ct: &str) -> bool {
    let lower = ct.to_ascii_lowercase();
    lower.starts_with("application/json") || lower.contains("+json")
}

/// Escape the *inner* characters of a JSON string — caller is responsible
/// for the surrounding quotes (the preset body_template already includes
/// them). Handles the minimum required by RFC 8259.
fn json_escape_inner(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escape_basics() {
        assert_eq!(json_escape_inner("hi"), "hi");
        assert_eq!(json_escape_inner("a\"b"), "a\\\"b");
        assert_eq!(json_escape_inner("a\nb"), "a\\nb");
        assert_eq!(json_escape_inner("a\\b"), "a\\\\b");
        assert_eq!(json_escape_inner("a\tb"), "a\\tb");
    }

    #[test]
    fn json_escape_control_char() {
        assert_eq!(json_escape_inner("\x01"), "\\u0001");
    }

    #[test]
    fn json_content_type_detection() {
        assert!(is_json_content_type("application/json"));
        assert!(is_json_content_type("application/json; charset=utf-8"));
        assert!(is_json_content_type("application/vnd.api+json"));
        assert!(!is_json_content_type("text/plain"));
        assert!(!is_json_content_type("application/x-www-form-urlencoded"));
    }
}
