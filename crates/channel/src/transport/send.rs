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

    let url = render_template(&send.url_template, &vars);
    let body = render_template(&send.body_template, &vars);

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
