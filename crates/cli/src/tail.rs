//! `brain tail` — observability tap for the daemon's BrainEvent bus.
//!
//! Subscribes to the daemon's `GET /v1/events` SSE stream and prints every
//! `brain_event` payload as a JSON line to stdout. Provided so observability
//! works when the UI is down, when the user is on a headless SSH session,
//! and when nothing else has access.

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::bootstrap;

#[derive(Debug, Default, Clone)]
pub struct TailFilter {
    /// `BrainEvent` variant discriminant (e.g. `signal_received`).
    pub kind: Option<String>,
    pub tool_id: Option<String>,
    pub principal: Option<String>,
    /// RFC3339 timestamp; only events with `ts >= since` are forwarded.
    pub since: Option<String>,
}

impl TailFilter {
    fn append_query(&self, url: &mut String) {
        let mut first = true;
        let mut push = |k: &str, v: &str| {
            if v.is_empty() {
                return;
            }
            url.push(if first { '?' } else { '&' });
            first = false;
            url.push_str(k);
            url.push('=');
            url.push_str(&urlencode(v));
        };
        if let Some(v) = &self.kind {
            push("kind", v);
        }
        if let Some(v) = &self.tool_id {
            push("tool_id", v);
        }
        if let Some(v) = &self.principal {
            push("principal", v);
        }
        if let Some(v) = &self.since {
            push("since", v);
        }
    }
}

/// Minimal RFC3986 component encoder (alnum + `-`, `_`, `.`, `~` are literal).
/// Avoids pulling in a URL crate for one tiny use.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Resolve the API key from `BRAIN_API_KEY` env or the first configured key.
fn resolve_api_key(config: &brain_core::BrainConfig) -> String {
    let env = std::env::var("BRAIN_API_KEY").unwrap_or_default();
    if !env.is_empty() {
        return env;
    }
    config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default()
}

/// Drive the tail loop until the stream closes or the user interrupts.
pub async fn cmd_tail(config: &brain_core::BrainConfig, filter: TailFilter) -> Result<()> {
    let base_url = bootstrap::require_daemon(config).await?;
    let mut url = format!("{base_url}/v1/events");
    filter.append_query(&mut url);
    let api_key = resolve_api_key(config);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(0)) // long-poll
        .build()?;
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Accept", "text/event-stream")
        .send()
        .await
        .context("connect to daemon SSE")?;
    if !resp.status().is_success() {
        anyhow::bail!("daemon returned {} for {url}", resp.status());
    }

    let mut stdout = BufWriter::new(tokio::io::stdout());
    let mut event_name = String::new();
    let mut data_buf = String::new();
    let mut byte_stream = resp.bytes_stream();
    let mut leftover = String::new();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("daemon stream errored")?;
        leftover.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(nl) = leftover.find('\n') {
            let line = leftover[..nl].trim_end_matches('\r').to_string();
            leftover.drain(..=nl);
            if line.is_empty() {
                if event_name == "brain_event" && !data_buf.is_empty() {
                    stdout.write_all(data_buf.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
                event_name.clear();
                data_buf.clear();
            } else if let Some(rest) = line.strip_prefix("event:") {
                event_name = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !data_buf.is_empty() {
                    data_buf.push('\n');
                }
                data_buf.push_str(rest.trim_start());
            }
            // Comments (lines starting ':') and other fields are ignored.
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_builder_skips_empty_fields() {
        let f = TailFilter {
            kind: Some("signal_received".into()),
            tool_id: None,
            principal: None,
            since: None,
        };
        let mut url = String::from("http://x/v1/events");
        f.append_query(&mut url);
        assert_eq!(url, "http://x/v1/events?kind=signal_received");
    }

    #[test]
    fn query_builder_concatenates_multiple_filters() {
        let f = TailFilter {
            kind: Some("tool_call_started".into()),
            tool_id: Some("mcp:fs:read".into()),
            principal: None,
            since: Some("2026-05-14T00:00:00Z".into()),
        };
        let mut url = String::from("http://x/v1/events");
        f.append_query(&mut url);
        assert!(url.contains("kind=tool_call_started"));
        assert!(url.contains("tool_id=mcp%3Afs%3Aread"));
        assert!(url.contains("since=2026-05-14T00%3A00%3A00Z"));
    }

    #[test]
    fn urlencode_handles_reserved_chars() {
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("a:b/c"), "a%3Ab%2Fc");
        assert_eq!(urlencode("a-b_c.d~e"), "a-b_c.d~e");
    }
}
