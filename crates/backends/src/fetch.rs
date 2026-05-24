//! URL-fetch backend — bounded HTTP GET + HTML→text reduction.
//!
//! Used by `cortex::actions::ActionDispatcher` to inline the body of
//! user-pasted URLs alongside web search results. The fetcher imposes
//! its own timeout and body cap so a runaway page can't blow the
//! LLM context window or stall the answer.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use brain::metrics::SubsystemMetrics;
use cortex::actions::{ActionError, FetchedPage, UrlFetchBackend};

use crate::resilience::{http_breaker, resilient_send, CircuitBreaker};

/// Hard cap on bytes read from any single page. Anything bigger is
/// truncated *after* HTML stripping so most real pages still produce
/// useful text.
const MAX_BODY_BYTES: usize = 256 * 1024;
/// Hard cap on cleaned-text bytes returned to the dispatcher.
const MAX_TEXT_BYTES: usize = 4 * 1024;
/// Connect/read timeout for a single fetch. Tight on purpose — the
/// answering LLM is waiting and a slow site shouldn't hold up the
/// whole response.
const DEFAULT_FETCH_TIMEOUT: Duration = Duration::from_secs(8);

/// Default HTTP fetcher: GET, follow redirects, strip HTML, return
/// truncated text. No JavaScript execution, no auth, no cookies.
pub struct BasicUrlFetcher {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
    /// When false (production default), `assert_url_safe` rejects URLs whose
    /// resolved address is in any reserved range — loopback, private,
    /// link-local, etc. Tests and a deployment that legitimately points the
    /// LLM at a localhost service flip this on.
    allow_internal: bool,
}

impl BasicUrlFetcher {
    pub fn new(resilience: &brain::config::ResilienceConfig) -> anyhow::Result<Self> {
        Self::new_with_metrics(resilience, None)
    }

    pub fn new_with_metrics(
        resilience: &brain::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_FETCH_TIMEOUT)
            .user_agent("brainos/url-fetch (+https://github.com/keshavashiya/brain)")
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| anyhow::anyhow!("URL fetch client init failed: {e}"))?;
        let cb = http_breaker(
            "url-fetch",
            resilience.circuit_breaker_threshold,
            resilience.circuit_breaker_cooldown_secs,
            metrics,
        );
        Ok(Self {
            client,
            circuit_breaker: Arc::new(cb),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
            allow_internal: false,
        })
    }

    /// Opt out of the SSRF guard for reserved-range targets. Used by tests
    /// (mockito binds 127.0.0.1) and by deployments pointing the LLM at a
    /// trusted internal service.
    pub fn with_allow_internal(mut self, allow: bool) -> Self {
        self.allow_internal = allow;
        self
    }
}

#[async_trait::async_trait]
impl UrlFetchBackend for BasicUrlFetcher {
    async fn fetch(&self, url: &str) -> Result<FetchedPage, ActionError> {
        assert_url_safe(url, self.allow_internal).await?;
        let client = self.client.clone();
        let url_owned = url.to_string();
        let response = resilient_send(
            || client.get(&url_owned),
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(ActionError::ExecutionFailed(format!(
                "{url} returned HTTP {}",
                response.status()
            )));
        }

        let final_url = response.url().to_string();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| ActionError::ExecutionFailed(format!("body read failed: {e}")))?;

        let head = &bytes[..bytes.len().min(MAX_BODY_BYTES)];
        let raw = String::from_utf8_lossy(head);
        let title = extract_title(&raw).unwrap_or_else(|| final_url.clone());
        let text = html_to_text(&raw);
        let text = if text.len() > MAX_TEXT_BYTES {
            let mut t: String = text.chars().take(MAX_TEXT_BYTES).collect();
            t.push_str("\n…[truncated]");
            t
        } else {
            text
        };

        Ok(FetchedPage {
            url: final_url,
            title,
            text,
        })
    }
}

/// Issue 122: SSRF guard. The fetcher is reachable from LLM-controlled
/// inputs (web-search hits, user-pasted URLs in chat) so we have to assume
/// the URL is adversarial. Reject:
///   * non-`http(s)` schemes (file:, gopher:, ftp:, …)
///   * embedded credentials (`user:pass@`) which most reqwest builds honor
///   * IP-literal hosts in any reserved range
///   * DNS hostnames whose resolved addresses include any reserved range
///
/// Remaining gap: a TOCTOU between this resolve and reqwest's own resolve
/// could theoretically be exploited by DNS rebinding. Closing it would
/// require pinning the connect target to a vetted IP via a custom
/// reqwest::dns::Resolve — deferred. The current resolver still raises
/// the floor materially.
pub(crate) async fn assert_url_safe(url: &str, allow_internal: bool) -> Result<(), ActionError> {
    let parsed = url::Url::parse(url)
        .map_err(|e| ActionError::InvalidArguments(format!("invalid url '{url}': {e}")))?;

    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(ActionError::InvalidArguments(format!(
                "url-fetch requires http(s):// scheme, got {other}://"
            )));
        }
    }

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(ActionError::InvalidArguments(
            "url-fetch refuses URLs with embedded credentials".into(),
        ));
    }

    let host = parsed
        .host()
        .ok_or_else(|| ActionError::InvalidArguments("url-fetch requires a host".into()))?;

    match host {
        url::Host::Ipv4(v4) => reject_reserved_ip(IpAddr::V4(v4), &host.to_string(), allow_internal),
        url::Host::Ipv6(v6) => reject_reserved_ip(IpAddr::V6(v6), &host.to_string(), allow_internal),
        url::Host::Domain(name) => {
            // Hostname — resolve and reject if any answer is reserved.
            // `lookup_host` needs a `host:port` string; synthesise a port
            // if the URL omits one.
            let port = parsed.port_or_known_default().unwrap_or(0);
            let addrs = tokio::net::lookup_host((name, port))
                .await
                .map_err(|e| {
                    ActionError::ExecutionFailed(format!("dns lookup for {name} failed: {e}"))
                })?
                .collect::<Vec<_>>();

            if addrs.is_empty() {
                return Err(ActionError::ExecutionFailed(format!(
                    "dns lookup for {name} returned no addresses"
                )));
            }
            for sa in addrs {
                reject_reserved_ip(sa.ip(), name, allow_internal)?;
            }
            Ok(())
        }
    }
}

fn reject_reserved_ip(ip: IpAddr, host: &str, allow_internal: bool) -> Result<(), ActionError> {
    if allow_internal {
        return Ok(());
    }
    if is_reserved(ip) {
        return Err(ActionError::InvalidArguments(format!(
            "url-fetch refuses host '{host}' resolving to reserved address {ip}"
        )));
    }
    Ok(())
}

fn is_reserved(ip: IpAddr) -> bool {
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return true;
    }
    match ip {
        IpAddr::V4(v4) => {
            if v4.is_private() || v4.is_link_local() || v4.is_broadcast() || v4.is_documentation() {
                return true;
            }
            // Cloud metadata endpoints (AWS / GCP / Azure all live at 169.254.169.254).
            // 169.254.0.0/16 is link-local and already caught above; explicit for clarity.
            if v4.octets()[0] == 169 && v4.octets()[1] == 254 {
                return true;
            }
            // Carrier-grade NAT (RFC 6598).
            if v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1]) {
                return true;
            }
            // Reserved for future use (240.0.0.0/4) — `is_reserved` is unstable; check manually.
            if v4.octets()[0] >= 240 {
                return true;
            }
            false
        }
        IpAddr::V6(v6) => {
            // Unique local (fc00::/7).
            if (v6.segments()[0] & 0xfe00) == 0xfc00 {
                return true;
            }
            // Link-local (fe80::/10).
            if (v6.segments()[0] & 0xffc0) == 0xfe80 {
                return true;
            }
            // IPv4-mapped — recurse on the embedded IPv4 so we reject
            // ::ffff:127.0.0.1 etc.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_reserved(IpAddr::V4(v4));
            }
            false
        }
    }
}

/// Pull the `<title>` content from raw HTML. Case-insensitive on the
/// tag name so `<TITLE>` and `<Title>` both match.
fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    // Skip past the closing `>` of the opening tag (which may include attrs).
    let after_open = lower[start..].find('>').map(|i| start + i + 1)?;
    let end_rel = lower[after_open..].find("</title>")?;
    let raw = &html[after_open..after_open + end_rel];
    let cleaned = decode_entities(raw).trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Strip HTML tags + scripts/styles + decode common entities. Not a
/// full parser — good enough to feed an LLM a readable excerpt.
fn html_to_text(html: &str) -> String {
    let mut s = String::with_capacity(html.len());
    let stripped = strip_block(html, "script");
    let stripped = strip_block(&stripped, "style");
    let stripped = strip_block(&stripped, "noscript");
    let mut in_tag = false;
    for ch in stripped.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                s.push(' ');
            }
            _ if in_tag => {}
            _ => s.push(ch),
        }
    }
    let decoded = decode_entities(&s);
    collapse_whitespace(&decoded)
}

/// Remove every `<tag …>…</tag>` block (case-insensitive on the tag).
fn strip_block(input: &str, tag: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let open_pat = format!("<{tag}");
    let close_pat = format!("</{tag}>");
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    while cursor < input.len() {
        match lower[cursor..].find(&open_pat) {
            None => {
                out.push_str(&input[cursor..]);
                break;
            }
            Some(rel) => {
                let abs_open = cursor + rel;
                out.push_str(&input[cursor..abs_open]);
                let after_open = match lower[abs_open..].find('>') {
                    Some(i) => abs_open + i + 1,
                    None => break,
                };
                match lower[after_open..].find(&close_pat) {
                    Some(end_rel) => {
                        cursor = after_open + end_rel + close_pat.len();
                    }
                    None => {
                        cursor = after_open;
                    }
                }
            }
        }
    }
    out
}

/// Decode the handful of HTML entities common enough to matter for
/// readable LLM context. Anything else is passed through verbatim.
fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

/// Collapse runs of whitespace (including newlines) into single spaces,
/// preserving paragraph breaks (a sequence with two or more newlines
/// becomes a single `\n`).
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_was_space = false;
    let mut newline_run = 0usize;
    for ch in s.chars() {
        if ch == '\n' {
            newline_run += 1;
            continue;
        }
        if ch.is_whitespace() {
            if !last_was_space && newline_run == 0 {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }
        if newline_run >= 2 {
            out.push('\n');
        } else if newline_run == 1 && !last_was_space {
            out.push(' ');
        }
        newline_run = 0;
        last_was_space = false;
        out.push(ch);
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_title_handles_attrs_and_case() {
        let html = "<html><head><TITLE id=\"x\">Hello World</TITLE></head><body/></html>";
        assert_eq!(extract_title(html).as_deref(), Some("Hello World"));
    }

    #[test]
    fn extract_title_returns_none_when_missing() {
        assert!(extract_title("<html><body>no title</body></html>").is_none());
    }

    #[test]
    fn html_to_text_strips_scripts_and_styles() {
        let html = "<html><head><style>body{color:red}</style></head>\
                    <body><script>alert(1)</script><p>Hello <b>world</b>!</p></body></html>";
        let txt = html_to_text(html);
        assert!(!txt.contains("alert"));
        assert!(!txt.contains("color:red"));
        assert!(txt.contains("Hello"));
        assert!(txt.contains("world"));
    }

    #[test]
    fn html_to_text_decodes_common_entities() {
        let html = "<p>tom &amp; jerry &lt;3 &nbsp; coffee</p>";
        let txt = html_to_text(html);
        assert!(txt.contains("tom & jerry"));
        assert!(txt.contains("<3"));
    }

    #[test]
    fn collapse_whitespace_keeps_paragraph_breaks() {
        let s = "alpha\n\nbeta   gamma\n\n\ndelta";
        let out = collapse_whitespace(s);
        // Multi-newline sequences collapse to a single \n; runs of spaces collapse to one.
        assert_eq!(out, "alpha\nbeta gamma\ndelta");
    }

    fn fast_resilience() -> brain::config::ResilienceConfig {
        brain::config::ResilienceConfig {
            max_retries: 0,
            retry_base_ms: 1,
            circuit_breaker_threshold: 5,
            circuit_breaker_cooldown_secs: 60,
        }
    }

    #[tokio::test]
    async fn fetch_returns_text_and_title() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/page")
            .with_status(200)
            .with_header("content-type", "text/html; charset=utf-8")
            .with_body(
                "<html><head><title>Demo Page</title></head>\
                 <body><script>x=1</script><h1>Hi</h1><p>Some &amp; text.</p></body></html>",
            )
            .create_async()
            .await;

        let fetcher = BasicUrlFetcher::new(&fast_resilience())
            .unwrap()
            .with_allow_internal(true);
        let url = format!("{}/page", server.url());
        let page = fetcher.fetch(&url).await.unwrap();
        assert_eq!(page.title, "Demo Page");
        assert!(page.text.contains("Hi"));
        assert!(page.text.contains("Some & text"));
        assert!(!page.text.contains("x=1"));
    }

    #[tokio::test]
    async fn fetch_rejects_non_http_schemes() {
        let fetcher = BasicUrlFetcher::new(&fast_resilience()).unwrap();
        let err = fetcher.fetch("file:///etc/passwd").await.unwrap_err();
        assert!(matches!(err, ActionError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn fetch_surfaces_non_2xx_as_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/missing")
            .with_status(404)
            .create_async()
            .await;
        let fetcher = BasicUrlFetcher::new(&fast_resilience())
            .unwrap()
            .with_allow_internal(true);
        let url = format!("{}/missing", server.url());
        assert!(fetcher.fetch(&url).await.is_err());
    }

    #[tokio::test]
    async fn fetch_rejects_loopback_by_default() {
        let fetcher = BasicUrlFetcher::new(&fast_resilience()).unwrap();
        let err = fetcher.fetch("http://127.0.0.1:9").await.unwrap_err();
        assert!(matches!(err, ActionError::InvalidArguments(_)), "{err:?}");
    }

    #[tokio::test]
    async fn fetch_rejects_private_ranges() {
        let fetcher = BasicUrlFetcher::new(&fast_resilience()).unwrap();
        for url in [
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://[::ffff:127.0.0.1]/",
        ] {
            let err = fetcher.fetch(url).await.unwrap_err();
            assert!(
                matches!(err, ActionError::InvalidArguments(_)),
                "{url} should have been rejected (got {err:?})"
            );
        }
    }

    #[tokio::test]
    async fn fetch_rejects_embedded_credentials() {
        let fetcher = BasicUrlFetcher::new(&fast_resilience())
            .unwrap()
            .with_allow_internal(true);
        let err = fetcher
            .fetch("http://user:pass@127.0.0.1:9/")
            .await
            .unwrap_err();
        assert!(matches!(err, ActionError::InvalidArguments(_)), "{err:?}");
    }

    #[tokio::test]
    async fn fetch_rejects_unspecified_address() {
        let fetcher = BasicUrlFetcher::new(&fast_resilience()).unwrap();
        let err = fetcher.fetch("http://0.0.0.0/").await.unwrap_err();
        assert!(matches!(err, ActionError::InvalidArguments(_)), "{err:?}");
    }
}
