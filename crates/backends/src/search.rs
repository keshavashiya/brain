//! Web search backends — DuckDuckGo (zero-config built-in), SearXNG,
//! Tavily, and custom endpoint providers.

use std::sync::Arc;

use metrics::SubsystemMetrics;

use crate::resilience::{http_breaker, resilient_send, CircuitBreaker};

/// The capability this backend declares: outbound HTTP for web search + URL
/// fetch, gated on `actions.web_search.enabled` (the same flag the dispatcher
/// keys the search/fetch backends off).
pub fn capabilities(config: &brain::BrainConfig) -> Vec<intent::ToolDescriptor> {
    use crate::capabilities::{backend, native, read_only, usage};
    if !config.actions.web_search.enabled {
        return Vec::new();
    }
    vec![native(
        "net",
        "http",
        backend("net"),
        read_only(),
        usage(
            "The answer needs fresh, external, or post-training-cutoff information, or the user references a URL to read.",
            "The answer is in memory, the conversation, or general knowledge; for plain host reachability or a connectivity diagnosis (rather than reading a page) use net.check.",
            &["actions.web_search.enabled = true", "Network egress is permitted."],
            "network call (latency + possible API quota)",
            "\"What's the latest release of ripgrep?\"",
        ),
    )]
}

fn make_cb(
    name: &str,
    resilience: &brain::config::ResilienceConfig,
    metrics: Option<Arc<SubsystemMetrics>>,
) -> CircuitBreaker {
    http_breaker(
        name,
        resilience.circuit_breaker_threshold,
        resilience.circuit_breaker_cooldown_secs,
        metrics,
    )
}

/// Parse a JSON array of search results into `SearchHit`s with flexible field names.
fn parse_search_results(
    candidates: Vec<serde_json::Value>,
    top_k: usize,
) -> Vec<cortex::actions::SearchHit> {
    candidates
        .into_iter()
        .filter_map(|entry| {
            let title = entry
                .get("title")
                .and_then(serde_json::Value::as_str)
                .or_else(|| entry.get("name").and_then(serde_json::Value::as_str))
                .unwrap_or("untitled")
                .to_string();
            let url = entry
                .get("url")
                .and_then(serde_json::Value::as_str)
                .or_else(|| entry.get("link").and_then(serde_json::Value::as_str))
                .unwrap_or_default()
                .to_string();
            if url.is_empty() {
                return None;
            }
            let snippet = entry
                .get("snippet")
                .and_then(serde_json::Value::as_str)
                .or_else(|| entry.get("description").and_then(serde_json::Value::as_str))
                .or_else(|| entry.get("content").and_then(serde_json::Value::as_str))
                .unwrap_or_default()
                .to_string();
            Some(cortex::actions::SearchHit {
                title,
                url,
                snippet,
            })
        })
        .take(top_k.max(1))
        .collect()
}

pub fn build_search_client(
    timeout_ms: u64,
) -> Result<reqwest::Client, crate::error::BackendInitError> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
        .build()
        .map_err(|e| crate::error::BackendInitError::HttpClient("search client", e))
}

/// DuckDuckGo HTML provider — zero-config, no API key, no Docker.
/// Hits the public `html.duckduckgo.com` endpoint and parses result
/// blocks out of the response. Quality is more limited than a metasearch
/// aggregator like SearXNG (single engine, no rich snippets), but it
/// works on every install with no setup. Falls back gracefully when
/// the HTML layout changes — bad parses return empty rather than panic.
pub struct DuckDuckGoSearchBackend {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl DuckDuckGoSearchBackend {
    pub fn new(
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
    ) -> Result<Self, crate::error::BackendInitError> {
        Self::new_with_metrics(timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> Result<Self, crate::error::BackendInitError> {
        // DDG's HTML endpoint inspects the User-Agent and serves the
        // post-only "we redirected you" stub if it looks too generic.
        // A normal-looking UA gets the real result list.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
            .user_agent(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
                 AppleWebKit/537.36 (KHTML, like Gecko) \
                 Chrome/124.0.0.0 Safari/537.36",
            )
            .redirect(reqwest::redirect::Policy::limited(3))
            .build()
            .map_err(|e| crate::error::BackendInitError::HttpClient("DuckDuckGo client", e))?;
        Ok(Self {
            client,
            circuit_breaker: Arc::new(make_cb("duckduckgo", resilience, metrics)),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for DuckDuckGoSearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        // Strategy: try the official Instant Answer JSON API first (always
        // reachable, no anti-bot, but only returns hits when DDG has a
        // categorized answer). If it produces nothing useful, fall back
        // to scraping `html.duckduckgo.com` — which DDG sometimes blocks
        // with an anomaly/CAPTCHA page; that case is detected and surfaced
        // as a clear error pointing to SearXNG / Tavily.
        let from_ia = self.search_instant_answer(query, top_k).await?;
        if !from_ia.is_empty() {
            return Ok(from_ia);
        }
        self.search_html(query, top_k).await
    }
}

impl DuckDuckGoSearchBackend {
    async fn search_instant_answer(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let url = "https://api.duckduckgo.com/";
        let client = self.client.clone();
        let q = query.to_string();
        let resp = resilient_send(
            || {
                client.get(url).query(&[
                    ("q", q.as_str()),
                    ("format", "json"),
                    ("no_html", "1"),
                    ("skip_disambig", "1"),
                ])
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(Vec::new());
        }
        let body: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return Ok(Vec::new()),
        };
        Ok(parse_instant_answer(&body, top_k))
    }

    async fn search_html(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        // POST to the same `html.duckduckgo.com/html/` endpoint with
        // browser-style headers — this is the form a real search submission
        // takes, and DDG is somewhat less aggressive about challenging it.
        let url = "https://html.duckduckgo.com/html/";
        let client = self.client.clone();
        let q = query.to_string();
        let response = resilient_send(
            || {
                client
                    .post(url)
                    .header(
                        reqwest::header::ACCEPT,
                        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    )
                    .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9")
                    .header(reqwest::header::REFERER, "https://duckduckgo.com/")
                    .header(reqwest::header::ORIGIN, "https://duckduckgo.com")
                    .form(&[("q", q.as_str())])
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "DuckDuckGo returned HTTP {}",
                response.status()
            )));
        }

        let html = response
            .text()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        // DDG's anti-bot interstitial swaps out result markup for an
        // "anomaly-modal" CAPTCHA. Detect and surface a clean error so
        // callers know to switch backends instead of seeing silent zero
        // hits.
        if html.contains("anomaly-modal") {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "DuckDuckGo served a bot-challenge page instead of results. \
                 Run `brain deps up` to use SearXNG, or set \
                 `actions.web_search.provider: tavily` with an API key."
                    .into(),
            ));
        }

        Ok(parse_duckduckgo_html(&html, top_k))
    }
}

/// Convert DuckDuckGo Instant Answer API response into search hits.
/// Picks up the abstract (with source URL) and any RelatedTopics that
/// have first-class URL+text — skipping disambiguation/category headers.
fn parse_instant_answer(body: &serde_json::Value, top_k: usize) -> Vec<cortex::actions::SearchHit> {
    let mut hits = Vec::new();

    let abstract_text = body
        .get("AbstractText")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let abstract_url = body
        .get("AbstractURL")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let heading = body.get("Heading").and_then(|v| v.as_str()).unwrap_or("");
    if !abstract_text.is_empty() && !abstract_url.is_empty() {
        hits.push(cortex::actions::SearchHit {
            title: if heading.is_empty() {
                "DuckDuckGo Instant Answer".to_string()
            } else {
                heading.to_string()
            },
            url: abstract_url.to_string(),
            snippet: abstract_text.to_string(),
        });
    }

    if let Some(topics) = body.get("RelatedTopics").and_then(|v| v.as_array()) {
        for t in topics {
            if hits.len() >= top_k.max(1) {
                break;
            }
            let url = t.get("FirstURL").and_then(|v| v.as_str()).unwrap_or("");
            let text = t.get("Text").and_then(|v| v.as_str()).unwrap_or("");
            if url.is_empty() || text.is_empty() {
                continue;
            }
            // Use the first sentence of `Text` as the title, the rest as snippet.
            let (title, snippet) = match text.split_once(" - ") {
                Some((t, s)) => (t.to_string(), s.to_string()),
                None => (text.to_string(), text.to_string()),
            };
            hits.push(cortex::actions::SearchHit {
                title,
                url: url.to_string(),
                snippet,
            });
        }
    }

    hits.into_iter().take(top_k.max(1)).collect()
}

/// Pull title/url/snippet out of DuckDuckGo's HTML response. Each result
/// is wrapped in a `<div class="result …">` block; inside it the title
/// link has class `result__a`, the snippet has class `result__snippet`.
/// DDG rewrites destination URLs through `/l/?uddg=<percent-encoded>`,
/// so we have to undo that to surface the real link.
fn parse_duckduckgo_html(html: &str, top_k: usize) -> Vec<cortex::actions::SearchHit> {
    let mut hits = Vec::new();
    let mut cursor = 0usize;
    let max = top_k.max(1);

    while hits.len() < max {
        // Find the next result block. Match on the class anchor that's
        // stable across DDG's HTML variants: every result row contains
        // an `<a class="result__a"` for the title link.
        let Some(rel) = html[cursor..].find("class=\"result__a\"") else {
            break;
        };
        let block_start = cursor + rel;
        // Walk back to find the opening `<a` of that anchor.
        let Some(a_open) = html[..block_start].rfind("<a ") else {
            cursor = block_start + 1;
            continue;
        };
        // Extract href.
        let after_a_open = a_open + 3;
        let Some(href_rel) = html[after_a_open..].find("href=\"") else {
            cursor = block_start + 1;
            continue;
        };
        let href_start = after_a_open + href_rel + 6;
        let Some(href_end_rel) = html[href_start..].find('"') else {
            cursor = block_start + 1;
            continue;
        };
        let href_raw = &html[href_start..href_start + href_end_rel];
        let url = decode_ddg_redirect(href_raw);

        // Title text: everything between the `>` after the anchor open
        // and the matching `</a>`.
        let Some(gt_rel) = html[block_start..].find('>') else {
            cursor = block_start + 1;
            continue;
        };
        let title_start = block_start + gt_rel + 1;
        let Some(title_end_rel) = html[title_start..].find("</a>") else {
            cursor = block_start + 1;
            continue;
        };
        let title = strip_html(&html[title_start..title_start + title_end_rel]);
        let snippet_search_from = title_start + title_end_rel;

        // Snippet — the next `class="result__snippet"` block within a
        // bounded window so we don't pull a snippet from the next result.
        let window_end = html[snippet_search_from..]
            .find("class=\"result__a\"")
            .map(|r| snippet_search_from + r)
            .unwrap_or(html.len());
        let window = &html[snippet_search_from..window_end];
        let snippet = window
            .find("class=\"result__snippet\"")
            .and_then(|s| {
                let from = snippet_search_from + s;
                let gt = html[from..].find('>')? + from + 1;
                let end = html[gt..].find("</a>")? + gt;
                Some(strip_html(&html[gt..end]))
            })
            .unwrap_or_default();

        if !url.is_empty() && !title.is_empty() {
            hits.push(cortex::actions::SearchHit {
                title,
                url,
                snippet,
            });
        }
        cursor = window_end;
    }

    hits
}

/// DDG wraps every result URL in `https://duckduckgo.com/l/?uddg=…&…` (or
/// the protocol-relative `//duckduckgo.com/l/?uddg=…`). Extract the
/// `uddg` parameter and percent-decode it; if the input isn't a redirect,
/// return it unchanged so direct links still work.
fn decode_ddg_redirect(href: &str) -> String {
    let trimmed = href
        .trim()
        .trim_start_matches("https:")
        .trim_start_matches("http:");
    if !(trimmed.starts_with("//duckduckgo.com/l/") || trimmed.starts_with("/l/?uddg=")) {
        // Not a redirect — but still normalise protocol-relative URLs
        // so downstream consumers get a fully-qualified scheme.
        if let Some(stripped) = href.strip_prefix("//") {
            return format!("https://{stripped}");
        }
        return href.to_string();
    }
    // Find `uddg=` and percent-decode up to the next `&` or end of string.
    let Some(start) = href.find("uddg=") else {
        return href.to_string();
    };
    let after = &href[start + 5..];
    let end = after.find('&').unwrap_or(after.len());
    percent_decode(&after[..end])
}

/// Minimal percent-decoder. We can't pull `percent-encoding` for one
/// call site without bloating the dep tree; the input is well-formed
/// query-string output so this covers the cases that matter.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = hex_digit(bytes[i + 1]);
            let lo = hex_digit(bytes[i + 2]);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Strip HTML tags + decode the entities DDG actually emits in titles
/// and snippets (`&amp;`, `&#39;`, `&quot;`, `&lt;`, `&gt;`, `&nbsp;`).
/// Whitespace runs collapse to a single space so the snippet renders
/// cleanly when the LLM quotes it back.
fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if in_tag => {}
            _ => out.push(ch),
        }
    }
    let decoded = out
        .replace("&amp;", "&")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// SearXNG provider — self-hosted metasearch engine.
pub struct SearxngSearchBackend {
    endpoint: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl SearxngSearchBackend {
    pub fn new(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
    ) -> Result<Self, crate::error::BackendInitError> {
        Self::new_with_metrics(endpoint, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> Result<Self, crate::error::BackendInitError> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(make_cb("searxng", resilience, metrics)),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for SearxngSearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let url = format!("{}/search", self.endpoint);
        let client = self.client.clone();
        let url_clone = url.clone();
        let query_owned = query.to_string();
        let response = resilient_send(
            || {
                client
                    .get(&url_clone)
                    .query(&[("q", query_owned.as_str()), ("format", "json")])
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "SearXNG returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        let candidates = match body.get("results").and_then(|v| v.as_array()) {
            Some(arr) => arr.clone(),
            None => {
                tracing::warn!(
                    backend = "searxng",
                    "Response missing 'results' array — returning empty"
                );
                Vec::new()
            }
        };

        Ok(parse_search_results(candidates, top_k))
    }
}

/// Tavily provider — AI-focused search API.
pub struct TavilySearchBackend {
    endpoint: String,
    api_key: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl TavilySearchBackend {
    pub fn new(
        endpoint: &str,
        api_key: &str,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
    ) -> Result<Self, crate::error::BackendInitError> {
        Self::new_with_metrics(endpoint, api_key, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        endpoint: &str,
        api_key: &str,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> Result<Self, crate::error::BackendInitError> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(make_cb("tavily", resilience, metrics)),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for TavilySearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let url = format!("{}/search", self.endpoint);
        let client = self.client.clone();
        let url_clone = url.clone();
        let api_key = self.api_key.clone();
        let query_owned = query.to_string();
        let response = resilient_send(
            || {
                client
                    .post(&url_clone)
                    .bearer_auth(&api_key)
                    .json(&serde_json::json!({
                        "query": query_owned,
                        "max_results": top_k,
                        "search_depth": "basic",
                    }))
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "Tavily returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        let candidates = match body.get("results").and_then(|v| v.as_array()) {
            Some(arr) => {
                if !arr.is_empty() && arr[0].get("url").is_none() {
                    tracing::warn!(
                        backend = "tavily",
                        "Results missing 'url' field — response schema may have changed"
                    );
                }
                arr.clone()
            }
            None => {
                tracing::warn!(
                    backend = "tavily",
                    "Response missing 'results' array — returning empty"
                );
                Vec::new()
            }
        };

        Ok(parse_search_results(candidates, top_k))
    }
}

/// Custom provider — raw JSON POST to a user-configured endpoint.
pub struct CustomSearchBackend {
    endpoint: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl CustomSearchBackend {
    pub fn new(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
    ) -> Result<Self, crate::error::BackendInitError> {
        Self::new_with_metrics(endpoint, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> Result<Self, crate::error::BackendInitError> {
        Ok(Self {
            endpoint: endpoint.to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(make_cb("custom-search", resilience, metrics)),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for CustomSearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let query_owned = query.to_string();
        let response = resilient_send(
            || {
                client.post(&endpoint).json(&serde_json::json!({
                    "query": query_owned,
                    "top_k": top_k,
                }))
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "search endpoint returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        let candidates: Vec<serde_json::Value> = body
            .get("hits")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| body.get("results").and_then(|v| v.as_array()).cloned())
            .or_else(|| body.as_array().cloned())
            .unwrap_or_default();

        Ok(parse_search_results(candidates, top_k))
    }
}

#[cfg(test)]
mod ddg_parser_tests {
    use super::*;

    const SAMPLE_HTML: &str = r#"
        <div class="result results_links">
          <h2 class="result__title">
            <a class="result__a" rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F&amp;rut=abc">
              The Rust Programming Language
            </a>
          </h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rust-lang.org%2F">
            A language empowering everyone to build <b>reliable</b> and efficient software.
          </a>
        </div>
        <div class="result results_links">
          <h2 class="result__title">
            <a class="result__a" rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2F&amp;rut=def">
              The Rust Programming Language - The Rust Book
            </a>
          </h2>
          <a class="result__snippet">
            An introductory book about Rust&#39;s ownership &amp; lifetimes.
          </a>
        </div>
    "#;

    #[test]
    fn parses_title_url_and_snippet_from_real_layout() {
        let hits = parse_duckduckgo_html(SAMPLE_HTML, 10);
        assert_eq!(hits.len(), 2, "should pick up both result blocks");
        assert_eq!(hits[0].title, "The Rust Programming Language");
        assert_eq!(hits[0].url, "https://www.rust-lang.org/");
        assert!(hits[0].snippet.contains("reliable and efficient"));
        assert_eq!(hits[1].url, "https://doc.rust-lang.org/book/");
        // HTML entities in the snippet should be decoded.
        assert!(hits[1].snippet.contains("Rust's ownership & lifetimes"));
    }

    #[test]
    fn respects_top_k() {
        let hits = parse_duckduckgo_html(SAMPLE_HTML, 1);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn returns_empty_on_unrecognised_layout() {
        let hits = parse_duckduckgo_html("<html><body>no results</body></html>", 5);
        assert!(hits.is_empty());
    }

    #[test]
    fn decodes_redirect_and_passes_direct_urls_through() {
        assert_eq!(
            decode_ddg_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x"),
            "https://example.com/a"
        );
        // Direct https URL — pass through unchanged.
        assert_eq!(
            decode_ddg_redirect("https://example.com/direct"),
            "https://example.com/direct"
        );
        // Protocol-relative non-redirect — normalise to https.
        assert_eq!(
            decode_ddg_redirect("//example.com/x"),
            "https://example.com/x"
        );
    }

    #[test]
    fn percent_decoder_handles_plus_and_hex() {
        assert_eq!(percent_decode("hello+world"), "hello world");
        assert_eq!(percent_decode("a%20b%2Fc"), "a b/c");
        // Malformed escapes are passed through verbatim instead of panicking.
        assert_eq!(percent_decode("a%ZZb"), "a%ZZb");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex::actions::WebSearchBackend;

    fn fast_resilience() -> brain::config::ResilienceConfig {
        brain::config::ResilienceConfig {
            max_retries: 0,
            retry_base_ms: 10,
            circuit_breaker_threshold: 5,
            circuit_breaker_cooldown_secs: 60,
        }
    }

    #[tokio::test]
    async fn test_searxng_successful_search() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/search")
            .match_query(mockito::Matcher::UrlEncoded("q".into(), "rust".into()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "results": [
                        {"title": "Rust docs", "url": "https://doc.rust-lang.org", "content": "language docs"},
                        {"title": "Rust book", "url": "https://rust-book.rs", "content": "book"}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let backend = SearxngSearchBackend::new(&server.url(), 5000, &fast_resilience()).unwrap();
        let hits = backend.search("rust", 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "Rust docs");
        assert_eq!(hits[0].url, "https://doc.rust-lang.org");
        assert_eq!(hits[0].snippet, "language docs");
    }

    #[tokio::test]
    async fn test_searxng_empty_results() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/search")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"results": []}"#)
            .create_async()
            .await;

        let backend = SearxngSearchBackend::new(&server.url(), 5000, &fast_resilience()).unwrap();
        let hits = backend.search("nothing", 10).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn test_searxng_5xx_surfaces_as_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/search")
            .match_query(mockito::Matcher::Any)
            .with_status(500)
            .with_body("internal error")
            .expect_at_least(1)
            .create_async()
            .await;

        let backend = SearxngSearchBackend::new(&server.url(), 5000, &fast_resilience()).unwrap();
        let result = backend.search("boom", 10).await;
        assert!(result.is_err(), "expected 5xx to surface as error");
    }

    #[tokio::test]
    async fn test_searxng_top_k_limit() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/search")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "results": [
                        {"title": "a", "url": "https://a.com", "content": ""},
                        {"title": "b", "url": "https://b.com", "content": ""},
                        {"title": "c", "url": "https://c.com", "content": ""},
                        {"title": "d", "url": "https://d.com", "content": ""}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let backend = SearxngSearchBackend::new(&server.url(), 5000, &fast_resilience()).unwrap();
        let hits = backend.search("q", 2).await.unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn test_searxng_missing_results_field_returns_empty() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/search")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"something": "else"}"#)
            .create_async()
            .await;

        let backend = SearxngSearchBackend::new(&server.url(), 5000, &fast_resilience()).unwrap();
        let hits = backend.search("q", 10).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn test_tavily_successful_search_sends_bearer() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/search")
            .match_header("authorization", "Bearer tvly-test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "results": [
                        {"title": "Tavily hit", "url": "https://example.com", "content": "snippet"}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let backend =
            TavilySearchBackend::new(&server.url(), "tvly-test-key", 5000, &fast_resilience())
                .unwrap();
        let hits = backend.search("question", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Tavily hit");
    }
}
