//! Web search backends — SearXNG, Tavily, and custom endpoint providers.

use std::sync::Arc;

use brain_core::metrics::SubsystemMetrics;

use crate::resilience::{resilient_send, CircuitBreaker};

fn make_cb(
    name: &str,
    resilience: &brain_core::config::ResilienceConfig,
    metrics: Option<Arc<SubsystemMetrics>>,
) -> CircuitBreaker {
    let cb = CircuitBreaker::new(
        name,
        resilience.circuit_breaker_threshold,
        resilience.circuit_breaker_cooldown_secs,
    );
    if let Some(m) = metrics {
        cb.with_metrics(m)
    } else {
        cb
    }
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

pub fn build_search_client(timeout_ms: u64) -> anyhow::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
        .build()
        .map_err(|e| anyhow::anyhow!("search client init failed: {e}"))
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
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Self::new_with_metrics(endpoint, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> anyhow::Result<Self> {
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
        .await?;

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
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Self::new_with_metrics(endpoint, api_key, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        endpoint: &str,
        api_key: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> anyhow::Result<Self> {
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
        .await?;

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
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Self::new_with_metrics(endpoint, timeout_ms, resilience, None)
    }

    pub fn new_with_metrics(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
        metrics: Option<Arc<SubsystemMetrics>>,
    ) -> anyhow::Result<Self> {
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
        .await?;

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
mod tests {
    use super::*;
    use cortex::actions::WebSearchBackend;

    fn fast_resilience() -> brain_core::config::ResilienceConfig {
        brain_core::config::ResilienceConfig {
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
