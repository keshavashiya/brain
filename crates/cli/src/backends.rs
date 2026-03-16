//! External service backends — web search, scheduling, messaging, and memory.

use std::collections::HashMap;
use std::sync::Arc;

use crate::resilience::{resilient_send, CircuitBreaker};

// ─── Memory backend for action dispatcher ────────────────────────────────────

#[derive(Clone)]
pub(crate) struct CliMemoryBackend {
    pub(crate) semantic: Option<hippocampus::SemanticStore>,
    pub(crate) embedder: Arc<tokio::sync::Mutex<Option<hippocampus::Embedder>>>,
    pub(crate) embedding_dim: usize,
}

#[async_trait::async_trait]
impl cortex::actions::MemoryBackend for CliMemoryBackend {
    async fn store_fact(
        &self,
        namespace: &str,
        _category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String, cortex::actions::ActionError> {
        let Some(semantic) = &self.semantic else {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "Semantic store unavailable".to_string(),
            ));
        };

        let content = format!("{subject} {predicate} {object}");
        let vector = {
            let mut guard = self.embedder.lock().await;
            if let Some(embedder) = guard.as_mut() {
                match embedder.embed(&content).await {
                    Ok(v) => {
                        hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, &content)
                    }
                    Err(e) => {
                        tracing::warn!("CLI ActionDispatcher embedding failed: {e}");
                        hippocampus::embedding::deterministic_fallback_embedding(
                            &content,
                            self.embedding_dim,
                        )
                    }
                }
            } else {
                hippocampus::embedding::deterministic_fallback_embedding(
                    &content,
                    self.embedding_dim,
                )
            }
        };

        semantic
            .store_fact(
                namespace, _category, subject, predicate, object, 1.0, None, vector, None,
            )
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))
    }

    async fn recall(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<cortex::actions::MemoryFact>, cortex::actions::ActionError> {
        let Some(semantic) = &self.semantic else {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "Semantic store unavailable".to_string(),
            ));
        };

        let vector = {
            let mut guard = self.embedder.lock().await;
            if let Some(embedder) = guard.as_mut() {
                match embedder.embed(query).await {
                    Ok(v) => {
                        hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, query)
                    }
                    Err(e) => {
                        tracing::warn!("CLI ActionDispatcher embedding failed: {e}");
                        hippocampus::embedding::deterministic_fallback_embedding(
                            query,
                            self.embedding_dim,
                        )
                    }
                }
            } else {
                hippocampus::embedding::deterministic_fallback_embedding(query, self.embedding_dim)
            }
        };

        let results = semantic
            .search_similar(vector, top_k.max(1), namespace, None)
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| cortex::actions::MemoryFact {
                namespace: r.fact.namespace,
                subject: r.fact.subject,
                predicate: r.fact.predicate,
                object: r.fact.object,
                confidence: r.fact.confidence,
            })
            .collect())
    }
}

// ─── Web Search Backends ─────────────────────────────────────────────────────

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

fn build_search_client(timeout_ms: u64) -> anyhow::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
        .build()
        .map_err(|e| anyhow::anyhow!("search client init failed: {e}"))
}

/// SearXNG provider — self-hosted metasearch engine.
pub(crate) struct SearxngSearchBackend {
    endpoint: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl SearxngSearchBackend {
    pub(crate) fn new(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "searxng",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
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
pub(crate) struct TavilySearchBackend {
    endpoint: String,
    api_key: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl TavilySearchBackend {
    pub(crate) fn new(
        endpoint: &str,
        api_key: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "tavily",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
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
pub(crate) struct CustomSearchBackend {
    endpoint: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl CustomSearchBackend {
    pub(crate) fn new(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: endpoint.to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "custom-search",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
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

// ─── Scheduling Backend ──────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct CliSchedulingBackend {
    pub(crate) db: storage::SqlitePool,
    pub(crate) mode: brain_core::config::SchedulingMode,
}

#[async_trait::async_trait]
impl cortex::actions::SchedulingBackend for CliSchedulingBackend {
    async fn schedule(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
    ) -> Result<cortex::actions::ScheduleOutcome, cortex::actions::ActionError> {
        if self.mode != brain_core::config::SchedulingMode::PersistOnly {
            return Err(cortex::actions::ActionError::InvalidArguments(format!(
                "Unsupported scheduling mode: {:?}",
                self.mode
            )));
        }

        let metadata = serde_json::json!({
            "source": "action_dispatcher",
            "mode": "persist_only",
        })
        .to_string();

        let schedule_id = self
            .db
            .insert_scheduled_intent(description, cron, namespace, Some(&metadata))
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        Ok(cortex::actions::ScheduleOutcome {
            schedule_id,
            status: "scheduled".to_string(),
        })
    }
}

// ─── Message Backend ─────────────────────────────────────────────────────────

pub(crate) const DEFAULT_MESSAGE_BODY: &str = r#"{"channel":"{{channel}}","recipient":"{{recipient}}","content":"{{content}}","namespace":"{{namespace}}","timestamp":"{{timestamp}}"}"#;

/// JSON-escape a string value (without surrounding quotes).
pub(crate) fn json_escape(s: &str) -> String {
    let escaped = serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s));
    escaped[1..escaped.len() - 1].to_string()
}

/// Render a message template by replacing `{{placeholder}}` tokens.
pub(crate) fn render_message_template(
    template: &str,
    channel: &str,
    recipient: &str,
    content: &str,
    namespace: &str,
    timestamp: &str,
) -> String {
    template
        .replace("{{channel}}", &json_escape(channel))
        .replace("{{recipient}}", &json_escape(recipient))
        .replace("{{content}}", &json_escape(content))
        .replace("{{namespace}}", &json_escape(namespace))
        .replace("{{timestamp}}", &json_escape(timestamp))
}

pub(crate) struct WebhookMessageBackend {
    channels: HashMap<String, brain_core::config::ChannelConfig>,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl WebhookMessageBackend {
    pub(crate) fn new(
        channels: &HashMap<String, brain_core::config::ChannelConfig>,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
            .build()
            .map_err(|e| anyhow::anyhow!("message client init failed: {e}"))?;
        Ok(Self {
            channels: channels
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect(),
            client,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "webhook-message",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl signal::notification::WebhookSender for WebhookMessageBackend {
    async fn send_notification(
        &self,
        channel: &str,
        content: &str,
        namespace: &str,
    ) -> Result<(), String> {
        let channel_cfg = self
            .channels
            .get(&channel.to_ascii_lowercase())
            .ok_or_else(|| format!("No webhook mapping for channel '{channel}'"))?
            .clone();

        let client = self.client.clone();
        let template = if channel_cfg.body.is_empty() {
            DEFAULT_MESSAGE_BODY.to_string()
        } else {
            channel_cfg.body.clone()
        };
        let url = channel_cfg.url.clone();
        let headers = channel_cfg.headers.clone();
        let channel_owned = channel.to_string();
        let content_owned = content.to_string();
        let namespace_owned = namespace.to_string();

        let response = resilient_send(
            || {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let rendered = render_message_template(
                    &template,
                    &channel_owned,
                    "",
                    &content_owned,
                    &namespace_owned,
                    &timestamp,
                );
                let mut req = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(rendered);
                for (key, value) in &headers {
                    req = req.header(key.as_str(), value.as_str());
                }
                req
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| format!("webhook send failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!(
                "webhook for channel '{}' returned HTTP {}",
                channel,
                response.status()
            ));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl cortex::actions::MessageBackend for WebhookMessageBackend {
    async fn send(
        &self,
        channel: &str,
        recipient: &str,
        content: &str,
        namespace: &str,
    ) -> Result<cortex::actions::MessageOutcome, cortex::actions::ActionError> {
        let channel_cfg = self
            .channels
            .get(&channel.to_ascii_lowercase())
            .ok_or_else(|| {
                cortex::actions::ActionError::InvalidArguments(format!(
                    "No webhook mapping for channel '{}'",
                    channel
                ))
            })?
            .clone();

        let client = self.client.clone();
        let template = if channel_cfg.body.is_empty() {
            DEFAULT_MESSAGE_BODY.to_string()
        } else {
            channel_cfg.body.clone()
        };
        let url = channel_cfg.url.clone();
        let headers = channel_cfg.headers.clone();
        let channel_owned = channel.to_string();
        let recipient_owned = recipient.to_string();
        let content_owned = content.to_string();
        let namespace_owned = namespace.to_string();

        let response = resilient_send(
            || {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let rendered = render_message_template(
                    &template,
                    &channel_owned,
                    &recipient_owned,
                    &content_owned,
                    &namespace_owned,
                    &timestamp,
                );
                let mut req = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(rendered);
                for (key, value) in &headers {
                    req = req.header(key.as_str(), value.as_str());
                }
                req
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "webhook for channel '{}' returned HTTP {}",
                channel,
                response.status()
            )));
        }

        let body = response.text().await.unwrap_or_default();
        let mut delivery_id = format!("msg-{}", chrono::Utc::now().timestamp_micros());
        let mut status = "accepted".to_string();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(id) = value
                .get("id")
                .or_else(|| value.get("delivery_id"))
                .and_then(serde_json::Value::as_str)
            {
                delivery_id = id.to_string();
            }
            if let Some(s) = value.get("status").and_then(serde_json::Value::as_str) {
                status = s.to_string();
            }
        }

        Ok(cortex::actions::MessageOutcome {
            delivery_id,
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduling_backend_persists_intent() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let backend = CliSchedulingBackend {
            db: db.clone(),
            mode: brain_core::config::SchedulingMode::PersistOnly,
        };

        let outcome = cortex::actions::SchedulingBackend::schedule(
            &backend,
            "ship release",
            Some("0 9 * * 1-5"),
            "work",
        )
        .await
        .unwrap();

        assert_eq!(outcome.status, "scheduled");
        let intents = db.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].id, outcome.schedule_id);
        assert_eq!(intents[0].description, "ship release");
    }

    #[test]
    fn test_render_message_template_default() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            "deploy done",
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("default template should produce valid JSON");
        assert_eq!(parsed["channel"], "alerts");
        assert_eq!(parsed["recipient"], "alice");
        assert_eq!(parsed["content"], "deploy done");
        assert_eq!(parsed["namespace"], "work");
        assert_eq!(parsed["timestamp"], "2026-03-08T12:00:00Z");
    }

    #[test]
    fn test_render_message_template_custom_slack() {
        let template = r#"{"text": "[{{channel}}] {{content}}"}"#;
        let rendered = render_message_template(
            template,
            "ops",
            "bob",
            "server is down",
            "personal",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("custom template should produce valid JSON");
        assert_eq!(parsed["text"], "[ops] server is down");
    }

    #[test]
    fn test_render_message_template_escapes_quotes() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            r#"He said "hello""#,
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("escaped content should produce valid JSON");
        assert_eq!(parsed["content"], r#"He said "hello""#);
    }

    #[test]
    fn test_render_message_template_escapes_newlines() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            "line1\nline2",
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value =
            serde_json::from_str(&rendered).expect("newline content should produce valid JSON");
        assert_eq!(parsed["content"], "line1\nline2");
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape(r#"say "hi""#), r#"say \"hi\""#);
        assert_eq!(json_escape("a\nb"), r#"a\nb"#);
        assert_eq!(json_escape("back\\slash"), r#"back\\slash"#);
    }
}
