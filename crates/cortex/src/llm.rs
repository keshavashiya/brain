//! LLM client — hybrid provider with trait-based adapter.
//!
//! `LlmProvider` trait with multiple implementations:
//! - `OllamaProvider` — local Ollama server
//! - `OpenAiProvider` — OpenAI compatible APIs

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from the LLM layer.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Invalid response format: {0}")]
    InvalidFormat(String),

    #[error("Provider not available: {0}")]
    ProviderUnavailable(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Timeout")]
    Timeout,
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message roles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// LLM response chunk (for streaming).
#[derive(Debug, Clone)]
pub struct ResponseChunk {
    pub content: String,
    pub is_done: bool,
}

/// Complete LLM response.
#[derive(Debug, Clone)]
pub struct Response {
    pub content: String,
    pub usage: Option<Usage>,
}

/// Token usage statistics.
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ─── Provider Trait ─────────────────────────────────────────────────────────

/// Trait for LLM providers.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a complete response (non-streaming).
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError>;

    /// Generate a streaming response.
    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>;

    /// Check if the provider is available.
    async fn health_check(&self) -> bool;

    /// Get the provider name.
    fn name(&self) -> &str;
}

// ─── Ollama Provider ────────────────────────────────────────────────────────

/// Ollama API request body.
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: Option<OllamaOptions>,
}

#[derive(Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OllamaOptions {
    temperature: f64,
    num_predict: i32,
}

/// Ollama API response (works for both streaming and non-streaming).
#[derive(Deserialize)]
struct OllamaResponse {
    message: Option<OllamaMessage>,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

/// Ollama LLM provider.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    temperature: f64,
    max_tokens: i32,
}

impl OllamaProvider {
    /// Create a new Ollama provider.
    pub fn new(base_url: &str, model: &str, temperature: f64, max_tokens: i32) -> Result<Self, LlmError> {
        // Ollama may need to load a large model on first call — allow up to 5 min
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| LlmError::ProviderUnavailable(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
        })
    }

    /// Create with default config. Panics only if TLS initialisation fails (extremely rare).
    pub fn default_config() -> Self {
        Self::new("http://localhost:11434", "qwen2.5-coder:7b", 0.7, 4096)
            .expect("Failed to initialise default Ollama HTTP client")
    }

    fn convert_messages(messages: &[Message]) -> Vec<OllamaMessage> {
        messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                content: m.content.clone(),
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl LlmProvider for OllamaProvider {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let request = OllamaRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            }),
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let data: OllamaResponse = resp.json().await?;

        let content = data
            .message
            .map(|m| m.content)
            .unwrap_or_default();

        Ok(Response {
            content,
            usage: Some(Usage {
                prompt_tokens: data.prompt_eval_count.unwrap_or(0),
                completion_tokens: data.eval_count.unwrap_or(0),
                total_tokens: data.prompt_eval_count.unwrap_or(0) + data.eval_count.unwrap_or(0),
            }),
        })
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        use futures::stream::try_unfold;

        let url = format!("{}/api/chat", self.base_url);
        let request = OllamaRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: true,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            }),
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let byte_stream = resp.bytes_stream();

        // State: (byte_stream, leftover buffer for incomplete lines)
        let stream = try_unfold(
            (Box::pin(byte_stream), String::new()),
            |(mut byte_stream, mut buf)| async move {
                use futures::TryStreamExt;

                loop {
                    // Try to extract a complete line from the buffer
                    if let Some(newline_pos) = buf.find('\n') {
                        let line: String = buf[..newline_pos].to_string();
                        buf = buf[newline_pos + 1..].to_string();

                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<OllamaResponse>(line) {
                            Ok(data) => {
                                let content = data
                                    .message
                                    .map(|m| m.content)
                                    .unwrap_or_default();
                                let chunk = ResponseChunk {
                                    content,
                                    is_done: data.done,
                                };
                                if data.done {
                                    return Ok(Some((chunk, (byte_stream, buf))));
                                }
                                return Ok(Some((chunk, (byte_stream, buf))));
                            }
                            Err(e) => {
                                return Err(LlmError::InvalidFormat(format!(
                                    "Failed to parse streaming response: {e}"
                                )));
                            }
                        }
                    }

                    // Need more data from the network
                    match byte_stream.try_next().await {
                        Ok(Some(bytes)) => {
                            buf.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Ok(None) => {
                            // Stream ended — parse any remaining data in buffer
                            let remaining = buf.trim();
                            if !remaining.is_empty() {
                                if let Ok(data) = serde_json::from_str::<OllamaResponse>(remaining) {
                                    let content = data
                                        .message
                                        .map(|m| m.content)
                                        .unwrap_or_default();
                                    return Ok(Some((
                                        ResponseChunk {
                                            content,
                                            is_done: true,
                                        },
                                        (byte_stream, String::new()),
                                    )));
                                }
                            }
                            return Ok(None);
                        }
                        Err(e) => return Err(LlmError::Http(e)),
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

// ─── OpenAI-Compatible Provider ─────────────────────────────────────────────

/// OpenAI API request body.
#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f64,
    max_tokens: Option<i32>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

/// OpenAI API response.
#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// OpenAI-compatible provider (works with OpenAI, OpenRouter, etc.)
pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f64,
    max_tokens: Option<i32>,
}

impl OpenAiProvider {
    /// Create a new OpenAI-compatible provider.
    pub fn new(
        base_url: &str,
        api_key: Option<&str>,
        model: &str,
        temperature: f64,
        max_tokens: Option<i32>,
    ) -> Result<Self, LlmError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| LlmError::ProviderUnavailable(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.map(|s| s.to_string()),
            model: model.to_string(),
            temperature,
            max_tokens,
        })
    }

    /// Create for OpenAI API.
    pub fn openai(api_key: &str, model: &str) -> Self {
        Self::new("https://api.openai.com/v1", Some(api_key), model, 0.7, Some(4096))
            .expect("Failed to initialise OpenAI HTTP client")
    }

    /// Create for OpenRouter.
    pub fn openrouter(api_key: &str, model: &str) -> Self {
        Self::new("https://openrouter.ai/api/v1", Some(api_key), model, 0.7, Some(4096))
            .expect("Failed to initialise OpenRouter HTTP client")
    }

    fn convert_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
        messages
            .iter()
            .map(|m| OpenAiMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                content: m.content.clone(),
            })
            .collect()
    }

    fn build_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder;
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }
        builder
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: false,
        };

        let resp = self
            .build_request(self.client.post(&url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let data: OpenAiResponse = resp.json().await?;
        let content = data
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(Response {
            content,
            usage: data.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        })
    }

    async fn generate_stream(
        &self,
        _messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        // Streaming support planned for v0.2
        Err(LlmError::Stream(
            "Streaming support is planned for v0.2".to_string(),
        ))
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/models", self.base_url);
        match self.build_request(self.client.get(&url)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn name(&self) -> &str {
        "openai"
    }
}

// ─── Provider Factory ───────────────────────────────────────────────────────

/// Configuration for LLM provider selection.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f64,
    pub max_tokens: i32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            model: "qwen2.5-coder:7b".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

/// Create an LLM provider from configuration.
pub fn create_provider(config: &ProviderConfig) -> Box<dyn LlmProvider> {
    match config.provider.as_str() {
        "ollama" => Box::new(
            OllamaProvider::new(
                &config.base_url,
                &config.model,
                config.temperature,
                config.max_tokens,
            )
            .unwrap_or_else(|e| {
                tracing::error!(error = %e, "Failed to create Ollama provider, falling back to default");
                OllamaProvider::default_config()
            }),
        ),
        "openai" => Box::new(
            OpenAiProvider::new(
                &config.base_url,
                config.api_key.as_deref(),
                &config.model,
                config.temperature,
                Some(config.max_tokens),
            )
            // TLS initialisation failure is unrecoverable — surface clearly.
            .expect("Failed to initialise OpenAI HTTP client"),
        ),
        _ => Box::new(OllamaProvider::default_config()),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config_default() {
        let config = ProviderConfig::default();
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.model, "qwen2.5-coder:7b");
    }

    #[test]
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new("http://localhost:11434", "llama3:8b", 0.5, 2048)
            .expect("OllamaProvider::new should not fail in test");
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_openai_provider_creation() {
        let provider = OpenAiProvider::openai("test-key", "gpt-4");
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_openrouter_provider_creation() {
        let provider = OpenAiProvider::openrouter("test-key", "anthropic/claude-3-opus");
        assert_eq!(provider.name(), "openai");
    }
}
