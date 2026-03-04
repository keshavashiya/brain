//! Embedding pipeline — Ollama and OpenAI-compatible backends.
//!
//! The active provider is determined by `llm.provider` in Brain config:
//! - `"ollama"` → calls `POST /api/embed` on the local Ollama server
//! - `"openai"` → calls `POST /v1/embeddings` on any OpenAI-compatible endpoint
//!
//! The embedding model and output dimension are explicit config values:
//!
//! ```yaml
//! llm:
//!   provider: "ollama"
//!   base_url: "http://localhost:11434"
//!
//! embedding:
//!   model: "nomic-embed-text"   # must be pulled in Ollama / available via OpenAI
//!   dimensions: 768              # must match the model's actual output size
//! ```
//!
//! For OpenAI:
//! ```yaml
//! llm:
//!   provider: "openai"
//!   base_url: "https://api.openai.com/v1"
//!   api_key: "sk-..."
//!
//! embedding:
//!   model: "text-embedding-3-small"
//!   dimensions: 1536
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the embedding pipeline.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Response parse error: {0}")]
    Parse(String),

    #[error("Shape error: {0}")]
    Shape(String),

    #[error("Provider not available: {0}")]
    ProviderUnavailable(String),
}

// ─── Ollama Provider ─────────────────────────────────────────────────────────

/// Ollama embedding provider — calls `POST /api/embed`.
///
/// Shares the same Ollama instance used for LLM inference.  Pull the model
/// once with `ollama pull <model>` and it works immediately.
#[derive(Debug)]
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaProvider {
    pub fn new(base_url: &str, model: &str) -> Self {
        // Ollama may need to load the model on first call — allow up to 120s
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Check if the Ollama server is reachable.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/embed", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&OllamaEmbedRequest {
                model: &self.model,
                input: texts.to_vec(),
            })
            .send()
            .await
            .map_err(|e| EmbeddingError::Http(format!("Request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbeddingError::Http(format!("HTTP {status}: {body}")));
        }

        let parsed: OllamaEmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbeddingError::Parse(format!("Failed to parse Ollama response: {e}")))?;

        if parsed.embeddings.len() != texts.len() {
            return Err(EmbeddingError::Shape(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                parsed.embeddings.len()
            )));
        }
        Ok(parsed.embeddings)
    }
}

// ─── OpenAI-compatible Provider ──────────────────────────────────────────────

/// OpenAI-compatible embedding provider — calls `POST /v1/embeddings`.
///
/// Works with OpenAI, OpenRouter, Azure OpenAI, or any OpenAI-compatible
/// local server (e.g. vLLM, LM Studio, Ollama in OpenAI-compat mode).
#[derive(Debug)]
pub struct OpenAIProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

#[derive(Serialize)]
struct OpenAIEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct OpenAIEmbedResponse {
    data: Vec<OpenAIEmbedData>,
}

#[derive(Deserialize)]
struct OpenAIEmbedData {
    embedding: Vec<f32>,
    index: usize,
}

impl OpenAIProvider {
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("Failed to create HTTP client");
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
        }
    }

    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&OpenAIEmbedRequest {
                model: &self.model,
                input: texts.to_vec(),
            })
            .send()
            .await
            .map_err(|e| EmbeddingError::Http(format!("Request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbeddingError::Http(format!("HTTP {status}: {body}")));
        }

        let mut parsed: OpenAIEmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbeddingError::Parse(format!("Failed to parse OpenAI response: {e}")))?;

        if parsed.data.len() != texts.len() {
            return Err(EmbeddingError::Shape(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                parsed.data.len()
            )));
        }
        // Sort by index to guarantee order (OpenAI may reorder for batching)
        parsed.data.sort_by_key(|d| d.index);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

// ─── Embedder ─────────────────────────────────────────────────────────────────

/// Active embedding backend.
///
/// Constructed once at startup via [`Embedder::for_ollama`] or [`Embedder::for_openai`],
/// then shared (behind `tokio::sync::Mutex`) across the signal pipeline.
#[allow(clippy::large_enum_variant)]
pub enum Embedder {
    Ollama(OllamaProvider),
    OpenAI(OpenAIProvider),
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama(_) => write!(f, "Embedder::Ollama"),
            Self::OpenAI(_) => write!(f, "Embedder::OpenAI"),
        }
    }
}

impl Embedder {
    /// Create an Ollama-backed embedder.
    pub fn for_ollama(base_url: &str, model: &str) -> Self {
        info!(model, "Embedding provider: Ollama");
        Self::Ollama(OllamaProvider::new(base_url, model))
    }

    /// Create an OpenAI-compatible embedder.
    pub fn for_openai(base_url: &str, model: &str, api_key: &str) -> Self {
        info!(model, base_url, "Embedding provider: OpenAI-compatible");
        Self::OpenAI(OpenAIProvider::new(base_url, model, api_key))
    }

    /// Embed a single text string.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut batch = self.embed_batch(&[text]).await?;
        Ok(batch.remove(0))
    }

    /// Embed a batch of texts for throughput.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match self {
            Self::Ollama(p) => p.embed_batch(texts).await,
            Self::OpenAI(p) => p.embed_batch(texts).await,
        }
    }

    /// Provider name for logging.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Ollama(_) => "ollama",
            Self::OpenAI(_) => "openai",
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_new() {
        let p = OllamaProvider::new("http://localhost:11434", "nomic-embed-text");
        assert_eq!(p.model, "nomic-embed-text");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_provider_trims_trailing_slash() {
        let p = OllamaProvider::new("http://localhost:11434/", "nomic-embed-text");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_openai_provider_new() {
        let p = OpenAIProvider::new("https://api.openai.com/v1", "text-embedding-3-small", "sk-x");
        assert_eq!(p.model, "text-embedding-3-small");
        assert_eq!(p.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn test_embedder_provider_name() {
        let e = Embedder::for_ollama("http://localhost:11434", "nomic-embed-text");
        assert_eq!(e.provider_name(), "ollama");

        let e2 = Embedder::for_openai("https://api.openai.com/v1", "text-embedding-3-small", "k");
        assert_eq!(e2.provider_name(), "openai");
    }

    /// Requires Ollama running locally with nomic-embed-text pulled.
    #[tokio::test]
    #[ignore = "Requires Ollama server running locally with nomic-embed-text"]
    async fn test_ollama_embed_live() {
        let e = Embedder::for_ollama("http://localhost:11434", "nomic-embed-text");
        let v = e.embed("Hello, world!").await.unwrap();
        assert_eq!(v.len(), 768, "nomic-embed-text produces 768-dim vectors");
    }
}
