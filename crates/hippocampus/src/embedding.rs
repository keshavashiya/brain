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

use async_trait::async_trait;
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

    #[error("Provider initialization error: {0}")]
    Provider(String),
}

/// Build a `reqwest::Client` with the given timeout, mapping construction
/// failure to [`EmbeddingError::Provider`]. Shared by Ollama + OpenAI
/// embedding providers so the timeout/error-mapping policy stays
/// single-sourced.
fn build_http_client(timeout: std::time::Duration) -> Result<reqwest::Client, EmbeddingError> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| EmbeddingError::Provider(format!("Failed to create HTTP client: {e}")))
}

/// Deterministically generate a non-zero fallback embedding and normalize it.
///
/// This is used when the embedding provider is unavailable or returns an invalid
/// vector shape/value. The output is stable for the same `(seed, dimensions)`.
pub fn deterministic_fallback_embedding(seed: &str, dimensions: usize) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }

    // FNV-1a 64-bit hash as deterministic PRNG seed.
    let mut state: u64 = 0xcbf29ce484222325;
    for b in seed.as_bytes() {
        state ^= u64::from(*b);
        state = state.wrapping_mul(0x100000001b3);
    }
    if state == 0 {
        state = 1;
    }

    let mut out = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let r = state.wrapping_mul(0x2545f4914f6cdd1d);
        let unit = (r as f64 / u64::MAX as f64) as f32;
        out.push(unit * 2.0 - 1.0);
    }

    normalize_or_unit(out)
}

/// Validate and normalize an embedding vector, with deterministic fallback.
///
/// Conditions enforced:
/// - exact `dimensions`
/// - finite values only
/// - non-zero norm
/// - normalized output
pub fn sanitize_embedding(candidate: Vec<f32>, dimensions: usize, seed: &str) -> Vec<f32> {
    if dimensions == 0 {
        return Vec::new();
    }
    if candidate.len() != dimensions || candidate.iter().any(|x| !x.is_finite()) {
        return deterministic_fallback_embedding(seed, dimensions);
    }

    let norm_sq: f32 = candidate.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq <= 1e-12 {
        return deterministic_fallback_embedding(seed, dimensions);
    }

    let normalized = normalize_or_unit(candidate);
    if normalized.iter().all(|x| x.is_finite()) {
        normalized
    } else {
        deterministic_fallback_embedding(seed, dimensions)
    }
}

fn normalize_or_unit(mut vector: Vec<f32>) -> Vec<f32> {
    if vector.is_empty() {
        return vector;
    }

    let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
    if !norm_sq.is_finite() || norm_sq <= 1e-12 {
        let mut unit = vec![0.0_f32; vector.len()];
        unit[0] = 1.0;
        return unit;
    }

    let norm = norm_sq.sqrt();
    for v in &mut vector {
        *v /= norm;
    }
    vector
}

// ─── EmbeddingProvider trait ─────────────────────────────────────────────────

/// Pluggable embedding backend.
///
/// New providers (Voyage, Cohere, …) implement this trait and register in
/// [`Embedder::from_config`] — no other call site changes.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync + std::fmt::Debug {
    /// Embed a batch of texts. Implementations preserve input order.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Stable lower-case identifier used in logs and metrics
    /// (e.g. `"ollama"`, `"openai"`).
    fn provider_name(&self) -> &str;
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
    pub fn new(base_url: &str, model: &str) -> Result<Self, EmbeddingError> {
        // Ollama may need to load the model on first call — allow up to 120s
        let client = build_http_client(brain::timeouts::EMBEDDING_OLLAMA)?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        })
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
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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

    fn provider_name(&self) -> &str {
        "ollama"
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
    pub fn new(base_url: &str, model: &str, api_key: &str) -> Result<Self, EmbeddingError> {
        let client = build_http_client(brain::timeouts::EMBEDDING_OPENAI)?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
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

    fn provider_name(&self) -> &str {
        "openai"
    }
}

// ─── Embedder ─────────────────────────────────────────────────────────────────

/// Active embedding backend — owns a trait object so new providers can be
/// added without touching this type.
///
/// Constructed once at startup via [`Embedder::for_ollama`], [`Embedder::for_openai`]
/// or [`Embedder::from_config`], then shared (behind `tokio::sync::Mutex`)
/// across the signal pipeline.
pub struct Embedder {
    inner: Box<dyn EmbeddingProvider>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Embedder({})", self.inner.provider_name())
    }
}

impl Embedder {
    /// Wrap any provider impl. Lets crates outside this module register
    /// custom backends (Voyage, Cohere, in-process test doubles, …).
    pub fn new(inner: Box<dyn EmbeddingProvider>) -> Self {
        Self { inner }
    }

    /// Create an Ollama-backed embedder.
    pub fn for_ollama(base_url: &str, model: &str) -> Result<Self, EmbeddingError> {
        info!(model, "Embedding provider: Ollama");
        Ok(Self::new(Box::new(OllamaProvider::new(base_url, model)?)))
    }

    /// Create an OpenAI-compatible embedder.
    pub fn for_openai(base_url: &str, model: &str, api_key: &str) -> Result<Self, EmbeddingError> {
        info!(model, base_url, "Embedding provider: OpenAI-compatible");
        Ok(Self::new(Box::new(OpenAIProvider::new(
            base_url, model, api_key,
        )?)))
    }

    /// Create an embedder from Brain config settings.
    ///
    /// Selects the appropriate backend (Ollama or OpenAI-compatible)
    /// based on `provider`. Returns `None` if no provider is configured.
    pub fn from_config(
        provider: &str,
        base_url: &str,
        model: &str,
        api_key: &str,
    ) -> Result<Option<Self>, EmbeddingError> {
        match provider {
            "openai" => Ok(Some(Self::for_openai(base_url, model, api_key)?)),
            _ => Ok(Some(Self::for_ollama(base_url, model)?)),
        }
    }

    /// Embed a single text string.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut batch = self.embed_batch(&[text]).await?;
        Ok(batch.remove(0))
    }

    /// Embed a batch of texts for throughput.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.inner.embed_batch(texts).await
    }

    /// Provider name for logging.
    pub fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_provider_new() {
        let p = OllamaProvider::new("http://localhost:11434", "nomic-embed-text").unwrap();
        assert_eq!(p.model, "nomic-embed-text");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_provider_trims_trailing_slash() {
        let p = OllamaProvider::new("http://localhost:11434/", "nomic-embed-text").unwrap();
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_openai_provider_new() {
        let p = OpenAIProvider::new(
            "https://api.openai.com/v1",
            "text-embedding-3-small",
            "sk-x",
        )
        .unwrap();
        assert_eq!(p.model, "text-embedding-3-small");
        assert_eq!(p.base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn test_embedder_provider_name() {
        let e = Embedder::for_ollama("http://localhost:11434", "nomic-embed-text").unwrap();
        assert_eq!(e.provider_name(), "ollama");

        let e2 = Embedder::for_openai("https://api.openai.com/v1", "text-embedding-3-small", "k")
            .unwrap();
        assert_eq!(e2.provider_name(), "openai");
    }

    /// Requires Ollama running locally with nomic-embed-text pulled.
    #[tokio::test]
    #[ignore = "Requires Ollama server running locally with nomic-embed-text"]
    async fn test_ollama_embed_live() {
        let e = Embedder::for_ollama("http://localhost:11434", "nomic-embed-text").unwrap();
        let v = e.embed("Hello, world!").await.unwrap();
        assert_eq!(v.len(), 768, "nomic-embed-text produces 768-dim vectors");
    }

    #[test]
    fn test_deterministic_fallback_embedding_is_stable_and_normalized() {
        let a = deterministic_fallback_embedding("remember rust", 16);
        let b = deterministic_fallback_embedding("remember rust", 16);
        let c = deterministic_fallback_embedding("remember bun", 16);

        assert_eq!(a.len(), 16);
        assert_eq!(a, b, "same seed must produce same fallback vector");
        assert_ne!(a, c, "different seeds should produce different vectors");

        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "fallback vector must be normalized"
        );
    }

    #[test]
    fn test_sanitize_embedding_rejects_invalid_inputs() {
        let zero = vec![0.0_f32; 8];
        let nan = vec![f32::NAN; 8];
        let wrong = vec![0.1_f32; 4];

        let a = sanitize_embedding(zero, 8, "seed-a");
        let b = sanitize_embedding(nan, 8, "seed-b");
        let c = sanitize_embedding(wrong, 8, "seed-c");

        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 8);
        assert_eq!(c.len(), 8);
        assert!(a.iter().all(|x| x.is_finite()));
        assert!(b.iter().all(|x| x.is_finite()));
        assert!(c.iter().all(|x| x.is_finite()));
    }

    // ─── Mock HTTP tests ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_ollama_embed_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"embeddings": [[0.1, 0.2, 0.3, 0.4]]}"#)
            .create_async()
            .await;

        let embedder = Embedder::for_ollama(&server.url(), "test-model").unwrap();
        let v = embedder.embed("hello world").await.unwrap();
        assert_eq!(v, vec![0.1, 0.2, 0.3, 0.4]);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_ollama_embed_500_error_returns_http_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/api/embed")
            .with_status(500)
            .with_body("server error")
            .create_async()
            .await;

        let embedder = Embedder::for_ollama(&server.url(), "test-model").unwrap();
        let err = embedder.embed("hello").await.unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Http(_)),
            "expected Http error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_ollama_embed_malformed_json() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/api/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body("not json at all")
            .create_async()
            .await;

        let embedder = Embedder::for_ollama(&server.url(), "test-model").unwrap();
        let err = embedder.embed("hello").await.unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Parse(_)),
            "expected Parse error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_ollama_embed_shape_mismatch() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/api/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            // Request asks for 2 texts, server returns 1 embedding
            .with_body(r#"{"embeddings": [[0.1, 0.2]]}"#)
            .create_async()
            .await;

        let embedder = Embedder::for_ollama(&server.url(), "test-model").unwrap();
        let err = embedder
            .embed_batch(&["first text", "second text"])
            .await
            .unwrap_err();
        assert!(
            matches!(err, EmbeddingError::Shape(_)),
            "expected Shape error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_openai_embed_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/embeddings")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "data": [
                        {"embedding": [0.9, 0.8, 0.7], "index": 0}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            Embedder::for_openai(&server.url(), "text-embedding-3-small", "test-key").unwrap();
        let v = embedder.embed("hello").await.unwrap();
        assert_eq!(v, vec![0.9, 0.8, 0.7]);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_openai_embed_reorders_by_index() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            // Intentionally out of order
            .with_body(
                r#"{
                    "data": [
                        {"embedding": [0.2], "index": 1},
                        {"embedding": [0.1], "index": 0}
                    ]
                }"#,
            )
            .create_async()
            .await;

        let embedder = Embedder::for_openai(&server.url(), "model", "key").unwrap();
        let batch = embedder.embed_batch(&["a", "b"]).await.unwrap();
        assert_eq!(batch[0], vec![0.1]);
        assert_eq!(batch[1], vec![0.2]);
    }
}
