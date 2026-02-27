//! Embedding pipeline with pluggable providers.
//!
//! Supports multiple embedding backends:
//! - `OllamaProvider` — calls Ollama's `/api/embed` endpoint (default, zero setup)
//! - `LocalProvider` — ONNX Runtime inference with BGE-small-en-v1.5 (offline, fastest)
//!
//! Provider selection strategy:
//! 1. Config: user can explicitly set `embedding.provider`
//! 2. Auto: try Ollama first (health check), fall back to Local ONNX

use std::path::{Path, PathBuf};

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn};

/// Embedding vector dimension.
pub const EMBEDDING_DIM: usize = 384;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the embedding pipeline.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Model not found at {0}. Run `brain init` or download manually.")]
    ModelNotFound(PathBuf),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Download error: {0}")]
    Download(String),

    #[error("Shape error: {0}")]
    Shape(String),

    #[error("Ollama error: {0}")]
    Ollama(String),

    #[error("Provider not available: {0}")]
    ProviderUnavailable(String),
}

// ─── Provider Trait ──────────────────────────────────────────────────────────

/// The embedding provider trait — all backends implement this.
///
/// Performance notes:
/// - `OllamaProvider`: ~5ms per embed call (local HTTP, GPU-accelerated)
/// - `LocalProvider`: ~2ms per embed call (in-process ONNX, CPU/CoreML)
/// - Batch operations amortize HTTP overhead for Ollama
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text string.
    fn embed(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, EmbeddingError>> + Send;

    /// Embed a batch of texts. Implementations should optimize for throughput.
    fn embed_batch(
        &mut self,
        texts: &[&str],
    ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>, EmbeddingError>> + Send;

    /// Vector dimension this provider produces.
    fn dimension(&self) -> usize;

    /// Human-readable provider name (for logging/status).
    fn name(&self) -> &str;
}

// ─── Ollama Provider ─────────────────────────────────────────────────────────

/// Ollama embedding provider — calls the local Ollama server.
///
/// Uses `POST /api/embed` which supports batch input natively.
/// Fastest startup (no model download), uses GPU if available via Ollama.
#[derive(Debug)]
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dim: usize,
}

/// Ollama /api/embed request body.
#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

/// Ollama /api/embed response body.
#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaProvider {
    /// Create a new Ollama embedding provider.
    pub fn new(base_url: &str, model: &str, dim: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            dim,
        }
    }

    /// Create with default config (localhost:11434, nomic-embed-text, 384-dim).
    pub fn default_config() -> Self {
        Self::new("http://localhost:11434", "nomic-embed-text", EMBEDDING_DIM)
    }

    /// Check if Ollama server is reachable.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
}

impl EmbeddingProvider for OllamaProvider {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut batch = self.embed_batch(&[text]).await?;
        Ok(batch.remove(0))
    }

    async fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base_url);
        let request = OllamaEmbedRequest {
            model: &self.model,
            input: texts.to_vec(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| EmbeddingError::Ollama(format!("Request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(EmbeddingError::Ollama(format!("HTTP {status}: {body}")));
        }

        let response: OllamaEmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbeddingError::Ollama(format!("Failed to parse response: {e}")))?;

        if response.embeddings.len() != texts.len() {
            return Err(EmbeddingError::Shape(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                response.embeddings.len()
            )));
        }

        Ok(response.embeddings)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &str {
        "ollama"
    }
}

// ─── Local ONNX Provider ────────────────────────────────────────────────────

/// Local ONNX embedding provider — runs inference in-process.
///
/// Fastest per-call latency (no HTTP overhead), works fully offline.
/// Requires model files (model.onnx + tokenizer.json) in the model directory.
#[derive(Debug)]
pub struct LocalProvider {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

/// Files required for the local ONNX model.
const MODEL_FILENAME: &str = "model.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// HuggingFace Hub for model downloads.
const HF_REPO: &str = "BAAI/bge-small-en-v1.5";
const HF_BASE: &str = "https://huggingface.co";

impl LocalProvider {
    /// Load the ONNX model and tokenizer from a directory.
    pub fn load(model_dir: &Path) -> Result<Self, EmbeddingError> {
        let model_path = model_dir.join(MODEL_FILENAME);
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);

        if !model_path.exists() {
            return Err(EmbeddingError::ModelNotFound(model_path));
        }

        let session = ort::session::Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbeddingError::Tokenizer(e.to_string()))?;

        info!(
            "Local ONNX embedder loaded from {} (inputs: {}, outputs: {})",
            model_dir.display(),
            session.inputs().len(),
            session.outputs().len()
        );

        Ok(Self { session, tokenizer })
    }

    /// Synchronous embed (no async needed for local inference).
    pub fn embed_sync(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let batch = self.embed_batch_sync(&[text])?;
        Ok(batch.into_iter().next().unwrap())
    }

    /// Synchronous batch embed.
    pub fn embed_batch_sync(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbeddingError::Tokenizer(e.to_string()))?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build padded input tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();

            for (j, (&id, (&m, &t))) in ids.iter().zip(mask.iter().zip(types.iter())).enumerate() {
                let idx = i * max_len + j;
                input_ids[idx] = id as i64;
                attention_mask[idx] = m as i64;
                token_type_ids[idx] = t as i64;
            }
        }

        let ids_array = Array2::from_shape_vec((batch_size, max_len), input_ids)
            .map_err(|e| EmbeddingError::Shape(e.to_string()))?;
        let mask_array = Array2::from_shape_vec((batch_size, max_len), attention_mask.clone())
            .map_err(|e| EmbeddingError::Shape(e.to_string()))?;
        let types_array = Array2::from_shape_vec((batch_size, max_len), token_type_ids)
            .map_err(|e| EmbeddingError::Shape(e.to_string()))?;

        // Run ONNX inference
        let ids_tensor = ort::value::TensorRef::from_array_view(&ids_array)?;
        let mask_tensor = ort::value::TensorRef::from_array_view(&mask_array)?;
        let types_tensor = ort::value::TensorRef::from_array_view(&types_array)?;

        let outputs = self
            .session
            .run(ort::inputs![ids_tensor, mask_tensor, types_tensor])?;

        // Extract output: (batch_size, seq_len, EMBEDDING_DIM)
        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::Shape(format!("Output extraction failed: {e}")))?;

        let dims: Vec<usize> = output_shape.iter().map(|&d| d as usize).collect();
        if dims.len() != 3 || dims[2] != EMBEDDING_DIM {
            return Err(EmbeddingError::Shape(format!(
                "Unexpected shape: {:?}, expected (batch, seq, {EMBEDDING_DIM})",
                dims
            )));
        }

        let seq_len = dims[1];

        // Mean pooling with attention mask + L2 normalization
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; EMBEDDING_DIM];
            let mut mask_sum: f32 = 0.0;

            for j in 0..seq_len {
                let mask_val = attention_mask[i * max_len + j] as f32;
                if mask_val > 0.0 {
                    mask_sum += mask_val;
                    let offset = i * seq_len * EMBEDDING_DIM + j * EMBEDDING_DIM;
                    for k in 0..EMBEDDING_DIM {
                        pooled[k] += output_data[offset + k];
                    }
                }
            }

            if mask_sum > 0.0 {
                for val in pooled.iter_mut() {
                    *val /= mask_sum;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in pooled.iter_mut() {
                    *val /= norm;
                }
            }

            results.push(pooled);
        }

        Ok(results)
    }

    /// Download the BGE-small-en-v1.5 ONNX model from HuggingFace.
    pub async fn download_model(target_dir: &Path) -> Result<PathBuf, EmbeddingError> {
        std::fs::create_dir_all(target_dir)?;

        let client = reqwest::Client::builder()
            .user_agent("brain/0.1")
            .build()
            .map_err(|e| EmbeddingError::Download(e.to_string()))?;

        for filename in [MODEL_FILENAME, TOKENIZER_FILENAME] {
            let target_path = target_dir.join(filename);
            if target_path.exists() {
                info!("Model file already exists: {}", target_path.display());
                continue;
            }

            let url = format!("{HF_BASE}/{HF_REPO}/resolve/main/{filename}");
            info!("Downloading {filename} from {url}...");

            let resp = client.get(&url).send().await.map_err(|e| {
                EmbeddingError::Download(format!("Failed to download {filename}: {e}"))
            })?;

            if !resp.status().is_success() {
                return Err(EmbeddingError::Download(format!(
                    "HTTP {} downloading {filename}",
                    resp.status()
                )));
            }

            let bytes = resp
                .bytes()
                .await
                .map_err(|e| EmbeddingError::Download(format!("Failed to read {filename}: {e}")))?;

            std::fs::write(&target_path, &bytes)?;
            info!(
                "Downloaded {} ({} bytes)",
                target_path.display(),
                bytes.len()
            );
        }

        Ok(target_dir.to_path_buf())
    }
}

impl EmbeddingProvider for LocalProvider {
    async fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.embed_sync(text)
    }

    async fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.embed_batch_sync(texts)
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }

    fn name(&self) -> &str {
        "localonnx"
    }
}

// ─── Embedder (auto-selecting wrapper) ───────────────────────────────────────

/// Which embedding provider to use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// Try Ollama first, fall back to Local ONNX.
    #[default]
    Auto,
    /// Always use Ollama.
    Ollama,
    /// Always use local ONNX model.
    Local,
}

/// Configuration for the embedding system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: ProviderType,
    /// Ollama base URL.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    /// Model name for Ollama embeddings.
    #[serde(default = "default_ollama_model")]
    pub ollama_model: String,
    /// Directory for local ONNX model files.
    #[serde(default = "default_model_dir")]
    pub model_dir: String,
    /// Embedding dimension.
    #[serde(default = "default_dim")]
    pub dimensions: usize,
}

fn default_ollama_url() -> String {
    "http://localhost:11434".into()
}
fn default_ollama_model() -> String {
    "nomic-embed-text".into()
}
fn default_model_dir() -> String {
    "~/.brain/models/bge-small-en-v1.5".into()
}
fn default_dim() -> usize {
    EMBEDDING_DIM
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: ProviderType::Auto,
            ollama_url: default_ollama_url(),
            ollama_model: default_ollama_model(),
            model_dir: default_model_dir(),
            dimensions: EMBEDDING_DIM,
        }
    }
}

/// The main embedder — wraps the active provider.
///
/// Use `Embedder::from_config()` for auto-detection:
/// - In `Auto` mode, probes Ollama first (fast health check). If Ollama is
///   running, uses it (zero model download, GPU-accelerated).
/// - Falls back to local ONNX if Ollama is unavailable and model files exist.
#[allow(clippy::large_enum_variant)]
pub enum Embedder {
    Ollama(OllamaProvider),
    Local(LocalProvider),
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ollama(p) => write!(f, "Embedder::Ollama({:?})", p),
            Self::Local(p) => write!(f, "Embedder::Local({:?})", p),
        }
    }
}

impl Embedder {
    /// Create an embedder from configuration with auto-detection.
    pub async fn from_config(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        match config.provider {
            ProviderType::Ollama => {
                let provider = OllamaProvider::new(
                    &config.ollama_url,
                    &config.ollama_model,
                    config.dimensions,
                );
                if !provider.health_check().await {
                    return Err(EmbeddingError::ProviderUnavailable(format!(
                        "Ollama not reachable at {}",
                        config.ollama_url
                    )));
                }
                info!(
                    "Using Ollama embedding provider (model: {})",
                    config.ollama_model
                );
                Ok(Self::Ollama(provider))
            }
            ProviderType::Local => {
                let model_dir = shellexpand::tilde(&config.model_dir).to_string();
                let provider = LocalProvider::load(Path::new(&model_dir))?;
                info!("Using Local ONNX embedding provider");
                Ok(Self::Local(provider))
            }
            ProviderType::Auto => {
                // Try Ollama first (fast health check, ~10ms)
                let ollama = OllamaProvider::new(
                    &config.ollama_url,
                    &config.ollama_model,
                    config.dimensions,
                );
                if ollama.health_check().await {
                    info!(
                        "Auto-detected Ollama at {} — using for embeddings",
                        config.ollama_url
                    );
                    return Ok(Self::Ollama(ollama));
                }

                // Fall back to local ONNX
                warn!("Ollama not available, trying local ONNX model...");
                let model_dir = shellexpand::tilde(&config.model_dir).to_string();
                let provider = LocalProvider::load(Path::new(&model_dir))?;
                info!("Using Local ONNX embedding provider (fallback)");
                Ok(Self::Local(provider))
            }
        }
    }

    /// Embed a single text.
    pub async fn embed(&mut self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        match self {
            Self::Ollama(p) => p.embed(text).await,
            Self::Local(p) => p.embed(text).await,
        }
    }

    /// Embed a batch of texts.
    pub async fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match self {
            Self::Ollama(p) => p.embed_batch(texts).await,
            Self::Local(p) => p.embed_batch(texts).await,
        }
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        match self {
            Self::Ollama(p) => p.dimension(),
            Self::Local(p) => p.dimension(),
        }
    }

    /// Get the active provider name.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Ollama(p) => p.name(),
            Self::Local(p) => p.name(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".brain/models/bge-small-en-v1.5")
    }

    #[test]
    fn test_model_not_found() {
        let result = LocalProvider::load(Path::new("/nonexistent/path"));
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbeddingError::ModelNotFound(_) => {}
            other => panic!("Expected ModelNotFound, got: {other}"),
        }
    }

    #[test]
    fn test_provider_type_default() {
        let pt = ProviderType::default();
        assert_eq!(pt, ProviderType::Auto);
    }

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.provider, ProviderType::Auto);
        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.ollama_model, "nomic-embed-text");
        assert_eq!(config.dimensions, EMBEDDING_DIM);
    }

    #[test]
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new("http://localhost:11434", "nomic-embed-text", 384);
        assert_eq!(provider.dimension(), 384);
        assert_eq!(provider.name(), "ollama");
    }

    /// Downloads the ONNX model from HuggingFace. Requires network access.
    #[tokio::test]
    #[ignore = "Requires network access to download ~100MB model from HuggingFace"]
    async fn test_download_model() {
        let dir = model_dir();
        LocalProvider::download_model(&dir).await.unwrap();
        assert!(dir.join(MODEL_FILENAME).exists());
        assert!(dir.join(TOKENIZER_FILENAME).exists());
    }

    /// Tests local ONNX embedding. Requires the model to be downloaded first.
    #[test]
    #[ignore = "Requires ONNX model files (~100MB) to be downloaded first"]
    fn test_local_embed_single() {
        let dir = model_dir();
        let mut provider = LocalProvider::load(&dir).unwrap();
        let vec = provider.embed_sync("Hello, world!").unwrap();
        assert_eq!(vec.len(), EMBEDDING_DIM);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "Norm should be ~1.0, got {norm}");
    }

    /// Tests batch embedding via local ONNX. Requires model.
    #[test]
    #[ignore = "Requires ONNX model files (~100MB) to be downloaded first"]
    fn test_local_embed_batch() {
        let dir = model_dir();
        let mut provider = LocalProvider::load(&dir).unwrap();

        let texts = vec![
            "Hello world",
            "Goodbye world",
            "Something completely different",
        ];
        let vecs = provider.embed_batch_sync(&texts).unwrap();
        assert_eq!(vecs.len(), 3);
        for vec in &vecs {
            assert_eq!(vec.len(), EMBEDDING_DIM);
        }

        let sim_close = cosine_sim(&vecs[0], &vecs[1]);
        let sim_far = cosine_sim(&vecs[0], &vecs[2]);
        assert!(
            sim_close > sim_far,
            "Similar texts should have higher similarity"
        );
    }

    /// Tests Ollama provider. Requires Ollama running locally.
    #[tokio::test]
    #[ignore = "Requires Ollama server running locally with embedding model"]
    async fn test_ollama_embed() {
        let mut provider = OllamaProvider::default_config();
        if !provider.health_check().await {
            eprintln!("Ollama not running, skipping test");
            return;
        }
        let vec = provider.embed("Hello, world!").await.unwrap();
        assert!(!vec.is_empty(), "Should produce a non-empty embedding");
    }

    /// Tests auto-detection. Requires either Ollama or local model.
    #[tokio::test]
    #[ignore = "Requires either Ollama server or ONNX model files"]
    async fn test_auto_detection() {
        let config = EmbeddingConfig::default();
        let embedder = Embedder::from_config(&config).await;
        if let Ok(e) = &embedder {
            eprintln!("Auto-detected provider: {}", e.provider_name());
        }
        // Should succeed if either Ollama or local model is available
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }
}
