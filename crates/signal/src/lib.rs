//! # Brain Signal Processor
//!
//! Central hub that converts all input signals (CLI, HTTP, WebSocket, MCP, gRPC)
//! into a unified Signal type and routes them through the Brain pipeline.
//!
//! The SignalProcessor wires together:
//! - Thalamus (intent classification)
//! - Amygdala (importance scoring)
//! - Hippocampus (episodic + semantic memory)
//! - Cortex (LLM reasoning + context assembly)

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use hippocampus::embedding::EmbeddingProvider;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the signal processing layer.
#[derive(Debug, Error)]
pub enum SignalError {
    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("LLM error: {0}")]
    Llm(#[from] cortex::LlmError),

    #[error("Initialization error: {0}")]
    Init(String),
}

// ─── Signal Types ─────────────────────────────────────────────────────────────

/// The source protocol of an incoming signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalSource {
    Cli,
    Http,
    WebSocket,
    Mcp,
    Grpc,
}

/// A unified signal — the single input type for all protocol adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,
    pub source: SignalSource,
    pub channel: String,
    pub sender: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl Signal {
    /// Create a new Signal with a generated UUID and current timestamp.
    pub fn new(
        source: SignalSource,
        channel: impl Into<String>,
        sender: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source,
            channel: channel.into(),
            sender: sender.into(),
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }
}

// ─── Response Types ───────────────────────────────────────────────────────────

/// Status of a signal response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseStatus {
    Ok,
    Error,
    Processing,
}

/// Content payload of a signal response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ResponseContent {
    Text(String),
    Json(serde_json::Value),
    Error(String),
}

/// Memory context included in every response — tracks what memory was used.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Number of semantic facts used to construct the response.
    pub facts_used: usize,
    /// Number of episodic memories used to construct the response.
    pub episodes_used: usize,
}

/// The response to a processed signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResponse {
    pub signal_id: Uuid,
    pub status: ResponseStatus,
    pub response: ResponseContent,
    pub memory_context: MemoryContext,
}

impl SignalResponse {
    /// Create a successful text response.
    pub fn ok(signal_id: Uuid, text: impl Into<String>) -> Self {
        Self {
            signal_id,
            status: ResponseStatus::Ok,
            response: ResponseContent::Text(text.into()),
            memory_context: MemoryContext::default(),
        }
    }

    /// Create an error response.
    pub fn error(signal_id: Uuid, error: impl Into<String>) -> Self {
        Self {
            signal_id,
            status: ResponseStatus::Error,
            response: ResponseContent::Error(error.into()),
            memory_context: MemoryContext::default(),
        }
    }
}

// ─── Signal Adapter Trait ─────────────────────────────────────────────────────

/// Trait implemented by all protocol adapters (HTTP, WebSocket, MCP, gRPC, CLI).
///
/// Each adapter converts protocol-specific messages into Signal values,
/// submits them to SignalProcessor, and delivers the SignalResponse back
/// to the originating client via `send()`.
#[async_trait::async_trait]
pub trait SignalAdapter: Send + Sync {
    /// Return the source type for this adapter.
    fn source(&self) -> SignalSource;

    /// Send a response back to the adapter's client.
    async fn send(&self, response: SignalResponse) -> Result<(), SignalError>;
}

// ─── Signal Processor ─────────────────────────────────────────────────────────

/// Central processor that wires all Brain subsystems together.
///
/// One instance is shared across all adapters. Each incoming Signal is routed
/// through intent classification → importance scoring → memory → LLM → response.
pub struct SignalProcessor {
    config: brain_core::BrainConfig,
    classifier: thalamus::IntentClassifier,
    importance: amygdala::ImportanceScorer,
    episodic: hippocampus::EpisodicStore,
    semantic: Option<hippocampus::SemanticStore>,
    embedder: tokio::sync::Mutex<hippocampus::embedding::OllamaProvider>,
    recall_engine: hippocampus::RecallEngine,
    llm: Box<dyn cortex::LlmProvider>,
    context_assembler: cortex::context::ContextAssembler,
}

impl SignalProcessor {
    /// Initialize the signal processor, wiring all Brain subsystems.
    ///
    /// Opens the SQLite database, connects to RuVector, creates the LLM provider,
    /// and wires the intent classifier, importance scorer, and context assembler.
    pub async fn new(config: brain_core::BrainConfig) -> Result<Self, SignalError> {
        // Open SQLite pool
        let db = storage::SqlitePool::open(&config.sqlite_path())
            .map_err(|e| SignalError::Init(format!("SQLite: {e}")))?;

        // Create episodic store
        let episodic = hippocampus::EpisodicStore::new(db.clone());

        // Create semantic store (optional — fails gracefully if RuVector unavailable)
        let semantic = match storage::RuVectorStore::open(&config.ruvector_path()).await {
            Ok(ruv) => {
                ruv.ensure_tables().await.ok();
                Some(hippocampus::SemanticStore::new(db.clone(), ruv))
            }
            Err(e) => {
                tracing::warn!("RuVector unavailable, semantic memory disabled: {e}");
                None
            }
        };

        // Create LLM provider
        let llm_config = cortex::llm::ProviderConfig {
            provider: config.llm.provider.clone(),
            base_url: config.llm.base_url.clone(),
            api_key: None,
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens as i32,
        };
        let llm = cortex::llm::create_provider(&llm_config);

        // Create Ollama embedder (falls back to zero vector at call time if Ollama unavailable)
        let embedder =
            tokio::sync::Mutex::new(hippocampus::embedding::OllamaProvider::default_config());

        // Create recall engine with default RRF config
        let recall_engine = hippocampus::RecallEngine::with_defaults();

        Ok(Self {
            config,
            classifier: thalamus::IntentClassifier::new(),
            importance: amygdala::ImportanceScorer::new(),
            episodic,
            semantic,
            embedder,
            recall_engine,
            llm,
            context_assembler: cortex::context::ContextAssembler::with_defaults(),
        })
    }

    /// Process a signal through the full Brain pipeline.
    ///
    /// Routes by intent:
    /// - `StoreFact`  → Amygdala importance → Hippocampus semantic store → confirmation
    /// - `Recall`     → Hippocampus hybrid search → Cortex context assembly → LLM response
    /// - `Chat`       → Hippocampus context → Cortex LLM → Hippocampus episode store
    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError> {
        let signal_id = signal.id;

        // 1. Score importance via Amygdala
        let importance = self.importance.score(&signal.content);

        // 2. Classify intent via Thalamus
        let classification = self.classifier.classify(&signal.content);

        tracing::debug!(
            signal_id = %signal_id,
            source = ?signal.source,
            intent = ?classification.intent,
            importance = importance,
            "Signal classified"
        );

        match classification.intent {
            // ── STORE_FACT: importance score → semantic memory → confirmation ────
            thalamus::Intent::StoreFact {
                subject,
                predicate,
                object,
            } => {
                let fact_text = format!("{subject} {predicate} {object}");
                let vector = self.embed_text(&fact_text).await;

                let mut facts_stored = 0;
                if let Some(semantic) = &self.semantic {
                    match semantic
                        .store_fact(
                            "signal",
                            &subject,
                            &predicate,
                            &object,
                            importance as f64,
                            None,
                            vector,
                        )
                        .await
                    {
                        Ok(_) => facts_stored = 1,
                        Err(e) => tracing::warn!("Failed to store fact in semantic memory: {e}"),
                    }
                }

                Ok(SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(format!(
                        "Stored: {subject} {predicate} {object} (importance: {importance:.2})"
                    )),
                    memory_context: MemoryContext {
                        facts_used: facts_stored,
                        episodes_used: 0,
                    },
                })
            }

            // ── RECALL_MEMORY: hybrid search → Cortex context → LLM response ───
            thalamus::Intent::Recall { query } => {
                let query_vector = self.embed_text(&query).await;
                let (memories, facts_used, episodes_used) =
                    self.do_recall(&query, query_vector, 10).await;

                let messages = self.context_assembler.assemble(&query, &memories, &[]);
                let llm_response = self.llm.generate(&messages).await?;

                Ok(SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(llm_response.content),
                    memory_context: MemoryContext {
                        facts_used,
                        episodes_used,
                    },
                })
            }

            // ── CHAT: context fetch → LLM response → episode store ───────────
            thalamus::Intent::Chat { content } => {
                // Fetch relevant memory context
                let query_vector = self.embed_text(&content).await;
                let (memories, facts_used, episodes_used) =
                    self.do_recall(&content, query_vector, 10).await;

                // Create session and store the user turn
                let session_id = self
                    .episodic
                    .create_session(&signal.channel)
                    .map_err(|e| SignalError::Storage(e.to_string()))?;

                self.episodic
                    .store_episode(&session_id, "user", &signal.content, importance as f64)
                    .map_err(|e| SignalError::Storage(e.to_string()))?;

                // Generate LLM response with assembled context
                let messages = self.context_assembler.assemble(&content, &memories, &[]);
                let llm_response = self.llm.generate(&messages).await?;

                // Store the assistant turn in episodic memory
                self.episodic
                    .store_episode(&session_id, "assistant", &llm_response.content, 0.5)
                    .map_err(|e| SignalError::Storage(e.to_string()))?;

                Ok(SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(llm_response.content),
                    memory_context: MemoryContext {
                        facts_used,
                        episodes_used,
                    },
                })
            }

            // ── Other intents: route acknowledgement ─────────────────────────
            other => Ok(SignalResponse::ok(
                signal_id,
                format!("Intent classified: {:?}", other),
            )),
        }
    }

    /// Generate a vector embedding for text.
    ///
    /// Falls back to a zero vector if the Ollama embedder is unavailable,
    /// so the pipeline degrades gracefully without panicking.
    async fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut embedder = self.embedder.lock().await;
        match embedder.embed(text).await {
            Ok(vec) => vec,
            Err(e) => {
                tracing::warn!("Embedding failed, using zero vector: {e}");
                vec![0.0_f32; hippocampus::EMBEDDING_DIM]
            }
        }
    }

    /// Run hybrid recall (BM25 + ANN via RecallEngine) and return memories with counts.
    ///
    /// If the semantic store is unavailable, falls back to BM25-only episodic search.
    async fn do_recall(
        &self,
        query: &str,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> (Vec<hippocampus::Memory>, usize, usize) {
        if let Some(semantic) = &self.semantic {
            match self
                .recall_engine
                .recall(query, query_vector, &self.episodic, semantic, top_k)
                .await
            {
                Ok(memories) => {
                    let facts_used = memories
                        .iter()
                        .filter(|m| m.source == hippocampus::MemorySource::Semantic)
                        .count();
                    let episodes_used = memories
                        .iter()
                        .filter(|m| m.source == hippocampus::MemorySource::Episodic)
                        .count();
                    (memories, facts_used, episodes_used)
                }
                Err(e) => {
                    tracing::warn!("Recall engine failed: {e}");
                    (Vec::new(), 0, 0)
                }
            }
        } else {
            // Semantic store unavailable — fall back to episodic BM25 only
            let bm25 = self.episodic.search_bm25(query, top_k).unwrap_or_default();
            let episodes_used = bm25.len();
            let memories = bm25
                .into_iter()
                .map(|r| hippocampus::Memory {
                    id: r.episode_id,
                    content: r.content,
                    source: hippocampus::MemorySource::Episodic,
                    score: r.rank as f64,
                    importance: 0.5,
                    timestamp: String::new(),
                })
                .collect();
            (memories, 0, episodes_used)
        }
    }

    /// Expose the config (for adapter use).
    pub fn config(&self) -> &brain_core::BrainConfig {
        &self.config
    }

    /// Expose the episodic store (for adapter use).
    pub fn episodic(&self) -> &hippocampus::EpisodicStore {
        &self.episodic
    }

    /// Expose the semantic store (for adapter use).
    pub fn semantic(&self) -> Option<&hippocampus::SemanticStore> {
        self.semantic.as_ref()
    }

    /// Expose the LLM provider (for adapter use).
    pub fn llm(&self) -> &dyn cortex::LlmProvider {
        self.llm.as_ref()
    }

    /// Expose the context assembler (for adapter use).
    pub fn context_assembler(&self) -> &cortex::context::ContextAssembler {
        &self.context_assembler
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_new() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello world");
        assert!(!signal.id.is_nil());
        assert_eq!(signal.source, SignalSource::Cli);
        assert_eq!(signal.channel, "cli");
        assert_eq!(signal.sender, "user");
        assert_eq!(signal.content, "hello world");
        assert!(signal.metadata.is_empty());
    }

    #[test]
    fn test_signal_response_ok() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "success");
        assert_eq!(resp.signal_id, id);
        assert_eq!(resp.status, ResponseStatus::Ok);
        assert!(matches!(resp.response, ResponseContent::Text(_)));
        assert_eq!(resp.memory_context.facts_used, 0);
        assert_eq!(resp.memory_context.episodes_used, 0);
    }

    #[test]
    fn test_signal_response_error() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::error(id, "something went wrong");
        assert_eq!(resp.status, ResponseStatus::Error);
        assert!(matches!(resp.response, ResponseContent::Error(_)));
    }

    #[test]
    fn test_memory_context_default() {
        let ctx = MemoryContext::default();
        assert_eq!(ctx.facts_used, 0);
        assert_eq!(ctx.episodes_used, 0);
    }

    #[test]
    fn test_signal_source_serde() {
        let sources = vec![
            SignalSource::Cli,
            SignalSource::Http,
            SignalSource::WebSocket,
            SignalSource::Mcp,
            SignalSource::Grpc,
        ];
        for s in &sources {
            let json = serde_json::to_string(s).unwrap();
            let back: SignalSource = serde_json::from_str(&json).unwrap();
            assert_eq!(s, &back);
        }
    }

    #[test]
    fn test_signal_serde() {
        let signal = Signal::new(SignalSource::Http, "http", "api-client", "Remember coffee");
        let json = serde_json::to_string(&signal).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(signal.id, back.id);
        assert_eq!(signal.content, back.content);
    }

    #[test]
    fn test_signal_response_serde() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "hello");
        let json = serde_json::to_string(&resp).unwrap();
        let back: SignalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.signal_id, back.signal_id);
        assert_eq!(resp.status, back.status);
    }

    /// Integration test: CLI input → SignalProcessor → StoreFact → memory stored.
    ///
    /// Tests the full STORE_FACT pipeline without requiring a running LLM or
    /// embedding model (embedding gracefully degrades to zero vector).
    #[tokio::test]
    async fn test_process_store_fact_integration() {
        let temp_dir = tempfile::tempdir().unwrap();

        let mut config = brain_core::BrainConfig::default();
        // Point data_dir to a temp directory so we don't touch ~/.brain
        config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

        let processor = SignalProcessor::new(config).await.unwrap();

        // "Remember that Rust is fast" → StoreFact intent
        let signal = Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "Remember that Rust is fast",
        );

        let response = processor.process(signal).await.unwrap();

        assert_eq!(response.status, ResponseStatus::Ok);
        // StoreFact stores in semantic memory → facts_used = 1
        assert_eq!(response.memory_context.facts_used, 1);
        assert_eq!(response.memory_context.episodes_used, 0);
        // Response text should confirm the stored fact
        if let ResponseContent::Text(text) = &response.response {
            assert!(text.contains("Rust"));
        } else {
            panic!("Expected Text response");
        }
    }

    /// Integration test: chat signal creates episodic memory entries.
    ///
    /// Requires Ollama running locally. Without it, the test hangs for ~120s
    /// waiting for the HTTP timeout, so it is skipped in normal CI.
    #[tokio::test]
    #[ignore = "Requires Ollama server running locally"]
    async fn test_process_chat_reaches_llm() {
        let temp_dir = tempfile::tempdir().unwrap();

        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

        let processor = SignalProcessor::new(config).await.unwrap();

        let signal = Signal::new(SignalSource::Cli, "cli", "user", "Hello, how are you?");

        // The LLM call will fail if Ollama is not running — that's expected.
        // The important thing is that the pipeline doesn't panic and that the
        // error is a SignalError::Llm (not a storage or routing error).
        let result = processor.process(signal).await;
        match result {
            Ok(resp) => {
                // Ollama is running — verify response structure
                assert_eq!(resp.status, ResponseStatus::Ok);
            }
            Err(SignalError::Llm(_)) => {
                // Expected when Ollama is not running — pipeline is wired correctly
            }
            Err(other) => {
                panic!("Unexpected error (should be Llm, not storage/routing): {other}");
            }
        }
    }
}
