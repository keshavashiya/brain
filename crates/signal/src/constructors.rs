//! SignalProcessor constructors and initialization.

use std::sync::Arc;

use crate::types::*;
use crate::SignalProcessor;

/// Resolve the LLM API key from config, with env var fallback for backwards compatibility.
fn resolve_llm_api_key(config: &brain_core::BrainConfig) -> String {
    let from_config = config.llm.api_key.trim().to_string();
    if !from_config.is_empty() {
        return from_config;
    }
    std::env::var("BRAIN_LLM__API_KEY").unwrap_or_default()
}

impl SignalProcessor {
    /// Initialize the signal processor, wiring all Brain subsystems.
    ///
    /// Opens the SQLite database, connects to RuVector, creates the LLM provider,
    /// and wires the intent classifier, importance scorer, and context assembler.
    pub async fn new(config: brain_core::BrainConfig) -> Result<Self, SignalError> {
        Self::new_with_encryptor(
            config,
            #[cfg(feature = "encryption")]
            None,
        )
        .await
    }

    /// Like `new`, but wires an `Encryptor` into all storage backends.
    ///
    /// When provided, `episodes.content` and `semantic_facts.object` are
    /// AES-256-GCM encrypted at rest; RuVector `content` fields are also
    /// encrypted in their JSON files on disk.
    pub async fn new_with_encryptor(
        config: brain_core::BrainConfig,
        #[cfg(feature = "encryption")] encryptor: Option<storage::Encryptor>,
    ) -> Result<Self, SignalError> {
        // Open SQLite pool — attach encryptor if provided
        let db = {
            let pool = storage::SqlitePool::open(&config.sqlite_path())
                .map_err(|e| SignalError::Init(format!("SQLite: {e}")))?;
            #[cfg(feature = "encryption")]
            let pool = if let Some(enc) = encryptor.clone() {
                tracing::info!("Encryption enabled: SQLite content columns will be encrypted");
                pool.with_encryptor(enc)
            } else {
                pool
            };
            pool
        };

        // Create episodic store
        let episodic = hippocampus::EpisodicStore::new(db.clone());

        // Create procedure store (cerebellum) — initialises its own table
        let procedures = cerebellum::ProcedureStore::new(db.clone());
        if let Err(e) = procedures.ensure_tables() {
            tracing::warn!("ProcedureStore table init failed (non-fatal): {e}");
        }

        // Create LLM provider
        let llm_api_key = resolve_llm_api_key(&config);
        let llm_config = cortex::llm::ProviderConfig {
            provider: config.llm.provider.clone(),
            base_url: config.llm.base_url.clone(),
            api_key: if llm_api_key.is_empty() {
                None
            } else {
                Some(llm_api_key.clone())
            },
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens as i32,
        };
        let llm: Arc<dyn cortex::LlmProvider> = cortex::llm::create_provider(&llm_config)
            .map_err(|e| SignalError::Init(format!("Failed to create LLM provider: {e}")))?
            .into();

        // Create embedder — provider is selected from llm.provider config.
        // The model and dimension come from the embedding config section.
        // embedding.dimensions MUST match the model's actual output size.
        let embedding_dim = config.embedding.dimensions as usize;
        tracing::info!(
            provider = config.llm.provider,
            model = config.embedding.model,
            dim = embedding_dim,
            "Embedding provider selected"
        );
        let embedder_inner = hippocampus::Embedder::from_config(
            &config.llm.provider,
            &config.llm.base_url,
            &config.embedding.model,
            &llm_api_key,
        )
        .map_err(|e| SignalError::Init(format!("Failed to create embedding provider: {e}")))?;
        let embedder = tokio::sync::Mutex::new(embedder_inner);

        // Create semantic store (optional — fails gracefully if RuVector unavailable).
        // Pass the probed embedding_dim so VectorDB is sized to match the provider.
        // Note: ruvector-core stores only IDs; content encryption is handled by SQLite.
        let semantic = match storage::RuVectorStore::open(&config.ruvector_path(), embedding_dim)
            .await
        {
            Ok(ruv) => {
                #[cfg(feature = "encryption")]
                if encryptor.is_some() {
                    tracing::info!("Encryption enabled: vector IDs stored in ruvector-core, content encrypted in SQLite");
                }
                match ruv.ensure_tables().await {
                    Ok(()) => Some(hippocampus::SemanticStore::new(db.clone(), ruv)),
                    Err(e) => {
                        tracing::warn!(
                            "RuVector table initialization failed, semantic memory disabled: {e}. \
                             Another brain process may be holding the file lock — \
                             check `ps aux | grep brain` and stop standalone brain mcp processes."
                        );
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!("RuVector unavailable, semantic memory disabled: {e}");
                None
            }
        };

        // Create recall engine from user config
        let search_cfg = &config.memory.search;
        let recall_engine = hippocampus::RecallEngine::new(hippocampus::RecallConfig::from_config(
            search_cfg.rrf_k,
            search_cfg.pre_fusion_limit,
            search_cfg.importance_weight,
            search_cfg.recency_weight,
            search_cfg.decay_rate,
            config.memory.semantic.similarity_threshold,
        ));
        let (events_tx, _) = tokio::sync::broadcast::channel(512);

        let classifier = thalamus::IntentClassifier::new()
            .with_llm_fallback(Arc::new(thalamus::LlmIntentFallback::new(llm.clone())));

        let processor = Self {
            config,
            classifier,
            importance: amygdala::ImportanceScorer::with_llm(llm.clone()),
            episodic,
            semantic,
            embedder,
            embedding_dim,
            recall_engine,
            llm,
            context_assembler: cortex::context::ContextAssembler::with_defaults(),
            procedures,
            events_tx,
            notification_router: None,
            action_dispatcher: None,
            metrics: Arc::new(brain_core::metrics::SubsystemMetrics::new()),
        };

        // Warm up the LLM model in the background to avoid first-call timeout
        // (cold Ollama starts can exceed the 15s classification timeout while
        // loading weights into VRAM).
        let warmup_llm = processor.llm.clone();
        tokio::spawn(async move {
            let warmup = vec![cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: "hi".to_string(),
            }];
            match warmup_llm.generate(&warmup).await {
                Ok(_) => tracing::info!("LLM warm-up complete"),
                Err(e) => tracing::debug!("LLM warm-up skipped (provider unavailable): {e}"),
            }
        });

        Ok(processor)
    }
}
