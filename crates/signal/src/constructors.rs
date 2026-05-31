//! SignalProcessor constructors and initialization.

use std::sync::Arc;

use crate::types::*;
use crate::SignalProcessor;

/// Resolve the LLM API key from config, with env var fallback for backwards compatibility.
///
/// An empty/whitespace-only `BRAIN_LLM__API_KEY` is treated as a user error
/// (e.g. unset-but-exported in a shell profile) and reported as a clear
/// init-time failure rather than silently falling back to an empty key.
fn resolve_llm_api_key(config: &brain::BrainConfig) -> Result<String, SignalError> {
    #[allow(deprecated)]
    let from_config = config.llm.api_key.trim().to_string();
    if !from_config.is_empty() {
        return Ok(from_config);
    }
    match std::env::var("BRAIN_LLM__API_KEY") {
        Ok(v) if v.trim().is_empty() => Err(SignalError::Init(
            "BRAIN_LLM__API_KEY is set but empty — unset it or provide a real key".to_string(),
        )),
        Ok(v) => Ok(v),
        Err(_) => Ok(String::new()),
    }
}

impl SignalProcessor {
    /// Initialize the signal processor, wiring all Brain subsystems.
    ///
    /// Opens the SQLite database, connects to RuVector, creates the LLM provider,
    /// and wires the intent classifier, importance scorer, and context assembler.
    pub async fn new(config: brain::BrainConfig) -> Result<Self, SignalError> {
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
        config: brain::BrainConfig,
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

        // Create the learned capability-fitness store (cerebellum) alongside it.
        let fitness_cfg = &config.learning.capability_fitness;
        let fitness = cerebellum::CapabilityFitnessStore::new(
            db.clone(),
            fitness_cfg.enabled,
            fitness_cfg.half_life_hours(),
        );
        if let Err(e) = fitness.ensure_tables() {
            tracing::warn!("CapabilityFitnessStore table init failed (non-fatal): {e}");
        }

        // Probe configured providers (multi-entry if `llm.providers` is set;
        // otherwise synthesised from the legacy single-provider fields) and
        // select the first reachable one. Env-var API key stays backfilled
        // onto the legacy field for single-provider configs.
        let llm_api_key = resolve_llm_api_key(&config)?;
        let mut llm_cfg = config.llm.clone();
        if llm_cfg.providers.is_empty() {
            #[allow(deprecated)]
            {
                llm_cfg.api_key = llm_api_key.clone();
            }
        }
        let llm: Arc<dyn cortex::LlmProvider> = Arc::new(
            cortex::llm::build_failover_chain(&llm_cfg)
                .await
                .map_err(|e| SignalError::Init(format!("Failed to create LLM provider: {e}")))?,
        );

        // Create embedder — the embedding transport key still comes from
        // the legacy `llm.provider` field (Issue 40 keeps it #[deprecated]
        // but load-bearing for embedder selection). The cortex LLM
        // provider is picked by `providers[]` when set; the embedder API
        // hasn't followed yet, so this single read is intentional.
        let embedding_dim = config.embedding.dimensions as usize;
        #[allow(deprecated)]
        let embed_provider = config.llm.provider.clone();
        #[allow(deprecated)]
        let embed_base = config.llm.base_url.clone();
        tracing::info!(
            provider = %embed_provider,
            model = config.embedding.model,
            dim = embedding_dim,
            "Embedding provider selected"
        );
        let embedder = hippocampus::Embedder::from_config(
            &embed_provider,
            &embed_base,
            &config.embedding.model,
            &llm_api_key,
        )
        .map_err(|e| SignalError::Init(format!("Failed to create embedding provider: {e}")))?
        .map(std::sync::Arc::new);

        // Create semantic store (optional — fails gracefully if RuVector unavailable).
        // Pass the probed embedding_dim so VectorDB is sized to match the provider.
        // Note: ruvector-core stores only IDs; content encryption is handled by SQLite.
        // Issue 37: HNSW tuning comes from `storage.hnsw` in config
        // instead of the hardcoded values that used to live inside the
        // storage crate. Convert the user-facing `brain::HnswConfig`
        // to the storage-crate type at the boundary.
        let hnsw = storage::HnswConfig {
            m: config.storage.hnsw.m,
            ef_construction: config.storage.hnsw.ef_construction,
            ef_search: config.storage.hnsw.ef_search,
            max_elements: config.storage.hnsw.max_elements,
        };
        let semantic = match storage::RuVectorStore::open_with_config(
            &config.ruvector_path(),
            embedding_dim,
            hnsw,
        )
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
        let classifier = thalamus::IntentClassifier::new()
            .with_llm_fallback(Arc::new(thalamus::LlmIntentFallback::new(llm.clone())));

        let proactivity_enabled = Arc::new(std::sync::atomic::AtomicBool::new(
            config.proactivity.enabled,
        ));
        // Capture before `config` is moved into the struct below — the prompt
        // assembler's budget scales to the model's real context window.
        let context_window = config.llm.context_window;
        let processor = Self {
            config,
            classifier,
            importance: amygdala::ImportanceScorer::with_llm(llm.clone()),
            episodic,
            semantic,
            embedder,
            embedding_dim,
            embedding_cache: std::sync::Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(256).expect("256 is non-zero"),
            )),
            recall_engine,
            llm,
            context_assembler: cortex::context::ContextAssembler::new(
                cortex::context::TokenBudget::for_context_size(context_window),
            ),
            history_summary_cache: std::sync::Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(64).expect("64 is non-zero"),
            )),
            procedures,
            fitness,
            metrics: Arc::new(brain::metrics::SubsystemMetrics::new()),
            proactivity_enabled,

            // Opt-in bundles — all builder-wired
            safety: crate::bundles::SafetyBundle::default(),
            channels: crate::bundles::ChannelBundle::default(),
            capability: crate::bundles::CapabilityBundle::default(),
            observability: crate::bundles::ObservabilityBundle::new(),

            // Top-level optionals
            dual_memory_reader: None,
            orchestrator: None,
            agent_registry: None,
            identity_store: None,
            client_rate_limits: None,
            product_self_model: None,
        };

        // Warm up the LLM model in the background to avoid first-call timeout
        // (cold Ollama starts can exceed the 15s classification timeout while
        // loading weights into VRAM).
        //
        // Fire-and-forget by design — the daemon must boot whether or not
        // the warm-up succeeds. We still want panics surfaced rather than
        // silently dropped, so the inner spawn awaits the handle and logs
        // anything other than a clean exit.
        let warmup_llm = processor.llm.clone();
        let warmup_handle = tokio::spawn(async move {
            let warmup = vec![cortex::llm::Message::user("hi")];
            match warmup_llm.generate(&warmup).await {
                Ok(_) => tracing::info!("LLM warm-up complete"),
                Err(e) => tracing::debug!("LLM warm-up skipped (provider unavailable): {e}"),
            }
        });
        tokio::spawn(async move {
            if let Err(e) = warmup_handle.await {
                if e.is_panic() {
                    tracing::error!("LLM warm-up task panicked: {e}");
                }
            }
        });

        Ok(processor)
    }
}
