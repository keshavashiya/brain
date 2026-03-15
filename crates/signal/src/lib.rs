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
//! - NotificationRouter (proactive delivery)

pub mod notification;

use std::{collections::HashMap, sync::Arc};

use chrono::{DateTime, Utc};
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
    /// Memory namespace for this signal (default: "personal").
    #[serde(default = "default_namespace")]
    pub namespace: String,
    /// Originating AI agent (e.g. "claude-code", "opencode"). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

fn default_namespace() -> String {
    "personal".to_string()
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
            namespace: "personal".to_string(),
            agent: None,
        }
    }

    /// Builder: set the originating agent identity.
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    /// Builder: set the memory namespace.
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Builder: set the metadata map.
    pub fn with_metadata(mut self, meta: HashMap<String, String>) -> Self {
        self.metadata = meta;
        self
    }

    /// Builder: set namespace from an Option (no-op if None).
    pub fn with_namespace_opt(mut self, ns: Option<String>) -> Self {
        if let Some(n) = ns {
            self.namespace = n;
        }
        self
    }

    /// Builder: set agent from an Option (no-op if None).
    pub fn with_agent_opt(mut self, agent: Option<String>) -> Self {
        if let Some(a) = agent {
            self.agent = Some(a);
        }
        self
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

/// Broadcast event emitted after a signal has been processed successfully.
#[derive(Debug, Clone)]
pub struct SignalProcessedEvent {
    pub signal_id: Uuid,
    pub source: SignalSource,
    pub channel: String,
    pub sender: String,
    pub namespace: String,
    pub status: ResponseStatus,
    pub response: String,
    pub facts_used: usize,
    pub episodes_used: usize,
    pub timestamp: DateTime<Utc>,
}

// ─── Pipeline Result ─────────────────────────────────────────────────────────

/// Result of the `prepare()` pipeline phase.
///
/// Either the intent was handled directly (StoreFact, Forget, SystemStatus, Actions)
/// and a complete response is returned, or the pipeline assembled LLM messages
/// and the caller decides whether to use streaming or batch generation.
pub enum PipelineResult {
    /// Intent handled directly. Response is complete.
    Complete(SignalResponse),
    /// Chat/Recall: pipeline done, LLM messages assembled.
    /// Caller chooses streaming vs batch generation.
    LlmReady {
        signal_id: Uuid,
        messages: Vec<cortex::llm::Message>,
        memory_context: MemoryContext,
        session_id: Option<String>,
        user_content: String,
        namespace: String,
        agent: Option<String>,
    },
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
    embedder: tokio::sync::Mutex<Option<hippocampus::Embedder>>,
    /// Actual output dimension of the active embedding provider (probed at startup).
    embedding_dim: usize,
    recall_engine: hippocampus::RecallEngine,
    llm: Arc<dyn cortex::LlmProvider>,
    context_assembler: cortex::context::ContextAssembler,
    procedures: cerebellum::ProcedureStore,
    events_tx: tokio::sync::broadcast::Sender<SignalProcessedEvent>,
    /// Notification router for proactive message delivery (set via builder).
    notification_router: Option<notification::NotificationRouter>,
    /// Action dispatcher for executing tool intents (set via builder).
    action_dispatcher: Option<cortex::actions::ActionDispatcher>,
}

/// Resolve the LLM API key from config, with env var fallback for backwards compatibility.
fn resolve_llm_api_key(config: &brain_core::BrainConfig) -> String {
    let from_config = config.llm.api_key.trim().to_string();
    if !from_config.is_empty() {
        return from_config;
    }
    std::env::var("BRAIN_LLM__API_KEY").unwrap_or_default()
}

/// Extract text content from a ResponseContent variant.
pub fn response_to_text(content: &ResponseContent) -> String {
    match content {
        ResponseContent::Text(t) => t.clone(),
        ResponseContent::Json(v) => v.to_string(),
        ResponseContent::Error(e) => e.clone(),
    }
}

impl SignalProcessor {
    /// Initialize the signal processor, wiring all Brain subsystems.
    ///
    /// Opens the SQLite database, connects to RuVector, creates the LLM provider,
    /// and wires the intent classifier, importance scorer, and context assembler.
    pub async fn new(config: brain_core::BrainConfig) -> Result<Self, SignalError> {
        Self::new_with_encryptor(config, None).await
    }

    /// Like `new`, but wires an `Encryptor` into all storage backends.
    ///
    /// When provided, `episodes.content` and `semantic_facts.object` are
    /// AES-256-GCM encrypted at rest; RuVector `content` fields are also
    /// encrypted in their JSON files on disk.
    pub async fn new_with_encryptor(
        config: brain_core::BrainConfig,
        encryptor: Option<storage::Encryptor>,
    ) -> Result<Self, SignalError> {
        // Open SQLite pool — attach encryptor if provided
        let db = {
            let pool = storage::SqlitePool::open(&config.sqlite_path())
                .map_err(|e| SignalError::Init(format!("SQLite: {e}")))?;
            if let Some(enc) = encryptor.clone() {
                tracing::info!("Encryption enabled: SQLite content columns will be encrypted");
                pool.with_encryptor(enc)
            } else {
                pool
            }
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
        let llm: Arc<dyn cortex::LlmProvider> = cortex::llm::create_provider(&llm_config).into();

        // Create embedder — provider is selected from llm.provider config.
        // The model and dimension come from the embedding config section.
        // embedding.dimensions MUST match the model's actual output size.
        let embedding_dim = config.embedding.dimensions as usize;
        let embedder_inner = match config.llm.provider.as_str() {
            "openai" => {
                tracing::info!(
                    model = config.embedding.model,
                    dim = embedding_dim,
                    "Embedding provider: OpenAI-compatible"
                );
                Some(hippocampus::Embedder::for_openai(
                    &config.llm.base_url,
                    &config.embedding.model,
                    &llm_api_key,
                ))
            }
            _ => {
                // Default: Ollama (covers "ollama" and any custom provider name)
                tracing::info!(
                    model = config.embedding.model,
                    dim = embedding_dim,
                    "Embedding provider: Ollama"
                );
                Some(hippocampus::Embedder::for_ollama(
                    &config.llm.base_url,
                    &config.embedding.model,
                ))
            }
        };
        let embedder = tokio::sync::Mutex::new(embedder_inner);

        // Create semantic store (optional — fails gracefully if RuVector unavailable).
        // Pass the probed embedding_dim so VectorDB is sized to match the provider.
        // Note: ruvector-core stores only IDs; content encryption is handled by SQLite.
        let semantic = match storage::RuVectorStore::open(&config.ruvector_path(), embedding_dim)
            .await
        {
            Ok(ruv) => {
                if encryptor.is_some() {
                    tracing::info!("Encryption enabled: vector IDs stored in ruvector-core, content encrypted in SQLite");
                }
                match ruv.ensure_tables().await {
                    Ok(()) => Some(hippocampus::SemanticStore::new(db.clone(), ruv)),
                    Err(e) => {
                        tracing::warn!(
                            "RuVector table initialization failed, semantic memory disabled: {e}"
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

        // Create recall engine with default RRF config
        let recall_engine = hippocampus::RecallEngine::with_defaults();
        let (events_tx, _) = tokio::sync::broadcast::channel(512);

        let classifier = thalamus::IntentClassifier::new()
            .with_llm_fallback(Arc::new(thalamus::LlmIntentFallback::new(llm.clone())));

        Ok(Self {
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
        })
    }

    /// Process a signal through the full Brain pipeline.
    ///
    /// Delegates to `prepare()` for pipeline work, then handles LLM generation
    /// for intents that require it (Chat, Recall).
    ///
    /// Routes by intent:
    /// - `StoreFact`  → Amygdala importance → Hippocampus semantic store → confirmation
    /// - `Recall`     → Hippocampus hybrid search → Cortex context assembly → LLM response
    /// - `Chat`       → Hippocampus context → Cortex LLM → Hippocampus episode store
    /// - `Forget`     → search + delete matching facts
    /// - `SystemStatus` → memory counts
    /// - Action intents → ActionDispatcher
    #[tracing::instrument(
        name = "signal.process",
        skip(self, signal),
        fields(
            signal_id = %signal.id,
            source = ?signal.source,
            namespace = %signal.namespace
        )
    )]
    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError> {
        match self.prepare(&signal, None).await? {
            PipelineResult::Complete(resp) => {
                self.publish_event(&signal, &resp);
                Ok(resp)
            }
            PipelineResult::LlmReady {
                signal_id,
                messages,
                memory_context,
                session_id,
                namespace,
                agent,
                ..
            } => {
                let llm_resp = self.llm.generate(&messages).await?;

                // Store assistant episode for Chat/Recall
                if let Some(sid) = &session_id {
                    self.episodic
                        .store_episode(
                            sid,
                            "assistant",
                            &llm_resp.content,
                            0.5,
                            Some(&namespace),
                            agent.as_deref(),
                        )
                        .map_err(|e| SignalError::Storage(e.to_string()))?;
                }

                let resp = SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(llm_resp.content),
                    memory_context,
                };
                self.publish_event(&signal, &resp);
                Ok(resp)
            }
        }
    }

    /// Prepare the pipeline up to (but not including) LLM generation.
    ///
    /// Returns either a complete response (for StoreFact, Forget, SystemStatus,
    /// Actions) or assembled LLM messages (for Chat, Recall). The caller can
    /// then choose streaming vs batch LLM generation.
    ///
    /// If `conversation_history` is provided, it is used instead of an empty
    /// history when assembling context (useful for CLI which manages its own).
    pub async fn prepare(
        &self,
        signal: &Signal,
        conversation_history: Option<&[cortex::llm::Message]>,
    ) -> Result<PipelineResult, SignalError> {
        let signal_id = signal.id;

        // 0. Drain any pending proactive notifications from the outbox
        let pending_notifications = if let Some(router) = &self.notification_router {
            router.drain_pending(10)
        } else {
            Vec::new()
        };

        // 1. Score importance via Amygdala (keyword heuristic — sync so the LLM
        //    slot stays free for classification which extracts facts)
        let importance = self.importance.score(&signal.content);

        // 2. Classify intent via Thalamus
        let classification = self.classifier.classify(&signal.content).await;

        tracing::info!(
            signal_id = %signal_id,
            source = ?signal.source,
            intent = ?classification.intent,
            importance = importance,
            method = ?classification.method,
            extracted_facts = classification.extracted_facts.len(),
            "Signal classified"
        );

        // ── Store any facts extracted during classification ───────────────────
        if !classification.extracted_facts.is_empty() {
            for fact in &classification.extracted_facts {
                match self
                    .store_fact_direct(
                        &signal.namespace,
                        "extracted",
                        &fact.subject,
                        &fact.predicate,
                        &fact.object,
                        signal.agent.as_deref(),
                    )
                    .await
                {
                    Ok(_) => {
                        tracing::info!(
                            "Extracted fact: {} {} {}",
                            fact.subject,
                            fact.predicate,
                            fact.object
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to store extracted fact: {e}");
                    }
                }
            }
        }

        // ── Cerebellum: match stored procedures ───────────────────────────────
        let procedure_context: Vec<String> = match self.procedures.match_trigger(&signal.content) {
            Ok(procs) if !procs.is_empty() => {
                tracing::debug!(
                    count = procs.len(),
                    "Procedure(s) matched — injecting steps into context"
                );
                let mut steps: Vec<String> = Vec::new();
                for proc in &procs {
                    let _ = self.procedures.record_execution(&proc.id);
                    steps.extend(proc.steps.clone());
                }
                steps
            }
            Ok(_) => Vec::new(),
            Err(e) => {
                tracing::warn!("Procedure match failed (non-fatal): {e}");
                Vec::new()
            }
        };

        // Helper: prepend notification nudges to a response
        let prepend_nudges = |mut resp: SignalResponse| -> SignalResponse {
            if !pending_notifications.is_empty() {
                let nudge_text: String = pending_notifications
                    .iter()
                    .map(|n| format!("[nudge] {}", n.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                if let ResponseContent::Text(ref text) = resp.response {
                    resp.response = ResponseContent::Text(format!("{nudge_text}\n\n{text}"));
                }
            }
            resp
        };

        match classification.intent {
            // ── STORE_FACT ──
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
                            &signal.namespace,
                            "signal",
                            &subject,
                            &predicate,
                            &object,
                            importance as f64,
                            None,
                            vector,
                            signal.agent.as_deref(),
                        )
                        .await
                    {
                        Ok(_) => facts_stored = 1,
                        Err(e) => tracing::warn!("Failed to store fact in semantic memory: {e}"),
                    }
                }

                let resp = prepend_nudges(SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(format!(
                        "Stored: {subject} {predicate} {object} (importance: {importance:.2})"
                    )),
                    memory_context: MemoryContext {
                        facts_used: facts_stored,
                        episodes_used: 0,
                    },
                });
                Ok(PipelineResult::Complete(resp))
            }

            // ── RECALL ──
            thalamus::Intent::Recall { query } => {
                let query_vector = self.embed_text(&query).await;
                let (memories, facts_used, episodes_used) = self
                    .do_recall(&query, query_vector, 10, Some(&signal.namespace))
                    .await;

                // Agent callers get structured data
                if signal.agent.is_some() {
                    let text = if memories.is_empty() {
                        "No relevant memories found.".to_string()
                    } else {
                        memories
                            .iter()
                            .map(|m| format!("[{:?}] {}", m.source, m.content))
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    let resp = prepend_nudges(SignalResponse {
                        signal_id,
                        status: ResponseStatus::Ok,
                        response: ResponseContent::Text(text),
                        memory_context: MemoryContext {
                            facts_used,
                            episodes_used,
                        },
                    });
                    return Ok(PipelineResult::Complete(resp));
                }

                // Human callers: assemble LLM messages
                let proc_history: Vec<cortex::llm::Message> = procedure_context
                    .iter()
                    .map(|step| cortex::llm::Message {
                        role: cortex::llm::Role::User,
                        content: format!("[procedure step] {step}"),
                    })
                    .collect();

                let history = conversation_history.unwrap_or(&proc_history);
                let messages = self
                    .context_assembler
                    .assemble(&query, &memories, history);

                Ok(PipelineResult::LlmReady {
                    signal_id,
                    messages,
                    memory_context: MemoryContext {
                        facts_used,
                        episodes_used,
                    },
                    session_id: None,
                    user_content: query,
                    namespace: signal.namespace.clone(),
                    agent: signal.agent.clone(),
                })
            }

            // ── CHAT ──
            thalamus::Intent::Chat { content } => {
                let query_vector = self.embed_text(&content).await;
                let (memories, facts_used, episodes_used) = self
                    .do_recall(&content, query_vector, 10, Some(&signal.namespace))
                    .await;

                // Create session and store the user turn
                let session_id = self
                    .episodic
                    .create_session(&signal.channel)
                    .map_err(|e| SignalError::Storage(e.to_string()))?;

                self.episodic
                    .store_episode(
                        &session_id,
                        "user",
                        &signal.content,
                        importance as f64,
                        Some(&signal.namespace),
                        signal.agent.as_deref(),
                    )
                    .map_err(|e| SignalError::Storage(e.to_string()))?;

                // Agent callers get structured memory context
                if signal.agent.is_some() {
                    let response_text = if memories.is_empty() {
                        format!("Stored episode. No relevant memories found for: {}", content)
                    } else {
                        let mem_lines: String = memories
                            .iter()
                            .map(|m| format!("[{:?}] {}", m.source, m.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!(
                            "Stored episode. Relevant memories:\n{}",
                            mem_lines
                        )
                    };

                    let resp = prepend_nudges(SignalResponse {
                        signal_id,
                        status: ResponseStatus::Ok,
                        response: ResponseContent::Text(response_text),
                        memory_context: MemoryContext {
                            facts_used,
                            episodes_used,
                        },
                    });
                    return Ok(PipelineResult::Complete(resp));
                }

                // Human callers: assemble LLM messages
                let proc_history: Vec<cortex::llm::Message> = procedure_context
                    .iter()
                    .map(|step| cortex::llm::Message {
                        role: cortex::llm::Role::User,
                        content: format!("[procedure step] {step}"),
                    })
                    .collect();

                let history = conversation_history.unwrap_or(&proc_history);
                let messages = self
                    .context_assembler
                    .assemble(&content, &memories, history);

                Ok(PipelineResult::LlmReady {
                    signal_id,
                    messages,
                    memory_context: MemoryContext {
                        facts_used,
                        episodes_used,
                    },
                    session_id: Some(session_id),
                    user_content: content,
                    namespace: signal.namespace.clone(),
                    agent: signal.agent.clone(),
                })
            }

            // ── FORGET ──
            thalamus::Intent::Forget { target } => {
                let mut deleted_count = 0usize;

                if let Some(semantic) = &self.semantic {
                    match semantic.find_facts_matching(&target, Some(&signal.namespace)) {
                        Ok(facts) if !facts.is_empty() => {
                            for fact in &facts {
                                if let Err(e) = semantic.delete_fact(&fact.id).await {
                                    tracing::warn!(
                                        fact_id = %fact.id,
                                        "Failed to delete fact: {e}"
                                    );
                                } else {
                                    deleted_count += 1;
                                }
                            }
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("Forget search failed: {e}");
                        }
                    }
                }

                let message = if deleted_count > 0 {
                    format!(
                        "Memory erased: removed {deleted_count} engram(s) matching \"{target}\""
                    )
                } else {
                    format!("No engrams found matching \"{target}\" to erase")
                };

                let resp = prepend_nudges(SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(message),
                    memory_context: MemoryContext {
                        facts_used: 0,
                        episodes_used: 0,
                    },
                });
                Ok(PipelineResult::Complete(resp))
            }

            // ── SystemStatus ──
            thalamus::Intent::SystemStatus => {
                let semantic_count = self
                    .semantic
                    .as_ref()
                    .and_then(|s| s.count().ok())
                    .unwrap_or(0);
                let episode_count = self.episodic.count().unwrap_or(0);

                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!("Brain status: {semantic_count} facts, {episode_count} episodes"),
                ));
                Ok(PipelineResult::Complete(resp))
            }

            // ── Action intents ──
            ref intent @ (thalamus::Intent::WebSearch { .. }
            | thalamus::Intent::Schedule { .. }
            | thalamus::Intent::SendMessage { .. }
            | thalamus::Intent::ExecuteCommand { .. }) => {
                let router = thalamus::SignalRouter::new();
                let resp = match (router.intent_to_action(intent), &self.action_dispatcher) {
                    (Some(action), Some(dispatcher)) => {
                        let result = dispatcher.dispatch(&action).await;
                        if result.success {
                            if matches!(&action, cortex::actions::Action::WebSearch { .. })
                                && !result.output.is_empty()
                            {
                                let search_context = format!(
                                    "The user asked: \"{}\"\n\nHere are web search results:\n{}\n\nUsing these search results, provide a helpful and concise answer to the user's question. Cite sources when relevant.",
                                    signal.content, result.output
                                );
                                let messages = vec![
                                    cortex::llm::Message {
                                        role: cortex::llm::Role::System,
                                        content: "You are Brain OS. Answer the user's question using the provided web search results. Be concise and cite your sources.".to_string(),
                                    },
                                    cortex::llm::Message {
                                        role: cortex::llm::Role::User,
                                        content: search_context,
                                    },
                                ];
                                match self.llm.generate(&messages).await {
                                    Ok(llm_response) => SignalResponse::ok(signal_id, llm_response.content),
                                    Err(_) => SignalResponse::ok(signal_id, result.output),
                                }
                            } else {
                                SignalResponse::ok(signal_id, result.output)
                            }
                        } else {
                            SignalResponse::error(
                                signal_id,
                                result.error.unwrap_or_else(|| "Action failed".to_string()),
                            )
                        }
                    }
                    (Some(_action), None) => SignalResponse::error(
                        signal_id,
                        format!(
                            "Action {:?} recognized but no dispatcher configured — \
                             enable the relevant backend in config",
                            intent
                        ),
                    ),
                    (None, _) => SignalResponse::ok(
                        signal_id,
                        format!("Intent classified: {:?}", intent),
                    ),
                };
                let resp = prepend_nudges(resp);
                Ok(PipelineResult::Complete(resp))
            }
        }
    }

    /// Store the assistant response in episodic memory after streaming completes.
    ///
    /// Call this after streaming LLM generation finishes to persist the
    /// assistant turn in episodic memory. The `session_id` comes from the
    /// `PipelineResult::LlmReady` variant.
    pub fn finalize_streaming(
        &self,
        session_id: &str,
        assistant_content: &str,
        namespace: &str,
        agent: Option<&str>,
    ) -> Result<(), SignalError> {
        self.episodic
            .store_episode(session_id, "assistant", assistant_content, 0.5, Some(namespace), agent)
            .map_err(|e| SignalError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Generate a vector embedding for text.
    ///
    /// Uses whichever provider was selected at startup (Ollama or OpenAI-compatible).
    /// Falls back to a deterministic, non-zero normalized vector if no provider
    /// is available or if the call fails.
    async fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut guard = self.embedder.lock().await;
        match &mut *guard {
            Some(embedder) => match embedder.embed(text).await {
                Ok(vec) => {
                    hippocampus::embedding::sanitize_embedding(vec, self.embedding_dim, text)
                }
                Err(e) => {
                    tracing::warn!("Embedding failed, using deterministic fallback vector: {e}");
                    hippocampus::embedding::deterministic_fallback_embedding(
                        text,
                        self.embedding_dim,
                    )
                }
            },
            None => {
                hippocampus::embedding::deterministic_fallback_embedding(text, self.embedding_dim)
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
        namespace: Option<&str>,
    ) -> (Vec<hippocampus::Memory>, usize, usize) {
        if let Some(semantic) = &self.semantic {
            match self
                .recall_engine
                .recall(
                    query,
                    query_vector,
                    &self.episodic,
                    semantic,
                    top_k,
                    namespace,
                    None,
                )
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
                    tracing::warn!(
                        "Recall engine failed, falling back to BM25-only episodic search: {e}"
                    );
                    let bm25 = self
                        .episodic
                        .search_bm25(query, top_k, namespace, None)
                        .unwrap_or_default();
                    let episodes_used = bm25.len();
                    let memories = bm25
                        .into_iter()
                        .map(|r| hippocampus::Memory {
                            id: r.episode_id,
                            content: r.content,
                            source: hippocampus::MemorySource::Episodic,
                            score: r.rank,
                            importance: 0.5,
                            timestamp: r.timestamp,
                            agent: r.agent,
                        })
                        .collect();
                    (memories, 0, episodes_used)
                }
            }
        } else {
            // Semantic store unavailable — fall back to episodic BM25 only
            let bm25 = self
                .episodic
                .search_bm25(query, top_k, namespace, None)
                .unwrap_or_default();
            let episodes_used = bm25.len();
            let memories = bm25
                .into_iter()
                .map(|r| hippocampus::Memory {
                    id: r.episode_id,
                    content: r.content,
                    source: hippocampus::MemorySource::Episodic,
                    score: r.rank,
                    importance: 0.5,
                    timestamp: r.timestamp,
                    agent: r.agent,
                })
                .collect();
            (memories, 0, episodes_used)
        }
    }

    fn publish_event(&self, signal: &Signal, response: &SignalResponse) {
        let event = SignalProcessedEvent {
            signal_id: response.signal_id,
            source: signal.source.clone(),
            channel: signal.channel.clone(),
            sender: signal.sender.clone(),
            namespace: signal.namespace.clone(),
            status: response.status.clone(),
            response: response_to_text(&response.response),
            facts_used: response.memory_context.facts_used,
            episodes_used: response.memory_context.episodes_used,
            timestamp: Utc::now(),
        };
        let _ = self.events_tx.send(event);
    }

    /// Subscribe to live signal-processing events.
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<SignalProcessedEvent> {
        self.events_tx.subscribe()
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
    pub fn llm(&self) -> &Arc<dyn cortex::LlmProvider> {
        &self.llm
    }

    /// Expose the context assembler (for adapter use).
    pub fn context_assembler(&self) -> &cortex::context::ContextAssembler {
        &self.context_assembler
    }

    /// Expose the embedding dimension (for adapter use).
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get a cloneable handle to the LLM provider (for adapter use).
    pub fn llm_arc(&self) -> Arc<dyn cortex::LlmProvider> {
        self.llm.clone()
    }

    /// Recall memories using hybrid search (BM25 + ANN), with embedding done internally.
    ///
    /// Returns merged and ranked memories. Falls back to BM25-only episodic search
    /// when the semantic store is unavailable.
    pub async fn recall_memories(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Vec<hippocampus::Memory> {
        let query_vector = self.embed_text(query).await;
        let (memories, _, _) = self.do_recall(query, query_vector, top_k, namespace).await;
        memories
    }

    /// Search semantic facts by text query (embed → vector ANN search).
    ///
    /// Returns up to `top_k` facts ranked by similarity. If `namespace` is
    /// provided, only facts in that namespace are returned. Falls back to an
    /// empty list if the semantic store is unavailable.
    pub async fn search_facts(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Vec<hippocampus::SemanticResult> {
        if let Some(semantic) = &self.semantic {
            let qv = self.embed_text(query).await;
            match semantic.search_similar(qv, top_k, namespace, None).await {
                Ok(results) => results,
                Err(e) => {
                    tracing::warn!("search_facts failed: {e}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    }

    /// List all active semantic facts (non-superseded), optionally scoped to a namespace.
    pub fn list_facts(&self, namespace: Option<&str>) -> Vec<hippocampus::Fact> {
        if let Some(semantic) = &self.semantic {
            semantic.list_by_namespace(namespace).unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get all facts about a specific subject.
    pub fn facts_about(&self, subject: &str, namespace: Option<&str>) -> Vec<hippocampus::Fact> {
        if let Some(semantic) = &self.semantic {
            semantic
                .get_facts_about_in_namespace(subject, namespace)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// List all namespaces with fact and episode counts.
    pub fn list_namespaces(&self) -> Vec<hippocampus::NamespaceStats> {
        if let Some(semantic) = &self.semantic {
            semantic.list_namespaces().unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get the most recent episodes across all sessions.
    pub fn recent_episodes(
        &self,
        limit: usize,
        namespace: Option<&str>,
    ) -> Vec<hippocampus::Episode> {
        self.episodic.recent(limit, namespace).unwrap_or_default()
    }

    /// Attach a notification router (builder pattern).
    pub fn with_notification_router(mut self, router: notification::NotificationRouter) -> Self {
        self.notification_router = Some(router);
        self
    }

    /// Expose the notification router.
    pub fn notification_router(&self) -> Option<&notification::NotificationRouter> {
        self.notification_router.as_ref()
    }

    /// Attach an action dispatcher for executing tool intents (builder pattern).
    pub fn with_action_dispatcher(mut self, dispatcher: cortex::actions::ActionDispatcher) -> Self {
        self.action_dispatcher = Some(dispatcher);
        self
    }

    /// Set the namespace used by the action dispatcher (if attached).
    ///
    /// Call this before `prepare()` when the active namespace changes
    /// (e.g. CLI session namespace switch).
    pub fn set_action_namespace(&mut self, ns: &str) {
        if let Some(d) = &mut self.action_dispatcher {
            d.set_namespace(ns);
        }
    }

    /// Flush all in-flight writes and checkpoint the SQLite WAL.
    ///
    /// Call this on graceful shutdown to ensure no committed data is lost.
    /// Safe to call from any async context; completes synchronously on the
    /// calling thread (WAL checkpoint is a fast O(WAL-size) operation).
    pub fn shutdown(&self) {
        if let Err(e) = self.episodic.pool().wal_checkpoint() {
            tracing::warn!("WAL checkpoint on shutdown failed: {e}");
        } else {
            tracing::info!("SQLite WAL checkpoint complete");
        }
    }

    /// Expose the procedure store (for adapter / MCP use).
    pub fn procedures(&self) -> &cerebellum::ProcedureStore {
        &self.procedures
    }

    /// Store a semantic fact directly (bypasses intent classification).
    ///
    /// Used by the MCP `memory_store` tool and extracted-fact storage.
    /// The `namespace` scopes the fact (default: "personal").
    /// Importance is scored via Amygdala rather than hardcoded.
    pub async fn store_fact_direct(
        &self,
        namespace: &str,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        agent: Option<&str>,
    ) -> Result<String, SignalError> {
        if let Some(semantic) = &self.semantic {
            let fact_text = format!("{subject} {predicate} {object}");
            let importance = self.importance.score(&fact_text);
            let vector = self.embed_text(&fact_text).await;
            let id = semantic
                .store_fact(
                    namespace, category, subject, predicate, object, importance as f64, None,
                    vector, agent,
                )
                .await
                .map_err(|e| SignalError::Storage(e.to_string()))?;
            Ok(id)
        } else {
            Err(SignalError::Storage(
                "Semantic store unavailable".to_string(),
            ))
        }
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
        let signal = Signal::new(SignalSource::Http, "http", "apiclient", "Remember coffee");
        let json = serde_json::to_string(&signal).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(signal.id, back.id);
        assert_eq!(signal.content, back.content);
    }

    #[test]
    fn test_signal_with_agent() {
        let signal =
            Signal::new(SignalSource::Http, "http", "apiclient", "hello").with_agent("claude-code");
        assert_eq!(signal.agent.as_deref(), Some("claude-code"));

        // Serialization round-trip preserves agent
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("claude-code"));
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agent.as_deref(), Some("claude-code"));
    }

    #[test]
    fn test_signal_without_agent_omits_field() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello");
        assert!(signal.agent.is_none());
        let json = serde_json::to_string(&signal).unwrap();
        // skip_serializing_if = "Option::is_none" should omit agent entirely
        assert!(!json.contains("agent"));
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
    /// embedding model (embedding gracefully degrades to deterministic fallback).
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

    /// Integration test: CLI signal → SignalProcessor → store fact → search fact → verify result.
    ///
    /// Tests the full round-trip: process() stores a fact via the StoreFact pipeline,
    /// list_facts() confirms persistence, and search_facts() confirms the fact is
    /// retrievable via the vector search API. Deterministic fallback vectors are
    /// used when embeddings are unavailable.
    #[tokio::test]
    async fn test_store_fact_then_search_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

        let processor = SignalProcessor::new(config).await.unwrap();

        // Store a fact via the CLI signal pipeline
        let signal = Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "Remember that Rust is fast",
        );
        let resp = processor.process(signal).await.unwrap();
        assert_eq!(resp.status, ResponseStatus::Ok);
        assert_eq!(
            resp.memory_context.facts_used, 1,
            "StoreFact should persist 1 fact"
        );

        // Verify persistence: list_facts returns the stored fact
        let facts = processor.list_facts(None);
        assert!(
            !facts.is_empty(),
            "Stored fact should appear in list_facts()"
        );

        // Verify search: search_facts returns results
        let results = processor
            .search_facts("Rust programming language", 5, None)
            .await;
        assert!(
            !results.is_empty(),
            "search_facts() should return the stored fact"
        );
    }

    #[tokio::test]
    async fn test_forget_is_namespace_scoped() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
        let processor = SignalProcessor::new(config).await.unwrap();

        processor
            .store_fact_direct("personal", "test", "project", "uses", "bun", None)
            .await
            .unwrap();
        processor
            .store_fact_direct("work", "test", "project", "uses", "bun", None)
            .await
            .unwrap();

        let mut forget_signal = Signal::new(SignalSource::Cli, "cli", "user", "forget bun");
        forget_signal.namespace = "work".to_string();
        let _ = processor.process(forget_signal).await.unwrap();

        let personal = processor.list_facts(Some("personal"));
        let work = processor.list_facts(Some("work"));
        assert_eq!(personal.len(), 1, "personal namespace fact should remain");
        assert_eq!(work.len(), 0, "work namespace fact should be deleted");
    }

    #[tokio::test]
    async fn test_store_fact_preserves_agent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();

        let processor = SignalProcessor::new(config).await.unwrap();

        // Store a fact with agent identity
        let signal = Signal::new(
            SignalSource::Http,
            "http",
            "apiclient",
            "Remember that Python is versatile",
        )
        .with_agent("open-code");

        let resp = processor.process(signal).await.unwrap();
        assert_eq!(resp.status, ResponseStatus::Ok);

        // Verify the agent is persisted on the fact
        let facts = processor.list_facts(None);
        assert!(!facts.is_empty());
        let fact = &facts[0];
        assert_eq!(fact.agent.as_deref(), Some("open-code"));
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
