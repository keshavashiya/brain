//! Recall helpers for SignalProcessor — embedding, BM25 conversion, hybrid recall.

use crate::SignalProcessor;

impl SignalProcessor {
    /// Generate a vector embedding for text.
    ///
    /// Uses whichever provider was selected at startup (Ollama or OpenAI-compatible).
    /// Falls back to a deterministic, non-zero normalized vector if no provider
    /// is available or if the call fails.
    pub(super) async fn embed_text(&self, text: &str) -> Vec<f32> {
        self.metrics.inc_embedding_request();
        let mut guard = self.embedder.lock().await;
        match &mut *guard {
            Some(embedder) => match embedder.embed(text).await {
                Ok(vec) => {
                    hippocampus::embedding::sanitize_embedding(vec, self.embedding_dim, text)
                }
                Err(e) => {
                    tracing::warn!("Embedding failed, using deterministic fallback vector: {e}");
                    self.metrics.inc_embedding_fallback();
                    hippocampus::embedding::deterministic_fallback_embedding(
                        text,
                        self.embedding_dim,
                    )
                }
            },
            None => {
                self.metrics.inc_embedding_fallback();
                hippocampus::embedding::deterministic_fallback_embedding(text, self.embedding_dim)
            }
        }
    }

    /// Convert BM25 search results to Memory objects.
    pub(super) fn bm25_to_memories(
        results: Vec<hippocampus::episodic::FtsResult>,
    ) -> Vec<hippocampus::Memory> {
        results
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
            .collect()
    }

    /// Run hybrid recall (BM25 + ANN via RecallEngine) and return memories with counts.
    ///
    /// If the semantic store is unavailable, falls back to BM25-only episodic search.
    pub(super) async fn do_recall(
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
                    let memories = Self::bm25_to_memories(bm25);
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
            let memories = Self::bm25_to_memories(bm25);
            (memories, 0, episodes_used)
        }
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

    /// Load the last `limit` episodes for a session as LLM messages, in
    /// chronological order. Used to give the LLM continuity across turns.
    pub(crate) fn load_session_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Vec<cortex::llm::Message> {
        let episodes = match self.episodic.get_session_history(session_id, limit) {
            Ok(eps) => eps,
            Err(e) => {
                tracing::debug!(session_id, "session history unavailable: {e}");
                return Vec::new();
            }
        };
        episodes
            .into_iter()
            .filter_map(|ep| {
                let role = match ep.role.as_str() {
                    "user" => cortex::llm::Role::User,
                    "assistant" => cortex::llm::Role::Assistant,
                    "system" => cortex::llm::Role::System,
                    _ => return None,
                };
                Some(cortex::llm::Message {
                    role,
                    content: ep.content,
                })
            })
            .collect()
    }
}
