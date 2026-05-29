//! Recall helpers for SignalProcessor — embedding, BM25 conversion, hybrid recall.

use crate::SignalProcessor;
use std::hash::{Hash, Hasher};

fn embedding_cache_key(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

impl SignalProcessor {
    /// Generate a vector embedding for text.
    ///
    /// Uses whichever provider was selected at startup (Ollama or OpenAI-compatible).
    /// Falls back to a deterministic, non-zero normalized vector if no provider
    /// is available or if the call fails. Successful embeddings flow through
    /// an LRU cache keyed by text hash so repeated recall/chat queries don't
    /// re-hit the provider.
    pub(super) async fn embed_text(&self, text: &str) -> Vec<f32> {
        self.metrics.inc_embedding_request();
        let key = embedding_cache_key(text);
        if let Some(cached) = self.embedding_cache.lock().unwrap().get(&key).cloned() {
            return (*cached).clone();
        }
        let (vector, cacheable) = match &self.embedder {
            Some(embedder) => match embedder.embed(text).await {
                Ok(vec) => (
                    hippocampus::embedding::sanitize_embedding(vec, self.embedding_dim, text),
                    true,
                ),
                Err(e) => {
                    tracing::warn!("Embedding failed, using deterministic fallback vector: {e}");
                    self.metrics.inc_embedding_fallback();
                    (
                        hippocampus::embedding::deterministic_fallback_embedding(
                            text,
                            self.embedding_dim,
                        ),
                        false,
                    )
                }
            },
            None => {
                self.metrics.inc_embedding_fallback();
                (
                    hippocampus::embedding::deterministic_fallback_embedding(
                        text,
                        self.embedding_dim,
                    ),
                    false,
                )
            }
        };
        if cacheable {
            let shared = std::sync::Arc::new(vector.clone());
            self.embedding_cache.lock().unwrap().put(key, shared);
        }
        vector
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
    /// If the semantic store is unavailable or fails, falls back to BM25-only
    /// episodic search. If BM25 also fails, returns the storage error rather
    /// than masking it as an empty result set — callers downgrade the request
    /// rather than silently serving stale context.
    pub(super) async fn do_recall(
        &self,
        query: &str,
        query_vector: Vec<f32>,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<(Vec<hippocampus::Memory>, usize, usize), crate::SignalError> {
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
                    self.dual_memory_reader.as_ref(),
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
                    return Ok((memories, facts_used, episodes_used));
                }
                Err(e) => {
                    tracing::warn!(
                        "Recall engine failed, falling back to BM25-only episodic search: {e}"
                    );
                }
            }
        }
        // Either semantic store is unavailable or hybrid recall failed.
        let bm25 = self
            .episodic
            .search_bm25(query, top_k, namespace, None)
            .map_err(|e| crate::SignalError::Storage(e.to_string()))?;
        let episodes_used = bm25.len();
        let memories = Self::bm25_to_memories(bm25);
        Ok((memories, 0, episodes_used))
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

    /// Quick existence probe — no rows materialized, no decryption. Used
    /// by chat / recall to decide whether to attach the onboarding
    /// addendum. Returns `true` when there are no facts *and* no episodes
    /// for `namespace`; storage errors fail closed (treat as non-empty so
    /// we don't show a misleading onboarding hint on a transient DB blip).
    pub fn namespace_is_empty(&self, namespace: &str) -> bool {
        let has_facts = match &self.semantic {
            Some(semantic) => semantic
                .has_facts_in_namespace(Some(namespace))
                .unwrap_or(true),
            None => false,
        };
        if has_facts {
            return false;
        }
        let has_episodes = self
            .episodic
            .has_episodes_in_namespace(Some(namespace))
            .unwrap_or(true);
        !has_episodes
    }

    /// List all active semantic facts (non-superseded), optionally scoped to a namespace.
    ///
    /// **Unbounded.** Kept for callers (memory summary, gRPC `get_facts`)
    /// that need the full set. New HTTP surface should prefer
    /// [`list_facts_paginated`] so a multi-thousand-fact store doesn't
    /// emit a single mega-response.
    pub fn list_facts(&self, namespace: Option<&str>) -> Vec<hippocampus::Fact> {
        if let Some(semantic) = &self.semantic {
            semantic.list_by_namespace(namespace).unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Paginated fact listing. `limit = None` matches the unbounded
    /// [`list_facts`] behavior; `Some(n)` translates to a SQL
    /// `LIMIT n OFFSET offset` at the storage layer.
    pub fn list_facts_paginated(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
        offset: usize,
    ) -> Vec<hippocampus::Fact> {
        if let Some(semantic) = &self.semantic {
            semantic
                .list_by_namespace_paginated(namespace, limit, offset)
                .unwrap_or_default()
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
                    ..Default::default()
                })
            })
            .collect()
    }
}
