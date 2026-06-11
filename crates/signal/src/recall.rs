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
    ///
    /// `namespace` is the namespace the text belongs to (fact storage) or
    /// the scope of the recall it serves (queries). When that namespace is
    /// `local_only` and the embedder is remote, the text never leaves the
    /// machine: the deterministic fallback vector is used instead — BM25
    /// still carries recall for those namespaces.
    pub(super) async fn embed_text(&self, text: &str, namespace: &str) -> Vec<f32> {
        self.metrics.inc_embedding_request();
        if let Some(embedder) = &self.memory.embedder {
            if !embedder.is_local() && self.config.memory.residency_of(namespace).is_local_only() {
                tracing::debug!(
                    namespace,
                    "residency: remote embedder skipped for local-only content"
                );
                self.metrics.inc_embedding_fallback();
                return hippocampus::embedding::deterministic_fallback_embedding(
                    text,
                    self.memory.embedding_dim,
                );
            }
        }
        let key = embedding_cache_key(text);
        if let Some(cached) = self
            .memory
            .embedding_cache
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
        {
            return (*cached).clone();
        }
        let (vector, cacheable) = match &self.memory.embedder {
            Some(embedder) => match embedder.embed(text).await {
                Ok(vec) => (
                    hippocampus::embedding::sanitize_embedding(
                        vec,
                        self.memory.embedding_dim,
                        text,
                    ),
                    true,
                ),
                Err(e) => {
                    tracing::warn!("Embedding failed, using deterministic fallback vector: {e}");
                    self.metrics.inc_embedding_fallback();
                    (
                        hippocampus::embedding::deterministic_fallback_embedding(
                            text,
                            self.memory.embedding_dim,
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
                        self.memory.embedding_dim,
                    ),
                    false,
                )
            }
        };
        if cacheable {
            let shared = std::sync::Arc::new(vector.clone());
            self.memory.embedding_cache.lock().unwrap().put(key, shared);
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
                namespace: Some(r.namespace),
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
        if let Some(semantic) = &self.memory.semantic {
            match self
                .memory
                .recall_engine
                .recall(
                    query,
                    query_vector,
                    &self.memory.episodic,
                    semantic,
                    top_k,
                    namespace,
                    None,
                    self.memory.dual_memory_reader.as_ref(),
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
            .memory
            .episodic
            .search_bm25(query, top_k, namespace, None)
            .map_err(|e| crate::SignalError::Storage(e.to_string()))?;
        let episodes_used = bm25.len();
        let mut memories = Self::bm25_to_memories(bm25);
        // The degraded path bypasses the recall engine's trust-weighted
        // scoring, and the raw BM25 rank semantics don't admit a clean
        // multiplicative term — so enforce trust on the *ordering*
        // instead: lower-trust memories sink below all higher-trust
        // ones, BM25 order preserved within each trust level. With the
        // default (identity) policy the stable sort is a no-op.
        let trust = self.config.memory.trust.policy();
        if !trust.is_noop() {
            memories.sort_by(|a, b| {
                trust
                    .trust_of(b.agent.as_deref())
                    .partial_cmp(&trust.trust_of(a.agent.as_deref()))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        Ok((memories, 0, episodes_used))
    }

    /// Enforce the namespace data-residency policy on recall results bound
    /// for the LLM: when the active chain can leave the machine, memories
    /// from `local_only` namespaces are withheld from the prompt. Items
    /// without a namespace (legacy conversions) inherit `scope_namespace`
    /// — the namespace the recall was scoped to — so nothing slips through
    /// unlabeled. Returns the kept memories and the withheld count.
    ///
    /// Agent-caller responses and user-facing renderings are *not* run
    /// through this filter: they stay on the machine.
    pub(crate) fn withhold_nonresident_memories(
        &self,
        memories: Vec<hippocampus::Memory>,
        scope_namespace: &str,
    ) -> (Vec<hippocampus::Memory>, usize) {
        if self.config.memory.namespaces.is_empty() || self.llm.is_local() {
            return (memories, 0);
        }
        let total = memories.len();
        let kept: Vec<hippocampus::Memory> = memories
            .into_iter()
            .filter(|m| {
                let ns = m.namespace.as_deref().unwrap_or(scope_namespace);
                !self.config.memory.residency_of(ns).is_local_only()
            })
            .collect();
        let withheld = total - kept.len();
        if withheld > 0 {
            tracing::info!(
                withheld,
                scope = scope_namespace,
                provider = self.llm.name(),
                "residency: local-only memories withheld from remote-bound prompt"
            );
        }
        (kept, withheld)
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
        if let Some(semantic) = &self.memory.semantic {
            let qv = self.embed_text(query, namespace.unwrap_or("")).await;
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
        let has_facts = match &self.memory.semantic {
            Some(semantic) => semantic
                .has_facts_in_namespace(Some(namespace))
                .unwrap_or(true),
            None => false,
        };
        if has_facts {
            return false;
        }
        let has_episodes = self
            .memory
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
        if let Some(semantic) = &self.memory.semantic {
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
        if let Some(semantic) = &self.memory.semantic {
            semantic
                .list_by_namespace_paginated(namespace, limit, offset)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get all facts about a specific subject.
    pub fn facts_about(&self, subject: &str, namespace: Option<&str>) -> Vec<hippocampus::Fact> {
        if let Some(semantic) = &self.memory.semantic {
            semantic
                .get_facts_about_in_namespace(subject, namespace)
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// List all namespaces with fact and episode counts.
    pub fn list_namespaces(&self) -> Vec<hippocampus::NamespaceStats> {
        if let Some(semantic) = &self.memory.semantic {
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
        self.memory
            .episodic
            .recent(limit, namespace)
            .unwrap_or_default()
    }

    /// Load the last `limit` episodes for a session as LLM messages, in
    /// chronological order. Used to give the LLM continuity across turns.
    pub(crate) fn load_session_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Vec<cortex::llm::Message> {
        let episodes = match self.memory.episodic.get_session_history(session_id, limit) {
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
