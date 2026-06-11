//! Memory backend for the action dispatcher.

use std::sync::Arc;

/// The capabilities this backend declares for the one manifest. Semantic-memory
/// store/delete are always wired, so the config is not consulted.
pub fn capabilities(_config: &brain::BrainConfig) -> Vec<intent::ToolDescriptor> {
    use crate::capabilities::{backend, destructive, native, usage};
    use intent::ToolAnnotations;
    let mem = || backend("memory");
    vec![
        native(
            "memory",
            "store",
            mem(),
            ToolAnnotations::default(),
            usage(
                "The user states a durable fact about themselves, their world, projects, or preferences that should survive the session.",
                "Transient chit-chat, or content already captured as an episodic turn.",
                &["A subject-predicate-object triple can be extracted from the statement."],
                "free / local SQLite + embedding",
                "\"Remember that my deploy script lives in ops/deploy.sh\"",
            ),
        ),
        native(
            "memory",
            "delete",
            mem(),
            destructive(),
            usage(
                "The user asks to forget or correct a previously stored fact.",
                "When unsure which facts match — deletion is irreversible.",
                &["A matching subject/predicate is known."],
                "free / local",
                "\"Forget what I said about the old API key\"",
            ),
        ),
    ]
}

#[derive(Clone)]
pub struct DefaultMemoryBackend {
    pub semantic: Option<hippocampus::SemanticStore>,
    pub embedder: Arc<tokio::sync::Mutex<Option<hippocampus::Embedder>>>,
    pub embedding_dim: usize,
    /// Namespace residency policy — content from `local_only`
    /// namespaces is never sent to a remote embedder (deterministic
    /// fallback instead).
    pub residency: brain::ResidencyPolicy,
}

impl DefaultMemoryBackend {
    /// True when embedding `namespace` content through the wired
    /// embedder would take it off the machine in violation of policy.
    async fn remote_embed_blocked(&self, namespace: &str) -> bool {
        if self.residency.is_empty() || !self.residency.is_local_only(namespace) {
            return false;
        }
        match self.embedder.lock().await.as_ref() {
            Some(embedder) => !embedder.is_local(),
            None => false,
        }
    }
}

#[async_trait::async_trait]
impl cortex::actions::MemoryBackend for DefaultMemoryBackend {
    async fn store_fact(
        &self,
        namespace: &str,
        _category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String, cortex::actions::ActionError> {
        let Some(semantic) = &self.semantic else {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "Semantic store unavailable".to_string(),
            ));
        };

        let content = format!("{subject} {predicate} {object}");
        let vector = if self.remote_embed_blocked(namespace).await {
            hippocampus::embedding::deterministic_fallback_embedding(&content, self.embedding_dim)
        } else {
            let mut guard = self.embedder.lock().await;
            if let Some(embedder) = guard.as_mut() {
                match embedder.embed(&content).await {
                    Ok(v) => {
                        hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, &content)
                    }
                    Err(e) => {
                        tracing::warn!("ActionDispatcher embedding failed: {e}");
                        hippocampus::embedding::deterministic_fallback_embedding(
                            &content,
                            self.embedding_dim,
                        )
                    }
                }
            } else {
                hippocampus::embedding::deterministic_fallback_embedding(
                    &content,
                    self.embedding_dim,
                )
            }
        };

        semantic
            .store_fact(
                namespace, _category, subject, predicate, object, 1.0, None, vector, None,
            )
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))
    }

    async fn recall(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<cortex::actions::MemoryFact>, cortex::actions::ActionError> {
        let Some(semantic) = &self.semantic else {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "Semantic store unavailable".to_string(),
            ));
        };

        let vector = if self.remote_embed_blocked(namespace.unwrap_or("")).await {
            hippocampus::embedding::deterministic_fallback_embedding(query, self.embedding_dim)
        } else {
            let mut guard = self.embedder.lock().await;
            if let Some(embedder) = guard.as_mut() {
                match embedder.embed(query).await {
                    Ok(v) => {
                        hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, query)
                    }
                    Err(e) => {
                        tracing::warn!("ActionDispatcher embedding failed: {e}");
                        hippocampus::embedding::deterministic_fallback_embedding(
                            query,
                            self.embedding_dim,
                        )
                    }
                }
            } else {
                hippocampus::embedding::deterministic_fallback_embedding(query, self.embedding_dim)
            }
        };

        let results = semantic
            .search_similar(vector, top_k.max(1), namespace, None)
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| cortex::actions::MemoryFact {
                namespace: r.fact.namespace,
                subject: r.fact.subject,
                predicate: r.fact.predicate,
                object: r.fact.object,
                confidence: r.fact.confidence,
            })
            .collect())
    }
}
