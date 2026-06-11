//! Export / import operations for SignalProcessor.

use crate::types::ExportedEpisode;
use crate::types::ExportedFact;
use crate::SignalError;
use crate::SignalProcessor;
use futures::future::join_all;

/// A single fact to be stored (subject-predicate-object triple).
pub struct FactToStore {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl SignalProcessor {
    /// Export all semantic facts.
    pub fn export_facts(&self) -> Result<Vec<ExportedFact>, SignalError> {
        self.memory
            .episodic
            .pool()
            .export_all_facts()
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Export all episodes with their session info.
    pub fn export_episodes(&self) -> Result<Vec<ExportedEpisode>, SignalError> {
        self.memory
            .episodic
            .pool()
            .export_all_episodes()
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Import facts into SQLite (ON CONFLICT DO NOTHING). Returns (imported_count, new_fact_indices).
    pub fn import_facts(&self, facts: &[ExportedFact]) -> Result<(usize, Vec<usize>), SignalError> {
        self.memory
            .episodic
            .pool()
            .import_facts(facts)
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Import episodes into SQLite (ON CONFLICT DO NOTHING). Returns count of newly imported episodes.
    pub fn import_episodes(&self, episodes: &[ExportedEpisode]) -> Result<usize, SignalError> {
        self.memory
            .episodic
            .pool()
            .import_episodes(episodes)
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Re-embed facts into the vector index. Returns (embedded_count, failed_count).
    ///
    /// Generates embeddings concurrently, then inserts vectors sequentially
    /// (SQLite is single-writer).
    pub async fn reembed_facts(&self, facts: &[ExportedFact]) -> (usize, usize) {
        let semantic = match &self.memory.semantic {
            Some(s) => s,
            None => return (0, 0),
        };

        // Step 1: Generate embeddings concurrently
        let texts: Vec<String> = facts
            .iter()
            .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
            .collect();

        let embedding_futures: Vec<_> = facts
            .iter()
            .zip(texts.iter())
            .map(|(f, t)| self.embed_text(t, &f.namespace))
            .collect();
        let mut embeddings: Vec<Vec<f32>> = join_all(embedding_futures).await;

        // Step 2: Insert vectors sequentially (SQLite is single-writer).
        // Move each embedding out of the buffer via `mem::take` — each
        // vector is 3-6 KB and we never re-read `embeddings` after the
        // loop, so the per-iteration `.clone()` was pure waste.
        let mut embedded = 0usize;
        let mut failed = 0usize;

        for (i, f) in facts.iter().enumerate() {
            let vector = std::mem::take(&mut embeddings[i]);
            match semantic
                .add_vector(&f.id, &texts[i], vector, "semantic")
                .await
            {
                Ok(()) => embedded += 1,
                Err(e) => {
                    tracing::warn!("RuVector insert failed for fact {}: {e}", f.id);
                    failed += 1;
                }
            }
        }

        (embedded, failed)
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
        if let Some(semantic) = &self.memory.semantic {
            let fact_text = format!("{subject} {predicate} {object}");
            let importance = self.importance.score(&fact_text);
            let vector = self.embed_text(&fact_text, namespace).await;
            let id = semantic
                .store_fact(
                    namespace,
                    category,
                    subject,
                    predicate,
                    object,
                    importance as f64,
                    None,
                    vector,
                    agent,
                )
                .await
                .map_err(|e| SignalError::Storage(e.to_string()))?;
            self.quarantine_fact_if_unattested(&id, agent).await;
            Ok(id)
        } else {
            Err(SignalError::Storage(
                "Semantic store unavailable".to_string(),
            ))
        }
    }

    /// Store multiple facts concurrently.
    ///
    /// Generates embeddings for all facts in parallel, then stores each
    /// sequentially (SQLite is single-writer). Returns (stored_ids, errors).
    pub async fn store_facts_batch(
        &self,
        namespace: &str,
        category: &str,
        facts: &[FactToStore],
        agent: Option<&str>,
    ) -> (Vec<String>, Vec<(String, SignalError)>) {
        let semantic = match &self.memory.semantic {
            Some(s) => s,
            None => {
                let errors: Vec<_> = facts
                    .iter()
                    .map(|f| {
                        (
                            format!("{} {} {}", f.subject, f.predicate, f.object),
                            SignalError::Storage("Semantic store unavailable".to_string()),
                        )
                    })
                    .collect();
                return (Vec::new(), errors);
            }
        };

        // Step 1: Generate embeddings concurrently
        let texts: Vec<String> = facts
            .iter()
            .map(|f| format!("{} {} {}", f.subject, f.predicate, f.object))
            .collect();

        let embedding_futures: Vec<_> = texts
            .iter()
            .map(|t| self.embed_text(t, namespace))
            .collect();
        let mut embeddings: Vec<Vec<f32>> = join_all(embedding_futures).await;

        // Step 2: Store sequentially (SQLite is single-writer). Move
        // each embedding out via `mem::take` rather than cloning — see
        // `reembed_facts` for the same shape.
        let mut stored = Vec::new();
        let mut errors = Vec::new();

        // One attestation check covers the whole batch (same writer).
        let attested = self.memory_writer_attested(agent).await;

        for (i, fact) in facts.iter().enumerate() {
            let importance = self.importance.score(&texts[i]);
            let vector = std::mem::take(&mut embeddings[i]);
            match semantic
                .store_fact(
                    namespace,
                    category,
                    &fact.subject,
                    &fact.predicate,
                    &fact.object,
                    importance as f64,
                    None,
                    vector,
                    agent,
                )
                .await
            {
                Ok(id) => {
                    if !attested {
                        let agent = agent.expect("unattested implies an agent id");
                        if let Err(e) = semantic.quarantine_fact(&id, agent) {
                            tracing::warn!(id, agent, "failed to quarantine fact: {e}");
                        }
                    }
                    stored.push(id);
                }
                Err(e) => errors.push((texts[i].clone(), SignalError::Storage(e.to_string()))),
            }
        }

        (stored, errors)
    }
}
