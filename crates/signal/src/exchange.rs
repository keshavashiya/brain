//! Export / import operations for SignalProcessor.

use crate::types::ExportedFact;
use crate::types::ExportedEpisode;
use crate::SignalError;
use crate::SignalProcessor;

impl SignalProcessor {
    /// Export all semantic facts.
    pub fn export_facts(&self) -> Result<Vec<ExportedFact>, SignalError> {
        self.episodic
            .pool()
            .export_all_facts()
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Export all episodes with their session info.
    pub fn export_episodes(&self) -> Result<Vec<ExportedEpisode>, SignalError> {
        self.episodic
            .pool()
            .export_all_episodes()
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Import facts into SQLite (ON CONFLICT DO NOTHING). Returns (imported_count, new_fact_indices).
    pub fn import_facts(&self, facts: &[ExportedFact]) -> Result<(usize, Vec<usize>), SignalError> {
        self.episodic
            .pool()
            .import_facts(facts)
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Import episodes into SQLite (ON CONFLICT DO NOTHING). Returns count of newly imported episodes.
    pub fn import_episodes(
        &self,
        episodes: &[ExportedEpisode],
    ) -> Result<usize, SignalError> {
        self.episodic
            .pool()
            .import_episodes(episodes)
            .map_err(|e| SignalError::Storage(e.to_string()))
    }

    /// Re-embed facts into the vector index. Returns (embedded_count, failed_count).
    pub async fn reembed_facts(&self, facts: &[ExportedFact]) -> (usize, usize) {
        let semantic = match &self.semantic {
            Some(s) => s,
            None => return (0, 0),
        };

        let mut embedded = 0usize;
        let mut failed = 0usize;

        for f in facts {
            let text = format!("{} {} {}", f.subject, f.predicate, f.object);
            let vector = self.embed_text(&text).await;

            match semantic.add_vector(&f.id, &text, vector, "semantic").await {
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
        if let Some(semantic) = &self.semantic {
            let fact_text = format!("{subject} {predicate} {object}");
            let importance = self.importance.score(&fact_text);
            let vector = self.embed_text(&fact_text).await;
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
            Ok(id)
        } else {
            Err(SignalError::Storage(
                "Semantic store unavailable".to_string(),
            ))
        }
    }
}
