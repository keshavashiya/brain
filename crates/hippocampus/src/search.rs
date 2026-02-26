//! Recall engine — hybrid search with RRF fusion.
//!
//! Combines episodic BM25 search and semantic vector search
//! using Reciprocal Rank Fusion (RRF), then applies importance
//! and recency reranking with a forgetting curve.

use std::collections::HashMap;

use crate::episodic::EpisodicStore;
use crate::semantic::SemanticStore;

/// A unified memory result from the recall engine.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub source: MemorySource,
    pub score: f64,
    pub importance: f64,
    pub timestamp: String,
}

/// Where this memory came from.
#[derive(Debug, Clone, PartialEq)]
pub enum MemorySource {
    Episodic,
    Semantic,
}

/// Configuration for the recall engine.
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// RRF constant (default: 60).
    pub rrf_k: f64,
    /// How many candidates to fetch from each source before fusion.
    pub pre_fusion_limit: usize,
    /// Weight for importance in final reranking (0.0–1.0).
    pub importance_weight: f64,
    /// Weight for recency in final reranking (0.0–1.0).
    pub recency_weight: f64,
    /// Decay rate for the forgetting curve (higher = faster decay).
    pub decay_rate: f64,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            pre_fusion_limit: 50,
            importance_weight: 0.3,
            recency_weight: 0.2,
            decay_rate: 0.01,
        }
    }
}

/// Reciprocal Rank Fusion (RRF) algorithm.
///
/// Given multiple ranked lists, produces a single fused ranking.
/// Score for item i = Σ (1 / (k + rank_i)) across all lists.
pub fn rrf_fuse(ranked_lists: &[Vec<(String, f64)>], k: f64) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for list in ranked_lists {
        for (rank, (id, _original_score)) in list.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f64 + 1.0));
            *scores.entry(id.clone()).or_default() += rrf_score;
        }
    }

    let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Calculate retention using a simplified forgetting curve.
///
/// `retention = importance * e^(-decay_rate * hours_since_access)`
pub fn forgetting_curve(importance: f64, hours_since_access: f64, decay_rate: f64) -> f64 {
    importance * (-decay_rate * hours_since_access).exp()
}

/// The recall engine orchestrates memory retrieval.
///
/// It queries both episodic (BM25) and semantic (vector) stores,
/// fuses results with RRF, and reranks by importance + recency.
pub struct RecallEngine {
    config: RecallConfig,
}

impl RecallEngine {
    pub fn new(config: RecallConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(RecallConfig::default())
    }

    /// Recall memories relevant to a query.
    ///
    /// Pipeline:
    /// 1. Query episodic store (BM25 full-text search)
    /// 2. Query semantic store (ANN vector search, optionally scoped to namespace)
    /// 3. Fuse with Reciprocal Rank Fusion (k=60)
    /// 4. Rerank by importance × recency (forgetting curve)
    /// 5. Return top_k results
    pub async fn recall(
        &self,
        query: &str,
        query_vector: Vec<f32>,
        episodic: &EpisodicStore,
        semantic: &SemanticStore,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<Memory>, RecallError> {
        let limit = self.config.pre_fusion_limit;

        // 1. BM25 search on episodic store
        let bm25_results = episodic
            .search_bm25(query, limit)
            .map_err(RecallError::Episodic)?;

        let bm25_ranked: Vec<(String, f64)> = bm25_results
            .iter()
            .map(|r| (r.episode_id.clone(), r.rank))
            .collect();

        // 2. ANN search on semantic store (filtered by namespace if provided)
        let ann_results = semantic
            .search_similar(query_vector, limit, namespace)
            .await
            .map_err(RecallError::Semantic)?;

        let ann_ranked: Vec<(String, f64)> = ann_results
            .iter()
            .map(|r| (r.fact.id.clone(), 1.0 / (1.0 + r.distance as f64)))
            .collect();

        // 3. RRF fusion
        let fused = rrf_fuse(&[bm25_ranked, ann_ranked], self.config.rrf_k);

        // 4. Build Memory objects and rerank
        let _now = chrono::Utc::now();
        let mut memories: Vec<Memory> = Vec::new();

        for (id, rrf_score) in &fused {
            // Try episodic first
            if let Some(fts) = bm25_results.iter().find(|r| &r.episode_id == id) {
                let importance = 0.5; // default for BM25 hits
                let hours = 1.0; // simplified — would parse timestamp
                let retention = forgetting_curve(importance, hours, self.config.decay_rate);
                let final_score = rrf_score
                    + self.config.importance_weight * importance
                    + self.config.recency_weight * retention;

                memories.push(Memory {
                    id: id.clone(),
                    content: fts.content.clone(),
                    source: MemorySource::Episodic,
                    score: final_score,
                    importance,
                    timestamp: String::new(),
                });
                continue;
            }

            // Try semantic
            if let Some(sr) = ann_results.iter().find(|r| &r.fact.id == id) {
                let importance = sr.fact.confidence;
                let hours = 1.0;
                let retention = forgetting_curve(importance, hours, self.config.decay_rate);
                let final_score = rrf_score
                    + self.config.importance_weight * importance
                    + self.config.recency_weight * retention;

                let content = format!(
                    "{} {} {}",
                    sr.fact.subject, sr.fact.predicate, sr.fact.object
                );

                memories.push(Memory {
                    id: id.clone(),
                    content,
                    source: MemorySource::Semantic,
                    score: final_score,
                    importance,
                    timestamp: String::new(),
                });
            }
        }

        // Sort by final score descending
        memories.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        memories.truncate(top_k);

        Ok(memories)
    }
}

/// Errors from the recall engine.
#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("Episodic search failed: {0}")]
    Episodic(crate::episodic::EpisodicError),

    #[error("Semantic search failed: {0}")]
    Semantic(crate::semantic::SemanticError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let lists = vec![vec![
            ("a".to_string(), 10.0),
            ("b".to_string(), 5.0),
            ("c".to_string(), 1.0),
        ]];

        let fused = rrf_fuse(&lists, 60.0);
        assert_eq!(fused[0].0, "a");
        assert_eq!(fused[1].0, "b");
        assert_eq!(fused[2].0, "c");

        // rank 1: 1/(60+1) ≈ 0.01639
        assert!((fused[0].1 - 1.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_two_lists() {
        let lists = vec![
            vec![("a".to_string(), 10.0), ("b".to_string(), 5.0)],
            vec![("b".to_string(), 10.0), ("a".to_string(), 5.0)],
        ];

        let fused = rrf_fuse(&lists, 60.0);

        // Both a and b appear at rank 1 and rank 2 in different lists
        // Both should have score 1/61 + 1/62
        assert_eq!(fused.len(), 2);
        let score_a = fused.iter().find(|(id, _)| id == "a").unwrap().1;
        let score_b = fused.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!((score_a - score_b).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_disjoint_lists() {
        let lists = vec![vec![("a".to_string(), 10.0)], vec![("b".to_string(), 10.0)]];

        let fused = rrf_fuse(&lists, 60.0);
        assert_eq!(fused.len(), 2);
        // Both at rank 1 in their respective lists
        let score_a = fused.iter().find(|(id, _)| id == "a").unwrap().1;
        let score_b = fused.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!((score_a - score_b).abs() < 1e-10);
    }

    #[test]
    fn test_rrf_overlap_boost() {
        let lists = vec![
            vec![
                ("a".to_string(), 10.0),
                ("b".to_string(), 5.0),
                ("c".to_string(), 1.0),
            ],
            vec![("a".to_string(), 10.0), ("c".to_string(), 5.0)],
        ];

        let fused = rrf_fuse(&lists, 60.0);

        // 'a' appears at rank 1 in both lists → highest score
        assert_eq!(fused[0].0, "a");

        // 'c' appears in both lists (rank 3 + rank 2) → higher than 'b' (rank 2 only)
        let score_b = fused.iter().find(|(id, _)| id == "b").unwrap().1;
        let score_c = fused.iter().find(|(id, _)| id == "c").unwrap().1;
        assert!(score_c > score_b, "c (in both) should rank > b (in one)");
    }

    #[test]
    fn test_forgetting_curve_no_decay() {
        let retention = forgetting_curve(1.0, 0.0, 0.01);
        assert!((retention - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_forgetting_curve_decay() {
        let retention_1h = forgetting_curve(1.0, 1.0, 0.01);
        let retention_24h = forgetting_curve(1.0, 24.0, 0.01);
        let retention_168h = forgetting_curve(1.0, 168.0, 0.01); // 1 week

        // Retention should decrease over time
        assert!(retention_1h > retention_24h);
        assert!(retention_24h > retention_168h);

        // High importance slows decay
        let retention_high = forgetting_curve(1.0, 24.0, 0.01);
        let retention_low = forgetting_curve(0.5, 24.0, 0.01);
        assert!(retention_high > retention_low);
    }

    #[test]
    fn test_forgetting_curve_importance_scaling() {
        let ret_a = forgetting_curve(1.0, 10.0, 0.01);
        let ret_b = forgetting_curve(0.5, 10.0, 0.01);
        // ret_a should be exactly 2x ret_b (linear in importance)
        assert!((ret_a / ret_b - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rrf_empty_lists() {
        let fused = rrf_fuse(&[], 60.0);
        assert!(fused.is_empty());

        let fused2 = rrf_fuse(&[vec![]], 60.0);
        assert!(fused2.is_empty());
    }

    #[test]
    fn test_recall_config_defaults() {
        let config = RecallConfig::default();
        assert_eq!(config.rrf_k, 60.0);
        assert_eq!(config.pre_fusion_limit, 50);
        assert!((config.importance_weight - 0.3).abs() < 1e-6);
        assert!((config.recency_weight - 0.2).abs() < 1e-6);
    }
}
