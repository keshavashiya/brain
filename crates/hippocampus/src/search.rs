//! Recall engine — hybrid search with RRF fusion.
//!
//! Combines episodic BM25 search and semantic vector search
//! using Reciprocal Rank Fusion (RRF), then applies importance
//! and recency reranking with a forgetting curve.

use std::collections::HashMap;

use crate::episodic::{EpisodicStore, FtsResult};
use crate::semantic::{SemanticResult, SemanticStore};

/// A unified memory result from the recall engine.
#[derive(Debug, Clone)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub source: MemorySource,
    pub score: f64,
    pub importance: f64,
    pub timestamp: String,
    /// Originating agent that stored this memory (if known).
    pub agent: Option<String>,
}

/// Where this memory came from.
#[derive(Debug, Clone, PartialEq)]
pub enum MemorySource {
    Episodic,
    Semantic,
    /// The episodic graph (`nodes`) — terminal/tool events surfaced via
    /// graph FTS or ANN over `graph_vec`.
    Graph,
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
    /// Minimum similarity score for semantic results (0.0–1.0).
    /// ANN results with similarity below this threshold are discarded before fusion.
    pub similarity_threshold: f64,
}

impl RecallConfig {
    /// Build from individual config values (avoids cross-crate dependency on brain).
    pub fn from_config(
        rrf_k: u32,
        pre_fusion_limit: u32,
        importance_weight: f64,
        recency_weight: f64,
        decay_rate: f64,
        similarity_threshold: f64,
    ) -> Self {
        Self {
            rrf_k: rrf_k as f64,
            pre_fusion_limit: pre_fusion_limit as usize,
            importance_weight,
            recency_weight,
            decay_rate,
            similarity_threshold,
        }
    }
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            pre_fusion_limit: 50,
            importance_weight: 0.3,
            recency_weight: 0.2,
            decay_rate: 0.01,
            similarity_threshold: 0.65,
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
    /// 3. Query the episodic graph, if wired (FTS over node bodies + ANN over `graph_vec`)
    /// 4. Fuse all lists with Reciprocal Rank Fusion (k=60)
    /// 5. Rerank by importance × recency (forgetting curve)
    /// 6. Return top_k results
    #[allow(clippy::too_many_arguments)]
    pub async fn recall(
        &self,
        query: &str,
        query_vector: Vec<f32>,
        episodic: &EpisodicStore,
        semantic: &SemanticStore,
        top_k: usize,
        namespace: Option<&str>,
        agent: Option<&str>,
        graph: Option<&crate::DualMemoryReader>,
    ) -> Result<Vec<Memory>, RecallError> {
        let limit = self.config.pre_fusion_limit;
        let threshold = self.config.similarity_threshold;

        // 1. BM25 search on episodic store
        let bm25_results = episodic
            .search_bm25(query, limit, namespace, agent)
            .map_err(RecallError::Episodic)?;

        let bm25_ranked: Vec<(String, f64)> = bm25_results
            .iter()
            .map(|r| (r.episode_id.clone(), r.rank))
            .collect();

        // 2. ANN search on semantic store (filtered by namespace and/or agent if provided)
        let ann_results = semantic
            .search_similar(query_vector.clone(), limit, namespace, agent)
            .await
            .map_err(RecallError::Semantic)?;

        // Convert distance to similarity and filter by threshold.
        // distance is L2; similarity = 1/(1+d). Higher = more similar.
        let ann_ranked: Vec<(String, f64)> = ann_results
            .iter()
            .map(|r| (r.fact.id.clone(), 1.0 / (1.0 + r.distance as f64)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        // 3. Graph candidates (FTS + ANN over the episodic graph), if a
        //    reader is wired. The agent filter doesn't apply — graph nodes
        //    carry no per-agent column today. A hard store error surfaces as
        //    RecallError::Graph; the ANN half degrades to FTS-only internally
        //    (see DualMemoryReader::recall_candidates).
        let graph_candidates = match graph {
            Some(reader) => reader
                .recall_candidates(query, query_vector, limit, namespace)
                .await
                .map_err(RecallError::Graph)?,
            None => crate::GraphCandidates::default(),
        };
        let graph_fts_ranked = graph_candidates.fts.clone();
        let graph_ann_ranked: Vec<(String, f64)> = graph_candidates
            .ann
            .iter()
            .filter(|(_, sim)| *sim >= threshold)
            .cloned()
            .collect();

        // 4. RRF fusion across every candidate list
        let fused = rrf_fuse(
            &[bm25_ranked, ann_ranked, graph_fts_ranked, graph_ann_ranked],
            self.config.rrf_k,
        );

        // 5. Build lookup maps to avoid O(n*m) linear scans during reranking
        let bm25_map: HashMap<&str, &FtsResult> = bm25_results
            .iter()
            .map(|r| (r.episode_id.as_str(), r))
            .collect();
        let ann_map: HashMap<&str, &SemanticResult> = ann_results
            .iter()
            .map(|r| (r.fact.id.as_str(), r))
            .collect();

        // 6. Build Memory objects and rerank
        let now = chrono::Utc::now();
        let mut memories: Vec<Memory> = Vec::new();

        for (id, rrf_score) in &fused {
            // Try episodic first
            if let Some(fts) = bm25_map.get(id.as_str()) {
                let importance = fts.importance;
                let hours = parse_elapsed_hours(&fts.timestamp, &now);
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
                    timestamp: fts.timestamp.clone(),
                    agent: fts.agent.clone(),
                });
                continue;
            }

            // Try semantic
            if let Some(sr) = ann_map.get(id.as_str()) {
                let importance = sr.fact.confidence;
                let hours = parse_elapsed_hours(&sr.created_at, &now);
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
                    timestamp: sr.created_at.clone(),
                    agent: sr.fact.agent.clone(),
                });
                continue;
            }

            // Try the episodic graph. `weight` stands in for importance
            // (the compactor's half-life decay already lives there).
            if let Some(gc) = graph_candidates.hydration.get(id.as_str()) {
                let importance = gc.weight as f64;
                let timestamp = gc.created_at.to_rfc3339();
                let hours = parse_elapsed_hours(&timestamp, &now);
                let retention = forgetting_curve(importance, hours, self.config.decay_rate);
                let final_score = rrf_score
                    + self.config.importance_weight * importance
                    + self.config.recency_weight * retention;

                memories.push(Memory {
                    id: id.clone(),
                    content: gc.content.clone(),
                    source: MemorySource::Graph,
                    score: final_score,
                    importance,
                    timestamp,
                    agent: None,
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

/// Parse an ISO 8601 or SQLite datetime string and return hours elapsed since `now`.
///
/// Falls back to 1.0 hour if parsing fails (e.g., empty or malformed timestamp).
/// A fallback of 1.0h applies mild decay without artificially boosting (0.0) or
/// aggressively penalizing the memory. Logs a warning so serialization bugs are visible.
fn parse_elapsed_hours(timestamp: &str, now: &chrono::DateTime<chrono::Utc>) -> f64 {
    if timestamp.is_empty() {
        tracing::warn!("Empty timestamp in recall — using 1.0h fallback");
        return 1.0;
    }
    // Try RFC 3339 first (e.g. "2025-03-01T12:00:00+00:00")
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(timestamp) {
        let elapsed = *now - dt.with_timezone(&chrono::Utc);
        return (elapsed.num_seconds() as f64 / 3600.0).max(0.01);
    }
    // Try SQLite datetime format (e.g. "2025-03-01 12:00:00")
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%d %H:%M:%S") {
        let dt = naive.and_utc();
        let elapsed = *now - dt;
        return (elapsed.num_seconds() as f64 / 3600.0).max(0.01);
    }
    tracing::warn!(
        timestamp,
        "Unparseable timestamp in recall — using 1.0h fallback"
    );
    1.0 // fallback
}

/// Errors from the recall engine.
#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("Episodic search failed: {0}")]
    Episodic(crate::episodic::EpisodicError),

    #[error("Semantic search failed: {0}")]
    Semantic(crate::semantic::SemanticError),

    #[error("Graph recall failed: {0}")]
    Graph(crate::dual_memory::DualMemoryError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodic::EpisodicStore;
    use crate::graph::{EpisodicGraph, Node, NodeKind, SqliteGraph};
    use crate::semantic::SemanticStore;
    use crate::DualMemoryReader;
    use std::sync::Arc;
    use storage::{RuVectorStore, SqlitePool};

    /// End-to-end: a graph-only event (no episodic/semantic row) must
    /// surface in `recall` results via the graph FTS path. This is the
    /// acceptance criterion for the graph→recall composition.
    #[tokio::test]
    async fn recall_fuses_graph_only_fts_hit() {
        let episodic = EpisodicStore::new(SqlitePool::open_memory().unwrap());
        let ruv_dir = tempfile::tempdir().unwrap();
        let ruv = RuVectorStore::open(ruv_dir.path(), 384).await.unwrap();
        ruv.ensure_tables().await.unwrap();
        let semantic = SemanticStore::new(SqlitePool::open_memory().unwrap(), ruv);

        let g: Arc<dyn EpisodicGraph> =
            Arc::new(SqliteGraph::new(SqlitePool::open_memory().unwrap()));
        let node = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"verb": "terminal.open", "program": "htop"}),
            "personal",
            None,
        );
        g.add_node(&node).unwrap();
        let reader = DualMemoryReader::graph_only(g);

        let engine = RecallEngine::with_defaults();
        let results = engine
            .recall(
                "htop",
                vec![0.0; 384],
                &episodic,
                &semantic,
                10,
                None,
                None,
                Some(&reader),
            )
            .await
            .unwrap();

        let graph_hit = results
            .iter()
            .find(|m| m.source == MemorySource::Graph)
            .expect("the graph node must appear in recall results");
        assert_eq!(graph_hit.id, node.id);
        assert!(graph_hit.content.contains("htop"));

        // Control: without the graph reader, recall yields nothing.
        let without = engine
            .recall(
                "htop",
                vec![0.0; 384],
                &episodic,
                &semantic,
                10,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            without.is_empty(),
            "the hit must come from the graph path, not episodic/semantic"
        );
    }

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

    // ── Property tests ────────────────────────────────────────────────
    //
    // `rrf_fuse` ranks recall candidates and `forgetting_curve` reranks them
    // by recency; a fusion that dropped or duplicated an id, returned an
    // unsorted list, or a curve that grew retention with elapsed time would
    // silently corrupt what the kernel remembers. These pin the algebra for
    // arbitrary inputs.
    use proptest::prelude::*;
    use std::collections::HashSet;

    /// A ranked list with ids unique *within* the list, so each id has a
    /// well-defined rank (the only shape `rrf_fuse` is meant to receive).
    fn ranked_list() -> impl Strategy<Value = Vec<(String, f64)>> {
        prop::collection::vec(0u32..50, 0..12).prop_map(|idxs| {
            let mut seen = HashSet::new();
            idxs.into_iter()
                .filter(|i| seen.insert(*i))
                .map(|i| (format!("id{i}"), 0.0))
                .collect()
        })
    }

    fn ranked_lists() -> impl Strategy<Value = Vec<Vec<(String, f64)>>> {
        prop::collection::vec(ranked_list(), 0..5)
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// The fused output is sorted by score, highest first.
        #[test]
        fn rrf_output_is_sorted_descending(lists in ranked_lists(), k in 0.5f64..200.0) {
            let fused = rrf_fuse(&lists, k);
            for w in fused.windows(2) {
                prop_assert!(w[0].1 >= w[1].1);
            }
        }

        /// Fusion neither invents nor drops ids: the output is exactly the
        /// set union of the inputs, each appearing once.
        #[test]
        fn rrf_output_is_exactly_the_union(lists in ranked_lists(), k in 0.5f64..200.0) {
            let fused = rrf_fuse(&lists, k);
            let union: HashSet<&str> = lists
                .iter()
                .flat_map(|l| l.iter().map(|(id, _)| id.as_str()))
                .collect();
            let got: HashSet<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
            prop_assert_eq!(fused.len(), union.len(), "no duplicate ids in output");
            prop_assert_eq!(got, union);
        }

        /// Every fused score is positive and bounded by `n_lists / (k+1)` —
        /// the score an id earns by topping every list.
        #[test]
        fn rrf_scores_are_positive_and_bounded(lists in ranked_lists(), k in 0.5f64..200.0) {
            let fused = rrf_fuse(&lists, k);
            let ceiling = lists.len() as f64 / (k + 1.0);
            for (_, score) in &fused {
                prop_assert!(*score > 0.0);
                prop_assert!(*score <= ceiling + 1e-12);
            }
        }

        /// RRF is additive across lists: fusing a list alongside an identical
        /// copy doubles every id's score (each appears at the same rank in
        /// both). Pins the `Σ 1/(k+rank)` contribution.
        #[test]
        fn rrf_is_additive_over_repeated_lists(list in ranked_list(), k in 0.5f64..200.0) {
            let single = rrf_fuse(std::slice::from_ref(&list), k);
            let doubled = rrf_fuse(&[list.clone(), list], k);
            let single_map: HashMap<&str, f64> =
                single.iter().map(|(id, s)| (id.as_str(), *s)).collect();
            for (id, s) in &doubled {
                let one = single_map[id.as_str()];
                prop_assert!((s - 2.0 * one).abs() <= 1e-12);
            }
        }

        /// Retention scales linearly with importance: doubling importance
        /// doubles retention (the curve only touches the time term).
        #[test]
        fn forgetting_curve_is_linear_in_importance(
            importance in 0.0f64..1.0,
            hours in 0.0f64..10_000.0,
            decay in 0.0f64..1.0,
            factor in 0.0f64..5.0,
        ) {
            let base = forgetting_curve(importance, hours, decay);
            let scaled = forgetting_curve(importance * factor, hours, decay);
            prop_assert!((scaled - factor * base).abs() <= 1e-9 + base.abs() * 1e-9);
        }

        /// For non-negative inputs, retention never exceeds the starting
        /// importance and never goes negative — recall can't manufacture
        /// salience out of decay.
        #[test]
        fn forgetting_curve_stays_within_zero_and_importance(
            importance in 0.0f64..1e6,
            hours in 0.0f64..10_000.0,
            decay in 0.0f64..1.0,
        ) {
            let r = forgetting_curve(importance, hours, decay);
            prop_assert!(r >= 0.0);
            prop_assert!(r <= importance + 1e-9);
        }

        /// Retention is monotone non-increasing in elapsed time: a memory
        /// touched longer ago never scores higher on recency.
        #[test]
        fn forgetting_curve_is_monotone_in_elapsed(
            importance in 0.0f64..1e3,
            decay in 0.0f64..1.0,
            h1 in 0.0f64..10_000.0,
            h2 in 0.0f64..10_000.0,
        ) {
            let (lo, hi) = if h1 <= h2 { (h1, h2) } else { (h2, h1) };
            prop_assert!(forgetting_curve(importance, lo, decay) + 1e-12
                >= forgetting_curve(importance, hi, decay));
        }

        /// Zero elapsed time (or zero decay) means full retention — the
        /// importance passes through untouched.
        #[test]
        fn forgetting_curve_is_identity_without_decay(
            importance in 0.0f64..1e3,
            hours in 0.0f64..10_000.0,
            decay in 0.0f64..1.0,
        ) {
            prop_assert_eq!(forgetting_curve(importance, 0.0, decay), importance);
            prop_assert_eq!(forgetting_curve(importance, hours, 0.0), importance);
        }
    }
}
