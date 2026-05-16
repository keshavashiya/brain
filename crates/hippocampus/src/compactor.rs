//! Graph compactor — applies half-life decay to every node's weight
//! and prunes nodes whose decayed weight falls below the cutoff.
//!
//! Cron-driven: the existing `orchestrate` scheduler invokes this on
//! a configured cadence. Each invocation is independent — the decay
//! math is referenced to `now - node.created_at`, not to the time of
//! the last compaction, so missed cycles self-correct.
//!
//! Embeddings (the `vector_id` link on each node) are **not** touched
//! here. Vector reclamation is a separate maintenance task because
//! the vector store may share embeddings across nodes once
//! deduplication ships.

use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use tracing::{debug, info};

use crate::graph::{EpisodicGraph, GraphError};

/// Tuning knobs for the [`DefaultCompactor`].
#[derive(Debug, Clone)]
pub struct CompactConfig {
    /// Half-life: a node's weight halves every this many seconds of
    /// wall-clock since `created_at`. Default 7 days — short enough
    /// to surface decay during a development week, long enough that
    /// production memories stay around weeks before pruning.
    pub half_life: Duration,
    /// Nodes whose decayed weight falls strictly below this are
    /// pruned. Default `0.05` — about 4.3 half-lives after a node
    /// starts at weight `1.0`.
    pub eviction_cutoff: f32,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self {
            half_life: Duration::from_secs(7 * 24 * 3600),
            eviction_cutoff: 0.05,
        }
    }
}

/// One compaction cycle's summary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactStats {
    /// Total nodes inspected.
    pub scanned: usize,
    /// Nodes whose weight was updated in place.
    pub decayed: usize,
    /// Nodes pruned because their decayed weight fell below the
    /// cutoff. Edges cascade per the migration v20 FK.
    pub evicted: usize,
}

/// Compactor trait — async so future impls (cold-tier export, vector
/// store cleanup) can do real I/O off the reactor thread.
#[async_trait]
pub trait Compactor: Send + Sync {
    async fn compact(&self, store: &dyn EpisodicGraph) -> Result<CompactStats, GraphError>;
}

/// Pure half-life decay compactor.
pub struct DefaultCompactor {
    config: CompactConfig,
}

impl DefaultCompactor {
    pub fn new(config: CompactConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &CompactConfig {
        &self.config
    }

    /// Compute the decayed weight at `now` for a node that started at
    /// `initial_weight` `elapsed_secs` ago. Pure function, exposed
    /// for tests pinning the decay curve.
    pub fn decayed_weight(initial_weight: f32, elapsed_secs: f64, half_life_secs: f64) -> f32 {
        if half_life_secs <= 0.0 || elapsed_secs <= 0.0 {
            return initial_weight;
        }
        // weight halves every half_life seconds:
        //   w(t) = w0 * 0.5 ^ (t / half_life)
        // which is the same shape as exp(-t * ln(2) / half_life).
        let factor = 0.5_f64.powf(elapsed_secs / half_life_secs);
        (initial_weight as f64 * factor) as f32
    }
}

#[async_trait]
impl Compactor for DefaultCompactor {
    async fn compact(&self, store: &dyn EpisodicGraph) -> Result<CompactStats, GraphError> {
        let now = Utc::now();
        let half_life_secs = self.config.half_life.as_secs_f64();
        let cutoff = self.config.eviction_cutoff;
        let mut stats = CompactStats::default();

        let nodes = store.list_all_nodes()?;
        for node in nodes {
            stats.scanned += 1;
            let elapsed = (now - node.created_at).num_milliseconds() as f64 / 1000.0;
            let new_weight = Self::decayed_weight(node.weight, elapsed, half_life_secs);
            if new_weight < cutoff {
                debug!(
                    node_id = %node.id,
                    weight = new_weight,
                    cutoff = cutoff,
                    "compactor evicting node"
                );
                if store.delete_node(&node.id)? {
                    stats.evicted += 1;
                }
            } else if (new_weight - node.weight).abs() > f32::EPSILON {
                store.update_weight(&node.id, new_weight)?;
                stats.decayed += 1;
            }
        }
        info!(
            scanned = stats.scanned,
            decayed = stats.decayed,
            evicted = stats.evicted,
            "compactor pass complete"
        );
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, EdgeKind, Node, NodeKind, SqliteGraph};
    use chrono::Duration as ChronoDuration;
    use storage::SqlitePool;

    fn store() -> SqliteGraph {
        SqliteGraph::new(SqlitePool::open_memory().expect("memory pool"))
    }

    fn make_node(weight: f32, age_secs: i64) -> Node {
        let mut n = Node::new(NodeKind::new("t"), serde_json::json!({}), "personal", None);
        n.weight = weight;
        n.created_at = Utc::now() - ChronoDuration::seconds(age_secs);
        n
    }

    #[test]
    fn decay_math_halves_at_half_life() {
        let w = DefaultCompactor::decayed_weight(1.0, 60.0, 60.0);
        assert!(
            (w - 0.5).abs() < 1e-5,
            "weight after one half-life should be 0.5, got {w}"
        );
    }

    #[test]
    fn decay_math_quarters_after_two_half_lives() {
        let w = DefaultCompactor::decayed_weight(1.0, 120.0, 60.0);
        assert!((w - 0.25).abs() < 1e-5, "two half-lives → 0.25, got {w}");
    }

    #[test]
    fn decay_math_zero_elapsed_is_identity() {
        let w = DefaultCompactor::decayed_weight(0.73, 0.0, 60.0);
        assert!((w - 0.73).abs() < 1e-6);
    }

    #[test]
    fn decay_math_zero_half_life_is_identity() {
        // Defensive: half_life <= 0 returns the input unchanged
        // rather than dividing by zero.
        let w = DefaultCompactor::decayed_weight(0.5, 99.0, 0.0);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn compactor_decays_recent_node_in_place() {
        let g = store();
        // 5s old, half_life 10s → weight should drop ~30% (0.5^0.5 ≈ 0.707).
        let n = make_node(1.0, 5);
        g.add_node(&n).unwrap();
        let compactor = DefaultCompactor::new(CompactConfig {
            half_life: Duration::from_secs(10),
            eviction_cutoff: 0.0,
        });
        let stats = compactor.compact(&g).await.unwrap();
        assert_eq!(stats.scanned, 1);
        assert_eq!(stats.decayed, 1);
        assert_eq!(stats.evicted, 0);
        let got = g.get_node(&n.id).unwrap().expect("node remains");
        assert!(
            got.weight < 1.0 && got.weight > 0.5,
            "weight {} should be decayed",
            got.weight
        );
    }

    #[tokio::test]
    async fn compactor_evicts_node_below_cutoff() {
        let g = store();
        // 1000s old, half_life 10s → weight ≈ 0.5^100 → effectively 0.
        let n = make_node(1.0, 1000);
        g.add_node(&n).unwrap();
        let compactor = DefaultCompactor::new(CompactConfig {
            half_life: Duration::from_secs(10),
            eviction_cutoff: 0.05,
        });
        let stats = compactor.compact(&g).await.unwrap();
        assert_eq!(stats.evicted, 1);
        assert!(g.get_node(&n.id).unwrap().is_none());
    }

    #[tokio::test]
    async fn compactor_evict_cascades_edges() {
        let g = store();
        let old = make_node(1.0, 1000);
        let young = make_node(1.0, 0);
        g.add_node(&old).unwrap();
        g.add_node(&young).unwrap();
        g.add_edge(&Edge::new(&old.id, &young.id, EdgeKind::new("k")))
            .unwrap();
        // Sanity: edge exists before compaction.
        assert_eq!(g.incoming(&young.id).unwrap().len(), 1);

        let compactor = DefaultCompactor::new(CompactConfig {
            half_life: Duration::from_secs(10),
            eviction_cutoff: 0.05,
        });
        let stats = compactor.compact(&g).await.unwrap();
        assert!(stats.evicted >= 1);
        // The edge cascaded with the deleted src node.
        assert!(g.incoming(&young.id).unwrap().is_empty());
    }

    #[tokio::test]
    async fn compactor_preserves_vector_id_reference_on_decayed_node() {
        let g = store();
        let mut n = make_node(1.0, 5);
        n.vector_id = Some("vec-keep-me".into());
        g.add_node(&n).unwrap();
        let compactor = DefaultCompactor::new(CompactConfig {
            half_life: Duration::from_secs(10),
            eviction_cutoff: 0.0,
        });
        compactor.compact(&g).await.unwrap();
        let got = g.get_node(&n.id).unwrap().expect("node remains");
        assert_eq!(got.vector_id.as_deref(), Some("vec-keep-me"));
    }

    #[tokio::test]
    async fn compactor_empty_graph_returns_zero_stats() {
        let g = store();
        let compactor = DefaultCompactor::new(CompactConfig::default());
        let stats = compactor.compact(&g).await.unwrap();
        assert_eq!(stats, CompactStats::default());
    }

    #[tokio::test]
    async fn compactor_mixed_decays_and_evicts() {
        let g = store();
        let young = make_node(1.0, 1);
        let old = make_node(1.0, 1000);
        g.add_node(&young).unwrap();
        g.add_node(&old).unwrap();
        let compactor = DefaultCompactor::new(CompactConfig {
            half_life: Duration::from_secs(10),
            eviction_cutoff: 0.05,
        });
        let stats = compactor.compact(&g).await.unwrap();
        assert_eq!(stats.scanned, 2);
        assert_eq!(stats.decayed, 1);
        assert_eq!(stats.evicted, 1);
    }
}
