//! Dual memory model — reconciliation layer between the legacy `episodes`
//! table and the newer episodic graph.
//!
//! ## What this is for
//!
//! The episodic graph ([`crate::graph`]) carries typed `Node`/`Edge` records.
//! The original conversation log (`episodes` table + [`crate::EpisodicStore`])
//! still ships — too many production code paths read from it for an
//! all-at-once cutover. Both stores coexist, **writes target the graph
//! going forward**, and reads reconcile through the [`DualMemoryReader`]
//! helper here.
//!
//! ## Read semantics
//!
//! `DualMemoryReader::read_by_id(id)`:
//! - tries `graph.get_node(id)` first — graph nodes are the authoritative
//!   shape for everything written after the graph schema landed;
//! - falls back to `legacy.get_episode(id)` so historic content stays
//!   reachable.
//!
//! The returned [`MemoryEntry`] keeps the underlying shape so callers
//! that need either field set can still discriminate (graph nodes carry
//! typed bodies; episodes carry role/content/decay metadata).
//!
//! ## Forward migration plan
//!
//! 1. **Backfill release.** A backfill task converts every row in
//!    `episodes` / `semantic_facts` into graph nodes (`node_kind: "episode"`
//!    / `"fact"`), preserving id so consumers don't break. Run once at upgrade.
//! 2. **Switch reads.** [`DualMemoryReader`] flips its default to graph-only;
//!    the legacy code path emits a deprecation warning.
//! 3. **Cleanup migration.** Drop the legacy tables (`episodes`,
//!    `semantic_facts`, `episodes_fts`, `episode_promotions`, related
//!    indexes). FTS5 over the graph becomes the new search path —
//!    body_json field-restricted MATCH.
//! 4. **Crate cleanup.** Remove `EpisodicStore` / [`crate::semantic::SemanticStore`]
//!    surfaces or hide them behind a `legacy` feature for two more minor releases.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use storage::RuVectorStore;

use crate::episodic::EpisodicStore;
use crate::graph::{EpisodicGraph, Node};
use crate::Episode;

/// RuVector collection holding graph-node embeddings — written by the
/// terminal graph sink, queried here for ANN recall. Must match the
/// constant in `signal::terminal_graph_mirror`.
const GRAPH_VEC: &str = "graph_vec";

/// Errors from the dual-memory read path.
#[derive(Debug, thiserror::Error)]
pub enum DualMemoryError {
    #[error("graph read: {0}")]
    Graph(#[from] crate::graph::GraphError),
    #[error("legacy read: {0}")]
    Legacy(#[from] crate::episodic::EpisodicError),
}

/// Read-side variant. Callers inspect the variant to pick the field
/// set they need — graph nodes carry typed `body_json`, episodes
/// carry `role`/`content`/decay metadata.
#[derive(Debug, Clone)]
pub enum MemoryEntry {
    Graph(Node),
    Legacy(Episode),
}

impl MemoryEntry {
    pub fn id(&self) -> &str {
        match self {
            MemoryEntry::Graph(n) => &n.id,
            MemoryEntry::Legacy(e) => &e.id,
        }
    }

    pub fn is_graph(&self) -> bool {
        matches!(self, MemoryEntry::Graph(_))
    }

    pub fn is_legacy(&self) -> bool {
        matches!(self, MemoryEntry::Legacy(_))
    }
}

/// One hydrated graph candidate for recall fusion — enough to build a
/// `Memory` without re-reading the node.
#[derive(Debug, Clone)]
pub struct GraphCandidate {
    pub content: String,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
    /// Namespace of the underlying node — recall threads this through
    /// to the residency filter at prompt assembly.
    pub namespace: String,
}

/// Graph candidate lists for one recall query, ready to fold into RRF.
/// `fts` and `ann` are ordered best-first; `hydration` maps every id in
/// either list to its `Memory`-building metadata.
#[derive(Debug, Clone, Default)]
pub struct GraphCandidates {
    /// `(node_id, bm25_rank)` — lower rank is a better text match.
    pub fts: Vec<(String, f64)>,
    /// `(node_id, similarity)` — `1/(1+distance)`, higher is closer.
    pub ann: Vec<(String, f64)>,
    pub hydration: HashMap<String, GraphCandidate>,
}

/// Read facade unifying the graph and the legacy episodic store.
/// Cheap to clone — both inner handles are `Arc`-shared.
#[derive(Clone)]
pub struct DualMemoryReader {
    graph: Option<Arc<dyn EpisodicGraph>>,
    legacy: Option<Arc<EpisodicStore>>,
    /// Vector store backing graph-node ANN (`graph_vec`). When unset the
    /// ANN half of [`Self::recall_candidates`] is skipped (FTS still runs).
    vectors: Option<RuVectorStore>,
}

impl DualMemoryReader {
    /// Reader that only consults the graph. Useful in tests and on
    /// fresh installs that never had the legacy tables populated.
    pub fn graph_only(graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph: Some(graph),
            legacy: None,
            vectors: None,
        }
    }

    /// Reader that only consults the legacy store. Used during the
    /// transition while a new SqlitePool hasn't yet been wired to a
    /// graph adapter.
    pub fn legacy_only(legacy: Arc<EpisodicStore>) -> Self {
        Self {
            graph: None,
            legacy: Some(legacy),
            vectors: None,
        }
    }

    /// The production wiring: prefer the graph, fall back to legacy.
    pub fn dual(legacy: Arc<EpisodicStore>, graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph: Some(graph),
            legacy: Some(legacy),
            vectors: None,
        }
    }

    /// Attach the vector store so [`Self::recall_candidates`] can run the
    /// ANN half over `graph_vec`. Without it, only graph FTS contributes.
    pub fn with_vector_store(mut self, vectors: RuVectorStore) -> Self {
        self.vectors = Some(vectors);
        self
    }

    /// Look up one entry by id. Tries the graph first, then the
    /// legacy store. Returns `Ok(None)` if no row matches anywhere.
    pub fn read_by_id(&self, id: &str) -> Result<Option<MemoryEntry>, DualMemoryError> {
        if let Some(graph) = &self.graph {
            if let Some(node) = graph.get_node(id)? {
                return Ok(Some(MemoryEntry::Graph(node)));
            }
        }
        if let Some(legacy) = &self.legacy {
            if let Some(ep) = legacy.get_episode(id)? {
                return Ok(Some(MemoryEntry::Legacy(ep)));
            }
        }
        Ok(None)
    }

    /// Gather graph candidates for a recall query: a BM25 list from the
    /// `nodes_fts` index and an ANN list from the `graph_vec` collection.
    /// Both are scoped to `namespace` (FTS at the SQL layer; ANN by
    /// hydrating the node and checking its namespace). Returns ordered
    /// id-lists plus a hydration map so the recall engine can fuse and
    /// materialize hits without a second round-trip.
    ///
    /// Returns empty when no graph is wired. The ANN half is skipped when
    /// no vector store is attached; FTS still runs.
    pub async fn recall_candidates(
        &self,
        query: &str,
        query_vector: Vec<f32>,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<GraphCandidates, DualMemoryError> {
        let Some(graph) = &self.graph else {
            return Ok(GraphCandidates::default());
        };
        let mut out = GraphCandidates::default();

        // FTS half — body-text BM25 over nodes_fts.
        for hit in graph.search_text(query, limit, namespace)? {
            out.fts.push((hit.id.clone(), hit.rank));
            out.hydration.entry(hit.id).or_insert(GraphCandidate {
                content: hit.text,
                weight: hit.weight,
                created_at: hit.created_at,
                namespace: hit.namespace,
            });
        }

        // ANN half — vector search over graph_vec, hydrated through the
        // graph. Best-effort: a vector-store error degrades to FTS-only.
        if let Some(vectors) = &self.vectors {
            match vectors.search(GRAPH_VEC, query_vector, limit).await {
                Ok(results) => {
                    for vr in results {
                        let Some(node) = graph.get_node(&vr.id)? else {
                            continue; // vector outlived its node
                        };
                        if namespace.is_some_and(|ns| !namespace_matches(ns, &node.namespace)) {
                            continue;
                        }
                        let similarity = 1.0 / (1.0 + vr.distance as f64);
                        out.ann.push((node.id.clone(), similarity));
                        out.hydration
                            .entry(node.id.clone())
                            .or_insert_with(|| GraphCandidate {
                                content: node_content(&node),
                                weight: node.weight,
                                created_at: node.created_at,
                                namespace: node.namespace.clone(),
                            });
                    }
                }
                Err(e) => {
                    tracing::warn!("graph_vec ANN search failed, FTS-only graph recall: {e}");
                }
            }
        }

        Ok(out)
    }
}

/// Namespace scope test mirroring the SQL filter: exact match or a
/// `ns/…` sub-namespace.
fn namespace_matches(scope: &str, ns: &str) -> bool {
    ns == scope || ns.starts_with(&format!("{scope}/"))
}

/// Content projection for an ANN-hydrated node — the JSON body as text,
/// matching what the FTS index stores so fused hits read consistently.
fn node_content(node: &Node) -> String {
    serde_json::to_string(&node.body).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeKind, SqliteGraph};
    use storage::SqlitePool;

    fn pool() -> SqlitePool {
        SqlitePool::open_memory().expect("memory pool")
    }

    /// 384-dim one-hot vector so seeded + query vectors can be made
    /// identical (distance ≈ 0 → similarity ≈ 1).
    fn unit_vector(idx: usize) -> Vec<f32> {
        let mut v = vec![0.0; 384];
        v[idx % 384] = 1.0;
        v
    }

    #[tokio::test]
    async fn recall_candidates_returns_graph_fts_hit() {
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool()));
        let mut n = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"program": "ripgrep"}),
            "personal",
            None,
        );
        n.weight = 0.7;
        g.add_node(&n).unwrap();

        let reader = DualMemoryReader::graph_only(g);
        let cands = reader
            .recall_candidates("ripgrep", vec![0.0; 384], 10, None)
            .await
            .unwrap();

        assert_eq!(cands.fts.len(), 1, "FTS should surface the ripgrep node");
        assert_eq!(cands.fts[0].0, n.id);
        let hyd = cands.hydration.get(&n.id).expect("hydration entry");
        assert!((hyd.weight - 0.7).abs() < 1e-6);
        assert!(cands.ann.is_empty(), "no vector store wired → no ANN list");
    }

    #[tokio::test]
    async fn recall_candidates_returns_graph_ann_hit() {
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool()));
        let n = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"program": "opaque-binary"}),
            "personal",
            None,
        );
        g.add_node(&n).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let ruv = RuVectorStore::open(dir.path(), 384).await.unwrap();
        ruv.ensure_tables().await.unwrap();
        let seeded = unit_vector(42);
        ruv.add_vectors(
            GRAPH_VEC,
            vec![n.id.clone()],
            vec!["opaque-binary".into()],
            vec![seeded.clone()],
            vec![n.created_at.to_rfc3339()],
            "graph",
        )
        .await
        .unwrap();

        let reader = DualMemoryReader::graph_only(g).with_vector_store(ruv);
        // Query text that FTS would NOT match ("xyzzy") so the hit can
        // only come from the ANN half.
        let cands = reader
            .recall_candidates("xyzzy", seeded, 10, None)
            .await
            .unwrap();

        assert!(cands.fts.is_empty(), "text query must not match via FTS");
        assert_eq!(cands.ann.len(), 1, "ANN should surface the seeded node");
        assert_eq!(cands.ann[0].0, n.id);
        assert!(cands.ann[0].1 > 0.9, "identical vector → high similarity");
        assert!(cands.hydration.contains_key(&n.id));
    }

    #[tokio::test]
    async fn recall_candidates_scopes_fts_to_namespace() {
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool()));
        let work = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"program": "deploy"}),
            "work",
            None,
        );
        let personal = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"program": "deploy"}),
            "personal",
            None,
        );
        g.add_node(&work).unwrap();
        g.add_node(&personal).unwrap();

        let reader = DualMemoryReader::graph_only(g);
        let cands = reader
            .recall_candidates("deploy", vec![0.0; 384], 10, Some("work"))
            .await
            .unwrap();
        assert_eq!(cands.fts.len(), 1);
        assert_eq!(cands.fts[0].0, work.id);
    }

    #[tokio::test]
    async fn recall_candidates_empty_without_graph() {
        let store = EpisodicStore::new(pool());
        let reader = DualMemoryReader::legacy_only(Arc::new(store));
        let cands = reader
            .recall_candidates("anything", vec![0.0; 384], 10, None)
            .await
            .unwrap();
        assert!(cands.fts.is_empty() && cands.ann.is_empty());
    }

    #[test]
    fn graph_only_reader_finds_graph_node() {
        let p = pool();
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(p));
        let n = Node::new(
            NodeKind::new("episode"),
            serde_json::json!({"x": 1}),
            "personal",
            None,
        );
        g.add_node(&n).unwrap();
        let r = DualMemoryReader::graph_only(g);
        let got = r.read_by_id(&n.id).unwrap().expect("found");
        assert!(got.is_graph());
        assert_eq!(got.id(), n.id);
    }

    #[test]
    fn legacy_only_reader_finds_episode() {
        let pool = pool();
        let store = EpisodicStore::new(pool);
        let sid = store.create_session("test").unwrap();
        let eid = store
            .store_episode(&sid, "user", "hello", 0.5, None, None)
            .unwrap();
        let r = DualMemoryReader::legacy_only(Arc::new(store));
        let got = r.read_by_id(&eid).unwrap().expect("found");
        assert!(got.is_legacy());
        assert_eq!(got.id(), &eid);
    }

    #[test]
    fn dual_reader_prefers_graph_when_both_exist() {
        // Same SqlitePool backs both stores so the test is in-process
        // realistic. The graph node and legacy episode coincidentally
        // share an id to prove the graph wins on conflict.
        let pool = pool();
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool.clone()));
        let legacy = Arc::new(EpisodicStore::new(pool));

        let sid = legacy.create_session("test").unwrap();
        let eid = legacy
            .store_episode(&sid, "user", "legacy text", 0.5, None, None)
            .unwrap();
        // Add a graph node under the *same* id by hand — bypasses
        // `Node::new`'s UUID minting to simulate a hypothetical
        // backfill that preserves ids.
        let n = Node {
            id: eid.clone(),
            session_id: Some(sid),
            namespace: "personal".into(),
            kind: NodeKind::new("episode"),
            body: serde_json::json!({"text": "graph text"}),
            vector_id: None,
            weight: 1.0,
            created_at: chrono::Utc::now(),
        };
        g.add_node(&n).unwrap();

        let r = DualMemoryReader::dual(legacy, g);
        let got = r.read_by_id(&eid).unwrap().expect("found");
        assert!(got.is_graph(), "graph must win when both exist");
    }

    #[test]
    fn dual_reader_falls_back_to_legacy_when_graph_misses() {
        let pool = pool();
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool.clone()));
        let legacy = Arc::new(EpisodicStore::new(pool));
        let sid = legacy.create_session("test").unwrap();
        let eid = legacy
            .store_episode(&sid, "user", "only in legacy", 0.5, None, None)
            .unwrap();
        let r = DualMemoryReader::dual(legacy, g);
        let got = r.read_by_id(&eid).unwrap().expect("found");
        assert!(got.is_legacy(), "must fall back to legacy on graph miss");
    }

    #[test]
    fn dual_reader_returns_none_when_neither_has_id() {
        let pool = pool();
        let g: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool.clone()));
        let legacy = Arc::new(EpisodicStore::new(pool));
        let r = DualMemoryReader::dual(legacy, g);
        assert!(r.read_by_id("does-not-exist").unwrap().is_none());
    }
}
