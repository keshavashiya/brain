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

use std::sync::Arc;

use crate::episodic::EpisodicStore;
use crate::graph::{EpisodicGraph, Node};
use crate::Episode;

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

/// Read facade unifying the graph and the legacy episodic store.
/// Cheap to clone — both inner handles are `Arc`-shared.
#[derive(Clone)]
pub struct DualMemoryReader {
    graph: Option<Arc<dyn EpisodicGraph>>,
    legacy: Option<Arc<EpisodicStore>>,
}

impl DualMemoryReader {
    /// Reader that only consults the graph. Useful in tests and on
    /// fresh installs that never had the legacy tables populated.
    pub fn graph_only(graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph: Some(graph),
            legacy: None,
        }
    }

    /// Reader that only consults the legacy store. Used during the
    /// transition while a new SqlitePool hasn't yet been wired to a
    /// graph adapter.
    pub fn legacy_only(legacy: Arc<EpisodicStore>) -> Self {
        Self {
            graph: None,
            legacy: Some(legacy),
        }
    }

    /// The production wiring: prefer the graph, fall back to legacy.
    pub fn dual(legacy: Arc<EpisodicStore>, graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph: Some(graph),
            legacy: Some(legacy),
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Node, NodeKind, SqliteGraph};
    use storage::SqlitePool;

    fn pool() -> SqlitePool {
        SqlitePool::open_memory().expect("memory pool")
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
