//! Episodic graph store — typed nodes + typed edges backed by the
//! `nodes` / `edges` tables.
//!
//! Coexists with [`crate::EpisodicStore`] (the legacy flat conversation
//! log); see [`crate::dual_memory`] for the reconciliation layer. Reads
//! should prefer the graph if a node exists for the requested id, else
//! fall back to the legacy store.

use std::str::FromStr;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use storage::SqlitePool;
use thiserror::Error;
use uuid::Uuid;

/// Errors from the graph layer.
#[derive(Debug, Error)]
pub enum GraphError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] storage::sqlite::SqliteError),
    #[error("rusqlite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),
    #[error("invalid node body json: {0}")]
    Body(#[from] serde_json::Error),
    #[error("invalid timestamp: {0}")]
    Timestamp(String),
}

/// Opaque newtype wrapper around the `node_kind` column. Brain stays
/// schemaless on the wire — callers pick their own taxonomy
/// (`tool_call`, `terminal_event`, `signal`, `fact`, …) and the graph
/// preserves it verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeKind(pub String);

impl NodeKind {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for NodeKind {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Opaque newtype wrapper around the `edge_kind` column. Same
/// rationale as [`NodeKind`] — examples: `causal_produced`,
/// `references`, `superseded_by`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeKind(pub String);

impl EdgeKind {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for EdgeKind {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// One row of the `nodes` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub session_id: Option<String>,
    pub namespace: String,
    pub kind: NodeKind,
    pub body: serde_json::Value,
    /// Embedding handle (when promoted to the vector store). `None`
    /// during the v1.0 dual-store period if no embedding was generated.
    pub vector_id: Option<String>,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
}

impl Node {
    /// Build a node with a fresh UUID and `created_at = now`. Default
    /// weight is `1.0`; the compactor decays it over time.
    pub fn new(
        kind: NodeKind,
        body: serde_json::Value,
        namespace: impl Into<String>,
        session_id: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            namespace: namespace.into(),
            kind,
            body,
            vector_id: None,
            weight: 1.0,
            created_at: Utc::now(),
        }
    }
}

/// One row of the `edges` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub src: String,
    pub dst: String,
    pub kind: EdgeKind,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
}

impl Edge {
    pub fn new(src: impl Into<String>, dst: impl Into<String>, kind: EdgeKind) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            kind,
            weight: 1.0,
            created_at: Utc::now(),
        }
    }
}

/// One full-text hit from [`EpisodicGraph::search_text`]. Carries the
/// BM25 `rank` (lower = better, FTS5 convention) plus enough node
/// metadata for the recall engine to build a `Memory` and fold the hit
/// into RRF fusion without a second lookup.
#[derive(Debug, Clone)]
pub struct GraphHit {
    pub id: String,
    /// The indexed text (the node's `body_json`).
    pub text: String,
    /// FTS5 BM25 rank — lower is a better match.
    pub rank: f64,
    pub namespace: String,
    pub node_kind: NodeKind,
    pub created_at: DateTime<Utc>,
    pub weight: f32,
    pub vector_id: Option<String>,
}

/// Repository surface for the graph store. Sync (matches
/// [`crate::EpisodicStore`]) — `tokio` callers wrap in
/// `spawn_blocking` if they need to avoid blocking the reactor.
pub trait EpisodicGraph: Send + Sync {
    fn add_node(&self, node: &Node) -> Result<(), GraphError>;
    fn add_edge(&self, edge: &Edge) -> Result<(), GraphError>;
    /// Full-text search over node bodies via the `nodes_fts` index
    /// (BM25). Optionally scoped to a namespace (exact or `ns/%`
    /// sub-namespace prefix, matching the episodic store). Returns up
    /// to `limit` hits ordered by rank (best first). An empty or
    /// punctuation-only query yields an empty result set.
    fn search_text(
        &self,
        query: &str,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<GraphHit>, GraphError>;
    fn get_node(&self, id: &str) -> Result<Option<Node>, GraphError>;
    /// Outgoing neighbors of `id`. Each row is `(edge, destination
    /// node)` so callers can render provenance without a second
    /// lookup.
    fn neighbors(&self, id: &str) -> Result<Vec<(Edge, Node)>, GraphError>;
    /// Incoming neighbors of `id` (`(edge, source node)`).
    fn incoming(&self, id: &str) -> Result<Vec<(Edge, Node)>, GraphError>;
    /// Find a directed path from `src` to `dst` up to `max_hops` long.
    /// Returns the node-id chain (inclusive on both ends) or `None`
    /// if no path exists within the limit.
    fn path(&self, src: &str, dst: &str, max_hops: u32) -> Result<Option<Vec<String>>, GraphError>;
    /// Enumerate every node — used by the compactor. Real impls may
    /// stream eventually; v1.0 returns a `Vec` because compaction
    /// runs on a cron and the graph stays modest.
    fn list_all_nodes(&self) -> Result<Vec<Node>, GraphError>;
    /// Update one node's weight in place. The compactor's primary
    /// write path.
    fn update_weight(&self, id: &str, weight: f32) -> Result<(), GraphError>;
    /// Remove one node. Edges cascade via the migration v20 FK.
    /// Returns `true` if a row was deleted. The associated embedding
    /// (referenced by `vector_id`) is *not* touched — vector
    /// reclamation is a separate maintenance task.
    fn delete_node(&self, id: &str) -> Result<bool, GraphError>;
}

/// SQLite-backed [`EpisodicGraph`] over the shared pool.
pub struct SqliteGraph {
    db: SqlitePool,
}

impl SqliteGraph {
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.db
    }
}

fn parse_ts(s: &str) -> Result<DateTime<Utc>, GraphError> {
    // Writes go through to_rfc3339; reads accept both that and the
    // SQLite `datetime('now')` default for direct SQL inserts.
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(naive.and_utc());
    }
    Err(GraphError::Timestamp(s.to_string()))
}

fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<NodeRow> {
    Ok(NodeRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        namespace: row.get(2)?,
        node_kind: row.get(3)?,
        body_json: row.get(4)?,
        vector_id: row.get(5)?,
        weight: row.get::<_, f64>(6)? as f32,
        created_at: row.get(7)?,
    })
}

struct NodeRow {
    id: String,
    session_id: Option<String>,
    namespace: String,
    node_kind: String,
    body_json: String,
    vector_id: Option<String>,
    weight: f32,
    created_at: String,
}

/// Raw `search_text` row — timestamp stays a string until `parse_ts`
/// runs outside the conn closure (same pattern as `NodeRow`).
struct HitRow {
    id: String,
    text: String,
    rank: f64,
    namespace: String,
    node_kind: String,
    created_at: String,
    weight: f64,
    vector_id: Option<String>,
}

impl NodeRow {
    fn into_node(self) -> Result<Node, GraphError> {
        Ok(Node {
            id: self.id,
            session_id: self.session_id,
            namespace: self.namespace,
            kind: NodeKind(self.node_kind),
            body: serde_json::Value::from_str(&self.body_json)?,
            vector_id: self.vector_id,
            weight: self.weight,
            created_at: parse_ts(&self.created_at)?,
        })
    }
}

struct EdgeRow {
    src: String,
    dst: String,
    edge_kind: String,
    weight: f32,
    created_at: String,
}

impl EdgeRow {
    fn into_edge(self) -> Result<Edge, GraphError> {
        Ok(Edge {
            src: self.src,
            dst: self.dst,
            kind: EdgeKind(self.edge_kind),
            weight: self.weight,
            created_at: parse_ts(&self.created_at)?,
        })
    }
}

impl EpisodicGraph for SqliteGraph {
    fn add_node(&self, node: &Node) -> Result<(), GraphError> {
        let body = serde_json::to_string(&node.body)?;
        let created = node.created_at.to_rfc3339();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO nodes
                   (id, session_id, namespace, node_kind, body_json,
                    vector_id, weight, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    node.id,
                    node.session_id,
                    node.namespace,
                    node.kind.0,
                    body,
                    node.vector_id,
                    node.weight as f64,
                    created,
                ],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    fn add_edge(&self, edge: &Edge) -> Result<(), GraphError> {
        let created = edge.created_at.to_rfc3339();
        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO edges
                   (src_id, dst_id, edge_kind, weight, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![edge.src, edge.dst, edge.kind.0, edge.weight as f64, created,],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    fn search_text(
        &self,
        query: &str,
        limit: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<GraphHit>, GraphError> {
        let sanitized = crate::episodic::sanitize_fts5_query(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        // Raw rows (timestamp still a string) come out of the conn
        // closure; `parse_ts` runs outside since its `GraphError`
        // doesn't fit the closure's `SqliteError` channel — same split
        // `get_node` uses with `NodeRow::into_node`.
        let raw: Vec<HitRow> = self.db.with_conn(|conn| {
            // Mirrors EpisodicStore::search_bm25 — MATCH on the FTS
            // index, join back to `nodes` via the shared rowid, optional
            // namespace scope (exact or `ns/%` sub-namespace), order by
            // BM25 rank (ascending = best).
            let mut sql = String::from(
                "SELECT n.id, f.text, f.rank, n.namespace, n.node_kind,
                        n.created_at, n.weight, n.vector_id
                 FROM nodes_fts f
                 JOIN nodes n ON n.rowid = f.rowid
                 WHERE nodes_fts MATCH ?1",
            );
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(sanitized)];

            if let Some(ns) = namespace {
                sql.push_str(&format!(
                    " AND (n.namespace = ?{} OR n.namespace LIKE ?{})",
                    params.len() + 1,
                    params.len() + 2
                ));
                params.push(Box::new(ns.to_string()));
                params.push(Box::new(format!("{ns}/%")));
            }

            sql.push_str(&format!(" ORDER BY f.rank LIMIT ?{}", params.len() + 1));
            params.push(Box::new(limit as i64));

            let mut stmt = conn.prepare(&sql)?;
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                params.iter().map(|p| p.as_ref()).collect();
            let rows = stmt
                .query_map(param_refs.as_slice(), |row| {
                    Ok(HitRow {
                        id: row.get(0)?,
                        text: row.get(1)?,
                        rank: row.get(2)?,
                        namespace: row.get(3)?,
                        node_kind: row.get(4)?,
                        created_at: row.get(5)?,
                        weight: row.get(6)?,
                        vector_id: row.get(7)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;

        raw.into_iter()
            .map(|r| {
                Ok(GraphHit {
                    id: r.id,
                    text: r.text,
                    rank: r.rank,
                    namespace: r.namespace,
                    node_kind: NodeKind(r.node_kind),
                    created_at: parse_ts(&r.created_at)?,
                    weight: r.weight as f32,
                    vector_id: r.vector_id,
                })
            })
            .collect()
    }

    fn get_node(&self, id: &str) -> Result<Option<Node>, GraphError> {
        let row: Option<NodeRow> = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, namespace, node_kind, body_json,
                        vector_id, weight, created_at
                 FROM nodes WHERE id = ?1",
            )?;
            let mut rows = stmt.query([id])?;
            if let Some(row) = rows.next()? {
                Ok(Some(row_to_node(row)?))
            } else {
                Ok(None)
            }
        })?;
        row.map(NodeRow::into_node).transpose()
    }

    fn neighbors(&self, id: &str) -> Result<Vec<(Edge, Node)>, GraphError> {
        let raw: Vec<(EdgeRow, NodeRow)> = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT e.src_id, e.dst_id, e.edge_kind, e.weight, e.created_at,
                        n.id, n.session_id, n.namespace, n.node_kind, n.body_json,
                        n.vector_id, n.weight, n.created_at
                 FROM edges e JOIN nodes n ON n.id = e.dst_id
                 WHERE e.src_id = ?1
                 ORDER BY e.created_at",
            )?;
            let mut rows = stmt.query([id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                let edge = EdgeRow {
                    src: row.get(0)?,
                    dst: row.get(1)?,
                    edge_kind: row.get(2)?,
                    weight: row.get::<_, f64>(3)? as f32,
                    created_at: row.get(4)?,
                };
                let node = NodeRow {
                    id: row.get(5)?,
                    session_id: row.get(6)?,
                    namespace: row.get(7)?,
                    node_kind: row.get(8)?,
                    body_json: row.get(9)?,
                    vector_id: row.get(10)?,
                    weight: row.get::<_, f64>(11)? as f32,
                    created_at: row.get(12)?,
                };
                out.push((edge, node));
            }
            Ok(out)
        })?;
        raw.into_iter()
            .map(|(e, n)| Ok((e.into_edge()?, n.into_node()?)))
            .collect()
    }

    fn incoming(&self, id: &str) -> Result<Vec<(Edge, Node)>, GraphError> {
        let raw: Vec<(EdgeRow, NodeRow)> = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT e.src_id, e.dst_id, e.edge_kind, e.weight, e.created_at,
                        n.id, n.session_id, n.namespace, n.node_kind, n.body_json,
                        n.vector_id, n.weight, n.created_at
                 FROM edges e JOIN nodes n ON n.id = e.src_id
                 WHERE e.dst_id = ?1
                 ORDER BY e.created_at",
            )?;
            let mut rows = stmt.query([id])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                let edge = EdgeRow {
                    src: row.get(0)?,
                    dst: row.get(1)?,
                    edge_kind: row.get(2)?,
                    weight: row.get::<_, f64>(3)? as f32,
                    created_at: row.get(4)?,
                };
                let node = NodeRow {
                    id: row.get(5)?,
                    session_id: row.get(6)?,
                    namespace: row.get(7)?,
                    node_kind: row.get(8)?,
                    body_json: row.get(9)?,
                    vector_id: row.get(10)?,
                    weight: row.get::<_, f64>(11)? as f32,
                    created_at: row.get(12)?,
                };
                out.push((edge, node));
            }
            Ok(out)
        })?;
        raw.into_iter()
            .map(|(e, n)| Ok((e.into_edge()?, n.into_node()?)))
            .collect()
    }

    fn list_all_nodes(&self) -> Result<Vec<Node>, GraphError> {
        let rows: Vec<NodeRow> = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, namespace, node_kind, body_json,
                        vector_id, weight, created_at
                 FROM nodes",
            )?;
            let mut rows = stmt.query([])?;
            let mut out = Vec::new();
            while let Some(row) = rows.next()? {
                out.push(row_to_node(row)?);
            }
            Ok(out)
        })?;
        rows.into_iter().map(NodeRow::into_node).collect()
    }

    fn update_weight(&self, id: &str, weight: f32) -> Result<(), GraphError> {
        self.db.with_conn(|conn| {
            conn.execute(
                "UPDATE nodes SET weight = ?1 WHERE id = ?2",
                rusqlite::params![weight as f64, id],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    fn delete_node(&self, id: &str) -> Result<bool, GraphError> {
        // Enable FK so edges cascade per migration v20. Pragmas are
        // per-connection and the pool isn't pinned, so set it inline.
        let deleted = self.db.with_conn(|conn| {
            conn.execute("PRAGMA foreign_keys = ON", [])?;
            let n = conn.execute("DELETE FROM nodes WHERE id = ?1", [id])?;
            Ok(n > 0)
        })?;
        Ok(deleted)
    }

    fn path(&self, src: &str, dst: &str, max_hops: u32) -> Result<Option<Vec<String>>, GraphError> {
        // BFS via recursive CTE. The `path` column is a separator-
        // delimited node-id chain; `instr` prevents cycles. We pick
        // the shortest matching path by depth.
        let max_depth = max_hops.max(1) as i64;
        let result: Option<String> = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "WITH RECURSIVE walk(node_id, depth, path) AS (
                    SELECT ?1, 0, ?1
                    UNION ALL
                    SELECT e.dst_id, w.depth + 1, w.path || '\u{1f}' || e.dst_id
                    FROM edges e JOIN walk w ON e.src_id = w.node_id
                    WHERE w.depth < ?2
                      AND instr(w.path || '\u{1f}', e.dst_id || '\u{1f}') = 0
                 )
                 SELECT path FROM walk WHERE node_id = ?3
                 ORDER BY depth LIMIT 1",
            )?;
            let mut rows = stmt.query(rusqlite::params![src, max_depth, dst])?;
            if let Some(row) = rows.next()? {
                let p: String = row.get(0)?;
                Ok(Some(p))
            } else {
                Ok(None)
            }
        })?;
        Ok(result.map(|p| p.split('\u{1f}').map(|s| s.to_string()).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SqliteGraph {
        SqliteGraph::new(SqlitePool::open_memory().expect("memory pool"))
    }

    fn node(kind: &str, name: &str) -> Node {
        Node::new(
            NodeKind::new(kind),
            serde_json::json!({"name": name}),
            "personal",
            None,
        )
    }

    #[test]
    fn search_text_finds_node_by_body_term() {
        let g = store();
        let mut n = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"verb": "terminal.open", "program": "ripgrep"}),
            "personal",
            None,
        );
        n.weight = 0.5;
        g.add_node(&n).unwrap();
        // unrelated node that must not match.
        g.add_node(&node("terminal_event", "cargo")).unwrap();

        let hits = g.search_text("ripgrep", 10, None).unwrap();
        assert_eq!(hits.len(), 1, "only the ripgrep node should match");
        assert_eq!(hits[0].id, n.id);
        assert_eq!(hits[0].node_kind.as_str(), "tool_call");
        assert!((hits[0].weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn search_text_respects_namespace_scope() {
        let g = store();
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

        let hits = g.search_text("deploy", 10, Some("work")).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, work.id);
    }

    #[test]
    fn search_text_index_stays_in_sync_on_delete() {
        let g = store();
        let n = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"program": "ephemeral-tool"}),
            "personal",
            None,
        );
        g.add_node(&n).unwrap();
        assert_eq!(g.search_text("ephemeral-tool", 10, None).unwrap().len(), 1);
        assert!(g.delete_node(&n.id).unwrap());
        assert!(
            g.search_text("ephemeral-tool", 10, None)
                .unwrap()
                .is_empty(),
            "delete trigger must drop the FTS row"
        );
    }

    #[test]
    fn search_text_empty_query_returns_empty() {
        let g = store();
        g.add_node(&node("tool_call", "x")).unwrap();
        assert!(g.search_text("   ", 10, None).unwrap().is_empty());
        assert!(g.search_text("!@#$", 10, None).unwrap().is_empty());
    }

    #[test]
    fn round_trip_node() {
        let g = store();
        let mut n = node("tool_call", "echo");
        n.weight = 0.75;
        n.vector_id = Some("vec-123".into());
        g.add_node(&n).unwrap();
        let got = g.get_node(&n.id).unwrap().expect("node should exist");
        assert_eq!(got.id, n.id);
        assert_eq!(got.kind, n.kind);
        assert_eq!(got.body, n.body);
        assert!((got.weight - 0.75).abs() < 1e-6);
        assert_eq!(got.vector_id.as_deref(), Some("vec-123"));
    }

    #[test]
    fn get_node_returns_none_for_missing() {
        let g = store();
        assert!(g.get_node("nope").unwrap().is_none());
    }

    #[test]
    fn neighbors_returns_outgoing_edges_with_destination_nodes() {
        let g = store();
        let a = node("tool_call", "a");
        let b = node("tool_event", "b");
        let c = node("tool_event", "c");
        g.add_node(&a).unwrap();
        g.add_node(&b).unwrap();
        g.add_node(&c).unwrap();
        g.add_edge(&Edge::new(&a.id, &b.id, EdgeKind::new("causal_produced")))
            .unwrap();
        g.add_edge(&Edge::new(&a.id, &c.id, EdgeKind::new("references")))
            .unwrap();

        let nb = g.neighbors(&a.id).unwrap();
        assert_eq!(nb.len(), 2);
        let kinds: Vec<&str> = nb.iter().map(|(e, _)| e.kind.as_str()).collect();
        assert!(kinds.contains(&"causal_produced"));
        assert!(kinds.contains(&"references"));
        let dst_ids: std::collections::HashSet<&str> =
            nb.iter().map(|(_, n)| n.id.as_str()).collect();
        assert!(dst_ids.contains(b.id.as_str()));
        assert!(dst_ids.contains(c.id.as_str()));
    }

    #[test]
    fn incoming_is_the_reverse_view() {
        let g = store();
        let a = node("t", "a");
        let b = node("t", "b");
        g.add_node(&a).unwrap();
        g.add_node(&b).unwrap();
        g.add_edge(&Edge::new(&a.id, &b.id, EdgeKind::new("k")))
            .unwrap();

        let inb = g.incoming(&b.id).unwrap();
        assert_eq!(inb.len(), 1);
        assert_eq!(inb[0].1.id, a.id);
        assert!(g.incoming(&a.id).unwrap().is_empty());
    }

    #[test]
    fn path_finds_chain_through_three_nodes() {
        let g = store();
        let a = node("t", "a");
        let b = node("t", "b");
        let c = node("t", "c");
        g.add_node(&a).unwrap();
        g.add_node(&b).unwrap();
        g.add_node(&c).unwrap();
        let k = EdgeKind::new("rel");
        g.add_edge(&Edge::new(&a.id, &b.id, k.clone())).unwrap();
        g.add_edge(&Edge::new(&b.id, &c.id, k.clone())).unwrap();

        let p = g.path(&a.id, &c.id, 5).unwrap().expect("path exists");
        assert_eq!(p, vec![a.id.clone(), b.id.clone(), c.id.clone()]);
    }

    #[test]
    fn path_respects_max_hops() {
        let g = store();
        let a = node("t", "a");
        let b = node("t", "b");
        let c = node("t", "c");
        g.add_node(&a).unwrap();
        g.add_node(&b).unwrap();
        g.add_node(&c).unwrap();
        let k = EdgeKind::new("rel");
        g.add_edge(&Edge::new(&a.id, &b.id, k.clone())).unwrap();
        g.add_edge(&Edge::new(&b.id, &c.id, k.clone())).unwrap();
        // Two hops required; cap at one → no path returned.
        assert!(g.path(&a.id, &c.id, 1).unwrap().is_none());
    }

    #[test]
    fn path_returns_none_when_disconnected() {
        let g = store();
        let a = node("t", "a");
        let b = node("t", "b");
        g.add_node(&a).unwrap();
        g.add_node(&b).unwrap();
        assert!(g.path(&a.id, &b.id, 5).unwrap().is_none());
    }

    #[test]
    fn path_handles_cycles_without_diverging() {
        let g = store();
        let a = node("t", "a");
        let b = node("t", "b");
        g.add_node(&a).unwrap();
        g.add_node(&b).unwrap();
        let k = EdgeKind::new("rel");
        // Cycle: a→b and b→a. Path a→b stays length 2 (not infinite).
        g.add_edge(&Edge::new(&a.id, &b.id, k.clone())).unwrap();
        g.add_edge(&Edge::new(&b.id, &a.id, k)).unwrap();
        let p = g.path(&a.id, &b.id, 5).unwrap().expect("path exists");
        assert_eq!(p, vec![a.id, b.id]);
    }
}
