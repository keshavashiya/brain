//! Bridge implementation that mirrors terminal lifecycles into a
//! [`hippocampus::EpisodicGraph`].
//!
//! Lives in `signal` because this crate is the natural assembly point
//! for `terminal` + `hippocampus` — keeping the adapter here means
//! `terminal` stays storage-free and `hippocampus` stays
//! terminal-free.
//!
//! Emission per session lifecycle:
//! - **Open** writes two nodes (`tool_call` for the invocation,
//!   `terminal_event` for the open side) plus a `causal_produced`
//!   edge between them.
//! - **Close** writes one node (`terminal_event` for the close side)
//!   plus a second `causal_produced` edge from the open event to the
//!   close event.
//!
//! Net: three nodes + two edges per session.

use std::sync::Arc;

use async_trait::async_trait;
use hippocampus::{Edge, EdgeKind, Embedder, EpisodicGraph, Node, NodeKind};
use identity::Principal;
use storage::RuVectorStore;
use terminal::{MirrorError, TerminalGraphHandles, TerminalGraphSink};

use crate::graph_embed::NodeEmbedder;

/// `TerminalGraphSink` impl backed by an [`EpisodicGraph`].
///
/// When an embedder + vector store are wired (via [`Self::with_embedding`]),
/// every mirrored node is also embedded into the `graph_vec` collection and
/// gets its `vector_id` set — this is what lets graph events participate in
/// ANN recall, not just FTS. Embedding is best-effort: a failure logs a
/// warning and leaves `vector_id` unset rather than dropping the node.
pub struct HippocampusTerminalSink {
    graph: Arc<dyn EpisodicGraph>,
    namespace: String,
    embedder: NodeEmbedder,
}

impl HippocampusTerminalSink {
    pub fn new(graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph,
            namespace: "personal".to_string(),
            embedder: NodeEmbedder::default(),
        }
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Wire the namespace residency policy: when this sink's namespace is
    /// `local_only` and the embedder is remote, mirrored nodes get the
    /// deterministic fallback vector instead of a remote embed.
    pub fn with_residency(mut self, residency: brain::ResidencyPolicy) -> Self {
        self.embedder.set_residency(residency);
        self
    }

    /// Wire the embedding write-path. Mirrored nodes are embedded through
    /// `embedder` and stored in the shared `vectors` store's `graph_vec`
    /// collection (sized to `embedding_dim`).
    pub fn with_embedding(
        mut self,
        embedder: Option<Arc<Embedder>>,
        vectors: RuVectorStore,
        embedding_dim: usize,
    ) -> Self {
        self.embedder
            .set_embedding(embedder, vectors, embedding_dim);
        self
    }
}

fn principal_summary(p: Option<&Principal>) -> serde_json::Value {
    match p {
        Some(p) => serde_json::json!({
            "agent_id": p.agent_id,
            "tier": p.tier.to_string(),
        }),
        None => serde_json::Value::Null,
    }
}

#[async_trait]
impl TerminalGraphSink for HippocampusTerminalSink {
    async fn record_open(
        &self,
        session_id: &str,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        principal: Option<&Principal>,
    ) -> Result<TerminalGraphHandles, MirrorError> {
        let mut tool_call = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({
                "verb": "terminal.open",
                "program": program,
                "args": args,
                "cwd": cwd,
                "session_id": session_id,
                "principal": principal_summary(principal),
            }),
            self.namespace.clone(),
            None,
        );
        let mut open_event = Node::new(
            NodeKind::new("terminal_event"),
            serde_json::json!({
                "phase": "open",
                "session_id": session_id,
                "program": program,
            }),
            self.namespace.clone(),
            None,
        );
        self.embedder.embed_and_link(&mut tool_call).await;
        self.embedder.embed_and_link(&mut open_event).await;
        self.graph
            .add_node(&tool_call)
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        self.graph
            .add_node(&open_event)
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        self.graph
            .add_edge(&Edge::new(
                &tool_call.id,
                &open_event.id,
                EdgeKind::new("causal_produced"),
            ))
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        Ok(TerminalGraphHandles {
            tool_call_node_id: tool_call.id,
            open_event_node_id: open_event.id,
        })
    }

    async fn record_close(
        &self,
        handles: &TerminalGraphHandles,
        session_id: &str,
        exit_code: i32,
        was_killed: bool,
    ) -> Result<(), MirrorError> {
        let mut close_event = Node::new(
            NodeKind::new("terminal_event"),
            serde_json::json!({
                "phase": "close",
                "session_id": session_id,
                "exit_code": exit_code,
                "was_killed": was_killed,
            }),
            self.namespace.clone(),
            None,
        );
        self.embedder.embed_and_link(&mut close_event).await;
        self.graph
            .add_node(&close_event)
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        self.graph
            .add_edge(&Edge::new(
                &handles.open_event_node_id,
                &close_event.id,
                EdgeKind::new("causal_produced"),
            ))
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_embed::GRAPH_VEC;
    use hippocampus::{EmbeddingError, EmbeddingProvider, SqliteGraph};
    use storage::SqlitePool;

    /// Fixed-vector embedder so the write-path is deterministic offline.
    #[derive(Debug)]
    struct FixedEmbedder;

    #[async_trait]
    impl EmbeddingProvider for FixedEmbedder {
        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.1_f32; 384]).collect())
        }
        fn provider_name(&self) -> &str {
            "fixed-test"
        }
    }

    #[tokio::test]
    async fn record_open_embeds_and_links_nodes() {
        let graph: Arc<dyn EpisodicGraph> =
            Arc::new(SqliteGraph::new(SqlitePool::open_memory().unwrap()));
        let dir = tempfile::tempdir().unwrap();
        let ruv = RuVectorStore::open(dir.path(), 384).await.unwrap();
        ruv.ensure_tables().await.unwrap();

        let embedder = Arc::new(Embedder::new(Box::new(FixedEmbedder)));
        let sink = HippocampusTerminalSink::new(graph.clone()).with_embedding(
            Some(embedder),
            ruv.clone(),
            384,
        );

        let handles = sink
            .record_open("sess-1", "ripgrep", &["-n".into()], Some("/tmp"), None)
            .await
            .unwrap();

        // The tool_call node gained a vector_id pointing at itself, and a
        // matching vector landed in graph_vec.
        let node = graph
            .get_node(&handles.tool_call_node_id)
            .unwrap()
            .expect("tool_call node");
        assert_eq!(node.vector_id.as_deref(), Some(node.id.as_str()));
        assert!(ruv.table_count(GRAPH_VEC).await.unwrap() >= 1);
    }

    #[tokio::test]
    async fn record_open_without_embedding_leaves_vector_id_unset() {
        let graph: Arc<dyn EpisodicGraph> =
            Arc::new(SqliteGraph::new(SqlitePool::open_memory().unwrap()));
        let sink = HippocampusTerminalSink::new(graph.clone());
        let handles = sink
            .record_open("sess-1", "ls", &[], None, None)
            .await
            .unwrap();
        let node = graph
            .get_node(&handles.tool_call_node_id)
            .unwrap()
            .expect("tool_call node");
        assert!(node.vector_id.is_none());
    }
}
