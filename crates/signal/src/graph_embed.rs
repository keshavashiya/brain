//! Shared embedding write-path for the graph mirrors.
//!
//! Both the terminal mirror and the observation mirror write nodes into the
//! episodic graph and want those nodes to participate in ANN recall, not just
//! FTS. This module holds the one copy of that write-path: flatten the node
//! to a text projection, embed it (respecting namespace residency), store the
//! vector in the shared `graph_vec` collection, and link the node via
//! `vector_id`. Embedding is best-effort throughout — a failure logs a
//! warning and leaves `vector_id` unset rather than dropping the node.

use std::sync::Arc;

use hippocampus::{Embedder, Node};
use storage::RuVectorStore;

/// RuVector collection holding graph-node embeddings. Each entry's id is
/// the node id, so recall can hydrate ANN hits back through the graph.
pub(crate) const GRAPH_VEC: &str = "graph_vec";

/// Optional embedding write-path shared by the graph mirrors. Unwired
/// (the default), [`Self::embed_and_link`] is a no-op.
#[derive(Default)]
pub(crate) struct NodeEmbedder {
    embedder: Option<Arc<Embedder>>,
    vectors: Option<RuVectorStore>,
    embedding_dim: usize,
    residency: brain::ResidencyPolicy,
}

impl NodeEmbedder {
    /// Wire the embedding write-path: nodes are embedded through `embedder`
    /// and stored in the shared `vectors` store's `graph_vec` collection
    /// (sized to `embedding_dim`).
    pub(crate) fn set_embedding(
        &mut self,
        embedder: Option<Arc<Embedder>>,
        vectors: RuVectorStore,
        embedding_dim: usize,
    ) {
        self.embedder = embedder;
        self.vectors = Some(vectors);
        self.embedding_dim = embedding_dim;
    }

    /// Wire the namespace residency policy: when a node's namespace is
    /// `local_only` and the embedder is remote, the node gets the
    /// deterministic fallback vector instead of a remote embed.
    pub(crate) fn set_residency(&mut self, residency: brain::ResidencyPolicy) {
        self.residency = residency;
    }

    /// Best-effort embed of `node` into `graph_vec`, setting `node.vector_id`
    /// on success. No-op (leaves `vector_id` as-is) when the embedder or
    /// vector store is unwired, or on any embed/store error.
    pub(crate) async fn embed_and_link(&self, node: &mut Node) {
        let (Some(embedder), Some(vectors)) = (&self.embedder, &self.vectors) else {
            return;
        };
        let text = node_text(node);
        // Residency: a local-only namespace never reaches a remote
        // embedder — the deterministic fallback keeps the ANN link
        // functional without the egress.
        let vector = if !embedder.is_local() && self.residency.is_local_only(&node.namespace) {
            hippocampus::embedding::deterministic_fallback_embedding(&text, self.embedding_dim)
        } else {
            match embedder.embed(&text).await {
                Ok(v) => hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, &text),
                Err(e) => {
                    tracing::warn!(node_id = %node.id, "graph node embed failed, skipping ANN link: {e}");
                    return;
                }
            }
        };
        if let Err(e) = vectors
            .add_vectors(
                GRAPH_VEC,
                vec![node.id.clone()],
                vec![text],
                vec![vector],
                vec![node.created_at.to_rfc3339()],
                "graph",
            )
            .await
        {
            tracing::warn!(node_id = %node.id, "graph_vec insert failed, skipping ANN link: {e}");
            return;
        }
        node.vector_id = Some(node.id.clone());
    }
}

/// Flatten a node into a compact text projection for embedding: the node
/// kind followed by every scalar value in its JSON body, space-joined.
/// Keeps the embedded text close to how a user would phrase a recall
/// query ("terminal.open ripgrep …") rather than raw JSON punctuation.
pub(crate) fn node_text(node: &Node) -> String {
    let mut parts = vec![node.kind.as_str().to_string()];
    collect_scalars(&node.body, &mut parts);
    parts.join(" ")
}

fn collect_scalars(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Number(n) => out.push(n.to_string()),
        serde_json::Value::Bool(b) => out.push(b.to_string()),
        serde_json::Value::Array(a) => a.iter().for_each(|e| collect_scalars(e, out)),
        serde_json::Value::Object(o) => o.values().for_each(|e| collect_scalars(e, out)),
        serde_json::Value::Null => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hippocampus::NodeKind;

    #[test]
    fn node_text_flattens_scalars_with_kind() {
        let n = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({"verb": "terminal.open", "program": "rg", "args": ["-n", "foo"]}),
            "personal",
            None,
        );
        let text = node_text(&n);
        assert!(text.starts_with("tool_call"));
        for term in ["terminal.open", "rg", "-n", "foo"] {
            assert!(text.contains(term), "missing {term} in {text}");
        }
    }
}
