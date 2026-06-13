//! Embedding write-path for semantic capability retrieval (S5).
//!
//! Two surfaces share one residency-aware embed routine:
//! - **Descriptor embedding** ([`CapabilityEmbedder::embed_descriptors`]):
//!   populates each [`ToolDescriptor::embedding`] from its
//!   [`embedding_text`](intent::ToolDescriptor::embedding_text) at registration
//!   time, so the router / tool-loop / index can score by cosine.
//! - **Query embedding** (the [`intent::QueryEmbedder`] impl): embeds the
//!   user's surface text so the router can add a semantic term in `resolve`.
//!
//! Best-effort throughout — a failed embed leaves `embedding` unset (descriptor
//! side) or returns `None` (query side), and scoring silently falls back to
//! lexical-only. Residency is honoured on the query side: a `local_only`
//! namespace never reaches a remote embedder (deterministic fallback instead),
//! mirroring [`crate::graph_embed`]. Tool descriptions are catalog metadata,
//! not namespaced user data, so the descriptor side embeds unconditionally.

use std::sync::Arc;

use hippocampus::Embedder;
use intent::ToolDescriptor;

/// Residency-aware embedder shared by the descriptor and query surfaces. Holds
/// the concrete [`Embedder`] so the schema crate (`intent`) stays embedder-free
/// — it only sees the [`intent::QueryEmbedder`] trait this type satisfies.
#[derive(Clone)]
pub struct CapabilityEmbedder {
    embedder: Arc<Embedder>,
    embedding_dim: usize,
    residency: brain::ResidencyPolicy,
}

impl CapabilityEmbedder {
    pub fn new(
        embedder: Arc<Embedder>,
        embedding_dim: usize,
        residency: brain::ResidencyPolicy,
    ) -> Self {
        Self {
            embedder,
            embedding_dim,
            residency,
        }
    }

    /// Embed catalog text (a tool's projection). No residency gate — tool
    /// descriptions are catalog metadata, not namespaced user data. Returns
    /// `None` on any embed error so the caller leaves `embedding` unset.
    async fn embed_catalog(&self, text: &str) -> Option<Vec<f32>> {
        match self.embedder.embed(text).await {
            Ok(v) => Some(hippocampus::embedding::sanitize_embedding(
                v,
                self.embedding_dim,
                text,
            )),
            Err(e) => {
                tracing::warn!(error = %e, "capability descriptor embed failed; leaving unembedded");
                None
            }
        }
    }

    /// Populate `.embedding` on each descriptor from its
    /// [`embedding_text`](intent::ToolDescriptor::embedding_text). Descriptors
    /// that already carry an embedding are left untouched (idempotent re-runs).
    pub async fn embed_descriptors(&self, descriptors: &mut [ToolDescriptor]) {
        for d in descriptors.iter_mut() {
            if d.embedding.is_some() {
                continue;
            }
            let text = d.embedding_text();
            if text.trim().is_empty() {
                continue;
            }
            d.embedding = self.embed_catalog(&text).await;
        }
    }
}

#[async_trait::async_trait]
impl intent::DescriptorEmbedder for CapabilityEmbedder {
    async fn embed_descriptor(&self, text: &str) -> Option<Vec<f32>> {
        self.embed_catalog(text).await
    }
}

#[async_trait::async_trait]
impl intent::QueryEmbedder for CapabilityEmbedder {
    async fn embed_query(&self, text: &str, namespace: &str) -> Option<Vec<f32>> {
        // Residency: a local-only namespace never reaches a remote embedder.
        // The deterministic fallback keeps the cosine term *self-consistent*
        // with descriptors embedded under the same fallback would be — but
        // descriptors embed via the real model, so a remote embedder under a
        // local-only query simply yields a vector that won't match well. That
        // is the correct trade: privacy first, semantic recall degrades to
        // lexical-only for that turn.
        if !self.embedder.is_local() && self.residency.is_local_only(namespace) {
            return Some(hippocampus::embedding::deterministic_fallback_embedding(
                text,
                self.embedding_dim,
            ));
        }
        match self.embedder.embed(text).await {
            Ok(v) => Some(hippocampus::embedding::sanitize_embedding(
                v,
                self.embedding_dim,
                text,
            )),
            Err(e) => {
                tracing::debug!(error = %e, "capability query embed failed; lexical-only this turn");
                None
            }
        }
    }
}
