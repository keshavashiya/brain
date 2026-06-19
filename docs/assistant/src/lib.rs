//! In-browser retrieval engine for the Brain OS docs assistant.
//!
//! The browser embeds the visitor's question with the same sentence model used
//! at build time, then hands the query vector here. This engine ranks the
//! pre-embedded doc chunks and returns the best matching **sections, verbatim**
//! — it never generates prose, so it cannot state anything that isn't already
//! in the docs. Ranking reuses the engine's own [`synapse`] primitives:
//! [`synapse::cosine_similarity`] for semantic closeness and
//! [`synapse::lexical_overlap`] for literal keyword hits, combined into one
//! hybrid score so an exact phrase still surfaces when the embedding is weak.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Weight on the semantic (cosine) term. Mirrors the router's bias toward a
/// strong embedding match while letting a literal keyword hit break ties.
const SEMANTIC_WEIGHT: f32 = 1.0;
/// Weight on the lexical (keyword-overlap) term.
const LEXICAL_WEIGHT: f32 = 0.5;

/// One indexed doc section, as emitted by the build-time indexer.
#[derive(Debug, Clone, Deserialize)]
pub struct Chunk {
    /// Page/section title for display, e.g. "Operations · Security".
    pub title: String,
    /// Deep link to the section anchor, e.g. "operations/security.html#data-residency".
    pub url: String,
    /// The verbatim section prose (what gets shown to the user).
    pub text: String,
    /// Pre-computed embedding of `text`, same model/dim as the query vector.
    pub vector: Vec<f32>,
}

/// The deserialized index file shipped as a static asset.
#[derive(Debug, Clone, Deserialize)]
pub struct Index {
    /// Model id the chunks were embedded with — the browser must embed the
    /// query with the same one or cosine is meaningless.
    pub model: String,
    /// Embedding dimensionality.
    pub dim: usize,
    pub chunks: Vec<Chunk>,
}

/// A ranked result handed back to the widget.
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f32,
}

impl Index {
    /// Rank chunks for a query, highest score first, capped at `k`.
    ///
    /// Pure and wasm-free so it is unit-testable on the host. `query_vec` is the
    /// browser-side embedding of `query`; when it is empty or mis-dimensioned
    /// the cosine term drops out and ranking falls back to lexical overlap.
    pub fn rank(&self, query: &str, query_vec: &[f32], k: usize) -> Vec<Hit> {
        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .map(|c| {
                let semantic = synapse::cosine_similarity(query_vec, &c.vector);
                let lexical = synapse::lexical_overlap(query, &c.text);
                (semantic * SEMANTIC_WEIGHT + lexical * LEXICAL_WEIGHT, c)
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only the best-scoring section per page, so a single page can't
        // fill the list with near-duplicate chunks. `page` = the url without its
        // `#fragment`; chunks are already sorted by score, so the first one seen
        // for a page is its strongest.
        let mut seen_pages = std::collections::HashSet::new();
        scored
            .into_iter()
            .filter(|(_, c)| {
                let page = c.url.split('#').next().unwrap_or(&c.url);
                seen_pages.insert(page.to_string())
            })
            .take(k)
            .map(|(score, c)| Hit {
                title: c.title.clone(),
                url: c.url.clone(),
                snippet: snippet(&c.text),
                score,
            })
            .collect()
    }
}

/// Trim a section down to a readable preview without splitting mid-word.
fn snippet(text: &str) -> String {
    const MAX: usize = 320;
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(MAX).collect();
    if let Some(idx) = out.rfind(char::is_whitespace) {
        out.truncate(idx);
    }
    out.push('…');
    out
}

// ─── WASM surface ────────────────────────────────────────────────────────────

/// The browser-facing handle. Construct once from the index JSON, then call
/// [`DocsEngine::search`] per query.
#[wasm_bindgen]
pub struct DocsEngine {
    index: Index,
}

#[wasm_bindgen]
impl DocsEngine {
    /// Parse the index JSON shipped alongside the site.
    #[wasm_bindgen(constructor)]
    pub fn new(index_json: &str) -> Result<DocsEngine, JsValue> {
        let index: Index = serde_json::from_str(index_json)
            .map_err(|e| JsValue::from_str(&format!("invalid index json: {e}")))?;
        Ok(DocsEngine { index })
    }

    /// Model id the index was built with, so the widget can load the matching
    /// embedding model.
    #[wasm_bindgen(getter)]
    pub fn model(&self) -> String {
        self.index.model.clone()
    }

    /// Embedding dimensionality expected for query vectors.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.index.dim
    }

    /// Rank doc sections for a query. `query_vec` is a `Float32Array` of the
    /// query embedding; returns `[{title, url, snippet, score}]`.
    pub fn search(&self, query: &str, query_vec: &[f32], k: usize) -> Result<JsValue, JsValue> {
        let hits = self.index.rank(query, query_vec, k);
        serde_wasm_bindgen::to_value(&hits).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx() -> Index {
        Index {
            model: "test".into(),
            dim: 2,
            chunks: vec![
                Chunk {
                    title: "Security · Data residency".into(),
                    url: "operations/security.html#residency".into(),
                    text: "Your data never leaves the device; embeddings stay local.".into(),
                    vector: vec![1.0, 0.0],
                },
                Chunk {
                    title: "Install".into(),
                    url: "install/README.html".into(),
                    text: "Download the binary and run brain serve to start.".into(),
                    vector: vec![0.0, 1.0],
                },
            ],
        }
    }

    #[test]
    fn semantic_match_ranks_first() {
        // Query vector points at the residency chunk; words don't overlap.
        let hits = idx().rank("keep information off the cloud", &[1.0, 0.0], 5);
        assert_eq!(hits[0].title, "Security · Data residency");
    }

    #[test]
    fn lexical_hit_survives_without_embedding() {
        // No query vector at all → cosine drops out, keyword "brain serve" wins.
        let hits = idx().rank("how do I run brain serve", &[], 5);
        assert_eq!(hits[0].title, "Install");
    }

    #[test]
    fn dedups_to_one_section_per_page() {
        let index = Index {
            model: "test".into(),
            dim: 2,
            chunks: vec![
                Chunk {
                    title: "MCP · A".into(),
                    url: "mcp/index.html#a".into(),
                    text: "connect a client over mcp".into(),
                    vector: vec![1.0, 0.0],
                },
                Chunk {
                    title: "MCP · B".into(),
                    url: "mcp/index.html#b".into(),
                    text: "connect another client over mcp".into(),
                    vector: vec![1.0, 0.0],
                },
                Chunk {
                    title: "WebSocket".into(),
                    url: "api/websocket.html".into(),
                    text: "connect over websocket".into(),
                    vector: vec![0.9, 0.1],
                },
            ],
        };
        let hits = index.rank("connect a client", &[1.0, 0.0], 5);
        let pages: Vec<&str> = hits
            .iter()
            .map(|h| h.url.split('#').next().unwrap())
            .collect();
        assert_eq!(pages, vec!["mcp/index.html", "api/websocket.html"]);
    }

    #[test]
    fn snippet_truncates_on_word_boundary() {
        let long = "word ".repeat(200);
        let s = snippet(&long);
        assert!(s.ends_with('…'));
        assert!(s.len() <= 324);
    }
}
