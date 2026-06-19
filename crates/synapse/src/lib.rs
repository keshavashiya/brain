//! # Brain Synapse — similarity primitives
//!
//! The shared, dependency-free math that scores how close two things are.
//! Both the capability **router** (which tool answers an intent) and the docs
//! **retrieval** assistant (which doc section answers a question) rank
//! candidates with the exact same functions defined here — there is no second
//! copy. Because the crate is pure `std` with no dependencies, it compiles
//! unchanged to `wasm32-unknown-unknown`, so the in-browser docs assistant runs
//! the engine's real ranking code rather than a re-implementation.
//!
//! - [`cosine_similarity`] — semantic closeness between two embedding vectors.
//! - [`jaccard`] — set overlap between two capability/term lists.
//! - [`tokenize`] / [`lexical_overlap`] — keyword-level overlap for hybrid
//!   scoring when an exact semantic match is ambiguous.

/// Cosine similarity between two embedding vectors, clamped to `[0, 1]`.
///
/// Returns `0.0` when either vector is empty, dimensions disagree, or either
/// magnitude is zero — i.e. the semantic term silently drops out rather than
/// poisoning the score. Negative cosines (semantically opposed directions)
/// are floored to `0.0`: an unrelated candidate should contribute nothing,
/// never a penalty that could reorder the keyword/verb signal beneath it.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0)
}

/// Jaccard overlap between two string sets: `|A ∩ B| / |A ∪ B|`, in `[0, 1]`.
///
/// Empty-on-both-sides returns `0.0` (no signal, not a perfect match).
pub fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let set_a: std::collections::HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(String::as_str).collect();
    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Split text into lower-case alphanumeric word tokens.
///
/// Deliberately simple and allocation-light: runs of ASCII-alphanumeric
/// characters become tokens, everything else is a separator. Shared by both
/// sides of [`lexical_overlap`] so query and document tokenize identically.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Fraction of the query's distinct terms that appear in `text`, in `[0, 1]`.
///
/// This is the keyword half of a hybrid score: it rewards literal term hits so
/// an exact phrase in the docs still surfaces even when the embedding model is
/// unavailable and the cosine term has dropped to `0.0`. Returns `0.0` for an
/// empty query.
pub fn lexical_overlap(query: &str, text: &str) -> f32 {
    let q_terms: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
    if q_terms.is_empty() {
        return 0.0;
    }
    let doc_terms: std::collections::HashSet<String> = tokenize(text).into_iter().collect();
    let hits = q_terms.iter().filter(|t| doc_terms.contains(*t)).count();
    hits as f32 / q_terms.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one_orthogonal_is_zero() {
        assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_negative_is_floored_to_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_degenerate_inputs_drop_out() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn jaccard_overlap_basics() {
        let a = vec!["x".to_string(), "y".to_string()];
        let b = vec!["y".to_string(), "z".to_string()];
        assert!((jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(jaccard(&[], &[]), 0.0);
    }

    #[test]
    fn tokenize_splits_on_non_alphanumeric_and_lowercases() {
        assert_eq!(
            tokenize("Local-First, on device!"),
            vec!["local", "first", "on", "device"]
        );
    }

    #[test]
    fn lexical_overlap_counts_distinct_query_terms() {
        // 2 of 3 distinct query terms ("keep data cloud") appear.
        let score = lexical_overlap(
            "keep data cloud",
            "your data never leaves the device, kept local",
        );
        assert!((score - 1.0 / 3.0).abs() < 1e-6, "got {score}");
        assert_eq!(lexical_overlap("", "anything"), 0.0);
    }
}
