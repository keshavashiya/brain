//! # Brain Amygdala
//!
//! Importance scoring and urgency detection with novelty tracking.
//!
//! The canonical keyword-based scoring logic lives in
//! `hippocampus::importance::ImportanceScorer` (stateless). This module wraps it
//! with per-process novelty detection (a `HashSet` of previously seen tokens).
//!
//! v1: Keyword-based heuristics (no LLM dependency).
//! Post-v1: LLM-based sentiment analysis, tone adaptation, mood tracking.

use std::collections::HashSet;

/// Importance scorer with per-process novelty tracking.
///
/// Delegates keyword-based scoring to the canonical weights defined in
/// `hippocampus::importance` and adds a novelty bonus for previously unseen
/// topic tokens.
pub struct ImportanceScorer {
    /// Previously seen topic tokens (for novelty detection).
    seen_topics: std::sync::Mutex<HashSet<String>>,
}

// Canonical scoring weights — kept in sync with hippocampus::importance.
const BASE_SCORE: f32 = 0.3;
const EXPLICIT_BOOST: f32 = 0.3;
const URGENCY_BOOST: f32 = 0.2;
const EMOTIONAL_BOOST: f32 = 0.15;
const NOVELTY_BOOST: f32 = 0.1;

/// Keywords that signal explicit memory intent.
const EXPLICIT_KEYWORDS: &[&str] = &[
    "remember",
    "important",
    "don't forget",
    "dont forget",
    "note that",
    "keep in mind",
    "make sure to remember",
    "never forget",
    "always remember",
];

/// Keywords that signal urgency.
const URGENCY_KEYWORDS: &[&str] = &[
    "asap",
    "urgent",
    "deadline",
    "emergency",
    "immediately",
    "right now",
    "timesensitive",
    "critical",
    "due date",
    "overdue",
];

/// Keywords that signal emotional intensity.
const EMOTIONAL_KEYWORDS: &[&str] = &[
    "stressed",
    "excited",
    "frustrated",
    "anxious",
    "worried",
    "happy",
    "angry",
    "overwhelmed",
    "thrilled",
    "exhausted",
    "passionate",
    "terrified",
];

impl ImportanceScorer {
    /// Create a new `ImportanceScorer` with an empty novelty history.
    pub fn new() -> Self {
        Self {
            seen_topics: std::sync::Mutex::new(HashSet::new()),
        }
    }

    /// Score the importance of `text`. Returns a value in [0.0, 1.0].
    ///
    /// Uses the same keyword lists and weights as `hippocampus::ImportanceScorer`,
    /// plus per-process novelty tracking.
    pub fn score(&self, text: &str) -> f32 {
        let lower = text.to_lowercase();

        let mut score: f32 = BASE_SCORE;

        if EXPLICIT_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
            score += EXPLICIT_BOOST;
        }

        if URGENCY_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
            score += URGENCY_BOOST;
        }

        if EMOTIONAL_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
            score += EMOTIONAL_BOOST;
        }

        // Novelty detection: a word is "novel" if it is a substantial
        // token (len > 4) and has not been seen before in this process.
        let is_novel = {
            let tokens: Vec<String> = lower
                .split(|c: char| !c.is_alphabetic())
                .filter(|w| w.len() > 4)
                .map(|w| w.to_string())
                .collect();

            let mut seen = self.seen_topics.lock().unwrap();
            let mut found_new = false;
            for token in tokens {
                if seen.insert(token) {
                    found_new = true;
                }
            }
            found_new
        };

        if is_novel {
            score += NOVELTY_BOOST;
        }

        score.clamp(0.0, 1.0)
    }
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_score_with_novelty() {
        // Fresh scorer — novel tokens present → base(0.3) + novelty(0.1) = 0.4
        let scorer = ImportanceScorer::new();
        let s = scorer.score("hello world");
        assert!((s - 0.4).abs() < 1e-6, "Expected ~0.4, got {s}");
    }

    #[test]
    fn test_base_score_no_novelty() {
        let scorer = ImportanceScorer::new();
        // First call — seeds the seen set
        scorer.score("hello world");
        // Second call with same tokens — no novelty
        let s = scorer.score("hello world");
        assert!((s - 0.3).abs() < 1e-6, "Expected ~0.3, got {s}");
    }

    #[test]
    fn test_explicit_signal() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("Please remember this for later");
        // base(0.3) + explicit(0.3) + novelty(0.1) = 0.7
        assert!((s - 0.7).abs() < 1e-6, "Expected ~0.7, got {s}");
    }

    #[test]
    fn test_urgency_signal() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("This is urgent please handle");
        // base(0.3) + urgency(0.2) + novelty(0.1) = 0.6
        assert!((s - 0.6).abs() < 1e-6, "Expected ~0.6, got {s}");
    }

    #[test]
    fn test_emotion_signal() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("I am so excited about this");
        // base(0.3) + emotion(0.15) + novelty(0.1) = 0.55
        assert!((s - 0.55).abs() < 1e-6, "Expected ~0.55, got {s}");
    }

    #[test]
    fn test_novelty_signal_adds_0_1_only_first_time() {
        let scorer = ImportanceScorer::new();
        let s1 = scorer.score("quantum computing breakthrough");
        assert!(s1 >= 0.4, "First call should include novelty bonus");

        let s2 = scorer.score("quantum computing breakthrough");
        assert!(
            (s2 - 0.3).abs() < 1e-6,
            "Second call with same text should be base only: got {s2}"
        );
    }

    #[test]
    fn test_combined_explicit_and_urgency() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("Important: fix this ASAP deadline");
        // base(0.3) + explicit(0.3) + urgency(0.2) + novelty(0.1) = 0.9
        assert!((s - 0.9).abs() < 1e-6, "Expected ~0.9, got {s}");
    }

    #[test]
    fn test_all_signals_clamped_to_1() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("Remember this urgent ASAP excited stressed deadline");
        // base(0.3) + explicit(0.3) + urgency(0.2) + emotion(0.15) + novelty(0.1) = 1.05 → clamped to 1.0
        assert!((s - 1.0).abs() < 1e-6, "Expected 1.0, got {s}");
    }

    #[test]
    fn test_empty_text() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("");
        // base(0.3) + no novelty = 0.3
        assert!((s - 0.3).abs() < 1e-6, "Expected ~0.3, got {s}");
    }

    #[test]
    fn test_case_insensitive() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("REMEMBER THIS IMPORTANT URGENT");
        assert!(s >= 0.8, "Uppercase keywords should trigger signals");
    }
}
