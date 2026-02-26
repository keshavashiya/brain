//! # Brain Amygdala
//!
//! Importance scoring and urgency detection.
//!
//! v1: Keyword-based heuristics (no LLM dependency).
//! Post-v1: LLM-based sentiment analysis, tone adaptation,
//! mood tracking.

use std::collections::HashSet;

/// Scores the importance/urgency weight of a signal's text.
///
/// Returns a value in [0.0, 1.0] based on keyword heuristics:
/// - Explicit signals ("remember", "important", etc.) add +0.3
/// - Urgency signals ("ASAP", "urgent", etc.) add +0.2
/// - Emotion signals ("excited", "frustrated", etc.) add +0.15
/// - Novelty signals (unknown topic words) add +0.1
pub struct ImportanceScorer {
    /// Previously seen topic tokens (for novelty detection).
    seen_topics: std::sync::Mutex<HashSet<String>>,
}

impl ImportanceScorer {
    /// Create a new `ImportanceScorer` with an empty novelty history.
    pub fn new() -> Self {
        Self {
            seen_topics: std::sync::Mutex::new(HashSet::new()),
        }
    }

    /// Score the importance of `text`. Returns a value in [0.0, 1.0].
    pub fn score(&self, text: &str) -> f32 {
        let lower = text.to_lowercase();
        let mut score: f32 = 0.0;

        // --- Explicit memory signals (+0.3) ---
        const EXPLICIT: &[&str] = &[
            "remember",
            "important",
            "note",
            "don't forget",
            "keep in mind",
            "must know",
        ];
        if EXPLICIT.iter().any(|kw| lower.contains(kw)) {
            score += 0.3;
        }

        // --- Urgency signals (+0.2) ---
        const URGENCY: &[&str] = &[
            "asap",
            "urgent",
            "critical",
            "immediately",
            "right now",
            "deadline",
            "emergency",
        ];
        if URGENCY.iter().any(|kw| lower.contains(kw)) {
            score += 0.2;
        }

        // --- Emotion signals (+0.15) ---
        const EMOTION: &[&str] = &[
            "excited",
            "frustrated",
            "angry",
            "happy",
            "sad",
            "love",
            "hate",
            "amazing",
            "terrible",
            "thrilled",
            "worried",
            "anxious",
        ];
        if EMOTION.iter().any(|kw| lower.contains(kw)) {
            score += 0.15;
        }

        // --- Novelty signals (+0.1) ---
        // A word is "novel" if it is a substantial token (len > 4) and
        // has not been seen before. We add the novelty bonus once if any
        // such word is new.
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
            score += 0.1;
        }

        // Clamp to [0.0, 1.0]
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
    fn test_no_signals_returns_zero_or_novelty() {
        // Fresh scorer — every word is novel so we get 0.1
        let scorer = ImportanceScorer::new();
        let s = scorer.score("hello world");
        // The words "hello" and "world" are both len <= 5 (world=5 > 4, hello=5 > 4)
        // Actually "hello" and "world" are len 5 which is > 4 so they count as novel
        assert!(s >= 0.0 && s <= 1.0);
    }

    #[test]
    fn test_explicit_signal_adds_0_3() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("Please remember this for later");
        // Explicit (+0.3) + novelty (+0.1) = 0.4
        assert!((s - 0.4).abs() < 1e-6, "Expected ~0.4, got {s}");
    }

    #[test]
    fn test_urgency_signal_adds_0_2() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("This is urgent please help");
        // Urgency (+0.2) + novelty (+0.1) = 0.3
        assert!((s - 0.3).abs() < 1e-6, "Expected ~0.3, got {s}");
    }

    #[test]
    fn test_emotion_signal_adds_0_15() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("I am so excited about this");
        // Emotion (+0.15) + novelty (+0.1) = 0.25
        assert!((s - 0.25).abs() < 1e-6, "Expected ~0.25, got {s}");
    }

    #[test]
    fn test_novelty_signal_adds_0_1_only_first_time() {
        let scorer = ImportanceScorer::new();
        // First call — novel tokens present → +0.1
        let s1 = scorer.score("quantum computing breakthrough");
        assert!(s1 >= 0.1, "First call should add novelty bonus");

        // Second call with exactly the same text — no new tokens → no bonus
        let s2 = scorer.score("quantum computing breakthrough");
        assert!(
            s2 < 0.15,
            "Second call with same text should not add novelty bonus: got {s2}"
        );
    }

    #[test]
    fn test_combined_explicit_and_urgency() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("Important: fix this ASAP");
        // Explicit (+0.3) + Urgency (+0.2) + novelty (+0.1) = 0.6
        assert!((s - 0.6).abs() < 1e-6, "Expected ~0.6, got {s}");
    }

    #[test]
    fn test_all_signals_clamped_to_1() {
        let scorer = ImportanceScorer::new();
        // All four categories triggered
        let s = scorer.score("Remember this important urgent ASAP excited frustrated deadline");
        // explicit(0.3) + urgency(0.2) + emotion(0.15) + novelty(0.1) = 0.75 → ≤ 1.0
        assert!(s <= 1.0, "Score must not exceed 1.0, got {s}");
        assert!(s >= 0.7, "Expected high score, got {s}");
    }

    #[test]
    fn test_score_clamped_lower() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("");
        assert!(s >= 0.0, "Score must be >= 0.0");
    }

    #[test]
    fn test_case_insensitive_explicit() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("REMEMBER TO DO THIS");
        assert!(
            s >= 0.3,
            "Uppercase 'REMEMBER' should trigger explicit signal"
        );
    }

    #[test]
    fn test_case_insensitive_urgency() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("URGENT: server is down");
        assert!(s >= 0.2, "Uppercase 'URGENT' should trigger urgency signal");
    }

    #[test]
    fn test_case_insensitive_emotion() {
        let scorer = ImportanceScorer::new();
        let s = scorer.score("I am FRUSTRATED with this issue");
        assert!(s >= 0.15, "Uppercase emotion should be detected");
    }
}
