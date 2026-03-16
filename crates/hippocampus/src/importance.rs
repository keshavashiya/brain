//! Importance scoring — keyword-based relevance tagging.
//!
//! Assigns an importance score to each incoming message based on
//! textual signals. v1 is purely keyword-based (no LLM cost);
//! post-v1 will add LLM-based sentiment analysis.

/// Score signals detected in the text.
#[derive(Debug, Clone, Default)]
pub struct ImportanceSignals {
    /// Explicit memory requests ("remember", "important", "don't forget")
    pub explicit: bool,
    /// Urgency markers ("asap", "urgent", "deadline", "emergency")
    pub urgency: bool,
    /// Emotional intensity ("stressed", "excited", "frustrated")
    pub emotional: bool,
    /// Whether the content is novel (caller-provided)
    pub novelty: bool,
}

/// Keyword-based importance scorer.
///
/// Scoring weights (additive, clamped to [0.0, 1.0]):
/// - Base: 0.3
/// - Explicit signals: +0.3
/// - Urgency: +0.2
/// - Emotional intensity: +0.15
/// - Novelty: +0.1
///
/// Uses word-boundary matching for single-word keywords to avoid false
/// positives (e.g., "insurgent" no longer triggers "urgent"). Multi-word
/// phrases use substring matching since they are already specific enough.
pub struct ImportanceScorer;

impl ImportanceScorer {
    const BASE_SCORE: f64 = 0.3;
    const EXPLICIT_BOOST: f64 = 0.3;
    const URGENCY_BOOST: f64 = 0.2;
    const EMOTIONAL_BOOST: f64 = 0.15;
    const NOVELTY_BOOST: f64 = 0.1;

    /// Multi-word phrases that signal explicit memory intent (use substring match).
    const EXPLICIT_PHRASES: &[&str] = &[
        "don't forget",
        "dont forget",
        "note that",
        "keep in mind",
        "make sure to remember",
        "never forget",
        "always remember",
    ];

    /// Single-word keywords that signal explicit memory intent (use word-boundary match).
    const EXPLICIT_WORDS: &[&str] = &["remember", "important"];

    /// Multi-word phrases that signal urgency (use substring match).
    const URGENCY_PHRASES: &[&str] = &["right now", "due date"];

    /// Single-word keywords that signal urgency (use word-boundary match).
    const URGENCY_WORDS: &[&str] = &[
        "asap",
        "urgent",
        "deadline",
        "emergency",
        "immediately",
        "timesensitive",
        "critical",
        "overdue",
    ];

    /// Single-word keywords that signal emotional intensity (use word-boundary match).
    /// All emotional keywords are single words.
    const EMOTIONAL_WORDS: &[&str] = &[
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

    /// Score a piece of text for importance.
    ///
    /// Returns a value in [0.0, 1.0] indicating how important
    /// this memory is. Higher = more important, resists decay longer.
    pub fn score(text: &str, novelty: bool) -> f64 {
        let signals = Self::detect(text, novelty);
        Self::score_from_signals(&signals)
    }

    /// Detect importance signals in text.
    pub fn detect(text: &str, novelty: bool) -> ImportanceSignals {
        let lower = text.to_lowercase();

        ImportanceSignals {
            explicit: has_phrase(&lower, Self::EXPLICIT_PHRASES)
                || has_word(&lower, Self::EXPLICIT_WORDS),
            urgency: has_phrase(&lower, Self::URGENCY_PHRASES)
                || has_word(&lower, Self::URGENCY_WORDS),
            emotional: has_word(&lower, Self::EMOTIONAL_WORDS),
            novelty,
        }
    }

    /// Compute score from pre-detected signals.
    pub fn score_from_signals(signals: &ImportanceSignals) -> f64 {
        let mut score = Self::BASE_SCORE;

        if signals.explicit {
            score += Self::EXPLICIT_BOOST;
        }
        if signals.urgency {
            score += Self::URGENCY_BOOST;
        }
        if signals.emotional {
            score += Self::EMOTIONAL_BOOST;
        }
        if signals.novelty {
            score += Self::NOVELTY_BOOST;
        }

        // Clamp to [0.0, 1.0]
        score.clamp(0.0, 1.0)
    }
}

/// Check if `text` contains any of `phrases` as a substring.
///
/// Used for multi-word phrases where substring matching is specific enough.
fn has_phrase(text: &str, phrases: &[&str]) -> bool {
    phrases.iter().any(|p| text.contains(p))
}

/// Check if `text` contains any of `words` as whole words.
///
/// A "word boundary" is the start/end of the string or any non-alphabetic
/// character. This avoids false positives like "insurgent" matching "urgent".
fn has_word(text: &str, words: &[&str]) -> bool {
    words.iter().any(|word| contains_whole_word(text, word))
}

/// Returns true if `word` appears in `text` surrounded by word boundaries.
fn contains_whole_word(text: &str, word: &str) -> bool {
    let word_bytes = word.as_bytes();
    let text_bytes = text.as_bytes();
    let word_len = word_bytes.len();

    if word_len > text_bytes.len() {
        return false;
    }

    for start in 0..=(text_bytes.len() - word_len) {
        if &text_bytes[start..start + word_len] != word_bytes {
            continue;
        }
        // Check left boundary: start of string or non-alphabetic
        let left_ok = start == 0 || !text_bytes[start - 1].is_ascii_alphabetic();
        // Check right boundary: end of string or non-alphabetic
        let right_ok = start + word_len == text_bytes.len()
            || !text_bytes[start + word_len].is_ascii_alphabetic();

        if left_ok && right_ok {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_score() {
        let score = ImportanceScorer::score("Hello, how are you?", false);
        assert!((score - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_explicit_boost() {
        let score = ImportanceScorer::score("Remember that I prefer Rust", false);
        assert!((score - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_urgency_boost() {
        let score = ImportanceScorer::score("This is urgent, I need help ASAP", false);
        assert!((score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_emotional_boost() {
        let score = ImportanceScorer::score("I'm really stressed about this", false);
        assert!((score - 0.45).abs() < f64::EPSILON);
    }

    #[test]
    fn test_novelty_boost() {
        let score = ImportanceScorer::score("Something mundane", true);
        assert!((score - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combined_max() {
        // All signals active should clamp to 1.0
        let score = ImportanceScorer::score(
            "Remember this urgent thing, I'm stressed about the deadline",
            true,
        );
        // 0.3 + 0.3 + 0.2 + 0.15 + 0.1 = 1.05, clamped to 1.0
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_case_insensitive() {
        let score = ImportanceScorer::score("REMEMBER THIS IMPORTANT THING", false);
        assert!((score - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_text() {
        let score = ImportanceScorer::score("", false);
        assert!((score - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_signals() {
        let signals = ImportanceScorer::detect("Remember this urgent moment", true);
        assert!(signals.explicit);
        assert!(signals.urgency);
        assert!(!signals.emotional);
        assert!(signals.novelty);
    }

    // ── Word-boundary false-positive prevention ─────────────────────────────

    #[test]
    fn test_insurgent_does_not_trigger_urgency() {
        let score = ImportanceScorer::score("The insurgent was captured", false);
        assert!(
            (score - 0.3).abs() < f64::EPSILON,
            "'insurgent' should NOT trigger urgency, got {score}"
        );
    }

    #[test]
    fn test_misremember_does_not_trigger_explicit() {
        let score = ImportanceScorer::score("I tend to misremember things", false);
        assert!(
            (score - 0.3).abs() < f64::EPSILON,
            "'misremember' should NOT trigger explicit, got {score}"
        );
    }

    #[test]
    fn test_unhappy_does_not_trigger_emotional() {
        let score = ImportanceScorer::score("I am unhappy about it", false);
        assert!(
            (score - 0.3).abs() < f64::EPSILON,
            "'unhappy' should NOT trigger emotional (happy), got {score}"
        );
    }

    #[test]
    fn test_unimportant_does_not_trigger_explicit() {
        let score = ImportanceScorer::score("This is unimportant", false);
        assert!(
            (score - 0.3).abs() < f64::EPSILON,
            "'unimportant' should NOT trigger explicit, got {score}"
        );
    }

    #[test]
    fn test_word_at_boundaries() {
        // keyword at start of string
        assert!((ImportanceScorer::score("urgent: fix this", false) - 0.5).abs() < f64::EPSILON);
        // keyword at end of string
        assert!((ImportanceScorer::score("this is urgent", false) - 0.5).abs() < f64::EPSILON);
        // keyword with punctuation
        assert!((ImportanceScorer::score("it's urgent!", false) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_word_phrases_still_work() {
        // "don't forget" is a multi-word phrase — still uses substring matching
        let score = ImportanceScorer::score("don't forget about the meeting", false);
        assert!(
            (score - 0.6).abs() < f64::EPSILON,
            "multi-word phrase 'don't forget' should trigger explicit, got {score}"
        );
    }

    // ── Internal word-boundary function ─────────────────────────────────────

    #[test]
    fn test_contains_whole_word() {
        assert!(contains_whole_word("this is urgent", "urgent"));
        assert!(contains_whole_word("urgent fix needed", "urgent"));
        assert!(contains_whole_word("it's urgent!", "urgent"));
        assert!(contains_whole_word("urgent", "urgent"));
        assert!(!contains_whole_word("insurgent attack", "urgent"));
        assert!(!contains_whole_word("urgently needed", "urgent"));
    }
}
