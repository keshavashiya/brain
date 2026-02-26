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
pub struct ImportanceScorer;

impl ImportanceScorer {
    const BASE_SCORE: f64 = 0.3;
    const EXPLICIT_BOOST: f64 = 0.3;
    const URGENCY_BOOST: f64 = 0.2;
    const EMOTIONAL_BOOST: f64 = 0.15;
    const NOVELTY_BOOST: f64 = 0.1;

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
        "time-sensitive",
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
            explicit: Self::EXPLICIT_KEYWORDS.iter().any(|kw| lower.contains(kw)),
            urgency: Self::URGENCY_KEYWORDS.iter().any(|kw| lower.contains(kw)),
            emotional: Self::EMOTIONAL_KEYWORDS.iter().any(|kw| lower.contains(kw)),
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
}
