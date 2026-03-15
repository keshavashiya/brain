//! # Brain Amygdala
//!
//! Importance scoring and urgency detection with novelty tracking.
//!
//! The canonical keyword-based scoring logic lives in
//! `hippocampus::importance::ImportanceScorer` (stateless). This module wraps it
//! with per-process novelty detection (a `HashSet` of previously seen tokens).
//!
//! When an LLM provider is available, `score_async()` uses a prompt-based
//! approach for richer importance scoring. The keyword heuristic stays as a
//! 0ms fallback when LLM is unavailable or times out.

use std::collections::HashSet;
use std::sync::Arc;

/// Importance scorer with per-process novelty tracking.
///
/// Delegates keyword-based scoring to the canonical weights and keyword lists
/// defined in `hippocampus::importance::ImportanceScorer` and adds a novelty
/// bonus for previously unseen topic tokens.
///
/// When constructed with `with_llm()`, `score_async()` uses LLM-driven scoring
/// with automatic fallback to keywords on failure/timeout.
pub struct ImportanceScorer {
    /// Previously seen topic tokens (for novelty detection).
    seen_topics: std::sync::Mutex<HashSet<String>>,
    /// Optional LLM provider for prompt-based scoring.
    llm: Option<Arc<dyn cortex::LlmProvider>>,
}

impl ImportanceScorer {
    /// Create a new `ImportanceScorer` with an empty novelty history (keyword-only).
    pub fn new() -> Self {
        Self {
            seen_topics: std::sync::Mutex::new(HashSet::new()),
            llm: None,
        }
    }

    /// Create an `ImportanceScorer` backed by the given LLM provider.
    pub fn with_llm(llm: Arc<dyn cortex::LlmProvider>) -> Self {
        Self {
            seen_topics: std::sync::Mutex::new(HashSet::new()),
            llm: Some(llm),
        }
    }

    /// Score the importance of `text`. Returns a value in [0.0, 1.0].
    ///
    /// Uses the canonical keyword lists and weights from
    /// `hippocampus::ImportanceScorer`, plus per-process novelty tracking.
    pub fn score(&self, text: &str) -> f32 {
        let is_novel = self.check_novelty(text);
        hippocampus::ImportanceScorer::score(text, is_novel) as f32
    }

    /// Score importance using LLM when available, keyword fallback otherwise.
    pub async fn score_async(&self, text: &str) -> f32 {
        let is_novel = self.check_novelty(text);

        if let Some(llm) = &self.llm {
            let timeout = tokio::time::Duration::from_millis(1000);
            match tokio::time::timeout(timeout, self.score_with_llm(llm, text, is_novel)).await {
                Ok(Ok(score)) => return score,
                Ok(Err(e)) => tracing::debug!("LLM importance scoring failed: {e}"),
                Err(_) => tracing::debug!("LLM importance scoring timed out"),
            }
        }

        hippocampus::ImportanceScorer::score(text, is_novel) as f32
    }

    /// Ask the LLM to score text importance.
    async fn score_with_llm(
        &self,
        llm: &Arc<dyn cortex::LlmProvider>,
        text: &str,
        is_novel: bool,
    ) -> Result<f32, cortex::LlmError> {
        let prompt = format!(
            "Score this text's importance from 0.0 to 1.0.\n\
             Criteria: memory_request (user asks to remember/note something),\n\
             urgency (deadlines, ASAP), emotional_intensity (strong feelings),\n\
             actionable (contains tasks/commitments), specificity (concrete vs vague).\n\
             Novel content: {is_novel}\n\
             Return ONLY JSON: {{\"score\":0.X,\"reason\":\"brief\"}}\n\
             Text: {text}"
        );

        let messages = vec![cortex::Message {
            role: cortex::Role::User,
            content: prompt,
        }];

        let response = llm.generate(&messages).await?;
        parse_importance_response(&response.content)
    }

    /// Check whether the text contains novel tokens (not seen before in this
    /// process). Side-effect: records all substantial tokens as "seen".
    fn check_novelty(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
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
    }
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse LLM response into a score. Tries direct JSON, then finds `{...}`.
fn parse_importance_response(raw: &str) -> Result<f32, cortex::LlmError> {
    #[derive(serde::Deserialize)]
    struct ImportancePayload {
        score: f32,
    }

    let trimmed = raw.trim();

    // Try direct parse
    if let Ok(payload) = serde_json::from_str::<ImportancePayload>(trimmed) {
        return Ok(payload.score.clamp(0.0, 1.0));
    }

    // Try extracting JSON object
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if let Ok(payload) =
                serde_json::from_str::<ImportancePayload>(&trimmed[start..=end])
            {
                return Ok(payload.score.clamp(0.0, 1.0));
            }
        }
    }

    Err(cortex::LlmError::InvalidFormat(format!(
        "Could not parse importance score from: {trimmed}"
    )))
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

    // ── JSON parsing tests ──────────────────────────────────────────────────

    #[test]
    fn test_parse_importance_clean_json() {
        let score = parse_importance_response(r#"{"score":0.7,"reason":"memory request"}"#).unwrap();
        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_parse_importance_embedded_json() {
        let score = parse_importance_response(
            r#"Here is the score: {"score":0.85,"reason":"urgent"} done"#,
        )
        .unwrap();
        assert!((score - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_parse_importance_clamped() {
        let score = parse_importance_response(r#"{"score":1.5,"reason":"over"}"#).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_importance_invalid() {
        assert!(parse_importance_response("no json here").is_err());
    }

    // ── Async fallback tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_score_async_no_llm_equals_sync() {
        let scorer = ImportanceScorer::new();
        let sync_score = scorer.score("remember my cat Luna");
        // Reset novelty state
        let scorer2 = ImportanceScorer::new();
        let async_score = scorer2.score_async("remember my cat Luna").await;
        assert!(
            (sync_score - async_score).abs() < 1e-6,
            "No-LLM async should equal sync: {sync_score} vs {async_score}"
        );
    }

    // ── Mock LLM tests ──────────────────────────────────────────────────────

    struct MockLlm {
        response: String,
    }

    #[async_trait::async_trait]
    impl cortex::LlmProvider for MockLlm {
        async fn generate(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<cortex::Response, cortex::LlmError> {
            Ok(cortex::Response {
                content: self.response.clone(),
                usage: None,
            })
        }

        async fn generate_stream(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = Result<cortex::ResponseChunk, cortex::LlmError>>
                        + Send,
                >,
            >,
            cortex::LlmError,
        > {
            Err(cortex::LlmError::ProviderUnavailable(
                "mock".to_string(),
            ))
        }

        async fn health_check(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_score_async_with_mock_llm() {
        let mock = Arc::new(MockLlm {
            response: r#"{"score":0.9,"reason":"memory request detected"}"#.to_string(),
        });
        let scorer = ImportanceScorer::with_llm(mock);
        let score = scorer.score_async("remember my cat is named Luna").await;
        assert!(
            (score - 0.9).abs() < 1e-6,
            "LLM score should be 0.9, got {score}"
        );
    }

    #[tokio::test]
    async fn test_score_async_llm_bad_json_falls_back() {
        let mock = Arc::new(MockLlm {
            response: "I cannot parse this".to_string(),
        });
        let scorer = ImportanceScorer::with_llm(mock);
        let score = scorer.score_async("remember my cat is named Luna").await;
        // Should fall back to keyword scoring: base(0.3) + explicit(0.3) + novelty(0.1) = 0.7
        assert!(
            (score - 0.7).abs() < 1e-6,
            "Should fallback to keyword score 0.7, got {score}"
        );
    }

    struct SlowMockLlm;

    #[async_trait::async_trait]
    impl cortex::LlmProvider for SlowMockLlm {
        async fn generate(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<cortex::Response, cortex::LlmError> {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            Ok(cortex::Response {
                content: r#"{"score":0.9,"reason":"slow"}"#.to_string(),
                usage: None,
            })
        }

        async fn generate_stream(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = Result<cortex::ResponseChunk, cortex::LlmError>>
                        + Send,
                >,
            >,
            cortex::LlmError,
        > {
            Err(cortex::LlmError::ProviderUnavailable(
                "mock".to_string(),
            ))
        }

        async fn health_check(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "slow-mock"
        }
    }

    #[tokio::test]
    async fn test_score_async_timeout_falls_back() {
        let scorer = ImportanceScorer::with_llm(Arc::new(SlowMockLlm));
        let score = scorer.score_async("hello world").await;
        // Should timeout and fallback: base(0.3) + novelty(0.1) = 0.4
        assert!(
            (score - 0.4).abs() < 1e-6,
            "Should fallback on timeout, got {score}"
        );
    }
}
