//! Conversation-history compaction planning.
//!
//! When a chat thread grows past its slice of the token budget, the assembler
//! would otherwise *drop* the oldest turns outright. Compaction instead keeps
//! the recent turns verbatim and folds the overflow prefix into a short
//! summary, so older context survives in compressed form rather than
//! vanishing.
//!
//! This module is the **pure planning half**: it decides which turns fit and
//! which overflow, with no LLM and no I/O, so the policy is fully unit-tested.
//! The signal pipeline owns the LLM-backed summarization + caching and feeds
//! the kept/overflow split from here.

use crate::context::estimate_tokens;
use crate::llm::Message;

/// How `history` should be split to fit a token budget.
#[derive(Debug, Clone)]
pub struct HistoryPlan<'a> {
    /// Oldest turns that don't fit — to be compacted into a single summary
    /// note by the caller. Empty when the whole history already fits.
    pub to_summarize: &'a [Message],
    /// The most recent turns that fit verbatim, in chronological order.
    pub keep_recent: &'a [Message],
}

impl HistoryPlan<'_> {
    /// True when nothing overflows — the caller can use the history as-is and
    /// skip the (LLM-costing) summary entirely.
    pub fn is_noop(&self) -> bool {
        self.to_summarize.is_empty()
    }
}

/// Plan history compaction against a `budget` (in tokens), holding back
/// `reserve` tokens for the summary note the caller will prepend so the final
/// `summary + keep_recent` still fits.
///
/// Walks newest→oldest, keeping turns while they fit within `budget - reserve`;
/// the oldest prefix that doesn't fit becomes `to_summarize`. When everything
/// fits, `to_summarize` is empty (a no-op). A single turn larger than the
/// effective budget falls into `to_summarize` rather than being kept, so the
/// caller never emits a verbatim block that blows the budget.
pub fn plan_history_compaction<'a>(
    history: &'a [Message],
    budget: usize,
    reserve: usize,
) -> HistoryPlan<'a> {
    let effective = budget.saturating_sub(reserve);
    let mut used = 0usize;
    // Index where the kept (recent) turns begin; defaults to len() == keep none.
    let mut keep_from = history.len();

    for (i, msg) in history.iter().enumerate().rev() {
        let tokens = estimate_tokens(&msg.content);
        if used + tokens > effective {
            break;
        }
        used += tokens;
        keep_from = i;
    }

    HistoryPlan {
        to_summarize: &history[..keep_from],
        keep_recent: &history[keep_from..],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;

    // ~30 chars ≈ 10 tokens at 3 chars/token.
    fn turn(n: usize) -> Message {
        Message::user(format!("turn {n}: {}", "x".repeat(27)))
    }

    #[test]
    fn nothing_overflows_is_a_noop() {
        let history: Vec<Message> = (0..3).map(turn).collect();
        let plan = plan_history_compaction(&history, 1000, 100);
        assert!(plan.is_noop());
        assert_eq!(plan.keep_recent.len(), 3);
        assert!(plan.to_summarize.is_empty());
    }

    #[test]
    fn oldest_turns_overflow_into_summary_keeping_recent() {
        let history: Vec<Message> = (0..10).map(turn).collect();
        // Each turn ~11 tokens; budget 40, reserve 5 → effective 35 → ~3 turns fit.
        let plan = plan_history_compaction(&history, 40, 5);
        assert!(!plan.is_noop());
        // Kept turns are the most recent, in order, and contiguous with overflow.
        assert_eq!(
            plan.to_summarize.len() + plan.keep_recent.len(),
            history.len()
        );
        assert!(plan.keep_recent.len() < history.len());
        // The newest turn is always kept.
        assert_eq!(
            plan.keep_recent.last().unwrap().content,
            history.last().unwrap().content
        );
        // The split is chronological: overflow is the oldest prefix.
        assert_eq!(
            plan.to_summarize.first().unwrap().content,
            history[0].content
        );
    }

    #[test]
    fn single_oversized_turn_goes_to_summary_not_kept_verbatim() {
        let big = Message::user("y".repeat(10_000));
        let history = vec![big];
        let plan = plan_history_compaction(&history, 40, 5);
        assert_eq!(plan.keep_recent.len(), 0);
        assert_eq!(plan.to_summarize.len(), 1);
    }

    #[test]
    fn empty_history_is_a_noop() {
        let plan = plan_history_compaction(&[], 100, 10);
        assert!(plan.is_noop());
        assert!(plan.keep_recent.is_empty());
    }
}
