//! Budget enforcement helpers for LLM call sites.
//!
//! Wraps `CostBudget::check` + `CostBudget::record` so pipeline and orchestrator
//! call sites can be wired with a single guard call. Honors the contract from
//! `docs/VISION.md` §v0.2.0 "Autonomous within a budget": ceilings block
//! *before* tokens burn, not after the fact.
//!
//! Behavior:
//! - No `CostBudget` wired                  → permissive no-op
//! - Provider absent from `BudgetPolicy`    → permissive no-op (configure to enforce)
//! - `Allowed`                              → proceed
//! - `Warn`                                 → proceed (already traced by budget impl)
//! - `Exceeded`                             → block; caller surfaces a friendly message

use std::sync::Arc;

use budget::{BudgetDecision, BudgetError, CostBudget, ResourceKind};
use cortex::llm::Message;

/// Outcome of a pre-flight budget check.
pub enum BudgetGate {
    /// Proceed with the call. `estimated_input_tokens` is the value the caller
    /// should pass to `record_llm_usage` if the provider does not return usage.
    Proceed { estimated_input_tokens: u64 },
    /// Budget exceeded. The caller must surface this message to the user and
    /// not invoke the LLM.
    Blocked { message: String },
}

/// Estimate input tokens for a message list using the same `chars/3`
/// heuristic the context assembler uses (`cortex::context::CHARS_PER_TOKEN`).
/// Kept as a caller-side mirror so the budget crate stays independent of
/// cortex; the ratio is duplicated deliberately, not imported.
pub(crate) fn estimate_input_tokens(messages: &[Message]) -> u64 {
    let chars: usize = messages.iter().map(|m| m.content.chars().count()).sum();
    chars.div_ceil(3) as u64
}

fn is_provider_unconfigured(err: &BudgetError) -> bool {
    matches!(
        err,
        BudgetError::Policy(budget::policy::PolicyError::ProviderNotFound(_))
            | BudgetError::ProviderNotFound(_)
    )
}

/// Run a pre-flight LLM input-token check.
pub async fn check_llm_input(
    budget: Option<&Arc<dyn CostBudget>>,
    provider: &str,
    messages: &[Message],
) -> BudgetGate {
    let estimated = estimate_input_tokens(messages);
    let Some(budget) = budget else {
        return BudgetGate::Proceed {
            estimated_input_tokens: estimated,
        };
    };

    match budget
        .check(provider, &ResourceKind::LlmInputTokens, estimated)
        .await
    {
        Ok(BudgetDecision::Exceeded { ceiling, consumed }) => BudgetGate::Blocked {
            message: format!(
                "Budget exceeded for provider `{provider}` ({} of {} input tokens consumed this period). \
                 The request was blocked before any tokens were spent. \
                 Raise the ceiling in `budget.providers.{provider}.hourly_input_tokens` or wait for the rolling window to expire.",
                consumed, ceiling
            ),
        },
        Ok(BudgetDecision::Warn { .. }) | Ok(BudgetDecision::Allowed) => BudgetGate::Proceed {
            estimated_input_tokens: estimated,
        },
        Err(e) if is_provider_unconfigured(&e) => BudgetGate::Proceed {
            estimated_input_tokens: estimated,
        },
        Err(e) => {
            tracing::warn!(provider = %provider, error = %e, "budget check failed; allowing call");
            BudgetGate::Proceed {
                estimated_input_tokens: estimated,
            }
        }
    }
}

/// Record actual consumption after an LLM call. Prefer real `Usage` from the
/// provider; fall back to the pre-flight estimate when usage isn't reported.
pub async fn record_llm_usage(
    budget: Option<&Arc<dyn CostBudget>>,
    provider: &str,
    usage: Option<&cortex::llm::Usage>,
    estimated_input_tokens: u64,
) {
    let Some(budget) = budget else { return };

    let (input, output) = match usage {
        Some(u) => (u.prompt_tokens as u64, u.completion_tokens as u64),
        None => (estimated_input_tokens, 0),
    };

    if let Err(e) = budget
        .record(provider, &ResourceKind::LlmInputTokens, input)
        .await
    {
        if !is_provider_unconfigured(&e) {
            tracing::warn!(provider = %provider, error = %e, "failed to record llm input tokens");
        }
    }
    if output > 0 {
        if let Err(e) = budget
            .record(provider, &ResourceKind::LlmOutputTokens, output)
            .await
        {
            if !is_provider_unconfigured(&e) {
                tracing::warn!(provider = %provider, error = %e, "failed to record llm output tokens");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortex::llm::Message;

    fn msg(role: cortex::llm::Role, content: &str) -> Message {
        Message {
            role,
            content: content.to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn no_budget_proceeds() {
        let messages = vec![msg(cortex::llm::Role::User, "hi")];
        match check_llm_input(None, "any", &messages).await {
            BudgetGate::Proceed { .. } => {}
            BudgetGate::Blocked { .. } => panic!("expected proceed when no budget wired"),
        }
    }

    #[test]
    fn estimate_uses_chars_over_three() {
        let messages = vec![
            msg(cortex::llm::Role::User, "abcdef"),
            msg(cortex::llm::Role::Assistant, "ghij"),
        ];
        // 10 chars at 3 chars/token → ceil(10/3) = 4.
        assert_eq!(estimate_input_tokens(&messages), 4);
    }
}
