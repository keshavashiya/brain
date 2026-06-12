//! Per-tier LLM usage accounting.
//!
//! [`TierUsageRecorder`] wraps a tier's provider chain and, once a
//! [`budget::CostBudget`] is wired (via
//! [`SignalProcessor::with_cost_budget`](crate::SignalProcessor)), records
//! every completed generation under the pseudo-provider key
//! `tier:<fast|balanced|deep>`. That makes the spend split visible in
//! `BudgetStatus` / `/budget` regardless of which subsystem made the call
//! (classifier fallback, importance, compaction, chat, decomposition) —
//! and, since the budget policy keys ceilings by provider string, lets a
//! user cap a tier outright (`budget.providers."tier:deep"`). Recording
//! only; enforcement stays at the call sites that already check.

use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use cortex::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk, TaskTier, ToolDef};
use futures::Stream;

/// Shared slot for the cost budget, filled once at bootstrap. The tier
/// chains are built (and handed to the classifier/scorer) before the
/// budget exists, so they hold this cell rather than the budget itself.
pub(crate) type BudgetCell = Arc<OnceLock<Arc<dyn budget::CostBudget>>>;

pub(crate) struct TierUsageRecorder {
    inner: Arc<dyn LlmProvider>,
    tier: TaskTier,
    budget: BudgetCell,
}

impl TierUsageRecorder {
    pub(crate) fn wrap(
        inner: Arc<dyn LlmProvider>,
        tier: TaskTier,
        budget: BudgetCell,
    ) -> Arc<dyn LlmProvider> {
        Arc::new(Self {
            inner,
            tier,
            budget,
        })
    }

    async fn record(&self, messages: &[Message], usage: Option<&cortex::llm::Usage>) {
        let Some(budget) = self.budget.get() else {
            return;
        };
        let estimated = crate::budget_guard::estimate_input_tokens(messages);
        crate::budget_guard::record_llm_usage(
            Some(budget),
            &format!("tier:{}", self.tier),
            usage,
            estimated,
        )
        .await;
    }
}

#[async_trait::async_trait]
impl LlmProvider for TierUsageRecorder {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let resp = self.inner.generate(messages).await?;
        self.record(messages, resp.usage.as_ref()).await;
        Ok(resp)
    }

    async fn generate_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Response, LlmError> {
        let resp = self.inner.generate_with_tools(messages, tools).await?;
        self.record(messages, resp.usage.as_ref()).await;
        Ok(resp)
    }

    /// Streaming chunks carry no usage; the stream passes through
    /// unrecorded (same gap as the provider-keyed accounting).
    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        self.inner.generate_stream(messages).await
    }

    async fn health_check(&self) -> bool {
        self.inner.health_check().await
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.inner.list_models().await
    }

    async fn fetch_context_window(&self) -> Option<usize> {
        self.inner.fetch_context_window().await
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubLlm;

    #[async_trait::async_trait]
    impl LlmProvider for StubLlm {
        async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
            Ok(Response::text(
                "ok",
                Some(cortex::llm::Usage {
                    prompt_tokens: 7,
                    completion_tokens: 3,
                    total_tokens: 10,
                }),
            ))
        }
        async fn generate_stream(
            &self,
            _messages: &[Message],
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
        {
            unimplemented!()
        }
        async fn health_check(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "stub"
        }
        fn model(&self) -> &str {
            "stub"
        }
        async fn list_models(&self) -> Result<Vec<String>, LlmError> {
            Ok(vec![])
        }
    }

    fn memory_budget() -> Arc<dyn budget::CostBudget> {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let b = budget::SqliteBudget::new(pool, budget::BudgetPolicy::default());
        b.ensure_tables().unwrap();
        Arc::new(b)
    }

    #[tokio::test]
    async fn records_usage_under_the_tier_key_once_budget_is_wired() {
        let cell: BudgetCell = Arc::new(OnceLock::new());
        let llm = TierUsageRecorder::wrap(Arc::new(StubLlm), TaskTier::Fast, cell.clone());

        // Before the budget is wired: generation works, nothing recorded.
        llm.generate(&[Message::user("hi")]).await.unwrap();

        let budget = memory_budget();
        cell.set(budget.clone()).ok();
        llm.generate(&[Message::user("hi again")]).await.unwrap();

        let status = budget.status().await.unwrap();
        assert_eq!(
            status
                .hourly_consumption
                .get("tier:fast:llm_input_tokens")
                .copied(),
            Some(7),
            "input tokens recorded under the tier pseudo-provider"
        );
        assert_eq!(
            status
                .hourly_consumption
                .get("tier:fast:llm_output_tokens")
                .copied(),
            Some(3)
        );
    }
}
