use std::pin::Pin;

use futures::Stream;

use super::{LlmError, LlmProvider, Message, Response, ResponseChunk, ToolDef};

/// An error is retriable (try next provider) for any failure EXCEPT
/// `InvalidFormat` — a parse error in our own SSE/JSON handling that would
/// fail identically against every provider.  Network failures, rate limits,
/// auth errors, and 5xx/4xx responses are all worth trying a fallback for.
fn is_retriable(e: &LlmError) -> bool {
    !matches!(e, LlmError::InvalidFormat(_))
}

/// Wraps a ranked list of providers and retries the next one whenever the
/// current provider returns a retriable error (rate limit, 5xx, timeout).
///
/// Non-retriable errors (4xx auth, bad request, invalid format) propagate
/// immediately without trying fallbacks.
pub struct FailoverProvider {
    providers: Vec<Box<dyn LlmProvider>>,
}

impl FailoverProvider {
    pub fn new(providers: Vec<Box<dyn LlmProvider>>) -> Self {
        Self { providers }
    }
}

#[async_trait::async_trait]
impl LlmProvider for FailoverProvider {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let mut last_err = LlmError::ProviderUnavailable("no providers configured".into());
        for provider in &self.providers {
            match provider.generate(messages).await {
                Ok(resp) => return Ok(resp),
                Err(e) if is_retriable(&e) => {
                    tracing::warn!(
                        provider = provider.name(),
                        model = provider.model(),
                        error = %e,
                        "provider failed — falling over to next"
                    );
                    last_err = e;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err)
    }

    async fn generate_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Response, LlmError> {
        let mut last_err = LlmError::ProviderUnavailable("no providers configured".into());
        for provider in &self.providers {
            match provider.generate_with_tools(messages, tools).await {
                Ok(resp) => return Ok(resp),
                Err(e) if is_retriable(&e) => {
                    tracing::warn!(
                        provider = provider.name(),
                        model = provider.model(),
                        error = %e,
                        "provider failed — falling over to next"
                    );
                    last_err = e;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err)
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        let mut last_err = LlmError::ProviderUnavailable("no providers configured".into());
        for provider in &self.providers {
            match provider.generate_stream(messages).await {
                Ok(stream) => return Ok(stream),
                Err(e) if is_retriable(&e) => {
                    tracing::warn!(
                        provider = provider.name(),
                        model = provider.model(),
                        error = %e,
                        "provider stream setup failed — falling over to next"
                    );
                    last_err = e;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err)
    }

    /// The failover chain is healthy if **any** provider is reachable — a
    /// down primary with a live fallback is still serviceable, which is the
    /// whole point of failover. Probing only the first provider would report
    /// the chain unhealthy precisely when failover would otherwise save it.
    async fn health_check(&self) -> bool {
        for provider in &self.providers {
            if provider.health_check().await {
                return true;
            }
        }
        false
    }

    fn name(&self) -> &str {
        self.providers.first().map(|p| p.name()).unwrap_or("none")
    }

    fn model(&self) -> &str {
        self.providers.first().map(|p| p.model()).unwrap_or("none")
    }

    /// Union the model lists of every reachable provider (de-duplicated,
    /// first-seen order). A provider that errors is skipped rather than
    /// failing the whole call — the same "any healthy provider keeps the
    /// chain usable" principle as [`Self::health_check`]. Only when *every*
    /// provider fails do we surface the last error.
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let mut models: Vec<String> = Vec::new();
        let mut last_err = LlmError::ProviderUnavailable("no providers configured".into());
        let mut any_ok = false;
        for provider in &self.providers {
            match provider.list_models().await {
                Ok(list) => {
                    any_ok = true;
                    for m in list {
                        if !models.contains(&m) {
                            models.push(m);
                        }
                    }
                }
                Err(e) => last_err = e,
            }
        }
        if any_ok {
            Ok(models)
        } else {
            Err(last_err)
        }
    }

    async fn fetch_context_window(&self) -> Option<usize> {
        self.providers.first()?.fetch_context_window().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ResponseChunk;

    /// A stub provider with controllable health + model list. `generate` is
    /// irrelevant to these tests; the suite exercises health/model aggregation.
    struct StubProvider {
        healthy: bool,
        models: Vec<String>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for StubProvider {
        async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
            Ok(Response::text("stub", None))
        }
        async fn generate_stream(
            &self,
            _messages: &[Message],
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
        {
            unimplemented!()
        }
        async fn health_check(&self) -> bool {
            self.healthy
        }
        fn name(&self) -> &str {
            "stub"
        }
        fn model(&self) -> &str {
            "stub"
        }
        async fn list_models(&self) -> Result<Vec<String>, LlmError> {
            if self.healthy {
                Ok(self.models.clone())
            } else {
                Err(LlmError::ProviderUnavailable("stub down".into()))
            }
        }
    }

    fn stub(healthy: bool, models: &[&str]) -> Box<dyn LlmProvider> {
        Box::new(StubProvider {
            healthy,
            models: models.iter().map(|s| s.to_string()).collect(),
        })
    }

    #[tokio::test]
    async fn health_check_is_true_when_any_provider_is_up() {
        // Primary down, fallback up — the chain is still serviceable. This is
        // the regression for the old "only checks first provider" bug.
        let chain = FailoverProvider::new(vec![stub(false, &[]), stub(true, &["m"])]);
        assert!(chain.health_check().await);
    }

    #[tokio::test]
    async fn health_check_is_false_only_when_all_providers_are_down() {
        let chain = FailoverProvider::new(vec![stub(false, &[]), stub(false, &[])]);
        assert!(!chain.health_check().await);
    }

    #[tokio::test]
    async fn list_models_unions_reachable_providers_and_dedups() {
        let chain = FailoverProvider::new(vec![
            stub(true, &["a", "b"]),
            stub(false, &["x"]), // down — skipped, not fatal
            stub(true, &["b", "c"]),
        ]);
        let models = chain.list_models().await.unwrap();
        assert_eq!(
            models,
            vec!["a", "b", "c"],
            "first-seen order, de-duplicated"
        );
    }

    #[tokio::test]
    async fn list_models_errors_only_when_every_provider_fails() {
        let chain = FailoverProvider::new(vec![stub(false, &[]), stub(false, &[])]);
        assert!(chain.list_models().await.is_err());
    }
}
