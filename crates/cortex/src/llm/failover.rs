use std::pin::Pin;

use futures::Stream;

use super::{LlmError, LlmProvider, Message, Response, ResponseChunk};

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
pub struct FalloverProvider {
    providers: Vec<Box<dyn LlmProvider>>,
}

impl FalloverProvider {
    pub fn new(providers: Vec<Box<dyn LlmProvider>>) -> Self {
        Self { providers }
    }
}

#[async_trait::async_trait]
impl LlmProvider for FalloverProvider {
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

    async fn health_check(&self) -> bool {
        match self.providers.first() {
            Some(p) => p.health_check().await,
            None => false,
        }
    }

    fn name(&self) -> &str {
        self.providers.first().map(|p| p.name()).unwrap_or("none")
    }

    fn model(&self) -> &str {
        self.providers.first().map(|p| p.model()).unwrap_or("none")
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.providers
            .first()
            .ok_or_else(|| LlmError::ProviderUnavailable("no providers".into()))?
            .list_models()
            .await
    }
}
