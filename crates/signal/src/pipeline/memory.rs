//! Memory-category intent handlers: episodic / semantic mutations.
//!
//! Variants: [`thalamus::Intent::StoreFact`], [`thalamus::Intent::Forget`].

use uuid::Uuid;

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{HandlerContext, MemoryAuth, MemoryHandler, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

impl MemoryAuth for SignalProcessor {
    fn auth_memory(intent: &thalamus::Intent) -> Option<(AuthorizationRequest, Tier)> {
        match intent {
            thalamus::Intent::StoreFact { .. } => {
                Some((AuthorizationRequest::new("memory", "store"), Tier::Write))
            }
            thalamus::Intent::Forget { .. } => Some((
                AuthorizationRequest::new("memory", "delete"),
                Tier::Destructive,
            )),
            _ => None,
        }
    }
}

#[async_trait::async_trait]
impl MemoryHandler for SignalProcessor {
    async fn dispatch_memory(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::StoreFact {
                subject,
                predicate,
                object,
            } => {
                self.handle_store_fact(
                    ctx.signal_id,
                    &ctx.signal.namespace,
                    ctx.signal.agent.as_deref(),
                    subject,
                    predicate,
                    object,
                    ctx.importance,
                    prepend_nudges,
                )
                .await
            }
            thalamus::Intent::Forget { target } => {
                self.handle_forget(ctx.signal_id, ctx.signal, target, prepend_nudges)
                    .await
            }
            other => unreachable!(
                "non-memory variant routed to dispatch_memory: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_store_fact(
        &self,
        signal_id: Uuid,
        namespace: &str,
        agent: Option<&str>,
        subject: String,
        predicate: String,
        object: String,
        importance: f32,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let fact_text = format!("{subject} {predicate} {object}");
        let vector = self.embed_text(&fact_text).await;

        let mut facts_stored = 0;
        if let Some(semantic) = &self.semantic {
            match semantic
                .store_fact(
                    namespace,
                    "signal",
                    &subject,
                    &predicate,
                    &object,
                    importance as f64,
                    None,
                    vector,
                    agent,
                )
                .await
            {
                Ok(_) => facts_stored = 1,
                Err(e) => tracing::warn!("Failed to store fact in semantic memory: {e}"),
            }
        }

        let resp = prepend_nudges(SignalResponse {
            signal_id,
            status: ResponseStatus::Ok,
            response: ResponseContent::Text(format!(
                "Stored: {subject} {predicate} {object} (importance: {importance:.2})"
            )),
            memory_context: MemoryContext {
                facts_used: facts_stored,
                episodes_used: 0,
            },
            session_id: None,
        });
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_forget(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        target: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let mut deleted_count = 0usize;

        if let Some(semantic) = &self.semantic {
            match semantic.find_facts_matching(&target, Some(&signal.namespace)) {
                Ok(facts) if !facts.is_empty() => {
                    for fact in &facts {
                        if let Err(e) = semantic.delete_fact(&fact.id).await {
                            tracing::warn!(fact_id = %fact.id, "Failed to delete fact: {e}");
                        } else {
                            deleted_count += 1;
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => tracing::warn!("Forget search failed: {e}"),
            }
        }

        let message = if deleted_count > 0 {
            format!("Memory erased: removed {deleted_count} engram(s) matching \"{target}\"")
        } else {
            format!("No engrams found matching \"{target}\" to erase")
        };

        let resp = prepend_nudges(SignalResponse {
            signal_id,
            status: ResponseStatus::Ok,
            response: ResponseContent::Text(message),
            memory_context: MemoryContext {
                facts_used: 0,
                episodes_used: 0,
            },
            session_id: None,
        });
        Ok(PipelineResult::Complete(resp))
    }
}
