//! Memory-category intent handlers: episodic / semantic mutations.
//!
//! Variants: [`thalamus::Intent::StoreFact`], [`thalamus::Intent::Forget`].

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
                self.handle_store_fact(&ctx, subject, predicate, object, prepend_nudges)
                    .await
            }
            thalamus::Intent::Forget { target } => {
                self.handle_forget(&ctx, target, prepend_nudges).await
            }
            other => unreachable!(
                "non-memory variant routed to dispatch_memory: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    pub(super) async fn handle_store_fact(
        &self,
        ctx: &HandlerContext<'_>,
        subject: String,
        predicate: String,
        object: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let &HandlerContext {
            signal_id,
            signal,
            importance,
            ..
        } = ctx;
        let namespace = &signal.namespace;
        let agent = signal.agent.as_deref();
        let fact_text = format!("{subject} {predicate} {object}");
        let vector = self.embed_text(&fact_text, namespace).await;

        let mut facts_stored = 0;
        if let Some(semantic) = &self.memory.semantic {
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
                Ok(id) => {
                    self.quarantine_fact_if_unattested(&id, agent).await;
                    facts_stored = 1;
                }
                Err(e) => tracing::warn!("Failed to store fact in semantic memory: {e}"),
            }
        }

        let resp = prepend_nudges(SignalResponse {
            signal_id,
            status: ResponseStatus::Ok,
            // User-facing confirmation stays clean — the importance score is
            // internal ranking jargon, not something the user asked about.
            response: ResponseContent::Text(format!("Stored: {subject} {predicate} {object}")),
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
        ctx: &HandlerContext<'_>,
        target: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let &HandlerContext {
            signal_id, signal, ..
        } = ctx;
        let mut deleted_count = 0usize;

        if let Some(semantic) = &self.memory.semantic {
            match semantic.find_facts_matching(&target, Some(&signal.namespace)) {
                Ok(facts) if !facts.is_empty() => {
                    let ids: Vec<&str> = facts.iter().map(|f| f.id.as_str()).collect();
                    match semantic.delete_facts_batch(&ids).await {
                        Ok(n) => deleted_count = n,
                        Err(e) => {
                            tracing::warn!("Forget batch delete failed: {e}");
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
