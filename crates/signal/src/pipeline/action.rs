//! Action-category intent handlers: external side effects (shell / net /
//! agent delegation). The umbrella [`SignalProcessor::handle_action`]
//! covers [`thalamus::Intent::WebSearch`], [`thalamus::Intent::Schedule`]
//! (lifecycle-categorised but transported through the same dispatcher),
//! [`thalamus::Intent::SendMessage`], and
//! [`thalamus::Intent::ExecuteCommand`] by routing through the configured
//! [`cortex::actions::ActionDispatcher`].
//!
//! Variants: [`thalamus::Intent::ExecuteCommand`],
//! [`thalamus::Intent::WebSearch`], [`thalamus::Intent::SendMessage`],
//! [`thalamus::Intent::DelegateTask`].

use uuid::Uuid;

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{ActionAuth, ActionHandler, HandlerContext, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

/// System prompt for grounding a chat answer in web-search material. Kept as a
/// named constant (not inline at the call site) so every LLM prompt in this
/// crate is auditable in one read.
const WEB_SYNTHESIS_SYSTEM: &str = "You are Brain OS. Answer the user's question \
     using the supplied research material. Be concise, \
     cite sources by URL, and never invent content not \
     present in the material.";

/// Render the user-turn that hands the model the question + research material
/// for web-search synthesis.
fn web_synthesis_user(question: &str, material: &str) -> String {
    format!(
        "The user asked: \"{question}\"\n\nResearch material:\n{material}\n\n\
         Answer the user's question grounded in the material above. \
         The `Linked sources` block (when present) is content fetched \
         directly from URLs the user pasted — treat it as authoritative \
         over the generic search hits. Quote page titles and URLs when \
         you reference them. If the material is silent on the user's \
         question, say so honestly instead of speculating."
    )
}

impl ActionAuth for SignalProcessor {
    fn auth_action(intent: &thalamus::Intent) -> Option<(AuthorizationRequest, Tier)> {
        match intent {
            thalamus::Intent::ExecuteCommand { .. } => {
                Some((AuthorizationRequest::new("shell", "exec"), Tier::Execute))
            }
            thalamus::Intent::WebSearch { .. } => {
                Some((AuthorizationRequest::new("net", "http"), Tier::External))
            }
            thalamus::Intent::SendMessage { .. } => {
                Some((AuthorizationRequest::new("notify", "send"), Tier::External))
            }
            thalamus::Intent::DelegateTask { .. } => Some((
                AuthorizationRequest::new("agent", "delegate"),
                Tier::Execute,
            )),
            _ => None,
        }
    }
}

#[async_trait::async_trait]
impl ActionHandler for SignalProcessor {
    async fn dispatch_action(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::DelegateTask { agent, prompt } => {
                self.handle_delegate_task(ctx.signal_id, agent, prompt, prepend_nudges)
                    .await
            }
            intent @ (thalamus::Intent::WebSearch { .. }
            | thalamus::Intent::SendMessage { .. }
            | thalamus::Intent::ExecuteCommand { .. }) => {
                self.handle_action(ctx.signal_id, ctx.signal, &intent, prepend_nudges)
                    .await
            }
            other => unreachable!(
                "non-action variant routed to dispatch_action: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    pub(super) async fn handle_delegate_task(
        &self,
        signal_id: Uuid,
        agent: String,
        prompt: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let registry = match self.agent_registry() {
            Some(r) => r,
            None => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    "Agent registry is not wired — delegation unavailable.".to_string(),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };

        if prompt.trim().is_empty() {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                format!("Asked to delegate to '{agent}' but no prompt was supplied."),
            ));
            return Ok(PipelineResult::Complete(resp));
        }

        let delegate = match registry.get(&agent) {
            Ok(d) => d,
            Err(e) => {
                let known: Vec<String> = registry.list();
                let hint = if known.is_empty() {
                    "no agents are currently registered.".to_string()
                } else {
                    format!("registered: {}", known.join(", "))
                };
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!("Could not delegate to '{agent}': {e}. {hint}"),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };

        let task = delegate::AgentTask::new(prompt.clone());
        let task_id = task.id.clone();
        match delegate.delegate(task).await {
            Ok(result) => {
                let summary = if result.summary.trim().is_empty() {
                    result.stdout.clone()
                } else {
                    result.summary.clone()
                };
                let body = if summary.trim().is_empty() {
                    format!(
                        "Delegate '{agent}' completed (status: {:?}, task_id: {}). \
                         No summary produced.",
                        result.status, task_id
                    )
                } else {
                    format!(
                        "Delegate '{agent}' ({:?}, task_id: {}):\n\n{}",
                        result.status, task_id, summary
                    )
                };
                let resp = prepend_nudges(SignalResponse::ok(signal_id, body));
                Ok(PipelineResult::Complete(resp))
            }
            Err(e) => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!("Delegate '{agent}' failed: {e}"),
                ));
                Ok(PipelineResult::Complete(resp))
            }
        }
    }

    pub(super) async fn handle_action(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        intent: &thalamus::Intent,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let router = thalamus::SignalRouter::new();
        let resp = match (
            router.intent_to_action(intent),
            &self.capability.action_dispatcher,
        ) {
            (Some(action), Some(dispatcher)) => {
                let result = dispatcher.dispatch(&action).await;
                if result.success {
                    if matches!(&action, cortex::actions::Action::WebSearch { .. })
                        && !result.output.is_empty()
                    {
                        let search_context = web_synthesis_user(&signal.content, &result.output);
                        let messages = vec![
                            cortex::llm::Message::system(WEB_SYNTHESIS_SYSTEM),
                            cortex::llm::Message::user(search_context),
                        ];
                        match self.llm.generate(&messages).await {
                            Ok(llm_response) => SignalResponse::ok(signal_id, llm_response.content),
                            Err(_) => SignalResponse::ok(signal_id, result.output),
                        }
                    } else {
                        SignalResponse::ok(signal_id, result.output)
                    }
                } else {
                    SignalResponse::error(
                        signal_id,
                        result.error.unwrap_or_else(|| "Action failed".to_string()),
                    )
                }
            }
            (Some(_action), None) => SignalResponse::error(
                signal_id,
                format!(
                    "Action {:?} recognized but no dispatcher configured — \
                     enable the relevant backend in config",
                    intent
                ),
            ),
            (None, _) => SignalResponse::ok(signal_id, format!("Intent classified: {:?}", intent)),
        };
        let resp = prepend_nudges(resp);
        Ok(PipelineResult::Complete(resp))
    }
}
