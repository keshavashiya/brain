//! Capability-category intent handler: the kernel-routed Standardized
//! Intent Token envelope. When an [`intent::IntentRouter`] is wired we
//! resolve the token to a [`intent::ToolRoute`] and dispatch via the
//! matching backend (MCP / terminal / native); otherwise we return a
//! deterministic "router not configured" placeholder.
//!
//! Variant: [`thalamus::Intent::ToolCall`].

use uuid::Uuid;

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{CapabilityAuth, CapabilityHandler, HandlerContext, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

impl CapabilityAuth for SignalProcessor {
    fn auth_capability(intent: &thalamus::Intent) -> Option<(AuthorizationRequest, Tier)> {
        match intent {
            // Derive verb_ns / action from the SIT, infer tier from the verb.
            // Conservative defaults (see `tier_for_verb`): destructive verbs
            // bump to `Destructive`; HTTP / mount verbs bump to `External`;
            // unknown lands at `Execute` so the gate prompts the user.
            thalamus::Intent::ToolCall(token) => {
                let verb_ns = token.verb.namespace.as_str();
                let verb_action = token.verb.action.as_str();
                let tier = crate::authz::tier_for_verb(verb_ns, verb_action);
                let req = AuthorizationRequest::new(verb_ns, verb_action)
                    .with_modifiers(token.object.value.clone());
                Some((req, tier))
            }
            _ => None,
        }
    }
}

#[async_trait::async_trait]
impl CapabilityHandler for SignalProcessor {
    async fn dispatch_capability(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::ToolCall(token) => {
                self.handle_tool_call(ctx.signal_id, *token, prepend_nudges)
                    .await
            }
            other => unreachable!(
                "non-capability variant routed to dispatch_capability: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    pub(super) async fn handle_tool_call(
        &self,
        signal_id: Uuid,
        token: intent::IntentToken,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let Some(router) = self.intent_router() else {
            let message = format!(
                "Capability router not configured; cannot resolve tool call '{}.{}'.",
                token.verb.namespace, token.verb.action
            );
            let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
            return Ok(PipelineResult::Complete(resp));
        };
        let route = match router.resolve(&token).await {
            Ok(r) => r,
            Err(e) => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!("Tool resolution failed: {e}"),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };
        let message = match route {
            intent::ToolRoute::Mcp { server, tool } => match self.mcp_host() {
                None => format!(
                    "Resolved '{}.{}' → mcp:{server}:{tool}, but MCP host not configured.",
                    token.verb.namespace, token.verb.action
                ),
                Some(host) => {
                    let args = if token.object.value.is_null() {
                        serde_json::json!({})
                    } else {
                        token.object.value.clone()
                    };
                    let tool_id = format!("mcp:{server}:{tool}");
                    let outcome_result = host.call(&server, &tool, args).await;
                    // Record into the per-tool breaker (if wired). A
                    // transport error or an `is_error: true` outcome both
                    // count as failures; otherwise success.
                    if let Some(breakers) = self.breaker_registry() {
                        let healthy = matches!(&outcome_result, Ok(o) if !o.is_error);
                        if healthy {
                            breakers.record_success(&tool_id).await;
                        } else {
                            breakers.record_failure(&tool_id).await;
                        }
                    }
                    match outcome_result {
                        Ok(outcome) => {
                            let status = if outcome.is_error { "error" } else { "ok" };
                            let body = serde_json::to_string(&outcome.content)
                                .unwrap_or_else(|_| "<unserializable>".into());
                            format!(
                                "mcp:{}:{} ({status}, {}ms): {body}",
                                outcome.server, outcome.tool, outcome.elapsed_ms,
                            )
                        }
                        Err(e) => format!("Tool call mcp:{server}:{tool} failed: {e}"),
                    }
                }
            },
            intent::ToolRoute::HumanConfirm { ask } => ask,
            intent::ToolRoute::Terminal { session_hint } => format!(
                "Terminal routing for '{}.{}' is not yet wired (session_hint={:?}).",
                token.verb.namespace, token.verb.action, session_hint
            ),
            intent::ToolRoute::NativeBackend { backend } => format!(
                "Native-backend routing for '{}.{}' → {} is not yet wired.",
                token.verb.namespace,
                token.verb.action,
                backend.as_str()
            ),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }
}

#[cfg(test)]
mod tool_call_dispatch_tests {
    use super::*;
    use std::sync::Arc;

    async fn make_processor() -> SignalProcessor {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);
        processor
    }

    fn sample_token(verb_ns: &str, verb_action: &str) -> intent::IntentToken {
        intent::IntentToken::new(
            intent::Verb::new(verb_ns, verb_action),
            intent::Object {
                kind: "intent_args".into(),
                value: serde_json::json!({ "text": "hi" }),
            },
            intent::Provenance::User {
                raw_input: format!("/{verb_ns} {verb_action}"),
                ui_origin: None,
                ts: chrono::Utc::now(),
            },
            "personal".into(),
        )
    }

    fn identity(r: SignalResponse) -> SignalResponse {
        r
    }

    fn body(resp: SignalResponse) -> String {
        match resp.response {
            ResponseContent::Text(t) => t,
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn router_not_configured_returns_placeholder() {
        let processor = make_processor().await;
        let result = processor
            .handle_tool_call(uuid::Uuid::new_v4(), sample_token("fs", "read"), &identity)
            .await
            .unwrap();
        match result {
            PipelineResult::Complete(resp) => {
                let t = body(resp);
                assert!(t.contains("Capability router not configured"), "{t}");
                assert!(t.contains("fs.read"), "{t}");
            }
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    #[tokio::test]
    async fn router_no_candidates_renders_human_confirm() {
        let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
        let router: Arc<dyn intent::IntentRouter> =
            Arc::new(intent::DefaultIntentRouter::new(registry));
        let processor = make_processor().await.with_intent_router(router);
        let result = processor
            .handle_tool_call(
                uuid::Uuid::new_v4(),
                sample_token("memory", "store"),
                &identity,
            )
            .await
            .unwrap();
        match result {
            PipelineResult::Complete(resp) => {
                let t = body(resp);
                assert!(t.contains("memory.store"), "{t}");
                assert!(t.contains("No tool registered"), "{t}");
            }
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    #[tokio::test]
    async fn router_resolves_to_mcp_but_host_unwired() {
        let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
        registry
            .register(intent::ToolDescriptor {
                tool_id: "mcp:echo:echo".into(),
                source: intent::ToolSource::McpServer {
                    server: "echo".into(),
                },
                verb: intent::Verb::new("mcp", "echo"),
                description: "echo".into(),
                input_schema: serde_json::json!({ "type": "object" }),
                output_schema: None,
                capabilities: vec![],
                annotations: intent::ToolAnnotations::default(),
                usage: intent::ToolUsage::default(),
                embedding: None,
            })
            .await
            .unwrap();
        let router: Arc<dyn intent::IntentRouter> =
            Arc::new(intent::DefaultIntentRouter::new(registry));
        let processor = make_processor().await.with_intent_router(router);
        let result = processor
            .handle_tool_call(uuid::Uuid::new_v4(), sample_token("mcp", "echo"), &identity)
            .await
            .unwrap();
        match result {
            PipelineResult::Complete(resp) => {
                let t = body(resp);
                assert!(t.contains("mcp:echo:echo"), "{t}");
                assert!(t.contains("MCP host not configured"), "{t}");
            }
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    #[tokio::test]
    async fn router_resolves_to_mcp_with_host_renders_transport_error() {
        let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
        registry
            .register(intent::ToolDescriptor {
                tool_id: "mcp:echo:echo".into(),
                source: intent::ToolSource::McpServer {
                    server: "echo".into(),
                },
                verb: intent::Verb::new("mcp", "echo"),
                description: "echo".into(),
                input_schema: serde_json::json!({ "type": "object" }),
                output_schema: None,
                capabilities: vec![],
                annotations: intent::ToolAnnotations::default(),
                usage: intent::ToolUsage::default(),
                embedding: None,
            })
            .await
            .unwrap();
        let router: Arc<dyn intent::IntentRouter> =
            Arc::new(intent::DefaultIntentRouter::new(registry));
        // Use the no-transport in-memory host. Mount echo so call() reaches
        // the no-transport error path rather than NotMounted.
        let host: Arc<dyn mcphost::MCPHost> = Arc::new(mcphost::InMemoryMcpHost::new());
        host.mount(
            "echo".into(),
            mcphost::ServerConfig::Stdio {
                command: "echo".into(),
                args: vec![],
                env: Default::default(),
                cwd: None,
            },
        )
        .await
        .unwrap();
        let processor = make_processor()
            .await
            .with_intent_router(router)
            .with_mcp_host(host);
        let result = processor
            .handle_tool_call(uuid::Uuid::new_v4(), sample_token("mcp", "echo"), &identity)
            .await
            .unwrap();
        match result {
            PipelineResult::Complete(resp) => {
                let t = body(resp);
                assert!(t.contains("Tool call mcp:echo:echo failed"), "{t}");
                assert!(t.contains("no transport configured"), "{t}");
            }
            _ => panic!("expected PipelineResult::Complete"),
        }
    }
}
