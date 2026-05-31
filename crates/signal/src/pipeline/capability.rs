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
        let message = self.dispatch_tool_route(router.as_ref(), &token).await;
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    /// Resolve `token` to a [`intent::ToolRoute`] and execute it, returning a
    /// readable outcome string. Shared by the `Intent::ToolCall` handler and
    /// the chat tool-use loop. Records breaker outcomes for MCP calls.
    ///
    /// Consent is **not** gated here — callers run
    /// [`confirmation_gate`](SignalProcessor::confirmation_gate) on the
    /// wrapping intent first.
    pub(super) async fn dispatch_tool_route(
        &self,
        router: &dyn intent::IntentRouter,
        token: &intent::IntentToken,
    ) -> String {
        let route = match router.resolve(token).await {
            Ok(r) => r,
            Err(e) => return format!("Tool resolution failed: {e}"),
        };
        match route {
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
            // Both Terminal-sourced (`shell.exec`) and NativeBackend-sourced
            // verbs execute through the same shared ActionDispatcher the
            // classic intent paths use — see `execute_native_token`. Consent
            // was already gated upstream (see this fn's doc comment).
            intent::ToolRoute::Terminal { .. } | intent::ToolRoute::NativeBackend { .. } => {
                self.execute_native_token(token).await
            }
        }
    }

    /// Execute a resolved native/terminal SIT token by mapping it to a
    /// [`cortex::actions::Action`] and dispatching through the wired
    /// [`ActionDispatcher`](cortex::actions::ActionDispatcher) — the same
    /// executor the classic `Intent::{WebSearch,StoreFact,…}` paths use.
    ///
    /// This is the WS2 fix for the silent-failure trap: previously the chat
    /// tool-loop advertised `net.http`/`shell.exec`/… and cleared the consent
    /// gate, then returned a "not yet wired" placeholder the model paraphrased
    /// into a bare "I couldn't do that." Now an approved call either runs and
    /// returns its real output, or returns an explicit, model-legible reason
    /// (no executor for this verb / backend not configured / backend error) —
    /// never a vague giveup.
    async fn execute_native_token(&self, token: &intent::IntentToken) -> String {
        let ns = token.verb.namespace.as_str();
        let action = token.verb.action.as_str();

        let Some(act) = token_to_action(token) else {
            // No ActionDispatcher executor for this verb. Be specific about
            // why and point at the real path so the model relays the truth
            // rather than inventing a failure.
            return format!(
                "'{ns}.{action}' can't be run from the chat tool-loop. {}",
                native_unexecutable_hint(ns, action)
            );
        };
        let Some(dispatcher) = self.capability.action_dispatcher.as_ref() else {
            return format!(
                "'{ns}.{action}' needs its backend enabled in config, but no \
                 action dispatcher is wired in this deployment."
            );
        };

        let result = dispatcher.dispatch(&act).await;
        if result.success {
            if result.output.trim().is_empty() {
                format!("'{ns}.{action}' completed with no output.")
            } else {
                result.output
            }
        } else {
            format!(
                "'{ns}.{action}' ran but failed: {}",
                result
                    .error
                    .unwrap_or_else(|| "the backend reported no reason".to_string())
            )
        }
    }
}

/// Map a resolved native/terminal SIT token to an executable
/// [`cortex::actions::Action`]. Only verbs with a real ActionDispatcher
/// backend map; everything else returns `None` (handled with an honest
/// reason by [`SignalProcessor::execute_native_token`]).
///
/// Argument extraction is tolerant of the key names the model picks, because
/// the native descriptors advertise a bare `{"type":"object"}` schema (no
/// declared properties) — so the model improvises (`query` vs `q` vs `url`).
fn token_to_action(token: &intent::IntentToken) -> Option<cortex::actions::Action> {
    use cortex::actions::Action;
    let v = &token.object.value;
    match (token.verb.namespace.as_str(), token.verb.action.as_str()) {
        ("net", "http") => Some(Action::WebSearch {
            query: first_str(v, &["query", "q", "url", "text", "input", "prompt"])?,
        }),
        ("shell", "exec") => Some(Action::ExecuteCommand {
            command: first_str(v, &["command", "cmd", "program"])?,
            args: str_array(v, &["args", "arguments"]),
        }),
        ("memory", "store") => Some(Action::StoreFact {
            subject: first_str(v, &["subject"])?,
            predicate: first_str(v, &["predicate", "relation"])?,
            object: first_str(v, &["object", "value"])?,
        }),
        ("schedule", "create") => Some(Action::ScheduleTask {
            description: first_str(v, &["description", "task", "text"])?,
            cron: first_str(v, &["cron", "schedule"]),
        }),
        ("notify", "send") => Some(Action::SendMessage {
            channel: first_str(v, &["channel"])?,
            recipient: first_str(v, &["recipient", "to"]).unwrap_or_default(),
            content: first_str(v, &["content", "message", "text", "body"])?,
        }),
        _ => None,
    }
}

/// Native / terminal SIT verbs the chat tool-loop can actually execute via the
/// shared [`ActionDispatcher`](cortex::actions::ActionDispatcher) — exactly the
/// set [`token_to_action`] maps. The single source of truth the tool-loop
/// advertiser ([`SignalProcessor::advertised_tools`](crate::SignalProcessor))
/// consults so it never surfaces a verb it can't dispatch.
///
/// This is the F6 fix: native backends register into the `ToolRegistry` so the
/// reasoner is *aware* of them, but verbs like `fs.read`, `memory.delete`,
/// `schedule.cancel`, and the `terminal.*` lifecycle verbs are reachable only
/// through *other* paths (path-grounding, the `Forget`/`CancelSchedule`
/// intents, the terminal bridge) — not this loop. Advertising them here would
/// let the SOUL over-promise. A drift-guard test keeps this list in lockstep
/// with `token_to_action`.
pub(crate) const TOOL_LOOP_NATIVE_VERBS: &[(&str, &str)] = &[
    ("net", "http"),
    ("shell", "exec"),
    ("memory", "store"),
    ("schedule", "create"),
    ("notify", "send"),
];

/// True when a native- or terminal-sourced verb has a chat-tool-loop executor
/// (i.e. [`token_to_action`] maps it). MCP-sourced verbs are dispatched by the
/// router/host on a separate path and are always loop-dispatchable, so this
/// predicate is only consulted for native/terminal sources.
pub(crate) fn native_verb_executable_in_tool_loop(ns: &str, action: &str) -> bool {
    TOOL_LOOP_NATIVE_VERBS.contains(&(ns, action))
}

/// Per-verb hint for native verbs that have no chat-loop executor, so the
/// model can tell the user the real path instead of a bare failure.
fn native_unexecutable_hint(ns: &str, action: &str) -> &'static str {
    match (ns, action) {
        ("fs", "read") => {
            "Just name the file path in your message — Brain reads it automatically \
             as grounding; there's no separate fetch step."
        }
        ("memory", "delete") => {
            "Forgetting a fact runs as its own confirmed step, not through this loop."
        }
        ("schedule", "cancel") => {
            "Cancelling a schedule runs as its own step — list the schedules first, \
             then cancel by id."
        }
        ("terminal", "open") | ("terminal", "close") => {
            "Interactive PTY sessions are managed by the terminal bridge, not the \
             chat tool-loop."
        }
        _ => "No native executor is wired for this verb in this deployment.",
    }
}

/// First non-empty string value among `keys` in a JSON object, else `None`.
fn first_str(v: &serde_json::Value, keys: &[&str]) -> Option<String> {
    let obj = v.as_object()?;
    keys.iter().find_map(|k| {
        obj.get(*k)
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    })
}

/// String array under the first present key among `keys`, or empty.
fn str_array(v: &serde_json::Value, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(|x| x.as_array()))
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
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

    fn token_with_args(
        verb_ns: &str,
        verb_action: &str,
        args: serde_json::Value,
    ) -> intent::IntentToken {
        intent::IntentToken::new(
            intent::Verb::new(verb_ns, verb_action),
            intent::Object {
                kind: "json".into(),
                value: args,
            },
            intent::Provenance::User {
                raw_input: format!("/{verb_ns} {verb_action}"),
                ui_origin: None,
                ts: chrono::Utc::now(),
            },
            "personal".into(),
        )
    }

    struct FakeSearch;
    #[async_trait::async_trait]
    impl cortex::actions::WebSearchBackend for FakeSearch {
        async fn search(
            &self,
            query: &str,
            _top_k: usize,
        ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
            Ok(vec![cortex::actions::SearchHit {
                title: format!("Result for {query}"),
                url: "https://example.com".into(),
                snippet: "a snippet".into(),
            }])
        }
    }

    fn web_search_dispatcher() -> cortex::actions::ActionDispatcher {
        let config = cortex::actions::ActionConfig {
            enable_web_search: true,
            ..Default::default()
        };
        cortex::actions::ActionDispatcher::new(config).with_web_search_backend(Arc::new(FakeSearch))
    }

    #[test]
    fn token_to_action_maps_native_verbs_and_is_arg_tolerant() {
        // net.http accepts several arg key names for the query.
        for key in ["query", "q", "url", "text", "input", "prompt"] {
            let token = token_with_args("net", "http", serde_json::json!({ key: "ripgrep" }));
            match token_to_action(&token) {
                Some(cortex::actions::Action::WebSearch { query }) => assert_eq!(query, "ripgrep"),
                other => panic!("net.http via `{key}` -> {other:?}"),
            }
        }
        // shell.exec carries command + args.
        let token = token_with_args(
            "shell",
            "exec",
            serde_json::json!({ "command": "git", "args": ["status"] }),
        );
        match token_to_action(&token) {
            Some(cortex::actions::Action::ExecuteCommand { command, args }) => {
                assert_eq!(command, "git");
                assert_eq!(args, vec!["status".to_string()]);
            }
            other => panic!("shell.exec -> {other:?}"),
        }
        // A verb with no ActionDispatcher executor maps to None.
        assert!(token_to_action(&token_with_args("fs", "read", serde_json::json!({}))).is_none());
        // Missing the required field also yields None (no half-built action).
        assert!(token_to_action(&token_with_args("net", "http", serde_json::json!({}))).is_none());
    }

    /// Drift guard for F6: the advertised-verb allowlist
    /// (`TOOL_LOOP_NATIVE_VERBS`) and the actual executor (`token_to_action`)
    /// must stay in lockstep. Every allowlisted verb maps to an Action when
    /// given well-formed args; the known advertised-but-unexecutable verbs do
    /// not. If someone adds a `token_to_action` arm without listing the verb
    /// (or vice versa), this fails.
    #[test]
    fn tool_loop_native_verbs_agree_with_token_to_action() {
        // Representative well-formed args per allowlisted verb.
        let args_for = |ns: &str, action: &str| -> serde_json::Value {
            match (ns, action) {
                ("net", "http") => serde_json::json!({ "query": "x" }),
                ("shell", "exec") => serde_json::json!({ "command": "ls" }),
                ("memory", "store") => {
                    serde_json::json!({ "subject": "s", "predicate": "p", "object": "o" })
                }
                ("schedule", "create") => serde_json::json!({ "description": "d" }),
                ("notify", "send") => {
                    serde_json::json!({ "channel": "c", "content": "m" })
                }
                _ => serde_json::json!({}),
            }
        };
        for (ns, action) in super::TOOL_LOOP_NATIVE_VERBS {
            let token = token_with_args(ns, action, args_for(ns, action));
            assert!(
                token_to_action(&token).is_some(),
                "{ns}.{action} is allowlisted but token_to_action can't map it",
            );
            assert!(super::native_verb_executable_in_tool_loop(ns, action));
        }
        // Verbs advertised in the manifest but deliberately routed elsewhere
        // must NOT be loop-executable.
        for (ns, action) in [
            ("fs", "read"),
            ("memory", "delete"),
            ("schedule", "cancel"),
            ("terminal", "open"),
            ("terminal", "close"),
        ] {
            assert!(
                !super::native_verb_executable_in_tool_loop(ns, action),
                "{ns}.{action} must not be advertised to the tool-loop",
            );
        }
    }

    #[tokio::test]
    async fn native_token_executes_through_dispatcher() {
        // The WS2 fix: an approved net.http now actually runs and returns the
        // real search material, not a "not yet wired" placeholder.
        let processor = make_processor()
            .await
            .with_action_dispatcher(web_search_dispatcher());
        let out = processor
            .execute_native_token(&token_with_args(
                "net",
                "http",
                serde_json::json!({ "query": "ripgrep latest" }),
            ))
            .await;
        assert!(out.contains("Result for ripgrep latest"), "{out}");
        assert!(out.contains("example.com"), "{out}");
        assert!(!out.contains("not yet wired"), "{out}");
    }

    #[tokio::test]
    async fn native_token_without_executor_gives_honest_reason() {
        // fs.read has no dispatcher executor — the model must get a clear,
        // truthful reason (and the real path) rather than a vague giveup.
        let processor = make_processor()
            .await
            .with_action_dispatcher(web_search_dispatcher());
        let out = processor
            .execute_native_token(&token_with_args("fs", "read", serde_json::json!({})))
            .await;
        assert!(out.contains("fs.read"), "{out}");
        assert!(out.contains("reads it automatically"), "{out}");
        assert!(!out.contains("not yet wired"), "{out}");
    }

    #[tokio::test]
    async fn native_token_surfaces_backend_failure_reason() {
        // A wired-but-failing backend (web search disabled) surfaces the
        // backend's own reason, not a blank failure.
        let dispatcher = cortex::actions::ActionDispatcher::new(cortex::actions::ActionConfig {
            enable_web_search: false,
            ..Default::default()
        });
        let processor = make_processor().await.with_action_dispatcher(dispatcher);
        let out = processor
            .execute_native_token(&token_with_args(
                "net",
                "http",
                serde_json::json!({ "query": "x" }),
            ))
            .await;
        assert!(out.contains("net.http"), "{out}");
        assert!(out.contains("ran but failed"), "{out}");
        assert!(out.to_lowercase().contains("disabled"), "{out}");
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
