//! Chat tool-use loop: the second consumer of the unified capability
//! manifest (the first being the SOUL digest / `tools/list`).
//!
//! A normal Chat turn now advertises a small, relevance-ranked slice of the
//! manifest to the model over the provider tools channel. When the model
//! proposes a call we resolve it to an [`intent::IntentToken`], run the
//! **same** tier-based consent gate every other capability invocation uses
//! ([`SignalProcessor::confirmation_gate`]), execute approved calls through
//! the shared [`dispatch_tool_route`](SignalProcessor::dispatch_tool_route),
//! and feed the result back as a [`cortex::Role::Tool`] turn — looping until
//! the model answers in plain text or we hit [`MAX_TOOL_ROUNDS`].
//!
//! Awareness ≠ permission: advertising a tool only lets the model *propose*
//! it. Nothing executes until the consent gate clears for that call's tier.
//!
//! Degrades to a plain `generate` when no tool registry is wired or the
//! manifest is empty, so a bare deployment behaves exactly as before.

use chrono::Utc;

use cortex::llm::{Message, ProposedToolCall, Response, ToolDef, Usage};

use crate::types::*;
use crate::SignalProcessor;

/// Max model⇄tool round-trips before we stop looping and return the model's
/// last text. Bounds cost and stops a model that keeps proposing calls from
/// looping forever.
const MAX_TOOL_ROUNDS: usize = 4;

/// How many tools we advertise per turn — a relevance-ranked slice of the
/// manifest, never the whole catalogue (keeps the prompt small and focused).
const TOOL_ADVERTISE_K: usize = 8;

impl SignalProcessor {
    /// Run one chat turn, using the tools channel when a manifest is wired.
    ///
    /// Returns the final assistant [`Response`]; its `usage` is summed across
    /// every round so the caller's budget accounting covers the whole turn.
    pub(super) async fn run_chat_turn(
        &self,
        signal: &Signal,
        signal_id: uuid::Uuid,
        mut messages: Vec<Message>,
    ) -> Result<Response, SignalError> {
        let tools = self.advertised_tools(&messages).await;
        if tools.is_empty() {
            // No manifest / no tools → unchanged plain-text behaviour.
            return Ok(self.llm.generate(&messages).await?);
        }

        let mut total_usage: Option<Usage> = None;
        let mut last = Response::default();

        for round in 0..MAX_TOOL_ROUNDS {
            let resp = self.llm.generate_with_tools(&messages, &tools).await?;
            accumulate_usage(&mut total_usage, resp.usage.as_ref());

            // Plain text answer, or we've exhausted our round budget — done.
            if resp.tool_calls.is_empty() || round + 1 == MAX_TOOL_ROUNDS {
                last = resp;
                break;
            }

            // Replay the assistant tool-call turn, then resolve each proposed
            // call through consent + dispatch and append its result.
            let calls = resp.tool_calls.clone();
            messages.push(Message::assistant_with_tool_calls(
                resp.content.clone(),
                calls.clone(),
            ));
            for call in &calls {
                let outcome = self.resolve_proposed_call(signal, signal_id, call).await;
                let id = call.id.clone().unwrap_or_else(|| call.name.clone());
                messages.push(Message::tool_result(id, outcome));
            }
            last = resp;
        }

        last.usage = total_usage;
        Ok(last)
    }

    /// Build the relevance-ranked [`ToolDef`] slice advertised to the model
    /// this turn. Sourced from the unified [`intent::ToolRegistry`] manifest
    /// (native + MCP), scored against the latest user message, and capped at
    /// [`TOOL_ADVERTISE_K`]. Empty when no registry is wired.
    ///
    /// Each description is routed through
    /// [`intent::sanitization::render_tool_description_for_prompt`] before it
    /// reaches a provider — an MCP server's description is untrusted text.
    async fn advertised_tools(&self, messages: &[Message]) -> Vec<ToolDef> {
        let Some(registry) = self.tool_registry() else {
            return Vec::new();
        };
        let manifest = registry.list().await;
        if manifest.is_empty() {
            return Vec::new();
        }
        let query = latest_user_text(messages);
        let ranked = score_top_k(manifest, &query, TOOL_ADVERTISE_K);
        ranked
            .into_iter()
            .map(|t| ToolDef {
                name: t.verb.dotted(),
                description: intent::sanitization::render_tool_description_for_prompt(
                    &t.verb.namespace,
                    &t.verb.action,
                    &t.description,
                ),
                parameters: t.input_schema,
            })
            .collect()
    }

    /// Resolve, consent-gate, and execute one model-proposed call, returning
    /// the result text fed back to the model. A denied/timed-out gate yields
    /// the gate's own message (so the model learns the call was refused) and
    /// nothing executes.
    async fn resolve_proposed_call(
        &self,
        signal: &Signal,
        signal_id: uuid::Uuid,
        call: &ProposedToolCall,
    ) -> String {
        let Some(router) = self.intent_router() else {
            return "Capability router not configured; tool call skipped.".to_string();
        };
        let token = proposed_call_to_token(call, self.llm.model());
        let intent = thalamus::Intent::ToolCall(Box::new(token.clone()));

        // Same tier-based consent gate every other capability invocation uses.
        if let Some(blocked) = self.confirmation_gate(signal, signal_id, &intent).await {
            return format!("Tool call refused: {}", response_text(&blocked));
        }

        self.dispatch_tool_route(router.as_ref(), &token).await
    }
}

/// The most recent user-turn content, used to rank the advertised tools.
fn latest_user_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == cortex::Role::User)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// Parse a model-proposed call into an [`intent::IntentToken`]. The tool name
/// is split on the first `.` into `(namespace, action)`; a dotless name lands
/// in the empty namespace. Arguments become the token object verbatim.
fn proposed_call_to_token(call: &ProposedToolCall, model: &str) -> intent::IntentToken {
    let (ns, action) = match call.name.split_once('.') {
        Some((ns, action)) => (ns.to_string(), action.to_string()),
        None => (String::new(), call.name.clone()),
    };
    intent::IntentToken::new(
        intent::Verb::new(ns.clone(), action),
        intent::Object {
            kind: "json".to_string(),
            value: call.arguments.clone(),
        },
        intent::Provenance::Llm {
            model: model.to_string(),
            call_id: call.id.clone().unwrap_or_default(),
            raw_input: None,
            ts: Utc::now(),
        },
        ns,
    )
}

/// Extract the human-readable text from a gate's [`SignalResponse`].
fn response_text(resp: &SignalResponse) -> String {
    match &resp.response {
        ResponseContent::Text(t) | ResponseContent::Error(t) => t.clone(),
        ResponseContent::Json(v) => v.to_string(),
    }
}

/// Fold one round's usage into the running total (treats `None` as zero).
fn accumulate_usage(total: &mut Option<Usage>, round: Option<&Usage>) {
    let Some(round) = round else { return };
    let acc = total.get_or_insert(Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });
    acc.prompt_tokens += round.prompt_tokens;
    acc.completion_tokens += round.completion_tokens;
    acc.total_tokens += round.total_tokens;
}

/// Rank `tools` by keyword overlap with `query` and keep the best `k`.
///
/// Case-insensitive alphanumeric term overlap over each tool's dotted verb
/// and description — no model/embedding dependency. Sorted by score then verb
/// for stability; zero-score tools fill remaining slots up to `k`, so a query
/// that matches nothing still advertises *some* tools rather than none.
fn score_top_k(
    mut tools: Vec<intent::ToolDescriptor>,
    query: &str,
    k: usize,
) -> Vec<intent::ToolDescriptor> {
    if k == 0 {
        return Vec::new();
    }
    let terms = tokenize(query);
    tools.sort_by(|a, b| {
        let sa = score_tool(a, &terms);
        let sb = score_tool(b, &terms);
        sb.cmp(&sa)
            .then_with(|| a.verb.dotted().cmp(&b.verb.dotted()))
    });
    tools.truncate(k);
    tools
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

fn score_tool(tool: &intent::ToolDescriptor, terms: &[String]) -> usize {
    if terms.is_empty() {
        return 0;
    }
    let haystack = format!("{} {}", tool.verb.dotted(), tool.description).to_lowercase();
    terms
        .iter()
        .filter(|t| haystack.contains(t.as_str()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(ns: &str, action: &str, description: &str) -> intent::ToolDescriptor {
        intent::ToolDescriptor {
            tool_id: format!("{ns}.{action}"),
            source: intent::ToolSource::NativeBackend {
                backend: intent::BackendId(ns.to_string()),
            },
            verb: intent::Verb::new(ns, action),
            description: description.to_string(),
            input_schema: serde_json::json!({"type": "object"}),
            output_schema: None,
            capabilities: Vec::new(),
            annotations: Default::default(),
            usage: Default::default(),
            embedding: None,
        }
    }

    #[test]
    fn score_top_k_ranks_keyword_matches_first() {
        let tools = vec![
            td("fs", "read_text_file", "Read the contents of a file"),
            td("web", "search", "Search the web for a query"),
            td("git", "commit", "Record changes to the repository"),
        ];
        let hits = score_top_k(tools, "search the web", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].verb.dotted(), "web.search");
    }

    #[test]
    fn score_top_k_caps_and_zero_short_circuits() {
        let tools = vec![
            td("fs", "read", "read"),
            td("fs", "write", "write"),
            td("fs", "list", "list"),
        ];
        assert_eq!(score_top_k(tools.clone(), "file", 2).len(), 2);
        assert!(score_top_k(tools, "file", 0).is_empty());
    }

    #[test]
    fn score_top_k_no_match_fills_slots_verb_sorted() {
        let tools = vec![
            td("git", "commit", "x"),
            td("fs", "read", "y"),
            td("web", "search", "z"),
        ];
        let hits = score_top_k(tools, "zzz_no_match", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].verb.dotted(), "fs.read");
        assert_eq!(hits[1].verb.dotted(), "git.commit");
    }

    #[test]
    fn proposed_call_splits_namespace_on_first_dot() {
        let call = ProposedToolCall {
            id: Some("call_1".into()),
            name: "fs.read_text_file".into(),
            arguments: serde_json::json!({ "path": "/tmp/x" }),
        };
        let token = proposed_call_to_token(&call, "test-model");
        assert_eq!(token.verb.namespace, "fs");
        assert_eq!(token.verb.action, "read_text_file");
        assert_eq!(token.namespace, "fs");
        assert_eq!(token.object.value, serde_json::json!({ "path": "/tmp/x" }));
        match token.provenance {
            intent::Provenance::Llm { model, call_id, .. } => {
                assert_eq!(model, "test-model");
                assert_eq!(call_id, "call_1");
            }
            other => panic!("expected Llm provenance, got {other:?}"),
        }
    }

    #[test]
    fn proposed_call_dotless_name_lands_in_empty_namespace() {
        let call = ProposedToolCall {
            id: None,
            name: "ping".into(),
            arguments: serde_json::Value::Null,
        };
        let token = proposed_call_to_token(&call, "m");
        assert_eq!(token.verb.namespace, "");
        assert_eq!(token.verb.action, "ping");
    }

    #[test]
    fn latest_user_text_finds_most_recent_user_turn() {
        let messages = vec![
            Message::user("first"),
            Message::assistant("reply"),
            Message::user("second"),
            Message::tool_result("id", "result"),
        ];
        assert_eq!(latest_user_text(&messages), "second");
    }

    #[test]
    fn accumulate_usage_sums_rounds_and_ignores_none() {
        let mut total = None;
        accumulate_usage(&mut total, None);
        assert!(total.is_none());
        accumulate_usage(
            &mut total,
            Some(&Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        );
        accumulate_usage(
            &mut total,
            Some(&Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            }),
        );
        let total = total.unwrap();
        assert_eq!(total.prompt_tokens, 13);
        assert_eq!(total.completion_tokens, 7);
        assert_eq!(total.total_tokens, 20);
    }

    #[test]
    fn response_text_extracts_each_variant() {
        let id = uuid::Uuid::new_v4();
        assert_eq!(
            response_text(&SignalResponse::ok(id, "hello")),
            "hello".to_string()
        );
        assert_eq!(
            response_text(&SignalResponse::error(id, "boom")),
            "boom".to_string()
        );
    }
}
