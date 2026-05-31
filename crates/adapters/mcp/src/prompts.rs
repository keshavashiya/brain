//! MCP `prompts/list` + `prompts/get` — canned prompt templates that steer an
//! MCP client toward Brain's memory tools for common workflows.

use serde_json::{json, Value};

use crate::protocol::JsonRpcRequest;
use crate::server::McpServer;

const PROMPT_RECALL_CONTEXT: &str = "recall-context";
const PROMPT_DAILY_REVIEW: &str = "daily-review";

/// Rendered body for `recall-context`. `{query}` is interpolated with the
/// caller's argument. Kept beside its sibling template so the prompt text is
/// auditable in one place rather than buried in the `prompts/get` match.
const RECALL_CONTEXT_TEMPLATE: &str =
    "Before answering, use Brain's `memory_search` tool to recall context \
     relevant to: \"{query}\". Summarize what you find, then answer the \
     question grounded in that recalled context. If nothing relevant is \
     stored, say so.";

/// Rendered body for `daily-review` (no arguments).
const DAILY_REVIEW_TEMPLATE: &str =
    "Use Brain's `memory_episodes` tool to fetch recent activity, then \
     produce a short daily review: what happened, any decisions made, and \
     open loops or follow-ups worth surfacing. Keep it concise and actionable.";

impl McpServer {
    /// `prompts/list` — advertise the available templates.
    pub(crate) fn handle_prompts_list(&self) -> Result<Value, (i32, String)> {
        Ok(json!({
            "prompts": [
                {
                    "name": PROMPT_RECALL_CONTEXT,
                    "description": "Pull memory relevant to a question before answering it.",
                    "arguments": [
                        {
                            "name": "query",
                            "description": "What to recall context about.",
                            "required": true
                        }
                    ]
                },
                {
                    "name": PROMPT_DAILY_REVIEW,
                    "description": "Review recent activity and surface open loops worth following up.",
                    "arguments": []
                }
            ]
        }))
    }

    /// `prompts/get` — render one template, interpolating arguments.
    pub(crate) fn handle_prompts_get(&self, req: &JsonRpcRequest) -> Result<Value, (i32, String)> {
        let params = req
            .params
            .as_ref()
            .ok_or((-32602, "Missing params".to_string()))?;
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing prompt name".to_string()))?;
        let args = params.get("arguments");

        match name {
            PROMPT_RECALL_CONTEXT => {
                let query = args
                    .and_then(|a| a.get("query"))
                    .and_then(Value::as_str)
                    .filter(|q| !q.trim().is_empty())
                    .ok_or((
                        -32602,
                        "recall-context requires a 'query' argument".to_string(),
                    ))?;
                let text = RECALL_CONTEXT_TEMPLATE.replace("{query}", query);
                Ok(prompt_result(
                    "Recall relevant memory for a query, then answer grounded in it.",
                    &text,
                ))
            }
            PROMPT_DAILY_REVIEW => Ok(prompt_result(
                "Review recent activity and surface open loops.",
                DAILY_REVIEW_TEMPLATE,
            )),
            other => Err((-32602, format!("Unknown prompt: {other}"))),
        }
    }
}

/// Build a standard MCP prompt result: a description plus a single user
/// message carrying the rendered template text.
fn prompt_result(description: &str, text: &str) -> Value {
    json!({
        "description": description,
        "messages": [
            {
                "role": "user",
                "content": { "type": "text", "text": text }
            }
        ]
    })
}
