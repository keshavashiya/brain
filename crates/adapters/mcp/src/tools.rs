//! Per-tool implementations exposed via `tools/call`. Each method is invoked
//! by [`crate::server::McpServer::handle_tools_call`] after argument parsing.

use serde_json::{json, Value};

use crate::server::McpServer;

impl McpServer {
    pub(crate) async fn tool_memory_search(&self, args: &Value) -> Result<Value, (i32, String)> {
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: query".to_string()))?;

        let top_k = args.get("top_k").and_then(Value::as_u64).unwrap_or(10) as usize;

        let namespace = args.get("namespace").and_then(Value::as_str);

        let results = self.processor.search_facts(query, top_k, namespace).await;

        let text = if results.is_empty() {
            "No relevant facts found in memory.".to_string()
        } else {
            let lines: Vec<String> = results
                .iter()
                .map(|r| {
                    format!(
                        "[{}:{}] {} {} {} (confidence: {:.2}, distance: {:.3})",
                        r.fact.namespace,
                        r.fact.category,
                        r.fact.subject,
                        r.fact.predicate,
                        r.fact.object,
                        r.fact.confidence,
                        r.distance
                    )
                })
                .collect();
            lines.join("\n")
        };

        Ok(tool_result_text(text))
    }

    pub(crate) async fn tool_memory_store(&self, args: &Value) -> Result<Value, (i32, String)> {
        let subject = args
            .get("subject")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: subject".to_string()))?;

        let predicate = args
            .get("predicate")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: predicate".to_string()))?;

        let object = args
            .get("object")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: object".to_string()))?;

        let category = args
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("general");

        let namespace = args
            .get("namespace")
            .and_then(Value::as_str)
            .unwrap_or("personal");

        let agent = args.get("agent").and_then(Value::as_str);

        match self
            .processor
            .store_fact_direct(namespace, category, subject, predicate, object, agent)
            .await
        {
            Ok(id) => Ok(tool_result_text(format!(
                "Stored fact [{id}]: {subject} {predicate} {object} (namespace: {namespace}, category: {category})"
            ))),
            Err(e) => Err((-32603, format!("Failed to store fact: {e}"))),
        }
    }

    pub(crate) fn tool_memory_facts(&self, args: &Value) -> Result<Value, (i32, String)> {
        let subject = args
            .get("subject")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: subject".to_string()))?;
        let namespace = args.get("namespace").and_then(Value::as_str);

        let facts = self.processor.facts_about(subject, namespace);

        let text = if facts.is_empty() {
            format!("No facts found about '{subject}'.")
        } else {
            let lines: Vec<String> = facts
                .iter()
                .map(|f| {
                    format!(
                        "[{}:{}] {} {} {} (confidence: {:.2})",
                        f.namespace, f.category, f.subject, f.predicate, f.object, f.confidence
                    )
                })
                .collect();
            lines.join("\n")
        };

        Ok(tool_result_text(text))
    }

    pub(crate) fn tool_memory_episodes(&self, args: &Value) -> Result<Value, (i32, String)> {
        let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;

        let episodes = self.processor.recent_episodes(limit, None);

        let text = if episodes.is_empty() {
            "No conversation episodes found.".to_string()
        } else {
            let lines: Vec<String> = episodes
                .iter()
                .map(|e| format!("[{}] {}: {}", e.timestamp, e.role, e.content))
                .collect();
            lines.join("\n")
        };

        Ok(tool_result_text(text))
    }

    pub(crate) fn tool_memory_procedures(&self, args: &Value) -> Result<Value, (i32, String)> {
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .ok_or((-32602, "Missing required argument: action".to_string()))?;

        let procs = self.processor.procedures();

        match action {
            "list" => {
                let list = procs
                    .list_procedures()
                    .map_err(|e| (-32603, format!("Failed to list procedures: {e}")))?;
                let text = if list.is_empty() {
                    "No procedures stored.".to_string()
                } else {
                    list.iter()
                        .map(|p| {
                            format!(
                                "[{}] trigger='{}' steps={} use_count={}",
                                p.id,
                                p.trigger_pattern,
                                p.steps.join(" → "),
                                p.use_count
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                Ok(tool_result_text(text))
            }
            "store" => {
                let trigger = args
                    .get("trigger")
                    .and_then(Value::as_str)
                    .ok_or((-32602, "Missing argument: trigger".to_string()))?;
                let steps: Vec<String> = args
                    .get("steps")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_str)
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default();
                let id = procs
                    .store_procedure(trigger, &steps)
                    .map_err(|e| (-32603, format!("Failed to store procedure: {e}")))?;
                Ok(tool_result_text(format!(
                    "Stored procedure [{id}]: trigger_pattern='{trigger}' with {} step(s)",
                    steps.len()
                )))
            }
            "delete" => {
                let id = args
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or((-32602, "Missing argument: id".to_string()))?;
                procs
                    .delete_procedure(id)
                    .map_err(|e| (-32603, format!("Failed to delete procedure: {e}")))?;
                Ok(tool_result_text(format!("Deleted procedure {id}")))
            }
            other => Err((
                -32602,
                format!("Unknown action: {other}. Use 'list', 'store', or 'delete'"),
            )),
        }
    }

    pub(crate) fn tool_user_profile(&self) -> Result<Value, (i32, String)> {
        Ok(tool_result_text(self.profile_json()))
    }

    /// Pretty-printed profile JSON, shared by the `user_profile` tool and the
    /// `brain://profile` resource so both render identically.
    pub(crate) fn profile_json(&self) -> String {
        let config = self.processor.config();
        // Surface the active LLM transport in the profile JSON. The
        // legacy `llm.provider` field is #[deprecated] (Issue 40) but
        // still load-bearing as the single-provider fallback; reading
        // it here is the deliberate path.
        #[allow(deprecated)]
        let llm_provider = config.llm.provider.clone();
        #[allow(deprecated)]
        let llm_model = config.llm.model.clone();
        let profile = json!({
            "llm": {
                "provider": llm_provider,
                "model": llm_model
            },
            "embedding": {
                "model": config.embedding.model,
                "dimensions": config.embedding.dimensions
            },
            "data_dir": config.data_dir().to_string_lossy(),
            "encryption_enabled": config.encryption.enabled
        });
        serde_json::to_string_pretty(&profile).unwrap_or_default()
    }

    /// `brain_capabilities` — expose Brain's live capability manifest to
    /// external MCP clients. Same text the internal
    /// `List { resource: Capabilities }` intent renders, so internal reasoner
    /// and external clients read one manifest.
    pub(crate) async fn tool_capabilities(&self) -> Result<Value, (i32, String)> {
        Ok(tool_result_text(self.processor.capability_manifest().await))
    }
}

/// Build a standard MCP tool result with a single text content block.
fn tool_result_text(text: impl Into<String>) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text.into()
            }
        ]
    })
}
