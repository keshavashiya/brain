//! Conversation-category intent handler: free-form chat (the catch-all
//! classification). Returns [`PipelineResult::LlmReady`] so the caller
//! (`SignalProcessor::process`) can run LLM generation — for agent callers
//! it returns [`PipelineResult::Complete`] with structured memory context
//! instead.
//!
//! Variant: [`thalamus::Intent::Chat`].

use uuid::Uuid;

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{ConversationAuth, ConversationHandler, HandlerContext, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

impl ConversationAuth for SignalProcessor {
    fn auth_conversation(_intent: &thalamus::Intent) -> Option<(AuthorizationRequest, Tier)> {
        // Pure conversation — no authorization needed.
        None
    }
}

#[async_trait::async_trait]
impl ConversationHandler for SignalProcessor {
    async fn dispatch_conversation(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::Chat { content } => {
                self.handle_chat(
                    ctx.signal_id,
                    ctx.signal,
                    content,
                    ctx.importance,
                    ctx.conversation_history,
                    ctx.procedure_context,
                    prepend_nudges,
                    ctx.progress,
                )
                .await
            }
            other => unreachable!(
                "non-conversation variant routed to dispatch_conversation: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_chat(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        content: String,
        importance: f32,
        conversation_history: Option<&[cortex::llm::Message]>,
        procedure_context: &[String],
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
        progress: Option<&tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let top_k = self.config.memory.semantic.max_results as usize;
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&content).await;
        let (memories, facts_used, episodes_used) = self
            .do_recall(&content, query_vector, top_k, Some(&signal.namespace))
            .await?;

        // Reuse caller-supplied session or create a new one
        let session_id = if let Some(ref sid) = signal.session_id {
            // Ensure the session row exists so FK constraints on episodes never fail.
            // This handles the case where a client reuses a session_id from a
            // previous daemon run that was cleared.
            self.episodic
                .ensure_session(sid, &signal.channel)
                .map_err(|e| SignalError::Storage(e.to_string()))?;
            sid.clone()
        } else {
            self.episodic
                .create_session(&signal.channel)
                .map_err(|e| SignalError::Storage(e.to_string()))?
        };

        self.episodic
            .store_episode(
                &session_id,
                "user",
                &signal.content,
                importance as f64,
                Some(&signal.namespace),
                signal.agent.as_deref(),
            )
            .map_err(|e| SignalError::Storage(e.to_string()))?;

        // Agent callers get structured memory context
        if signal.agent.is_some() {
            let response_text = if memories.is_empty() {
                format!(
                    "Stored episode. No relevant memories found for: {}",
                    content
                )
            } else {
                let mem_lines: String = memories
                    .iter()
                    .map(|m| format!("[{:?}] {}", m.source, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("Stored episode. Relevant memories:\n{}", mem_lines)
            };

            let resp = prepend_nudges(SignalResponse {
                signal_id,
                status: ResponseStatus::Ok,
                response: ResponseContent::Text(response_text),
                memory_context: MemoryContext {
                    facts_used,
                    episodes_used,
                },
                session_id: Some(session_id.clone()),
            });
            return Ok(PipelineResult::Complete(resp));
        }

        let proc_history: Vec<cortex::llm::Message> = procedure_context
            .iter()
            .map(|step| cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: format!("[procedure step] {step}"),
            })
            .collect();
        let history = conversation_history.unwrap_or(&proc_history);
        let addendum = if self.namespace_is_empty(&signal.namespace) {
            Some(cortex::context::ONBOARDING_ADDENDUM)
        } else {
            None
        };

        // Path-attachment grounding: if the user named one or more
        // local paths in `content`, read each on their behalf and pass
        // the snapshots to the assembler. Replaces the old
        // `Intent::ProjectInspect` branch that bypassed SOUL entirely.
        //
        // The snapshot pipeline performs blocking `std::fs::*` calls
        // (canonicalize, metadata, read_dir, read). Off-load to the
        // blocking pool so other async tasks on this runtime aren't
        // stalled while we read a directory.
        let attachments = {
            let content_owned = content.clone();
            let allowed = self.config.security.allowed_paths.clone();
            tokio::task::spawn_blocking(move || {
                crate::attachment::build_chat_attachments(&content_owned, &allowed)
            })
            .await
            .map_err(|e| SignalError::Processing(format!("attachment task panicked: {e}")))?
        };
        if !attachments.is_empty() {
            tracing::debug!(
                attached = attachments.attached.len(),
                skipped = attachments.skipped.len(),
                "chat turn carries path attachments"
            );
        }

        // Hand the SOUL prompt a *live* capability digest
        // rendered from the currently-wired tools and agents, so the
        // reasoner describes its real catalog instead of a hardcoded
        // string. Read-only awareness — execution stays gated downstream.
        let capability_digest = self.capability_digest().await;

        let messages = self.context_assembler.assemble_full(
            &content,
            &memories,
            history,
            addendum,
            Some(&capability_digest),
            &attachments.attached,
            &attachments.skipped,
        );

        Ok(PipelineResult::LlmReady {
            signal_id,
            messages,
            memory_context: MemoryContext {
                facts_used,
                episodes_used,
            },
            session_id: Some(session_id),
            user_content: content,
            namespace: signal.namespace.clone(),
            agent: signal.agent.clone(),
        })
    }

    /// Build the live "Your Capabilities" section for the SOUL prompt
    /// Reads the currently-wired tool registry and agent
    /// registry so the reasoner's self-description tracks what is mounted
    /// *right now* — mount a new MCP server and the next turn's digest
    /// reflects it. Returns the always-on faculties even when nothing
    /// extra is wired.
    pub(super) async fn capability_digest(&self) -> String {
        let tools = match &self.capability.tool_registry {
            Some(registry) => registry.list().await,
            None => Vec::new(),
        };
        let agents = self
            .agent_registry
            .as_ref()
            .map(|r| r.list())
            .unwrap_or_default();
        render_capability_digest(&tools, &agents)
    }
}

/// Cap on tools listed per MCP server in the digest, keeping it token-bounded.
const MAX_TOOLS_PER_SERVER: usize = 15;
/// Cap on delegate agents listed in the digest.
const MAX_AGENTS: usize = 20;

/// Render the live capability section injected into the SOUL prompt
/// Starts from the always-on cognitive faculties
/// ([`cortex::context::DEFAULT_CAPABILITIES`]) and appends whatever tools
/// and delegate agents are *currently* wired.
///
/// Read-only awareness only: this lists what exists; execution stays gated
/// by the capability/consent/audit path. Only trusted fields are inlined
/// (verb vocabulary, source kind, agent names) — untrusted MCP tool
/// *descriptions* are deliberately not rendered here (that enrichment is
/// a later phase, and would route through [`intent::sanitization`]).
/// Token-bounded: per-server tool lists and the agent list are capped.
fn render_capability_digest(tools: &[intent::ToolDescriptor], agents: &[String]) -> String {
    use intent::ToolSource;
    use std::collections::BTreeMap;

    let mut out = String::from(cortex::context::DEFAULT_CAPABILITIES);

    // Group registered tools by source. The tool registry is the unified
    // catalog the MCP host and native backends register into on mount.
    let mut mcp: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut native: Vec<String> = Vec::new();
    let mut terminal = false;
    for t in tools {
        match &t.source {
            ToolSource::McpServer { server } => mcp
                .entry(server.clone())
                .or_default()
                .push(t.verb.action.clone()),
            ToolSource::NativeBackend { backend } => native.push(backend.as_str().to_string()),
            ToolSource::Terminal => terminal = true,
        }
    }

    if !mcp.is_empty() || !native.is_empty() || terminal {
        out.push_str("\n\nMounted tools (live — reflects what is connected right now):\n");
        for (server, mut actions) in mcp {
            actions.sort();
            actions.dedup();
            out.push_str(&format!(
                "- MCP server \"{server}\": {}\n",
                render_capped_list(&actions, MAX_TOOLS_PER_SERVER)
            ));
        }
        if !native.is_empty() {
            native.sort();
            native.dedup();
            out.push_str(&format!("- Native backends: {}\n", native.join(", ")));
        }
        if terminal {
            out.push_str("- Terminal: run shell commands in sandboxed sessions.\n");
        }
    }

    if !agents.is_empty() {
        out.push_str(&format!(
            "\nDelegated agents (specialist tasks you can hand off): {}\n",
            render_capped_list(agents, MAX_AGENTS)
        ));
    }

    out
}

/// Join `items` with commas, capping at `cap` and summarizing the overflow.
fn render_capped_list(items: &[String], cap: usize) -> String {
    if items.len() <= cap {
        items.join(", ")
    } else {
        format!(
            "{}, … (+{} more)",
            items[..cap].join(", "),
            items.len() - cap
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{render_capability_digest, render_capped_list};
    use intent::{BackendId, ToolAnnotations, ToolDescriptor, ToolSource, Verb};

    fn tool(tool_id: &str, source: ToolSource, action: &str) -> ToolDescriptor {
        ToolDescriptor {
            tool_id: tool_id.to_string(),
            source,
            verb: Verb {
                namespace: "mcp".to_string(),
                action: action.to_string(),
            },
            description: "untrusted".to_string(),
            input_schema: serde_json::json!({}),
            output_schema: None,
            capabilities: vec![],
            annotations: ToolAnnotations::default(),
            usage: intent::ToolUsage::default(),
            embedding: None,
        }
    }

    #[test]
    fn digest_always_includes_static_faculties() {
        let digest = render_capability_digest(&[], &[]);
        assert!(digest.starts_with(cortex::context::DEFAULT_CAPABILITIES));
        assert!(digest.contains("Episodic Memory"));
        // Nothing wired → no "Mounted tools" / agents sections.
        assert!(!digest.contains("Mounted tools"));
        assert!(!digest.contains("Delegated agents"));
    }

    #[test]
    fn digest_groups_tools_by_source_and_lists_agents() {
        let tools = vec![
            tool(
                "github::create_issue",
                ToolSource::McpServer {
                    server: "github".to_string(),
                },
                "create_issue",
            ),
            tool(
                "github::list_prs",
                ToolSource::McpServer {
                    server: "github".to_string(),
                },
                "list_prs",
            ),
            tool(
                "fs::read",
                ToolSource::NativeBackend {
                    backend: BackendId::new("fs"),
                },
                "read",
            ),
            tool("sh", ToolSource::Terminal, "exec"),
        ];
        let agents = vec!["aider".to_string(), "claude".to_string()];
        let digest = render_capability_digest(&tools, &agents);

        assert!(digest.contains("MCP server \"github\": create_issue, list_prs"));
        assert!(digest.contains("Native backends: fs"));
        assert!(digest.contains("Terminal: run shell commands"));
        assert!(digest.contains("Delegated agents"));
        assert!(digest.contains("aider, claude"));
        // Untrusted descriptions are never inlined.
        assert!(!digest.contains("untrusted"));
    }

    #[test]
    fn capped_list_summarizes_overflow() {
        let items: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        assert_eq!(render_capped_list(&items, 10), "t0, t1, t2, t3, t4");
        assert_eq!(render_capped_list(&items, 2), "t0, t1, … (+3 more)");
    }
}
