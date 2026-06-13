//! Conversation-category intent handler: free-form chat (the catch-all
//! classification). Returns [`PipelineResult::LlmReady`] so the caller
//! (`SignalProcessor::process`) can run LLM generation — for agent callers
//! it returns [`PipelineResult::Complete`] with structured memory context
//! instead.
//!
//! Variant: [`thalamus::Intent::Chat`].

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
                self.handle_chat(&ctx, content, prepend_nudges).await
            }
            other => unreachable!(
                "non-conversation variant routed to dispatch_conversation: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

/// Recount the per-source split after the residency filter changed the set.
fn count_sources(memories: &[hippocampus::Memory]) -> (usize, usize) {
    let facts = memories
        .iter()
        .filter(|m| m.source == hippocampus::MemorySource::Semantic)
        .count();
    let episodes = memories
        .iter()
        .filter(|m| m.source == hippocampus::MemorySource::Episodic)
        .count();
    (facts, episodes)
}

/// Render the write-side grounding block appended to the chat capability
/// section. Lists the facts the pipeline actually persisted this turn so the
/// reasoner can truthfully confirm a save — the write-side analogue of the
/// "Relevant memories:" recall block, cited by the SOUL `MEMORY WRITES` rule.
/// Returns an empty string when nothing was written (the common case), so a
/// no-write turn leaves the reasoner with nothing to claim.
fn render_saved_this_turn(writes: &[crate::exchange::FactToStore]) -> String {
    if writes.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "\n\nSaved this turn (persisted to memory just now — you MAY confirm these were saved):\n",
    );
    for f in writes {
        out.push_str(&format!("- {} {} {}\n", f.subject, f.predicate, f.object));
    }
    out
}

impl SignalProcessor {
    pub(super) async fn handle_chat(
        &self,
        ctx: &HandlerContext<'_>,
        content: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let &HandlerContext {
            signal_id,
            signal,
            importance,
            conversation_history,
            procedure_context,
            progress,
            writes_this_turn,
        } = ctx;
        // Scale the number of memory recall candidates with the available
        // memory budget so large-window models surface more relevant context
        // instead of being clipped to the conservative static default.
        let top_k_base = self.config.memory.semantic.max_results as usize;
        let memory_budget = self.context_assembler.budget().memory_budget();
        let top_k = top_k_base
            .max(memory_budget / 50) // ~50 tokens per memory entry
            .min(200); // sanity cap
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&content, &signal.namespace).await;
        let (memories, facts_used, episodes_used) = self
            .do_recall(&content, query_vector, top_k, Some(&signal.namespace))
            .await?;

        // Reuse caller-supplied session or create a new one
        let session_id = if let Some(ref sid) = signal.session_id {
            // Ensure the session row exists so FK constraints on episodes never fail.
            // This handles the case where a client reuses a session_id from a
            // previous daemon run that was cleared.
            self.memory
                .episodic
                .ensure_session(sid, &signal.channel)
                .map_err(|e| SignalError::Storage(e.to_string()))?;
            sid.clone()
        } else {
            self.memory
                .episodic
                .create_session(&signal.channel)
                .map_err(|e| SignalError::Storage(e.to_string()))?
        };

        let episode_id = self
            .memory
            .episodic
            .store_episode(
                &session_id,
                "user",
                &signal.content,
                importance as f64,
                Some(&signal.namespace),
                signal.agent.as_deref(),
            )
            .map_err(|e| SignalError::Storage(e.to_string()))?;
        self.quarantine_episode_if_unattested(&episode_id, signal.agent.as_deref())
            .await;

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

        // Residency gate: this prompt may leave the machine, so memories
        // from local-only namespaces must not ride along. The agent branch
        // above is exempt — it returns memory to a local caller.
        let (memories, withheld) = self.withhold_nonresident_memories(memories, &signal.namespace);
        let (facts_used, episodes_used) = if withheld > 0 {
            count_sources(&memories)
        } else {
            (facts_used, episodes_used)
        };

        let proc_history: Vec<cortex::llm::Message> = procedure_context
            .iter()
            .map(|step| cortex::llm::Message::user(format!("[procedure step] {step}")))
            .collect();
        let history_ref = conversation_history.unwrap_or(&proc_history);
        // Compact overflow turns into a summary instead of dropping them when
        // the thread exceeds its history budget (no-op + no LLM call when it
        // fits, which is the norm on a generously-sized context window).
        let history = self.compact_history(history_ref).await;
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
            let budget = self.context_assembler.budget().attachments;
            tokio::task::spawn_blocking(move || {
                crate::attachment::build_chat_attachments(&content_owned, &allowed, budget)
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

        // Hand the SOUL prompt a *live* capability digest rendered from the
        // currently-wired tools and agents plus Brain's grounded self-model,
        // so the reasoner describes its real catalog and product surface
        // instead of a hardcoded or fabricated one. Read-only awareness —
        // execution stays gated downstream.
        let mut capability_digest = self.chat_capability_section(&content).await;
        // Write-side grounding: surface what the pipeline actually persisted
        // this turn so the reasoner can truthfully confirm a save (and only
        // then), instead of fabricating one. The MEMORY WRITES operating
        // principle in the SOUL prompt cites this exact block.
        capability_digest.push_str(&render_saved_this_turn(writes_this_turn));

        let messages = self.context_assembler.assemble_full(
            &content,
            &memories,
            &history,
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

    /// The full "Your Capabilities" + "About Brain" prompt section for one
    /// chat turn: the live capability digest (mounted tools/agents) followed,
    /// when a product self-model is wired, by the code-derived self-knowledge
    /// (real CLI commands, config schema, policy) scored against `content`.
    /// The self-model section is what stops the SOUL fabricating Brain's own
    /// commands and config keys; processors without one (tests, background
    /// tasks) get the digest alone — back-compat.
    pub(super) async fn chat_capability_section(&self, content: &str) -> String {
        let mut section = self.capability_digest().await;
        if let Some(model) = self.product_self_model() {
            section.push_str("\n\n");
            section.push_str(&model.render_grounding(content, SELF_MODEL_CONFIG_K));
        }
        section
    }

    /// Compact conversation history that overflows its token budget: keep the
    /// most recent turns verbatim and fold the oldest overflow into a single
    /// cached summary note, so older context survives in compressed form
    /// instead of being silently dropped.
    ///
    /// A no-op (returns the history unchanged, no LLM call) whenever the
    /// thread already fits — which is the common case, and always true once
    /// `llm.context_window` is set generously. Only long threads on small
    /// windows pay the one extra summarization call, and repeated turns reuse
    /// the cached summary.
    pub(super) async fn compact_history(
        &self,
        history: &[cortex::llm::Message],
    ) -> Vec<cortex::llm::Message> {
        // Room held back for the summary note so summary + kept turns still fit.
        const SUMMARY_RESERVE_TOKENS: usize = 256;
        let budget = self.context_assembler.budget().conversation_history;
        let plan =
            cortex::compaction::plan_history_compaction(history, budget, SUMMARY_RESERVE_TOKENS);
        if plan.is_noop() {
            return history.to_vec();
        }

        let mut out = Vec::with_capacity(plan.keep_recent.len() + 1);
        // On summarization failure we fall back to recent turns only — the
        // overflow is dropped exactly as it was before compaction existed.
        if let Some(summary) = self.summarize_overflow(plan.to_summarize).await {
            out.push(cortex::llm::Message::system(format!(
                "Summary of earlier conversation (older turns compacted to fit the context window):\n{summary}"
            )));
        }
        out.extend_from_slice(plan.keep_recent);
        out
    }

    /// Summarize the overflow turns into a short note, caching by content hash.
    /// Returns `None` when there's nothing to summarize or the LLM call fails.
    async fn summarize_overflow(
        &self,
        turns: &[cortex::llm::Message],
    ) -> Option<std::sync::Arc<str>> {
        if turns.is_empty() {
            return None;
        }
        let key = history_summary_key(turns);
        if let Some(hit) = self
            .history_summary_cache
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
        {
            return Some(hit);
        }

        let transcript = turns
            .iter()
            .map(|m| format!("{}: {}", role_label(&m.role), m.content))
            .collect::<Vec<_>>()
            .join("\n");
        let prompt = vec![
            cortex::llm::Message::system(
                "You compress conversation history. Summarize the turns below in 2-5 \
                 sentences, preserving facts, decisions, names, numbers, and any \
                 unresolved questions. Output only the summary.",
            ),
            cortex::llm::Message::user(transcript),
        ];
        // Fast-tier work: a compression chore, not a quality-sensitive
        // generation — and on a configured local fast lane it stays here.
        match self.llm_fast.generate(&prompt).await {
            Ok(resp) => {
                let summary: std::sync::Arc<str> = std::sync::Arc::from(resp.content.trim());
                self.history_summary_cache
                    .lock()
                    .unwrap()
                    .put(key, summary.clone());
                Some(summary)
            }
            Err(e) => {
                tracing::warn!(error = %e, "history compaction summary failed; keeping recent turns only");
                None
            }
        }
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
        // Learned self-model: tools that have a proven track record here.
        // Empty when learning is off / nothing proven yet.
        let proven = self
            .fitness()
            .proven_tools(
                cerebellum::MIN_USES_TO_SURFACE,
                cerebellum::MIN_RATIO_TO_SURFACE,
                MAX_PROVEN_TOOLS,
            )
            .unwrap_or_default();
        let mut digest = render_capability_digest(&tools, &agents, &proven);
        // Situated grounding: name the machine (and its class) so the
        // reasoner sizes suggestions — local model picks, batch work — to
        // the hardware it actually runs on.
        if let Some(host) = self.host_model() {
            digest.push('\n');
            digest.push_str(&host.digest_line());
            digest.push('\n');
        }
        // Situated grounding: name the model actually serving THIS turn. The
        // reasoner kept telling users "I don't have that — which model am I?"
        // while the daemon plainly knew. active_llm() already reflects the
        // offline→local-tier switch, so the line stays truthful when the
        // network drops and the turn rides a local model instead.
        let active = self.active_llm();
        let locality = if active.is_local() {
            "local, runs on this machine"
        } else {
            "remote"
        };
        digest.push_str(&format!(
            "\nActive model: {}/{} ({locality}) — the model generating this reply right now.\n",
            active.name(),
            active.model(),
        ));
        // Situated grounding: when the network view is anything but Online,
        // say so — the reasoner must not promise web search or remote tools
        // it cannot reach, and offline turns are already riding a local tier.
        match self.connectivity.state() {
            brain::ConnectivityState::Online => {
                // State it positively too: "are you online?" was unanswerable
                // when this arm stayed silent — the reasoner could only infer
                // online-ness indirectly. One short line closes that.
                digest.push_str("\nConnectivity: online.\n");
            }
            brain::ConnectivityState::Degraded => {
                digest.push_str(
                    "\nConnectivity: degraded — some network endpoints are unreachable; \
                     remote tools and web search may fail.\n",
                );
            }
            brain::ConnectivityState::Offline => {
                digest.push_str(
                    "\nConnectivity: OFFLINE — the network is unreachable. Web search, \
                     remote tools, and outbound messages will not work right now; answer \
                     from the local model and stored memory, and say plainly that you are \
                     offline when a request needs the network.\n",
                );
            }
        }
        // Same situated grounding for power: on battery the reasoner should
        // prefer lighter work and not volunteer heavy batch jobs. The
        // deferral clause tracks the actual config so the digest never
        // claims behaviour the operator turned off.
        if self.power.is_battery() {
            if self.config.monitoring.power.defer_maintenance {
                digest.push_str(
                    "\nPower: on battery — heavy background maintenance (memory \
                     consolidation, graph sweeps) is held until external power returns; \
                     prefer lighter work over big batch jobs.\n",
                );
            } else {
                digest.push_str("\nPower: on battery.\n");
            }
        }
        // Live capability health: name the capabilities that won't work right
        // now (embedding model down, breaker open, …) so the reasoner doesn't
        // promise a dead faculty. Empty before the first sweep / without one.
        digest.push_str(&render_health_block(&self.manifest_health().snapshot()));
        // Quarantined-and-waiting must be visible, not a silent hole:
        // memories from unvouched writers exist but are excluded from
        // recall until the user reviews them.
        let quarantined = self.quarantined_memory_counts();
        if !quarantined.is_empty() {
            digest.push_str("\nUnreviewed memory (excluded from recall until approved):\n");
            for q in &quarantined {
                digest.push_str(&format!(
                    "- {} fact(s) and {} episode(s) from agent \"{}\" — approve with /memory-approve {}\n",
                    q.facts, q.episodes, q.agent, q.agent,
                ));
            }
        }
        digest
    }

    /// Concise capability summary lines for the task planner — one line
    /// per faculty (mounted MCP servers with their action verbs, native
    /// backends, the terminal) drawn from the live tool registry. Distinct
    /// from [`Self::capability_digest`]: that builds a SOUL-prompt section
    /// prefixed with the always-on faculties; this returns bare lines the
    /// decomposer folds into its own prompt. Empty when no registry is
    /// wired, in which case the planner falls back to the sandbox allowlist
    /// alone (its prior behavior).
    pub(super) async fn planner_capabilities(&self) -> Vec<String> {
        let tools = match &self.capability.tool_registry {
            Some(registry) => registry.list().await,
            None => Vec::new(),
        };
        capability_lines(&tools)
    }
}

/// Group registered tools by source into one summary line per faculty.
/// Trusted fields only (server names + verb actions, source kind) — the
/// same restraint as [`render_capability_digest`]; untrusted MCP tool
/// *descriptions* are not inlined.
fn capability_lines(tools: &[intent::ToolDescriptor]) -> Vec<String> {
    use intent::ToolSource;
    use std::collections::BTreeMap;

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

    let mut lines = Vec::new();
    for (server, mut actions) in mcp {
        actions.sort();
        actions.dedup();
        lines.push(format!(
            "MCP server \"{server}\": {}",
            render_capped_list(&actions, MAX_TOOLS_PER_SERVER)
        ));
    }
    if !native.is_empty() {
        native.sort();
        native.dedup();
        lines.push(format!("Native backends: {}", native.join(", ")));
    }
    if terminal {
        lines.push("Terminal: run shell commands in sandboxed sessions".to_string());
    }
    lines
}

/// Top-k config sections injected into the per-turn product self-model
/// grounding. Small so a relevant slice (e.g. the messaging webhook schema)
/// reaches the SOUL without dumping the whole config.
const SELF_MODEL_CONFIG_K: usize = 3;

/// Cap on tools listed per MCP server in the digest, keeping it token-bounded.
const MAX_TOOLS_PER_SERVER: usize = 15;
/// Cap on delegate agents listed in the digest.
const MAX_AGENTS: usize = 20;
/// Cap on proven tools listed in the learned "Proven here" digest line.
const MAX_PROVEN_TOOLS: usize = 8;

/// Render the "Degraded capabilities" digest block from the live manifest
/// health map. Lists every capability that isn't [`Verified`], sorted by
/// `tool_id` for stable output, each with its kind + reason. Empty string when
/// everything is healthy (or no sweep has run yet) — the digest then carries no
/// health block at all, so a healthy deployment reads exactly as before.
///
/// [`Verified`]: brain::CapabilityHealth::Verified
fn render_health_block(
    health: &std::collections::HashMap<String, brain::CapabilityHealth>,
) -> String {
    let mut unhealthy: Vec<(&String, &brain::CapabilityHealth)> =
        health.iter().filter(|(_, h)| !h.is_healthy()).collect();
    if unhealthy.is_empty() {
        return String::new();
    }
    unhealthy.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::from(
        "\nDegraded capabilities (not working right now — don't promise these; \
         say plainly they're unavailable if asked):\n",
    );
    for (tool_id, h) in unhealthy {
        out.push_str(&format!(
            "- {tool_id}: {} ({})\n",
            h.reason().unwrap_or_default(),
            h.as_str(),
        ));
    }
    out
}

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
fn render_capability_digest(
    tools: &[intent::ToolDescriptor],
    agents: &[String],
    proven: &[cerebellum::Fitness],
) -> String {
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

    // Learned self-model: tools with a proven track record in this deployment.
    // Trusted fields only (internal tool ids → bare verb labels); no untrusted
    // MCP description text. Awareness/preference only — still consent-gated.
    if !proven.is_empty() {
        let labels: Vec<String> = proven.iter().map(|f| proven_label(&f.tool_id)).collect();
        out.push_str(&format!(
            "\nProven here (you've used these successfully before — prefer them when they fit): {}\n",
            render_capped_list(&labels, MAX_PROVEN_TOOLS),
        ));
    }

    // Closed-world boundary: the always-on faculties plus whatever is listed
    // above are the *complete* set of things actually executable in this
    // deployment. Stops the reasoner over-claiming faculties (shell, web,
    // filesystem, a specific MCP tool) that aren't wired — the transcript's
    // "I can run shell commands, query the filesystem" failure when neither
    // was mounted.
    out.push_str(
        "\nThis is the complete set of actions you can actually execute right now. \
         If a capability — shell/command execution, web search, file access, or a \
         specific MCP tool — is not listed above, you do NOT have it in this \
         deployment: say so plainly instead of claiming, promising, or simulating it.\n",
    );

    out
}

/// Bare verb/tool label for a proven `tool_id` in the digest — drops the
/// source prefix (`native:` / `mcp:`) so the reasoner sees `net.http` rather
/// than `native:net.http`. The tool_id is an internal, trusted string.
fn proven_label(tool_id: &str) -> String {
    tool_id
        .split_once(':')
        .map(|(_, rest)| rest.to_string())
        .unwrap_or_else(|| tool_id.to_string())
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

/// Stable hash of the overflow turns being summarized — the
/// history-summary cache key. Order-sensitive; role + content of each turn.
fn history_summary_key(turns: &[cortex::llm::Message]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for m in turns {
        role_label(&m.role).hash(&mut h);
        m.content.hash(&mut h);
    }
    h.finish()
}

/// Lowercase wire label for a chat role, used in the compaction transcript.
fn role_label(role: &cortex::llm::Role) -> &'static str {
    match role {
        cortex::llm::Role::User => "user",
        cortex::llm::Role::Assistant => "assistant",
        cortex::llm::Role::System => "system",
        cortex::llm::Role::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        capability_lines, history_summary_key, render_capability_digest, render_capped_list,
        render_health_block, render_saved_this_turn,
    };
    use crate::exchange::FactToStore;
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
    fn saved_this_turn_block_is_empty_when_nothing_was_written() {
        // The common case: no extraction → no block → the reasoner has no
        // evidence to claim a save (MEMORY WRITES then makes it offer instead).
        assert!(render_saved_this_turn(&[]).is_empty());
    }

    #[test]
    fn saved_this_turn_block_lists_persisted_triples() {
        let writes = vec![
            FactToStore {
                subject: "user".into(),
                predicate: "likes".into(),
                object: "pizza".into(),
            },
            FactToStore {
                subject: "user".into(),
                predicate: "works_at".into(),
                object: "Acme Corp".into(),
            },
        ];
        let block = render_saved_this_turn(&writes);
        // Labelled exactly as the SOUL MEMORY WRITES rule cites it.
        assert!(block.contains("Saved this turn"));
        assert!(block.contains("user likes pizza"));
        assert!(block.contains("user works_at Acme Corp"));
    }

    #[test]
    fn digest_always_includes_static_faculties() {
        let digest = render_capability_digest(&[], &[], &[]);
        assert!(digest.starts_with(cortex::context::DEFAULT_CAPABILITIES));
        assert!(digest.contains("Episodic Memory"));
        // Nothing wired → no "Mounted tools" / agents / proven sections.
        assert!(!digest.contains("Mounted tools"));
        assert!(!digest.contains("Delegated agents"));
        assert!(!digest.contains("Proven here"));
        // The closed-world boundary is always present, especially when nothing
        // is wired — that's when over-claiming is worst.
        assert!(digest.contains("complete set of actions you can actually execute"));
        assert!(digest.contains("you do NOT have it in this deployment"));
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
        let digest = render_capability_digest(&tools, &agents, &[]);

        assert!(digest.contains("MCP server \"github\": create_issue, list_prs"));
        assert!(digest.contains("Native backends: fs"));
        assert!(digest.contains("Terminal: run shell commands"));
        assert!(digest.contains("Delegated agents"));
        assert!(digest.contains("aider, claude"));
        // Untrusted descriptions are never inlined.
        assert!(!digest.contains("untrusted"));
    }

    #[test]
    fn digest_surfaces_proven_tools_with_bare_labels() {
        let fit = |tool_id: &str| cerebellum::Fitness {
            tool_id: tool_id.to_string(),
            success: 5.0,
            failure: 0.0,
            uses: 5,
            ratio: 1.0,
        };
        let proven = vec![fit("native:net.http"), fit("mcp:github:create_issue")];
        let digest = render_capability_digest(&[], &[], &proven);
        assert!(digest.contains("Proven here"), "{digest}");
        // Source prefixes are stripped to bare labels.
        assert!(digest.contains("net.http"), "{digest}");
        assert!(digest.contains("github:create_issue"), "{digest}");
        assert!(!digest.contains("native:net.http"), "{digest}");
    }

    #[test]
    fn capability_lines_summarize_faculties_for_planner() {
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
        let lines = capability_lines(&tools);
        assert!(lines
            .iter()
            .any(|l| l == "MCP server \"github\": create_issue, list_prs"));
        assert!(lines.iter().any(|l| l == "Native backends: fs"));
        assert!(lines
            .iter()
            .any(|l| l.starts_with("Terminal: run shell commands")));
        // Untrusted descriptions never leak into the planner lines.
        assert!(!lines.iter().any(|l| l.contains("untrusted")));
    }

    #[test]
    fn capability_lines_empty_when_nothing_wired() {
        assert!(capability_lines(&[]).is_empty());
    }

    #[test]
    fn capped_list_summarizes_overflow() {
        let items: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        assert_eq!(render_capped_list(&items, 10), "t0, t1, t2, t3, t4");
        assert_eq!(render_capped_list(&items, 2), "t0, t1, … (+3 more)");
    }

    #[test]
    fn history_summary_key_is_order_sensitive_and_stable() {
        use cortex::llm::Message;
        let a = vec![Message::user("one"), Message::assistant("two")];
        let b = vec![Message::assistant("two"), Message::user("one")];
        assert_eq!(history_summary_key(&a), history_summary_key(&a));
        assert_ne!(history_summary_key(&a), history_summary_key(&b));
    }

    #[tokio::test]
    async fn compact_history_is_noop_when_within_budget() {
        use cortex::llm::Message;
        let processor = make_processor().await;
        // A handful of short turns is far under the 8k-default history budget,
        // so compaction must return them verbatim and make no LLM call.
        let history = vec![
            Message::user("hello"),
            Message::assistant("hi there"),
            Message::user("how are you?"),
        ];
        let out = processor.compact_history(&history).await;
        assert_eq!(out.len(), history.len());
        for (o, h) in out.iter().zip(&history) {
            assert_eq!(o.content, h.content);
        }
        // No summary note was injected.
        assert!(!out.iter().any(|m| m.content.contains("Summary of earlier")));
    }

    use crate::SignalProcessor;
    use brain::BrainConfig;
    use selfmodel::{CommandDoc, ProductSelfModel, SignalDoc};
    use std::sync::Arc;

    async fn make_processor() -> SignalProcessor {
        let temp = tempfile::tempdir().unwrap();
        let mut config = BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);
        processor
    }

    fn self_model() -> Arc<ProductSelfModel> {
        Arc::new(ProductSelfModel::new(
            vec![CommandDoc {
                name: "chat".to_string(),
                summary: "interactive chat session".to_string(),
                args: vec![],
            }],
            vec![SignalDoc {
                name: "/status".to_string(),
                summary: "show cortex, memory, and synapse status".to_string(),
            }],
            BrainConfig::default_config_content(),
        ))
    }

    #[tokio::test]
    async fn chat_section_includes_self_model_grounding_when_wired() {
        let processor = make_processor().await.with_product_self_model(self_model());
        let section = processor
            .chat_capability_section("how do I configure telegram in config.yaml")
            .await;
        // Capability digest prefix is still there…
        assert!(section.contains("Your Capabilities"));
        // …and the authoritative self-model grounding is appended for this turn.
        assert!(section.contains("About Brain"));
        assert!(section.contains("brain chat"));
        assert!(section.contains("NO native Telegram"));
        // The telegram query pulled in the real messaging webhook schema.
        assert!(section.contains("channels"));
    }

    #[tokio::test]
    async fn chat_section_is_digest_only_without_self_model() {
        let processor = make_processor().await;
        let section = processor.chat_capability_section("hello").await;
        assert!(section.contains("Your Capabilities"));
        assert!(!section.contains("About Brain"));
    }

    #[tokio::test]
    async fn digest_names_machine_class_when_host_model_wired() {
        let host = Arc::new(selfmodel::HostModel::probe(None));
        let expected_class = host.machine_class();
        let processor = make_processor().await.with_host_model(host);
        let digest = processor.capability_digest().await;
        assert!(digest.contains("Host machine:"), "{digest}");
        assert!(
            digest.contains(&format!("machine class: {expected_class}")),
            "digest must name the machine class: {digest}"
        );

        // Unwired processors keep the digest host-free (back-compat).
        let bare = make_processor().await.capability_digest().await;
        assert!(!bare.contains("Host machine:"));
    }

    /// The reasoner was blind to its own serving model — asked "which model
    /// are you?" it told the user to supply the answer the daemon already had.
    /// The digest the SOUL reads must name the active provider/model so that
    /// self-knowledge question is answerable. Asserts against the processor's
    /// own `active_llm()` so the test holds regardless of the wired provider.
    #[tokio::test]
    async fn digest_names_the_active_serving_model() {
        let processor = make_processor().await;
        let active = processor.active_llm();
        let digest = processor.capability_digest().await;
        assert!(digest.contains("Active model:"), "{digest}");
        assert!(
            digest.contains(active.name()),
            "digest must name the active provider: {digest}"
        );
        assert!(
            digest.contains(active.model()),
            "digest must name the active model: {digest}"
        );
        // Connectivity defaults to Online; "are you online?" must be answerable
        // from the digest, not just inferable from the model's locality.
        assert!(
            digest.contains("Connectivity: online"),
            "online connectivity must be stated positively: {digest}"
        );
    }

    #[test]
    fn health_block_empty_when_all_verified() {
        use std::collections::HashMap;
        let mut h = HashMap::new();
        h.insert(
            "memory.store".to_string(),
            brain::CapabilityHealth::Verified,
        );
        assert_eq!(render_health_block(&h), "");
        // No sweep yet → empty map → no block.
        assert_eq!(render_health_block(&HashMap::new()), "");
    }

    #[test]
    fn health_block_lists_degraded_sorted_with_reason() {
        use std::collections::HashMap;
        let mut h = HashMap::new();
        h.insert(
            "web.search".to_string(),
            brain::CapabilityHealth::BreakerOpen,
        );
        h.insert(
            "memory.store".to_string(),
            brain::CapabilityHealth::Degraded {
                reason: "local embedding model unreachable".to_string(),
            },
        );
        h.insert("net.http".to_string(), brain::CapabilityHealth::Verified);
        let block = render_health_block(&h);
        assert!(block.contains("Degraded capabilities"), "{block}");
        // Sorted by tool_id: memory.store before web.search; net.http (healthy) absent.
        let mem = block.find("memory.store").expect("memory.store listed");
        let web = block.find("web.search").expect("web.search listed");
        assert!(mem < web, "entries sorted by tool_id: {block}");
        assert!(!block.contains("net.http"), "healthy tool omitted: {block}");
        assert!(
            block.contains("local embedding model unreachable"),
            "{block}"
        );
        assert!(block.contains("breaker-open"), "{block}");
    }

    /// DoD (M4): once the sweep stamps an embedding-dependent capability
    /// degraded, the capability digest the SOUL reads names it. Drives the same
    /// `ManifestHealth` handle the sweep writes — proving the writer→reader
    /// wiring, not just the renderer.
    #[tokio::test]
    async fn digest_surfaces_degraded_capability_from_manifest_health() {
        let processor = make_processor().await;
        // Healthy by default: no block.
        assert!(!processor
            .capability_digest()
            .await
            .contains("Degraded capabilities"));

        // Simulate one sweep finding the embedder down.
        let mut next = std::collections::HashMap::new();
        next.insert(
            "memory.store".to_string(),
            brain::CapabilityHealth::Degraded {
                reason: "local embedding model unreachable".to_string(),
            },
        );
        processor.manifest_health().replace(next);

        let digest = processor.capability_digest().await;
        assert!(digest.contains("Degraded capabilities"), "{digest}");
        assert!(digest.contains("memory.store"), "{digest}");
        assert!(
            digest.contains("local embedding model unreachable"),
            "{digest}"
        );
    }
}
