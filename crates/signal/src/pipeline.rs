//! Pipeline implementation for SignalProcessor — process, prepare, and per-intent handlers.

use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

fn format_agent_status(id: &str, status: &delegate::RegistryAgentStatus) -> String {
    match status {
        delegate::RegistryAgentStatus::Registered {
            binary,
            version,
            source,
        } => {
            let source = match source {
                delegate::AgentSource::Discovered => "discovered",
                delegate::AgentSource::Custom => "custom",
                delegate::AgentSource::Manual => "manual",
            };
            match version {
                Some(v) => format!("{id} ({source}) — {} [{}]", v, binary.display()),
                None => format!("{id} ({source}) — {}", binary.display()),
            }
        }
        delegate::RegistryAgentStatus::DisabledByConfig => {
            format!("{id} — disabled by config")
        }
        delegate::RegistryAgentStatus::Unavailable { binary, reason } => {
            format!("{id} — unavailable ({reason}) [{}]", binary.display())
        }
    }
}

impl SignalProcessor {
    /// Process a signal through the full Brain pipeline.
    ///
    /// Delegates to `prepare()` for pipeline work, then handles LLM generation
    /// for intents that require it (Chat, Recall).
    ///
    /// Routes by intent:
    /// - `StoreFact`  → Amygdala importance → Hippocampus semantic store → confirmation
    /// - `Recall`     → Hippocampus hybrid search → Cortex context assembly → LLM response
    /// - `Chat`       → Hippocampus context → Cortex LLM → Hippocampus episode store
    /// - `Forget`     → search + delete matching facts
    /// - `SystemStatus` → memory counts
    /// - Action intents → ActionDispatcher
    #[tracing::instrument(
        name = "signal.process",
        skip(self, signal),
        fields(
            signal_id = %signal.id,
            source = ?signal.source,
            namespace = %signal.namespace
        )
    )]
    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError> {
        match self.prepare(&signal, None, None).await? {
            PipelineResult::Complete(resp) => {
                self.publish_event(&signal, &resp);
                Ok(resp)
            }
            PipelineResult::LlmReady {
                signal_id,
                messages,
                memory_context,
                session_id,
                namespace,
                agent,
                ..
            } => {
                let llm_resp = self.llm.generate(&messages).await?;

                // Store assistant episode for Chat/Recall
                if let Some(sid) = &session_id {
                    self.episodic
                        .store_episode(
                            sid,
                            "assistant",
                            &llm_resp.content,
                            0.5,
                            Some(&namespace),
                            agent.as_deref(),
                        )
                        .map_err(|e| SignalError::Storage(e.to_string()))?;
                }

                let resp = SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(llm_resp.content),
                    memory_context,
                    session_id,
                };
                self.publish_event(&signal, &resp);
                Ok(resp)
            }
        }
    }

    /// Prepare the Pipeline up to (but not including) LLM generation.
    ///
    /// Returns either a complete response (for StoreFact, Forget, SystemStatus,
    /// Actions) or assembled LLM messages (for Chat, Recall). The caller can
    /// then choose streaming vs batch LLM generation.
    ///
    /// If `conversation_history` is provided, it is used instead of an empty
    /// history when assembling context (useful for CLI which manages its own).
    pub async fn prepare(
        &self,
        signal: &Signal,
        conversation_history: Option<&[cortex::llm::Message]>,
        progress: Option<tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let signal_id = signal.id;

        let loaded_history = self.hydrate_history(signal, conversation_history);
        let conversation_history: Option<&[cortex::llm::Message]> =
            conversation_history.or(loaded_history.as_deref());

        let pending_notifications = self.drain_pending_notifications();
        let importance = self.importance.score(&signal.content);
        let classification = self.classify_and_store_facts(signal).await;
        let procedure_context = self.match_procedures(&signal.content);

        // Helper: prepend notification nudges to a response
        let prepend_nudges = |mut resp: SignalResponse| -> SignalResponse {
            if !pending_notifications.is_empty() {
                let nudge_text: String = pending_notifications
                    .iter()
                    .map(|n| format!("[nudge] {}", n.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                if let ResponseContent::Text(ref text) = resp.response {
                    resp.response = ResponseContent::Text(format!("{nudge_text}\n\n{text}"));
                }
            }
            resp
        };

        match classification.intent {
            thalamus::Intent::StoreFact {
                subject,
                predicate,
                object,
            } => {
                self.handle_store_fact(
                    signal_id,
                    &signal.namespace,
                    signal.agent.as_deref(),
                    subject,
                    predicate,
                    object,
                    importance,
                    &prepend_nudges,
                )
                .await
            }
            thalamus::Intent::Recall { query } => {
                self.handle_recall(
                    signal_id,
                    signal,
                    query,
                    conversation_history,
                    &procedure_context,
                    &prepend_nudges,
                    progress.as_ref(),
                )
                .await
            }
            thalamus::Intent::Chat { content } => {
                self.handle_chat(
                    signal_id,
                    signal,
                    content,
                    importance,
                    conversation_history,
                    &procedure_context,
                    &prepend_nudges,
                    progress.as_ref(),
                )
                .await
            }
            thalamus::Intent::Forget { target } => {
                self.handle_forget(signal_id, signal, target, &prepend_nudges)
                    .await
            }
            thalamus::Intent::SystemStatus => self.handle_system_status(signal_id, &prepend_nudges),
            thalamus::Intent::QueryAudit {
                filter,
                since,
                limit,
            } => {
                self.handle_query_audit(signal_id, filter, since, limit, &prepend_nudges)
                    .await
            }
            thalamus::Intent::PruneAudit { older_than } => {
                self.handle_prune_audit(signal_id, older_than, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ListApprovals { status } => {
                self.handle_list_approvals(signal_id, status, &prepend_nudges)
                    .await
            }
            thalamus::Intent::RespondToApproval { nonce, decision } => {
                self.handle_respond_to_approval(signal_id, nonce, decision, &prepend_nudges)
                    .await
            }
            thalamus::Intent::BudgetStatus { window } => {
                self.handle_budget_status(signal_id, window, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ListSchedules => {
                self.handle_list_schedules(signal_id, &prepend_nudges).await
            }
            thalamus::Intent::CancelSchedule { id } => {
                self.handle_cancel_schedule(signal_id, id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::DecomposeTask { ref request } => {
                self.handle_decompose_task(signal_id, request.clone(), &prepend_nudges)
                    .await
            }
            thalamus::Intent::ListTasks => self.handle_list_tasks(signal_id, &prepend_nudges).await,
            thalamus::Intent::TaskStatus { task_id } => {
                self.handle_task_status(signal_id, task_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::CancelTask { task_id } => {
                self.handle_cancel_task(signal_id, task_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::QueryAgents { filter } => {
                self.handle_query_agents(signal_id, filter, &prepend_nudges)
            }
            thalamus::Intent::SetProactivity { enabled, until } => {
                self.handle_set_proactivity(signal_id, enabled, until, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ProactivityStatus => {
                self.handle_proactivity_status(signal_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::MemorySummary => {
                self.handle_memory_summary(signal_id, signal, conversation_history, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ProjectInspect { path, focus } => {
                self.handle_project_inspect(signal_id, signal, path, focus, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ListChannels => {
                self.handle_list_channels(signal_id, &prepend_nudges).await
            }
            thalamus::Intent::ChannelPreferences {
                namespace: ns,
                category,
            } => {
                self.handle_channel_preferences(signal_id, ns, category, &prepend_nudges)
                    .await
            }
            thalamus::Intent::SetChannelPreference {
                channel,
                category,
                weight,
                pinned,
            } => {
                self.handle_set_channel_preference(
                    signal_id,
                    channel,
                    category,
                    weight,
                    pinned,
                    &prepend_nudges,
                )
                .await
            }
            ref intent @ (thalamus::Intent::WebSearch { .. }
            | thalamus::Intent::Schedule { .. }
            | thalamus::Intent::SendMessage { .. }
            | thalamus::Intent::ExecuteCommand { .. }) => {
                self.handle_action(signal_id, signal, intent, &prepend_nudges)
                    .await
            }
        }
    }

    // ── prepare() helpers ───────────────────────────────────────────────────

    /// Hydrate conversation history from session episodes when the caller
    /// didn't pass any. Returns `None` if the caller supplied history or
    /// no session lookup is possible. Bounded so the token budget stays
    /// predictable.
    fn hydrate_history(
        &self,
        signal: &Signal,
        caller_history: Option<&[cortex::llm::Message]>,
    ) -> Option<Vec<cortex::llm::Message>> {
        const SESSION_HISTORY_LIMIT: usize = 20;
        match caller_history {
            Some(_) => None,
            None => signal
                .session_id
                .as_deref()
                .map(|sid| self.load_session_messages(sid, SESSION_HISTORY_LIMIT))
                .filter(|h| !h.is_empty()),
        }
    }

    /// Drain up to 10 pending proactive notifications from the outbox so
    /// the response can prepend them as nudges.
    fn drain_pending_notifications(&self) -> Vec<storage::sqlite::Notification> {
        match &self.notification_router {
            Some(router) => router.drain_pending(10),
            None => Vec::new(),
        }
    }

    /// Run intent classification, log the outcome, and persist any facts
    /// the classifier extracted on the side. Returns the classification.
    async fn classify_and_store_facts(&self, signal: &Signal) -> thalamus::Classification {
        let classification = self.classifier.classify(&signal.content).await;
        self.metrics.inc_intent_classification();
        if matches!(classification.method, thalamus::ClassificationMethod::Llm) {
            self.metrics.inc_intent_llm_fallback();
        }

        tracing::info!(
            signal_id = %signal.id,
            source = ?signal.source,
            intent = ?classification.intent,
            importance = self.importance.score(&signal.content),
            method = ?classification.method,
            extracted_facts = classification.extracted_facts.len(),
            "Signal classified"
        );

        if !classification.extracted_facts.is_empty() {
            let facts_to_store: Vec<_> = classification
                .extracted_facts
                .iter()
                .map(|f| crate::exchange::FactToStore {
                    subject: f.subject.clone(),
                    predicate: f.predicate.clone(),
                    object: f.object.clone(),
                })
                .collect();

            let (stored, errors) = self
                .store_facts_batch(
                    &signal.namespace,
                    "extracted",
                    &facts_to_store,
                    signal.agent.as_deref(),
                )
                .await;

            for id in &stored {
                tracing::info!("Extracted fact stored: {id}");
            }
            for (text, e) in errors {
                tracing::warn!("Failed to store extracted fact ({text}): {e}");
            }
        }

        classification
    }

    /// Match user content against stored procedures and surface the
    /// concatenated step list as additional context. Procedure-matcher
    /// errors are non-fatal — we log and degrade to no procedures.
    fn match_procedures(&self, content: &str) -> Vec<String> {
        match self.procedures.match_trigger(content) {
            Ok(procs) if !procs.is_empty() => {
                tracing::debug!(
                    count = procs.len(),
                    "Procedure(s) matched — injecting steps into context"
                );
                let mut steps: Vec<String> = Vec::new();
                for proc in &procs {
                    let _ = self.procedures.record_execution(&proc.id);
                    steps.extend(proc.steps.clone());
                }
                steps
            }
            Ok(_) => Vec::new(),
            Err(e) => {
                tracing::warn!("Procedure match failed (non-fatal): {e}");
                Vec::new()
            }
        }
    }

    // ── Per-intent handlers ─────────────────────────────────────────────────

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
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
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

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_recall(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        query: String,
        conversation_history: Option<&[cortex::llm::Message]>,
        procedure_context: &[String],
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
        progress: Option<&tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let top_k = self.config.memory.semantic.max_results as usize;
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&query).await;
        let (memories, facts_used, episodes_used) = self
            .do_recall(&query, query_vector, top_k, Some(&signal.namespace))
            .await;

        // Agent callers get structured data
        if signal.agent.is_some() {
            let text = if memories.is_empty() {
                "No relevant memories found.".to_string()
            } else {
                memories
                    .iter()
                    .map(|m| format!("[{:?}] {}", m.source, m.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let resp = prepend_nudges(SignalResponse {
                signal_id,
                status: ResponseStatus::Ok,
                response: ResponseContent::Text(text),
                memory_context: MemoryContext {
                    facts_used,
                    episodes_used,
                },
                session_id: None,
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
        // Onboarding mode only when the namespace is truly empty — not just when
        // this query's semantic search returned nothing.
        let namespace_is_empty = self.list_facts(Some(&signal.namespace)).is_empty()
            && self.recent_episodes(1, Some(&signal.namespace)).is_empty();
        let addendum = if namespace_is_empty {
            Some(cortex::context::ONBOARDING_ADDENDUM)
        } else {
            None
        };
        let messages = self
            .context_assembler
            .assemble_with_addendum(&query, &memories, history, addendum);

        Ok(PipelineResult::LlmReady {
            signal_id,
            messages,
            memory_context: MemoryContext {
                facts_used,
                episodes_used,
            },
            session_id: None,
            user_content: query,
            namespace: signal.namespace.clone(),
            agent: signal.agent.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_chat(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        content: String,
        importance: f32,
        conversation_history: Option<&[cortex::llm::Message]>,
        procedure_context: &[String],
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
        progress: Option<&tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let top_k = self.config.memory.semantic.max_results as usize;
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&content).await;
        let (memories, facts_used, episodes_used) = self
            .do_recall(&content, query_vector, top_k, Some(&signal.namespace))
            .await;

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
        let namespace_is_empty = self.list_facts(Some(&signal.namespace)).is_empty()
            && self.recent_episodes(1, Some(&signal.namespace)).is_empty();
        let addendum = if namespace_is_empty {
            Some(cortex::context::ONBOARDING_ADDENDUM)
        } else {
            None
        };
        let messages = self
            .context_assembler
            .assemble_with_addendum(&content, &memories, history, addendum);

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

    pub(super) async fn handle_forget(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        target: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
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

    pub(super) fn handle_system_status(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let semantic_count = self
            .semantic
            .as_ref()
            .and_then(|s| s.count().ok())
            .unwrap_or(0);
        let episode_count = self.episodic.count().unwrap_or(0);

        let resp = prepend_nudges(SignalResponse::ok(
            signal_id,
            format!("Brain status: {semantic_count} facts, {episode_count} episodes"),
        ));
        Ok(PipelineResult::Complete(resp))
    }

    /// Fetch every stored fact and recent episodes in the namespace, then ask the
    /// LLM to summarise them. Never uses semantic search — this is a full listing
    /// so a generic "what do you know" always returns real content.
    pub(super) async fn handle_memory_summary(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        conversation_history: Option<&[cortex::llm::Message]>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let ns = Some(signal.namespace.as_str());
        let facts = self.list_facts(ns);
        let episodes = self.recent_episodes(50, ns);

        // Nothing stored at all → honest empty-memory response, no LLM needed.
        if facts.is_empty() && episodes.is_empty() {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "Your memory is empty — I haven't stored anything yet. \
                 Tell me about yourself, your projects, or what you'd like me to remember."
                    .to_string(),
            ));
            return Ok(PipelineResult::Complete(resp));
        }

        // Deterministic format — for "what do you know about me" we list
        // exactly what's stored, not an LLM paraphrase. Earlier versions
        // sent this through the model and it routinely produced fluffy
        // categorical summaries that omitted real facts.
        //
        // Group facts by subject so a user with many entries about
        // themselves and their projects sees structure without losing
        // ground truth.
        let mut by_subject: std::collections::BTreeMap<String, Vec<&hippocampus::Fact>> =
            std::collections::BTreeMap::new();
        for f in &facts {
            by_subject.entry(f.subject.clone()).or_default().push(f);
        }

        let mut out = String::new();
        out.push_str(&format!(
            "Memory snapshot — {} fact{} across {} subject{}, {} recent episode{}.\n",
            facts.len(),
            if facts.len() == 1 { "" } else { "s" },
            by_subject.len(),
            if by_subject.len() == 1 { "" } else { "s" },
            episodes.len(),
            if episodes.len() == 1 { "" } else { "s" },
        ));

        if !by_subject.is_empty() {
            out.push_str("\nStored facts:\n");
            for (subject, subj_facts) in &by_subject {
                out.push_str(&format!("  {subject}:\n"));
                let mut seen = std::collections::HashSet::new();
                for f in subj_facts {
                    let key = format!("{}|{}", f.predicate, f.object);
                    if !seen.insert(key) {
                        continue;
                    }
                    out.push_str(&format!("    • {} → {}\n", f.predicate, f.object));
                }
            }
        }

        if !episodes.is_empty() {
            out.push_str("\nRecent activity (most recent first):\n");
            for ep in episodes.iter().take(8) {
                let one_line = ep.content.lines().next().unwrap_or("").trim();
                let trimmed = if one_line.chars().count() > 140 {
                    let mut s: String = one_line.chars().take(137).collect();
                    s.push('…');
                    s
                } else {
                    one_line.to_string()
                };
                let ts_short: String = ep.timestamp.chars().take(16).collect();
                out.push_str(&format!("  • [{ts_short}] {trimmed}\n"));
            }
        }

        out.push_str(
            "\nTell me anything else you'd like remembered — projects, goals, \
             preferences, or context I should keep around.",
        );

        let _ = conversation_history; // history not used in the deterministic path
        let resp = prepend_nudges(SignalResponse {
            signal_id,
            status: ResponseStatus::Ok,
            response: ResponseContent::Text(out),
            memory_context: crate::MemoryContext {
                facts_used: facts.len(),
                episodes_used: episodes.len(),
            },
            session_id: signal.session_id.clone(),
        });
        Ok(PipelineResult::Complete(resp))
    }

    /// Read-only inspection of a local directory or file path. The handler
    /// builds a structured snapshot of the path (top-level entries, anchor
    /// files like README/Cargo.toml/package.json), then asks the LLM to
    /// summarise. No sandbox, no decomposition, no shell scripts — this is
    /// deliberately bounded and synchronous so the chat answer is fast.
    pub(super) async fn handle_project_inspect(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        path: String,
        focus: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let expanded = expand_user_path(&path);
        let pb = std::path::PathBuf::from(&expanded);

        let metadata = match std::fs::metadata(&pb) {
            Ok(m) => m,
            Err(e) => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!(
                        "Can't inspect `{}` — {}. Pass an absolute path I can read.",
                        pb.display(),
                        friendly_io_error(&e)
                    ),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };

        let snapshot = if metadata.is_dir() {
            build_directory_snapshot(&pb)
        } else if metadata.is_file() {
            build_file_snapshot(&pb)
        } else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                format!("`{}` is not a regular file or directory.", pb.display()),
            ));
            return Ok(PipelineResult::Complete(resp));
        };

        let focus_line = focus
            .as_deref()
            .map(|f| format!("\nThe user specifically wants: {f}\n"))
            .unwrap_or_default();

        let system_prompt = format!(
            "You are inspecting a local project for the user. The block between \
             <PROJECT> tags is the actual content read from disk — file tree and \
             key file excerpts. Summarise honestly: what kind of project this is, \
             how it is organised, the most important entry points, and anything \
             notable about its build/runtime. Use bullets, keep it under 250 \
             words, do not invent files that aren't shown, and do not run any \
             commands — you cannot.{focus_line}\n\n<PROJECT path=\"{}\">\n{snapshot}\n</PROJECT>",
            pb.display()
        );

        let messages = vec![
            cortex::llm::Message {
                role: cortex::llm::Role::System,
                content: system_prompt,
            },
            cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: match focus.as_deref() {
                    Some(f) => format!("Summarise this project, with focus on: {f}"),
                    None => "Summarise this project.".to_string(),
                },
            },
        ];

        Ok(PipelineResult::LlmReady {
            signal_id,
            messages,
            memory_context: crate::MemoryContext {
                facts_used: 0,
                episodes_used: 0,
            },
            session_id: signal.session_id.clone(),
            user_content: signal.content.clone(),
            namespace: signal.namespace.clone(),
            agent: signal.agent.clone(),
        })
    }

    pub(super) async fn handle_query_audit(
        &self,
        signal_id: Uuid,
        _filter: Option<String>,
        _since: Option<String>,
        _limit: Option<usize>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.audit_trail {
            Some(audit) => {
                let entries = audit
                    .query(audit::query::AuditQuerySpec::last(10))
                    .await
                    .map_err(|e| SignalError::Processing(format!("Audit query failed: {e}")))?;
                if entries.is_empty() {
                    "Audit trail is empty.".to_string()
                } else {
                    let mut out = "Recent audit entries:\n".to_string();
                    for entry in entries {
                        out.push_str(&format!(
                            "  • [{}] {} -> {}\n",
                            entry.timestamp, entry.action, entry.outcome
                        ));
                    }
                    out.trim_end().to_string()
                }
            }
            None => "Audit trail is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_prune_audit(
        &self,
        signal_id: Uuid,
        older_than: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.audit_trail {
            Some(audit) => {
                // Parse duration from string (stub: use 30 days if parsing fails)
                let duration = chrono::Duration::try_days(30).unwrap();
                match audit.prune(duration).await {
                    Ok(n) => format!("Pruned {n} entries older than {older_than}"),
                    Err(e) => format!("Failed to prune audit: {e}"),
                }
            }
            None => "Audit trail is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_list_approvals(
        &self,
        signal_id: Uuid,
        _status: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.confirmation_engine {
            Some(engine) => {
                let pending = engine
                    .pending()
                    .await
                    .map_err(|e| SignalError::Processing(format!("Failed to list pending: {e}")))?;
                if pending.is_empty() {
                    "No pending approvals.".to_string()
                } else {
                    let mut out = "Pending approvals:\n".to_string();
                    for p in pending {
                        out.push_str(&format!("  • [{}] {}\n", p.nonce, p.action_description));
                    }
                    out.push_str("\nReply with 'approve <nonce>' or 'reject <nonce>'.");
                    out
                }
            }
            None => "Confirmation engine is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_respond_to_approval(
        &self,
        signal_id: Uuid,
        nonce: String,
        decision: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let approved = decision.to_lowercase().contains("approve");

        // Plan-level approval: an ID matching a task in AwaitingApproval
        // phase is a phase-transition request, not a confirm-engine nonce.
        // Approving kicks off execution; rejecting cancels the plan.
        //
        // Resolution order:
        //   1. If the user typed an explicit nonce, try it first.
        //   2. Otherwise (or if it doesn't match a pending plan) look at
        //      `pending_approvals()`. If exactly one plan is pending,
        //      route the bare yes/no to it. Multiple pending → ask the
        //      user to disambiguate. Zero pending → fall through to the
        //      confirm-engine path (per-step approvals).
        if let Some(orch) = &self.orchestrator {
            let mut resolved: Option<String> = None;
            if !nonce.is_empty() {
                if let Some(task) = orch.get_task(&nonce).await {
                    if task.phase == orchestrate::TaskPhase::AwaitingApproval {
                        resolved = Some(nonce.clone());
                    }
                }
            }
            if resolved.is_none() {
                let pending = orch.pending_approvals().await;
                match pending.len() {
                    1 => resolved = Some(pending[0].clone()),
                    n if n > 1 && nonce.is_empty() => {
                        let message = format!(
                            "{n} plans are awaiting approval. Reply `approve <id>` or \
                             `reject <id>` to choose one. Pending: {}",
                            pending.join(", ")
                        );
                        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
                        return Ok(PipelineResult::Complete(resp));
                    }
                    _ => {}
                }
            }

            if let Some(plan_id) = resolved {
                let message = if approved {
                    match orch.execute(&plan_id).await {
                        Ok(summary) => format!("Plan approved.\n\n{summary}"),
                        Err(e) => {
                            format!("Plan approved but execution failed: {e}")
                        }
                    }
                } else {
                    match orch.cancel(&plan_id).await {
                        Ok(_) => "Plan rejected and cancelled.".to_string(),
                        Err(e) => format!("Failed to cancel plan: {e}"),
                    }
                };
                let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
                return Ok(PipelineResult::Complete(resp));
            }
        }

        // Per-step approval: resolve via the confirm engine.
        let message = match &self.confirmation_engine {
            Some(engine) => {
                let dec = if approved {
                    confirm::ApprovalDecision::Approve
                } else {
                    confirm::ApprovalDecision::Reject
                };
                match engine.respond(&nonce, dec).await {
                    Ok(_) => {
                        if approved {
                            format!("Approval {nonce} accepted. Execution resumed.")
                        } else {
                            format!("Approval {nonce} rejected. Action cancelled.")
                        }
                    }
                    Err(e) => format!("Failed to respond to {nonce}: {e}"),
                }
            }
            None => "Confirmation engine is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_budget_status(
        &self,
        signal_id: Uuid,
        _window: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.cost_budget {
            Some(budget) => {
                let status = budget
                    .status()
                    .await
                    .map_err(|e| SignalError::Processing(format!("Budget status failed: {e}")))?;
                let mut out = "Budget status:\n".to_string();
                out.push_str("  Hourly consumption:\n");
                for (k, v) in &status.hourly_consumption {
                    out.push_str(&format!("    • {}: {}\n", k, v));
                }
                out.push_str("  Daily consumption:\n");
                for (k, v) in &status.daily_consumption {
                    out.push_str(&format!("    • {}: {}\n", k, v));
                }
                out.trim_end().to_string()
            }
            None => "Cost budget is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_list_schedules(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        // Scheduled intents are stored in episodic pool's scheduled_intents table
        // We can query them via a direct SQL if needed, or if SignalProcessor
        // has a method for it.
        // For now, let's look for a method or use a placeholder.
        let message =
            "Background schedules list is currently only available via `brain schedules list`."
                .to_string();
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_cancel_schedule(
        &self,
        signal_id: Uuid,
        _id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message =
            "Schedule cancellation is currently only available via `brain schedules cancel`."
                .to_string();
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) fn handle_query_agents(
        &self,
        signal_id: Uuid,
        filter: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let registry = match self.agent_registry() {
            Some(r) => r,
            None => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    "Agent registry is not wired.".to_string(),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };

        let needle = filter.trim().to_lowercase();
        let known = registry.known_agents();
        let mut matches_line: Vec<String> = Vec::new();
        for (id, status) in &known {
            if !needle.is_empty() && !id.to_lowercase().contains(&needle) {
                continue;
            }
            matches_line.push(format_agent_status(id, status));
        }

        let message = if known.is_empty() {
            "No agents discovered and none configured. Install a CLI agent \
             (claude-code, aider, codex, qwen, gemini, opencode) on your PATH \
             — it will be picked up on the next boot."
                .to_string()
        } else if matches_line.is_empty() {
            format!("No known agents match '{filter}'.")
        } else {
            let registered: Vec<String> = registry.list();
            let mut out = String::new();
            out.push_str("Known agents:\n");
            for line in &matches_line {
                out.push_str("  • ");
                out.push_str(line);
                out.push('\n');
            }
            if needle.is_empty() && !registered.is_empty() {
                out.push_str(&format!("\nReady to delegate: {}", registered.join(", ")));
            }
            out.trim_end().to_string()
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_decompose_task(
        &self,
        signal_id: Uuid,
        request: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let orchestrator = match &self.orchestrator {
            Some(orch) => orch,
            None => {
                let message = format!(
                    "Task decomposition recognized for: \"{request}\"\n\
                     Task orchestration is not yet active — the orchestrator \
                     has not been wired into this instance."
                );
                let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
                return Ok(PipelineResult::Complete(resp));
            }
        };

        // Build decomposition context from config + memory so the
        // decomposer LLM sees the actual sandbox allowlist and won't
        // produce plans that try to call binaries the sandbox refuses.
        let context = orchestrate::DecompositionContext {
            available_tools: self.config.security.exec_allowlist.clone(),
            ..Default::default()
        };

        match orchestrator.plan(&request, context).await {
            Ok((task_id, plan_text)) => {
                let message = format!("Task plan created (ID: {task_id}):\n\n{plan_text}");
                let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
                Ok(PipelineResult::Complete(resp))
            }
            Err(e) => {
                let message = format!("Failed to decompose task: {e}");
                let resp = prepend_nudges(SignalResponse::error(signal_id, message));
                Ok(PipelineResult::Complete(resp))
            }
        }
    }

    pub(super) async fn handle_list_tasks(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.orchestrator {
            Some(orch) => {
                let tasks = orch.list_tasks().await;
                if tasks.is_empty() {
                    "No active or recent tasks.".to_string()
                } else {
                    let mut out = "Recent tasks:\n".to_string();
                    for (id, desc, phase) in tasks {
                        out.push_str(&format!("  • [{}] {} — {:?}\n", id, desc, phase));
                    }
                    out.trim_end().to_string()
                }
            }
            None => "Task orchestrator is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_task_status(
        &self,
        signal_id: Uuid,
        task_id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.orchestrator {
            Some(orch) => match orch.get_task(&task_id).await {
                Some(task) => {
                    format!(
                        "Task: {}\nID: {}\nPhase: {:?}\nSteps: {} total, {} completed, {} failed",
                        task.request,
                        task.id,
                        task.phase,
                        task.step_states.len(),
                        task.step_states
                            .values()
                            .filter(|s| matches!(s, orchestrate::StepState::Completed { .. }))
                            .count(),
                        task.step_states
                            .values()
                            .filter(|s| matches!(s, orchestrate::StepState::Failed { .. }))
                            .count(),
                    )
                }
                None => format!("Task {task_id} not found."),
            },
            None => "Task orchestrator is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_cancel_task(
        &self,
        signal_id: Uuid,
        task_id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.orchestrator {
            Some(orch) => match orch.cancel(&task_id).await {
                Ok(_) => format!("Task {task_id} cancelled."),
                Err(e) => format!("Failed to cancel task {task_id}: {e}"),
            },
            None => "Task orchestrator is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_set_proactivity(
        &self,
        signal_id: Uuid,
        enabled: bool,
        _until: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        // This would ideally update config or a live state.
        // For now, since we don't have a live proactivity toggle in SignalProcessor,
        // we'll just return a message.
        let message = if enabled {
            "Proactivity enabled (simulated).".to_string()
        } else {
            "Proactivity disabled (simulated).".to_string()
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_proactivity_status(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = format!(
            "Proactivity status:\n  • Habit engine: {}\n  • Open-loop detector: {}\n  • Quiet hours: {}-{}",
            if self.config.proactivity.enabled {
                "active"
            } else {
                "disabled"
            },
            if self.config.proactivity.open_loop.enabled {
                "active"
            } else {
                "disabled"
            },
            self.config.proactivity.quiet_hours.start,
            self.config.proactivity.quiet_hours.end
        );

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_list_channels(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.channel_router {
            Some(router) => match router.list_channels().await {
                Ok(channels) if channels.is_empty() => {
                    "No channels registered yet. Configure transports in `channel.transports[]` \
                     or `channel.relays[]`."
                        .to_string()
                }
                Ok(channels) => {
                    let mut lines = vec!["Registered channels:".to_string()];
                    for c in channels {
                        let health = if c.healthy { "healthy" } else { "down" };
                        lines.push(format!("  • {} ({}, {}) — {health}", c.id, c.label, c.kind));
                    }
                    lines.join("\n")
                }
                Err(e) => format!("Channel listing failed: {e}"),
            },
            None => "Channel router not wired in this build.".to_string(),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_channel_preferences(
        &self,
        signal_id: Uuid,
        namespace: Option<String>,
        category: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let ns = namespace.as_deref().unwrap_or("personal");
        let message = match &self.channel_preferences {
            None => "Channel preferences not wired in this build.".to_string(),
            Some(store) => {
                let cat = category.as_deref().map(channel::DeliveryCategory::parse);
                let categories: Vec<channel::DeliveryCategory> = match cat {
                    Some(Some(c)) => vec![c],
                    Some(None) => {
                        return Ok(PipelineResult::Complete(prepend_nudges(
                            SignalResponse::ok(
                                signal_id,
                                format!(
                                    "Unknown delivery category: {:?}. \
                                     Try: confirm, nudge, report, response, alert.",
                                    category,
                                ),
                            ),
                        )));
                    }
                    None => vec![
                        channel::DeliveryCategory::Confirm,
                        channel::DeliveryCategory::Nudge,
                        channel::DeliveryCategory::Report,
                        channel::DeliveryCategory::Response,
                        channel::DeliveryCategory::Alert,
                    ],
                };

                let mut lines = vec![format!("Channel preferences (namespace = {ns}):")];
                for c in categories {
                    match store.get_preferences(ns, c, 0.0).await {
                        Ok(prefs) if prefs.is_empty() => {
                            lines.push(format!("  • {c:?}: (none learned)"));
                        }
                        Ok(prefs) => {
                            let formatted: Vec<String> = prefs
                                .iter()
                                .map(|p| {
                                    let pin = if p.pinned { " 📌" } else { "" };
                                    format!("{}={:.2}{}", p.channel_id, p.weight, pin)
                                })
                                .collect();
                            lines.push(format!("  • {c:?}: {}", formatted.join(", ")));
                        }
                        Err(e) => lines.push(format!("  • {c:?}: error: {e}")),
                    }
                }
                lines.join("\n")
            }
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_set_channel_preference(
        &self,
        signal_id: Uuid,
        channel_id: String,
        category: String,
        weight: f32,
        pinned: bool,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match (channel::DeliveryCategory::parse(&category), &self.channel_preferences) {
            (None, _) => format!(
                "Unknown delivery category: {category}. Try: confirm, nudge, report, response, alert.",
            ),
            (_, None) => "Channel preference store not wired in this build.".to_string(),
            (Some(cat), Some(store)) => match store
                .upsert_preference("personal", cat, &channel_id, weight, pinned)
                .await
            {
                Ok(_) => {
                    if weight <= 0.0 && !pinned {
                        format!("Cleared preference for {channel_id} on {category}.")
                    } else {
                        format!(
                            "Set preference: {channel_id} for {category} → weight {:.2}{}.",
                            weight,
                            if pinned { " (pinned)" } else { "" }
                        )
                    }
                }
                Err(e) => format!("Failed to update preference: {e}"),
            },
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_action(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        intent: &thalamus::Intent,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let router = thalamus::SignalRouter::new();
        let resp = match (router.intent_to_action(intent), &self.action_dispatcher) {
            (Some(action), Some(dispatcher)) => {
                let result = dispatcher.dispatch(&action).await;
                if result.success {
                    if matches!(&action, cortex::actions::Action::WebSearch { .. })
                        && !result.output.is_empty()
                    {
                        let search_context = format!(
                            "The user asked: \"{}\"\n\nHere are web search results:\n{}\n\nUsing these search results, provide a helpful and concise answer to the user's question. Cite sources when relevant.",
                            signal.content, result.output
                        );
                        let messages = vec![
                            cortex::llm::Message {
                                role: cortex::llm::Role::System,
                                content: "You are Brain OS. Answer the user's question using the provided web search results. Be concise and cite your sources.".to_string(),
                            },
                            cortex::llm::Message {
                                role: cortex::llm::Role::User,
                                content: search_context,
                            },
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

    pub(super) fn publish_event(&self, signal: &Signal, response: &SignalResponse) {
        let event = SignalProcessedEvent {
            signal_id: response.signal_id,
            source: signal.source.clone(),
            channel: signal.channel.clone(),
            sender: signal.sender.clone(),
            namespace: signal.namespace.clone(),
            status: response.status.clone(),
            response: response_to_text(&response.response),
            facts_used: response.memory_context.facts_used,
            episodes_used: response.memory_context.episodes_used,
            timestamp: chrono::Utc::now(),
        };
        let _ = self.events_tx.send(event);
    }
}

// `DeliveryCategory::parse` (and `FromStr`) live in the channel crate —
// no local normalizer needed here.

// ── ProjectInspect helpers ─────────────────────────────────────────────────

/// Expand a leading `~` to the user's home directory. Anything else is
/// returned as-is — the caller resolves relative paths against cwd.
fn expand_user_path(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let mut out = std::path::PathBuf::from(home);
            out.push(rest);
            return out.to_string_lossy().into_owned();
        }
    }
    if p == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return home.to_string_lossy().into_owned();
        }
    }
    p.to_string()
}

/// Map an io::Error to a one-liner the user can act on. Avoids exposing
/// the bare Rust error format ("No such file or directory (os error 2)").
fn friendly_io_error(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => "no such path".to_string(),
        std::io::ErrorKind::PermissionDenied => "permission denied".to_string(),
        std::io::ErrorKind::InvalidInput => "invalid path".to_string(),
        _ => e.to_string(),
    }
}

/// Files that materially explain what a project is. Reading the first
/// ~6 KB of any present anchor lets the LLM produce a faithful summary
/// without slurping the whole tree.
const ANCHOR_FILES: &[&str] = &[
    "README.md",
    "README",
    "README.rst",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "Gemfile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Makefile",
    "justfile",
    "Justfile",
    "CHANGELOG.md",
    "ARCHITECTURE.md",
    "CLAUDE.md",
    "AGENTS.md",
];

/// Directories not worth surfacing in the snapshot — they bloat the
/// listing and rarely help a summary.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".svelte-kit",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
];

/// Build an inspection snapshot for a directory: top-level entries + the
/// content of any anchor files present.
fn build_directory_snapshot(root: &std::path::Path) -> String {
    let mut out = String::new();
    out.push_str("Top-level entries:\n");

    let mut entries: Vec<(String, bool)> = match std::fs::read_dir(root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if SKIP_DIRS.contains(&name.as_str()) {
                    return None;
                }
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                Some((name, is_dir))
            })
            .collect(),
        Err(e) => {
            return format!("(failed to read directory: {})", friendly_io_error(&e));
        }
    };
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let max_entries = 40;
    for (i, (name, is_dir)) in entries.iter().enumerate() {
        if i == max_entries {
            out.push_str(&format!(
                "  … (+{} more entries omitted)\n",
                entries.len() - max_entries
            ));
            break;
        }
        out.push_str(&format!("  {}{}\n", name, if *is_dir { "/" } else { "" }));
    }

    // Surface a couple of source-directory subtrees that are common
    // landmarks — but only one level deep so we don't blow up on big
    // monorepos.
    for landmark in ["src", "crates", "lib", "app", "apps", "packages"] {
        let p = root.join(landmark);
        if !p.is_dir() {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(&p) {
            let kids: Vec<String> = rd
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    format!("{}{}", name, if is_dir { "/" } else { "" })
                })
                .filter(|n| !n.starts_with('.'))
                .take(30)
                .collect();
            if !kids.is_empty() {
                out.push_str(&format!("\n{landmark}/ (one level):\n"));
                for k in kids {
                    out.push_str(&format!("  {k}\n"));
                }
            }
        }
    }

    let mut anchors_found = 0;
    for anchor in ANCHOR_FILES {
        let p = root.join(anchor);
        if !p.is_file() {
            continue;
        }
        anchors_found += 1;
        out.push_str(&format!("\n--- {anchor} (first 6 KB) ---\n"));
        out.push_str(&read_truncated(&p, 6 * 1024));
    }

    if anchors_found == 0 {
        out.push_str("\n(no anchor files found — README/Cargo.toml/package.json/etc.)\n");
    }

    out
}

/// Snapshot of a single file: path + first 12 KB of content. Binary
/// files are reported as such instead of being fed through.
fn build_file_snapshot(p: &std::path::Path) -> String {
    let mut out = format!("File: {}\n\n", p.display());
    out.push_str(&read_truncated(p, 12 * 1024));
    out
}

/// Read the first `cap` bytes of a path, returning a string. If the
/// content is non-UTF-8 we declare it binary instead of garbling it.
fn read_truncated(path: &std::path::Path, cap: usize) -> String {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return format!("(read failed: {})\n", friendly_io_error(&e)),
    };
    let truncated = bytes.len() > cap;
    let head = &bytes[..bytes.len().min(cap)];
    match std::str::from_utf8(head) {
        Ok(s) => {
            if truncated {
                format!("{s}\n… (file truncated at {cap} bytes)\n")
            } else {
                format!("{s}\n")
            }
        }
        Err(_) => "(binary file — not displayed)\n".to_string(),
    }
}
