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
        // Register a cancellation notify for this signal id. The guard removes
        // it on drop so abort/error paths don't leak entries.
        let signal_id = signal.id;
        let cancel = self.register_cancel(signal_id).await;
        let _cancel_guard = CancelGuard {
            processor: self,
            signal_id,
        };

        self.publish_signal_received(&signal).await;

        // The fast-classify path inside prepare() may return Complete; the slow
        // LLM-generation path returns LlmReady. Both are protected by the
        // cancel notify below for any awaits that would otherwise block.
        let _ = &cancel; // silence unused warning when there's no await checkpoint
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
                let provider_name = self.llm.name().to_string();
                let gate = crate::budget_guard::check_llm_input(
                    self.cost_budget(),
                    &provider_name,
                    &messages,
                )
                .await;
                let estimated_input = match gate {
                    crate::budget_guard::BudgetGate::Blocked { message } => {
                        let resp = SignalResponse {
                            signal_id,
                            status: ResponseStatus::Ok,
                            response: ResponseContent::Text(message),
                            memory_context,
                            session_id,
                        };
                        self.publish_event(&signal, &resp);
                        return Ok(resp);
                    }
                    crate::budget_guard::BudgetGate::Proceed {
                        estimated_input_tokens,
                    } => estimated_input_tokens,
                };

                let llm_resp = tokio::select! {
                    biased;
                    _ = cancel.notified() => {
                        return Ok(self.cancelled_response(signal_id, &signal).await);
                    }
                    r = self.llm.generate(&messages) => r?,
                };

                crate::budget_guard::record_llm_usage(
                    self.cost_budget(),
                    &provider_name,
                    llm_resp.usage.as_ref(),
                    estimated_input,
                )
                .await;

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
        let classification = self
            .classify_and_store_facts(signal, conversation_history)
            .await;
        let procedure_context = self.match_procedures(&signal.content);

        // Identity gate. Runs after classification (so we know the intent)
        // and before any handler executes. Three short-circuit outcomes:
        // EscalateToUser → text response pointing to the pending approval;
        // Deny → error response; otherwise proceed.
        if let Some(early) = self
            .enforce_identity(signal, signal_id, &classification.intent)
            .await
        {
            return Ok(PipelineResult::Complete(early));
        }

        // Inline confirmation gate. Identity says "this principal may
        // attempt this verb"; the confirmation gate says "but for tiers
        // ≥ Destructive, a human (or a standing approval) must consent."
        // Reflex firings travel the same path — they have no way to
        // bypass this checkpoint.
        if let Some(early) = self
            .confirmation_gate(signal, signal_id, &classification.intent)
            .await
        {
            return Ok(PipelineResult::Complete(early));
        }

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
            thalamus::Intent::ListStandingApprovals => {
                self.handle_list_standing_approvals(signal_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::RevokeStandingApproval { id } => {
                self.handle_revoke_standing_approval(signal_id, id, &prepend_nudges)
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
            thalamus::Intent::CancelSignal {
                signal_id: target_id,
            } => {
                self.handle_cancel_signal(signal_id, target_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::QueryAgents { filter } => {
                self.handle_query_agents(signal_id, filter, &prepend_nudges)
            }
            thalamus::Intent::DelegateTask { agent, prompt } => {
                self.handle_delegate_task(signal_id, agent, prompt, &prepend_nudges)
                    .await
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
            thalamus::Intent::OpenTerminalSession { program, args, cwd } => {
                self.handle_open_terminal_session(
                    signal_id,
                    signal,
                    program,
                    args,
                    cwd,
                    &prepend_nudges,
                )
                .await
            }
            thalamus::Intent::ListTerminalSessions => {
                self.handle_list_terminal_sessions(signal_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::CloseTerminalSession { session_id } => {
                self.handle_close_terminal_session(signal_id, session_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::MountMcpServer {
                name,
                transport,
                command_or_url,
            } => {
                self.handle_mount_mcp_server(
                    signal_id,
                    name,
                    transport,
                    command_or_url,
                    &prepend_nudges,
                )
                .await
            }
            thalamus::Intent::UnmountMcpServer { name } => {
                self.handle_unmount_mcp_server(signal_id, name, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ListMcpServers => {
                self.handle_list_mcp_servers(signal_id, &prepend_nudges)
                    .await
            }
            thalamus::Intent::ToolCall(token) => {
                self.handle_tool_call(signal_id, *token, &prepend_nudges)
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
    /// the classifier extracted on the side. When `history` is supplied,
    /// the LLM fallback uses it to disambiguate follow-up replies from
    /// new biographical claims.
    async fn classify_and_store_facts(
        &self,
        signal: &Signal,
        history: Option<&[cortex::llm::Message]>,
    ) -> thalamus::Classification {
        let history = history.unwrap_or(&[]);
        let classification = self
            .classifier
            .classify_with_history(&signal.content, history)
            .await;
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

        if let Some(observer) = &self.observer {
            let intent_summary = intent_summary_of(&classification.intent);
            let ev = observe::BrainEvent::IntentClassified {
                id: signal.id,
                intent: intent_summary,
                confidence: classification.confidence as f32,
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }

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

        let mut md = crate::render::Markdown::new();
        md.push_line(format!(
            "**Memory snapshot** — {} fact{} across {} subject{}, {} recent episode{}.",
            facts.len(),
            if facts.len() == 1 { "" } else { "s" },
            by_subject.len(),
            if by_subject.len() == 1 { "" } else { "s" },
            episodes.len(),
            if episodes.len() == 1 { "" } else { "s" },
        ));

        if !by_subject.is_empty() {
            md.push_heading(3, "Stored facts");
            for (subject, subj_facts) in &by_subject {
                md.push_bullet(0, format!("**{subject}**"));
                let mut seen = std::collections::HashSet::new();
                for f in subj_facts {
                    let key = format!("{}|{}", f.predicate, f.object);
                    if !seen.insert(key) {
                        continue;
                    }
                    md.push_bullet(1, format!("`{}` → {}", f.predicate, f.object));
                }
            }
        }

        if !episodes.is_empty() {
            md.push_heading(3, "Recent activity");
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
                md.push_bullet(0, format!("*{ts_short}* — {trimmed}"));
            }
        }

        md.push_line(
            "Tell me anything else you'd like remembered — projects, goals, \
             preferences, or context I should keep around.",
        );

        let _ = conversation_history; // history not used in the deterministic path
        let resp = prepend_nudges(SignalResponse {
            signal_id,
            status: ResponseStatus::Ok,
            response: ResponseContent::Text(md.build()),
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
        let requested = std::path::PathBuf::from(&expanded);

        // Issue 119: canonicalize first, then enforce that the resolved
        // path lives under `security.allowed_paths` (default: `$HOME`).
        // `canonicalize` resolves `..` and symlinks so `~/code/../../etc`
        // can't escape the configured root.
        let canonical = match std::fs::canonicalize(&requested) {
            Ok(p) => p,
            Err(e) => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!(
                        "Can't inspect `{}` — {}. Pass an absolute path I can read.",
                        requested.display(),
                        friendly_io_error(&e)
                    ),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };

        let allowed_roots = resolve_allowed_roots(&self.config.security.allowed_paths);
        if !path_under_any_root(&canonical, &allowed_roots) {
            tracing::warn!(
                requested = %requested.display(),
                resolved = %canonical.display(),
                "project_inspect rejected — path outside security.allowed_paths"
            );
            let roots = allowed_roots
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                format!(
                    "I can't inspect `{}` — it's outside the configured allowed_paths ({}). \
                     Edit `security.allowed_paths` if you need to widen the sandbox.",
                    requested.display(),
                    if roots.is_empty() {
                        "$HOME".to_string()
                    } else {
                        roots
                    },
                ),
            ));
            return Ok(PipelineResult::Complete(resp));
        }
        let pb = canonical;

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
                    let mut md = crate::render::Markdown::new();
                    md.push_heading(3, "Recent audit entries");
                    for entry in entries {
                        md.push_bullet(
                            0,
                            format!(
                                "*{}* — `{}` → {}",
                                entry.timestamp, entry.action, entry.outcome
                            ),
                        );
                    }
                    md.build()
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
            Some(audit) => match parse_human_duration(&older_than) {
                Ok(duration) => match audit.prune(duration).await {
                    Ok(n) => format!("Pruned {n} entries older than {older_than}"),
                    Err(e) => format!("Failed to prune audit: {e}"),
                },
                Err(e) => format!(
                    "Couldn't parse duration {older_than:?}: {e}. \
                     Try forms like 24h, 7d, 4w, 1y."
                ),
            },
            None => "Audit trail is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_list_standing_approvals(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.standing_approvals {
            Some(store) => {
                let grants = store.list_active().await.map_err(|e| {
                    SignalError::Processing(format!("Failed to list standing approvals: {e}"))
                })?;
                if grants.is_empty() {
                    "No active standing approvals.".to_string()
                } else {
                    let mut md = crate::render::Markdown::new();
                    md.push_heading(3, "Active standing approvals");
                    for g in grants {
                        let note = g.note.as_deref().unwrap_or("");
                        let suffix = if note.is_empty() {
                            String::new()
                        } else {
                            format!(" — {note}")
                        };
                        md.push_bullet(
                            0,
                            format!(
                                "`{}` — {} for `{}.{}`{}",
                                g.id, g.agent_id, g.verb_ns, g.verb_action, suffix
                            ),
                        );
                    }
                    md.push_line("Revoke with `/approval-revoke <id>`.");
                    md.build()
                }
            }
            None => "Standing-approval store is not wired.".to_string(),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_revoke_standing_approval(
        &self,
        signal_id: Uuid,
        id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.standing_approvals {
            Some(store) => match store.revoke(&id).await {
                Ok(true) => format!("Revoked standing approval `{id}`."),
                Ok(false) => format!("Standing approval `{id}` not found or already revoked."),
                Err(e) => format!("Failed to revoke `{id}`: {e}"),
            },
            None => "Standing-approval store is not wired.".to_string(),
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
                    let mut md = crate::render::Markdown::new();
                    md.push_heading(3, "Pending approvals");
                    for p in pending {
                        md.push_bullet(0, format!("`{}` — {}", p.nonce, p.action_description));
                    }
                    md.push_line("Reply `approve <nonce>` or `reject <nonce>`.");
                    md.build()
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
                    // The user replied after the nonce already settled
                    // (timed_out / approved / rejected / NotFound). This
                    // is almost always benign: the chat client buffered
                    // the keystroke during a previous in-flight signal
                    // and flushed it slightly late. Surfacing
                    // "Approval already resolved" as a Brain: error
                    // just confuses the user, so we swallow it quietly
                    // with no body so the renderer skips it.
                    Err(confirm::ConfirmError::AlreadyResolved(_))
                    | Err(confirm::ConfirmError::NotFound(_)) => {
                        let resp = prepend_nudges(SignalResponse::ok(signal_id, String::new()));
                        return Ok(PipelineResult::Complete(resp));
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
                let mut md = crate::render::Markdown::new();
                md.push_heading(3, "Budget status");
                md.push_bullet(0, "**Hourly consumption**");
                for (k, v) in &status.hourly_consumption {
                    md.push_kv(1, k, v.to_string());
                }
                md.push_bullet(0, "**Daily consumption**");
                for (k, v) in &status.daily_consumption {
                    md.push_kv(1, k, v.to_string());
                }
                md.build()
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
        let intents: Vec<_> = self
            .list_scheduled_intents(None)?
            .into_iter()
            .filter(|i| i.status == "scheduled")
            .collect();
        let message = if intents.is_empty() {
            "No active scheduled intents.".to_string()
        } else {
            let mut md = crate::render::Markdown::new();
            md.push_heading(3, "Scheduled intents");
            for intent in &intents {
                let cadence = intent.cron.as_deref().unwrap_or("one-shot");
                md.push_bullet(
                    0,
                    format!(
                        "`{}` — {} ({}) [{}]",
                        intent.id, intent.description, cadence, intent.namespace
                    ),
                );
            }
            md.push_line("Cancel with `cancel schedule <id>`.");
            md.build()
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_cancel_schedule(
        &self,
        signal_id: Uuid,
        id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let trimmed = id.trim();
        let message = if trimmed.is_empty() {
            "Missing schedule id. Try `cancel schedule <id>`.".to_string()
        } else {
            match self.cancel_scheduled_intent(trimmed)? {
                true => format!("Cancelled schedule `{trimmed}`."),
                false => format!("No active schedule with id `{trimmed}`."),
            }
        };
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
            let mut md = crate::render::Markdown::new();
            md.push_heading(3, "Known agents");
            for line in &matches_line {
                md.push_bullet(0, line);
            }
            if needle.is_empty() && !registered.is_empty() {
                md.push_line(format!("**Ready to delegate**: {}", registered.join(", ")));
            }
            md.build()
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_delegate_task(
        &self,
        signal_id: Uuid,
        agent: String,
        prompt: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
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

        // Auto-expand: if the request names files or directories the
        // daemon can read, attach short excerpts so the decomposer plans
        // against real content instead of guessing. This is the general
        // fix for "user said run CI from .github/workflows/ci.yml and the
        // planner invented `gh run view` because it never saw the file".
        let relevant_facts = collect_path_excerpts(&request);

        // Build decomposition context from config + memory so the
        // decomposer LLM sees the actual sandbox allowlist and won't
        // produce plans that try to call binaries the sandbox refuses.
        let context = orchestrate::DecompositionContext {
            available_tools: self.config.security.exec_allowlist.clone(),
            relevant_facts,
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
                    let mut md = crate::render::Markdown::new();
                    md.push_heading(3, "Recent tasks");
                    for (id, desc, phase) in tasks {
                        md.push_bullet(0, format!("`{id}` — {desc} *({phase:?})*"));
                    }
                    md.build()
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
        until: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        if let Some(window) = until.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let message = format!(
                "Time-bounded proactivity pauses (`for {window}`) aren't supported yet — \
                 v0.4.0 only honours plain `enable nudges` / `disable nudges`. \
                 Re-issue without the duration suffix."
            );
            let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
            return Ok(PipelineResult::Complete(resp));
        }

        let previous = self
            .proactivity_enabled
            .swap(enabled, std::sync::atomic::Ordering::SeqCst);
        let startup_enabled = self.config.proactivity.enabled;
        let message = match (previous, enabled) {
            (false, true) if !startup_enabled => {
                "Proactivity flag set to enabled, but the background habit and \
                 open-loop tasks weren't spawned at startup (config had \
                 `proactivity.enabled: false`). Set it `true` in your config and \
                 restart to actually start generating nudges."
                    .to_string()
            }
            (false, true) => "Proactivity enabled. Nudges resume on the next tick.".to_string(),
            (true, false) => "Proactivity disabled. Background habit / open-loop tasks will \
                 skip generation on the next tick. Set `proactivity.enabled: false` \
                 in your config to keep it off across restarts."
                .to_string(),
            (true, true) => "Proactivity already enabled.".to_string(),
            (false, false) => "Proactivity already disabled.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_proactivity_status(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let runtime = self
            .proactivity_enabled
            .load(std::sync::atomic::Ordering::SeqCst);
        let startup = self.config.proactivity.enabled;
        let runtime_label = if runtime { "enabled" } else { "disabled" };
        let drift_note = if runtime == startup {
            String::new()
        } else {
            format!(
                " (toggled this session; startup config = {})",
                if startup { "enabled" } else { "disabled" }
            )
        };
        let message = format!(
            "Proactivity status:\n  • Runtime toggle: {runtime_label}{drift_note}\n  \
             • Habit engine (startup): {}\n  • Open-loop detector (startup): {}\n  \
             • Quiet hours: {}-{}",
            if startup { "active" } else { "disabled" },
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
                    let mut md = crate::render::Markdown::new();
                    md.push_heading(3, "Registered channels");
                    for c in channels {
                        let health = if c.healthy { "healthy" } else { "down" };
                        md.push_bullet(
                            0,
                            format!("**{}** *({}, {})* — {health}", c.id, c.label, c.kind),
                        );
                    }
                    md.build()
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
                            "The user asked: \"{}\"\n\nResearch material:\n{}\n\n\
                             Answer the user's question grounded in the material above. \
                             The `Linked sources` block (when present) is content fetched \
                             directly from URLs the user pasted — treat it as authoritative \
                             over the generic search hits. Quote page titles and URLs when \
                             you reference them. If the material is silent on the user's \
                             question, say so honestly instead of speculating.",
                            signal.content, result.output
                        );
                        let messages = vec![
                            cortex::llm::Message {
                                role: cortex::llm::Role::System,
                                content: "You are Brain OS. Answer the user's question \
                                          using the supplied research material. Be concise, \
                                          cite sources by URL, and never invent content not \
                                          present in the material."
                                    .to_string(),
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

    /// Publish a `BrainEvent::SignalReceived` to the observability bus if one
    /// is configured. Silent no-op when no observer or no subscribers are
    /// attached — observability must never block the pipeline.
    ///
    /// All string fields are passed through [`observe::Redactor`] first so a
    /// vault-marked secret embedded in `Signal.content` cannot leak onto the
    /// bus.
    pub async fn publish_signal_received(&self, signal: &Signal) {
        let Some(observer) = &self.observer else {
            return;
        };
        const PREVIEW_BYTES: usize = 256;
        let preview = if signal.content.len() <= PREVIEW_BYTES {
            signal.content.clone()
        } else {
            let mut end = PREVIEW_BYTES;
            while end > 0 && !signal.content.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &signal.content[..end])
        };
        let redactor = observe::Redactor::new();
        let scrub = |s: String| -> String {
            let mut v = serde_json::Value::String(s);
            redactor.redact(&mut v);
            v.as_str().map(|s| s.to_string()).unwrap_or_default()
        };
        let ev = observe::BrainEvent::SignalReceived {
            id: signal.id,
            signal: observe::SignalSummary {
                source: format!("{:?}", signal.source).to_lowercase(),
                channel: scrub(signal.channel.clone()),
                sender: scrub(signal.sender.clone()),
                namespace: scrub(signal.namespace.clone()),
                content_preview: scrub(preview),
            },
            ts: chrono::Utc::now(),
        };
        // BusClosed (no subscribers) is informational, not fatal.
        let _ = observer.publish(ev).await;
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

    // ── Identity gate ────────────────────────────────────────────────────

    /// Run the configured `IdentityStore::check` against the classified
    /// intent. Returns `Some(resp)` if the pipeline should short-circuit
    /// with that response, `None` to proceed.
    ///
    /// Skipped when no identity store is wired, when the signal carries no
    /// principal, or when the intent is unguarded (chat/inspection).
    pub(super) async fn enforce_identity(
        &self,
        signal: &Signal,
        signal_id: Uuid,
        intent: &thalamus::Intent,
    ) -> Option<SignalResponse> {
        let store = self.identity_store.as_ref()?;
        let principal = signal.principal.as_ref()?;
        let (req, required) = crate::authz::intent_to_auth(intent)?;

        match store.check(principal, &req, required).await {
            identity::CheckOutcome::Allow => None,
            identity::CheckOutcome::EscalateToUser { reason } => {
                if let Some(observer) = &self.observer {
                    let ev = observe::BrainEvent::ConfirmationRequested {
                        id: signal_id,
                        nonce: signal_id.to_string(),
                        reason: reason.clone(),
                        ts: chrono::Utc::now(),
                    };
                    let _ = observer.publish(ev).await;
                }
                Some(SignalResponse::ok(
                    signal_id,
                    format!(
                        "Approval required: {reason}. Awaiting your decision \
                         (Live tab → cancel/approve, or `respond` from chat)."
                    ),
                ))
            }
            identity::CheckOutcome::Deny { reason } => {
                if let Some(observer) = &self.observer {
                    let ev = observe::BrainEvent::Error {
                        id: signal_id,
                        source: "identity.deny".into(),
                        message: reason.clone(),
                        ts: chrono::Utc::now(),
                    };
                    let _ = observer.publish(ev).await;
                }
                Some(SignalResponse::error(signal_id, reason))
            }
        }
    }

    /// Inline confirmation gate. Runs after the identity check. When a
    /// confirmation engine is wired and the intent lands a tier that
    /// `requires_confirmation`, builds an [`ApprovalSpec`] with the
    /// principal-bound [`GrantKey`] (so [`StandingApprovalStore`] matches
    /// bypass any user prompt) and blocks on `engine.request`.
    ///
    /// Returns `Some(resp)` to short-circuit:
    /// - `ApprovalOutcome::Approved` → `None` (proceed to dispatch)
    /// - `Rejected` / `TimedOut` / `Aborted` → text response explaining
    ///   the outcome
    /// - `engine.request` error → error response
    ///
    /// Skipped when no engine is wired, the intent is unguarded, or the
    /// tier doesn't require confirmation. This is the single inline
    /// checkpoint that enforces the cardinal rule: every action that
    /// reaches a Destructive/External tier passes through the same gate
    /// regardless of provenance (user typing, LLM, reflex firing).
    pub(super) async fn confirmation_gate(
        &self,
        signal: &Signal,
        signal_id: Uuid,
        intent: &thalamus::Intent,
    ) -> Option<SignalResponse> {
        // identity::Tier and brain::security::ActionTier carry
        // identical variants; the converter keeps the cross-crate
        // boundary explicit rather than relying on shared serialization.
        fn convert_tier(t: identity::Tier) -> brain::security::ActionTier {
            use brain::security::ActionTier;
            match t {
                identity::Tier::Read => ActionTier::Read,
                identity::Tier::Write => ActionTier::Write,
                identity::Tier::Execute => ActionTier::Execute,
                identity::Tier::Destructive => ActionTier::Destructive,
                identity::Tier::External => ActionTier::External,
            }
        }

        let engine = self.confirmation_engine.as_ref()?;
        let (req, identity_tier) = crate::authz::intent_to_auth(intent)?;
        let action_tier = convert_tier(identity_tier);
        if !action_tier.requires_confirmation() {
            return None;
        }

        let description = format!("{}.{}", req.verb_ns, req.verb_action);
        let timeout = self
            .confirmation_timeout
            .unwrap_or_else(|| action_tier.default_timeout());
        let mut spec =
            confirm::ApprovalSpec::new(description.clone(), action_tier).with_timeout(timeout);
        if let Some(principal) = &signal.principal {
            spec = spec.with_grant_key(confirm::GrantKey::new(
                principal.agent_id.0.clone(),
                &req.verb_ns,
                &req.verb_action,
            ));
        }
        let nonce = spec.nonce.clone();

        if let Some(observer) = &self.observer {
            let ev = observe::BrainEvent::ConfirmationRequested {
                id: signal_id,
                nonce: nonce.clone(),
                reason: description.clone(),
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }

        let result = engine.request(spec).await;

        if let Some(observer) = &self.observer {
            let decision = match &result {
                Ok(confirm::ApprovalOutcome::Approved) => "approved",
                Ok(confirm::ApprovalOutcome::Rejected { .. }) => "rejected",
                Ok(confirm::ApprovalOutcome::TimedOut) => "timed_out",
                Ok(confirm::ApprovalOutcome::Aborted { .. }) => "aborted",
                Err(_) => "error",
            };
            let ev = observe::BrainEvent::ConfirmationResolved {
                id: signal_id,
                nonce,
                decision: decision.to_string(),
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }

        match result {
            Ok(confirm::ApprovalOutcome::Approved) => None,
            Ok(confirm::ApprovalOutcome::Rejected { reason }) => Some(SignalResponse::error(
                signal_id,
                format!("Approval rejected for `{description}`: {reason}"),
            )),
            Ok(confirm::ApprovalOutcome::TimedOut) => Some(SignalResponse::error(
                signal_id,
                format!("Approval timed out waiting for confirmation on `{description}`"),
            )),
            Ok(confirm::ApprovalOutcome::Aborted { reason }) => Some(SignalResponse::error(
                signal_id,
                format!("Approval aborted for `{description}`: {reason}"),
            )),
            Err(e) => Some(SignalResponse::error(
                signal_id,
                format!("Confirmation engine error on `{description}`: {e}"),
            )),
        }
    }

    // ── Signal cancellation ──────────────────────────────────────────────

    /// Register a cancellation notify for an in-flight signal and return a
    /// handle the pipeline can await. Idempotent — if a notify already exists
    /// for this id (re-entry), the existing one is returned so any pending
    /// cancel still fires on the new pipeline instance.
    pub async fn register_cancel(&self, signal_id: Uuid) -> std::sync::Arc<tokio::sync::Notify> {
        let mut reg = self.cancel_registry.lock().await;
        reg.entry(signal_id)
            .or_insert_with(|| std::sync::Arc::new(tokio::sync::Notify::new()))
            .clone()
    }

    /// Remove the cancellation notify for a signal. Called from `CancelGuard::drop`.
    pub(super) fn unregister_cancel(&self, signal_id: Uuid) {
        // Best-effort: avoid blocking the drop path on the lock. If the lock
        // is held, the entry will be GC'd by the next `register_cancel` call
        // for the same id, or stay live until the process restarts (rare).
        let registry = std::sync::Arc::clone(&self.cancel_registry);
        tokio::spawn(async move {
            registry.lock().await.remove(&signal_id);
        });
    }

    /// Trigger cancellation for an in-flight signal. Returns `true` if a
    /// notify was registered; `false` if the target id is unknown.
    pub async fn cancel_signal(&self, signal_id: Uuid) -> bool {
        let reg = self.cancel_registry.lock().await;
        match reg.get(&signal_id) {
            Some(notify) => {
                notify.notify_waiters();
                true
            }
            None => false,
        }
    }

    /// Build the response for a signal that was cancelled mid-flight.
    /// Also publishes a `BrainEvent::Error { source: "cancelled" }`
    /// correlated to the cancelled signal's id.
    pub(super) async fn cancelled_response(
        &self,
        signal_id: Uuid,
        signal: &Signal,
    ) -> SignalResponse {
        if let Some(observer) = &self.observer {
            let ev = observe::BrainEvent::Error {
                id: signal_id,
                source: "cancelled".into(),
                message: format!("signal {signal_id} cancelled by Intent::CancelSignal"),
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }
        SignalResponse {
            signal_id,
            status: ResponseStatus::Error,
            response: ResponseContent::Text(format!(
                "Signal {} cancelled before completion.",
                signal.id
            )),
            memory_context: MemoryContext::default(),
            session_id: None,
        }
    }

    /// Handle `Intent::OpenTerminalSession`. Requires a wired
    /// [`terminal::TerminalBridge`]; without one, returns a Complete response
    /// explaining the bridge isn't configured. The Signal's `Principal` (if
    /// any) is threaded into the session so audit events and `SessionMeta`
    /// carry it.
    pub(super) async fn handle_open_terminal_session(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        program: String,
        args: Vec<String>,
        cwd: Option<String>,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let Some(bridge) = self.terminal_bridge() else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "Terminal Bridge not configured on this instance.",
            ));
            return Ok(PipelineResult::Complete(resp));
        };
        let request = terminal::pb::OpenRequest {
            program: program.clone(),
            args: args.clone(),
            env: Default::default(),
            cwd: cwd.unwrap_or_default(),
            initial_size: None,
            set_controlling_tty: false,
            client_id: format!("signal:{signal_id}"),
        };
        let svc = bridge.svc();
        let message = match svc
            .open_via_pipeline(request, signal.principal.clone())
            .await
        {
            Ok(handle) => format!(
                "Opened terminal session {} for `{}` ({} args).",
                handle.session_id,
                program,
                args.len(),
            ),
            Err(s) => format!("Failed to open terminal: {}", s.message()),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    /// Handle `Intent::ListTerminalSessions`. Returns a compact human
    /// summary of currently-tracked sessions.
    pub(super) async fn handle_list_terminal_sessions(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let Some(bridge) = self.terminal_bridge() else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "Terminal Bridge not configured on this instance.",
            ));
            return Ok(PipelineResult::Complete(resp));
        };
        let metas = bridge.sessions().list().await;
        let message = if metas.is_empty() {
            "No active terminal sessions.".to_string()
        } else {
            let mut buf = format!("{} active terminal session(s):\n", metas.len());
            for m in &metas {
                use std::fmt::Write;
                let _ = writeln!(
                    buf,
                    "  {} — {} {} (opened {})",
                    m.session_id,
                    m.program,
                    m.args.join(" "),
                    m.opened_at.to_rfc3339(),
                );
            }
            buf
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message.trim_end()));
        Ok(PipelineResult::Complete(resp))
    }

    /// Handle `Intent::CloseTerminalSession`. Forwards to the bridge's
    /// `Close` path and reports the exit code / kill status.
    pub(super) async fn handle_close_terminal_session(
        &self,
        signal_id: Uuid,
        session_id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let Some(bridge) = self.terminal_bridge() else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "Terminal Bridge not configured on this instance.",
            ));
            return Ok(PipelineResult::Complete(resp));
        };
        let svc = bridge.svc();
        let message = match svc.close_via_pipeline(&session_id).await {
            Ok(ack) => format!(
                "Closed terminal session {session_id}: exit_code={}, was_killed={}.",
                ack.exit_code, ack.was_killed,
            ),
            Err(s) => format!(
                "Failed to close terminal session {session_id}: {}",
                s.message()
            ),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    /// Handle `Intent::MountMcpServer`. Builds a [`mcphost::ServerConfig`]
    /// from the slash-form payload and asks the wired host to mount it.
    /// Without a host wired, returns a "not configured" response.
    pub(super) async fn handle_mount_mcp_server(
        &self,
        signal_id: Uuid,
        name: String,
        transport: String,
        command_or_url: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let Some(host) = self.mcp_host() else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "MCP host not configured on this instance.",
            ));
            return Ok(PipelineResult::Complete(resp));
        };
        let cfg = match transport.as_str() {
            "stdio" => {
                let parts: Vec<&str> = command_or_url.split_whitespace().collect();
                let (command, args) = match parts.split_first() {
                    Some((head, rest)) => (
                        (*head).to_string(),
                        rest.iter().map(|s| (*s).to_string()).collect::<Vec<_>>(),
                    ),
                    None => (String::new(), Vec::new()),
                };
                if command.is_empty() {
                    let resp = prepend_nudges(SignalResponse::ok(
                        signal_id,
                        "MCP mount: stdio transport needs a command.",
                    ));
                    return Ok(PipelineResult::Complete(resp));
                }
                mcphost::ServerConfig::Stdio {
                    command,
                    args,
                    env: Default::default(),
                    cwd: None,
                }
            }
            "streamable_http" => mcphost::ServerConfig::StreamableHttp {
                url: command_or_url.clone(),
                oauth: None,
            },
            "http_sse" => mcphost::ServerConfig::HttpSse {
                url: command_or_url.clone(),
                oauth: None,
            },
            other => {
                let resp = prepend_nudges(SignalResponse::ok(
                    signal_id,
                    format!(
                        "MCP mount: unknown transport '{other}' (expected stdio, streamable_http, http_sse).",
                    ),
                ));
                return Ok(PipelineResult::Complete(resp));
            }
        };
        let message = match host.mount(name.clone(), cfg).await {
            Ok(()) => format!("Mounted MCP server '{name}' over {transport}."),
            Err(e) => format!("Failed to mount MCP server '{name}': {e}"),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    /// Handle `Intent::ToolCall`. When an [`intent::IntentRouter`] is
    /// wired, resolve the token to a [`intent::ToolRoute`] and dispatch:
    /// MCP routes flow through `MCPHost::call`; HumanConfirm routes are
    /// rendered as text (until the confirmation engine integration lands);
    /// Terminal / NativeBackend routes are placeholder until the matching
    /// dispatch slice ships. Without a router, returns the deterministic
    /// router-not-configured placeholder.
    pub(super) async fn handle_tool_call(
        &self,
        signal_id: Uuid,
        token: intent::IntentToken,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
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

    /// Handle `Intent::UnmountMcpServer`. Forwards to the wired host.
    pub(super) async fn handle_unmount_mcp_server(
        &self,
        signal_id: Uuid,
        name: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let Some(host) = self.mcp_host() else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "MCP host not configured on this instance.",
            ));
            return Ok(PipelineResult::Complete(resp));
        };
        let message = match host.unmount(&name).await {
            Ok(()) => format!("Unmounted MCP server '{name}'."),
            Err(e) => format!("Failed to unmount MCP server '{name}': {e}"),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    /// Handle `Intent::ListMcpServers`. Renders a compact human summary.
    pub(super) async fn handle_list_mcp_servers(
        &self,
        signal_id: Uuid,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let Some(host) = self.mcp_host() else {
            let resp = prepend_nudges(SignalResponse::ok(
                signal_id,
                "MCP host not configured on this instance.",
            ));
            return Ok(PipelineResult::Complete(resp));
        };
        let servers = host.list_servers().await;
        let message = if servers.is_empty() {
            "No mounted MCP servers.".to_string()
        } else {
            let mut buf = format!("{} mounted MCP server(s):\n", servers.len());
            for s in &servers {
                use std::fmt::Write;
                let _ = writeln!(
                    buf,
                    "  {} — {} tool(s) (mounted {})",
                    s.name,
                    s.tool_count,
                    s.mounted_at.to_rfc3339(),
                );
            }
            buf
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message.trim_end()));
        Ok(PipelineResult::Complete(resp))
    }

    /// Handle the `CancelSignal { signal_id }` intent. Parses the target id,
    /// triggers the notify if present, returns a status response.
    pub(super) async fn handle_cancel_signal(
        &self,
        signal_id: Uuid,
        target_id: String,
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
    ) -> Result<PipelineResult, SignalError> {
        let message = match Uuid::parse_str(&target_id) {
            Err(_) => format!("Invalid signal id: {target_id}"),
            Ok(target) => {
                if self.cancel_signal(target).await {
                    format!("Cancellation requested for signal {target}.")
                } else {
                    format!("No in-flight signal with id {target}.")
                }
            }
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }
}

/// RAII guard that drops a signal's cancel registry entry when the pipeline
/// returns — whether normally, via early-return, or via panic.
pub(super) struct CancelGuard<'a> {
    pub(super) processor: &'a SignalProcessor,
    pub(super) signal_id: Uuid,
}

impl<'a> Drop for CancelGuard<'a> {
    fn drop(&mut self) {
        self.processor.unregister_cancel(self.signal_id);
    }
}

// `DeliveryCategory::parse` (and `FromStr`) live in the channel crate —
// no local normalizer needed here.

/// Project a thalamus::Intent into the observe-crate IntentSummary shape.
/// Kept in the signal crate so we keep observe free of a thalamus dep.
/// Args are best-effort redacted via observe::Redactor.
fn intent_summary_of(intent: &thalamus::Intent) -> observe::IntentSummary {
    use thalamus::Intent;
    let (kind, mut args) = match intent {
        Intent::Chat { content } => ("Chat", serde_json::json!({ "content": content })),
        Intent::StoreFact {
            subject,
            predicate,
            object,
        } => (
            "StoreFact",
            serde_json::json!({ "subject": subject, "predicate": predicate, "object": object }),
        ),
        Intent::Forget { target } => ("Forget", serde_json::json!({ "target": target })),
        Intent::Recall { query } => ("Recall", serde_json::json!({ "query": query })),
        Intent::MemorySummary => ("MemorySummary", serde_json::Value::Null),
        Intent::ExecuteCommand { command, args } => (
            "ExecuteCommand",
            serde_json::json!({ "command": command, "args": args }),
        ),
        Intent::WebSearch { query } => ("WebSearch", serde_json::json!({ "query": query })),
        Intent::SendMessage {
            channel,
            recipient,
            content,
        } => (
            "SendMessage",
            serde_json::json!({ "channel": channel, "recipient": recipient, "content": content }),
        ),
        Intent::ProjectInspect { path, focus } => (
            "ProjectInspect",
            serde_json::json!({ "path": path, "focus": focus }),
        ),
        Intent::DelegateTask { agent, prompt } => (
            "DelegateTask",
            serde_json::json!({ "agent": agent, "prompt": prompt }),
        ),
        Intent::ToolCall(token) => (
            "ToolCall",
            serde_json::json!({
                "verb_ns": token.verb.namespace,
                "verb_action": token.verb.action,
            }),
        ),
        // For control-plane and inspection intents the variant name alone
        // suffices — no payload of interest for observers.
        other => (intent_variant_name(other), serde_json::Value::Null),
    };
    observe::Redactor::new().redact(&mut args);
    observe::IntentSummary {
        kind: kind.to_string(),
        args_redacted: args,
    }
}

/// Fallback variant-name extractor for intents we don't project explicitly.
/// Strips the `Intent::` prefix and any payload from the Debug rendering so
/// the resulting string is a stable enum-style tag.
fn intent_variant_name(intent: &thalamus::Intent) -> &'static str {
    use thalamus::Intent::*;
    match intent {
        Chat { .. } => "Chat",
        StoreFact { .. } => "StoreFact",
        Forget { .. } => "Forget",
        Recall { .. } => "Recall",
        MemorySummary => "MemorySummary",
        ExecuteCommand { .. } => "ExecuteCommand",
        WebSearch { .. } => "WebSearch",
        SendMessage { .. } => "SendMessage",
        ProjectInspect { .. } => "ProjectInspect",
        DelegateTask { .. } => "DelegateTask",
        ToolCall(_) => "ToolCall",
        QueryAudit { .. } => "QueryAudit",
        PruneAudit { .. } => "PruneAudit",
        ListApprovals { .. } => "ListApprovals",
        RespondToApproval { .. } => "RespondToApproval",
        BudgetStatus { .. } => "BudgetStatus",
        Schedule { .. } => "Schedule",
        ListSchedules => "ListSchedules",
        CancelSchedule { .. } => "CancelSchedule",
        SystemStatus => "SystemStatus",
        QueryAgents { .. } => "QueryAgents",
        ListTasks => "ListTasks",
        TaskStatus { .. } => "TaskStatus",
        CancelTask { .. } => "CancelTask",
        CancelSignal { .. } => "CancelSignal",
        SetProactivity { .. } => "SetProactivity",
        ProactivityStatus => "ProactivityStatus",
        DecomposeTask { .. } => "DecomposeTask",
        ListChannels => "ListChannels",
        ChannelPreferences { .. } => "ChannelPreferences",
        SetChannelPreference { .. } => "SetChannelPreference",
        OpenTerminalSession { .. } => "OpenTerminalSession",
        CloseTerminalSession { .. } => "CloseTerminalSession",
        ListTerminalSessions => "ListTerminalSessions",
        MountMcpServer { .. } => "MountMcpServer",
        UnmountMcpServer { .. } => "UnmountMcpServer",
        ListMcpServers => "ListMcpServers",
        ListStandingApprovals => "ListStandingApprovals",
        RevokeStandingApproval { .. } => "RevokeStandingApproval",
    }
}

// ── ProjectInspect helpers ─────────────────────────────────────────────────

/// Resolve `security.allowed_paths` into a list of canonicalized roots.
/// Empty input defaults to `$HOME`. Entries that fail to canonicalize
/// (e.g. typo, missing directory) are dropped with a warning — a
/// misconfigured entry must not silently widen the sandbox.
fn resolve_allowed_roots(configured: &[String]) -> Vec<std::path::PathBuf> {
    let raw: Vec<String> = if configured.is_empty() {
        std::env::var("HOME").into_iter().collect()
    } else {
        configured.to_vec()
    };
    raw.into_iter()
        .filter_map(|entry| {
            let expanded = expand_user_path(&entry);
            match std::fs::canonicalize(&expanded) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(
                        entry = %entry,
                        error = %e,
                        "security.allowed_paths entry could not be canonicalized — ignored"
                    );
                    None
                }
            }
        })
        .collect()
}

/// True when `candidate` is equal to or a descendant of any entry in
/// `roots`. Both sides should already be canonicalized.
fn path_under_any_root(candidate: &std::path::Path, roots: &[std::path::PathBuf]) -> bool {
    roots
        .iter()
        .any(|root| candidate == root.as_path() || candidate.starts_with(root))
}

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

/// Read the first `cap` bytes of a path, returning a string. Routes
/// through the format-aware extractor first so PDFs (and other
/// supported binary formats) come back as real text. Falls back to a
/// raw UTF-8 read for plain text files; non-text binaries return a
/// short "(binary)" stub so the LLM doesn't see garbled bytes.
fn read_truncated(path: &std::path::Path, cap: usize) -> String {
    match crate::extract::read_path_as_text(path, cap) {
        Ok(s) => format!("{s}\n"),
        Err(crate::extract::ExtractError::Io(e)) => {
            format!("(read failed: {})\n", friendly_io_error(&e))
        }
        Err(crate::extract::ExtractError::NotText) => "(binary file — not displayed)\n".to_string(),
        Err(crate::extract::ExtractError::Pdf(why)) => {
            format!("(PDF parse failed: {why})\n")
        }
    }
}

// ── Auto-context expansion for decompose ───────────────────────────────────

/// Maximum number of distinct paths we'll attach to a single decompose
/// request. Caps the prompt size so a request that pastes a dozen files
/// can't blow the LLM context window.
const MAX_DECOMPOSE_PATHS: usize = 4;
/// Per-file content cap when building the decomposer's relevant_facts.
/// Tighter than `read_truncated`'s 12 KB because the decomposer needs a
/// nudge, not a full code-review-quality excerpt.
const DECOMPOSE_FILE_BYTES: usize = 3 * 1024;
/// Bare filenames that are recognised as path tokens even without a
/// directory separator. Common manifests + CI files only — the goal is
/// to surface real grounding, not to scoop arbitrary identifiers.
const BARE_MANIFEST_NAMES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "Gemfile",
    "Makefile",
    "justfile",
    "Justfile",
    "README.md",
    "CHANGELOG.md",
    "ARCHITECTURE.md",
    "Dockerfile",
];

/// Scan a free-form request for path-like tokens. Conservative on
/// purpose — only tokens that are unambiguously paths (absolute,
/// home-relative, explicitly relative, or contain a slash plus a
/// recognisable file extension) qualify. A bare word like `brain` is
/// NOT treated as a path even if it happens to be a directory in cwd.
pub(crate) fn extract_path_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        // Trim wrapping punctuation from each end independently. We only
        // strip a trailing `.` because `.github/workflows/...` is a real
        // path token, while `.../ci.yml.` (sentence terminator) isn't.
        let token = raw.trim_start_matches(['(', '[', '{', '\'', '"', '`']);
        let token = token.trim_end_matches(['.', ')', ']', '}', '\'', '"', '!', '?', '`', ':']);
        if token.is_empty() {
            continue;
        }
        if !is_pathlike(token) {
            continue;
        }
        if !out.iter().any(|p: &String| p == token) {
            out.push(token.to_string());
        }
    }
    out
}

fn is_pathlike(s: &str) -> bool {
    if s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("../")
        || s.starts_with("~/")
        || s == "~"
    {
        return true;
    }
    if BARE_MANIFEST_NAMES.contains(&s) {
        return true;
    }
    if !s.contains('/') {
        return false;
    }
    // Relative path with at least one slash AND the basename has an
    // extension — covers `crates/foo/Cargo.toml`, `.github/workflows/ci.yml`,
    // etc., without falling for prose like `and/or`.
    let basename = s.rsplit('/').next().unwrap_or("");
    if basename.contains('.') {
        return true;
    }
    // Common workflow/config dot-dirs.
    s.starts_with(".github/") || s.starts_with(".vscode/") || s.starts_with(".cargo/")
}

/// Read short excerpts for every path token mentioned in `request`.
/// Each entry becomes one `relevant_facts` line on the decomposer's
/// prompt. Failures (missing path, unreadable, binary) are silently
/// dropped — we don't want a prompt full of "(read failed: ...)".
pub(crate) fn collect_path_excerpts(request: &str) -> Vec<String> {
    let cwd = std::env::current_dir().ok();
    extract_path_tokens(request)
        .into_iter()
        .take(MAX_DECOMPOSE_PATHS)
        .filter_map(|tok| {
            let expanded = expand_user_path(&tok);
            let mut pb = std::path::PathBuf::from(&expanded);
            if pb.is_relative() {
                if let Some(base) = &cwd {
                    pb = base.join(&pb);
                }
            }
            build_decompose_excerpt(&tok, &pb)
        })
        .collect()
}

fn build_decompose_excerpt(token: &str, pb: &std::path::Path) -> Option<String> {
    let meta = std::fs::metadata(pb).ok()?;
    if meta.is_file() {
        // Route through the extractor so PDFs (and any other binary
        // formats we add later) come back as real text, not a refusal
        // that pushes the planner into `grep -a` workarounds.
        match crate::extract::read_path_as_text(pb, DECOMPOSE_FILE_BYTES) {
            Ok(body) => Some(format!("File `{token}`:\n```\n{body}\n```")),
            Err(e) => {
                tracing::debug!(path = %pb.display(), error = %e, "decompose excerpt skipped");
                None
            }
        }
    } else if meta.is_dir() {
        let mut entries: Vec<String> = std::fs::read_dir(pb)
            .ok()?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if SKIP_DIRS.contains(&name.as_str()) {
                    return None;
                }
                let suffix = if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    "/"
                } else {
                    ""
                };
                Some(format!("{name}{suffix}"))
            })
            .collect();
        entries.sort();
        let shown: Vec<String> = entries.iter().take(20).cloned().collect();
        let extra = entries.len().saturating_sub(shown.len());
        let extra_line = if extra > 0 {
            format!(", +{extra} more")
        } else {
            String::new()
        };
        Some(format!(
            "Directory `{token}` ({} entries{extra_line}):\n  {}",
            entries.len(),
            shown.join("\n  ")
        ))
    } else {
        None
    }
}

/// Parse a short human duration like `24h`, `7d`, `4w`, `2y` into a
/// `chrono::Duration`. Used by intents that take a `older_than` field
/// from the user. Trailing whitespace is tolerated; case-insensitive
/// on the unit suffix; the numeric prefix must be a positive integer.
///
/// Supported units (single letter):
/// - `m` minutes (rarely useful for retention, kept for symmetry)
/// - `h` hours
/// - `d` days
/// - `w` weeks (7 days)
/// - `y` years (365 days — non-leap approximation, fine for prune
///   thresholds where ±1 day doesn't matter)
fn parse_human_duration(input: &str) -> Result<chrono::Duration, String> {
    let s = input.trim();
    if s.is_empty() {
        return Err("empty duration".into());
    }
    let bytes = s.as_bytes();
    let unit = bytes[bytes.len() - 1].to_ascii_lowercase() as char;
    if !matches!(unit, 'm' | 'h' | 'd' | 'w' | 'y') {
        return Err(format!("unknown unit {unit:?}"));
    }
    let n_str = &s[..s.len() - 1];
    let n: i64 = n_str
        .parse()
        .map_err(|_| format!("not a non-negative integer: {n_str:?}"))?;
    if n <= 0 {
        return Err(format!("duration must be positive, got {n}"));
    }
    let dur = match unit {
        'm' => chrono::Duration::try_minutes(n),
        'h' => chrono::Duration::try_hours(n),
        'd' => chrono::Duration::try_days(n),
        'w' => chrono::Duration::try_weeks(n),
        'y' => chrono::Duration::try_days(n.saturating_mul(365)),
        _ => unreachable!(),
    };
    dur.ok_or_else(|| format!("duration out of range: {n}{unit}"))
}

#[cfg(test)]
mod duration_parse_tests {
    use super::parse_human_duration;

    #[test]
    fn parses_common_forms() {
        assert_eq!(
            parse_human_duration("24h").unwrap(),
            chrono::Duration::try_hours(24).unwrap()
        );
        assert_eq!(
            parse_human_duration("7d").unwrap(),
            chrono::Duration::try_days(7).unwrap()
        );
        assert_eq!(
            parse_human_duration("4w").unwrap(),
            chrono::Duration::try_weeks(4).unwrap()
        );
        assert_eq!(
            parse_human_duration("1y").unwrap(),
            chrono::Duration::try_days(365).unwrap()
        );
        assert_eq!(
            parse_human_duration("30m").unwrap(),
            chrono::Duration::try_minutes(30).unwrap()
        );
    }

    #[test]
    fn ignores_trailing_whitespace_and_unit_case() {
        assert_eq!(
            parse_human_duration("30D ").unwrap(),
            chrono::Duration::try_days(30).unwrap()
        );
        assert_eq!(
            parse_human_duration("12H").unwrap(),
            chrono::Duration::try_hours(12).unwrap()
        );
    }

    #[test]
    fn rejects_zero_negative_and_garbage() {
        assert!(parse_human_duration("0d").is_err());
        assert!(parse_human_duration("-5d").is_err());
        assert!(parse_human_duration("").is_err());
        assert!(parse_human_duration("30").is_err());
        assert!(parse_human_duration("30x").is_err());
        assert!(parse_human_duration("abc").is_err());
    }
}

#[cfg(test)]
mod project_inspect_path_gate_tests {
    use super::{path_under_any_root, resolve_allowed_roots};
    use std::fs;

    #[test]
    fn empty_config_defaults_to_home() {
        let roots = resolve_allowed_roots(&[]);
        if let Some(home) = std::env::var_os("HOME") {
            let home_canonical = fs::canonicalize(home).expect("HOME must canonicalize");
            assert_eq!(roots, vec![home_canonical]);
        }
    }

    #[test]
    fn rejects_path_outside_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let inside = tmp.path().join("ok");
        fs::create_dir_all(&inside).unwrap();
        let outside_dir = tempfile::tempdir().unwrap();

        let root_canonical = fs::canonicalize(tmp.path()).unwrap();
        let inside_canonical = fs::canonicalize(&inside).unwrap();
        let outside_canonical = fs::canonicalize(outside_dir.path()).unwrap();

        assert!(path_under_any_root(
            &inside_canonical,
            std::slice::from_ref(&root_canonical)
        ));
        assert!(!path_under_any_root(
            &outside_canonical,
            std::slice::from_ref(&root_canonical)
        ));
    }

    #[test]
    fn symlink_escape_is_rejected_via_canonicalization() {
        // sandbox/inner -> /tmp/escape (symlink). resolve_allowed_roots
        // canonicalizes sandbox into the real path; path_under_any_root
        // sees the resolved escape target and refuses.
        let sandbox = tempfile::tempdir().unwrap();
        let escape = tempfile::tempdir().unwrap();
        let link = sandbox.path().join("inner");
        #[cfg(unix)]
        std::os::unix::fs::symlink(escape.path(), &link).unwrap();
        #[cfg(not(unix))]
        {
            // Symlinks on other platforms aren't reliable here — skip.
            return;
        }

        let sandbox_root = fs::canonicalize(sandbox.path()).unwrap();
        let resolved_link = fs::canonicalize(&link).unwrap();
        assert!(
            !path_under_any_root(&resolved_link, &[sandbox_root]),
            "symlink to outside path must not be considered inside the root"
        );
    }

    #[test]
    fn malformed_root_is_dropped_not_widened() {
        // A nonexistent path in `allowed_paths` should be silently
        // dropped, never reinterpreted as "allow everything".
        let roots = resolve_allowed_roots(&["/this/path/definitely/does/not/exist".to_string()]);
        assert!(
            roots.is_empty(),
            "broken entries must drop, not widen the sandbox"
        );
    }
}

#[cfg(test)]
mod list_schedules_tests {
    use super::*;
    use crate::types::SignalResponse;
    use uuid::Uuid;

    async fn make_processor() -> SignalProcessor {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);
        processor
    }

    fn body_of(result: PipelineResult) -> String {
        match result {
            PipelineResult::Complete(resp) => match resp.response {
                crate::types::ResponseContent::Text(t) => t,
                other => panic!("expected Text response, got {other:?}"),
            },
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    #[tokio::test]
    async fn empty_when_no_intents_scheduled() {
        let processor = make_processor().await;
        let result = processor
            .handle_list_schedules(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("No active scheduled intents"),
            "got: {body:?}"
        );
    }

    #[tokio::test]
    async fn renders_persisted_intents_with_id_and_cadence() {
        let processor = make_processor().await;
        let pool = processor.episodic().pool();
        let id_a = pool
            .insert_scheduled_intent("daily standup ping", Some("0 9 * * *"), "work", None)
            .unwrap();
        let id_b = pool
            .insert_scheduled_intent("write release notes", None, "personal", None)
            .unwrap();

        let result = processor
            .handle_list_schedules(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);

        assert!(body.contains("### Scheduled intents"), "got: {body:?}");
        assert!(body.contains(&id_a), "missing id_a in: {body:?}");
        assert!(body.contains("daily standup ping"), "got: {body:?}");
        assert!(body.contains("0 9 * * *"), "got: {body:?}");
        assert!(body.contains(&id_b), "missing id_b in: {body:?}");
        assert!(body.contains("one-shot"), "missing cadence label: {body:?}");
        assert!(
            body.contains("cancel schedule"),
            "missing hint line: {body:?}"
        );
    }

    #[tokio::test]
    async fn cancel_marks_intent_cancelled_and_drops_it_from_list() {
        let processor = make_processor().await;
        let pool = processor.episodic().pool();
        let id = pool
            .insert_scheduled_intent("nightly compact", Some("0 3 * * *"), "system", None)
            .unwrap();

        let result = processor
            .handle_cancel_schedule(Uuid::new_v4(), id.clone(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(body.contains("Cancelled schedule"), "got: {body:?}");
        assert!(body.contains(&id), "got: {body:?}");

        let listed = processor
            .handle_list_schedules(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let listed_body = body_of(listed);
        assert!(
            !listed_body.contains(&id),
            "cancelled id should drop from active list, got: {listed_body:?}"
        );
    }

    #[tokio::test]
    async fn cancel_unknown_id_reports_no_active_schedule() {
        let processor = make_processor().await;
        let result = processor
            .handle_cancel_schedule(
                Uuid::new_v4(),
                "does-not-exist".to_string(),
                &|r: SignalResponse| r,
            )
            .await
            .unwrap();
        let body = body_of(result);
        assert!(body.contains("No active schedule"), "got: {body:?}");
        assert!(body.contains("does-not-exist"), "got: {body:?}");
    }

    #[tokio::test]
    async fn cancel_empty_id_returns_usage_hint() {
        let processor = make_processor().await;
        let result = processor
            .handle_cancel_schedule(Uuid::new_v4(), "   ".to_string(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(body.contains("Missing schedule id"), "got: {body:?}");
    }
}

#[cfg(test)]
mod proactivity_tests {
    use super::*;
    use crate::types::SignalResponse;
    use std::sync::atomic::Ordering;
    use uuid::Uuid;

    async fn make_processor() -> SignalProcessor {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        // Pin proactivity disabled — these tests exercise the disabled→
        // enabled toggle path and must not inherit whichever value the
        // shipped default carries (Issue 36 made the YAML default true).
        config.proactivity.enabled = false;
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);
        processor
    }

    fn body_of(result: PipelineResult) -> String {
        match result {
            PipelineResult::Complete(resp) => match resp.response {
                crate::types::ResponseContent::Text(t) => t,
                other => panic!("expected Text response, got {other:?}"),
            },
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    #[tokio::test]
    async fn toggle_flips_runtime_flag_and_is_visible_in_status() {
        let processor = make_processor().await;
        // Default config has proactivity disabled, so the runtime flag starts false.
        assert!(!processor.proactivity_enabled.load(Ordering::SeqCst));

        let enable = processor
            .handle_set_proactivity(Uuid::new_v4(), true, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(enable);
        // Default config has proactivity disabled at startup, so the response
        // should warn that background tasks weren't spawned.
        assert!(body.contains("weren't spawned at startup"), "got: {body:?}");
        assert!(processor.proactivity_enabled.load(Ordering::SeqCst));

        let status = processor
            .handle_proactivity_status(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(status);
        assert!(
            body.contains("Runtime toggle: enabled"),
            "status missing runtime label: {body:?}"
        );
        assert!(
            body.contains("toggled this session"),
            "status missing drift marker: {body:?}"
        );

        let disable = processor
            .handle_set_proactivity(Uuid::new_v4(), false, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(disable);
        assert!(body.contains("Proactivity disabled"), "got: {body:?}");
        assert!(!processor.proactivity_enabled.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn repeat_toggle_reports_already_state() {
        let processor = make_processor().await;
        // Already disabled by default.
        let result = processor
            .handle_set_proactivity(Uuid::new_v4(), false, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("already disabled"),
            "expected idempotent ack: {body:?}"
        );

        processor.proactivity_enabled.store(true, Ordering::SeqCst);
        let result = processor
            .handle_set_proactivity(Uuid::new_v4(), true, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("already enabled"),
            "expected idempotent ack: {body:?}"
        );
    }

    #[tokio::test]
    async fn enable_when_startup_was_enabled_promises_next_tick() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.proactivity.enabled = true;
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);

        // Flip off then back on to land in the (false, true) branch with
        // startup_enabled = true.
        processor.proactivity_enabled.store(false, Ordering::SeqCst);
        let result = processor
            .handle_set_proactivity(Uuid::new_v4(), true, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("Nudges resume on the next tick"),
            "got: {body:?}"
        );
    }

    #[tokio::test]
    async fn until_window_is_rejected_without_mutating_flag() {
        let processor = make_processor().await;
        let before = processor.proactivity_enabled.load(Ordering::SeqCst);
        let result = processor
            .handle_set_proactivity(
                Uuid::new_v4(),
                false,
                Some("2h".to_string()),
                &|r: SignalResponse| r,
            )
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("aren't supported yet") && body.contains("2h"),
            "got: {body:?}"
        );
        assert_eq!(
            before,
            processor.proactivity_enabled.load(Ordering::SeqCst),
            "rejected request must not flip the flag"
        );
    }
}

#[cfg(test)]
mod path_extraction_tests {
    use super::*;

    #[test]
    fn extracts_absolute_relative_and_workflow_paths() {
        let text = "perform CI from .github/workflows/ci.yml \
                    in /Users/me/proj — also check ./crates/foo/Cargo.toml.";
        let paths = extract_path_tokens(text);
        assert!(paths.contains(&".github/workflows/ci.yml".to_string()));
        assert!(paths.contains(&"/Users/me/proj".to_string()));
        assert!(paths.contains(&"./crates/foo/Cargo.toml".to_string()));
    }

    #[test]
    fn ignores_prose_words_with_slashes() {
        let paths = extract_path_tokens("evaluate true/false logic and/or branches");
        assert!(paths.is_empty(), "got {paths:?}");
    }

    #[test]
    fn picks_up_bare_manifests() {
        let paths = extract_path_tokens("look at Cargo.toml and package.json please");
        assert!(paths.contains(&"Cargo.toml".to_string()));
        assert!(paths.contains(&"package.json".to_string()));
    }

    #[test]
    fn dedupes_repeated_paths() {
        let paths = extract_path_tokens("/a/b /a/b /a/b/c");
        assert_eq!(paths, vec!["/a/b".to_string(), "/a/b/c".to_string()]);
    }

    #[test]
    fn collect_excerpts_returns_real_file_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("snippet.txt");
        std::fs::write(&path, "hello world").expect("write");
        let request = format!("look at {} please", path.display());
        let excerpts = collect_path_excerpts(&request);
        assert_eq!(excerpts.len(), 1);
        assert!(excerpts[0].contains("hello world"));
    }

    #[test]
    fn collect_excerpts_silently_skips_missing_paths() {
        let excerpts = collect_path_excerpts("touch /tmp/does-not-exist-9384234");
        assert!(excerpts.is_empty());
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
