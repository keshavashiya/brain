//! Inspection-category intent handlers: read-only state queries. None
//! of these mutate state; all are unguarded in
//! `signal::authz::intent_to_auth`.
//!
//! Variants: [`thalamus::Intent::Recall`], [`thalamus::Intent::MemorySummary`],
//! [`thalamus::Intent::SystemStatus`], [`thalamus::Intent::ProactivityStatus`],
//! [`thalamus::Intent::BudgetStatus`], [`thalamus::Intent::ListApprovals`],
//! [`thalamus::Intent::ListStandingApprovals`],
//! [`thalamus::Intent::ListSchedules`], [`thalamus::Intent::ListTasks`],
//! [`thalamus::Intent::TaskStatus`], [`thalamus::Intent::QueryAgents`],
//! [`thalamus::Intent::QueryAudit`], [`thalamus::Intent::ListChannels`],
//! [`thalamus::Intent::ChannelPreferences`],
//! [`thalamus::Intent::ListTerminalSessions`],
//! [`thalamus::Intent::ListMcpServers`].

use uuid::Uuid;

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{HandlerContext, InspectionAuth, InspectionHandler, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

/// How far back the memory-summary "recent activity" list reaches. Older
/// episodes are still stored and searchable; they just don't crowd the
/// at-a-glance recent view.
const RECENT_ACTIVITY_WINDOW_DAYS: i64 = 30;

/// True if an episode timestamp falls on/after `cutoff`. Fails open: an
/// unparseable timestamp is treated as in-window rather than silently dropped.
fn episode_within_window(timestamp: &str, cutoff: chrono::DateTime<chrono::Utc>) -> bool {
    match chrono::DateTime::parse_from_rfc3339(timestamp) {
        Ok(ts) => ts.with_timezone(&chrono::Utc) >= cutoff,
        Err(_) => true,
    }
}

/// Render one budget window (hourly or daily) as `provider:resource — used /
/// limit` rows, merging recorded consumption with the configured ceilings so
/// the user sees the envelope, not just usage. When the window has neither
/// usage nor a configured limit, emit a single zero-state line instead of a
/// dangling header with no children (the W6 bug).
fn render_budget_window(
    md: &mut crate::render::Markdown,
    consumption: &std::collections::HashMap<String, u64>,
    limits: &std::collections::HashMap<String, u64>,
) {
    let mut keys: Vec<&String> = consumption.keys().chain(limits.keys()).collect();
    keys.sort();
    keys.dedup();
    if keys.is_empty() {
        md.push_bullet(1, "none this window");
        return;
    }
    for key in keys {
        let used = consumption.get(key).copied().unwrap_or(0);
        let value = match limits.get(key) {
            Some(limit) => format!("{used} / {limit}"),
            None => used.to_string(),
        };
        md.push_kv(1, key, value);
    }
}

impl InspectionAuth for SignalProcessor {
    fn auth_inspection(_intent: &thalamus::Intent) -> Option<(AuthorizationRequest, Tier)> {
        // Read-only state queries. Identity gate is skipped entirely.
        None
    }
}

#[async_trait::async_trait]
impl InspectionHandler for SignalProcessor {
    async fn dispatch_inspection(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::Recall { query } => {
                self.handle_recall(
                    ctx.signal_id,
                    ctx.signal,
                    query,
                    ctx.conversation_history,
                    ctx.procedure_context,
                    prepend_nudges,
                    ctx.progress,
                )
                .await
            }
            thalamus::Intent::MemorySummary => {
                self.handle_memory_summary(
                    ctx.signal_id,
                    ctx.signal,
                    ctx.conversation_history,
                    prepend_nudges,
                )
                .await
            }
            thalamus::Intent::SystemStatus => {
                self.handle_system_status(ctx.signal_id, prepend_nudges)
            }
            thalamus::Intent::ProactivityStatus => {
                self.handle_proactivity_status(ctx.signal_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::BudgetStatus { window } => {
                self.handle_budget_status(ctx.signal_id, window, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListApprovals { status } => {
                self.handle_list_approvals(ctx.signal_id, status, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListStandingApprovals => {
                self.handle_list_standing_approvals(ctx.signal_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListSchedules => {
                self.handle_list_schedules(ctx.signal_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListTasks => {
                self.handle_list_tasks(ctx.signal_id, prepend_nudges).await
            }
            thalamus::Intent::TaskStatus { task_id } => {
                self.handle_task_status(ctx.signal_id, task_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::QueryAgents { filter } => {
                self.handle_query_agents(ctx.signal_id, filter, prepend_nudges)
            }
            thalamus::Intent::QueryAudit {
                filter,
                since,
                limit,
            } => {
                self.handle_query_audit(ctx.signal_id, filter, since, limit, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListChannels => {
                self.handle_list_channels(ctx.signal_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::ChannelPreferences {
                namespace,
                category,
            } => {
                self.handle_channel_preferences(ctx.signal_id, namespace, category, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListTerminalSessions => {
                self.handle_list_terminal_sessions(ctx.signal_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListMcpServers => {
                self.handle_list_mcp_servers(ctx.signal_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::ListCapabilities => {
                self.handle_list_capabilities(ctx.signal_id, prepend_nudges)
                    .await
            }
            other => unreachable!(
                "non-inspection variant routed to dispatch_inspection: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_recall(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        query: String,
        conversation_history: Option<&[cortex::llm::Message]>,
        procedure_context: &[String],
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
        progress: Option<&tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let top_k = self.config.memory.semantic.max_results as usize;
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&query).await;
        let (memories, facts_used, episodes_used) = self
            .do_recall(&query, query_vector, top_k, Some(&signal.namespace))
            .await?;

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
            .map(|step| cortex::llm::Message::user(format!("[procedure step] {step}")))
            .collect();
        let history = conversation_history.unwrap_or(&proc_history);
        // Onboarding mode only when the namespace is truly empty — not just when
        // this query's semantic search returned nothing.
        let addendum = if self.namespace_is_empty(&signal.namespace) {
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

    pub(super) fn handle_system_status(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

        // "Recent" should mean recent — bound the activity list to a window so
        // a long-lived store doesn't surface months-old personal turns as if
        // they just happened. Unparseable timestamps are kept (fail-open).
        let cutoff = chrono::Utc::now() - chrono::Duration::days(RECENT_ACTIVITY_WINDOW_DAYS);
        let recent: Vec<&hippocampus::Episode> = episodes
            .iter()
            .filter(|ep| episode_within_window(&ep.timestamp, cutoff))
            .take(8)
            .collect();
        if !recent.is_empty() {
            md.push_heading(3, "Recent activity");
            for ep in recent {
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

    pub(super) async fn handle_query_audit(
        &self,
        signal_id: Uuid,
        _filter: Option<String>,
        _since: Option<String>,
        _limit: Option<usize>,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.safety.audit_trail {
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

    pub(super) async fn handle_list_standing_approvals(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.safety.standing_approvals {
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

    pub(super) async fn handle_list_approvals(
        &self,
        signal_id: Uuid,
        _status: Option<String>,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.safety.confirmation_engine {
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

    pub(super) async fn handle_budget_status(
        &self,
        signal_id: Uuid,
        _window: Option<String>,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.safety.cost_budget {
            Some(budget) => {
                let status = budget
                    .status()
                    .await
                    .map_err(|e| SignalError::Processing(format!("Budget status failed: {e}")))?;
                let mut md = crate::render::Markdown::new();
                md.push_heading(3, "Budget status");
                md.push_bullet(0, "**Hourly**");
                render_budget_window(&mut md, &status.hourly_consumption, &status.hourly_limits);
                md.push_bullet(0, "**Daily**");
                render_budget_window(&mut md, &status.daily_consumption, &status.daily_limits);
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
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    pub(super) fn handle_query_agents(
        &self,
        signal_id: Uuid,
        filter: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
            matches_line.push(super::observe::format_agent_status(id, status));
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

    pub(super) async fn handle_list_tasks(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    pub(super) async fn handle_proactivity_status(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.channels.channel_router {
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
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let ns = namespace.as_deref().unwrap_or("personal");
        let message = match &self.channels.channel_preferences {
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

    /// Handle `Intent::ListTerminalSessions`. Returns a compact human
    /// summary of currently-tracked sessions.
    pub(super) async fn handle_list_terminal_sessions(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    /// Handle `Intent::ListMcpServers`. Renders a compact human summary.
    pub(super) async fn handle_list_mcp_servers(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    /// Handle `Intent::ListCapabilities`. Renders the live
    /// capability manifest: every tool in the shared registry (native +
    /// terminal + mounted MCP), grouped by source and annotated with its
    /// tier, plus the registered delegate agents. This is the same
    /// registry the SOUL capability digest reads — one manifest, two
    /// consumers.
    pub(super) async fn handle_list_capabilities(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let manifest = self.capability_manifest().await;
        let resp = prepend_nudges(SignalResponse::ok(signal_id, manifest.trim_end()));
        Ok(PipelineResult::Complete(resp))
    }

    /// Render the live capability manifest as human-readable text: every
    /// tool in the shared registry (native + terminal + mounted MCP),
    /// grouped by source and tagged with its tier, plus the registered
    /// delegate agents. The same registry the SOUL capability digest reads
    /// — one manifest, two consumers (the `ListCapabilities` intent and
    /// the outward MCP `brain_capabilities` tool). Untrusted MCP-server
    /// descriptions are sanitized before they reach the output.
    pub async fn capability_manifest(&self) -> String {
        use std::fmt::Write;

        let tools = match self.tool_registry() {
            Some(reg) => reg.list().await,
            None => Vec::new(),
        };
        let agents = self.agent_registry().map(|r| r.list()).unwrap_or_default();

        // Partition by source: native + terminal are first-party
        // (trusted descriptions); MCP tools come from mounted servers
        // (untrusted descriptions — sanitized before display).
        let mut native: Vec<&intent::ToolDescriptor> = Vec::new();
        let mut mcp: Vec<&intent::ToolDescriptor> = Vec::new();
        for t in &tools {
            match t.source {
                intent::ToolSource::McpServer { .. } => mcp.push(t),
                _ => native.push(t),
            }
        }
        native.sort_by(|a, b| a.tool_id.cmp(&b.tool_id));
        mcp.sort_by(|a, b| a.tool_id.cmp(&b.tool_id));

        let tier_of =
            |t: &intent::ToolDescriptor| t.usage.tier.clone().unwrap_or_else(|| "?".to_string());

        let mut buf = format!(
            "Capability manifest — {} tool(s), {} agent(s):\n",
            tools.len(),
            agents.len()
        );

        if !native.is_empty() {
            buf.push_str("\nNative & terminal tools:\n");
            for t in &native {
                let _ = writeln!(
                    buf,
                    "  {} [{}] — {}",
                    t.verb.dotted(),
                    tier_of(t),
                    t.description
                );
                if let Some(when) = &t.usage.when_to_use {
                    let _ = writeln!(buf, "      when: {when}");
                }
            }
        }

        if !mcp.is_empty() {
            buf.push_str("\nMCP tools (mounted servers):\n");
            for t in &mcp {
                // Untrusted server-supplied description — strip control
                // bytes / ANSI before it touches the output.
                let desc = intent::sanitization::sanitize_description_body(&t.description);
                let _ = writeln!(buf, "  {} [{}] — {}", t.tool_id, tier_of(t), desc);
            }
        }

        if !agents.is_empty() {
            let _ = writeln!(
                buf,
                "\nDelegate agents (specialist tasks you can hand off): {}",
                agents.join(", ")
            );
        }

        buf.trim_end().to_string()
    }
}

#[cfg(test)]
mod list_schedules_tests {
    use crate::types::{PipelineResult, SignalResponse};
    use crate::SignalProcessor;
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
mod list_capabilities_tests {
    use crate::types::{PipelineResult, ResponseContent, SignalResponse};
    use crate::SignalProcessor;
    use std::sync::Arc;
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
                ResponseContent::Text(t) => t,
                other => panic!("expected Text response, got {other:?}"),
            },
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    fn descriptor(
        tool_id: &str,
        source: intent::ToolSource,
        verb: intent::Verb,
    ) -> intent::ToolDescriptor {
        intent::ToolDescriptor {
            tool_id: tool_id.to_string(),
            source,
            verb,
            description: "desc".to_string(),
            input_schema: serde_json::json!({}),
            output_schema: None,
            capabilities: vec![],
            annotations: intent::ToolAnnotations::default(),
            usage: intent::ToolUsage::default(),
            embedding: None,
        }
    }

    #[tokio::test]
    async fn renders_native_and_mcp_grouped_with_tier() {
        let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());

        let mut native = descriptor(
            "native:memory.store",
            intent::ToolSource::NativeBackend {
                backend: intent::BackendId::new("memory"),
            },
            intent::Verb::new("memory", "store"),
        );
        native.usage.tier = Some("write".to_string());
        native.usage.when_to_use = Some("State a durable fact.".to_string());
        registry.register(native).await.unwrap();

        // MCP description carries an ANSI escape — must be stripped on display.
        let mut mcp = descriptor(
            "mcp:github:create_issue",
            intent::ToolSource::McpServer {
                server: "github".to_string(),
            },
            intent::Verb::new("mcp", "create_issue"),
        );
        mcp.description = "open an \x1b[31missue\x1b[0m".to_string();
        mcp.usage.tier = Some("external".to_string());
        registry.register(mcp).await.unwrap();

        let processor = make_processor().await.with_tool_registry(registry);
        let result = processor
            .handle_list_capabilities(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);

        assert!(body.contains("2 tool(s)"), "got: {body:?}");
        assert!(body.contains("Native & terminal tools:"), "got: {body:?}");
        assert!(body.contains("memory.store [write]"), "got: {body:?}");
        assert!(
            body.contains("when: State a durable fact."),
            "got: {body:?}"
        );
        assert!(
            body.contains("MCP tools (mounted servers):"),
            "got: {body:?}"
        );
        assert!(
            body.contains("mcp:github:create_issue [external]"),
            "got: {body:?}"
        );
        // ANSI escape stripped from untrusted MCP description.
        assert!(!body.contains('\x1b'), "ANSI not stripped: {body:?}");
    }

    #[tokio::test]
    async fn handles_no_registry() {
        let processor = make_processor().await;
        let result = processor
            .handle_list_capabilities(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(body.contains("0 tool(s), 0 agent(s)"), "got: {body:?}");
    }
}

#[cfg(test)]
mod budget_render_tests {
    use super::render_budget_window;
    use crate::render::Markdown;
    use std::collections::HashMap;

    fn render(consumption: &[(&str, u64)], limits: &[(&str, u64)]) -> String {
        let c: HashMap<String, u64> = consumption
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        let l: HashMap<String, u64> = limits.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let mut md = Markdown::new();
        render_budget_window(&mut md, &c, &l);
        md.build()
    }

    #[test]
    fn empty_window_renders_zero_state_not_dangling_header() {
        let out = render(&[], &[]);
        assert!(out.contains("none this window"), "got: {out:?}");
    }

    #[test]
    fn configured_limit_with_no_usage_shows_zero_over_limit() {
        let out = render(&[], &[("openai:llm_input_tokens", 500_000)]);
        assert!(
            out.contains("openai:llm_input_tokens") && out.contains("0 / 500000"),
            "got: {out:?}"
        );
    }

    #[test]
    fn usage_against_limit_shows_used_over_limit() {
        let out = render(
            &[("openai:llm_input_tokens", 1200)],
            &[("openai:llm_input_tokens", 500_000)],
        );
        assert!(out.contains("1200 / 500000"), "got: {out:?}");
    }

    #[test]
    fn usage_without_a_limit_shows_bare_count() {
        let out = render(&[("local:llm_input_tokens", 42)], &[]);
        assert!(
            out.contains("local:llm_input_tokens") && out.contains("42"),
            "got: {out:?}"
        );
        assert!(
            !out.contains(" / "),
            "should have no limit divider: {out:?}"
        );
    }
}

#[cfg(test)]
mod recent_activity_window_tests {
    use super::episode_within_window;

    #[test]
    fn fresh_episode_is_in_window() {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(30);
        let now = chrono::Utc::now().to_rfc3339();
        assert!(episode_within_window(&now, cutoff));
    }

    #[test]
    fn stale_episode_is_excluded() {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(30);
        let old = (chrono::Utc::now() - chrono::Duration::days(90)).to_rfc3339();
        assert!(!episode_within_window(&old, cutoff));
    }

    #[test]
    fn unparseable_timestamp_fails_open() {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(30);
        assert!(episode_within_window("not-a-timestamp", cutoff));
    }
}
