use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

use super::super::dispatch::HandlerContext;
use super::{episode_within_window, render_budget_window, RECENT_ACTIVITY_WINDOW_DAYS};

/// Render a grant's TTL/scope qualifiers for the listing surfaces
/// (`/approval-list`, `/grants`). Empty for a plain indefinite grant.
pub(super) fn render_grant_constraints(g: &confirm::StandingApproval) -> String {
    let mut parts = Vec::new();
    if let Some(exp) = g.expires_at {
        parts.push(format!("expires {}", exp.format("%Y-%m-%d %H:%M UTC")));
    }
    if let Some(scope) = &g.scope {
        if let Some(p) = &scope.path_prefix {
            parts.push(format!("path `{p}`"));
        }
        if let Some(ns) = &scope.namespace {
            parts.push(format!("namespace `{ns}`"));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" [{}]", parts.join(", "))
    }
}

impl SignalProcessor {
    pub(super) async fn handle_recall(
        &self,
        ctx: &HandlerContext<'_>,
        query: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let &HandlerContext {
            signal_id,
            signal,
            conversation_history,
            procedure_context,
            progress,
            ..
        } = ctx;
        let top_k = self.config.memory.semantic.max_results as usize;
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&query, &signal.namespace).await;
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

        // Residency gate — same rule as chat: a remote-bound prompt must
        // not carry memories from local-only namespaces. (Agent callers
        // above receive them directly; that path stays on the machine.)
        let (memories, withheld) = self.withhold_nonresident_memories(memories, &signal.namespace);
        let (facts_used, episodes_used) = if withheld > 0 {
            let facts = memories
                .iter()
                .filter(|m| m.source == hippocampus::MemorySource::Semantic)
                .count();
            let episodes = memories
                .iter()
                .filter(|m| m.source == hippocampus::MemorySource::Episodic)
                .count();
            (facts, episodes)
        } else {
            (facts_used, episodes_used)
        };

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
            .memory
            .semantic
            .as_ref()
            .and_then(|s| s.count().ok())
            .unwrap_or(0);
        let episode_count = self.memory.episodic.count().unwrap_or(0);

        let mut message = format!("Brain status: {semantic_count} facts, {episode_count} episodes");
        // Residency split — only rendered when a namespace declares a
        // policy, so zero-config installs keep the one-line status.
        if !self.config.memory.namespaces.is_empty() {
            let (mut lo_facts, mut lo_episodes) = (0i64, 0i64);
            let mut lo_names: Vec<String> = Vec::new();
            for s in self.list_namespaces() {
                if self
                    .config
                    .memory
                    .residency_of(&s.namespace)
                    .is_local_only()
                {
                    lo_facts += s.fact_count;
                    lo_episodes += s.episode_count;
                    lo_names.push(s.namespace);
                }
            }
            let chain = if self.llm.is_local() {
                "local (loopback) — local-only memories are available to chat"
            } else {
                "remote — local-only memories are withheld from prompts"
            };
            message.push_str(&format!(
                "\nResidency: {lo_facts} facts and {lo_episodes} episodes stay on this machine \
                 (local-only namespaces: {}). LLM chain: {chain}.",
                if lo_names.is_empty() {
                    self.config.memory.local_only_namespaces().join(", ")
                } else {
                    lo_names.join(", ")
                },
            ));
        }
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
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
                                "`{}` — {} for `{}.{}`{}{}",
                                g.id,
                                g.agent_id,
                                g.verb_ns,
                                g.verb_action,
                                render_grant_constraints(&g),
                                suffix
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
                    md.push_line(
                        "Reply `approve <nonce>` or `reject <nonce>`. Add `for 1h` to also \
                         grant it for an hour, or `here` to grant it within this request's \
                         scope (e.g. `approve <nonce> here for 1h`).",
                    );
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
}
