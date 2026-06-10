use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

impl SignalProcessor {
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
            matches_line.push(crate::pipeline::observe::format_agent_status(id, status));
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

    // pub(in crate::pipeline), not pub(super): the governance test module — a
    // sibling under `pipeline` — drives this handler directly.
    pub(in crate::pipeline) async fn handle_proactivity_status(
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
                let quarantine_note = if s.quarantined {
                    " — ⚠ QUARANTINED: tool catalog changed since approval; \
                     tools disabled until `/mcp-reconsent` or unmount"
                } else {
                    ""
                };
                let _ = writeln!(
                    buf,
                    "  {} — {} tool(s) (mounted {}){quarantine_note}",
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
