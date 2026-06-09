//! Lifecycle-category intent handlers: create / cancel of schedules,
//! tasks, terminal sessions, and MCP server mounts.
//!
//! Variants: [`thalamus::Intent::Schedule`] (routed through
//! `handle_action` for transport, see action.rs),
//! [`thalamus::Intent::CancelSchedule`], [`thalamus::Intent::DecomposeTask`],
//! [`thalamus::Intent::CancelTask`], [`thalamus::Intent::CancelSignal`],
//! [`thalamus::Intent::OpenTerminalSession`],
//! [`thalamus::Intent::CloseTerminalSession`],
//! [`thalamus::Intent::MountMcpServer`],
//! [`thalamus::Intent::UnmountMcpServer`].

use uuid::Uuid;

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{HandlerContext, LifecycleAuth, LifecycleHandler, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

impl LifecycleAuth for SignalProcessor {
    fn auth_lifecycle(intent: &thalamus::Intent) -> Option<(AuthorizationRequest, Tier)> {
        match intent {
            // Issue 126 / W3: creating a schedule kicks off actions that fire
            // later, possibly while the user is away, so it still requires
            // up-front approval — but it is a *reversible* create (undo via
            // CancelSchedule), not an irreversible write. External is the
            // right tier: it gates (`requires_confirmation`) without the
            // misleading "destructive" label or the 5-minute timeout. This
            // must stay in sync with `authz::tier_for_verb("schedule", _)`.
            thalamus::Intent::Schedule { .. } => Some((
                AuthorizationRequest::new("schedule", "create"),
                Tier::External,
            )),
            // The lifecycle Cancel targets (schedule/task/signal) — each its own
            // verb, all Write. The StandingApproval target is Governance-category
            // and is handled in governance.rs, so it falls through to `_ => None`.
            thalamus::Intent::Cancel {
                target: thalamus::CancelTarget::Schedule,
                ..
            } => Some((AuthorizationRequest::new("schedule", "cancel"), Tier::Write)),
            thalamus::Intent::Cancel {
                target: thalamus::CancelTarget::Task,
                ..
            } => Some((AuthorizationRequest::new("task", "cancel"), Tier::Write)),
            thalamus::Intent::Cancel {
                target: thalamus::CancelTarget::Signal,
                ..
            } => Some((AuthorizationRequest::new("signal", "cancel"), Tier::Write)),
            thalamus::Intent::DecomposeTask { .. } => Some((
                AuthorizationRequest::new("task", "decompose"),
                Tier::Execute,
            )),
            thalamus::Intent::OpenTerminalSession { program, cwd, .. } => Some((
                AuthorizationRequest::new("terminal", "open").with_modifiers(serde_json::json!({
                    "program": program,
                    "cwd": cwd,
                })),
                Tier::Execute,
            )),
            thalamus::Intent::CloseTerminalSession { session_id } => Some((
                AuthorizationRequest::new("terminal", "close")
                    .with_modifiers(serde_json::json!({ "session_id": session_id })),
                Tier::Write,
            )),
            // MCP host control: mounting any server is External (HTTP transports
            // egress, stdio transports load untrusted tool descriptions into the
            // planning context). Both Destructive and External tiers route
            // through the confirmation engine, so this satisfies Issue 120's
            // "MCP mount requires human approval" — External is the
            // semantically tighter classification (machine-leaving capability
            // grant) so we keep it rather than downgrading to Destructive.
            // Unmount drops state and is a Write.
            thalamus::Intent::MountMcpServer {
                name,
                transport,
                command_or_url,
            } => Some((
                AuthorizationRequest::new("mcp", "mount").with_modifiers(serde_json::json!({
                    "name": name,
                    "transport": transport,
                    "command_or_url": command_or_url,
                })),
                Tier::External,
            )),
            thalamus::Intent::UnmountMcpServer { name } => Some((
                AuthorizationRequest::new("mcp", "unmount")
                    .with_modifiers(serde_json::json!({ "name": name })),
                Tier::Write,
            )),
            _ => None,
        }
    }
}

#[async_trait::async_trait]
impl LifecycleHandler for SignalProcessor {
    async fn dispatch_lifecycle(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::Cancel {
                target: thalamus::CancelTarget::Schedule,
                id,
            } => {
                self.handle_cancel_schedule(ctx.signal_id, id, prepend_nudges)
                    .await
            }
            thalamus::Intent::DecomposeTask { request } => {
                self.handle_decompose_task(ctx.signal_id, request, prepend_nudges)
                    .await
            }
            thalamus::Intent::Cancel {
                target: thalamus::CancelTarget::Task,
                id,
            } => {
                self.handle_cancel_task(ctx.signal_id, id, prepend_nudges)
                    .await
            }
            thalamus::Intent::Cancel {
                target: thalamus::CancelTarget::Signal,
                id,
            } => {
                self.handle_cancel_signal(ctx.signal_id, id, prepend_nudges)
                    .await
            }
            thalamus::Intent::OpenTerminalSession { program, args, cwd } => {
                self.handle_open_terminal_session(&ctx, program, args, cwd, prepend_nudges)
                    .await
            }
            thalamus::Intent::CloseTerminalSession { session_id } => {
                self.handle_close_terminal_session(ctx.signal_id, session_id, prepend_nudges)
                    .await
            }
            thalamus::Intent::MountMcpServer {
                name,
                transport,
                command_or_url,
            } => {
                self.handle_mount_mcp_server(
                    ctx.signal_id,
                    name,
                    transport,
                    command_or_url,
                    prepend_nudges,
                )
                .await
            }
            thalamus::Intent::UnmountMcpServer { name } => {
                self.handle_unmount_mcp_server(ctx.signal_id, name, prepend_nudges)
                    .await
            }
            // Schedule shares transport with the action umbrella —
            // SignalRouter::intent_to_action(Schedule) → Action::ScheduleTask
            // dispatched through ActionDispatcher. Lifecycle-categorised
            // because it creates persistent state.
            intent @ thalamus::Intent::Schedule { .. } => {
                self.handle_action(ctx.signal_id, ctx.signal, &intent, prepend_nudges)
                    .await
            }
            other => unreachable!(
                "non-lifecycle variant routed to dispatch_lifecycle: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    pub(super) async fn handle_cancel_schedule(
        &self,
        signal_id: Uuid,
        id: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    pub(super) async fn handle_decompose_task(
        &self,
        signal_id: Uuid,
        request: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
        //
        // Off-load to the blocking pool — `collect_path_excerpts` runs a
        // chain of `std::fs::metadata` / `read_dir` / `read` calls per
        // referenced path.
        let relevant_facts = {
            let request_owned = request.clone();
            tokio::task::spawn_blocking(move || super::paths::collect_path_excerpts(&request_owned))
                .await
                .map_err(|e| {
                    SignalError::Processing(format!("decompose excerpt task panicked: {e}"))
                })?
        };

        // Build decomposition context from config + memory + the live
        // capability registries so the decomposer LLM sees (a) the actual
        // sandbox allowlist, (b) the delegate agents it can hand off to,
        // and (c) the faculties wired right now. The planner then composes
        // against real capabilities, and an `implement` step naming an
        // agent that doesn't exist is rejected at plan time.
        let available_agents = self.agent_registry().map(|r| r.list()).unwrap_or_default();
        let available_capabilities = self.planner_capabilities().await;
        let context = orchestrate::DecompositionContext {
            available_tools: self.config.security.exec_allowlist.clone(),
            relevant_facts,
            available_agents,
            available_capabilities,
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

    pub(super) async fn handle_cancel_task(
        &self,
        signal_id: Uuid,
        task_id: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    /// Handle the `CancelSignal { signal_id }` intent. Parses the target id,
    /// triggers the notify if present, returns a status response.
    pub(super) async fn handle_cancel_signal(
        &self,
        signal_id: Uuid,
        target_id: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    /// Handle `Intent::OpenTerminalSession`. Requires a wired
    /// [`terminal::TerminalBridge`]; without one, returns a Complete response
    /// explaining the bridge isn't configured. The Signal's `Principal` (if
    /// any) is threaded into the session so audit events and `SessionMeta`
    /// carry it.
    pub(super) async fn handle_open_terminal_session(
        &self,
        ctx: &HandlerContext<'_>,
        program: String,
        args: Vec<String>,
        cwd: Option<String>,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let &HandlerContext {
            signal_id, signal, ..
        } = ctx;
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

    /// Handle `Intent::CloseTerminalSession`. Forwards to the bridge's
    /// `Close` path and reports the exit code / kill status.
    pub(super) async fn handle_close_terminal_session(
        &self,
        signal_id: Uuid,
        session_id: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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

    /// Handle `Intent::UnmountMcpServer`. Forwards to the wired host.
    pub(super) async fn handle_unmount_mcp_server(
        &self,
        signal_id: Uuid,
        name: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
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
}
