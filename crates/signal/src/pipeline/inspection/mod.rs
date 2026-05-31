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

use identity::{AuthorizationRequest, Tier};

use super::dispatch::{HandlerContext, InspectionAuth, InspectionHandler, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

mod read_state;
mod resources;

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
