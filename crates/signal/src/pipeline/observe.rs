//! Observability projections of [`thalamus::Intent`] for the `observe`
//! bus. Lives in the signal crate so the observe crate stays free of any
//! thalamus dependency. All payload fields flow through
//! [`observe::Redactor`] before being published.

/// Format a `delegate::RegistryAgentStatus` as a single human line.
/// Used by `handle_query_agents` to render the known-agents list.
pub(super) fn format_agent_status(id: &str, status: &delegate::RegistryAgentStatus) -> String {
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

/// Project a [`thalamus::Intent`] into the observe-crate
/// [`observe::IntentSummary`] shape. Args are best-effort redacted via
/// [`observe::Redactor`].
pub(super) fn intent_summary_of(intent: &thalamus::Intent) -> observe::IntentSummary {
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
        // Control-plane and inspection intents: the variant name alone
        // suffices — no payload of interest for observers.
        other => (intent_variant_name(other), serde_json::Value::Null),
    };
    observe::Redactor::new().redact(&mut args);
    observe::IntentSummary {
        kind: kind.to_string(),
        args_redacted: args,
    }
}

/// Stable enum-style tag for intents we don't project explicitly. Exhaustive
/// over [`thalamus::Intent`] so a new variant is a compile-error here.
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
        ListCapabilities => "ListCapabilities",
        ListStandingApprovals => "ListStandingApprovals",
        RevokeStandingApproval { .. } => "RevokeStandingApproval",
    }
}
