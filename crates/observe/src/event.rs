use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Everything Brain might want to surface to the user, audit log, or remote consumers.
///
/// Summary types are deliberately string-shaped payload bags so this crate
/// doesn't take a hard dependency on higher-level crates (`brainos-identity`
/// for `Principal`, `brainos-intent` for `IntentToken`, etc.). Tighter
/// payload types can be substituted without changing the variant set.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrainEvent {
    SignalReceived {
        id: Uuid,
        signal: SignalSummary,
        ts: DateTime<Utc>,
    },
    IntentClassified {
        id: Uuid,
        intent: IntentSummary,
        confidence: f32,
        ts: DateTime<Utc>,
    },
    /// Reserved for a future dispatch-time emission point. Currently
    /// not produced by any emitter — kept in the enum so consumers can
    /// already handle it and so the serde wire shape is fixed before
    /// the first emitter lands. Wire it from the dispatch layer
    /// (`pipeline/dispatch.rs`) once route selection is on the
    /// observability pillar's roadmap.
    ToolRouteResolved {
        id: Uuid,
        route: ToolRouteSummary,
        ts: DateTime<Utc>,
    },
    ConfirmationRequested {
        id: Uuid,
        nonce: String,
        reason: String,
        ts: DateTime<Utc>,
    },
    ConfirmationResolved {
        id: Uuid,
        nonce: String,
        decision: String,
        ts: DateTime<Utc>,
    },
    /// Reserved for future emission from the MCP / capability backends.
    /// Pairs with [`BrainEvent::ToolCallFinished`] to bracket a single
    /// tool invocation. Not produced yet — keep matching for forward
    /// compat.
    ToolCallStarted {
        id: Uuid,
        tool_id: String,
        args_redacted: serde_json::Value,
        ts: DateTime<Utc>,
    },
    /// See [`BrainEvent::ToolCallStarted`] — reserved, not yet emitted.
    ToolCallFinished {
        id: Uuid,
        tool_id: String,
        outcome: OutcomeSummary,
        duration_ms: u64,
        ts: DateTime<Utc>,
    },
    ReflexFired {
        id: Uuid,
        trigger_id: String,
        payload: serde_json::Value,
        ts: DateTime<Utc>,
    },
    AuditAppended {
        id: Uuid,
        /// The `AuditEntry.id` (UUID-as-string) — matches the existing audit
        /// schema in `crates/audit`. (Spec §8.1 said `audit_row_id: i64`, but
        /// the audit table is keyed by string UUID, not SQLite ROWID.)
        audit_entry_id: String,
        principal: Option<PrincipalSummary>,
        ts: DateTime<Utc>,
    },
    /// Reserved for emission from the budget enforcement layer when a
    /// watermark (warn / hard) is crossed. Not yet wired — the budget
    /// crate currently signals decisions via return values only.
    BudgetCrossed {
        id: Uuid,
        watermark: f32,
        window: String,
        ts: DateTime<Utc>,
    },
    BreakerStateChange {
        id: Uuid,
        tool_id: String,
        from: String,
        to: String,
        ts: DateTime<Utc>,
    },
    Error {
        id: Uuid,
        source: String,
        message: String,
        ts: DateTime<Utc>,
    },
    TerminalSessionOpened {
        id: Uuid,
        session_id: String,
        program: String,
        args: Vec<String>,
        cwd: Option<String>,
        principal: Option<PrincipalSummary>,
        ts: DateTime<Utc>,
    },
    TerminalSessionClosed {
        id: Uuid,
        session_id: String,
        exit_code: i32,
        was_killed: bool,
        principal: Option<PrincipalSummary>,
        ts: DateTime<Utc>,
    },
    /// Orchestrator phase transition. `from` is `"none"` for the initial
    /// entry into `Planning`; subsequent transitions name the prior phase.
    /// The string shape (rather than a typed enum) keeps `observe` free of
    /// an `orchestrate` dep, just like `BreakerStateChange` keeps it free
    /// of `resilience`.
    TaskStateChange {
        id: Uuid,
        task_id: String,
        from: String,
        to: String,
        ts: DateTime<Utc>,
    },
}

impl BrainEvent {
    /// The variant discriminant as a lowercase string. Used by SSE/WS filters
    /// (`?kind=tool_call_started`) without re-serialising the event.
    pub fn kind(&self) -> &'static str {
        match self {
            BrainEvent::SignalReceived { .. } => "signal_received",
            BrainEvent::IntentClassified { .. } => "intent_classified",
            BrainEvent::ToolRouteResolved { .. } => "tool_route_resolved",
            BrainEvent::ConfirmationRequested { .. } => "confirmation_requested",
            BrainEvent::ConfirmationResolved { .. } => "confirmation_resolved",
            BrainEvent::ToolCallStarted { .. } => "tool_call_started",
            BrainEvent::ToolCallFinished { .. } => "tool_call_finished",
            BrainEvent::ReflexFired { .. } => "reflex_fired",
            BrainEvent::AuditAppended { .. } => "audit_appended",
            BrainEvent::BudgetCrossed { .. } => "budget_crossed",
            BrainEvent::BreakerStateChange { .. } => "breaker_state_change",
            BrainEvent::Error { .. } => "error",
            BrainEvent::TerminalSessionOpened { .. } => "terminal_session_opened",
            BrainEvent::TerminalSessionClosed { .. } => "terminal_session_closed",
            BrainEvent::TaskStateChange { .. } => "task_state_change",
        }
    }

    /// The event correlation id. Multiple events sharing this id belong to one signal flow.
    pub fn id(&self) -> Uuid {
        match self {
            BrainEvent::SignalReceived { id, .. }
            | BrainEvent::IntentClassified { id, .. }
            | BrainEvent::ToolRouteResolved { id, .. }
            | BrainEvent::ConfirmationRequested { id, .. }
            | BrainEvent::ConfirmationResolved { id, .. }
            | BrainEvent::ToolCallStarted { id, .. }
            | BrainEvent::ToolCallFinished { id, .. }
            | BrainEvent::ReflexFired { id, .. }
            | BrainEvent::AuditAppended { id, .. }
            | BrainEvent::BudgetCrossed { id, .. }
            | BrainEvent::BreakerStateChange { id, .. }
            | BrainEvent::Error { id, .. }
            | BrainEvent::TerminalSessionOpened { id, .. }
            | BrainEvent::TerminalSessionClosed { id, .. }
            | BrainEvent::TaskStateChange { id, .. } => *id,
        }
    }

    /// Optional `tool_id` filter target; `None` for events not associated with a tool.
    pub fn tool_id(&self) -> Option<&str> {
        match self {
            BrainEvent::ToolCallStarted { tool_id, .. }
            | BrainEvent::ToolCallFinished { tool_id, .. }
            | BrainEvent::BreakerStateChange { tool_id, .. } => Some(tool_id.as_str()),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignalSummary {
    pub source: String,
    pub channel: String,
    pub sender: String,
    pub namespace: String,
    pub content_preview: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IntentSummary {
    pub kind: String,
    pub args_redacted: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolRouteSummary {
    pub tool_id: String,
    pub source: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutcomeSummary {
    pub status: String,
    pub error: Option<String>,
}

/// Loose summary view of `brainos_identity::Principal`. Kept summary-shaped
/// here so a tighter typed payload is a swap, not a variant rename, and so
/// `observe` doesn't depend on `identity`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrincipalSummary {
    pub user_id: String,
    pub agent_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_strings_are_snake_case() {
        let ev = BrainEvent::Error {
            id: Uuid::nil(),
            source: "test".into(),
            message: "m".into(),
            ts: Utc::now(),
        };
        assert_eq!(ev.kind(), "error");
    }

    #[test]
    fn roundtrip_tool_call_started_through_json() {
        let id = Uuid::new_v4();
        let ts = Utc::now();
        let original = BrainEvent::ToolCallStarted {
            id,
            tool_id: "mcp:fs:read".into(),
            args_redacted: serde_json::json!({"path": "/tmp/x"}),
            ts,
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "tool_call_started");
        assert_eq!(decoded.id(), id);
        assert_eq!(decoded.tool_id(), Some("mcp:fs:read"));
    }

    #[test]
    fn id_accessor_returns_per_variant_id() {
        let id = Uuid::new_v4();
        let ts = Utc::now();
        let ev = BrainEvent::BudgetCrossed {
            id,
            watermark: 0.75,
            window: "daily".into(),
            ts,
        };
        assert_eq!(ev.id(), id);
        assert_eq!(ev.tool_id(), None);
    }
}
