//! `BrainEvent` filter predicate and timestamp accessor used by the
//! `AgentService::brain_events` server-streaming RPC.

use crate::agent_proto::BrainEventsRequest;

/// Predicate matching a `BrainEvent` against the gRPC filter fields.
pub(crate) fn brain_event_matches(ev: &observe::BrainEvent, filter: &BrainEventsRequest) -> bool {
    if !filter.kind.is_empty() && ev.kind() != filter.kind {
        return false;
    }
    if !filter.tool_id.is_empty() && ev.tool_id() != Some(filter.tool_id.as_str()) {
        return false;
    }
    if !filter.principal.is_empty() {
        // Bus events do not yet carry a principal; the filter rejects
        // everything when set.
        return false;
    }
    if !filter.since.is_empty() {
        let Ok(since) = chrono::DateTime::parse_from_rfc3339(&filter.since) else {
            return false;
        };
        let since = since.with_timezone(&chrono::Utc);
        if brain_event_ts(ev) < since {
            return false;
        }
    }
    true
}

pub(crate) fn brain_event_ts(ev: &observe::BrainEvent) -> chrono::DateTime<chrono::Utc> {
    use observe::BrainEvent::*;
    match ev {
        SignalReceived { ts, .. }
        | IntentClassified { ts, .. }
        | ReasoningStep { ts, .. }
        | ToolRouteResolved { ts, .. }
        | ConfirmationRequested { ts, .. }
        | ConfirmationResolved { ts, .. }
        | ToolCallStarted { ts, .. }
        | ToolCallFinished { ts, .. }
        | ReflexFired { ts, .. }
        | AuditAppended { ts, .. }
        | BudgetCrossed { ts, .. }
        | BreakerStateChange { ts, .. }
        | Error { ts, .. }
        | TerminalSessionOpened { ts, .. }
        | TerminalSessionClosed { ts, .. }
        | TaskStateChange { ts, .. } => *ts,
    }
}
