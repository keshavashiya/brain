//! Intent → AuthorizationRequest + required Tier mapping (v1.0.0 Phase 1).
//!
//! Each Thalamus `Intent` variant maps to a verb in the dotted-namespace
//! taxonomy from `docs/v1.0.0.md` §12 OQ#1 (`fs.*`, `net.*`, `shell.*`,
//! `memory.*`, etc.) plus the minimum tier the principal must hold.
//! Returning `None` means the intent is unguarded — pure-conversation
//! intents (Chat) and inspection intents that touch nothing destructive
//! (ListChannels, BudgetStatus, SystemStatus).
//!
//! Phase 1 keeps the mapping conservative: when in doubt, classify as
//! `Execute` or higher and let the identity gate escalate to the user.
//! Phase 3's CapabilityIndex routing will refine this; Phase 5's
//! standing-approvals stop the prompts where the user has consented.

use identity::{AuthorizationRequest, Tier};
use thalamus::Intent;

/// Map a classified [`Intent`] to a verb/tier authorization request.
///
/// Returns `None` for intents that don't need a check (pure chat, inspection
/// of internal state that the user can already see in the UI).
///
/// Note: this is a coarse mapping. Path-scope modifiers (`path` / `cwd`)
/// are NOT populated here — they're filled in at the call site once the
/// concrete path is extracted from the intent payload. Phase 3 will fold
/// this into the `IntentRouter::abstract_from_signal` step.
pub fn intent_to_auth(intent: &Intent) -> Option<(AuthorizationRequest, Tier)> {
    match intent {
        // ── Pure conversation — no authorization needed ────────────────
        Intent::Chat { .. }
        | Intent::Recall { .. }
        | Intent::MemorySummary
        | Intent::SystemStatus
        | Intent::ProactivityStatus
        | Intent::ListApprovals { .. }
        | Intent::BudgetStatus { .. }
        | Intent::ListSchedules
        | Intent::ListTasks
        | Intent::TaskStatus { .. }
        | Intent::QueryAgents { .. }
        | Intent::QueryAudit { .. }
        | Intent::ListChannels
        | Intent::ChannelPreferences { .. }
        | Intent::ListTerminalSessions
        | Intent::ListMcpServers => None,

        // ── Memory mutations — Write tier under `memory.*` ─────────────
        Intent::StoreFact { .. } => {
            Some((AuthorizationRequest::new("memory", "store"), Tier::Write))
        }
        Intent::Forget { .. } => Some((
            AuthorizationRequest::new("memory", "delete"),
            Tier::Destructive,
        )),

        // ── Shell execution — Execute tier ─────────────────────────────
        Intent::ExecuteCommand { .. } => {
            Some((AuthorizationRequest::new("shell", "exec"), Tier::Execute))
        }

        // ── Network — External tier (any HTTP egress is External) ─────
        Intent::WebSearch { .. } => {
            Some((AuthorizationRequest::new("net", "http"), Tier::External))
        }
        Intent::SendMessage { .. } => {
            Some((AuthorizationRequest::new("notify", "send"), Tier::External))
        }

        // ── Schedules / tasks (creation) — Write tier ──────────────────
        Intent::Schedule { .. } => {
            Some((AuthorizationRequest::new("schedule", "create"), Tier::Write))
        }
        Intent::CancelSchedule { .. } => {
            Some((AuthorizationRequest::new("schedule", "cancel"), Tier::Write))
        }
        Intent::DecomposeTask { .. } => Some((
            AuthorizationRequest::new("task", "decompose"),
            Tier::Execute,
        )),
        Intent::CancelTask { .. } => {
            Some((AuthorizationRequest::new("task", "cancel"), Tier::Write))
        }
        Intent::CancelSignal { .. } => {
            Some((AuthorizationRequest::new("signal", "cancel"), Tier::Write))
        }

        // ── Agent delegation — Execute (delegate may run code) ────────
        Intent::DelegateTask { .. } => Some((
            AuthorizationRequest::new("agent", "delegate"),
            Tier::Execute,
        )),

        // ── Approval response — Write (state mutation of approval queue) ─
        Intent::RespondToApproval { .. } => Some((
            AuthorizationRequest::new("approval", "respond"),
            Tier::Write,
        )),
        Intent::PruneAudit { .. } => Some((
            AuthorizationRequest::new("audit", "prune"),
            Tier::Destructive,
        )),

        // ── Channel config mutation — Write ────────────────────────────
        Intent::SetChannelPreference { .. } => Some((
            AuthorizationRequest::new("channel", "configure"),
            Tier::Write,
        )),

        // ── Proactivity toggle — Write (modifies behavior) ────────────
        Intent::SetProactivity { .. } => Some((
            AuthorizationRequest::new("proactivity", "configure"),
            Tier::Write,
        )),

        // ── Project inspect — Read (reads files) with path scoping ────
        Intent::ProjectInspect { path, .. } => Some((
            AuthorizationRequest::new("fs", "read")
                .with_modifiers(serde_json::json!({ "path": path })),
            Tier::Read,
        )),

        // ── Terminal Bridge — same Execute tier as shell.exec ──────────
        Intent::OpenTerminalSession { program, cwd, .. } => Some((
            AuthorizationRequest::new("terminal", "open").with_modifiers(serde_json::json!({
                "program": program,
                "cwd": cwd,
            })),
            Tier::Execute,
        )),
        Intent::CloseTerminalSession { session_id } => Some((
            AuthorizationRequest::new("terminal", "close")
                .with_modifiers(serde_json::json!({ "session_id": session_id })),
            Tier::Write,
        )),

        // ── MCP host control — mounting any server is External (HTTP
        // transports egress the network; stdio transports load untrusted
        // tool descriptions into the planning context). Unmount drops state
        // and is a Write. ListMcpServers is unguarded (see above).
        Intent::MountMcpServer {
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
        Intent::UnmountMcpServer { name } => Some((
            AuthorizationRequest::new("mcp", "unmount")
                .with_modifiers(serde_json::json!({ "name": name })),
            Tier::Write,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_is_unguarded() {
        let intent = Intent::Chat {
            content: "hello".into(),
        };
        assert!(intent_to_auth(&intent).is_none());
    }

    #[test]
    fn execute_command_requires_execute_tier() {
        let intent = Intent::ExecuteCommand {
            command: "ls".into(),
            args: vec![],
        };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "shell");
        assert_eq!(req.verb_action, "exec");
        assert_eq!(tier, Tier::Execute);
    }

    #[test]
    fn web_search_is_external() {
        let intent = Intent::WebSearch { query: "x".into() };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "net");
        assert_eq!(tier, Tier::External);
    }

    #[test]
    fn forget_is_destructive() {
        let intent = Intent::Forget { target: "x".into() };
        let (_req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(tier, Tier::Destructive);
    }

    #[test]
    fn project_inspect_carries_path_modifier() {
        let intent = Intent::ProjectInspect {
            path: "/Users/k/proj".into(),
            focus: None,
        };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "fs");
        assert_eq!(req.verb_action, "read");
        assert_eq!(tier, Tier::Read);
        assert_eq!(req.modifier_str("path"), Some("/Users/k/proj"));
    }

    #[test]
    fn recall_and_memory_summary_unguarded() {
        assert!(intent_to_auth(&Intent::Recall { query: "x".into() }).is_none());
        assert!(intent_to_auth(&Intent::MemorySummary).is_none());
    }

    #[test]
    fn mount_mcp_server_is_external_tier() {
        let intent = Intent::MountMcpServer {
            name: "fs".into(),
            transport: "stdio".into(),
            command_or_url: "mcp-fs".into(),
        };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "mcp");
        assert_eq!(req.verb_action, "mount");
        assert_eq!(tier, Tier::External);
        assert_eq!(req.modifier_str("name"), Some("fs"));
        assert_eq!(req.modifier_str("transport"), Some("stdio"));
    }

    #[test]
    fn unmount_mcp_server_is_write_tier() {
        let intent = Intent::UnmountMcpServer { name: "fs".into() };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "mcp");
        assert_eq!(req.verb_action, "unmount");
        assert_eq!(tier, Tier::Write);
    }

    #[test]
    fn list_mcp_servers_unguarded() {
        assert!(intent_to_auth(&Intent::ListMcpServers).is_none());
    }

    #[test]
    fn store_fact_is_memory_write() {
        let intent = Intent::StoreFact {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "memory");
        assert_eq!(req.verb_action, "store");
        assert_eq!(tier, Tier::Write);
    }
}
