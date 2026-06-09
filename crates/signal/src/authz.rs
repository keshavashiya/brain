//! Intent → AuthorizationRequest + required Tier mapping.
//!
//! Each Thalamus `Intent` variant maps to a verb in the dotted-namespace
//! taxonomy (`fs.*`, `net.*`, `shell.*`, `memory.*`, etc.) plus the
//! minimum tier the principal must hold. Returning `None` means the
//! intent is unguarded — pure-conversation intents (Chat) and inspection
//! intents that touch nothing destructive (ListChannels, BudgetStatus,
//! SystemStatus).
//!
//! Per-category mappings live alongside their dispatchers in
//! `pipeline/<category>.rs`, behind the `<Category>Auth` traits in
//! `pipeline/dispatch.rs`. This module is the public free-function
//! entry point — call sites stay verb-stable while the actual logic
//! distributes across the seven sibling modules.
//!
//! The mapping is intentionally conservative: when in doubt, classify
//! as `Execute` or higher and let the identity gate escalate to the user.
//! CapabilityIndex routing refines verbs at dispatch time; standing
//! approvals suppress prompts where the user has pre-consented.

use identity::{AuthorizationRequest, Tier};
use thalamus::Intent;

use crate::pipeline::IntentAuthorizer;
use crate::SignalProcessor;

/// Map a classified [`Intent`] to a verb/tier authorization request.
///
/// Returns `None` for intents that don't need a check (pure chat, inspection
/// of internal state that the user can already see in the UI).
///
/// Note: this is a coarse mapping. Path-scope modifiers (`path` / `cwd`)
/// are NOT populated here — they're filled in at the call site once the
/// concrete path is extracted from the intent payload. The intent router
/// folds this into its `abstract_from_signal` step at dispatch time.
///
/// The body dispatches on [`thalamus::Intent::category`] via the
/// [`IntentAuthorizer`] super-trait, which fans out to seven per-category
/// `<Category>Auth` impls colocated with their dispatch impls. Adding a
/// new [`thalamus::IntentCategory`] forces every sub-trait impl site to
/// fail compilation until the new variant is handled.
pub fn intent_to_auth(intent: &Intent) -> Option<(AuthorizationRequest, Tier)> {
    <SignalProcessor as IntentAuthorizer>::intent_to_auth(intent)
}

/// Conservative tier inference from a verb pair. Mirrors the per-variant
/// mappings in `pipeline/<category>.rs` so the typed and abstract paths
/// converge on the same tier once the router lands. Used by
/// [`crate::pipeline::dispatch::CapabilityAuth`] for the `ToolCall` envelope.
pub(crate) fn tier_for_verb(verb_ns: &str, verb_action: &str) -> Tier {
    match (verb_ns, verb_action) {
        ("memory", "delete") | ("audit", "prune") => Tier::Destructive,
        (_, "delete") | (_, "drop") | (_, "destroy") => Tier::Destructive,
        ("net", _) | ("notify", _) => Tier::External,
        ("mcp", "mount") => Tier::External,
        // schedule.create gates up-front but is a reversible create — see the
        // matching rationale in `pipeline/lifecycle.rs` (Issue 126 / W3). Kept
        // here so the typed and abstract paths converge on the same tier.
        // schedule.cancel stays a Write (undo of a create), matching
        // `Intent::CancelSchedule` in lifecycle.rs.
        ("schedule", "create") => Tier::External,
        ("memory", "store")
        | ("memory", "import")
        | ("memory", "export")
        | ("mcp", "unmount")
        | ("schedule", "cancel")
        | ("task", "cancel")
        | ("signal", "cancel")
        | ("approval", _)
        | ("channel", "configure")
        | ("proactivity", "configure")
        | ("terminal", "close") => Tier::Write,
        ("fs", "read") | ("security", "audit") => Tier::Read,
        ("shell", _) | ("terminal", "open") | ("task", "decompose") | ("agent", "delegate") => {
            Tier::Execute
        }
        _ => Tier::Execute,
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
    fn recall_and_memory_summary_unguarded() {
        assert!(intent_to_auth(&Intent::Recall { query: "x".into() }).is_none());
        assert!(intent_to_auth(&Intent::MemorySummary).is_none());
    }

    #[test]
    fn schedule_create_tier_agrees_across_typed_and_abstract_paths() {
        // W3: the typed `Intent::Schedule` path (pipeline/lifecycle.rs) and
        // the abstract verb path (`tier_for_verb`, used by the ToolCall
        // envelope) must resolve schedule.create to the same tier, or a
        // scheduled reminder gets two different approval experiences.
        let (req, typed_tier) = intent_to_auth(&Intent::Schedule {
            description: "review PRs at 9am".into(),
            cron: None,
        })
        .unwrap();
        assert_eq!(req.verb_ns, "schedule");
        assert_eq!(req.verb_action, "create");
        assert_eq!(typed_tier, Tier::External);
        assert_eq!(tier_for_verb("schedule", "create"), typed_tier);
    }

    #[test]
    fn cancel_schedule_tier_agrees_across_typed_and_abstract_paths() {
        // schedule.cancel is the reversible undo of a create — a Write on
        // both paths.
        let (req, typed_tier) = intent_to_auth(&Intent::CancelSchedule {
            id: "review-prs".into(),
        })
        .unwrap();
        assert_eq!(req.verb_ns, "schedule");
        assert_eq!(req.verb_action, "cancel");
        assert_eq!(typed_tier, Tier::Write);
        assert_eq!(tier_for_verb("schedule", "cancel"), typed_tier);
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
    fn list_standing_approvals_is_unguarded() {
        assert!(intent_to_auth(&Intent::ListStandingApprovals).is_none());
    }

    #[test]
    fn revoke_standing_approval_is_write_tier_with_id_modifier() {
        let intent = Intent::RevokeStandingApproval {
            id: "grant-42".into(),
        };
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "approval");
        assert_eq!(req.verb_action, "revoke");
        assert_eq!(tier, Tier::Write);
        assert_eq!(req.modifier_str("id"), Some("grant-42"));
    }

    #[test]
    fn tool_call_destructive_verb_is_destructive_tier() {
        let token = intent::IntentToken::new(
            intent::Verb::new("memory", "delete"),
            intent::Object {
                kind: "intent_args".into(),
                value: serde_json::json!({ "target": "x" }),
            },
            intent::Provenance::User {
                raw_input: "forget x".into(),
                ui_origin: None,
                ts: chrono::Utc::now(),
            },
            "personal".into(),
        );
        let intent = Intent::ToolCall(Box::new(token));
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "memory");
        assert_eq!(req.verb_action, "delete");
        assert_eq!(tier, Tier::Destructive);
    }

    #[test]
    fn tool_call_net_verb_is_external_tier() {
        let token = intent::IntentToken::new(
            intent::Verb::new("net", "http"),
            intent::Object {
                kind: "intent_args".into(),
                value: serde_json::json!({ "query": "rust" }),
            },
            intent::Provenance::Reflex {
                trigger: "cron:hourly".into(),
                raw_input: None,
                ts: chrono::Utc::now(),
            },
            "personal".into(),
        );
        let intent = Intent::ToolCall(Box::new(token));
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "net");
        assert_eq!(tier, Tier::External);
    }

    #[test]
    fn tool_call_fs_read_is_read_tier() {
        let token = intent::IntentToken::new(
            intent::Verb::new("fs", "read"),
            intent::Object {
                kind: "intent_args".into(),
                value: serde_json::json!({ "path": "/etc/hosts" }),
            },
            intent::Provenance::User {
                raw_input: "show /etc/hosts".into(),
                ui_origin: None,
                ts: chrono::Utc::now(),
            },
            "personal".into(),
        );
        let intent = Intent::ToolCall(Box::new(token));
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "fs");
        assert_eq!(req.verb_action, "read");
        assert_eq!(tier, Tier::Read);
        assert_eq!(req.modifier_str("path"), Some("/etc/hosts"));
    }

    #[test]
    fn tool_call_unknown_verb_defaults_to_execute() {
        let token = intent::IntentToken::new(
            intent::Verb::new("custom", "thing"),
            intent::Object {
                kind: "intent_args".into(),
                value: serde_json::Value::Null,
            },
            intent::Provenance::User {
                raw_input: "do the thing".into(),
                ui_origin: None,
                ts: chrono::Utc::now(),
            },
            "personal".into(),
        );
        let intent = Intent::ToolCall(Box::new(token));
        let (_req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(tier, Tier::Execute);
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

    /// Cross-check: every typed-Intent variant that produces an
    /// `AuthorizationRequest` must use a verb that exists in
    /// `intent::verbs::VERBS`, and `tier_for_verb` must return the
    /// same tier as the registry's hint for every entry. Catches
    /// drift between `intent_to_auth` and the kernel vocabulary at
    /// test time — the v0.4.0 substitute for the boot-time
    /// `verbs.toml` registry RFC §158 nominally described.
    #[test]
    fn every_static_verb_is_in_registry() {
        use chrono::Utc;
        use intent::{IntentToken, Object, Provenance, Verb};

        // Build one representative of every authz-mapped Intent variant.
        // Pure-conversation variants (Chat, Recall, …) return `None`
        // from `intent_to_auth` and need no registry entry.
        let typed: Vec<Intent> = vec![
            Intent::StoreFact {
                subject: "s".into(),
                predicate: "p".into(),
                object: "o".into(),
            },
            Intent::Forget { target: "x".into() },
            Intent::ExecuteCommand {
                command: "ls".into(),
                args: vec![],
            },
            Intent::WebSearch { query: "x".into() },
            Intent::SendMessage {
                channel: "ch".into(),
                recipient: "u".into(),
                content: "hi".into(),
            },
            Intent::Schedule {
                description: "d".into(),
                cron: None,
            },
            Intent::CancelSchedule { id: "1".into() },
            Intent::DecomposeTask {
                request: "build".into(),
            },
            Intent::CancelTask {
                task_id: "1".into(),
            },
            Intent::CancelSignal {
                signal_id: "1".into(),
            },
            Intent::DelegateTask {
                agent: "claude-code".into(),
                prompt: "do thing".into(),
            },
            Intent::RespondToApproval {
                nonce: "n".into(),
                decision: "approve".into(),
            },
            Intent::PruneAudit {
                older_than: "30d".into(),
            },
            Intent::SetChannelPreference {
                channel: "ch".into(),
                category: "c".into(),
                weight: 1.0,
                pinned: false,
            },
            Intent::SetProactivity {
                enabled: true,
                until: None,
            },
            Intent::OpenTerminalSession {
                program: "bash".into(),
                args: vec![],
                cwd: None,
            },
            Intent::CloseTerminalSession {
                session_id: "s".into(),
            },
            Intent::MountMcpServer {
                name: "m".into(),
                transport: "stdio".into(),
                command_or_url: "x".into(),
            },
            Intent::UnmountMcpServer { name: "m".into() },
            Intent::RevokeStandingApproval { id: "1".into() },
        ];

        for variant in &typed {
            let (req, _tier) = intent_to_auth(variant)
                .unwrap_or_else(|| panic!("intent {variant:?} produced no AuthorizationRequest"));
            assert!(
                intent::verbs::lookup(&req.verb_ns, &req.verb_action).is_some(),
                "verb {}.{} produced by {:?} is not in intent::verbs::VERBS",
                req.verb_ns,
                req.verb_action,
                variant,
            );
        }

        // Drive `tier_for_verb` through every registered verb and
        // confirm the inferred tier matches the registry's tier_hint.
        // The hint is the source of truth for the kernel vocabulary;
        // `tier_for_verb` is its enforcement.
        for spec in intent::verbs::VERBS {
            let inferred = tier_for_verb(spec.ns, spec.action);
            let expected = match spec.tier_hint {
                intent::verbs::TierHint::Read => Tier::Read,
                intent::verbs::TierHint::Write => Tier::Write,
                intent::verbs::TierHint::Execute => Tier::Execute,
                intent::verbs::TierHint::Destructive => Tier::Destructive,
                intent::verbs::TierHint::External => Tier::External,
            };
            assert_eq!(
                inferred, expected,
                "tier mismatch for {}.{}: tier_for_verb={:?} registry={:?}",
                spec.ns, spec.action, inferred, expected,
            );
        }

        // Probe `Intent::ToolCall` with a registered verb and verify
        // the SIT path resolves cleanly through tier_for_verb.
        let token = IntentToken::new(
            Verb::new("memory", "store"),
            Object {
                kind: "fact".into(),
                value: serde_json::Value::Null,
            },
            Provenance::User {
                raw_input: "test".into(),
                ui_origin: None,
                ts: Utc::now(),
            },
            "personal".into(),
        );
        let intent = Intent::ToolCall(Box::new(token));
        let (req, tier) = intent_to_auth(&intent).unwrap();
        assert_eq!(req.verb_ns, "memory");
        assert_eq!(req.verb_action, "store");
        assert_eq!(tier, Tier::Write);
    }
}
