//! Single source of truth for the control-plane intent vocabulary.
//!
//! The `Intent` enum, the `PATTERNS` regex table, and the
//! `CLASSIFIER_SYSTEM_PROMPT` historically maintained the same set of verbs
//! in three disconnected forms that drifted: only [`Intent::category`] was
//! compiler-checked. This table is the declarative anchor the other three
//! forms are tested against, so adding an `Intent` variant without updating
//! the vocabulary becomes a compile-or-test failure, not a silent
//! natural-language regression.
//!
//! Pairing: every variant exposes a stable snake_case wire [`Intent::key`]
//! (an exhaustive match — a new variant won't compile until it has a key).
//! Each key gets exactly one [`IntentSpec`] row here, and the drift-guard
//! tests in `tests.rs` assert the enum, the regex coverage, and the
//! classifier prompt all agree with this table.
//!
//! The closed `Intent` enum stays closed by design — this is the kernel's
//! syscall table, deliberately *not* runtime-discoverable. Dynamism belongs
//! to the capability plane (`Intent::ToolCall` → SIT → `CapabilityIndex`),
//! which is out of scope here.

use crate::IntentCategory;

/// How an intent can be reached from user input. Pins the "26 vs 39"
/// coverage gap so each verb's natural-language reachability is a conscious
/// per-verb choice in this table, not an accident of which forms got wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlRouting {
    /// Reachable from free-form natural language via the LLM classifier
    /// (the verb's key appears in `CLASSIFIER_SYSTEM_PROMPT`). Most also
    /// have a deterministic regex fast-path and/or a slash form.
    LlmFallback,
    /// Reachable from natural language **only** via a `PATTERNS` regex —
    /// the LLM classifier prompt does not list it. A narrow, deterministic
    /// surface with no free-text generalization.
    RegexOnly,
    /// Reachable **only** via a `/slash` form (or, for `tool_call`, the
    /// chat tool-loop). Never produced by the classifier from prose.
    SlashOnly,
}

/// One row per `Intent` variant — the declarative source of truth for the
/// control-plane vocabulary. Keyed by the snake_case wire [`crate::Intent::key`].
#[derive(Debug, Clone, Copy)]
pub struct IntentSpec {
    /// snake_case wire name, e.g. `"budget_status"`. Matches
    /// [`crate::Intent::key`] for the corresponding variant.
    pub key: &'static str,
    /// Side-effect class. Must agree with [`crate::Intent::category`].
    pub category: IntentCategory,
    /// Natural-language reachability of this verb.
    pub nl_routable: NlRouting,
    /// One-line rule describing when this intent fires. For `LlmFallback`
    /// verbs this mirrors the rule shown to the classifier; for the others
    /// it documents the deterministic surface that reaches the verb.
    pub blurb: &'static str,
}

/// The vocabulary table. Exactly one row per `Intent` variant; the
/// drift-guard tests assert this is a bijection with the enum's keys.
pub const INTENT_SPECS: &[IntentSpec] = &[
    // ── Inspection ─ read-only state queries ──────────────────────────────
    IntentSpec {
        key: "recall",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Specific memory query naming a concrete topic.",
    },
    IntentSpec {
        key: "memory_summary",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Broad \"dump everything you know about me\" request.",
    },
    IntentSpec {
        key: "system_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Explicit status check like \"/status\".",
    },
    IntentSpec {
        key: "proactivity_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Check proactivity / nudge configuration.",
    },
    IntentSpec {
        key: "budget_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Check LLM usage / token budget.",
    },
    IntentSpec {
        key: "list_approvals",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Show pending confirmations.",
    },
    IntentSpec {
        key: "list_standing_approvals",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Audit pre-granted standing approvals (/approval-list).",
    },
    IntentSpec {
        key: "list_schedules",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "List active background schedules.",
    },
    IntentSpec {
        key: "list_tasks",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "List active or recent multi-step tasks.",
    },
    IntentSpec {
        key: "task_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Status of a specific task by id.",
    },
    IntentSpec {
        key: "query_agents",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Ask which specialist agents are available or why one isn't.",
    },
    IntentSpec {
        key: "query_audit",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Check past actions in the audit trail.",
    },
    IntentSpec {
        key: "list_channels",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::RegexOnly,
        blurb: "List registered channels (\"list channels\", \"what channels\").",
    },
    IntentSpec {
        key: "channel_preferences",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::RegexOnly,
        blurb: "Show learned channel preferences (\"channel preferences for ...\").",
    },
    IntentSpec {
        key: "list_terminal_sessions",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        blurb: "List active terminal sessions (/terminal-list).",
    },
    IntentSpec {
        key: "list_mcp_servers",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        blurb: "List mounted MCP servers (/mcp-list).",
    },
    IntentSpec {
        key: "list_capabilities",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        blurb: "List the live capability manifest (/capabilities, /caps).",
    },
    IntentSpec {
        key: "list_grants",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Unified grants ledger — every standing authority with its provenance and revoke path (/grants).",
    },
    // ── Memory ─ episodic / semantic mutations ─────────────────────────────
    IntentSpec {
        key: "store_fact",
        category: IntentCategory::Memory,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Explicit memory request stating a distilled fact triple.",
    },
    IntentSpec {
        key: "forget",
        category: IntentCategory::Memory,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Forget / delete a stored fact.",
    },
    // ── Action ─ external side effects ─────────────────────────────────────
    IntentSpec {
        key: "execute_command",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Explicit shell command request (\"run ls\", \"execute cargo build\").",
    },
    IntentSpec {
        key: "web_search",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Explicit search for internet / latest / external info.",
    },
    IntentSpec {
        key: "send_message",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Send content via a channel to a recipient.",
    },
    IntentSpec {
        key: "delegate_task",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Single-shot delegation to a named agent (\"delegate to X: ...\").",
    },
    // ── Lifecycle ─ create / cancel of schedules, tasks, sessions, mounts ──
    IntentSpec {
        key: "schedule",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Schedule a new future task (\"remind me in 5 minutes to ...\").",
    },
    IntentSpec {
        key: "cancel_schedule",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Cancel a background schedule by id.",
    },
    IntentSpec {
        key: "decompose_task",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Multi-step request that needs planning and execution.",
    },
    IntentSpec {
        key: "cancel_task",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Cancel a running multi-step task by id.",
    },
    IntentSpec {
        key: "cancel_signal",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Abort an in-flight signal by UUID (distinct from cancel_task).",
    },
    IntentSpec {
        key: "open_terminal_session",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Open a terminal session (/terminal-open).",
    },
    IntentSpec {
        key: "close_terminal_session",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Close a terminal session by id (/terminal-close).",
    },
    IntentSpec {
        key: "mount_mcp_server",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Mount an external MCP server (/mcp-mount).",
    },
    IntentSpec {
        key: "unmount_mcp_server",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Unmount an MCP server by name (/mcp-unmount).",
    },
    IntentSpec {
        key: "reconsent_mcp_server",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Re-approve a quarantined MCP server's changed tool catalog (/mcp-reconsent).",
    },
    // ── Governance ─ approvals, audit, config mutation, proactivity ────────
    IntentSpec {
        key: "approve_memory_writer",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Approve an agent as a memory writer, releasing its quarantined memories (/memory-approve).",
    },
    IntentSpec {
        key: "respond_to_approval",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Approve or reject a pending nonce (\"approve 1234\").",
    },
    IntentSpec {
        key: "revoke_standing_approval",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Revoke a standing approval by id (/approval-revoke).",
    },
    IntentSpec {
        key: "prune_audit",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Delete old audit entries (\"prune audit older than 30 days\").",
    },
    IntentSpec {
        key: "set_channel_preference",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::RegexOnly,
        blurb: "Pin / unpin a channel preference (\"pin <ch> for <category>\").",
    },
    IntentSpec {
        key: "set_proactivity",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Manage nudges / proactivity (\"pause nudges for 2h\").",
    },
    // ── Capability ─ kernel-routed SIT envelope ────────────────────────────
    IntentSpec {
        key: "tool_call",
        category: IntentCategory::Capability,
        nl_routable: NlRouting::SlashOnly,
        blurb: "Capability-plane tool invocation (/tool, chat tool-loop). \
                Never emitted by the classifier from free text.",
    },
    // ── Conversation ─ free-form chat (catch-all) ──────────────────────────
    IntentSpec {
        key: "chat",
        category: IntentCategory::Conversation,
        nl_routable: NlRouting::LlmFallback,
        blurb: "Free-form conversation; the classifier default when uncertain.",
    },
];

/// Look up the [`IntentSpec`] for a wire key, if any.
pub fn spec_for_key(key: &str) -> Option<&'static IntentSpec> {
    INTENT_SPECS.iter().find(|s| s.key == key)
}
