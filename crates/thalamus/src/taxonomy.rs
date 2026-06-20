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
//!
//! ## Usage data, one substrate (mirrors the capability plane)
//!
//! Each row carries an [`IntentUsage`] — the same metacognitive shape the
//! capability plane keeps on `intent::ToolUsage` (`when_to_use` /
//! `when_not_to` plus positive `examples` and negative `counter` phrasings).
//! Two consumers read it: [`crate::build_classifier_system_prompt`] renders
//! the per-intent classifier rules from it (so the prompt can't drift from
//! the table), and [`score_intent`]/[`counter_suppresses`] disambiguate
//! deterministically *before* the LLM tier — the data-driven replacement for
//! the hand-fitted phrasing-regexes (`REACHABILITY_RE`, `REMIND_RECALL_RE`)
//! that used to live in `classifier.rs`. Adding a new disambiguation is a
//! table edit (with a drift-guard test), not a new `LazyLock<Regex>`.

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

/// Per-intent usage guidance — the authored disambiguation data for one verb.
///
/// Deliberately mirrors `intent::ToolUsage` so the control plane and the
/// capability plane share one vocabulary (and, eventually, one renderer).
/// These are `&'static` because they're const table rows, not wire data.
///
/// - `when_to_use` / `when_not_to`: prose shown to the classifier LLM.
/// - `examples`: positive phrasings that *should* route to this intent.
/// - `counter`: negative phrasings that should **not** — the data form of
///   what the retired phrasing-regexes hard-coded. A counter phrase that is
///   fully token-contained in the input suppresses the intent
///   ([`counter_suppresses`]).
#[derive(Debug, Clone, Copy)]
pub struct IntentUsage {
    /// One-line rule describing when this intent fires. For `LlmFallback`
    /// verbs this is rendered into the classifier prompt; for the others it
    /// documents the deterministic surface that reaches the verb.
    pub when_to_use: &'static str,
    /// Optional disambiguation: when a look-alike phrasing should route
    /// elsewhere instead. Rendered into the prompt when present.
    pub when_not_to: Option<&'static str>,
    /// Positive example phrasings (also fed to [`score_intent`]).
    pub examples: &'static [&'static str],
    /// Negative phrasings that must not route here — the data that replaced
    /// the bespoke phrasing-regexes. See [`counter_suppresses`].
    pub counter: &'static [&'static str],
}

impl IntentUsage {
    /// A row with only a one-line description and no example/counter data —
    /// the common case for slash-only / regex-only verbs that don't need
    /// LLM-prompt rules or deterministic phrasing disambiguation.
    pub const fn simple(when_to_use: &'static str) -> Self {
        Self {
            when_to_use,
            when_not_to: None,
            examples: &[],
            counter: &[],
        }
    }
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
    /// Authored usage guidance — prompt rules + deterministic disambiguation.
    pub usage: IntentUsage,
}

/// The vocabulary table. Exactly one row per `Intent` variant; the
/// drift-guard tests assert this is a bijection with the enum's keys.
pub const INTENT_SPECS: &[IntentSpec] = &[
    // ── Inspection ─ read-only state queries ──────────────────────────────
    IntentSpec {
        key: "recall",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Specific memory query that names a concrete topic — the query MUST \
                          identify a topic.",
            when_not_to: None,
            examples: &[
                "what do you know about my project",
                "what did we discuss about Rust",
                "what do you remember about my goals",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "memory_summary",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Broad \"dump everything you know about me\" request; no query \
                          parameter needed.",
            when_not_to: None,
            examples: &[
                "summarise my memory",
                "what do you know",
                "what have you stored",
                "show me my memories",
                "tell me what you remember about me",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "system_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Explicit status check like \"/status\".",
            when_not_to: None,
            examples: &["/status"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "proactivity_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Check proactivity / nudge configuration.",
            when_not_to: None,
            examples: &["check proactivity status", "proactivity status"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "budget_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Check LLM usage / token budget.",
            when_not_to: None,
            examples: &[
                "how much have I spent",
                "what's my token budget",
                "budget status",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "list_approvals",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Show pending confirmations.",
            when_not_to: None,
            examples: &["what am I waiting to approve", "show pending approvals"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "list_standing_approvals",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("Audit pre-granted standing approvals (/approval-list)."),
    },
    IntentSpec {
        key: "list_schedules",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "List active background schedules.",
            when_not_to: None,
            examples: &["what's scheduled", "list schedules", "show schedules"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "list_tasks",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "List active or recent multi-step tasks.",
            when_not_to: None,
            examples: &["what tasks are running", "list tasks", "show tasks"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "task_status",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Status of a specific multi-step task by id.",
            when_not_to: None,
            examples: &["status of task 42", "task status 42"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "query_agents",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Ask which specialist agents are available or why a named one isn't.",
            when_not_to: None,
            examples: &[
                "what agents do you have",
                "which agents can code rust",
                "why aren't you using aider",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "query_audit",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Check past actions in the audit trail.",
            when_not_to: None,
            examples: &[
                "what did I run today",
                "show my audit entries",
                "what did I approve yesterday",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "list_channels",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::RegexOnly,
        usage: IntentUsage::simple(
            "List registered channels (\"list channels\", \"what channels\").",
        ),
    },
    IntentSpec {
        key: "channel_preferences",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::RegexOnly,
        usage: IntentUsage::simple(
            "Show learned channel preferences (\"channel preferences for ...\").",
        ),
    },
    IntentSpec {
        key: "list_terminal_sessions",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("List active terminal sessions (/terminal-list)."),
    },
    IntentSpec {
        key: "list_mcp_servers",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("List mounted MCP servers (/mcp-list)."),
    },
    IntentSpec {
        key: "list_capabilities",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("List the live capability manifest (/capabilities, /caps)."),
    },
    IntentSpec {
        key: "list_grants",
        category: IntentCategory::Inspection,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple(
            "Unified grants ledger — every standing authority with its provenance and revoke \
             path (/grants).",
        ),
    },
    // ── Memory ─ episodic / semantic mutations ─────────────────────────────
    IntentSpec {
        key: "store_fact",
        category: IntentCategory::Memory,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Explicit memory request that states a distilled fact triple. Set \
                          subject/predicate/object to the distilled triple — never copy the \
                          whole sentence into object, and never use \"said\" as the predicate. \
                          If you cannot extract a clean predicate/object pair, classify as chat.",
            when_not_to: Some(
                "A COMPOUND request that asks you to DO something and also remember it (\"run \
                 the build and remember the output\") is chat, NOT store_fact — the assistant \
                 performs the action and stores a distilled result itself.",
            ),
            examples: &[
                "remember that I use Postgres",
                "note that the deploy script is in ops/",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "forget",
        category: IntentCategory::Memory,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Forget / delete a stored fact.",
            when_not_to: None,
            examples: &[
                "forget my old address",
                "delete what you know about project X",
            ],
            counter: &[],
        },
    },
    // ── Action ─ external side effects ─────────────────────────────────────
    IntentSpec {
        key: "execute_command",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Explicit shell command request; the command must be a real shell \
                          binary (ls, git, cargo, …).",
            when_not_to: Some("Questions are NEVER execute_command."),
            examples: &["run ls", "execute cargo build"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "web_search",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Retrieve information or a page's contents from the internet; prefer it \
                          for explicit search / internet / latest / current-info requests. Set \
                          'query' to the exact optimal search terms, stripping conversational \
                          fluff.",
            when_not_to: Some(
                "Testing connectivity or reachability — \"is X reachable\", \"can you reach X\", \
                 \"is the site up/down\", \"ping/trace the route to X\", \"when does the cert for \
                 X expire\" — is NOT web_search: it is chat, where Brain runs a network-diagnostic \
                 capability (net.check/trace/cert).",
            ),
            examples: &[
                "search for rust async book",
                "look up the latest tokio release",
                "google Keshav Ashiya",
                "find information about AI",
            ],
            // The data form of the retired REACHABILITY_RE: a phrasing fully
            // contained in the input steers it to chat (→ net.check) instead.
            counter: &[
                "reachable",
                "unreachable",
                "reach",
                "ping",
                "is up",
                "is down",
                "is online",
                "is offline",
                "can you reach",
                "trace route",
                "cert expire",
                "is the site up",
                "is the site down",
            ],
        },
    },
    IntentSpec {
        key: "send_message",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Send content via a channel to a recipient.",
            when_not_to: None,
            examples: &[
                "send via email to bob: hi",
                "message alice saying running late",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "delegate_task",
        category: IntentCategory::Action,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Explicit single-shot delegation to a named agent. Set `agent` to the \
                          lowercase agent id and `prompt` to the task body. NOT for multi-step \
                          plans — those go to decompose_task and the orchestrator picks an agent.",
            when_not_to: None,
            examples: &[
                "delegate to claude-code: refactor X",
                "ask codex: explain Y",
                "@aider: fix the bug",
            ],
            counter: &[],
        },
    },
    // ── Lifecycle ─ create / cancel of schedules, tasks, sessions, mounts ──
    IntentSpec {
        key: "schedule",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Schedule a new FUTURE task, typically with a time — \"remind me [in \
                          5m] to <action>\", \"set up a daily reminder at 9am to …\", \"schedule \
                          a search every day\". The word \"remind\" alone does not make it a \
                          schedule.",
            when_not_to: Some(
                "\"remind me what/where/who/when …\" is a RECALL question about past facts or the \
                 conversation (answer from memory + history) and must be chat, never schedule. \
                 Only schedule when asked to be reminded TO DO something in the future.",
            ),
            examples: &[
                "remind me to call mom at 5pm",
                "remind me in 5 minutes to check the build",
                "set up a daily reminder at 9am to review my PRs",
                "schedule a backup every night",
            ],
            // The data form of the retired REMIND_RECALL_RE: an interrogative
            // "remind me <wh-word>" is recall, routed to chat, not a schedule.
            counter: &[
                "remind me what",
                "remind me where",
                "remind me who",
                "remind me whom",
                "remind me whose",
                "remind me which",
                "remind me when",
                "remind me why",
                "remind me how",
                "remind me whether",
                "remind me if",
            ],
        },
    },
    IntentSpec {
        key: "cancel_schedule",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Cancel a background schedule by id.",
            when_not_to: None,
            examples: &["cancel schedule 123"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "decompose_task",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Multi-step request that needs planning and execution (multiple steps \
                          or coordination). Simple single-step requests are NOT decompose_task.",
            when_not_to: Some(
                "A message that names a local path (\"summarise /Users/me/notes\", \"read this \
                 file: /tmp/foo.txt\") is chat — Brain reads the path as attached context and \
                 responds conversationally.",
            ),
            examples: &[
                "build a CSV export feature",
                "set up CI/CD pipeline",
                "refactor the auth module and add tests",
                "deploy to production",
            ],
            counter: &[],
        },
    },
    IntentSpec {
        key: "cancel_task",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Cancel a running multi-step task by id.",
            when_not_to: None,
            examples: &["cancel task 10"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "cancel_signal",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Abort an in-flight signal by its UUID — distinct from cancel_task; the \
                          signal_id field carries the UUID.",
            when_not_to: None,
            examples: &["cancel signal e4b8-..."],
            counter: &[],
        },
    },
    IntentSpec {
        key: "open_terminal_session",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("Open a terminal session (/terminal-open)."),
    },
    IntentSpec {
        key: "close_terminal_session",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("Close a terminal session by id (/terminal-close)."),
    },
    IntentSpec {
        key: "mount_mcp_server",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("Mount an external MCP server (/mcp-mount)."),
    },
    IntentSpec {
        key: "unmount_mcp_server",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("Unmount an MCP server by name (/mcp-unmount)."),
    },
    IntentSpec {
        key: "reconsent_mcp_server",
        category: IntentCategory::Lifecycle,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple(
            "Re-approve a quarantined MCP server's changed tool catalog (/mcp-reconsent).",
        ),
    },
    // ── Governance ─ approvals, audit, config mutation, proactivity ────────
    IntentSpec {
        key: "approve_memory_writer",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple(
            "Approve an agent as a memory writer, releasing its quarantined memories \
             (/memory-approve).",
        ),
    },
    IntentSpec {
        key: "respond_to_approval",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Approve or reject a pending nonce.",
            when_not_to: None,
            examples: &["approve 1234", "reject 5678"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "revoke_standing_approval",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple("Revoke a standing approval by id (/approval-revoke)."),
    },
    IntentSpec {
        key: "prune_audit",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Delete old audit entries.",
            when_not_to: None,
            examples: &["prune audit logs older than 30 days"],
            counter: &[],
        },
    },
    IntentSpec {
        key: "set_channel_preference",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::RegexOnly,
        usage: IntentUsage::simple(
            "Pin / unpin a channel preference (\"pin <ch> for <category>\").",
        ),
    },
    IntentSpec {
        key: "set_proactivity",
        category: IntentCategory::Governance,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Manage nudges / proactivity / the habit engine.",
            when_not_to: None,
            examples: &["pause nudges for 2h", "disable proactivity"],
            counter: &[],
        },
    },
    // ── Capability ─ kernel-routed SIT envelope ────────────────────────────
    IntentSpec {
        key: "tool_call",
        category: IntentCategory::Capability,
        nl_routable: NlRouting::SlashOnly,
        usage: IntentUsage::simple(
            "Capability-plane tool invocation (/tool, chat tool-loop). Never emitted by the \
             classifier from free text.",
        ),
    },
    // ── Conversation ─ free-form chat (catch-all) ──────────────────────────
    IntentSpec {
        key: "chat",
        category: IntentCategory::Conversation,
        nl_routable: NlRouting::LlmFallback,
        usage: IntentUsage {
            when_to_use: "Free-form conversation; the classifier default when uncertain. \
                          Conversational meta-questions about the CURRENT chat (\"what did we \
                          discuss?\", \"what did I just say?\", \"summarize our conversation\") \
                          are chat — answer from the live history, not a memory lookup. \
                          General-knowledge, opinion, and how-to questions are chat. \
                          Conversational statements (\"I've done X\", \"I like X\") are chat but \
                          ALSO extract any personal facts.",
            when_not_to: None,
            examples: &[],
            counter: &[],
        },
    },
];

impl IntentSpec {
    /// Render this spec as one line of the classifier prompt's `Rules:` block,
    /// generated from its [`IntentUsage`] — the data-substrate replacement for
    /// the former hand-written `CLASSIFIER_PROMPT_RULES` prose. Mirrors the
    /// capability plane's `advertised_description_body` (which folds the same
    /// `when_to_use` / `when_not_to` guidance into a tool's advertised text);
    /// the eventual single cross-crate `render_usage` is the unifying target.
    ///
    /// Counter phrasings are deliberately *not* rendered — they feed the
    /// deterministic [`score_intent`] / [`counter_suppresses`] tier, while the
    /// human-readable `when_not_to` carries the disambiguation for the LLM.
    pub fn render_prompt_rule(&self) -> String {
        let mut line = format!("- {}: {}", self.key, self.usage.when_to_use);
        if let Some(not) = self.usage.when_not_to {
            line.push_str(" NOT when: ");
            line.push_str(not);
        }
        if !self.usage.examples.is_empty() {
            let quoted = self
                .usage
                .examples
                .iter()
                .map(|e| format!("\"{e}\""))
                .collect::<Vec<_>>()
                .join(", ");
            line.push_str(" e.g. ");
            line.push_str(&quoted);
        }
        line
    }
}

/// Look up the [`IntentSpec`] for a wire key, if any.
pub fn spec_for_key(key: &str) -> Option<&'static IntentSpec> {
    INTENT_SPECS.iter().find(|s| s.key == key)
}

// ─── Deterministic, data-driven disambiguation ──────────────────────────────
//
// One keyword scorer over the table's `examples` / `counter` data replaces the
// hand-fitted phrasing-regexes that used to live in `classifier.rs`. It runs
// before the LLM tier, is zero-LLM, and — crucially — adding a new
// disambiguation is a table edit guarded by a drift test, not a new
// `LazyLock<Regex>` + a new branch in `classify_regex`.

/// Split text into lowercase alphanumeric tokens, dropping empties. Re-exported
/// from [`synapse`] — the one shared lexical primitive both the control plane
/// (this table's scorer) and the capability plane (`mcphost`, `signal`'s
/// tool-loop) tokenize with, so they score phrasings identically.
pub use synapse::tokenize;

/// True when every token of `phrase` appears among `terms` — i.e. the phrasing
/// is fully present in the input. This is the deterministic, host-agnostic test
/// the retired regexes encoded: a counter like `"reachable"` or `"is up"`
/// matches `"is github.com reachable"` / `"is myhost up"` without enumerating
/// hosts, because it keys on the marker tokens, not a literal anchor.
fn phrase_contained(terms: &[String], phrase: &str) -> bool {
    let needles = tokenize(phrase);
    !needles.is_empty() && needles.iter().all(|n| terms.iter().any(|t| t == n))
}

/// Fraction of `phrase`'s tokens present in `terms` (0.0..=1.0). 0.0 for an
/// empty phrase.
fn overlap(terms: &[String], phrase: &str) -> f32 {
    let needles = tokenize(phrase);
    if needles.is_empty() {
        return 0.0;
    }
    let hits = needles.iter().filter(|n| terms.contains(n)).count();
    hits as f32 / needles.len() as f32
}

/// Score `input` against one intent's authored phrasings: the strongest
/// positive example overlap minus the strongest negative counter overlap. A
/// counter-phrasing match suppresses the intent (drives the score negative),
/// which is exactly how the reachability / remind-recall guards steer
/// look-alikes to chat. Mirrors `mcphost::score_top_k` over `IntentSpec` data.
pub fn score_intent(input: &str, spec: &IntentSpec) -> f32 {
    let terms = tokenize(input);
    let pos = spec
        .usage
        .examples
        .iter()
        .map(|e| overlap(&terms, e))
        .fold(0.0_f32, f32::max);
    let neg = spec
        .usage
        .counter
        .iter()
        .map(|c| overlap(&terms, c))
        .fold(0.0_f32, f32::max);
    pos - neg
}

/// True when any of the intent's `counter` phrasings is fully present in
/// `input` — the data-driven replacement for `REACHABILITY_RE` (keyed on
/// `web_search`) and `REMIND_RECALL_RE` (keyed on `schedule`). Unknown keys
/// and keys without counters return `false`.
pub fn counter_suppresses(input: &str, key: &str) -> bool {
    let Some(spec) = spec_for_key(key) else {
        return false;
    };
    if spec.usage.counter.is_empty() {
        return false;
    }
    let terms = tokenize(input);
    spec.usage
        .counter
        .iter()
        .any(|c| phrase_contained(&terms, c))
}
