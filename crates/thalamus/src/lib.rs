//! # Brain Thalamus
//!
//! Signal router — first point of contact for all input.
//! Classifies intent using a two-tier approach:
//! 1. Regex fast-path for obvious intents (0ms)
//! 2. LLM fallback for ambiguous input (~300ms)
//!
//! Routes messages to the appropriate subsystem based on intent.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

mod classifier;
mod router;
pub mod taxonomy;

#[cfg(test)]
mod tests;

pub use classifier::IntentClassifier;
pub use intent::{IntentToken, Provenance};
pub use router::SignalRouter;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from the thalamus layer.
#[derive(Debug, Error)]
pub enum ThalamusError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Routing error: {0}")]
    RoutingError(String),
}

// ─── Intent Types ───────────────────────────────────────────────────────────

/// Classified intent for routing.
///
/// Variants are grouped by side-effect class. Every variant maps to exactly
/// one [`IntentCategory`] via [`Intent::category`] (an exhaustive match —
/// new variants get a compile-error until their category is declared).
///
/// The taxonomy mirrors the auth tiers in `signal::authz::intent_to_auth`
/// so trait-dispatch and tier resolution (Issue 112) converge on the
/// same cut of the enum.
///
/// JSON / serde shape is unaffected by the grouping — variant *names* are
/// the wire identifiers and remain unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    // ── Inspection ─ read-only state queries; never require auth ───────────
    /// Recall/search memory.
    Recall { query: String },
    /// Dump and summarise everything stored in memory.
    MemorySummary,
    /// Get system status.
    SystemStatus,
    /// Get proactivity status and configuration.
    ProactivityStatus,
    /// Check LLM budget and usage status.
    BudgetStatus { window: Option<String> },
    /// List a collection of control-plane resources (read-only inspection).
    /// One generic verb over a [`Resource`] replaces the former per-collection
    /// `List*` variants; `filter` narrows the listing where the resource
    /// supports it (e.g. approvals by status). Each `(List, resource)` pair
    /// keeps its own stable wire [`Intent::key`], so the classifier vocabulary
    /// is unchanged.
    List {
        resource: Resource,
        filter: Option<String>,
    },
    /// Get the status of a specific task.
    TaskStatus { task_id: String },
    /// Ask about available specialist agents (delegates). Optional
    /// `filter` narrows the answer: e.g. "rust", "aider", or "".
    QueryAgents { filter: String },
    /// Query the audit trail.
    QueryAudit {
        filter: Option<String>,
        since: Option<String>,
        limit: Option<usize>,
    },
    /// Show learned channel preferences for a (namespace, category).
    /// `category` is one of: confirm, nudge, report, response, alert.
    /// `namespace` defaults to "personal".
    ChannelPreferences {
        namespace: Option<String>,
        category: Option<String>,
    },

    // ── Memory ─ episodic / semantic mutations ─────────────────────────────
    /// Store a fact explicitly.
    StoreFact {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Forget/delete something.
    Forget { target: String },

    // ── Action ─ external side effects (shell / net / delegation) ──────────
    /// Execute a command.
    ExecuteCommand { command: String, args: Vec<String> },
    /// Search the web.
    WebSearch { query: String },
    /// Send via a channel.
    SendMessage {
        channel: String,
        recipient: String,
        content: String,
    },
    /// Run a single-turn delegation to a named specialist agent. Bypasses
    /// task decomposition — used when the user explicitly asks "delegate
    /// to claude-code: ..." or "@codex: ...". For multi-step plans the
    /// orchestrator picks the agent itself via [`Intent::DecomposeTask`].
    DelegateTask { agent: String, prompt: String },

    // ── Lifecycle ─ create / cancel of schedules, tasks, sessions, mounts ─
    /// Schedule something.
    Schedule {
        description: String,
        cron: Option<String>,
    },
    /// Decompose a complex request into an executable task plan.
    DecomposeTask { request: String },
    /// Cancel / revoke a single resource instance by id. One generic verb over
    /// a [`CancelTarget`] replaces the former `CancelSchedule` / `CancelTask` /
    /// `CancelSignal` / `RevokeStandingApproval` variants; each `(Cancel,
    /// target)` pair keeps its own stable wire [`Intent::key`] (and its own
    /// category — schedule/task/signal cancels are Lifecycle, a standing-approval
    /// revoke is Governance — resolved through the taxonomy table).
    Cancel { target: CancelTarget, id: String },
    /// Open a new terminal session via the Terminal Bridge. Returns the
    /// session id so the caller can `Attach` or close it later.
    OpenTerminalSession {
        program: String,
        args: Vec<String>,
        cwd: Option<String>,
    },
    /// Close an active terminal session by id. Kills the child if still running.
    CloseTerminalSession { session_id: String },
    /// Mount an external MCP server (stdio / streamable-http / http-sse) for
    /// tool routing through Brain. The `transport` string is one of
    /// `"stdio"`, `"streamable_http"`, `"http_sse"`. `command_or_url` is the
    /// child-process argv (space-separated) for stdio, or the endpoint URL
    /// for HTTP transports.
    MountMcpServer {
        name: String,
        transport: String,
        command_or_url: String,
    },
    /// Unmount a previously-mounted MCP server by name.
    UnmountMcpServer { name: String },
    /// Re-approve a mounted MCP server's current tool catalog, lifting the
    /// quarantine applied when the catalog changed after mount-time consent.
    ReconsentMcpServer { name: String },

    // ── Governance ─ approvals, audit, config mutation, proactivity ────────
    /// Respond to a pending approval.
    RespondToApproval { nonce: String, decision: String },
    /// Approve an agent as a memory writer: grants a standing
    /// `memory.write` approval (revocable via the standing-approval
    /// revoke path) and releases the agent's quarantined memories.
    ApproveMemoryWriter { agent: String },
    /// Prune the audit trail.
    PruneAudit { older_than: String },
    /// Pin or unpin a channel preference. Pinned weights bypass the
    /// min-weight threshold during routing.
    SetChannelPreference {
        channel: String,
        category: String,
        weight: f32,
        pinned: bool,
    },
    /// Configure proactivity / nudges.
    SetProactivity {
        enabled: bool,
        until: Option<String>,
    },

    // ── Capability ─ kernel-routed Standardized Intent Token envelope ──────
    /// Abstract tool invocation expressed as a Standardized Intent Token.
    /// Emitted by the classifier when the requested action can't be served by
    /// any of the typed variants above and must instead be resolved against
    /// the capability index (MCP tools, native backends, terminal sessions).
    /// The router scores candidates and dispatches the winner; until the
    /// router is wired the pipeline returns a deterministic placeholder.
    /// Boxed so the enum discriminant stays compact — the SIT envelope is
    /// the heaviest variant by far.
    ToolCall(Box<IntentToken>),

    // ── Conversation ─ free-form chat (catch-all) ──────────────────────────
    /// Regular chat/conversation.
    Chat { content: String },
}

/// A listable collection of control-plane resources — the operand of the
/// generic [`Intent::List`] verb. Each arm corresponds to exactly one
/// `list_*` wire key (see [`Intent::key`]); adding a listable resource is one
/// arm here plus its [`taxonomy::INTENT_SPECS`] row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resource {
    /// Pending confirmations (`list_approvals`; `filter` = status).
    Approvals,
    /// Standing approval grants (`list_standing_approvals`).
    StandingApprovals,
    /// Background schedules (`list_schedules`).
    Schedules,
    /// Multi-step orchestrator tasks (`list_tasks`).
    Tasks,
    /// Registered channels (`list_channels`).
    Channels,
    /// Active terminal sessions (`list_terminal_sessions`).
    TerminalSessions,
    /// Mounted MCP servers (`list_mcp_servers`).
    McpServers,
    /// The live capability manifest (`list_capabilities`).
    Capabilities,
    /// The unified grants ledger (`list_grants`): every standing authority
    /// Brain currently holds — runtime grants and config-declared ones —
    /// each with its provenance and revoke path.
    Grants,
}

/// A cancelable single resource instance — the operand of the generic
/// [`Intent::Cancel`] verb. Each arm corresponds to exactly one cancel/revoke
/// wire key (see [`Intent::key`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CancelTarget {
    /// A background schedule (`cancel_schedule`).
    Schedule,
    /// A running multi-step task (`cancel_task`).
    Task,
    /// An in-flight signal pipeline run (`cancel_signal`).
    Signal,
    /// A standing approval grant (`revoke_standing_approval`).
    StandingApproval,
}

/// Side-effect class of an [`Intent`]. Aligns with the auth tiers in
/// `signal::authz::intent_to_auth` so trait-dispatch (Issue 111) and tier
/// resolution (Issue 112) can share a single cut of the enum.
///
/// Every [`Intent`] variant maps to exactly one category via
/// [`Intent::category`]; the match is exhaustive, so adding a new variant
/// without declaring its category is a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentCategory {
    /// Read-only state queries. Never require authorization.
    Inspection,
    /// Mutations of episodic or semantic memory.
    Memory,
    /// External side effects (shell / network / agent delegation).
    Action,
    /// Create / cancel of schedules, tasks, sessions, MCP mounts.
    Lifecycle,
    /// Approvals, audit, channel prefs, proactivity — meta-configuration.
    Governance,
    /// Capability-kernel routed Standardized Intent Token envelope.
    Capability,
    /// Free-form chat (catch-all classification).
    Conversation,
}

impl Intent {
    /// Side-effect class of this intent, resolved through the taxonomy SSOT:
    /// [`Intent::key`] is the compiler-checked variant→key map, and
    /// [`taxonomy::INTENT_SPECS`] owns each key's category. Deriving it here
    /// (rather than a second parallel match) means a verb's category lives in
    /// exactly one place, and a generic verb whose category depends on its
    /// operand (e.g. `Cancel` — Lifecycle for schedule/task/signal, Governance
    /// for a standing-approval revoke) is handled per wire key for free. The
    /// key↔spec bijection is a hard invariant (drift-guard test), so the
    /// fallback is unreachable in practice.
    pub fn category(&self) -> IntentCategory {
        taxonomy::spec_for_key(self.key())
            .map(|s| s.category)
            .unwrap_or(IntentCategory::Conversation)
    }

    /// Stable snake_case wire key for this intent — the identifier shared by
    /// the LLM classifier prompt/parser, the `PATTERNS` regex table, and the
    /// [`taxonomy::INTENT_SPECS`](crate::taxonomy::INTENT_SPECS) source of
    /// truth. Exhaustive over all variants, so adding a new variant without
    /// declaring its key is a compile error — the forcing function that keeps
    /// the vocabulary from drifting. The drift-guard tests assert every key
    /// here has exactly one `IntentSpec` row.
    pub fn key(&self) -> &'static str {
        match self {
            // ── Inspection ────────────────────────────────────────────────
            Intent::Recall { .. } => "recall",
            Intent::MemorySummary => "memory_summary",
            Intent::SystemStatus => "system_status",
            Intent::ProactivityStatus => "proactivity_status",
            Intent::BudgetStatus { .. } => "budget_status",
            // Generic List over a Resource — each collection keeps its own key.
            Intent::List { resource, .. } => match resource {
                Resource::Approvals => "list_approvals",
                Resource::StandingApprovals => "list_standing_approvals",
                Resource::Schedules => "list_schedules",
                Resource::Tasks => "list_tasks",
                Resource::Channels => "list_channels",
                Resource::TerminalSessions => "list_terminal_sessions",
                Resource::McpServers => "list_mcp_servers",
                Resource::Capabilities => "list_capabilities",
                Resource::Grants => "list_grants",
            },
            Intent::TaskStatus { .. } => "task_status",
            Intent::QueryAgents { .. } => "query_agents",
            Intent::QueryAudit { .. } => "query_audit",
            Intent::ChannelPreferences { .. } => "channel_preferences",
            // ── Memory ────────────────────────────────────────────────────
            Intent::StoreFact { .. } => "store_fact",
            Intent::Forget { .. } => "forget",
            // ── Action ────────────────────────────────────────────────────
            Intent::ExecuteCommand { .. } => "execute_command",
            Intent::WebSearch { .. } => "web_search",
            Intent::SendMessage { .. } => "send_message",
            Intent::DelegateTask { .. } => "delegate_task",
            // ── Lifecycle ─────────────────────────────────────────────────
            Intent::Schedule { .. } => "schedule",
            Intent::DecomposeTask { .. } => "decompose_task",
            // Generic Cancel over a CancelTarget — each target keeps its own key
            // (and, via the taxonomy table, its own category).
            Intent::Cancel { target, .. } => match target {
                CancelTarget::Schedule => "cancel_schedule",
                CancelTarget::Task => "cancel_task",
                CancelTarget::Signal => "cancel_signal",
                CancelTarget::StandingApproval => "revoke_standing_approval",
            },
            Intent::OpenTerminalSession { .. } => "open_terminal_session",
            Intent::CloseTerminalSession { .. } => "close_terminal_session",
            Intent::MountMcpServer { .. } => "mount_mcp_server",
            Intent::UnmountMcpServer { .. } => "unmount_mcp_server",
            Intent::ReconsentMcpServer { .. } => "reconsent_mcp_server",
            // ── Governance ────────────────────────────────────────────────
            Intent::RespondToApproval { .. } => "respond_to_approval",
            Intent::ApproveMemoryWriter { .. } => "approve_memory_writer",
            Intent::PruneAudit { .. } => "prune_audit",
            Intent::SetChannelPreference { .. } => "set_channel_preference",
            Intent::SetProactivity { .. } => "set_proactivity",
            // ── Capability ────────────────────────────────────────────────
            Intent::ToolCall(_) => "tool_call",
            // ── Conversation ──────────────────────────────────────────────
            Intent::Chat { .. } => "chat",
        }
    }

    /// Convert a typed `Intent` into a Standardized Intent Token. Returns
    /// `None` for purely conversational variants (`Chat`, inspection
    /// variants) that don't carry a capability claim. Used when re-routing
    /// the existing taxonomy through the capability kernel — the same
    /// envelope a fresh `ToolCall` carries.
    pub fn to_intent_token(
        &self,
        provenance: Provenance,
        namespace: impl Into<String>,
    ) -> Option<IntentToken> {
        use intent::{Object, Verb};
        let namespace = namespace.into();
        let (verb, object_value, caps) = match self {
            Intent::ToolCall(token) => return Some(*token.clone()),
            Intent::StoreFact {
                subject,
                predicate,
                object,
            } => (
                Verb::new("memory", "store"),
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                }),
                vec!["memory.store".to_string()],
            ),
            Intent::Forget { target } => (
                Verb::new("memory", "delete"),
                serde_json::json!({ "target": target }),
                vec!["memory.delete".to_string()],
            ),
            Intent::ExecuteCommand { command, args } => (
                Verb::new("shell", "exec"),
                serde_json::json!({ "command": command, "args": args }),
                vec!["shell.exec".to_string()],
            ),
            Intent::WebSearch { query } => (
                Verb::new("net", "http"),
                serde_json::json!({ "query": query }),
                vec!["net.http".to_string()],
            ),
            Intent::SendMessage {
                channel,
                recipient,
                content,
            } => (
                Verb::new("notify", "send"),
                serde_json::json!({
                    "channel": channel,
                    "recipient": recipient,
                    "content": content,
                }),
                vec!["notify.send".to_string()],
            ),
            Intent::OpenTerminalSession { program, args, cwd } => (
                Verb::new("terminal", "open"),
                serde_json::json!({
                    "program": program,
                    "args": args,
                    "cwd": cwd,
                }),
                vec!["terminal.open".to_string()],
            ),
            Intent::CloseTerminalSession { session_id } => (
                Verb::new("terminal", "close"),
                serde_json::json!({ "session_id": session_id }),
                vec!["terminal.close".to_string()],
            ),
            Intent::MountMcpServer {
                name,
                transport,
                command_or_url,
            } => (
                Verb::new("mcp", "mount"),
                serde_json::json!({
                    "name": name,
                    "transport": transport,
                    "command_or_url": command_or_url,
                }),
                vec!["mcp.mount".to_string()],
            ),
            Intent::UnmountMcpServer { name } => (
                Verb::new("mcp", "unmount"),
                serde_json::json!({ "name": name }),
                vec!["mcp.unmount".to_string()],
            ),
            Intent::ReconsentMcpServer { name } => (
                Verb::new("mcp", "reconsent"),
                serde_json::json!({ "name": name }),
                vec!["mcp.reconsent".to_string()],
            ),
            // Purely conversational / inspection variants do not map to a
            // capability-routed verb. Callers fall back to the typed
            // dispatch path for these.
            _ => return None,
        };
        let object = Object {
            kind: "intent_args".into(),
            value: object_value,
        };
        let mut tok = IntentToken::new(verb, object, provenance, namespace);
        tok.required_capabilities = caps;
        Some(tok)
    }
}

/// A fact extracted from conversational input alongside intent classification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Classification result.
#[derive(Debug, Clone)]
pub struct Classification {
    pub intent: Intent,
    pub confidence: f64,
    pub method: ClassificationMethod,
    /// Facts extracted from the input (even when intent is Chat).
    pub extracted_facts: Vec<ExtractedFact>,
}

/// How the classification was made.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationMethod {
    /// Regex fast-path (instant).
    Regex,
    /// LLM-based classification.
    Llm,
    /// Default fallback.
    Fallback,
}

/// Optional LLM hook used for intent classification.
#[async_trait::async_trait]
pub trait IntentFallback: Send + Sync {
    /// Returns a best-effort classification for ambiguous input.
    /// Return `None` to allow the classifier's normal fallback behavior.
    async fn classify_with_llm(&self, input: &str) -> Option<Classification>;

    /// History-aware variant. Default impl ignores history so trait
    /// implementors stay backwards-compatible. Override to feed prior
    /// turns into the classifier prompt — that's how the LLM tells
    /// "username : foo" (a follow-up parameter) from a self-introduction.
    async fn classify_with_history(
        &self,
        input: &str,
        _history: &[cortex::llm::Message],
    ) -> Option<Classification> {
        self.classify_with_llm(input).await
    }

    /// Capability-aware variant. `capabilities` is a rendered, lightweight
    /// summary of the tools/agents the kernel can currently dispatch to — the
    /// same live manifest the SOUL prompt and external `tools/list` see. Feeding
    /// it here makes the classifier the *third* consumer of one manifest rather
    /// than a blind one, so it can disambiguate capability-shaped requests
    /// (route to chat/`tool_call`) from look-alikes. The default ignores it and
    /// delegates to [`Self::classify_with_history`], so existing implementors
    /// stay backwards-compatible.
    async fn classify_with_context(
        &self,
        input: &str,
        history: &[cortex::llm::Message],
        _capabilities: Option<&str>,
    ) -> Option<Classification> {
        self.classify_with_history(input, history).await
    }
}

#[derive(Debug, Deserialize)]
struct LlmIntentPayload {
    intent: String,
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
    query: Option<String>,
    filter: Option<String>,
    since: Option<String>,
    limit: Option<usize>,
    older_than: Option<String>,
    status: Option<String>,
    nonce: Option<String>,
    decision: Option<String>,
    window: Option<String>,
    id: Option<String>,
    task_id: Option<String>,
    signal_id: Option<String>,
    enabled: Option<bool>,
    until: Option<String>,
    target: Option<String>,
    command: Option<String>,
    args: Option<Vec<String>>,
    description: Option<String>,
    cron: Option<String>,
    channel: Option<String>,
    recipient: Option<String>,
    content: Option<String>,
    /// Specialist agent id for `delegate_task` (e.g. `"claude-code"`).
    agent: Option<String>,
    /// Prompt body for `delegate_task` (separate from `query`/`content`
    /// so the LLM can disambiguate from chat).
    prompt: Option<String>,
    /// Facts extracted from conversational input (populated for chat intent).
    facts: Option<Vec<LlmFactPayload>>,
}

#[derive(Debug, Deserialize)]
struct LlmFactPayload {
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
}

/// LLM-based intent classifier used as a fallback/override for routing.
pub struct LlmIntentFallback {
    llm: Arc<dyn cortex::llm::LlmProvider>,
}

impl LlmIntentFallback {
    pub fn new(llm: Arc<dyn cortex::llm::LlmProvider>) -> Self {
        Self { llm }
    }

    fn parse_json_payload(raw: &str) -> Option<LlmIntentPayload> {
        cortex::extract_json_from_response(raw)
    }

    fn split_command(raw: &str) -> (String, Vec<String>) {
        let parts: Vec<&str> = raw.split_whitespace().collect();
        if parts.is_empty() {
            return (String::new(), Vec::new());
        }
        let command = parts[0].to_string();
        let args = parts[1..].iter().map(|s| s.to_string()).collect();
        (command, args)
    }
}

/// Decide the intent for an LLM `store_fact` classification.
///
/// A store is emitted only when the classifier actually *distilled* a fact: a
/// non-empty predicate and object, and an object that isn't just the raw
/// request echoed back. Otherwise we route to chat — the SOUL handles the
/// turn, and any genuine personal fact is still captured through the separate
/// `extracted_facts` channel.
///
/// This closes the WS4 failure: a compound imperative ("access the terminal,
/// type a message, and share it to our memory") was mis-routed to `store_fact`
/// and, lacking a distilled object, the raw sentence was filed verbatim as
/// "user said <whole request>" — a low-value, mis-categorised memory echo.
/// Mirrors the `execute_command` / `send_message` / `delegate_task` arms,
/// which already fall back to chat when their required fields are absent.
fn store_fact_or_chat(
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
    input: &str,
) -> Intent {
    let predicate = predicate.unwrap_or_default().trim().to_string();
    let object = object.unwrap_or_default().trim().to_string();
    let distilled =
        !predicate.is_empty() && !object.is_empty() && !object.eq_ignore_ascii_case(input.trim());
    if distilled {
        Intent::StoreFact {
            subject: subject
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "user".to_string()),
            predicate,
            object,
        }
    } else {
        Intent::Chat {
            content: input.to_string(),
        }
    }
}

/// First line of the classifier prompt — fixed preamble that precedes the
/// generated `Valid intents:` line.
const CLASSIFIER_PROMPT_HEADER: &str =
    "You classify user input into exactly one intent for Brain OS.";

/// The genuinely cross-cutting tail of the classifier prompt — the
/// fact-extraction contract and the JSON output shape. This is *not*
/// per-intent (it applies regardless of the classified intent), so unlike the
/// former hand-written `Rules:` prose it stays fixed here; the per-intent rules
/// are generated from [`taxonomy::INTENT_SPECS`]'s [`taxonomy::IntentUsage`]
/// (see [`build_classifier_system_prompt`]).
const FACT_EXTRACTION_TAIL: &str = r#"FACT EXTRACTION: Regardless of intent, if the input contains personal facts about the user (name, role, company, projects, skills, interests, goals, location, preferences, habits), extract them into the "facts" array. Each fact is {"subject": "user", "predicate": "<snake_case_verb>", "object": "<value>"}.
Predicates: name_is, role_is, works_at, works_on, title_is, interested_in, lives_in, skill_is, goal_is, preference_is, likes, etc.
Only extract a fact when the user is making a clear self-statement in natural language ("my name is X", "I work at Y", "I'm a Z developer"). Do NOT extract facts from:
- Short parameter-shaped messages (`username : foo`, `email = bar`, `5 minutes`, `yes`, `no`) — these are almost always follow-up parameters to a previous turn. Classify the intent as `chat` and return facts: [].
- Bare identifiers, paths, or URLs typed alone — they're context for the prior request, not biography.
- Anything you wouldn't confidently restate as a sentence about the user.
When recent conversation history is supplied above your input, USE it: a one-line reply right after a question is a parameter to that question, not a new biographical claim. If no history is supplied, default to skepticism — extract facts only when the wording is self-evidently a self-statement.
If no facts qualify, set facts to [].

Return only JSON with keys: intent, subject, predicate, object, query, filter, since, limit, older_than, status, nonce, decision, window, id, task_id, enabled, until, target, command, args, description, cron, channel, recipient, content, facts.
Missing keys must be null. facts must be [] if none."#;

/// Assemble the classifier system prompt entirely from
/// [`taxonomy::INTENT_SPECS`]: the `Valid intents:` line and the per-intent
/// `Rules:` block are both generated from the `LlmFallback` rows (their keys
/// and authored [`taxonomy::IntentUsage`]), so the natural-language vocabulary
/// and its disambiguation can never silently drift from the `Intent` enum.
/// Only the cross-cutting [`FACT_EXTRACTION_TAIL`] is fixed. The keys/rules are
/// emitted in table order (grouped by category), the canonical SSOT ordering.
fn build_classifier_system_prompt() -> String {
    let llm_specs: Vec<&taxonomy::IntentSpec> = taxonomy::INTENT_SPECS
        .iter()
        .filter(|s| s.nl_routable == taxonomy::NlRouting::LlmFallback)
        .collect();
    let valid_intents = llm_specs
        .iter()
        .map(|s| s.key)
        .collect::<Vec<_>>()
        .join(", ");
    let rules = llm_specs
        .iter()
        .map(|s| s.render_prompt_rule())
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{CLASSIFIER_PROMPT_HEADER}\nValid intents: {valid_intents}.\nRules:\n{rules}\n\n{FACT_EXTRACTION_TAIL}"
    )
}

/// The full classifier system prompt, built once on first use. The
/// `Valid intents:` line is generated from the taxonomy SSOT
/// ([`build_classifier_system_prompt`]); everything else is the fixed
/// header/rules prose.
static CLASSIFIER_SYSTEM_PROMPT: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(build_classifier_system_prompt);

#[async_trait::async_trait]
impl IntentFallback for LlmIntentFallback {
    async fn classify_with_llm(&self, input: &str) -> Option<Classification> {
        self.classify_with_history(input, &[]).await
    }

    async fn classify_with_history(
        &self,
        input: &str,
        history: &[cortex::llm::Message],
    ) -> Option<Classification> {
        self.classify_with_context(input, history, None).await
    }

    async fn classify_with_context(
        &self,
        input: &str,
        history: &[cortex::llm::Message],
        capabilities: Option<&str>,
    ) -> Option<Classification> {
        use cortex::llm::{Message, Role};

        // Build a compact transcript from at most the last 4 turns. The
        // classifier only needs enough context to recognise follow-ups —
        // not the full thread. Keeping it short also caps token cost on
        // every classification call.
        const HISTORY_TURNS: usize = 4;
        const PER_TURN_CHARS: usize = 240;
        let transcript = if history.is_empty() {
            String::new()
        } else {
            let recent: Vec<&Message> = history.iter().rev().take(HISTORY_TURNS).collect();
            let mut lines = Vec::new();
            for msg in recent.into_iter().rev() {
                let label = match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System | Role::Tool => continue,
                };
                let trimmed: String = msg.content.chars().take(PER_TURN_CHARS).collect();
                let suffix = if msg.content.chars().count() > PER_TURN_CHARS {
                    "…"
                } else {
                    ""
                };
                lines.push(format!("{label}: {trimmed}{suffix}"));
            }
            format!(
                "Recent conversation (oldest first):\n{}\n\n",
                lines.join("\n")
            )
        };

        let user_content = if transcript.is_empty() {
            input.to_string()
        } else {
            format!("{transcript}New input to classify:\n{input}")
        };

        // Feed the live capability manifest so the classifier shares the same
        // view of available tools the SOUL and external clients have. It is
        // context for disambiguation only — the valid *intents* are still the
        // fixed control-plane vocabulary, and the classifier never emits
        // `tool_call` from prose (the tool-loop owns that path).
        let system_prompt = match capabilities {
            Some(caps) if !caps.trim().is_empty() => format!(
                "{}\n\nCurrently available capabilities (for disambiguation only — \
                 not new intents; route capability requests to chat):\n{}",
                CLASSIFIER_SYSTEM_PROMPT.as_str(),
                caps.trim()
            ),
            _ => CLASSIFIER_SYSTEM_PROMPT.clone(),
        };

        let messages = vec![Message::system(system_prompt), Message::user(user_content)];

        let response = match self.llm.generate(&messages).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM intent classification failed: {e}");
                return None;
            }
        };

        tracing::debug!(
            raw_len = response.content.len(),
            "LLM classifier raw response"
        );

        let payload = match Self::parse_json_payload(&response.content) {
            Some(p) => p,
            None => {
                tracing::warn!(
                    "LLM classifier returned unparseable JSON: {}",
                    &response.content[..response.content.len().min(200)]
                );
                return None;
            }
        };
        let key = payload.intent.to_ascii_lowercase();

        // Extract facts from the LLM response
        let extracted_facts: Vec<ExtractedFact> = payload
            .facts
            .unwrap_or_default()
            .into_iter()
            .filter_map(|f| {
                let predicate = f.predicate.unwrap_or_default();
                let object = f.object.unwrap_or_default();
                if predicate.is_empty() || object.is_empty() {
                    None
                } else {
                    Some(ExtractedFact {
                        subject: f.subject.unwrap_or_else(|| "user".to_string()),
                        predicate,
                        object,
                    })
                }
            })
            .collect();

        let intent = match key.as_str() {
            "store_fact" => {
                store_fact_or_chat(payload.subject, payload.predicate, payload.object, input)
            }
            "recall" => Intent::Recall {
                query: payload.query.unwrap_or_else(|| input.to_string()),
            },
            "forget" => Intent::Forget {
                target: payload.target.unwrap_or_else(|| input.to_string()),
            },
            "execute_command" => {
                let raw = payload
                    .command
                    .or(payload.content)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let (command, mut args) = Self::split_command(&raw);
                if payload.args.as_ref().is_some_and(|a| !a.is_empty()) {
                    args = payload.args.unwrap_or_default();
                }
                if command.is_empty() {
                    Intent::Chat {
                        content: input.to_string(),
                    }
                } else {
                    Intent::ExecuteCommand { command, args }
                }
            }
            "web_search" => Intent::WebSearch {
                query: payload.query.unwrap_or_else(|| input.to_string()),
            },
            "query_audit" => Intent::QueryAudit {
                filter: payload.filter,
                since: payload.since,
                limit: payload.limit,
            },
            "prune_audit" => Intent::PruneAudit {
                older_than: payload.older_than.unwrap_or_else(|| "30d".to_string()),
            },
            "list_approvals" => Intent::List {
                resource: Resource::Approvals,
                filter: payload.status,
            },
            "respond_to_approval" => Intent::RespondToApproval {
                nonce: payload.nonce.unwrap_or_default(),
                decision: payload.decision.unwrap_or_else(|| "approve".to_string()),
            },
            "budget_status" => Intent::BudgetStatus {
                window: payload.window,
            },
            "schedule" => {
                let description = payload
                    .description
                    .or(payload.content)
                    .unwrap_or_else(|| input.to_string());
                Intent::Schedule {
                    description,
                    cron: payload.cron,
                }
            }
            "list_schedules" => Intent::List {
                resource: Resource::Schedules,
                filter: None,
            },
            "cancel_schedule" => Intent::Cancel {
                target: CancelTarget::Schedule,
                id: payload.id.unwrap_or_default(),
            },
            "send_message" => {
                let channel = payload.channel.unwrap_or_default();
                let recipient = payload.recipient.unwrap_or_default();
                let content = payload.content.unwrap_or_default();
                if channel.is_empty() || recipient.is_empty() || content.is_empty() {
                    Intent::Chat {
                        content: input.to_string(),
                    }
                } else {
                    Intent::SendMessage {
                        channel,
                        recipient,
                        content,
                    }
                }
            }
            "system_status" => Intent::SystemStatus,
            "decompose_task" => Intent::DecomposeTask {
                request: payload
                    .content
                    .or(payload.description)
                    .unwrap_or_else(|| input.to_string()),
            },
            "list_tasks" => Intent::List {
                resource: Resource::Tasks,
                filter: None,
            },
            "task_status" => Intent::TaskStatus {
                task_id: payload.task_id.unwrap_or_default(),
            },
            "cancel_task" => Intent::Cancel {
                target: CancelTarget::Task,
                id: payload.task_id.unwrap_or_default(),
            },
            "cancel_signal" => Intent::Cancel {
                target: CancelTarget::Signal,
                id: payload.signal_id.unwrap_or_default(),
            },
            "query_agents" => Intent::QueryAgents {
                filter: payload.query.unwrap_or_default(),
            },
            "delegate_task" => {
                let agent = payload
                    .agent
                    .clone()
                    .unwrap_or_default()
                    .trim()
                    .to_lowercase();
                let prompt = payload
                    .prompt
                    .clone()
                    .or(payload.content.clone())
                    .or(payload.query.clone())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if agent.is_empty() || prompt.is_empty() {
                    Intent::Chat {
                        content: input.to_string(),
                    }
                } else {
                    Intent::DelegateTask { agent, prompt }
                }
            }
            "set_proactivity" => Intent::SetProactivity {
                enabled: payload.enabled.unwrap_or(true),
                until: payload.until,
            },
            "proactivity_status" => Intent::ProactivityStatus,
            "memory_summary" => Intent::MemorySummary,
            _ => Intent::Chat {
                content: input.to_string(),
            },
        };

        if !extracted_facts.is_empty() {
            tracing::info!(
                count = extracted_facts.len(),
                "LLM extracted facts from input"
            );
        }

        Some(Classification {
            intent,
            confidence: 0.7,
            method: ClassificationMethod::Llm,
            extracted_facts,
        })
    }
}

/// Normalized message format for all channels.
#[derive(Debug, Clone)]
pub struct NormalizedMessage {
    pub content: String,
    pub channel: String,
    pub sender: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub message_id: Option<String>,
}
