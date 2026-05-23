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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    /// Store a fact explicitly.
    StoreFact {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Recall/search memory.
    Recall { query: String },
    /// Forget/delete something.
    Forget { target: String },
    /// Execute a command.
    ExecuteCommand { command: String, args: Vec<String> },
    /// Search the web.
    WebSearch { query: String },
    /// Query the audit trail.
    QueryAudit {
        filter: Option<String>,
        since: Option<String>,
        limit: Option<usize>,
    },
    /// Prune the audit trail.
    PruneAudit { older_than: String },
    /// List pending approvals.
    ListApprovals { status: Option<String> },
    /// Respond to a pending approval.
    RespondToApproval { nonce: String, decision: String },
    /// Check LLM budget and usage status.
    BudgetStatus { window: Option<String> },
    /// Schedule something.
    Schedule {
        description: String,
        cron: Option<String>,
    },
    /// List active background schedules.
    ListSchedules,
    /// Cancel a scheduled intent.
    CancelSchedule { id: String },
    /// Send via a channel.
    SendMessage {
        channel: String,
        recipient: String,
        content: String,
    },
    /// Get system status.
    SystemStatus,
    /// Decompose a complex request into an executable task plan.
    DecomposeTask { request: String },
    /// List active or recent tasks.
    ListTasks,
    /// Get the status of a specific task.
    TaskStatus { task_id: String },
    /// Cancel a running task.
    CancelTask { task_id: String },
    /// Cancel an in-flight signal by its id. Wires the Live-tab cancel
    /// button in the observability UI. Distinct from `CancelTask` —
    /// that aborts an orchestrated multi-step plan; this aborts a
    /// single Signal's pipeline.
    CancelSignal { signal_id: String },
    /// Ask about available specialist agents (delegates). Optional
    /// `filter` narrows the answer: e.g. "rust", "aider", or "".
    QueryAgents { filter: String },
    /// Run a single-turn delegation to a named specialist agent. Bypasses
    /// task decomposition — used when the user explicitly asks "delegate
    /// to claude-code: ..." or "@codex: ...". For multi-step plans the
    /// orchestrator picks the agent itself via [`DecomposeTask`].
    DelegateTask { agent: String, prompt: String },
    /// Configure proactivity / nudges.
    SetProactivity {
        enabled: bool,
        until: Option<String>,
    },
    /// Get proactivity status and configuration.
    ProactivityStatus,
    /// Dump and summarise everything stored in memory.
    MemorySummary,
    /// List registered channels (router-known descriptors). The
    /// natural-language replacement for inspection CLIs.
    ListChannels,
    /// Show learned channel preferences for a (namespace, category).
    /// `category` is one of: confirm, nudge, report, response, alert.
    /// `namespace` defaults to "personal".
    ChannelPreferences {
        namespace: Option<String>,
        category: Option<String>,
    },
    /// Pin or unpin a channel preference. Pinned weights bypass the
    /// min-weight threshold during routing.
    SetChannelPreference {
        channel: String,
        category: String,
        weight: f32,
        pinned: bool,
    },
    /// Open a new terminal session via the Terminal Bridge. Returns the
    /// session id so the caller can `Attach` or close it later.
    OpenTerminalSession {
        program: String,
        args: Vec<String>,
        cwd: Option<String>,
    },
    /// List currently-active terminal sessions (read-only inspection).
    ListTerminalSessions,
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
    /// List currently-mounted MCP servers (read-only inspection).
    ListMcpServers,
    /// List active standing approvals — every `(agent_id, verb_ns,
    /// verb_action)` triple currently pre-granted to bypass the
    /// human-confirm prompt. Read-only inspection so the user can
    /// audit what their reflexes are allowed to do unattended.
    ListStandingApprovals,
    /// Revoke a previously-granted standing approval by id. Idempotent —
    /// revoking an unknown or already-revoked id returns a friendly
    /// "not found" rather than failing.
    RevokeStandingApproval { id: String },
    /// Abstract tool invocation expressed as a Standardized Intent Token.
    /// Emitted by the classifier when the requested action can't be served by
    /// any of the typed variants above and must instead be resolved against
    /// the capability index (MCP tools, native backends, terminal sessions).
    /// The router scores candidates and dispatches the winner; until the
    /// router is wired the pipeline returns a deterministic placeholder.
    /// Boxed so the enum discriminant stays compact — the SIT envelope is
    /// the heaviest variant by far.
    ToolCall(Box<IntentToken>),
    /// Regular chat/conversation.
    Chat { content: String },
}

impl Intent {
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

const CLASSIFIER_SYSTEM_PROMPT: &str = r#"You classify user input into exactly one intent for Brain OS.
Valid intents: store_fact, recall, forget, execute_command, web_search, query_audit, prune_audit, list_approvals, respond_to_approval, budget_status, schedule, list_schedules, cancel_schedule, send_message, system_status, decompose_task, list_tasks, task_status, cancel_task, cancel_signal, query_agents, delegate_task, set_proactivity, proactivity_status, memory_summary, chat.
Rules:
- query_audit is for checking past actions: "what did I run today", "show my audit entries", "what did I approve yesterday".
- prune_audit is for deleting old audit entries: "prune audit logs older than 30 days".
- list_approvals is for showing pending confirmations: "what am I waiting to approve", "show pending approvals".
- respond_to_approval is for approving or rejecting a nonce: "approve 1234", "reject 5678".
- budget_status is for checking usage: "how much have I spent", "what's my token budget".
- schedule is for new future tasks: "remind me in 5 minutes to...", "schedule a search every day for...".
- list_schedules/cancel_schedule are for managing background schedules: "what's scheduled", "cancel schedule 123".
- list_tasks/task_status/cancel_task are for managing complex multi-step tasks from decompose_task: "what tasks are running", "status of task 42", "cancel task 10".
- cancel_signal aborts an in-flight Signal by its UUID — distinct from cancel_task: "cancel signal e4b8…". The signal_id payload field carries the UUID.
- set_proactivity/proactivity_status are for managing nudges/habit engine: "pause nudges for 2h", "disable proactivity", "check proactivity status".
- memory_summary is for broad "dump everything you know about me" requests: "summarise my memory", "what do you know", "what have you stored", "show me my memories", "tell me what you remember about me". No query parameter needed.
- recall is for specific memory queries that name a concrete topic: "what do you know about my project", "what did we discuss about Rust", "what do you remember about my goals". The query MUST identify a topic.
- Conversational meta-questions about the current chat ("what did we discuss?", "what did I just say?", "summarize our conversation", "what did we talk about earlier today") are chat — the assistant answers from the live conversation history, not from memory lookup.
- Questions that are NOT about stored memories (general knowledge, opinions, how-to questions) are chat.
- Questions should NEVER be execute_command.
- store_fact is ONLY for explicit memory requests: "remember that ...", "note that ...", "keep in mind ...".
- execute_command is ONLY for explicit requests like "run ls", "execute cargo build". The command field must be a real shell command (ls, git, cargo, etc.).
- decompose_task is for multi-step requests that need planning and execution: "build a CSV export feature", "set up CI/CD pipeline", "refactor the auth module and add tests", "deploy to production". The request must involve multiple steps or coordination. Simple single-step requests are NOT decompose_task.
- A user message that names a local path (e.g. "summarise /Users/me/notes", "what's in ~/downloads/x", "read this file: /tmp/foo.txt") is chat — Brain reads the path as attached context and responds conversationally. Do NOT route these to decompose_task or any inspect-style intent; they go through the normal chat flow.
- query_agents is for asking which specialist agents are available or why a named agent is unavailable: "what agents do you have", "which agents can code rust", "why aren't you using aider".
- delegate_task is for explicit single-shot delegation to a named agent: "delegate to claude-code: refactor X", "ask codex: explain Y", "@aider: fix the bug". Set `agent` to the lowercase agent id and `prompt` to the task body. Do NOT use this for multi-step plans — those go to decompose_task and the orchestrator picks an agent itself.
- Conversational statements ("I've done X", "I completed X", "I like X") are chat but ALSO extract any personal facts (see below).
- Prefer web_search for explicit search requests about internet/google/latest/current external info.
- For web_search, set 'query' to the exact optimal search terms, stripping conversational fluff.
- Use system_status only for explicit status checks like "/status".
- Use chat when uncertain or for general conversation.

FACT EXTRACTION: Regardless of intent, if the input contains personal facts about the user (name, role, company, projects, skills, interests, goals, location, preferences, habits), extract them into the "facts" array. Each fact is {"subject": "user", "predicate": "<snake_case_verb>", "object": "<value>"}.
Predicates: name_is, role_is, works_at, works_on, title_is, interested_in, lives_in, skill_is, goal_is, preference_is, likes, etc.
Only extract a fact when the user is making a clear self-statement in natural language ("my name is X", "I work at Y", "I'm a Z developer"). Do NOT extract facts from:
- Short parameter-shaped messages (`username : foo`, `email = bar`, `5 minutes`, `yes`, `no`) — these are almost always follow-up parameters to a previous turn. Classify the intent as `chat` and return facts: [].
- Bare identifiers, paths, or URLs typed alone — they're context for the prior request, not biography.
- Anything you wouldn't confidently restate as a sentence about the user.
When recent conversation history is supplied above your input, USE it: a one-line reply right after a question is a parameter to that question, not a new biographical claim. If no history is supplied, default to skepticism — extract facts only when the wording is self-evidently a self-statement.
If no facts qualify, set facts to [].

Return only JSON with keys: intent, subject, predicate, object, query, filter, since, limit, older_than, status, nonce, decision, window, id, task_id, enabled, until, target, command, args, description, cron, channel, recipient, content, facts.
Missing keys must be null. facts must be [] if none."#;

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
                    Role::System => continue,
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

        let messages = vec![
            Message {
                role: Role::System,
                content: CLASSIFIER_SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: Role::User,
                content: user_content,
            },
        ];

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
            "store_fact" => Intent::StoreFact {
                subject: payload.subject.unwrap_or_else(|| "user".to_string()),
                predicate: payload.predicate.unwrap_or_else(|| "said".to_string()),
                object: payload.object.unwrap_or_else(|| input.to_string()),
            },
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
            "list_approvals" => Intent::ListApprovals {
                status: payload.status,
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
            "list_schedules" => Intent::ListSchedules,
            "cancel_schedule" => Intent::CancelSchedule {
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
            "list_tasks" => Intent::ListTasks,
            "task_status" => Intent::TaskStatus {
                task_id: payload.task_id.unwrap_or_default(),
            },
            "cancel_task" => Intent::CancelTask {
                task_id: payload.task_id.unwrap_or_default(),
            },
            "cancel_signal" => Intent::CancelSignal {
                signal_id: payload.signal_id.unwrap_or_default(),
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
