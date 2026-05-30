use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use regex::Regex;

use crate::{Classification, ClassificationMethod, Intent, IntentFallback};

/// Pre-compiled regex patterns compiled once at first access via LazyLock.
/// Each pattern is paired with its base intent and named capture group extractors.
pub(crate) struct PatternDef {
    pub(crate) regex: &'static LazyLock<Regex>,
    pub(crate) base_intent: Intent,
    pub(crate) extractors: &'static [(&'static str, usize)],
}

static RECALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Recall = "look up something in long-term memory by topic".
    // Conversational meta-questions like "what did we discuss?" must NOT match
    // here — they belong to Chat so the LLM can use the session's own history.
    // Triggers require an explicit recall verb followed by a topic.
    Regex::new(
        r"(?i)^(?:recall|what do you know about|what.*\bknow about|tell me about)\s+(.+?)\??$",
    )
    .expect("invariant: RECALL_RE must be valid")
});

static MEMORY_SUMMARY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?ix)
        ^(?:
            (?:summarise|summarize|sum\s+up|give\s+me\s+a\s+summary\s+of)\s+(?:my\s+)?(?:memory|memories|what\s+you\s+know)|
            what\s+(?:do\s+you\s+know|have\s+you\s+(?:learned|stored|remembered))\??|
            what\s+(?:are\s+)?(?:you\s+)?(?:in\s+my\s+memory|stored\s+about\s+me|you\s+remember(?:\s+about\s+me)?)\??|
            show\s+(?:me\s+)?(?:my\s+|all\s+)?(?:memory|memories|what\s+you\s+know|stored\s+facts?)|
            (?:share|give)\s+(?:me\s+)?(?:all\s+|my\s+|the\s+)?(?:memory|memories|stored\s+facts?|stored\s+memories?)(?:\s+with\s+me)?|
            tell\s+me\s+(?:everything|what)\s+you\s+(?:know|remember|have)(?:\s+about\s+me)?\??|
            (?:dump|list|display)\s+(?:my\s+|all\s+)?(?:memory|memories|all\s+facts|stored\s+facts?)
        )$
    ").expect("invariant: MEMORY_SUMMARY_RE must be valid")
});

static EXECUTE_COMMAND_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Capture the whole tail after the verb (and an optional "please"); the
    // binary/args split and filler stripping happen in `normalize_command`,
    // because natural phrasing wraps the command in filler ("the command:",
    // "the following:") and shells (backticks, quotes, `$(...)`) that a
    // single regex can't cleanly peel off.
    Regex::new(r"(?i)^(?:please\s+)?(?:run|exec|execute)\s+(.+)$")
        .expect("invariant: EXECUTE_COMMAND_RE must be valid")
});

/// Leading words that are conversational filler in "run the command: …"
/// phrasing, not part of the command. Compared case-insensitively with any
/// trailing colon stripped, so both `command` and `command:` match.
const COMMAND_FILLER: &[&str] = &["the", "this", "following", "please", "command"];

/// Peel the filler and shell wrappers off a captured command string so the
/// first whitespace token is the real binary. Handles:
///
/// - a `$(...)`, backtick, or matching-quote wrapper around the whole command
/// - leading filler tokens ("the", "the command:", "the following:", …)
///
/// Leaves a bare `cmd` token intact (it's a real Windows binary); only the
/// explicit `cmd:` form is treated as filler.
pub(crate) fn normalize_command(raw: &str) -> String {
    let mut s = raw.trim();

    // Peel matching outer wrappers, possibly nested ($(`...`) etc.).
    loop {
        let stripped = if let Some(inner) = s.strip_prefix("$(").and_then(|x| x.strip_suffix(')')) {
            inner.trim()
        } else if let Some(inner) = ['`', '"', '\'']
            .iter()
            .find_map(|&q| s.strip_prefix(q).and_then(|x| x.strip_suffix(q)))
        {
            inner.trim()
        } else {
            break;
        };
        s = stripped;
    }

    // Drop leading filler tokens.
    let mut tokens = s.split_whitespace().peekable();
    while let Some(&tok) = tokens.peek() {
        if is_command_filler(tok) {
            tokens.next();
        } else {
            break;
        }
    }
    tokens.collect::<Vec<_>>().join(" ")
}

/// True if `tok` is a leading filler word. `cmd:` is filler; bare `cmd` is not.
fn is_command_filler(tok: &str) -> bool {
    let lc = tok.to_ascii_lowercase();
    if lc == "cmd:" {
        return true;
    }
    COMMAND_FILLER.contains(&lc.trim_end_matches(':'))
}

static WEB_SEARCH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:(?:can you|could you|please|will you|would you)\s+)?(?:search|look up|google|web search|look for)\s+(?:for\s+|about\s+|up\s+)?(.+?)(?:\?)?$")
        .expect("invariant: WEB_SEARCH_RE (main) must be valid")
});

static WEB_SEARCH_FIND_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:(?:Can you|Could you|please|will you|would you)\s+)?find\s+(?:information\s+)?(?:about|for)\s+(.+?)(?:\?)?$")
        .expect("invariant: WEB_SEARCH_FIND_RE must be valid")
});

/// `delegate to <agent>: <prompt>` or `@<agent> <prompt>` — single-turn
/// invocation of a named specialist agent. Agent names are lowercase
/// alphanumeric with optional `-` (claude-code, codex, aider, gemini-cli).
static DELEGATE_TASK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
        ^(?:
            (?:please\s+)?(?:ask|delegate(?:\s+to)?|hand(?:\s+off|\s+over)?(?:\s+to)?|forward\s+to|use)\s+
            (?P<agent>[a-z][a-z0-9_-]*)\s*[:,\-]\s*(?P<prompt>.+)
            |
            @(?P<agent2>[a-z][a-z0-9_-]*)\s*[:,]?\s*(?P<prompt2>.+)
        )$
        ",
    )
    .expect("invariant: DELEGATE_TASK_RE must be valid")
});

/// URL grounding — message contains an http(s) URL alongside a fetch-like
/// verb (`fetch`, `read`, `open`, `summarise`, etc.). Routes to WebSearch
/// so the action runner pulls the URL body as grounding for the LLM.
/// The whole message is captured as the query because the WebSearch
/// backend's `extract_urls()` will harvest the URL itself and fetch it
/// in parallel with the search.
static URL_FETCH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)
        ^(?:.*\b(?:fetch|read|open|summari[sz]e|grab|pull|get|show|visit|browse|scrape|crawl|check|view|extract|parse|tell\s+me\s+about|what\s+does)\b.*)?
        \s*(?P<query>.*?https?://\S+.*?)\s*\??$
        ",
    )
    .expect("invariant: URL_FETCH_RE must be valid")
});

static QUERY_AUDIT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:what did I (?:run|do|approve)|show (?:my )?audit|list audit)(?:\s+(?:today|yesterday|since\s+(.+)))?\??$")
        .expect("invariant: QUERY_AUDIT_RE must be valid")
});

static PRUNE_AUDIT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^prune\s+audit\s+(?:logs?\s+)?(?:older\s+than\s+)?(.+?)$")
        .expect("invariant: PRUNE_AUDIT_RE must be valid")
});

static LIST_APPROVALS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:what am I waiting to approve|show pending approvals|list approvals|pending approvals)\??$")
        .expect("invariant: LIST_APPROVALS_RE must be valid")
});

static RESPOND_TO_APPROVAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Accept hyphens so UUID-style IDs (both confirm-engine nonces and
    // orchestrator task IDs) match the fast path.
    //
    // Two shapes:
    //   `approve <id>` / `reject <id>`            — explicit
    //   `approve` / `y` / `yes` / `reject` / `n`  — bare; resolved by the
    //                                               signal handler against
    //                                               `pending_approvals()`
    Regex::new(r"(?i)^(?P<decision>approve|reject|yes|no|y|n)(?:\s+(?P<nonce>[a-zA-Z0-9-]+))?$")
        .expect("invariant: RESPOND_TO_APPROVAL_RE must be valid")
});

static BUDGET_STATUS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:how much have I spent|what'?s my (?:token )?budget|budget status)\??$")
        .expect("invariant: BUDGET_STATUS_RE must be valid")
});

static SCHEDULE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:remind me|schedule|set reminder)\s+(?:to\s+)?(.+?)$")
        .expect("invariant: SCHEDULE_RE must be valid")
});

static LIST_SCHEDULES_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:what'?s scheduled|list schedules|show schedules)\??$")
        .expect("invariant: LIST_SCHEDULES_RE must be valid")
});

static CANCEL_SCHEDULE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^cancel\s+schedule\s+(.+?)$")
        .expect("invariant: CANCEL_SCHEDULE_RE must be valid")
});

static SEND_MESSAGE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:send|message|text)\s+(?:via\s+(\w+)\s+)?(?:to\s+)?(.+?)\s+(?:saying\s+|:\s+)(.+?)$")
        .expect("invariant: SEND_MESSAGE_RE must be valid")
});

static STATUS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^/status$").expect("invariant: STATUS_RE must be valid"));

static QUERY_AGENTS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(?:what|which|list)\s+(?:agents?|delegates?|specialists?)(?:\s+(?:do you have|are available|are there|can you use|can (?:code|write|do)\s+(.+?)))?\??$",
    )
    .expect("invariant: QUERY_AGENTS_RE must be valid")
});

static QUERY_AGENTS_WHY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^why\s+(?:aren'?t|isn'?t|can'?t)\s+you\s+(?:using|use|offering)\s+(.+?)\??$")
        .expect("invariant: QUERY_AGENTS_WHY_RE must be valid")
});

static LIST_TASKS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:what tasks are running|list tasks|show tasks)\??$")
        .expect("invariant: LIST_TASKS_RE must be valid")
});

static TASK_STATUS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:status of task|task status)\s+(.+?)\??$")
        .expect("invariant: TASK_STATUS_RE must be valid")
});

static CANCEL_TASK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^cancel\s+task\s+(.+?)$").expect("invariant: CANCEL_TASK_RE must be valid")
});

static CANCEL_SIGNAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^cancel\s+signal\s+(.+?)$").expect("invariant: CANCEL_SIGNAL_RE must be valid")
});

static SET_PROACTIVITY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(pause|disable|enable|turn (?:on|off))\s+(?:nudges|proactivity|reminders)(?:\s+(?:for\s+)?(.+))?$")
        .expect("invariant: SET_PROACTIVITY_RE must be valid")
});

static PROACTIVITY_STATUS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:check proactivity status|proactivity status)\??$")
        .expect("invariant: PROACTIVITY_STATUS_RE must be valid")
});

static DECOMPOSE_TASK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:decompose|plan|orchestrate)\s+(?:task\s+)?(.+?)$")
        .expect("invariant: DECOMPOSE_TASK_RE must be valid")
});

static LIST_CHANNELS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:list channels|show channels|what channels(?:\s+are\s+available)?)\??$")
        .expect("invariant: LIST_CHANNELS_RE must be valid")
});

static CHANNEL_PREFS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:show |list )?channel\s+(?:preferences|prefs)(?:\s+for\s+(\w+))?\??$")
        .expect("invariant: CHANNEL_PREFS_RE must be valid")
});

static PIN_CHANNEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(pin|unpin|prefer)\s+(\S+)\s+for\s+(confirm|confirms|nudge|nudges|report|reports|response|responses|alert|alerts)$")
        .expect("invariant: PIN_CHANNEL_RE must be valid")
});

/// All patterns in priority order — first match wins.
///
/// `Intent::StoreFact` (`remember/note/keep in mind …`) and
/// `Intent::Forget` (`forget/delete/remove …`) intentionally have no
/// entries here — those prefixes are handled earlier by
/// `classify_explicit`, which runs before `classify_regex` in
/// `classify_with_history`. The regex shadows were removed in v0.4.0
/// to drop dead code (audit Issues 13 + 14).
pub(crate) const PATTERNS: &[PatternDef] = &[
    PatternDef {
        regex: &MEMORY_SUMMARY_RE,
        base_intent: Intent::MemorySummary,
        extractors: &[],
    },
    PatternDef {
        regex: &RECALL_RE,
        base_intent: Intent::Recall {
            query: String::new(),
        },
        extractors: &[("query", 1)],
    },
    PatternDef {
        regex: &EXECUTE_COMMAND_RE,
        base_intent: Intent::ExecuteCommand {
            command: String::new(),
            args: Vec::new(),
        },
        extractors: &[("command", 1)],
    },
    PatternDef {
        regex: &WEB_SEARCH_RE,
        base_intent: Intent::WebSearch {
            query: String::new(),
        },
        extractors: &[("query", 1)],
    },
    PatternDef {
        regex: &WEB_SEARCH_FIND_RE,
        base_intent: Intent::WebSearch {
            query: String::new(),
        },
        extractors: &[("query", 1)],
    },
    // URL-bearing requests route to WebSearch so the action runner fetches
    // their bodies. Placed after the `search ...` patterns so explicit
    // search verbs win, but before downstream patterns can swallow URLs.
    PatternDef {
        regex: &URL_FETCH_RE,
        base_intent: Intent::WebSearch {
            query: String::new(),
        },
        extractors: &[("query", 0)],
    },
    PatternDef {
        regex: &QUERY_AUDIT_RE,
        base_intent: Intent::QueryAudit {
            filter: None,
            since: None,
            limit: None,
        },
        extractors: &[("since", 1)],
    },
    PatternDef {
        regex: &PRUNE_AUDIT_RE,
        base_intent: Intent::PruneAudit {
            older_than: String::new(),
        },
        extractors: &[("older_than", 1)],
    },
    PatternDef {
        regex: &LIST_APPROVALS_RE,
        base_intent: Intent::ListApprovals { status: None },
        extractors: &[],
    },
    PatternDef {
        regex: &RESPOND_TO_APPROVAL_RE,
        base_intent: Intent::RespondToApproval {
            nonce: String::new(),
            decision: String::new(),
        },
        // Named-group lookup; numeric indices are unused for this pattern.
        extractors: &[("decision", 0), ("nonce", 0)],
    },
    PatternDef {
        regex: &BUDGET_STATUS_RE,
        base_intent: Intent::BudgetStatus { window: None },
        extractors: &[],
    },
    PatternDef {
        regex: &SCHEDULE_RE,
        base_intent: Intent::Schedule {
            description: String::new(),
            cron: None,
        },
        extractors: &[("description", 1)],
    },
    PatternDef {
        regex: &LIST_SCHEDULES_RE,
        base_intent: Intent::ListSchedules,
        extractors: &[],
    },
    PatternDef {
        regex: &CANCEL_SCHEDULE_RE,
        base_intent: Intent::CancelSchedule { id: String::new() },
        extractors: &[("id", 1)],
    },
    PatternDef {
        regex: &SEND_MESSAGE_RE,
        base_intent: Intent::SendMessage {
            channel: String::new(),
            recipient: String::new(),
            content: String::new(),
        },
        extractors: &[("channel", 1), ("recipient", 2), ("content", 3)],
    },
    PatternDef {
        regex: &STATUS_RE,
        base_intent: Intent::SystemStatus,
        extractors: &[],
    },
    PatternDef {
        regex: &QUERY_AGENTS_RE,
        base_intent: Intent::QueryAgents {
            filter: String::new(),
        },
        extractors: &[("filter", 1)],
    },
    PatternDef {
        regex: &DELEGATE_TASK_RE,
        base_intent: Intent::DelegateTask {
            agent: String::new(),
            prompt: String::new(),
        },
        // Named-group lookup; numeric indices are unused.
        extractors: &[("agent", 0), ("prompt", 0)],
    },
    PatternDef {
        regex: &QUERY_AGENTS_WHY_RE,
        base_intent: Intent::QueryAgents {
            filter: String::new(),
        },
        extractors: &[("filter", 1)],
    },
    PatternDef {
        regex: &LIST_TASKS_RE,
        base_intent: Intent::ListTasks,
        extractors: &[],
    },
    PatternDef {
        regex: &TASK_STATUS_RE,
        base_intent: Intent::TaskStatus {
            task_id: String::new(),
        },
        extractors: &[("task_id", 1)],
    },
    PatternDef {
        regex: &CANCEL_TASK_RE,
        base_intent: Intent::CancelTask {
            task_id: String::new(),
        },
        extractors: &[("task_id", 1)],
    },
    PatternDef {
        regex: &CANCEL_SIGNAL_RE,
        base_intent: Intent::CancelSignal {
            signal_id: String::new(),
        },
        extractors: &[("signal_id", 1)],
    },
    PatternDef {
        regex: &SET_PROACTIVITY_RE,
        base_intent: Intent::SetProactivity {
            enabled: true,
            until: None,
        },
        extractors: &[("mode", 1), ("until", 2)],
    },
    PatternDef {
        regex: &PROACTIVITY_STATUS_RE,
        base_intent: Intent::ProactivityStatus,
        extractors: &[],
    },
    PatternDef {
        regex: &DECOMPOSE_TASK_RE,
        base_intent: Intent::DecomposeTask {
            request: String::new(),
        },
        extractors: &[("request", 1)],
    },
    PatternDef {
        regex: &LIST_CHANNELS_RE,
        base_intent: Intent::ListChannels,
        extractors: &[],
    },
    PatternDef {
        regex: &CHANNEL_PREFS_RE,
        base_intent: Intent::ChannelPreferences {
            namespace: None,
            category: None,
        },
        extractors: &[("category", 1)],
    },
    PatternDef {
        regex: &PIN_CHANNEL_RE,
        base_intent: Intent::SetChannelPreference {
            channel: String::new(),
            category: String::new(),
            weight: 0.0,
            pinned: false,
        },
        extractors: &[("verb", 1), ("channel", 2), ("category", 3)],
    },
];

/// Default LLM classification timeout. Generous because free-tier hosted
/// models (e.g. OpenRouter free pool) routinely take 20-40s for the first
/// classification of a session, and falling back to keyword-only is worse
/// than waiting a bit longer.
pub const DEFAULT_LLM_CLASSIFY_TIMEOUT: tokio::time::Duration =
    tokio::time::Duration::from_secs(45);

/// Intent classifier using two-tier approach.
pub struct IntentClassifier {
    patterns: Vec<(IntentPattern, Intent)>,
    llm_fallback: Option<Arc<dyn IntentFallback>>,
    llm_timeout: tokio::time::Duration,
}

struct IntentPattern {
    regex: Regex,
    extractors: HashMap<String, usize>,
}

impl IntentClassifier {
    #[allow(clippy::vec_init_then_push)]
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        for pdef in PATTERNS {
            let extractors: HashMap<String, usize> = pdef
                .extractors
                .iter()
                .map(|(name, idx)| (name.to_string(), *idx))
                .collect();
            let pattern = IntentPattern {
                regex: (**pdef.regex).clone(),
                extractors,
            };
            patterns.push((pattern, pdef.base_intent.clone()));
        }

        Self {
            patterns,
            llm_fallback: None,
            llm_timeout: DEFAULT_LLM_CLASSIFY_TIMEOUT,
        }
    }

    pub fn with_llm_fallback(mut self, fallback: Arc<dyn IntentFallback>) -> Self {
        self.llm_fallback = Some(fallback);
        self
    }

    /// Override the LLM classification timeout. Defaults to
    /// [`DEFAULT_LLM_CLASSIFY_TIMEOUT`]. On timeout the classifier falls
    /// through to keyword-only classification.
    pub fn with_llm_timeout(mut self, timeout: tokio::time::Duration) -> Self {
        self.llm_timeout = timeout;
        self
    }

    pub fn classify_regex(&self, input: &str) -> Option<Classification> {
        for (pattern, base_intent) in &self.patterns {
            if let Some(captures) = pattern.regex.captures(input) {
                let intent = self.extract_intent(base_intent, &captures, &pattern.extractors);
                return Some(Classification {
                    intent,
                    confidence: 0.9,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }

        None
    }

    fn extract_intent(
        &self,
        base: &Intent,
        captures: &regex::Captures,
        extractors: &HashMap<String, usize>,
    ) -> Intent {
        let get_group = |name: &str| -> String {
            extractors
                .get(name)
                .and_then(|&idx| captures.get(idx))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        };

        match base {
            // `StoreFact` / `Forget` are produced by `classify_explicit`,
            // not the regex pipeline — see PATTERNS doc-comment (audit
            // Issues 13 + 14).
            Intent::Recall { .. } => Intent::Recall {
                query: get_group("query"),
            },
            Intent::ExecuteCommand { .. } => {
                let cmd_str = normalize_command(&get_group("command"));
                let parts: Vec<&str> = cmd_str.split_whitespace().collect();
                if parts.is_empty() {
                    Intent::ExecuteCommand {
                        command: String::new(),
                        args: Vec::new(),
                    }
                } else {
                    let command = parts[0].to_string();
                    let args = parts[1..].iter().map(|s| s.to_string()).collect();
                    Intent::ExecuteCommand { command, args }
                }
            }
            Intent::WebSearch { .. } => Intent::WebSearch {
                query: get_group("query"),
            },
            Intent::Schedule { .. } => Intent::Schedule {
                description: get_group("description"),
                cron: None,
            },
            Intent::SendMessage { .. } => Intent::SendMessage {
                channel: get_group("channel").to_lowercase(),
                recipient: get_group("recipient"),
                content: get_group("content"),
            },
            Intent::QueryAgents { .. } => Intent::QueryAgents {
                filter: get_group("filter").trim().to_string(),
            },
            Intent::DelegateTask { .. } => {
                let agent = captures
                    .name("agent")
                    .or_else(|| captures.name("agent2"))
                    .map(|m| m.as_str().trim().to_lowercase())
                    .unwrap_or_default();
                let prompt = captures
                    .name("prompt")
                    .or_else(|| captures.name("prompt2"))
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_default();
                Intent::DelegateTask { agent, prompt }
            }
            Intent::RespondToApproval { .. } => {
                let raw = captures
                    .name("decision")
                    .map(|m| m.as_str().to_lowercase())
                    .unwrap_or_default();
                let decision = match raw.as_str() {
                    "yes" | "y" => "approve".to_string(),
                    "no" | "n" => "reject".to_string(),
                    other => other.to_string(),
                };
                let nonce = captures
                    .name("nonce")
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default();
                Intent::RespondToApproval { nonce, decision }
            }
            Intent::ChannelPreferences { .. } => {
                let cat = get_group("category");
                let category = if cat.is_empty() {
                    None
                } else {
                    Some(normalize_delivery_category(&cat))
                };
                Intent::ChannelPreferences {
                    namespace: None,
                    category,
                }
            }
            Intent::DecomposeTask { .. } => {
                // The PATTERNS entry maps `("request", 1)` but the
                // default arm fell back to the empty base intent. Pull
                // the captured group explicitly so `decompose this …`
                // doesn't reach the orchestrator with an empty request
                // (which then synthesizes a useless "ask for clarification"
                // plan).
                Intent::DecomposeTask {
                    request: get_group("request"),
                }
            }
            Intent::CancelSignal { .. } => Intent::CancelSignal {
                signal_id: get_group("signal_id").trim().to_string(),
            },
            Intent::SetChannelPreference { .. } => {
                let verb = get_group("verb").to_lowercase();
                let channel = get_group("channel");
                let category = normalize_delivery_category(&get_group("category"));
                let (weight, pinned) = match verb.as_str() {
                    "pin" => (1.0, true),
                    "unpin" => (0.0, false),
                    _ => (0.7, false), // "prefer"
                };
                Intent::SetChannelPreference {
                    channel,
                    category,
                    weight,
                    pinned,
                }
            }
            _ => base.clone(),
        }
    }

    fn classify_slash_command(&self, input: &str) -> Option<Classification> {
        if !input.starts_with('/') {
            return None;
        }
        let trimmed = input.trim();
        if trimmed == "/status" {
            return Some(Classification {
                intent: Intent::SystemStatus,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            });
        }

        // Terminal Bridge commands. The split-RPC shape stays out of natural
        // language for now — these slash forms are deterministic and easy to
        // test. LLM-payload paths can route the same intents later.
        if trimmed == "/terminal-list" {
            return Some(Classification {
                intent: Intent::ListTerminalSessions,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            });
        }
        if let Some(rest) = trimmed.strip_prefix("/terminal-open") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some((program, args)) = parts.split_first() {
                return Some(Classification {
                    intent: Intent::OpenTerminalSession {
                        program: (*program).to_string(),
                        args: args.iter().map(|s| (*s).to_string()).collect(),
                        cwd: None,
                    },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }
        if let Some(rest) = trimmed.strip_prefix("/terminal-close") {
            let id = rest.trim();
            if !id.is_empty() {
                return Some(Classification {
                    intent: Intent::CloseTerminalSession {
                        session_id: id.to_string(),
                    },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }

        // MCP host control commands. Slash forms only — full LLM-payload
        // arms are out of scope here and would land alongside a richer
        // natural-language story for MCP control.
        if trimmed == "/mcp-list" {
            return Some(Classification {
                intent: Intent::ListMcpServers,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            });
        }
        // Capability manifest. Slash form drives the
        // `brain capabilities` subcommand and lets a chat user ask "what
        // can you actually do?" deterministically.
        if trimmed == "/capabilities" || trimmed == "/caps" {
            return Some(Classification {
                intent: Intent::ListCapabilities,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            });
        }
        if let Some(rest) = trimmed.strip_prefix("/mcp-mount") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() >= 3 {
                let name = parts[0].to_string();
                let transport = parts[1].to_string();
                let command_or_url = parts[2..].join(" ");
                return Some(Classification {
                    intent: Intent::MountMcpServer {
                        name,
                        transport,
                        command_or_url,
                    },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }
        if let Some(rest) = trimmed.strip_prefix("/mcp-unmount") {
            let name = rest.trim();
            if !name.is_empty() {
                return Some(Classification {
                    intent: Intent::UnmountMcpServer {
                        name: name.to_string(),
                    },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }

        // Standing-approval inspection and revocation. The list path is
        // unguarded (read-only) and the revoke path is Tier::Write —
        // both wired in signal::authz.
        if trimmed == "/approval-list" || trimmed.starts_with("/approval-list ") {
            return Some(Classification {
                intent: Intent::ListStandingApprovals,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            });
        }
        if let Some(rest) = trimmed.strip_prefix("/approval-revoke") {
            let id = rest.trim();
            if !id.is_empty() {
                return Some(Classification {
                    intent: Intent::RevokeStandingApproval { id: id.to_string() },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }

        // Task lifecycle slash forms. Natural-language paths
        // (`cancel task <id>`, `list tasks`, `task status <id>`) still
        // work; these slashes are the deterministic UI / scripting hook,
        // matching the /terminal-* / /mcp-* / /approval-* shapes.
        if trimmed == "/task-list" {
            return Some(Classification {
                intent: Intent::ListTasks,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            });
        }
        if let Some(rest) = trimmed.strip_prefix("/task-status") {
            let id = rest.trim();
            if !id.is_empty() {
                return Some(Classification {
                    intent: Intent::TaskStatus {
                        task_id: id.to_string(),
                    },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }
        if let Some(rest) = trimmed.strip_prefix("/task-cancel") {
            let id = rest.trim();
            if !id.is_empty() {
                return Some(Classification {
                    intent: Intent::CancelTask {
                        task_id: id.to_string(),
                    },
                    confidence: 1.0,
                    method: ClassificationMethod::Regex,
                    extracted_facts: Vec::new(),
                });
            }
        }

        // /tool <verb_ns>.<verb_action> [json-args] — the deterministic
        // power-user / tester entrypoint to the capability router. The verb
        // pair routes against the wired `intent::ToolRegistry`; optional JSON
        // payload becomes the SIT's `object.value`.
        //
        // This is one of two — and only two — `Intent::ToolCall` producers,
        // by design: this explicit user path (`Provenance::User`), and the
        // chat tool-loop where the reasoning LLM proposes calls in-band
        // (`Provenance::Llm`, signal/pipeline/toolloop.rs). The classifier
        // deliberately does NOT try to emit `Intent::ToolCall` from arbitrary
        // free text — open-ended tool selection belongs to the tool-loop,
        // which has the full tool manifest, argument schemas, and a
        // multi-round protocol. A single-shot classifier guess would be a
        // strictly weaker duplicate of that path.
        if let Some(rest) = trimmed.strip_prefix("/tool") {
            let rest = rest.trim();
            if !rest.is_empty() {
                let (verb_str, payload_str) = match rest.split_once(char::is_whitespace) {
                    Some((v, p)) => (v.trim(), p.trim()),
                    None => (rest, ""),
                };
                if let Some((ns, action)) = verb_str.split_once('.') {
                    if !ns.is_empty() && !action.is_empty() {
                        let payload = if payload_str.is_empty() {
                            serde_json::Value::Null
                        } else {
                            match serde_json::from_str::<serde_json::Value>(payload_str) {
                                Ok(v) => v,
                                Err(_) => serde_json::Value::String(payload_str.to_string()),
                            }
                        };
                        let token = intent::IntentToken::new(
                            intent::Verb::new(ns, action),
                            intent::Object {
                                kind: "intent_args".into(),
                                value: payload,
                            },
                            intent::Provenance::User {
                                raw_input: trimmed.to_string(),
                                ui_origin: None,
                                ts: chrono::Utc::now(),
                            },
                            "personal".into(),
                        );
                        return Some(Classification {
                            intent: Intent::ToolCall(Box::new(token)),
                            confidence: 1.0,
                            method: ClassificationMethod::Regex,
                            extracted_facts: Vec::new(),
                        });
                    }
                }
            }
        }
        None
    }

    pub async fn classify(&self, input: &str) -> Classification {
        self.classify_with_history(input, &[]).await
    }

    /// History-aware classification. The regex fast-paths and explicit
    /// prefix matches don't need history (they're unambiguous on their
    /// own), but the LLM fallback does — that's how it can tell a
    /// follow-up parameter ("username : foo") from a self-introduction.
    pub async fn classify_with_history(
        &self,
        input: &str,
        history: &[cortex::llm::Message],
    ) -> Classification {
        if input.starts_with('/') {
            if let Some(c) = self.classify_slash_command(input) {
                return c;
            }
        }

        if let Some(c) = self.classify_explicit(input) {
            return c;
        }

        if let Some(classification) = self.classify_regex(input) {
            return classification;
        }

        if let Some(fallback) = &self.llm_fallback {
            let timeout = self.llm_timeout;
            match tokio::time::timeout(timeout, fallback.classify_with_history(input, history))
                .await
            {
                Ok(Some(classification)) => return classification,
                Ok(None) => {
                    tracing::warn!("LLM classifier returned None (error or parse failure)");
                }
                Err(_) => {
                    tracing::warn!(
                        "LLM intent classification timed out ({}s)",
                        timeout.as_secs()
                    );
                }
            }
        }

        Classification {
            intent: Intent::Chat {
                content: input.to_string(),
            },
            confidence: 1.0,
            method: ClassificationMethod::Fallback,
            extracted_facts: Vec::new(),
        }
    }

    fn classify_explicit(&self, input: &str) -> Option<Classification> {
        let trimmed = input.trim();
        let lower = trimmed.to_lowercase();

        let forget_prefixes = ["forget ", "delete ", "remove "];
        for prefix in &forget_prefixes {
            if lower.starts_with(prefix) {
                let rest = &trimmed[prefix.len()..];
                let target = if rest.to_lowercase().starts_with("about ") {
                    rest[6..].trim()
                } else {
                    rest.trim()
                };
                if !target.is_empty() {
                    return Some(Classification {
                        intent: Intent::Forget {
                            target: target.to_string(),
                        },
                        confidence: 1.0,
                        method: ClassificationMethod::Regex,
                        extracted_facts: Vec::new(),
                    });
                }
            }
        }

        let store_prefixes = ["remember ", "note ", "keep in mind "];
        for prefix in &store_prefixes {
            if lower.starts_with(prefix) {
                let rest = &trimmed[prefix.len()..];
                let content = if rest.to_lowercase().starts_with("that ") {
                    rest[5..].trim()
                } else {
                    rest.trim()
                };
                if !content.is_empty() {
                    return Some(Classification {
                        intent: Intent::StoreFact {
                            subject: "user".to_string(),
                            predicate: "said".to_string(),
                            object: content.to_string(),
                        },
                        confidence: 1.0,
                        method: ClassificationMethod::Regex,
                        extracted_facts: Vec::new(),
                    });
                }
            }
        }

        None
    }
}

/// Lower-case + trim a user-typed category token. Plural and unknown
/// forms are passed through; downstream `channel::DeliveryCategory::parse`
/// is lenient and handles both singular and plural, so this only needs
/// to normalize whitespace and case.
fn normalize_delivery_category(raw: &str) -> String {
    raw.trim().to_lowercase()
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}
