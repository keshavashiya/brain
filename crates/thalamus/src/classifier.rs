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

static STORE_FACT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:remember|note|keep in mind)\s+(?:that\s+)?(.+?)$")
        .expect("invariant: STORE_FACT_RE must be valid")
});

static RECALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:what did|recall|remember)\s+(.+?)\??$")
        .expect("invariant: RECALL_RE must be valid")
});

static FORGET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:forget|delete|remove)\s+(?:about\s+)?(.+?)$")
        .expect("invariant: FORGET_RE must be valid")
});

static EXECUTE_COMMAND_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:run|exec|execute)\s+(?:command\s+)?(\S+)(?:\s+(.*))?$")
        .expect("invariant: EXECUTE_COMMAND_RE must be valid")
});

static WEB_SEARCH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:(?:can you|could you|please|will you|would you)\s+)?(?:search|look up|google|web search|look for)\s+(?:for\s+|about\s+|up\s+)?(.+?)(?:\?)?$")
        .expect("invariant: WEB_SEARCH_RE (main) must be valid")
});

static WEB_SEARCH_FIND_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:(?:Can you|Could you|please|will you|would you)\s+)?find\s+(?:information\s+)?(?:about|for)\s+(.+?)(?:\?)?$")
        .expect("invariant: WEB_SEARCH_FIND_RE must be valid")
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
    Regex::new(r"(?i)^(approve|reject)\s+([a-zA-Z0-9]+)$")
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

/// All patterns in priority order — first match wins.
pub(crate) const PATTERNS: &[PatternDef] = &[
    PatternDef {
        regex: &STORE_FACT_RE,
        base_intent: Intent::StoreFact {
            subject: String::new(),
            predicate: String::new(),
            object: String::new(),
        },
        extractors: &[("content", 1)],
    },
    PatternDef {
        regex: &RECALL_RE,
        base_intent: Intent::Recall {
            query: String::new(),
        },
        extractors: &[("query", 1)],
    },
    PatternDef {
        regex: &FORGET_RE,
        base_intent: Intent::Forget {
            target: String::new(),
        },
        extractors: &[("target", 1)],
    },
    PatternDef {
        regex: &EXECUTE_COMMAND_RE,
        base_intent: Intent::ExecuteCommand {
            command: String::new(),
            args: Vec::new(),
        },
        extractors: &[("command", 1), ("args", 2)],
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
        extractors: &[("decision", 1), ("nonce", 2)],
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
];

/// Intent classifier using two-tier approach.
pub struct IntentClassifier {
    patterns: Vec<(IntentPattern, Intent)>,
    llm_fallback: Option<Arc<dyn IntentFallback>>,
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
        }
    }

    pub fn with_llm_fallback(mut self, fallback: Arc<dyn IntentFallback>) -> Self {
        self.llm_fallback = Some(fallback);
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
            Intent::StoreFact { .. } => {
                let content = get_group("content");
                Intent::StoreFact {
                    subject: "user".to_string(),
                    predicate: "said".to_string(),
                    object: content,
                }
            }
            Intent::Recall { .. } => Intent::Recall {
                query: get_group("query"),
            },
            Intent::Forget { .. } => Intent::Forget {
                target: get_group("target"),
            },
            Intent::ExecuteCommand { .. } => {
                let cmd_str = get_group("command");
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
            _ => base.clone(),
        }
    }

    fn classify_slash_command(&self, input: &str) -> Option<Classification> {
        if !input.starts_with('/') {
            return None;
        }
        match input.trim() {
            "/status" => Some(Classification {
                intent: Intent::SystemStatus,
                confidence: 1.0,
                method: ClassificationMethod::Regex,
                extracted_facts: Vec::new(),
            }),
            _ => None,
        }
    }

    pub async fn classify(&self, input: &str) -> Classification {
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
            let timeout = tokio::time::Duration::from_millis(15000);
            match tokio::time::timeout(timeout, fallback.classify_with_llm(input)).await {
                Ok(Some(classification)) => return classification,
                Ok(None) => {
                    tracing::warn!("LLM classifier returned None (error or parse failure)");
                }
                Err(_) => {
                    tracing::warn!("LLM intent classification timed out (15s)");
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

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}
