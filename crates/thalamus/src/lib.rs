//! # Brain Thalamus
//!
//! Signal router — first point of contact for all input.
//! Classifies intent using a two-tier approach:
//! 1. Regex fast-path for obvious intents (0ms)
//! 2. LLM fallback for ambiguous input (~300ms)
//!
//! Routes messages to the appropriate subsystem based on intent.

use cortex::actions::Action;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

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
    /// Schedule something.
    Schedule {
        description: String,
        cron: Option<String>,
    },
    /// Send via a channel.
    SendMessage {
        channel: String,
        recipient: String,
        content: String,
    },
    /// Get system status.
    SystemStatus,
    /// Regular chat/conversation.
    Chat { content: String },
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
}

#[derive(Debug, Deserialize)]
struct LlmIntentPayload {
    intent: String,
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
    query: Option<String>,
    target: Option<String>,
    command: Option<String>,
    args: Option<Vec<String>>,
    description: Option<String>,
    cron: Option<String>,
    channel: Option<String>,
    recipient: Option<String>,
    content: Option<String>,
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
        let trimmed = raw.trim();
        if let Ok(payload) = serde_json::from_str::<LlmIntentPayload>(trimmed) {
            return Some(payload);
        }

        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        serde_json::from_str::<LlmIntentPayload>(&trimmed[start..=end]).ok()
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

#[async_trait::async_trait]
impl IntentFallback for LlmIntentFallback {
    async fn classify_with_llm(&self, input: &str) -> Option<Classification> {
        use cortex::llm::{Message, Role};

        let prompt = format!(
            "Classify the user input into exactly one intent for Brain OS.\n\
             Valid intents: store_fact, recall, forget, execute_command, web_search, schedule, send_message, system_status, chat.\n\
             Rules:\n\
             - Questions (how/what/why/who/when/where/can/could/is/are) are ALWAYS chat. Never classify a question as execute_command.\n\
             - execute_command is ONLY for explicit requests like \"run ls\", \"execute cargo build\". The command field must be a real shell command (ls, git, cargo, etc.).\n\
             - Conversational statements (\"I've done X\", \"I completed X\", \"I like X\") are chat — but ALSO extract any personal facts (see below).\n\
             - store_fact is ONLY for explicit memory requests like \"remember that ...\", \"note that ...\", \"keep in mind ...\".\n\
             - recall is ONLY for explicit memory queries like \"what did we discuss\", \"recall ...\", \"what do you remember about ...\".\n\
             - Prefer web_search for explicit search requests about internet/google/latest/current external info.\n\
             - For web_search, set 'query' to the exact optimal search terms, stripping conversational fluff.\n\
             - Use system_status only for explicit status checks like \"/status\".\n\
             - Use chat when uncertain or for general conversation.\n\n\
             FACT EXTRACTION: Regardless of intent, if the input contains personal facts about the user \
             (name, role, company, projects, skills, interests, goals, location, preferences, habits), \
             extract them into the \"facts\" array. Each fact is {{\"subject\": \"user\", \"predicate\": \"<snake_case_verb>\", \"object\": \"<value>\"}}.\n\
             Predicates: name_is, role_is, works_at, works_on, title_is, interested_in, lives_in, skill_is, goal_is, preference_is, likes, etc.\n\
             Only extract clear factual statements. If no facts, set facts to [].\n\n\
             Return only JSON with keys: intent, subject, predicate, object, query, target, command, args, description, cron, channel, recipient, content, facts.\n\
             Missing keys must be null. facts must be [] if none.\n\
             Input: {input}"
        );

        let messages = vec![Message {
            role: Role::User,
            content: prompt,
        }];

        let response = match self.llm.generate(&messages).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("LLM intent classification failed: {e}");
                return None;
            }
        };

        tracing::debug!(raw_len = response.content.len(), "LLM classifier raw response");

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
                if !payload.args.clone().unwrap_or_default().is_empty() {
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

// ─── Message Types ──────────────────────────────────────────────────────────

/// Normalized message format for all channels.
#[derive(Debug, Clone)]
pub struct NormalizedMessage {
    /// Message content/text.
    pub content: String,
    /// Channel it came from (cli, whatsapp, telegram, etc.)
    pub channel: String,
    /// Sender identifier.
    pub sender: String,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Original message ID (if any).
    pub message_id: Option<String>,
    /// Metadata (channel-specific).
    #[allow(dead_code)]
    pub metadata: HashMap<String, String>,
}

// ─── Intent Classifier ─────────────────────────────────────────────────────

/// Intent classifier using two-tier approach.
pub struct IntentClassifier {
    patterns: Vec<(IntentPattern, Intent)>,
    llm_fallback: Option<Arc<dyn IntentFallback>>,
}

/// A regex pattern that maps to an intent.
struct IntentPattern {
    regex: Regex,
    extractors: HashMap<String, usize>,
}

impl IntentClassifier {
    /// Create a new classifier with built-in patterns.
    #[allow(clippy::vec_init_then_push)]
    pub fn new() -> Self {
        let mut patterns = Vec::new();

        // Store fact patterns
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:remember|note|keep in mind)\s+(?:that\s+)?(.+?)$",
                &[("content", 1)],
            ),
            Intent::StoreFact {
                subject: String::new(),
                predicate: String::new(),
                object: String::new(),
            },
        ));

        // Recall patterns
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:what did|recall|remember)\s+(.+?)\??$",
                &[("query", 1)],
            ),
            Intent::Recall {
                query: String::new(),
            },
        ));

        // Forget patterns
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:forget|delete|remove)\s+(?:about\s+)?(.+?)$",
                &[("target", 1)],
            ),
            Intent::Forget {
                target: String::new(),
            },
        ));

        // Execute command patterns
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:run|execute|do)\s+(?:command\s+)?(.+?)$",
                &[("command", 1)],
            ),
            Intent::ExecuteCommand {
                command: String::new(),
                args: Vec::new(),
            },
        ));

        // Web search patterns
        // Note: "find" requires "about/for/info" to avoid false positives like "find the bug"
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:(?:can you|could you|please|will you|would you)\s+)?(?:search|look up|google|web search|look for)\s+(?:for\s+|about\s+|up\s+)?(.+?)(?:\?)?$",
                &[("query", 1)],
            ),
            Intent::WebSearch {
                query: String::new(),
            },
        ));
        // "find" only triggers web search when paired with "about", "for", or "information"
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:(?:can you|could you|please|will you|would you)\s+)?find\s+(?:information\s+)?(?:about|for)\s+(.+?)(?:\?)?$",
                &[("query", 1)],
            ),
            Intent::WebSearch {
                query: String::new(),
            },
        ));

        // Schedule patterns
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:remind me|schedule|set reminder)\s+(?:to\s+)?(.+?)$",
                &[("description", 1)],
            ),
            Intent::Schedule {
                description: String::new(),
                cron: None,
            },
        ));

        // Send message patterns
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:send|message|text)\s+(?:via\s+(\w+)\s+)?(?:to\s+)?(.+?)\s+(?:saying\s+|:\s+)(.+?)$",
                &[("channel", 1), ("recipient", 2), ("content", 3)],
            ),
            Intent::SendMessage {
                channel: String::new(),
                recipient: String::new(),
                content: String::new(),
            },
        ));

        // Status patterns
        patterns.push((
            Self::build_pattern(r"(?i)^/status$", &[]),
            Intent::SystemStatus,
        ));

        Self {
            patterns,
            llm_fallback: None,
        }
    }

    /// Attach an optional async fallback classifier.
    pub fn with_llm_fallback(mut self, fallback: Arc<dyn IntentFallback>) -> Self {
        self.llm_fallback = Some(fallback);
        self
    }

    /// Build a pattern with named extractors.
    fn build_pattern(pattern: &str, extractors: &[(&str, usize)]) -> IntentPattern {
        IntentPattern {
            regex: Regex::new(pattern).expect("Invalid regex pattern"),
            extractors: extractors
                .iter()
                .map(|(name, idx)| (name.to_string(), *idx))
                .collect(),
        }
    }

    /// Classify input using regex patterns (fallback when LLM is unavailable).
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

    /// Extract intent with captured groups.
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
                // Regex fallback: use simple defaults. LLM-primary path
                // provides proper subject/predicate/object extraction.
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
            _ => base.clone(),
        }
    }

/// Classify slash commands (deterministic, not NLU).
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

    /// Classify input using a layered strategy.
    ///
    /// Order: slash commands → explicit fast-path → regex → LLM → Chat default.
    ///
    /// Regex patterns catch known intent shapes deterministically (0ms, no LLM
    /// needed). The LLM handles everything else — ambiguous natural language,
    /// conversational fact extraction, and nuanced intent detection that regex
    /// can't handle. This avoids small-model misclassification for patterns
    /// we already know how to match.
    pub async fn classify(&self, input: &str) -> Classification {
        // 1. Slash commands (deterministic, 0ms)
        if input.starts_with('/') {
            if let Some(c) = self.classify_slash_command(input) {
                return c;
            }
        }

        // 2. Unambiguous explicit commands (forget X, remember X)
        if let Some(c) = self.classify_explicit(input) {
            return c;
        }

        // 3. Regex patterns — known intent shapes (schedule, web search, etc.)
        if let Some(classification) = self.classify_regex(input) {
            return classification;
        }

        // 4. LLM classification — handles ambiguous / conversational input
        //    and extracts facts from natural language.
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

        // 5. Default to chat
        Classification {
            intent: Intent::Chat {
                content: input.to_string(),
            },
            confidence: 1.0,
            method: ClassificationMethod::Fallback,
            extracted_facts: Vec::new(),
        }
    }

    /// Fast-path for unambiguous explicit commands that don't need LLM.
    ///
    /// Matches only clear imperative patterns like "forget X", "remember X",
    /// "recall X" where the leading verb unambiguously signals intent.
    /// Preserves original case in extracted content.
    fn classify_explicit(&self, input: &str) -> Option<Classification> {
        let trimmed = input.trim();
        let lower = trimmed.to_lowercase();

        // "forget X" / "delete X" / "remove X" → Forget
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

        // "remember (that) X" / "note (that) X" / "keep in mind (that) X" → StoreFact
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

// ─── Signal Router ─────────────────────────────────────────────────────────

/// Routes normalized messages to appropriate handlers.
pub struct SignalRouter {
    classifier: IntentClassifier,
}

impl SignalRouter {
    /// Create a new router.
    pub fn new() -> Self {
        Self {
            classifier: IntentClassifier::new(),
        }
    }

    /// Attach LLM intent classification to this router.
    pub fn with_llm_fallback(mut self, fallback: Arc<dyn IntentFallback>) -> Self {
        self.classifier = self.classifier.with_llm_fallback(fallback);
        self
    }

    /// Route a message and return the classified intent.
    pub async fn route(&self, message: &NormalizedMessage) -> Classification {
        self.classifier.classify(&message.content).await
    }

    /// Convert intent to action (for action intents).
    ///
    /// Returns `None` for intents handled directly in SignalProcessor:
    /// - `Forget` — deletes matching facts (handled in SignalProcessor::process)
    /// - `SystemStatus` — returns memory counts (handled in SignalProcessor::process)
    /// - `Chat` / `Recall` — routed to LLM pipeline (not dispatched as actions)
    pub fn intent_to_action(&self, intent: &Intent) -> Option<Action> {
        match intent {
            Intent::StoreFact {
                subject,
                predicate,
                object,
            } => Some(Action::StoreFact {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object: object.clone(),
            }),
            Intent::Recall { query } => Some(Action::Recall {
                query: query.clone(),
            }),
            Intent::ExecuteCommand { command, args } => Some(Action::ExecuteCommand {
                command: command.clone(),
                args: args.clone(),
            }),
            Intent::WebSearch { query } => Some(Action::WebSearch {
                query: query.clone(),
            }),
            Intent::Schedule { description, cron } => Some(Action::ScheduleTask {
                description: description.clone(),
                cron: cron.clone(),
            }),
            Intent::SendMessage {
                channel,
                recipient,
                content,
            } => Some(Action::SendMessage {
                channel: channel.clone(),
                recipient: recipient.clone(),
                content: content.clone(),
            }),
            _ => None,
        }
    }
}

impl Default for SignalRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock IntentFallback that returns a fixed classification for testing.
    struct MockFallback {
        response: Option<Classification>,
    }

    impl MockFallback {
        fn chat() -> Self {
            Self {
                response: Some(Classification {
                    intent: Intent::Chat {
                        content: "mock".to_string(),
                    },
                    confidence: 0.7,
                    method: ClassificationMethod::Llm,
                    extracted_facts: Vec::new(),
                }),
            }
        }

        fn unavailable() -> Self {
            Self { response: None }
        }
    }

    #[async_trait::async_trait]
    impl IntentFallback for MockFallback {
        async fn classify_with_llm(&self, _input: &str) -> Option<Classification> {
            self.response.clone()
        }
    }

    // ── Regex fallback tests (no LLM attached) ───────────────────────────────

    #[tokio::test]
    async fn test_classify_store_fact_regex_fallback() {
        // Without LLM, regex kicks in as fallback
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Remember that I like coffee").await;

        assert!(
            matches!(result.intent, Intent::StoreFact { .. }),
            "Expected StoreFact, got {:?}",
            result.intent
        );
        assert_eq!(result.method, ClassificationMethod::Regex);
    }

    #[tokio::test]
    async fn test_classify_recall_regex_fallback() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What did we discuss yesterday?").await;

        assert!(
            matches!(result.intent, Intent::Recall { .. }),
            "Expected Recall, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_execute_command_regex_fallback() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Run ls -la").await;

        assert!(
            matches!(result.intent, Intent::ExecuteCommand { .. }),
            "Expected ExecuteCommand, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_web_search_regex_fallback() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Search for Rust programming").await;

        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_web_search_natural_phrasing_regex_fallback() {
        let classifier = IntentClassifier::new();

        let result = classifier
            .classify("can you search about Keshav Ashiya")
            .await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'can you search about ...', got {:?}",
            result.intent
        );

        let result = classifier.classify("please look up Rust language").await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'please look up ...', got {:?}",
            result.intent
        );

        let result = classifier
            .classify("could you find information about AI")
            .await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'could you find ...', got {:?}",
            result.intent
        );

        let result = classifier.classify("google Keshav Ashiya").await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'google ...', got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_schedule_regex_fallback() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Remind me to call mom").await;

        assert!(
            matches!(result.intent, Intent::Schedule { .. }),
            "Expected Schedule, got {:?}",
            result.intent
        );
    }

    // ── Slash command tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_classify_status_slash_command() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("/status").await;

        assert_eq!(result.intent, Intent::SystemStatus);
        // Slash command should be classified before LLM or regex
        assert_eq!(result.method, ClassificationMethod::Regex);
    }

    // ── Chat fallback tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_classify_chat_fallback() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Hello, how are you?").await;

        assert!(
            matches!(result.intent, Intent::Chat { .. }),
            "Expected Chat, got {:?}",
            result.intent
        );
        assert_eq!(result.method, ClassificationMethod::Fallback);
    }

    // ── LLM-first tests (with mock) ─────────────────────────────────────────

    #[tokio::test]
    async fn test_llm_classifies_ambiguous_input() {
        // For input that doesn't match regex patterns, LLM should classify
        let classifier = IntentClassifier::new()
            .with_llm_fallback(Arc::new(MockFallback::chat()));

        // "What's the weather?" doesn't match any regex pattern → LLM classifies as Chat
        let result = classifier.classify("What's the weather?").await;
        assert_eq!(result.method, ClassificationMethod::Llm);
        assert!(
            matches!(result.intent, Intent::Chat { .. }),
            "LLM should classify ambiguous input; got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_regex_fallback_when_llm_unavailable() {
        // LLM returns None → regex should kick in
        let classifier = IntentClassifier::new()
            .with_llm_fallback(Arc::new(MockFallback::unavailable()));

        let result = classifier.classify("Remember that I like coffee").await;
        assert_eq!(result.method, ClassificationMethod::Regex);
        assert!(
            matches!(result.intent, Intent::StoreFact { .. }),
            "Regex fallback should work; got {:?}",
            result.intent
        );
    }

    // ── Negative tests: false positive prevention ────────────────────────────

    #[tokio::test]
    async fn test_do_you_remember_is_not_store_fact() {
        // "Do you remember my birthday?" should NOT match StoreFact regex
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Do you remember my birthday?").await;

        assert!(
            !matches!(result.intent, Intent::StoreFact { .. }),
            "'Do you remember...' should NOT be StoreFact, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_find_the_bug_is_not_web_search() {
        // "Find the bug in my code" should NOT match WebSearch regex
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Find the bug in my code").await;

        assert!(
            !matches!(result.intent, Intent::WebSearch { .. }),
            "'Find the bug...' should NOT be WebSearch, got {:?}",
            result.intent
        );
    }

    // ── Router tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_intent_to_action_store_fact() {
        let router = SignalRouter::new();
        let intent = Intent::StoreFact {
            subject: "user".to_string(),
            predicate: "likes".to_string(),
            object: "coffee".to_string(),
        };

        let action = router.intent_to_action(&intent);
        assert!(
            matches!(action, Some(Action::StoreFact { .. })),
            "Expected StoreFact action"
        );
    }

    #[test]
    fn test_intent_to_action_system_status() {
        let router = SignalRouter::new();
        let intent = Intent::SystemStatus;

        let action = router.intent_to_action(&intent);
        assert!(action.is_none(), "SystemStatus should not map to action");
    }

    #[test]
    fn test_regex_classification_has_empty_extracted_facts() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify_regex("Remember that I like coffee");
        assert!(
            result.unwrap().extracted_facts.is_empty(),
            "Regex classification should have empty extracted_facts"
        );
    }

    #[test]
    fn test_fallback_classification_has_empty_extracted_facts() {
        let classifier = IntentClassifier::new();
        let result = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(classifier.classify("Hello, how are you?"));
        assert!(
            result.extracted_facts.is_empty(),
            "Fallback classification should have empty extracted_facts"
        );
    }

    #[test]
    fn test_parse_json_payload_with_facts() {
        let json = r#"{
            "intent": "chat",
            "content": "I'm Keshav, a software engineer",
            "facts": [
                {"subject": "user", "predicate": "name_is", "object": "Keshav"},
                {"subject": "user", "predicate": "role_is", "object": "software engineer"}
            ]
        }"#;

        let payload = LlmIntentFallback::parse_json_payload(json).unwrap();
        assert_eq!(payload.intent, "chat");
        let facts = payload.facts.unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].predicate.as_deref(), Some("name_is"));
        assert_eq!(facts[0].object.as_deref(), Some("Keshav"));
        assert_eq!(facts[1].predicate.as_deref(), Some("role_is"));
        assert_eq!(facts[1].object.as_deref(), Some("software engineer"));
    }

    #[test]
    fn test_parse_json_payload_with_empty_facts() {
        let json = r#"{"intent": "chat", "content": "hello", "facts": []}"#;
        let payload = LlmIntentFallback::parse_json_payload(json).unwrap();
        assert!(payload.facts.unwrap().is_empty());
    }

    #[test]
    fn test_parse_json_payload_without_facts_field() {
        let json = r#"{"intent": "chat", "content": "hello"}"#;
        let payload = LlmIntentFallback::parse_json_payload(json).unwrap();
        assert!(payload.facts.is_none());
    }

    #[test]
    fn test_extracted_fact_filters_empty_fields() {
        let raw_facts = vec![
            LlmFactPayload {
                subject: Some("user".to_string()),
                predicate: Some("name_is".to_string()),
                object: Some("Keshav".to_string()),
            },
            LlmFactPayload {
                subject: Some("user".to_string()),
                predicate: Some("".to_string()), // empty predicate
                object: Some("something".to_string()),
            },
            LlmFactPayload {
                subject: Some("user".to_string()),
                predicate: Some("likes".to_string()),
                object: None, // missing object
            },
        ];

        let extracted: Vec<ExtractedFact> = raw_facts
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

        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].predicate, "name_is");
        assert_eq!(extracted[0].object, "Keshav");
    }
}
