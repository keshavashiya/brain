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

/// Classification result.
#[derive(Debug, Clone)]
pub struct Classification {
    pub intent: Intent,
    pub confidence: f64,
    pub method: ClassificationMethod,
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
             - Conversational statements (\"I've done X\", \"I completed X\", \"I like X\") are chat, NOT execute_command or store_fact.\n\
             - store_fact is ONLY for explicit memory requests like \"remember that ...\", \"note that ...\", \"keep in mind ...\".\n\
             - recall is ONLY for explicit memory queries like \"what did we discuss\", \"recall ...\", \"what do you remember about ...\".\n\
             - Prefer web_search for explicit search requests about internet/google/latest/current external info.\n\
             - For web_search, set 'query' to the exact optimal search terms, stripping conversational fluff.\n\
             - Use system_status only for explicit status checks like \"/status\".\n\
             - Use chat when uncertain or for general conversation.\n\
             Return only JSON with keys: intent, subject, predicate, object, query, target, command, args, description, cron, channel, recipient, content.\n\
             Missing keys must be null.\n\
             Input: {input}"
        );

        let messages = vec![Message {
            role: Role::User,
            content: prompt,
        }];

        let response = self.llm.generate(&messages).await.ok()?;
        let payload = Self::parse_json_payload(&response.content)?;
        let key = payload.intent.to_ascii_lowercase();

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

        Some(Classification {
            intent,
            confidence: 0.7,
            method: ClassificationMethod::Llm,
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
        patterns.push((
            Self::build_pattern(
                r"(?i)^(?:(?:can you|could you|please|will you|would you)\s+)?(?:search|look up|find|google|web search|look for)\s+(?:for\s+|about\s+|up\s+)?(.+?)(?:\?)?$",
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

    /// Classify input using regex patterns (fast path).
    pub fn classify_fast(&self, input: &str) -> Option<Classification> {
        for (pattern, base_intent) in &self.patterns {
            if let Some(captures) = pattern.regex.captures(input) {
                let intent = self.extract_intent(base_intent, &captures, &pattern.extractors);
                return Some(Classification {
                    intent,
                    confidence: 0.9,
                    method: ClassificationMethod::Regex,
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
                let (subject, predicate, object) = Self::parse_store_fact_content(&content);
                Intent::StoreFact {
                    subject,
                    predicate,
                    object,
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

    /// Deterministically parse free-form text into a store-fact tuple.
    pub fn parse_store_fact_content(content: &str) -> (String, String, String) {
        let parts: Vec<&str> = content.splitn(3, ' ').collect();
        if parts.len() >= 3 {
            (
                parts[0].to_string(),
                parts[1].to_string(),
                parts[2].to_string(),
            )
        } else {
            ("user".to_string(), "said".to_string(), content.to_string())
        }
    }

    /// Classify with regex-first, LLM-fallback strategy.
    ///
    /// Regex patterns catch clear, structured commands deterministically (0ms).
    /// When no regex matches, the LLM classifies ambiguous input (~300ms–5s).
    /// This ensures reliable routing for known patterns while handling
    /// natural language variations through the LLM.
    pub async fn classify(&self, input: &str) -> Classification {
        // 1. Deterministic fast path — regex patterns for clear commands.
        if let Some(classification) = self.classify_fast(input) {
            return classification;
        }

        // 2. LLM fallback — classifies ambiguous/natural language input.
        if let Some(fallback) = &self.llm_fallback {
            let timeout = tokio::time::Duration::from_millis(5000);
            if let Ok(Some(classification)) =
                tokio::time::timeout(timeout, fallback.classify_with_llm(input)).await
            {
                return classification;
            }
        }

        // 3. Default to chat
        Classification {
            intent: Intent::Chat {
                content: input.to_string(),
            },
            confidence: 1.0,
            method: ClassificationMethod::Fallback,
        }
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

    #[tokio::test]
    async fn test_classify_store_fact() {
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
    async fn test_classify_recall() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("What did we discuss yesterday?").await;

        assert!(
            matches!(result.intent, Intent::Recall { .. }),
            "Expected Recall, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_execute_command() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Run ls -la").await;

        assert!(
            matches!(result.intent, Intent::ExecuteCommand { .. }),
            "Expected ExecuteCommand, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_web_search() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Search for Rust programming").await;

        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_web_search_natural_phrasing() {
        let classifier = IntentClassifier::new();

        // "can you search about ..."
        let result = classifier
            .classify("can you search about Keshav Ashiya")
            .await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'can you search about ...', got {:?}",
            result.intent
        );

        // "please look up ..."
        let result = classifier.classify("please look up Rust language").await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'please look up ...', got {:?}",
            result.intent
        );

        // "could you find ..."
        let result = classifier
            .classify("could you find information about AI")
            .await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'could you find ...', got {:?}",
            result.intent
        );

        // "google ..."
        let result = classifier.classify("google Keshav Ashiya").await;
        assert!(
            matches!(result.intent, Intent::WebSearch { .. }),
            "Expected WebSearch for 'google ...', got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_schedule() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("Remind me to call mom").await;

        assert!(
            matches!(result.intent, Intent::Schedule { .. }),
            "Expected Schedule, got {:?}",
            result.intent
        );
    }

    #[tokio::test]
    async fn test_classify_status() {
        let classifier = IntentClassifier::new();
        let result = classifier.classify("/status").await;

        assert_eq!(result.intent, Intent::SystemStatus);
    }

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
}
