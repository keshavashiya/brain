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
    /// Decompose a complex request into an executable task plan.
    DecomposeTask { request: String },
    /// Ask about available specialist agents (delegates). Optional
    /// `filter` narrows the answer: e.g. "rust", "aider", or "".
    QueryAgents { filter: String },
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
Valid intents: store_fact, recall, forget, execute_command, web_search, schedule, send_message, system_status, decompose_task, query_agents, chat.
Rules:
- recall is for memory queries: "what do you know about...", "what did we discuss", "what do you remember about...", "tell me about...", "what is my...", "do you remember...", "tell me everything about...". These ask about the user's stored memories.
- Questions that are NOT about stored memories (general knowledge, opinions, how-to questions) are chat.
- Questions should NEVER be execute_command.
- store_fact is ONLY for explicit memory requests: "remember that ...", "note that ...", "keep in mind ...".
- execute_command is ONLY for explicit requests like "run ls", "execute cargo build". The command field must be a real shell command (ls, git, cargo, etc.).
- decompose_task is for multi-step requests that need planning and execution: "build a CSV export feature", "set up CI/CD pipeline", "refactor the auth module and add tests", "deploy to production". The request must involve multiple steps or coordination. Simple single-step requests are NOT decompose_task.
- query_agents is for asking which specialist agents are available or why a named agent is unavailable: "what agents do you have", "which agents can code rust", "why aren't you using aider".
- Conversational statements ("I've done X", "I completed X", "I like X") are chat but ALSO extract any personal facts (see below).
- Prefer web_search for explicit search requests about internet/google/latest/current external info.
- For web_search, set 'query' to the exact optimal search terms, stripping conversational fluff.
- Use system_status only for explicit status checks like "/status".
- Use chat when uncertain or for general conversation.

FACT EXTRACTION: Regardless of intent, if the input contains personal facts about the user (name, role, company, projects, skills, interests, goals, location, preferences, habits), extract them into the "facts" array. Each fact is {"subject": "user", "predicate": "<snake_case_verb>", "object": "<value>"}.
Predicates: name_is, role_is, works_at, works_on, title_is, interested_in, lives_in, skill_is, goal_is, preference_is, likes, etc.
Only extract clear factual statements. If no facts, set facts to [].

Return only JSON with keys: intent, subject, predicate, object, query, target, command, args, description, cron, channel, recipient, content, facts.
Missing keys must be null. facts must be [] if none."#;

#[async_trait::async_trait]
impl IntentFallback for LlmIntentFallback {
    async fn classify_with_llm(&self, input: &str) -> Option<Classification> {
        use cortex::llm::{Message, Role};

        let messages = vec![
            Message {
                role: Role::System,
                content: CLASSIFIER_SYSTEM_PROMPT.to_string(),
            },
            Message {
                role: Role::User,
                content: input.to_string(),
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
            "decompose_task" => Intent::DecomposeTask {
                request: payload
                    .content
                    .or(payload.description)
                    .unwrap_or_else(|| input.to_string()),
            },
            "query_agents" => Intent::QueryAgents {
                filter: payload.query.unwrap_or_default(),
            },
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
