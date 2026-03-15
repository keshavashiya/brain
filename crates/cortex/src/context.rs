//! Context assembly — builds LLM prompts with token budget management.
//!
//! Manages the token budget for LLM context windows:
//! - System prompt (~500 tokens)
//! - User model snapshot (~300 tokens)
//! - Conversation history (~2000 tokens)
//! - Retrieved memories (remaining budget)
//! - Response buffer (~400 tokens)

use crate::llm::{Message, Role};
use hippocampus::search::Memory;

/// Default token budgets.
pub const TOKEN_BUDGETS: TokenBudget = TokenBudget {
    system_prompt: 500,
    user_model: 300,
    conversation_history: 2000,
    response_buffer: 400,
    total_context: 8192, // Default for most models
};

/// Hardcoded greeting for first-ever chat session (0 facts).
/// Printed directly — no LLM call needed.
pub const ONBOARDING_GREETING: &str = "Hey! I'm Brain \u{2014} your personal memory engine. \
I run locally on your machine and I'm here to remember what matters to you. \
I don't know anything about you yet, so let's fix that. What's your name?";

/// System-prompt addendum injected while the user has fewer than 5 facts.
/// Makes the LLM naturally curious and question-asking during onboarding.
pub const ONBOARDING_ADDENDUM: &str = r#"

[ONBOARDING MODE — the user is new and you know very little about them]
- After every user message, end your response with ONE short, focused follow-up question to learn about the user (name, role, projects, interests).
- Keep responses to 1-3 sentences plus the question.
- Sound warm, curious, and conversational — not like an intake form.
- NEVER say "I don't have that in my memory yet" — instead, be proactive about learning.
- Once you learn something, acknowledge it naturally and ask about the next thing."#;

/// Token budget allocation.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    pub system_prompt: usize,
    pub user_model: usize,
    pub conversation_history: usize,
    pub response_buffer: usize,
    pub total_context: usize,
}

impl TokenBudget {
    /// Calculate remaining budget for memories.
    pub fn memory_budget(&self) -> usize {
        self.total_context
            .saturating_sub(self.system_prompt)
            .saturating_sub(self.user_model)
            .saturating_sub(self.conversation_history)
            .saturating_sub(self.response_buffer)
    }

    /// Create budget for a specific model context size.
    pub fn for_context_size(total_tokens: usize) -> Self {
        let mut budget = TOKEN_BUDGETS;
        budget.total_context = total_tokens;
        budget
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        TOKEN_BUDGETS
    }
}

/// User profile data for context injection.
#[derive(Debug, Clone, Default)]
pub struct UserProfile {
    pub name: Option<String>,
    pub preferences: Vec<String>,
    pub goals: Vec<String>,
    pub facts: Vec<String>,
}

impl UserProfile {
    /// Format as a context string.
    pub fn to_context_string(&self) -> String {
        let mut parts = Vec::new();

        if let Some(name) = &self.name {
            parts.push(format!("The user's name is {}.", name));
        }

        if !self.preferences.is_empty() {
            parts.push(format!("User preferences: {}", self.preferences.join(", ")));
        }

        if !self.goals.is_empty() {
            parts.push(format!("User goals: {}", self.goals.join(", ")));
        }

        if !self.facts.is_empty() {
            parts.push(format!("Key facts: {}", self.facts.join("; ")));
        }

        parts.join(" ")
    }

    /// Estimate token count (rough approximation: ~4 chars per token).
    pub fn estimate_tokens(&self) -> usize {
        self.to_context_string().len() / 4
    }
}

/// Context assembler — builds prompts respecting token budgets.
pub struct ContextAssembler {
    budget: TokenBudget,
    system_prompt: String,
    user_profile: UserProfile,
}

impl ContextAssembler {
    /// Create a new context assembler.
    pub fn new(budget: TokenBudget) -> Self {
        Self {
            budget,
            system_prompt: Self::default_system_prompt(),
            user_profile: UserProfile::default(),
        }
    }

    /// Create with default budget.
    pub fn with_defaults() -> Self {
        Self::new(TOKEN_BUDGETS)
    }

    /// Set custom system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set user profile.
    pub fn with_user_profile(mut self, profile: UserProfile) -> Self {
        self.user_profile = profile;
        self
    }

    /// Get the default system prompt.
    fn default_system_prompt() -> String {
        r#"You are the SOUL of Brain OS — a biologically-inspired, proactive cognitive engine. You are not just an assistant; you are the user's digital hippocampus and prefrontal cortex, operating with deep context and long-term memory.

Your Identity:
- You are "Brain", the central intelligence of a local-first memory system.
- You are private, secure, and run entirely on the user's machine.
- Your purpose is to eliminate "context amnesia" by bridging the gap between siloed tools and the user's life.

Your Capabilities:
- Episodic Memory: You recall past experiences and conversations provided as context.
- Semantic Memory: You maintain a web of facts about the user's world, projects, and habits.
- Proactivity: You don't just react; you anticipate needs based on established patterns (provided in context).

Operating Principles:
1. TRUTH OVER HALLUCINATION: Answer based ONLY on the provided memories and general knowledge. If information is missing from memory, state: "I don't have that in my memory yet."
2. SEAMLESS RECALL: Reference memories naturally ("You mentioned earlier...", "Based on what we discussed...").
3. COGNITIVE CLARITY: Be concise, direct, and insightful. Avoid corporate fluff.
4. CONTEXTUAL AWARENESS: Use the provided User Profile to tailor your tone and relevance.
5. CURIOSITY: When you lack context about the user, ask one focused follow-up question. Learning about the user is part of your job — don't wait to be told.

You are the user's partner in thought. Your goal is to make their digital life feel like a continuous, coherent stream of intelligence."#
            .to_string()
    }

    /// Assemble context into messages.
    ///
    /// Takes retrieved memories and conversation history, returns
    /// messages ready for the LLM.
    pub fn assemble(
        &self,
        user_message: &str,
        memories: &[Memory],
        conversation_history: &[Message],
    ) -> Vec<Message> {
        let mut messages = Vec::new();
        let memory_budget = self.budget.memory_budget();

        // 1. System prompt with user profile
        let system_content = if self.user_profile.estimate_tokens() > 0 {
            format!(
                "{}\n\nUser Profile: {}",
                self.system_prompt,
                self.user_profile.to_context_string()
            )
        } else {
            self.system_prompt.clone()
        };
        messages.push(Message {
            role: Role::System,
            content: system_content,
        });

        // 2. Add memories as system context (if within budget)
        let mut current_tokens = messages[0].content.len() / 4;
        let mut memory_context = String::new();

        for memory in memories {
            let memory_text = if let Some(ref agent) = memory.agent {
                format!(
                    "- [{:?}, agent: {}] {}\n",
                    memory.source, agent, memory.content
                )
            } else {
                format!("- [{:?}] {}\n", memory.source, memory.content)
            };
            let memory_tokens = memory_text.len() / 4;

            if current_tokens + memory_tokens > memory_budget {
                break;
            }

            memory_context.push_str(&memory_text);
            current_tokens += memory_tokens;
        }

        if !memory_context.is_empty() {
            messages.push(Message {
                role: Role::System,
                content: format!("Relevant memories:\n{}", memory_context),
            });
        }

        // 3. Add conversation history (respecting budget)
        let mut history_tokens: usize = 0;
        let mut included_history: Vec<Message> = Vec::new();

        // Start from most recent and work backwards
        for msg in conversation_history.iter().rev() {
            let msg_tokens = msg.content.len() / 4;
            if history_tokens + msg_tokens > self.budget.conversation_history {
                break;
            }
            included_history.push(msg.clone());
            history_tokens += msg_tokens;
        }

        // Reverse to maintain chronological order
        included_history.reverse();
        messages.extend(included_history);

        // 4. Add current user message
        messages.push(Message {
            role: Role::User,
            content: user_message.to_string(),
        });

        messages
    }

    /// Quick estimate of total tokens in messages.
    pub fn estimate_tokens(messages: &[Message]) -> usize {
        messages.iter().map(|m| m.content.len() / 4).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget_memory_allocation() {
        let budget = TokenBudget::default();
        let memory_budget = budget.memory_budget();

        // 8192 - 500 - 300 - 2000 - 400 = 4992
        assert_eq!(memory_budget, 4992);
    }

    #[test]
    fn test_token_budget_for_context_size() {
        let budget = TokenBudget::for_context_size(128000);
        assert_eq!(budget.total_context, 128000);
        assert_eq!(budget.memory_budget(), 128000 - 500 - 300 - 2000 - 400);
    }

    #[test]
    fn test_user_profile_to_context() {
        let profile = UserProfile {
            name: Some("Alice".to_string()),
            preferences: vec!["coffee".to_string(), "quiet mornings".to_string()],
            goals: vec!["learn Rust".to_string()],
            facts: vec!["works remotely".to_string()],
        };

        let context = profile.to_context_string();
        assert!(context.contains("Alice"));
        assert!(context.contains("coffee"));
        assert!(context.contains("learn Rust"));
    }

    #[test]
    fn test_context_assembler_basic() {
        use hippocampus::search::MemorySource;

        let assembler = ContextAssembler::with_defaults();

        let memories = vec![Memory {
            id: "1".to_string(),
            content: "User likes Rust programming".to_string(),
            source: MemorySource::Semantic,
            score: 0.9,
            importance: 0.8,
            timestamp: "2026-01-01".to_string(),
            agent: None,
        }];

        let history = vec![];
        let messages = assembler.assemble("What language should I learn?", &memories, &history);

        // Should have: system prompt, memory context, user message
        assert!(messages.len() >= 2);
        assert_eq!(
            messages.last().unwrap().content,
            "What language should I learn?"
        );
        assert_eq!(messages.last().unwrap().role, Role::User);
    }

    #[test]
    fn test_context_assembler_agent_attribution() {
        use hippocampus::search::MemorySource;

        let assembler = ContextAssembler::with_defaults();

        let memories = vec![
            Memory {
                id: "1".to_string(),
                content: "User likes coffee".to_string(),
                source: MemorySource::Episodic,
                score: 0.9,
                importance: 0.8,
                timestamp: "2026-01-01".to_string(),
                agent: Some("slack-bot".to_string()),
            },
            Memory {
                id: "2".to_string(),
                content: "User works remotely".to_string(),
                source: MemorySource::Semantic,
                score: 0.85,
                importance: 0.7,
                timestamp: "2026-01-02".to_string(),
                agent: None,
            },
        ];

        let messages = assembler.assemble("Tell me about the user", &memories, &[]);

        let memory_msg = messages
            .iter()
            .find(|m| m.content.contains("Relevant memories"))
            .expect("should have memory context message");

        assert!(
            memory_msg.content.contains("agent: slack-bot"),
            "memory with agent should include attribution"
        );
        assert!(
            !memory_msg.content.contains("agent: ")
                || memory_msg.content.matches("agent: ").count() == 1,
            "memory without agent should NOT include agent label"
        );
    }

    #[test]
    fn test_context_assembler_with_history() {
        let assembler = ContextAssembler::with_defaults();

        let history = vec![
            Message {
                role: Role::User,
                content: "Hello".to_string(),
            },
            Message {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
            },
        ];

        let messages = assembler.assemble("How are you?", &[], &history);

        // Should include system + history + current message
        assert!(messages.len() >= 3);
        assert_eq!(messages.last().unwrap().content, "How are you?");
    }

    #[test]
    fn test_default_prompt_core_instructions() {
        let assembler = ContextAssembler::with_defaults();
        let messages = assembler.assemble("How do I connect OpenClaw?", &[], &[]);
        let system = &messages[0].content;

        assert!(system.contains("Brain"));
        assert!(system.contains("SOUL"));
        assert!(system.contains("biologically-inspired"));
        assert!(system.contains("Episodic Memory"));
        assert!(system.contains("Semantic Memory"));
        assert!(system.contains("Proactivity"));
        assert!(system.contains("TRUTH OVER HALLUCINATION"));
        assert!(
            system.contains("CURIOSITY"),
            "SOUL prompt must include CURIOSITY operating principle"
        );
    }

    #[test]
    fn test_onboarding_greeting_exists() {
        assert!(
            ONBOARDING_GREETING.contains("Brain"),
            "greeting must mention Brain"
        );
        assert!(
            ONBOARDING_GREETING.contains("name"),
            "greeting must ask for the user's name"
        );
    }

    #[test]
    fn test_onboarding_addendum_exists() {
        assert!(
            ONBOARDING_ADDENDUM.contains("ONBOARDING MODE"),
            "addendum must contain ONBOARDING MODE marker"
        );
        assert!(
            ONBOARDING_ADDENDUM.contains("follow-up question"),
            "addendum must instruct follow-up questions"
        );
    }

    #[test]
    fn test_estimate_tokens() {
        let messages = vec![Message {
            role: Role::User,
            content: "Hello world".to_string(),
        }];

        let tokens = ContextAssembler::estimate_tokens(&messages);
        assert!(tokens > 0);
        assert_eq!(tokens, 11 / 4); // "Hello world" is 11 chars
    }
}
