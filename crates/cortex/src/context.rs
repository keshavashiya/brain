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
        r#"You are Brain, a personal AI assistant with persistent memory. You have access to:
- Episodic memory (past conversations)
- Semantic memory (facts about the user)

Guidelines:
1. Be helpful, accurate, and concise
2. Reference relevant memories naturally in conversation
3. Ask clarifying questions when needed
4. Respect user privacy and preferences
5. Never make up information you don't have

You are running locally on the user's machine with full privacy."#
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
            let memory_text = format!("- [{:?}] {}\n", memory.source, memory.content);
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
