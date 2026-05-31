//! Context assembly — builds LLM prompts with token budget management.
//!
//! Manages the token budget for LLM context windows:
//! - System prompt (~500 tokens)
//! - User model snapshot (~300 tokens)
//! - Conversation history (~2000 tokens)
//! - Retrieved memories (remaining budget)
//! - Response buffer (~400 tokens)

use crate::llm::Message;
use hippocampus::search::Memory;

/// Default token budgets.
pub const TOKEN_BUDGETS: TokenBudget = TokenBudget {
    system_prompt: 500,
    user_model: 300,
    conversation_history: 2000,
    response_buffer: 400,
    attachments: 2500,
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

/// The always-on cognitive faculties, rendered as the fallback "Your
/// Capabilities" section of the SOUL prompt. Used verbatim when no live
/// capability digest is supplied (non-chat LLM paths, tests, custom
/// prompts) and as the prefix of the live digest the chat path builds
/// (see `signal::pipeline::conversation`). Keeping the
/// wording in one place stops the static and live views from drifting.
pub const DEFAULT_CAPABILITIES: &str = r#"Your Capabilities:
- Episodic Memory: You recall past experiences and conversations provided as context.
- Semantic Memory: You maintain a web of facts about the user's world, projects, and habits.
- Proactivity: You don't just react; you anticipate needs based on established patterns (provided in context)."#;

/// Token budget allocation.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    pub system_prompt: usize,
    pub user_model: usize,
    pub conversation_history: usize,
    pub response_buffer: usize,
    /// Cap on rendered path-attachments (snapshots of files/dirs the
    /// user referenced in chat). Truncated to fit by the assembler.
    pub attachments: usize,
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
            .saturating_sub(self.attachments)
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

/// Path-attachment grounding for a chat turn. When the user references
/// a local path in their message, the pipeline reads it on their behalf
/// and hands the snapshot here so the LLM can see *what's actually
/// there* alongside memories and history. The SOUL prompt's
/// "ATTACHED_CONTENT" instructions explain how to read these blocks.
#[derive(Debug, Clone)]
pub struct Attachment {
    /// Path token as the user wrote it. Preserved verbatim so the LLM
    /// can refer back to the user's own wording.
    pub display_path: String,
    /// Rendered snapshot — directory listing + histogram + inlined
    /// files for a directory, or file excerpt for a file. Built by
    /// `signal::pipeline::build_directory_snapshot` /
    /// `build_file_snapshot`.
    pub snapshot: String,
}

/// A path the user referenced that couldn't be attached (not found,
/// outside `security.allowed_paths`, wrong file kind). Rendered as a
/// `<SKIPPED_PATH>` tag so Brain can mention it instead of silently
/// dropping the reference.
#[derive(Debug, Clone)]
pub struct SkippedAttachment {
    pub display_path: String,
    pub reason: String,
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

    /// Estimate token count (conservative: ~2 chars per token to handle non-ASCII safely).
    pub fn estimate_tokens(&self) -> usize {
        self.to_context_string().chars().count() / 2
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

Operating Principles:
1. TRUTH OVER HALLUCINATION: Ground answers in (a) the provided memories, (b) the live conversation history above this message, and (c) general world knowledge. If a *fact about the user* is genuinely absent from memory AND not present in the conversation, state: "I don't have that in my memory yet." Do NOT say this when the user is asking about things discussed earlier in the current conversation — answer from the message thread itself.
   - SELF-KNOWLEDGE BOUNDARY: General world knowledge is fine for the world at large, but it is NOT a source for claims about Brain itself. Any statement about Brain's own CLI commands, config keys/schema, file layout, or features MUST come from the "About Brain" and "Your Capabilities" sections below — never from general knowledge or guesswork. If the answer isn't in those sections, say so plainly ("that isn't something Brain exposes" / "that's not a command/config key I have") and, where useful, point to the closest real command or config key. Never invent command names, config keys, templating syntax, or option flags — a confident, plausible-looking fabrication of Brain's surface is the worst failure mode.
   - MEMORY GROUNDING: Never assert a specific fact about the user unless it appears verbatim in the "Relevant memories:" block or earlier in this conversation. This applies with full force when you are *describing what you remember* (e.g. answering "what do you know about me?" or "what are your capabilities?"): do NOT manufacture illustrative examples — never say things like "you bike to work" or "you deploy on Fridays" to demonstrate recall. Describe the *kinds* of things you store (preferences, projects, habits, people, decisions) in the abstract, and cite only real entries from the memories block. A fabricated personal "memory" is a betrayal of a memory product's core promise — when memory is empty or lacks the detail, say so.
2. SEAMLESS RECALL: Reference memories and prior turns naturally ("You mentioned earlier...", "Based on what we discussed...").
3. COGNITIVE CLARITY: Be concise, direct, and insightful. Avoid corporate fluff. Match response length to the question — simple greetings get one or two sentences, not tables.
4. CONTEXTUAL AWARENESS: Use the provided User Profile to tailor your tone and relevance.
5. CURIOSITY: When you lack context about the user, ask one focused follow-up question. Learning about the user is part of your job — don't wait to be told.
6. FORMATTING: The user's terminal renders markdown. Use it lightly when it helps (lists for multi-item answers, **bold** for emphasis, `code` for identifiers). Skip headings and tables for short replies. Prefer bullet lists over tables — the terminal is narrow and wide tables render poorly; only use a table for genuinely tabular data with short cells.
7. ATTACHED CONTENT: When the user references a local path, an `<ATTACHED_CONTENT path="…">` block is provided below as grounding — that is what is actually on disk, read on the user's behalf. Adapt your response shape to the *content*, not to a template: a chat export deserves a conversational summary with themes, tone, and an honest opinion; a code project deserves a technical overview; a folder of photos or media deserves an honest "I can see these file types but I can't view the images themselves." Never describe a non-code folder as if it were a software project. If a `<SKIPPED_PATH reason="…"/>` tag appears, the user named a path I couldn't read — acknowledge it briefly and ask them to confirm or rephrase.

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
        self.assemble_with_addendum(user_message, memories, conversation_history, None)
    }

    /// Like [`assemble`], but appends `addendum` to the system prompt if provided.
    /// Used to switch prompt modes per-turn (e.g. onboarding) without mutating
    /// the shared assembler.
    pub fn assemble_with_addendum(
        &self,
        user_message: &str,
        memories: &[Memory],
        conversation_history: &[Message],
        addendum: Option<&str>,
    ) -> Vec<Message> {
        self.assemble_full(
            user_message,
            memories,
            conversation_history,
            addendum,
            None,
            &[],
            &[],
        )
    }

    /// Full assembly with path-attachment grounding. Attachments render
    /// as `<ATTACHED_CONTENT>` blocks in a System message positioned
    /// right before the user's actual message — closest attention slot
    /// to "what the user just put on the table." Skipped paths render
    /// as `<SKIPPED_PATH>` tags in the same block so Brain can mention
    /// them naturally.
    ///
    /// Per-attachment content is truncated to fit `budget.attachments`;
    /// when total snapshot text exceeds the budget, later attachments
    /// shrink first so the first (and usually primary) reference stays
    /// intact.
    ///
    /// `capabilities` is the "Your Capabilities" section of the SOUL
    /// prompt. The chat path passes a *live* digest rendered from the
    /// currently-wired tools and agents; every other path
    /// passes `None` and falls back to [`DEFAULT_CAPABILITIES`]. Either
    /// way the section is appended after the base prompt so the reasoner
    /// always sees an explicit capability manifest.
    pub fn assemble_full(
        &self,
        user_message: &str,
        memories: &[Memory],
        conversation_history: &[Message],
        addendum: Option<&str>,
        capabilities: Option<&str>,
        attachments: &[Attachment],
        skipped: &[SkippedAttachment],
    ) -> Vec<Message> {
        let mut messages = Vec::new();
        let memory_budget = self.budget.memory_budget();

        // 1. System prompt with optional addendum and user profile
        let base_prompt = match addendum {
            Some(extra) if !extra.is_empty() => {
                format!("{}{}", self.system_prompt, extra)
            }
            _ => self.system_prompt.clone(),
        };
        // Capability manifest: live digest from the chat path, or the
        // static always-on faculties everywhere else.
        let prompt_with_caps = format!(
            "{}\n\n{}",
            base_prompt,
            capabilities.unwrap_or(DEFAULT_CAPABILITIES)
        );
        let system_content = if self.user_profile.estimate_tokens() > 0 {
            format!(
                "{}\n\nUser Profile: {}",
                prompt_with_caps,
                self.user_profile.to_context_string()
            )
        } else {
            prompt_with_caps
        };
        messages.push(Message::system(system_content));

        // 2. Add memories as system context (if within budget)
        let mut current_tokens = messages[0].content.chars().count() / 2;
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
            let memory_tokens = memory_text.chars().count() / 2;

            if current_tokens + memory_tokens > memory_budget {
                break;
            }

            memory_context.push_str(&memory_text);
            current_tokens += memory_tokens;
        }

        if !memory_context.is_empty() {
            messages.push(Message::system(format!(
                "Relevant memories:\n{}",
                memory_context
            )));
        }

        // 3. Add conversation history (respecting budget)
        let mut history_tokens: usize = 0;
        let mut included_history: Vec<Message> = Vec::new();

        // Start from most recent and work backwards
        for msg in conversation_history.iter().rev() {
            let msg_tokens = msg.content.chars().count() / 2;
            if history_tokens + msg_tokens > self.budget.conversation_history {
                break;
            }
            included_history.push(msg.clone());
            history_tokens += msg_tokens;
        }

        // Reverse to maintain chronological order
        included_history.reverse();
        messages.extend(included_history);

        // 4. Attached path grounding (renders right before the user
        //    message so the LLM has it freshly in attention).
        if let Some(block) = render_attachments_block(attachments, skipped, self.budget.attachments)
        {
            messages.push(Message::system(block));
        }

        // 5. Add current user message
        messages.push(Message::user(user_message.to_string()));

        messages
    }

    /// Quick estimate of total tokens in messages.
    pub fn estimate_tokens(messages: &[Message]) -> usize {
        messages.iter().map(|m| m.content.chars().count() / 2).sum()
    }
}

/// Build the `<ATTACHED_CONTENT>` / `<SKIPPED_PATH>` block that goes
/// just before the user's message. Returns `None` when there's nothing
/// to render. Each attachment's snapshot is truncated to keep the
/// total under `budget_tokens` (2 chars ≈ 1 token); later attachments
/// shrink first so the primary reference stays intact.
fn render_attachments_block(
    attachments: &[Attachment],
    skipped: &[SkippedAttachment],
    budget_tokens: usize,
) -> Option<String> {
    if attachments.is_empty() && skipped.is_empty() {
        return None;
    }
    // 2 chars per token (matches the conservative estimator used
    // elsewhere in this module).
    let char_budget = budget_tokens.saturating_mul(2);
    let mut out = String::new();
    let mut chars_used = 0usize;

    for (i, att) in attachments.iter().enumerate() {
        // Per-attachment ceiling: equal share of remaining budget,
        // floored at 600 chars so a small attachment can always fit.
        let remaining_atts = attachments.len() - i;
        let per_attachment =
            (char_budget.saturating_sub(chars_used) / remaining_atts.max(1)).max(600);
        let body = truncate_snapshot(&att.snapshot, per_attachment);
        let block = format!(
            "<ATTACHED_CONTENT path=\"{}\">\n{}\n</ATTACHED_CONTENT>\n",
            att.display_path, body
        );
        chars_used = chars_used.saturating_add(block.chars().count());
        out.push_str(&block);
    }
    for sk in skipped {
        let tag = format!(
            "<SKIPPED_PATH path=\"{}\" reason=\"{}\"/>\n",
            sk.display_path,
            sk.reason.replace('"', "'"),
        );
        out.push_str(&tag);
    }
    Some(out)
}

/// Truncate a snapshot string to at most `cap_chars`, appending a
/// short marker so the LLM knows content was cut. Walks back to a
/// character boundary to avoid splitting multi-byte chars.
fn truncate_snapshot(s: &str, cap_chars: usize) -> String {
    if s.chars().count() <= cap_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(cap_chars.saturating_sub(20)).collect();
    out.push_str("\n…[truncated]");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Role;

    #[test]
    fn test_token_budget_memory_allocation() {
        let budget = TokenBudget::default();
        let memory_budget = budget.memory_budget();

        // 8192 - 500 - 300 - 2000 - 400 - 2500 = 2492
        assert_eq!(memory_budget, 2492);
    }

    #[test]
    fn test_token_budget_for_context_size() {
        let budget = TokenBudget::for_context_size(128000);
        assert_eq!(budget.total_context, 128000);
        assert_eq!(
            budget.memory_budget(),
            128000 - 500 - 300 - 2000 - 400 - 2500
        );
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
    fn test_assemble_with_addendum_injects_into_system_prompt() {
        let assembler = ContextAssembler::with_defaults();
        let messages = assembler.assemble_with_addendum("hi", &[], &[], Some(ONBOARDING_ADDENDUM));

        let system = messages
            .iter()
            .find(|m| matches!(m.role, Role::System))
            .expect("system message");
        assert!(
            system.content.contains("[ONBOARDING MODE"),
            "onboarding addendum should be present in system prompt"
        );
    }

    #[test]
    fn system_prompt_forbids_fabricated_memories() {
        // The SOUL prompt must carry the memory-grounding rule that stops the
        // reasoner inventing first-person "memories" (WS3). Anchored on the
        // base prompt so it's present on every turn, onboarding or not.
        let assembler = ContextAssembler::with_defaults();
        let messages = assembler.assemble("what do you know about me?", &[], &[]);
        let system = &messages[0].content;
        assert!(
            system.contains("MEMORY GROUNDING"),
            "memory-grounding rule missing from system prompt"
        );
        assert!(
            system.contains("Relevant memories:"),
            "rule should anchor on the real memories block label"
        );
    }

    #[test]
    fn test_assemble_without_addendum_matches_plain_assemble() {
        let assembler = ContextAssembler::with_defaults();
        let a = assembler.assemble("hi", &[], &[]);
        let b = assembler.assemble_with_addendum("hi", &[], &[], None);
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].content, b[0].content);
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
                agent: Some("chat-bot".to_string()),
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

        // The memories block is its own system message starting with the
        // label; `starts_with` avoids matching the base system prompt, which
        // now references "Relevant memories:" in its memory-grounding rule.
        let memory_msg = messages
            .iter()
            .find(|m| m.content.starts_with("Relevant memories:"))
            .expect("should have memory context message");

        assert!(
            memory_msg.content.contains("agent: chat-bot"),
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
                ..Default::default()
            },
            Message {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
                ..Default::default()
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
        assert!(
            system.contains("ATTACHED CONTENT"),
            "SOUL prompt must teach Brain how to handle <ATTACHED_CONTENT> blocks"
        );
        assert!(
            system.contains("chat export deserves a conversational summary"),
            "SOUL prompt must instruct response-shape adaptation by content type"
        );
    }

    #[test]
    fn default_capabilities_used_when_no_digest_supplied() {
        let assembler = ContextAssembler::with_defaults();
        let messages = assembler.assemble("what can you do?", &[], &[]);
        let system = &messages[0].content;
        // Falls back to the static always-on faculties.
        assert!(system.contains(DEFAULT_CAPABILITIES));
        assert!(system.contains("Episodic Memory"));
    }

    #[test]
    fn live_capability_digest_overrides_default() {
        let assembler = ContextAssembler::with_defaults();
        let digest = "Your Capabilities:\n- Episodic Memory: ...\n\nMounted tools:\n- MCP server \"github\": create_issue";
        let messages =
            assembler.assemble_full("what can you do?", &[], &[], None, Some(digest), &[], &[]);
        let system = &messages[0].content;
        assert!(
            system.contains("MCP server \"github\": create_issue"),
            "live digest must reach the system prompt"
        );
        // The supplied digest replaces the static block — the default's
        // Semantic/Proactivity bullets are not present unless the caller
        // included them.
        assert!(!system.contains("a web of facts about the user's world"));
    }

    #[test]
    fn attachments_render_as_a_dedicated_system_message_before_user() {
        let assembler = ContextAssembler::with_defaults();
        let attachments = vec![Attachment {
            display_path: "/Users/me/notes.md".to_string(),
            snapshot: "# my notes\nbuy milk".to_string(),
        }];
        let messages =
            assembler.assemble_full("read this", &[], &[], None, None, &attachments, &[]);

        // Penultimate message should be the attachments block; last is
        // the user message itself.
        let user_msg = messages.last().expect("non-empty");
        assert_eq!(user_msg.role, Role::User);
        assert_eq!(user_msg.content, "read this");

        let prev = &messages[messages.len() - 2];
        assert_eq!(prev.role, Role::System);
        assert!(
            prev.content
                .contains("<ATTACHED_CONTENT path=\"/Users/me/notes.md\">"),
            "missing attached-content block:\n{}",
            prev.content
        );
        assert!(prev.content.contains("buy milk"));
        assert!(prev.content.contains("</ATTACHED_CONTENT>"));
    }

    #[test]
    fn skipped_paths_render_as_a_tag_for_brain_to_mention() {
        let assembler = ContextAssembler::with_defaults();
        let skipped = vec![SkippedAttachment {
            display_path: "/Users/me/missing.txt".to_string(),
            reason: "path not found".to_string(),
        }];
        let messages = assembler.assemble_full("summarise it", &[], &[], None, None, &[], &skipped);
        let prev = &messages[messages.len() - 2];
        assert!(prev.content.contains("<SKIPPED_PATH"));
        assert!(prev.content.contains("/Users/me/missing.txt"));
        assert!(prev.content.contains("path not found"));
    }

    #[test]
    fn no_attachments_means_no_extra_block() {
        let assembler = ContextAssembler::with_defaults();
        let before = assembler.assemble("hi", &[], &[]);
        let after = assembler.assemble_full("hi", &[], &[], None, None, &[], &[]);
        assert_eq!(
            before.len(),
            after.len(),
            "no attachments must not add a message"
        );
    }

    #[test]
    fn large_attachment_is_truncated_to_budget() {
        // Snapshot is 60_000 chars (~30_000 tokens). Default attachments
        // budget is 2500 tokens ≈ 5000 chars; the rendered block must be
        // far smaller than the input snapshot.
        let huge = "x".repeat(60_000);
        let assembler = ContextAssembler::with_defaults();
        let attachments = vec![Attachment {
            display_path: "/Users/me/huge.txt".to_string(),
            snapshot: huge,
        }];
        let messages = assembler.assemble_full("read", &[], &[], None, None, &attachments, &[]);
        let prev = &messages[messages.len() - 2];
        assert!(
            prev.content.contains("[truncated]"),
            "huge attachment must be marked as truncated"
        );
        // Sanity: rendered block must be at least an order of magnitude
        // smaller than the input snapshot.
        assert!(
            prev.content.chars().count() < 10_000,
            "rendered block too large: {} chars",
            prev.content.chars().count()
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
        let messages = vec![Message::user("Hello world")];

        let tokens = ContextAssembler::estimate_tokens(&messages);
        assert!(tokens > 0);
        assert_eq!(tokens, 11 / 2); // "Hello world" is 11 chars, ~2 chars/token
    }
}
