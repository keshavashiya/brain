//! # Brain Cortex
//!
//! Reasoning core providing:
//! - LLM client (Ollama, OpenAI-compatible)
//! - Context assembly from memory + user model
//! - Token budget management
//! - Tool calling and action dispatch
//! - Structured output validation with retry logic

pub mod actions;
pub mod compaction;
pub mod context;
pub mod llm;
pub mod presets;

pub use llm::{
    build_tier_chains, create_provider, extract_json_from_response, select_provider, LlmError,
    LlmProvider, LlmTiers, Message, OllamaProvider, OpenAiProvider, ProposedToolCall,
    ProviderConfig, Response, ResponseChunk, Role, TaskTier, ToolDef, Usage,
};
pub use presets::Preset;
