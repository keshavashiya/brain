//! # Brain Cortex
//!
//! Reasoning core providing:
//! - LLM client (Ollama, OpenAI-compatible)
//! - Context assembly from memory + user model
//! - Token budget management
//! - Tool calling and action dispatch
//! - Structured output validation with retry logic

pub mod actions;
pub mod context;
pub mod llm;
pub mod presets;

pub use llm::{
    create_provider, extract_json_from_response, select_provider, LlmError, LlmProvider, Message,
    OllamaProvider, OpenAiProvider, ProposedToolCall, ProviderConfig, Response, ResponseChunk,
    Role, ToolDef, Usage,
};
pub use presets::Preset;
