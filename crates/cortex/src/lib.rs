//! # Brain Cortex
//!
//! Reasoning core providing:
//! - LLM client (Ollama, OpenAI compatible)
//! - Hybrid provider with trait-based adapter pattern
//! - Context assembly from memory + user model
//! - Token budget management
//! - Tool calling and action dispatch
//! - Structured output validation with retry logic

pub mod actions;
pub mod context;
pub mod llm;

pub use llm::{
    LlmProvider, OllamaProvider, OpenAiProvider, ProviderConfig, create_provider,
    Message, Role, Response, ResponseChunk, Usage, LlmError,
};
