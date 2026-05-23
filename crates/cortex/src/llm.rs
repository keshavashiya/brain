//! LLM client — hybrid provider with trait-based adapter.
//!
//! `LlmProvider` trait with multiple implementations:
//! - `OllamaProvider` — local Ollama server
//! - `OpenAiProvider` — OpenAI compatible APIs

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod ollama;
mod openai;

#[cfg(test)]
mod tests;

pub use ollama::OllamaProvider;
pub use openai::OpenAiProvider;

mod failover;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from the LLM layer.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Invalid response format: {0}")]
    InvalidFormat(String),

    #[error("Provider not available: {0}")]
    ProviderUnavailable(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Timeout")]
    Timeout,
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message roles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    /// Wire-format string used by every OpenAI-shaped chat API (OpenAI,
    /// OpenRouter, Ollama in chat mode, etc.). Centralised here so each
    /// provider's `convert_messages` body stays a one-line `.map(...)`.
    pub fn as_wire_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

// ─── Shared HTTP helpers (used by openai + ollama submodules) ───────────────

/// Build a `reqwest::Client` with the given timeout, mapping construction
/// failure to [`LlmError::ProviderUnavailable`].
pub(crate) fn build_http_client(timeout: std::time::Duration) -> Result<reqwest::Client, LlmError> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|e| LlmError::ProviderUnavailable(format!("Failed to create HTTP client: {e}")))
}

/// If `resp` is non-success, drain the body and turn it into
/// [`LlmError::Api`]. Otherwise pass the response through untouched so the
/// caller can `.json()` / `.bytes_stream()` it.
pub(crate) async fn ensure_ok(resp: reqwest::Response) -> Result<reqwest::Response, LlmError> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    Err(LlmError::Api {
        status: status.as_u16(),
        message: body,
    })
}

/// LLM response chunk (for streaming).
#[derive(Debug, Clone)]
pub struct ResponseChunk {
    pub content: String,
    pub is_done: bool,
}

/// Complete LLM response.
#[derive(Debug, Clone)]
pub struct Response {
    pub content: String,
    pub usage: Option<Usage>,
}

/// Token usage statistics.
#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ─── Provider Trait ─────────────────────────────────────────────────────────

/// Trait for LLM providers.
#[async_trait::async_trait]
pub trait LlmProvider: Send + Sync {
    /// Generate a complete response (non-streaming).
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError>;

    /// Generate a streaming response.
    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>;

    /// Check if the provider is available.
    async fn health_check(&self) -> bool;

    /// Get the provider name.
    fn name(&self) -> &str;

    /// Get the active model name.
    fn model(&self) -> &str;

    /// List models available from this provider. Used by `select_provider`
    /// to probe reachability and match `preferred_models` during startup.
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
}

// ─── Provider Factory ───────────────────────────────────────────────────────

/// Configuration for LLM provider selection.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub temperature: f64,
    pub max_tokens: i32,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            provider: "ollama".to_string(),
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            model: "qwen2.5-coder:7b".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
        }
    }
}

/// Create an LLM provider from configuration.
///
/// Resolution order:
/// 1. `ollama` → `OllamaProvider`.
/// 2. `openai_compat` (or a built-in preset: openai, openrouter, groq,
///    deepseek, together, gemini-compat) → OpenAI-compatible provider.
///    An explicit non-empty `base_url` overrides the preset default.
/// 3. Unknown provider → fall back to default Ollama with a warning.
pub fn create_provider(config: &ProviderConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    if config.provider == "ollama" {
        let provider = OllamaProvider::new(
            &config.base_url,
            &config.model,
            config.temperature,
            config.max_tokens,
        )
        .or_else(|e| {
            tracing::error!(error = %e, "Failed to create Ollama provider, falling back to default");
            OllamaProvider::default_config()
        })?;
        return Ok(Box::new(provider));
    }

    let preset_base = crate::presets::resolve(&config.provider).map(|p| p.base_url);

    if config.provider == "openai_compat" || preset_base.is_some() {
        let base_url = if !config.base_url.is_empty() {
            config.base_url.as_str()
        } else if let Some(b) = preset_base {
            b
        } else {
            return Err(LlmError::ProviderUnavailable(format!(
                "provider `{}` has no base_url configured",
                config.provider
            )));
        };
        return Ok(Box::new(OpenAiProvider::new(
            base_url,
            config.api_key.as_deref(),
            &config.model,
            config.temperature,
            Some(config.max_tokens),
        )?));
    }

    tracing::warn!(
        provider = %config.provider,
        "Unknown LLM provider, falling back to default Ollama"
    );
    Ok(Box::new(OllamaProvider::default_config()?))
}

// ─── Multi-provider selection ───────────────────────────────────────────────

/// Build a `ProviderConfig` from a `brain::ProviderEntry` and shared
/// temperature/max_tokens. `model_override` lets `select_provider` swap in
/// a preferred model discovered via `list_models`.
fn provider_config_from_entry(
    entry: &brain::ProviderEntry,
    temperature: f64,
    max_tokens: i32,
    model_override: Option<&str>,
) -> ProviderConfig {
    let api_key = entry.api_key.trim();
    ProviderConfig {
        provider: entry.kind.clone(),
        base_url: entry.base_url.clone(),
        api_key: if api_key.is_empty() {
            None
        } else {
            Some(api_key.to_string())
        },
        model: model_override.unwrap_or(&entry.model).to_string(),
        temperature,
        max_tokens,
    }
}

/// Probe every configured provider, pick the first reachable one whose
/// `preferred_models` intersects the live model list, and return it.
///
/// When `llm.providers` is empty we synthesise a single entry from the
/// legacy `llm.provider`/`model`/`base_url`/`api_key` fields — so existing
/// configs keep working unchanged.
///
/// Fail-safe: if no provider answers `list_models`, we still return the
/// first entry as a best effort rather than erroring out (the underlying
/// generate call will surface the real problem when used).
pub async fn select_provider(llm: &brain::LlmConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    let entries = synthesise_entries(llm);
    let max_tokens = llm.max_tokens as i32;

    if entries.is_empty() {
        return Err(LlmError::ProviderUnavailable(
            "no LLM providers configured".into(),
        ));
    }

    for entry in &entries {
        let cfg = provider_config_from_entry(entry, llm.temperature, max_tokens, None);
        let probe = match create_provider(&cfg) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(name = %entry.name, error = %e, "skipping provider — construction failed");
                continue;
            }
        };

        match probe.list_models().await {
            Ok(models) => {
                let chosen = pick_model(&entry.preferred_models, &models, &entry.model);
                tracing::info!(
                    name = %entry.name,
                    kind = %entry.kind,
                    model = %chosen,
                    "LLM provider selected"
                );
                let cfg =
                    provider_config_from_entry(entry, llm.temperature, max_tokens, Some(&chosen));
                return create_provider(&cfg);
            }
            Err(e) => {
                tracing::warn!(
                    name = %entry.name,
                    error = %e,
                    "provider unreachable — trying next"
                );
            }
        }
    }

    // All probes failed — fall back to the first entry so startup continues
    // and the caller surfaces the real failure on first generate().
    let first = &entries[0];
    tracing::warn!(
        name = %first.name,
        "no provider answered list_models — falling back to first entry"
    );
    let cfg = provider_config_from_entry(first, llm.temperature, max_tokens, None);
    create_provider(&cfg)
}

/// Build a failover chain from all configured providers.
///
/// The chain is ordered: the startup-probed winner goes first; the remaining
/// entries (built without probing) follow as fallbacks. At request time the
/// chain tries each in order whenever the current provider returns a retriable
/// error (429 / 5xx / unavailable / timeout).
pub async fn build_failover_chain(
    llm: &brain::LlmConfig,
) -> Result<failover::FalloverProvider, LlmError> {
    let entries = synthesise_entries(llm);
    let max_tokens = llm.max_tokens as i32;

    if entries.is_empty() {
        return Err(LlmError::ProviderUnavailable(
            "no LLM providers configured".into(),
        ));
    }

    // Find the primary via probing (same logic as select_provider).
    let mut primary_idx = None;
    for (i, entry) in entries.iter().enumerate() {
        let cfg = provider_config_from_entry(entry, llm.temperature, max_tokens, None);
        let probe = match create_provider(&cfg) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(name = %entry.name, error = %e, "skipping provider — construction failed");
                continue;
            }
        };
        match probe.list_models().await {
            Ok(models) => {
                let chosen = pick_model(&entry.preferred_models, &models, &entry.model);
                tracing::info!(
                    name = %entry.name,
                    kind = %entry.kind,
                    model = %chosen,
                    "LLM provider selected"
                );
                primary_idx = Some((i, chosen));
                break;
            }
            Err(e) => {
                tracing::warn!(name = %entry.name, error = %e, "provider unreachable — trying next");
            }
        }
    }

    // If no probe succeeded, fall back to index 0 (best-effort).
    let (primary_i, model_override) = primary_idx.unwrap_or_else(|| {
        tracing::warn!("no provider answered list_models — using first entry as primary");
        (0, entries[0].model.clone())
    });

    // Build all providers: primary first, rest appended in config order.
    let mut providers: Vec<Box<dyn LlmProvider>> = Vec::with_capacity(entries.len());
    let primary_cfg = provider_config_from_entry(
        &entries[primary_i],
        llm.temperature,
        max_tokens,
        Some(&model_override),
    );
    providers.push(create_provider(&primary_cfg)?);

    for (i, entry) in entries.iter().enumerate() {
        if i == primary_i {
            continue;
        }
        let cfg = provider_config_from_entry(entry, llm.temperature, max_tokens, None);
        match create_provider(&cfg) {
            Ok(p) => {
                tracing::info!(name = %entry.name, "registered as fallback provider");
                providers.push(p);
            }
            Err(e) => {
                tracing::warn!(name = %entry.name, error = %e, "fallback provider construction failed — skipping");
            }
        }
    }

    Ok(failover::FalloverProvider::new(providers))
}

fn synthesise_entries(llm: &brain::LlmConfig) -> Vec<brain::ProviderEntry> {
    if !llm.providers.is_empty() {
        return llm.providers.clone();
    }
    // Single-provider fallback path — legitimate use of the deprecated
    // `llm.provider` field (Issue 40). The startup warning fires when
    // both shapes are set; here `providers[]` is empty so it's the only
    // way to know which transport to talk to.
    #[allow(deprecated)]
    let kind = llm.provider.clone();
    vec![brain::ProviderEntry {
        name: "default".to_string(),
        kind,
        base_url: llm.base_url.clone(),
        api_key: llm.api_key.clone(),
        model: llm.model.clone(),
        preferred_models: Vec::new(),
    }]
}

fn pick_model(preferred: &[String], available: &[String], fallback: &str) -> String {
    for want in preferred {
        if available.iter().any(|m| m == want) {
            return want.clone();
        }
    }
    fallback.to_string()
}

/// Extract a JSON object from an LLM response string.
///
/// LLMs sometimes wrap JSON in markdown fences or explanatory text.
/// This tries direct parse first, then finds the outermost `{...}`.
pub fn extract_json_from_response<T: serde::de::DeserializeOwned>(raw: &str) -> Option<T> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<T>(trimmed) {
        return Some(parsed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    serde_json::from_str::<T>(&trimmed[start..=end]).ok()
}
