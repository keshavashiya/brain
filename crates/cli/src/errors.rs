//! User-friendly error mapping for CLI output.
//!
//! Translates internal error types into actionable messages
//! that guide users toward a fix rather than showing raw error chains.

use cortex::LlmError;
use hippocampus::EmbeddingError;
use signal::SignalError;
use storage::sqlite::SqliteError;

/// Convert an `anyhow::Error` into a user-friendly message.
///
/// Pattern-matches on known error types and returns actionable guidance.
/// Unknown errors pass through unchanged.
pub fn friendly_error(err: &anyhow::Error) -> String {
    // Try each known error type in order of specificity.

    if let Some(e) = err.downcast_ref::<SignalError>() {
        return match e {
            SignalError::Init(_) => format!(
                "Brain failed to initialize. Check your config with `brain status` \
                 or reset with `brain init`.\n\nDetails: {e}"
            ),
            SignalError::Llm(llm_err) => friendly_llm_error(llm_err),
            _ => format!("{e}"),
        };
    }

    if let Some(e) = err.downcast_ref::<LlmError>() {
        return friendly_llm_error(e);
    }

    if let Some(e) = err.downcast_ref::<EmbeddingError>() {
        return match e {
            EmbeddingError::ProviderUnavailable(_) | EmbeddingError::Http(_) => format!(
                "Embedding provider is unavailable. Semantic search will use approximate matching. \
                 Run `ollama pull nomic-embed-text` to restore full quality.\n\nDetails: {e}"
            ),
            _ => format!("{e}"),
        };
    }

    if let Some(e) = err.downcast_ref::<SqliteError>() {
        return match e {
            SqliteError::LockPoisoned => {
                "Memory database is locked. Another Brain process may be running — \
                 check `brain status`."
                    .to_string()
            }
            _ => format!("{e}"),
        };
    }

    // Check for common reqwest errors (connection refused, timeout)
    if let Some(e) = err.downcast_ref::<reqwest::Error>() {
        if e.is_timeout() {
            return "Request timed out. The service may be starting up — try again in a few seconds."
                .to_string();
        }
        if e.is_connect() {
            return format!(
                "Cannot connect to the server. Is the Brain daemon running? \
                 Start it with `brain start` or `brain serve`.\n\nDetails: {e}"
            );
        }
    }

    // Fallback: return the full error chain
    format!("{err:#}")
}

fn friendly_llm_error(e: &LlmError) -> String {
    match e {
        LlmError::ProviderUnavailable(_) | LlmError::Http(_) => format!(
            "Cannot reach the LLM provider. Check that Ollama is running (`ollama serve`) \
             and the model is pulled (`ollama list`).\n\nDetails: {e}"
        ),
        LlmError::Timeout => {
            "LLM request timed out. The model may be loading — try again in a few seconds."
                .to_string()
        }
        LlmError::RateLimited => {
            "LLM provider is rate-limiting requests. Wait a moment and try again.".to_string()
        }
        _ => format!("{e}"),
    }
}

/// Format an error for CLI display.
///
/// In verbose mode, shows both the friendly message and the full error chain.
/// In normal mode, shows only the friendly message.
pub fn format_error(err: &anyhow::Error, verbose: bool) -> String {
    let friendly = friendly_error(err);
    if verbose {
        format!("{friendly}\n\nFull error chain:\n{err:?}")
    } else {
        friendly
    }
}
