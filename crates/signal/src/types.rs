//! Type definitions for the signal processing layer.
//!
//! Contains signal, response, error types, the adapter trait, and
//! helper functions — everything that isn't an `impl SignalProcessor` method.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use uuid::Uuid;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from the signal processing layer.
#[derive(Debug, Error)]
pub enum SignalError {
    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("LLM error: {0}")]
    Llm(#[from] cortex::LlmError),

    #[error("Initialization error: {0}")]
    Init(String),
}

/// Network-safe rendering of a [`SignalError`].
///
/// Adapter responses (HTTP / WS / gRPC / MCP) MUST go through this instead
/// of `SignalError::to_string()` so internal details (storage paths, SQL
/// strings, LLM provider messages, init failures) never reach untrusted
/// callers. The original error is preserved in tracing logs for operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicError {
    /// Stable machine-readable code suitable for client branching.
    pub code: &'static str,
    /// Human-readable message safe to expose externally.
    pub message: &'static str,
}

impl PublicError {
    /// 500 Internal Server Error analog for unknown processing failures.
    pub const PROCESSING: PublicError = PublicError {
        code: "processing_failed",
        message: "Failed to process signal",
    };
    pub const STORAGE: PublicError = PublicError {
        code: "storage_unavailable",
        message: "Storage backend unavailable",
    };
    pub const LLM: PublicError = PublicError {
        code: "llm_error",
        message: "Language model error",
    };
    pub const INIT: PublicError = PublicError {
        code: "service_unavailable",
        message: "Service not ready",
    };
}

impl SignalError {
    /// Strip internal detail and return a sanitized error suitable for
    /// rendering to network clients. Keep the original error in tracing
    /// logs alongside any call site that uses this.
    pub fn to_public(&self) -> PublicError {
        match self {
            SignalError::Processing(_) => PublicError::PROCESSING,
            SignalError::Storage(_) => PublicError::STORAGE,
            SignalError::Llm(_) => PublicError::LLM,
            SignalError::Init(_) => PublicError::INIT,
        }
    }
}

// ─── Signal Types ─────────────────────────────────────────────────────────────

/// The source protocol of an incoming signal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalSource {
    Cli,
    Http,
    WebSocket,
    Mcp,
    Grpc,
}

impl SignalSource {
    /// Parse a source string into a SignalSource variant.
    /// Returns the given `default` for unrecognized or None values.
    pub fn parse(s: Option<&str>, default: SignalSource) -> SignalSource {
        match s {
            Some("cli") => SignalSource::Cli,
            Some("http") => SignalSource::Http,
            Some("ws") | Some("websocket") => SignalSource::WebSocket,
            Some("mcp") => SignalSource::Mcp,
            Some("grpc") => SignalSource::Grpc,
            _ => default,
        }
    }
}

/// A unified signal — the single input type for all protocol adapters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: Uuid,
    pub source: SignalSource,
    pub channel: String,
    pub sender: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    /// Memory namespace for this signal (default: "personal").
    #[serde(default = "default_namespace")]
    pub namespace: String,
    /// Originating AI agent (e.g. "claude-code", "opencode"). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    /// Optional session ID for conversation continuity.
    /// When provided, the processor reuses this session instead of creating a new one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Who is asking. Resolved by the
    /// originating adapter from its auth context. `None` means the adapter
    /// did not authenticate the caller; the pipeline's identity gate (if
    /// wired) treats this as anonymous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal: Option<identity::Principal>,
    /// How this signal entered the pipeline. Stamped by the originating
    /// surface: `Provenance::User` for typed input, `Provenance::Llm` for
    /// agent-emitted intents, `Provenance::Reflex` for trigger firings.
    /// Audit, recall, and the inline confirmation gate read this to
    /// distinguish user-typed signals from reflex-driven ones.
    /// `None` is back-compat for adapters that haven't been updated yet —
    /// the pipeline treats `None` the same as it did before this field
    /// existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<intent::Provenance>,
}

fn default_namespace() -> String {
    "personal".to_string()
}

impl Signal {
    /// Create a new Signal with a generated UUID and current timestamp.
    pub fn new(
        source: SignalSource,
        channel: impl Into<String>,
        sender: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source,
            channel: channel.into(),
            sender: sender.into(),
            content: content.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            namespace: "personal".to_string(),
            agent: None,
            session_id: None,
            principal: None,
            provenance: None,
        }
    }

    /// Builder: stamp the origin of this signal. Used by the reflex
    /// runner to mark trigger-fired signals so the pipeline + audit
    /// can distinguish them from user-typed input.
    pub fn with_provenance(mut self, provenance: intent::Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Builder: attach a `Principal` resolved by the adapter from its
    /// auth context. The identity gate consults this when authorizing the
    /// signal's intent.
    pub fn with_principal(mut self, principal: identity::Principal) -> Self {
        self.principal = Some(principal);
        self
    }

    /// Builder: attach a principal from an Option (no-op if None).
    pub fn with_principal_opt(mut self, principal: Option<identity::Principal>) -> Self {
        if let Some(p) = principal {
            self.principal = Some(p);
        }
        self
    }

    /// Builder: set the originating agent identity.
    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    /// Builder: set the memory namespace.
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Builder: set the metadata map.
    pub fn with_metadata(mut self, meta: HashMap<String, String>) -> Self {
        self.metadata = meta;
        self
    }

    /// Builder: set namespace from an Option (no-op if None).
    pub fn with_namespace_opt(mut self, ns: Option<String>) -> Self {
        if let Some(n) = ns {
            self.namespace = n;
        }
        self
    }

    /// Builder: set agent from an Option (no-op if None).
    pub fn with_agent_opt(mut self, agent: Option<String>) -> Self {
        if let Some(a) = agent {
            self.agent = Some(a);
        }
        self
    }

    /// Builder: set session ID for conversation continuity.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Builder: set session ID from an Option (no-op if None).
    pub fn with_session_id_opt(mut self, session_id: Option<String>) -> Self {
        self.session_id = session_id;
        self
    }

    /// Build a Signal from an [`AdapterRequest`], applying defaults for missing optional fields.
    pub fn from_adapter_request(req: AdapterRequest) -> Self {
        Signal::new(
            req.source,
            req.channel.unwrap_or(req.default_channel),
            req.sender.unwrap_or(req.default_sender),
            req.content,
        )
        .with_metadata(req.metadata.unwrap_or_default())
        .with_namespace_opt(req.namespace)
        .with_agent_opt(req.agent)
        .with_session_id_opt(req.session_id)
    }
}

/// Fields from an adapter request used to construct a [`Signal`].
///
/// Replaces the previous 10-parameter positional API to prevent argument mis-ordering.
pub struct AdapterRequest {
    pub source: SignalSource,
    pub content: String,
    pub channel: Option<String>,
    pub sender: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
    pub namespace: Option<String>,
    pub agent: Option<String>,
    pub session_id: Option<String>,
    pub default_channel: String,
    pub default_sender: String,
}

// ─── Response Types ───────────────────────────────────────────────────────────

/// Status of a signal response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseStatus {
    Ok,
    Error,
    Processing,
}

/// Content payload of a signal response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ResponseContent {
    Text(String),
    Json(serde_json::Value),
    Error(String),
}

// ─── Export / Import types ───────────────────────────────────────────────────

pub use storage::{ExportedEpisode, ExportedFact};

/// Memory context included in every response — tracks what memory was used.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Number of semantic facts used to construct the response.
    pub facts_used: usize,
    /// Number of episodic memories used to construct the response.
    pub episodes_used: usize,
}

/// The response to a processed signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResponse {
    pub signal_id: Uuid,
    pub status: ResponseStatus,
    pub response: ResponseContent,
    pub memory_context: MemoryContext,
    /// Session ID for conversation continuity. Clients should send this back
    /// in subsequent signals to maintain the same conversation context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl SignalResponse {
    /// Create a successful text response.
    pub fn ok(signal_id: Uuid, text: impl Into<String>) -> Self {
        Self {
            signal_id,
            status: ResponseStatus::Ok,
            response: ResponseContent::Text(text.into()),
            memory_context: MemoryContext::default(),
            session_id: None,
        }
    }

    /// Create an error response.
    pub fn error(signal_id: Uuid, error: impl Into<String>) -> Self {
        Self {
            signal_id,
            status: ResponseStatus::Error,
            response: ResponseContent::Error(error.into()),
            memory_context: MemoryContext::default(),
            session_id: None,
        }
    }
}

/// Broadcast event emitted after a signal has been processed successfully.
#[derive(Debug, Clone)]
pub struct SignalProcessedEvent {
    pub signal_id: Uuid,
    pub source: SignalSource,
    pub channel: String,
    pub sender: String,
    pub namespace: String,
    pub status: ResponseStatus,
    pub response: String,
    pub facts_used: usize,
    pub episodes_used: usize,
    pub timestamp: DateTime<Utc>,
}

// ─── Pipeline Result ─────────────────────────────────────────────────────────

/// Result of the `prepare()` pipeline phase.
///
/// Either the intent was handled directly (StoreFact, Forget, SystemStatus, Actions)
/// and a complete response is returned, or the pipeline assembled LLM messages
/// and the caller decides whether to use streaming or batch generation.
pub enum PipelineResult {
    /// Intent handled directly. Response is complete.
    Complete(SignalResponse),
    /// Chat/Recall: pipeline done, LLM messages assembled.
    /// Caller chooses streaming vs batch generation.
    LlmReady {
        signal_id: Uuid,
        messages: Vec<cortex::llm::Message>,
        memory_context: MemoryContext,
        session_id: Option<String>,
        user_content: String,
        namespace: String,
        agent: Option<String>,
    },
}

// ─── Signal Adapter Trait ─────────────────────────────────────────────────────

/// Trait implemented by all protocol adapters (HTTP, WebSocket, MCP, gRPC, CLI).
///
/// Each adapter converts protocol-specific messages into Signal values,
/// submits them to SignalProcessor, and delivers the SignalResponse back
/// to the originating client via `send()`.
#[async_trait::async_trait]
pub trait SignalAdapter: Send + Sync {
    /// Return the source type for this adapter.
    fn source(&self) -> SignalSource;

    /// Send a response back to the adapter's client.
    async fn send(&self, response: SignalResponse) -> Result<(), SignalError>;
}

/// Extract text content from a ResponseContent variant.
pub fn response_to_text(content: &ResponseContent) -> String {
    match content {
        ResponseContent::Text(t) => t.clone(),
        ResponseContent::Json(v) => v.to_string(),
        ResponseContent::Error(e) => e.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_public_strips_internal_detail() {
        let storage = SignalError::Storage("disk full at /var/lib/brain/db.sqlite".into());
        let public = storage.to_public();
        assert_eq!(public, PublicError::STORAGE);
        // Sanity-check: nothing in the public payload reveals the path.
        assert!(!public.message.contains("/var/lib"));
        assert!(!public.code.contains("/var/lib"));

        let processing = SignalError::Processing("panic in classify_explicit".into());
        assert_eq!(processing.to_public(), PublicError::PROCESSING);

        let init = SignalError::Init("SQLite migration v22 failed".into());
        assert_eq!(init.to_public(), PublicError::INIT);
    }

    #[test]
    fn signal_new_starts_with_no_provenance() {
        let s = Signal::new(SignalSource::Cli, "ch", "sender", "hello");
        assert!(s.provenance.is_none());
    }

    #[test]
    fn with_provenance_stamps_reflex_origin() {
        let s = Signal::new(SignalSource::Cli, "ch", "sender", "hello").with_provenance(
            intent::Provenance::Reflex {
                trigger: "fs:/tmp/foo".into(),
                raw_input: None,
                ts: chrono::Utc::now(),
            },
        );
        match s.provenance {
            Some(intent::Provenance::Reflex { trigger, .. }) => {
                assert_eq!(trigger, "fs:/tmp/foo");
            }
            other => panic!("expected Reflex provenance, got {other:?}"),
        }
    }

    #[test]
    fn provenance_round_trips_through_json() {
        let s = Signal::new(SignalSource::Cli, "ch", "sender", "hi").with_provenance(
            intent::Provenance::Reflex {
                trigger: "sys:battery_below_20".into(),
                raw_input: Some("{\"battery\":15}".into()),
                ts: chrono::Utc::now(),
            },
        );
        let wire = serde_json::to_string(&s).expect("serialize");
        let parsed: Signal = serde_json::from_str(&wire).expect("deserialize");
        assert!(matches!(
            parsed.provenance,
            Some(intent::Provenance::Reflex { .. })
        ));
    }
}
