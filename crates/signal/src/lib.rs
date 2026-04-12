//! # Brain Signal Processor
//!
//! Central hub that converts all input signals (CLI, HTTP, WebSocket, MCP, gRPC)
//! into a unified Signal type and routes them through the Brain pipeline.
//!
//! The SignalProcessor wires together:
//! - Thalamus (intent classification)
//! - Amygdala (importance scoring)
//! - Hippocampus (episodic + semantic memory)
//! - Cortex (LLM reasoning + context assembly)
//! - NotificationRouter (proactive delivery)

pub mod notification;
pub mod types;

mod constructors;
mod exchange;
mod pipeline;
mod recall;
mod streaming;
mod wiring;

// Re-export all public types so `use signal::X;` continues to work.
pub use types::*;

// ─── Signal Processor ─────────────────────────────────────────────────────────

/// Central processor that wires all Brain subsystems together.
///
/// One instance is shared across all adapters. Each incoming Signal is routed
/// through intent classification → importance scoring → memory → LLM → response.
pub struct SignalProcessor {
    config: brain_core::BrainConfig,
    classifier: thalamus::IntentClassifier,
    importance: amygdala::ImportanceScorer,
    episodic: hippocampus::EpisodicStore,
    semantic: Option<hippocampus::SemanticStore>,
    embedder: tokio::sync::Mutex<Option<hippocampus::Embedder>>,
    /// Actual output dimension of the active embedding provider (probed at startup).
    embedding_dim: usize,
    recall_engine: hippocampus::RecallEngine,
    llm: std::sync::Arc<dyn cortex::LlmProvider>,
    context_assembler: cortex::context::ContextAssembler,
    procedures: cerebellum::ProcedureStore,
    events_tx: tokio::sync::broadcast::Sender<SignalProcessedEvent>,
    /// Notification router for proactive message delivery (set via builder).
    notification_router: Option<notification::NotificationRouter>,
    /// Action dispatcher for executing tool intents (set via builder).
    action_dispatcher: Option<cortex::actions::ActionDispatcher>,
    /// Cross-subsystem metrics (embedding, consolidation, circuit breaker, intent).
    metrics: std::sync::Arc<brain_core::metrics::SubsystemMetrics>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_signal_new() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello world");
        assert!(!signal.id.is_nil());
        assert_eq!(signal.source, SignalSource::Cli);
        assert_eq!(signal.channel, "cli");
        assert_eq!(signal.sender, "user");
        assert_eq!(signal.content, "hello world");
        assert!(signal.metadata.is_empty());
    }

    #[test]
    fn test_signal_response_ok() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "success");
        assert_eq!(resp.signal_id, id);
        assert_eq!(resp.status, ResponseStatus::Ok);
        assert!(matches!(resp.response, ResponseContent::Text(_)));
        assert_eq!(resp.memory_context.facts_used, 0);
        assert_eq!(resp.memory_context.episodes_used, 0);
    }

    #[test]
    fn test_signal_response_error() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::error(id, "something went wrong");
        assert_eq!(resp.status, ResponseStatus::Error);
        assert!(matches!(resp.response, ResponseContent::Error(_)));
    }

    #[test]
    fn test_memory_context_default() {
        let ctx = MemoryContext::default();
        assert_eq!(ctx.facts_used, 0);
        assert_eq!(ctx.episodes_used, 0);
    }

    #[test]
    fn test_signal_source_serde() {
        let sources = vec![
            SignalSource::Cli,
            SignalSource::Http,
            SignalSource::WebSocket,
            SignalSource::Mcp,
            SignalSource::Grpc,
        ];
        for s in &sources {
            let json = serde_json::to_string(s).unwrap();
            let back: SignalSource = serde_json::from_str(&json).unwrap();
            assert_eq!(s, &back);
        }
    }

    #[test]
    fn test_signal_serde() {
        let signal = Signal::new(SignalSource::Http, "http", "apiclient", "Remember coffee");
        let json = serde_json::to_string(&signal).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(signal.id, back.id);
        assert_eq!(signal.content, back.content);
    }

    #[test]
    fn test_signal_with_agent() {
        let signal =
            Signal::new(SignalSource::Http, "http", "apiclient", "hello").with_agent("claude-code");
        assert_eq!(signal.agent.as_deref(), Some("claude-code"));

        // Serialization round-trip preserves agent
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("claude-code"));
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agent.as_deref(), Some("claude-code"));
    }

    #[test]
    fn test_signal_without_agent_omits_field() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello");
        assert!(signal.agent.is_none());
        let json = serde_json::to_string(&signal).unwrap();
        // skip_serializing_if = "Option::is_none" should omit agent entirely
        assert!(!json.contains("agent"));
    }

    #[test]
    fn test_signal_response_serde() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "hello");
        let json = serde_json::to_string(&resp).unwrap();
        let back: SignalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.signal_id, back.signal_id);
        assert_eq!(resp.status, back.status);
    }
}
