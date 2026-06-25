//! Per-turn telemetry acceptance (`monitoring.telemetry`).
//!
//! DoD: a completed chat turn publishes exactly one
//! `BrainEvent::TurnCompleted` on the observability bus, carrying the turn's
//! token usage and the model/locality that actually served it. A mock provider
//! stands in for the LLM so the turn answers deterministically and reports a
//! known usage tally; the assertion is on the emitted event, not on a live
//! model.

use std::pin::Pin;
use std::sync::Arc;

use brainos_signal::{Signal, SignalProcessor, SignalSource};
use cortex::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk, ToolDef, Usage};
use futures::Stream;

/// Minimal local provider: every generation returns a fixed answer with a
/// known usage tally, so the telemetry event's token fields are predictable.
struct MockLlm;

#[async_trait::async_trait]
impl LlmProvider for MockLlm {
    async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
        Ok(Response::text(
            "A short answer.",
            Some(Usage {
                prompt_tokens: 123,
                completion_tokens: 45,
                total_tokens: 168,
            }),
        ))
    }

    async fn generate_with_tools(
        &self,
        messages: &[Message],
        _tools: &[ToolDef],
    ) -> Result<Response, LlmError> {
        self.generate(messages).await
    }

    async fn generate_stream(
        &self,
        _messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        let chunk = ResponseChunk {
            content: "A short answer.".to_string(),
            is_done: true,
        };
        Ok(Box::pin(futures::stream::once(async move { Ok(chunk) })))
    }

    async fn health_check(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn model(&self) -> &str {
        "mock-model"
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec!["mock-model".to_string()])
    }

    fn is_local(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn chat_turn_emits_turn_completed_telemetry() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    // Telemetry is on by default; pin it so the test is independent of the
    // shipped default.
    config.monitoring.telemetry.enabled = true;

    let observer = observe::BroadcastObserver::new();
    let processor = SignalProcessor::new(config)
        .await
        .unwrap()
        .with_llm(Arc::new(MockLlm))
        .with_observer(observer.clone());

    let mut rx = processor.subscribe_brain_events().expect("observer wired");

    // A plainly conversational message routes to Chat → the unified
    // generation entry point → telemetry.
    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "Tell me a short story about a curious fox.",
    );
    let signal_id = signal.id;

    // Run the turn to completion first (classification can be slow), then
    // drain the buffered bus — broadcast retains the flow's events.
    let resp = processor
        .process(signal)
        .await
        .expect("chat turn succeeded");
    assert!(
        matches!(resp.status, brainos_signal::ResponseStatus::Ok),
        "chat turn returned Ok, got {:?}",
        resp.status
    );

    let mut found = None;
    while let Ok(ev) = rx.try_recv() {
        if let observe::BrainEvent::TurnCompleted { id, .. } = &ev {
            if *id == signal_id {
                found = Some(ev);
            }
        }
    }

    let turn = found.expect("a turn_completed event was published for this turn");
    match turn {
        observe::BrainEvent::TurnCompleted {
            provider,
            model,
            local,
            connectivity,
            input_tokens,
            output_tokens,
            tool_rounds,
            tools_invoked,
            ..
        } => {
            assert_eq!(provider, "mock");
            assert_eq!(model, "mock-model");
            assert!(local, "mock provider is loopback-local");
            assert_eq!(connectivity, "online", "no probe wired → online");
            assert_eq!(input_tokens, 123, "prompt tokens from the mock's usage");
            assert_eq!(output_tokens, 45, "completion tokens from the mock's usage");
            // No tool registry is wired, so the turn is a plain answer.
            assert_eq!(tool_rounds, 0);
            assert_eq!(tools_invoked, 0);
        }
        other => panic!("expected TurnCompleted, got {other:?}"),
    }
}

#[tokio::test]
async fn telemetry_disabled_emits_no_turn_completed() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp_dir.path().to_str().unwrap().to_string();
    config.monitoring.telemetry.enabled = false;

    let observer = observe::BroadcastObserver::new();
    let processor = SignalProcessor::new(config)
        .await
        .unwrap()
        .with_llm(Arc::new(MockLlm))
        .with_observer(observer.clone());

    let mut rx = processor.subscribe_brain_events().expect("observer wired");

    let signal = Signal::new(
        SignalSource::Cli,
        "cli",
        "user",
        "Tell me a short story about a curious fox.",
    );
    let handle = tokio::spawn(async move { processor.process(signal).await });
    let _ = handle.await.unwrap();

    // Drain whatever the flow published; none of it may be a turn_completed.
    let mut saw_turn = false;
    while let Ok(ev) = rx.try_recv() {
        if matches!(ev, observe::BrainEvent::TurnCompleted { .. }) {
            saw_turn = true;
        }
    }
    assert!(!saw_turn, "telemetry disabled → no turn_completed event");
}
