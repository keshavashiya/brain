//! Reflex → Signal runner.
//!
//! Bridges any [`reflex::ReflexSource`] into the signal pipeline so a
//! firing trigger produces a `Signal` that flows through identity,
//! confirmation, and dispatch like any user-typed input.
//!
//! ## Cardinal rule
//!
//! Reflexes have **no execution path of their own**. Every firing becomes
//! a Signal; the pipeline does the work. That means reflexes inherit
//! identity gating, the inline confirmation gate (bypassable per-(agent,
//! verb) by [`confirm::StandingApprovalStore`]), per-tool breakers, the
//! resilient MCP host, and audit — without any code in this module.
//!
//! The caller supplies a `signal_builder` closure that converts each
//! [`reflex::ReflexEvent`] into a [`Signal`]. Keeping that mapping in the
//! caller (instead of pinning a payload schema here) means reflexes
//! whose events carry different shapes (filesystem path, cron entry id,
//! battery percentage) all reach the pipeline with sensible content
//! without a one-size-fits-all envelope.

use std::sync::Arc;

use futures::stream::StreamExt;
use reflex::{ReflexError, ReflexEvent, ReflexSource};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::{Signal, SignalProcessor};

/// Subscribe to `source` and convert each emitted [`ReflexEvent`] into a
/// [`Signal`] via `signal_builder`, then dispatch through
/// `processor.process`.
///
/// Returns the [`JoinHandle`] for the spawned background task. Drop it
/// to detach; abort it for a hard stop. The task exits naturally when
/// the underlying reflex stream completes.
///
/// `name` is used in tracing spans so logs identify which reflex is
/// firing — useful when many reflexes share one processor.
pub async fn spawn_reflex<F>(
    name: impl Into<String>,
    source: Arc<dyn ReflexSource>,
    processor: Arc<SignalProcessor>,
    signal_builder: F,
) -> Result<JoinHandle<()>, ReflexError>
where
    F: Fn(ReflexEvent) -> Signal + Send + Sync + 'static,
{
    let name = name.into();
    let mut stream = source.subscribe().await?;
    let handle = tokio::spawn(async move {
        while let Some(event) = stream.next().await {
            let trigger = event.trigger.clone();
            let signal = signal_builder(event);
            debug!(
                reflex = %name,
                trigger = %trigger,
                signal_id = %signal.id,
                "reflex firing -> pipeline"
            );
            if let Err(e) = processor.process(signal).await {
                warn!(
                    reflex = %name,
                    trigger = %trigger,
                    error = %e,
                    "reflex-triggered signal failed in pipeline"
                );
            }
        }
        debug!(reflex = %name, "reflex stream ended");
    });
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SignalSource;
    use brain_core::BrainConfig;
    use reflex::NoopReflex;

    async fn make_processor() -> Arc<SignalProcessor> {
        let temp = tempfile::tempdir().unwrap();
        let mut config = BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);
        Arc::new(processor)
    }

    #[tokio::test]
    async fn spawn_reflex_drives_one_event_through_processor() {
        let processor = make_processor().await;
        let source: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("t", "noop:smoke"));

        let handle = spawn_reflex("t", source, processor.clone(), |_ev| {
            // Use `/approval-list` — a slash classifier path that
            // resolves synchronously and is unguarded, so the pipeline
            // returns without hitting the LLM or the confirmation gate.
            // Confirms only that the runner subscribes, builds a signal,
            // and dispatches it through `process()`.
            Signal::new(
                SignalSource::Cli,
                "reflex",
                "reflex-agent",
                "/approval-list",
            )
        })
        .await
        .expect("spawn");

        // NoopReflex emits one event then ends — the runner task
        // should exit on its own.
        let res = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
        assert!(res.is_ok(), "runner should exit after stream ends");
    }
}
