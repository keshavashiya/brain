//! In-flight signal cancellation infrastructure.
//!
//! Each running `process()` call registers a `tokio::sync::Notify` in the
//! [`SignalProcessor::cancel_registry`] keyed by signal id. The pipeline
//! awaits the notify on long-running checkpoints (LLM generation, etc.)
//! and short-circuits with a [`cancelled_response`] when it fires. The
//! [`CancelGuard`] RAII type guarantees the registry entry is removed when
//! `process()` returns, even on panic.

use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

impl SignalProcessor {
    /// Register a cancellation notify for an in-flight signal and return a
    /// handle the pipeline can await. Idempotent — if a notify already exists
    /// for this id (re-entry), the existing one is returned so any pending
    /// cancel still fires on the new pipeline instance.
    pub async fn register_cancel(&self, signal_id: Uuid) -> std::sync::Arc<tokio::sync::Notify> {
        let mut reg = self.observability.cancel_registry.lock().await;
        reg.entry(signal_id)
            .or_insert_with(|| std::sync::Arc::new(tokio::sync::Notify::new()))
            .clone()
    }

    /// Remove the cancellation notify for a signal. Called from `CancelGuard::drop`
    /// inside the standard pipeline; adapters that drive their own LLM loop
    /// (e.g. WS streaming) also call this when their stream finishes so the
    /// registry entry doesn't leak.
    pub fn unregister_cancel(&self, signal_id: Uuid) {
        // Best-effort: avoid blocking the drop path on the lock. If the lock
        // is held, the entry will be GC'd by the next `register_cancel` call
        // for the same id, or stay live until the process restarts (rare).
        let registry = std::sync::Arc::clone(&self.observability.cancel_registry);
        tokio::spawn(async move {
            registry.lock().await.remove(&signal_id);
        });
    }

    /// Trigger cancellation for an in-flight signal. Returns `true` if a
    /// notify was registered; `false` if the target id is unknown.
    pub async fn cancel_signal(&self, signal_id: Uuid) -> bool {
        let reg = self.observability.cancel_registry.lock().await;
        match reg.get(&signal_id) {
            Some(notify) => {
                notify.notify_waiters();
                true
            }
            None => false,
        }
    }

    /// Build the response for a signal that was cancelled mid-flight.
    /// Also publishes a `BrainEvent::Error { source: "cancelled" }`
    /// correlated to the cancelled signal's id.
    pub(super) async fn cancelled_response(
        &self,
        signal_id: Uuid,
        signal: &Signal,
    ) -> SignalResponse {
        if let Some(observer) = &self.observability.observer {
            let ev = observe::BrainEvent::Error {
                id: signal_id,
                source: "cancelled".into(),
                message: format!("signal {signal_id} cancelled by Intent::CancelSignal"),
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }
        SignalResponse {
            signal_id,
            status: ResponseStatus::Error,
            response: ResponseContent::Text(format!(
                "Signal {} cancelled before completion.",
                signal.id
            )),
            memory_context: MemoryContext::default(),
            session_id: None,
        }
    }
}

/// RAII guard that drops a signal's cancel registry entry when the pipeline
/// returns — whether normally, via early-return, or via panic.
pub(super) struct CancelGuard<'a> {
    pub(super) processor: &'a SignalProcessor,
    pub(super) signal_id: Uuid,
}

impl<'a> Drop for CancelGuard<'a> {
    fn drop(&mut self) {
        self.processor.unregister_cancel(self.signal_id);
    }
}
