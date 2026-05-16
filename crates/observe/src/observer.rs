use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::broadcast;

use crate::event::BrainEvent;

/// Bus capacity. Slow consumers see `Lagged(n)` rather than block the pipeline.
pub const DEFAULT_BROADCAST_CAPACITY: usize = 4096;

#[derive(Debug, Error)]
pub enum ObserveError {
    #[error("bus closed: no remaining subscribers")]
    BusClosed,
}

/// Single ingestion point for every consequential event in the system.
///
/// Implementations MUST be cheap to clone (via `Arc`) and MUST not block in
/// `publish` — slow subscribers should be dropped, not back-pressure the pipeline.
#[async_trait]
pub trait Observer: Send + Sync {
    /// Publish an event to all current subscribers. Errors are surfaced for
    /// liveness checks (e.g. "is anything listening?") but publishers SHOULD
    /// treat them as informational, not fatal.
    async fn publish(&self, ev: BrainEvent) -> Result<(), ObserveError>;

    /// Subscribe to all future events. Returns a `broadcast::Receiver` with
    /// lag-drop semantics — slow readers see `Err(RecvError::Lagged(n))`.
    fn subscribe(&self) -> broadcast::Receiver<BrainEvent>;
}

/// Default in-process implementation backed by `tokio::sync::broadcast`.
pub struct BroadcastObserver {
    tx: broadcast::Sender<BrainEvent>,
}

impl BroadcastObserver {
    pub fn new() -> Arc<Self> {
        Self::with_capacity(DEFAULT_BROADCAST_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Arc<Self> {
        let (tx, _) = broadcast::channel(capacity);
        Arc::new(Self { tx })
    }

    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

#[async_trait]
impl Observer for BroadcastObserver {
    async fn publish(&self, ev: BrainEvent) -> Result<(), ObserveError> {
        match self.tx.send(ev) {
            Ok(_n) => Ok(()),
            Err(_) => Err(ObserveError::BusClosed),
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<BrainEvent> {
        self.tx.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::BrainEvent;
    use chrono::Utc;
    use tokio::sync::broadcast::error::RecvError;
    use uuid::Uuid;

    fn err_event(msg: &str) -> BrainEvent {
        BrainEvent::Error {
            id: Uuid::new_v4(),
            source: "test".into(),
            message: msg.into(),
            ts: Utc::now(),
        }
    }

    #[tokio::test]
    async fn publish_with_no_subscribers_returns_bus_closed() {
        let obs = BroadcastObserver::new();
        let res = obs.publish(err_event("noone")).await;
        assert!(matches!(res, Err(ObserveError::BusClosed)));
    }

    #[tokio::test]
    async fn publish_reaches_subscriber() {
        let obs = BroadcastObserver::new();
        let mut rx = obs.subscribe();
        obs.publish(err_event("hi")).await.unwrap();

        let got = rx.recv().await.unwrap();
        assert_eq!(got.kind(), "error");
    }

    #[tokio::test]
    async fn slow_subscriber_sees_lagged_not_block() {
        let obs = BroadcastObserver::with_capacity(4);
        let mut rx = obs.subscribe();

        // Overrun the buffer without consuming.
        for i in 0..16 {
            obs.publish(err_event(&format!("burst-{i}"))).await.unwrap();
        }

        // First recv should report lag, then deliver subsequent events normally.
        match rx.recv().await {
            Err(RecvError::Lagged(n)) => assert!(n > 0),
            other => panic!("expected Lagged, got {other:?}"),
        }
        // Drain remainder without panicking.
        while let Ok(ev) = rx.try_recv() {
            assert_eq!(ev.kind(), "error");
        }
    }
}
