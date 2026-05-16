//! No-op reflex source — emits one canned event on subscribe and
//! ends the stream. Used by acceptance tests and downstream consumers
//! that want to verify their wiring without spinning up a real trigger.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use crate::{ReflexError, ReflexEvent, ReflexSource, ReflexStream};

/// Reflex that emits a single configured event and then completes.
pub struct NoopReflex {
    name: String,
    event: ReflexEvent,
}

impl NoopReflex {
    pub fn new(name: impl Into<String>, event: ReflexEvent) -> Self {
        Self {
            name: name.into(),
            event,
        }
    }

    /// Quick constructor for tests: emit one event with the given
    /// trigger and an empty JSON-object payload.
    pub fn simple(name: impl Into<String>, trigger: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            event: ReflexEvent::new(trigger, serde_json::json!({})),
            name,
        }
    }
}

#[async_trait]
impl ReflexSource for NoopReflex {
    fn name(&self) -> &str {
        &self.name
    }

    async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError> {
        let event = self.event.clone();
        Ok(stream::once(async move { event }).boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn noop_reflex_emits_one_event_then_ends() {
        let reflex: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("test", "noop:smoke"));
        let mut stream = reflex.subscribe().await.expect("subscribe");
        let ev = stream.next().await.expect("first event");
        assert_eq!(ev.trigger, "noop:smoke");
        assert!(stream.next().await.is_none(), "stream must end");
    }

    #[tokio::test]
    async fn noop_reflex_carries_configured_payload() {
        let event = ReflexEvent::new("fs:/etc/hosts", serde_json::json!({"kind": "modify"}));
        let reflex = Arc::new(NoopReflex::new("fs-test", event.clone()));
        let mut stream = reflex.subscribe().await.unwrap();
        let got = stream.next().await.unwrap();
        assert_eq!(got.trigger, event.trigger);
        assert_eq!(got.payload, event.payload);
    }

    #[tokio::test]
    async fn noop_reflex_exposes_name() {
        let reflex = NoopReflex::simple("named-source", "trig");
        assert_eq!(reflex.name(), "named-source");
    }
}
