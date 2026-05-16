//! Composite reflex — boolean AND/OR combinator over a set of child
//! [`ReflexSource`]s.
//!
//! ## Operators
//!
//! - **`Or`** — merges child streams and passes every event through
//!   unchanged. Useful when one downstream subscriber wants to react
//!   to "any of these triggers fired" without managing N subscriptions.
//!   Original child triggers are preserved so audit correlation stays
//!   stable.
//!
//! - **`And`** — emits one synthetic event when *every* child has
//!   fired at least once within a sliding `window`. Most-recent
//!   firing replaces older state per child; after emission, the
//!   per-child slots reset so the next composite firing requires a
//!   fresh round. Trigger format: `composite:<name>:and`. The payload
//!   lists every contributing child event, so a downstream consumer
//!   can inspect what combined.
//!
//! ## Why AND resets after emission
//!
//! Without reset, a composite would latch: once all children had ever
//! fired, every subsequent child event would re-trigger the composite.
//! Reset makes AND edge-triggered on the moment-of-completion — one
//! composite event per fresh round of all children, no flood.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{ReflexError, ReflexEvent, ReflexSource, ReflexStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    /// Pass through every child event unchanged.
    Or,
    /// Emit one composite event when every child has fired within
    /// the configured window.
    And,
}

#[derive(Debug, Clone)]
pub struct CompositeReflexConfig {
    pub op: CompositeOp,
    /// For [`CompositeOp::And`], the sliding window inside which
    /// every child must fire at least once. Ignored for `Or`.
    /// Default 5s.
    pub window: Duration,
}

impl Default for CompositeReflexConfig {
    fn default() -> Self {
        Self {
            op: CompositeOp::Or,
            window: Duration::from_secs(5),
        }
    }
}

impl CompositeReflexConfig {
    pub fn or() -> Self {
        Self {
            op: CompositeOp::Or,
            ..Self::default()
        }
    }

    pub fn and(window: Duration) -> Self {
        Self {
            op: CompositeOp::And,
            window,
        }
    }
}

pub struct CompositeReflex {
    name: String,
    children: Vec<Arc<dyn ReflexSource>>,
    config: CompositeReflexConfig,
}

impl CompositeReflex {
    pub fn new(
        name: impl Into<String>,
        children: Vec<Arc<dyn ReflexSource>>,
        config: CompositeReflexConfig,
    ) -> Self {
        Self {
            name: name.into(),
            children,
            config,
        }
    }

    pub fn config(&self) -> &CompositeReflexConfig {
        &self.config
    }

    pub fn children_len(&self) -> usize {
        self.children.len()
    }
}

#[async_trait]
impl ReflexSource for CompositeReflex {
    fn name(&self) -> &str {
        &self.name
    }

    async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError> {
        let (out_tx, out_rx) = mpsc::channel::<ReflexEvent>(64);
        let (inner_tx, mut inner_rx) = mpsc::channel::<(usize, ReflexEvent)>(64);

        // Subscribe to every child up front so failures surface
        // synchronously to the caller, not later from the spawned
        // task where the only recourse is a warn! log.
        let mut child_streams = Vec::with_capacity(self.children.len());
        for child in &self.children {
            let stream = Arc::clone(child).subscribe().await?;
            child_streams.push(stream);
        }

        // Per-child pump tasks fan into one ordered channel.
        for (idx, mut stream) in child_streams.into_iter().enumerate() {
            let tx = inner_tx.clone();
            tokio::spawn(async move {
                while let Some(ev) = stream.next().await {
                    if tx.send((idx, ev)).await.is_err() {
                        return;
                    }
                }
            });
        }
        drop(inner_tx);

        let me = Arc::clone(&self);
        tokio::spawn(async move {
            let n = me.children.len();
            let mut latest: Vec<Option<(ReflexEvent, Instant)>> = vec![None; n];
            let window = me.config.window;
            let name = me.name.clone();
            let op = me.config.op;

            loop {
                tokio::select! {
                    msg = inner_rx.recv() => {
                        match msg {
                            Some((idx, ev)) => {
                                match op {
                                    CompositeOp::Or => {
                                        if out_tx.send(ev).await.is_err() {
                                            return;
                                        }
                                    }
                                    CompositeOp::And => {
                                        latest[idx] = Some((ev, Instant::now()));
                                        if let Some(composite) =
                                            try_emit_and(&name, &latest, Instant::now(), window)
                                        {
                                            if out_tx.send(composite).await.is_err() {
                                                return;
                                            }
                                            // Reset for next round so AND
                                            // stays edge-triggered.
                                            for slot in latest.iter_mut() {
                                                *slot = None;
                                            }
                                        }
                                    }
                                }
                            }
                            None => return, // every child stream ended
                        }
                    }
                    _ = out_tx.closed() => return,
                }
            }
        });

        Ok(ReceiverStream::new(out_rx).boxed())
    }
}

/// Pure AND-completion check — pulled out so tests pin window
/// semantics without standing up the polling loop.
///
/// Returns `Some(composite_event)` iff every slot is `Some` and every
/// timestamp is within `window` of `now`. The caller is responsible
/// for resetting `latest` after a successful emission.
pub fn try_emit_and(
    name: &str,
    latest: &[Option<(ReflexEvent, Instant)>],
    now: Instant,
    window: Duration,
) -> Option<ReflexEvent> {
    if latest.is_empty() {
        return None;
    }
    let mut children = Vec::with_capacity(latest.len());
    for slot in latest {
        let (ev, t) = slot.as_ref()?;
        if now.duration_since(*t) > window {
            return None;
        }
        children.push(serde_json::json!({
            "trigger": ev.trigger,
            "payload": ev.payload,
            "ts": ev.ts,
        }));
    }
    Some(ReflexEvent::new(
        format!("composite:{name}:and"),
        serde_json::json!({
            "op": "and",
            "children": children,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NoopReflex;
    use futures::StreamExt;
    use std::collections::HashSet;

    #[tokio::test]
    async fn composite_or_passes_through_every_child_event() {
        let a: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("a", "noop:a"));
        let b: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("b", "noop:b"));
        let composite = Arc::new(CompositeReflex::new(
            "or-test",
            vec![a, b],
            CompositeReflexConfig::or(),
        ));
        let mut stream = composite.subscribe().await.expect("subscribe");

        // Collect both events — NoopReflex emits one each and ends.
        let mut triggers: HashSet<String> = HashSet::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(Duration::from_secs(2), stream.next())
                .await
                .expect("event within timeout")
                .expect("stream still open");
            triggers.insert(ev.trigger);
        }
        assert!(triggers.contains("noop:a"), "missing noop:a");
        assert!(triggers.contains("noop:b"), "missing noop:b");
    }

    #[tokio::test]
    async fn composite_and_emits_one_event_after_all_children_fire() {
        let a: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("a", "noop:a"));
        let b: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("b", "noop:b"));
        let composite = Arc::new(CompositeReflex::new(
            "and-test",
            vec![a, b],
            CompositeReflexConfig::and(Duration::from_secs(5)),
        ));
        let mut stream = composite.subscribe().await.expect("subscribe");

        let ev = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("event within timeout")
            .expect("stream still open");
        assert_eq!(ev.trigger, "composite:and-test:and");
        let children = ev
            .payload
            .get("children")
            .and_then(|v| v.as_array())
            .expect("children array");
        assert_eq!(children.len(), 2);

        // No second event — both NoopReflexes have completed, so the
        // AND state can't be refilled.
        let second = tokio::time::timeout(Duration::from_millis(100), stream.next()).await;
        assert!(second.is_err() || second.unwrap().is_none());
    }

    #[tokio::test]
    async fn composite_and_does_not_fire_until_every_child_fires() {
        // Only one child — but `subscribe` returns a one-shot stream
        // that completes, so the composite can never see the other
        // half. Use one Noop + a stub that subscribes successfully but
        // never emits.
        struct SilentReflex;
        #[async_trait]
        impl ReflexSource for SilentReflex {
            fn name(&self) -> &str {
                "silent"
            }
            async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError> {
                let (_tx, rx) = mpsc::channel::<ReflexEvent>(1);
                Ok(ReceiverStream::new(rx).boxed())
            }
        }

        let a: Arc<dyn ReflexSource> = Arc::new(NoopReflex::simple("a", "noop:a"));
        let silent: Arc<dyn ReflexSource> = Arc::new(SilentReflex);
        let composite = Arc::new(CompositeReflex::new(
            "and-partial",
            vec![a, silent],
            CompositeReflexConfig::and(Duration::from_secs(5)),
        ));
        let mut stream = composite.subscribe().await.expect("subscribe");
        let res = tokio::time::timeout(Duration::from_millis(100), stream.next()).await;
        match res {
            Err(_) => {}   // timeout — no event, good
            Ok(None) => {} // stream ended — also good
            Ok(Some(ev)) => panic!("AND must stay silent but emitted: {ev:?}"),
        }
    }

    #[test]
    fn try_emit_and_returns_none_when_any_slot_is_empty() {
        let ev = ReflexEvent::new("a", serde_json::json!({}));
        let now = Instant::now();
        let latest = vec![Some((ev, now)), None];
        assert!(try_emit_and("c", &latest, now, Duration::from_secs(5)).is_none());
    }

    #[test]
    fn try_emit_and_returns_none_when_timestamp_is_outside_window() {
        let ev_a = ReflexEvent::new("a", serde_json::json!({}));
        let ev_b = ReflexEvent::new("b", serde_json::json!({}));
        let now = Instant::now();
        let stale = now - Duration::from_secs(10);
        let latest = vec![Some((ev_a, stale)), Some((ev_b, now))];
        assert!(try_emit_and("c", &latest, now, Duration::from_secs(5)).is_none());
    }

    #[test]
    fn try_emit_and_emits_when_all_slots_within_window() {
        let ev_a = ReflexEvent::new("a", serde_json::json!({}));
        let ev_b = ReflexEvent::new("b", serde_json::json!({}));
        let now = Instant::now();
        let latest = vec![
            Some((ev_a, now - Duration::from_millis(100))),
            Some((ev_b, now)),
        ];
        let composite = try_emit_and("work", &latest, now, Duration::from_secs(5)).expect("event");
        assert_eq!(composite.trigger, "composite:work:and");
        let children = composite
            .payload
            .get("children")
            .and_then(|v| v.as_array())
            .expect("children");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn try_emit_and_returns_none_for_empty_latest() {
        let now = Instant::now();
        assert!(try_emit_and("c", &[], now, Duration::from_secs(5)).is_none());
    }

    #[test]
    fn config_helpers_set_op_correctly() {
        assert_eq!(CompositeReflexConfig::or().op, CompositeOp::Or);
        assert_eq!(
            CompositeReflexConfig::and(Duration::from_secs(1)).op,
            CompositeOp::And
        );
    }
}
