//! Cron reflex — bridges the existing scheduled-intent table onto the
//! reflex pipeline.
//!
//! Brain's pre-Phase-5 "scheduler" was a 60s poller in `cli::serve`
//! that read `SqlitePool::due_scheduled_intents()` and fired
//! `ProactiveNotification`s directly. That bypasses identity,
//! confirmation, and per-tool breakers — exactly what Phase 5's
//! cardinal rule forbids ("triggers emit signals, never execute").
//!
//! `CronReflex` replaces that direct-execution path with a
//! [`ReflexSource`]: every due row is published as a
//! [`ReflexEvent`] with trigger `"cron:<scheduled_intent_id>"` and a
//! payload carrying the persisted `description` / `cron` / `namespace`.
//! Downstream consumers turn it into a `Signal { provenance::Reflex }`
//! and run the normal pipeline.
//!
//! ## Cron semantics
//!
//! `SqlitePool::due_scheduled_intents()` returns every row with
//! `status = 'scheduled'` — it does **not** evaluate the `cron`
//! expression. CronReflex mirrors that behavior on purpose: this slice
//! is the wiring change, not the cron-parser change. A future slice
//! can add a `cron`-expression evaluator without disturbing the
//! reflex surface.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt;
use storage::SqlitePool;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

use crate::{ReflexError, ReflexEvent, ReflexSource, ReflexStream};

/// Tuning for [`CronReflex`].
#[derive(Debug, Clone)]
pub struct CronReflexConfig {
    /// Interval between `due_scheduled_intents` polls. Default 60s to
    /// match the historical `cli::serve` ticker.
    pub poll_interval: Duration,
    /// Optional namespace filter — when set, only intents matching the
    /// given namespace fire. `None` lets every namespace through.
    pub namespace_filter: Option<String>,
    /// After emitting a `ReflexEvent`, mark the underlying row as
    /// `fired` so the next poll doesn't replay it. Default `true`;
    /// tests and dry-run consumers can flip it off.
    pub mark_fired: bool,
}

impl Default for CronReflexConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(60),
            namespace_filter: None,
            mark_fired: true,
        }
    }
}

impl CronReflexConfig {
    pub fn new(poll_interval: Duration) -> Self {
        Self {
            poll_interval,
            ..Self::default()
        }
    }

    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace_filter = Some(ns.into());
        self
    }

    pub fn mark_fired(mut self, mark: bool) -> Self {
        self.mark_fired = mark;
        self
    }
}

/// Reflex source that polls `scheduled_intents` and emits one event
/// per due row.
pub struct CronReflex {
    name: String,
    pool: SqlitePool,
    config: CronReflexConfig,
}

impl CronReflex {
    pub fn new(name: impl Into<String>, pool: SqlitePool, config: CronReflexConfig) -> Self {
        Self {
            name: name.into(),
            pool,
            config,
        }
    }

    pub fn config(&self) -> &CronReflexConfig {
        &self.config
    }
}

#[async_trait]
impl ReflexSource for CronReflex {
    fn name(&self) -> &str {
        &self.name
    }

    async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError> {
        // Bound is loose — at human-scale cadences (≥ seconds) the
        // channel never fills, but a small buffer absorbs the case
        // where many intents come due in the same tick.
        let (out_tx, out_rx) = mpsc::channel::<ReflexEvent>(64);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(self.config.poll_interval);
            // Skip the immediate tick interval emits on creation —
            // consumers expect the first firing to follow a real poll
            // cadence, not happen synchronously with subscribe.
            ticker.tick().await;
            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        if out_tx.is_closed() {
                            return;
                        }
                        if let Err(e) = self.poll_once(&out_tx).await {
                            warn!(error = %e, "cron reflex poll failed");
                        }
                    }
                    _ = out_tx.closed() => return,
                }
            }
        });

        Ok(ReceiverStream::new(out_rx).boxed())
    }
}

impl CronReflex {
    async fn poll_once(&self, out_tx: &mpsc::Sender<ReflexEvent>) -> Result<(), String> {
        let due = self
            .pool
            .due_scheduled_intents()
            .map_err(|e| e.to_string())?;

        for intent in due {
            if let Some(ns) = &self.config.namespace_filter {
                if &intent.namespace != ns {
                    continue;
                }
            }

            let trigger = format!("cron:{}", intent.id);
            let payload = serde_json::json!({
                "scheduled_intent_id": intent.id,
                "description": intent.description,
                "cron": intent.cron,
                "namespace": intent.namespace,
            });
            info!(
                id = %intent.id,
                description = %intent.description,
                "cron reflex firing scheduled intent"
            );
            let event = ReflexEvent::new(trigger, payload);
            if out_tx.send(event).await.is_err() {
                return Ok(());
            }

            if self.config.mark_fired {
                if let Err(e) = self
                    .pool
                    .update_scheduled_intent_status(&intent.id, "fired")
                {
                    warn!(id = %intent.id, error = %e, "failed to mark scheduled intent fired");
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn make_pool() -> SqlitePool {
        SqlitePool::open_memory().expect("open in-memory pool")
    }

    #[tokio::test]
    async fn cron_reflex_emits_one_event_per_due_intent_and_marks_fired() {
        let pool = make_pool();
        let id = pool
            .insert_scheduled_intent("ship release", Some("0 9 * * 1-5"), "work", None)
            .expect("insert");

        let reflex = Arc::new(CronReflex::new(
            "cron-test",
            pool.clone(),
            CronReflexConfig::new(Duration::from_millis(30)),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");

        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("event within timeout")
            .expect("stream still open");

        assert_eq!(event.trigger, format!("cron:{id}"));
        assert_eq!(
            event
                .payload
                .get("scheduled_intent_id")
                .and_then(|v| v.as_str()),
            Some(id.as_str())
        );
        assert_eq!(
            event.payload.get("description").and_then(|v| v.as_str()),
            Some("ship release")
        );
        assert_eq!(
            event.payload.get("namespace").and_then(|v| v.as_str()),
            Some("work")
        );

        // Drop the subscriber so the background task exits before we
        // assert against the table — otherwise it can fire the same
        // row a second time before we read status.
        drop(stream);
        // Give the task a moment to observe the closed channel.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let remaining = pool.due_scheduled_intents().expect("query");
        assert!(
            remaining.iter().all(|i| i.id != id),
            "fired intent should not appear in due list"
        );
    }

    #[tokio::test]
    async fn cron_reflex_emits_nothing_when_no_intents_are_due() {
        let pool = make_pool();
        let reflex = Arc::new(CronReflex::new(
            "cron-empty",
            pool,
            CronReflexConfig::new(Duration::from_millis(20)),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");

        // Two ticks worth of headroom — if nothing arrives, the
        // empty-table branch is exercised correctly.
        let res = tokio::time::timeout(Duration::from_millis(80), stream.next()).await;
        assert!(res.is_err(), "no events should arrive when table is empty");
    }

    #[tokio::test]
    async fn cron_reflex_respects_namespace_filter() {
        let pool = make_pool();
        let _wrong = pool
            .insert_scheduled_intent("personal task", None, "home", None)
            .expect("insert wrong-ns");
        let right_id = pool
            .insert_scheduled_intent("work task", None, "work", None)
            .expect("insert right-ns");

        let reflex = Arc::new(CronReflex::new(
            "cron-ns",
            pool.clone(),
            CronReflexConfig::new(Duration::from_millis(30)).namespace("work"),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");

        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("event within timeout")
            .expect("stream still open");
        assert_eq!(event.trigger, format!("cron:{right_id}"));

        // No second event should arrive for the filtered-out namespace
        // — bound the wait so this test stays fast.
        let second = tokio::time::timeout(Duration::from_millis(80), stream.next()).await;
        assert!(
            second.is_err(),
            "filter must exclude the 'home' namespace row"
        );
    }

    #[tokio::test]
    async fn cron_reflex_does_not_mark_fired_when_disabled() {
        let pool = make_pool();
        let id = pool
            .insert_scheduled_intent("dry run", None, "work", None)
            .expect("insert");

        let reflex = Arc::new(CronReflex::new(
            "cron-dry",
            pool.clone(),
            CronReflexConfig::new(Duration::from_millis(30)).mark_fired(false),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");

        let _event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("event within timeout")
            .expect("stream still open");

        drop(stream);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let remaining = pool.due_scheduled_intents().expect("query");
        assert!(
            remaining.iter().any(|i| i.id == id),
            "row must remain scheduled when mark_fired=false"
        );
    }

    #[test]
    fn config_defaults_are_sensible() {
        let cfg = CronReflexConfig::default();
        assert_eq!(cfg.poll_interval, Duration::from_secs(60));
        assert!(cfg.namespace_filter.is_none());
        assert!(cfg.mark_fired);
    }
}
