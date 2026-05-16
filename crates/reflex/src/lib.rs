//! # Brain Reflex
//!
//! Reactive signal sources. A *reflex* is anything that observes the
//! world (filesystem, cron, system state, composite predicates) and
//! emits a [`ReflexEvent`] when its trigger condition fires.
//!
//! ## Cardinal rule
//!
//! **Triggers emit signals — they never execute.** Every firing
//! produces a `Signal { provenance: intent::Provenance::Reflex { … } }`
//! that flows through the normal pipeline (intent classification,
//! identity gate, confirmation, capability routing). This keeps the
//! security model invariant: a reflex cannot bypass the
//! `ConfirmationEngine` or the per-tool breakers because it has no
//! authority to call tools directly.
//!
//! ## Built-in sources
//!
//! - `NoopReflex` — never fires; useful as a placeholder.
//! - `FsReflex` — debounced filesystem watcher built on `notify`.
//! - `CronReflex` — bridges the existing scheduler into the reflex stream.
//! - `SysStateReflex` — edge-triggered rules over pluggable samplers
//!   (battery / network / lock-state, cfg-gated per platform).
//! - `CompositeReflex` — boolean AND/OR combinator over child sources.
//!
//! Reflexes integrate with the `ConfirmationEngine` and the standing-approval
//! store so pre-granted `(verb, principal)` consent can bypass an interactive
//! prompt while still being audited and revocable.

use std::pin::Pin;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod composite;
pub mod cron;
pub mod fs;
mod noop;
pub mod sys;

pub use composite::{CompositeOp, CompositeReflex, CompositeReflexConfig};
pub use cron::{CronReflex, CronReflexConfig};
pub use fs::{FsChange, FsReflex, FsReflexConfig};
pub use noop::NoopReflex;
pub use sys::{
    NetworkState, NoopSampler, SysSnapshot, SysStateReflex, SysStateReflexConfig, SysStateRule,
    SysStateSampler,
};

/// Errors a [`ReflexSource`] can surface at subscribe time. Per-event
/// errors live inside the stream as `ReflexEvent::error_message` so
/// downstream consumers can survive transient blips without tearing
/// the subscription down.
#[derive(Debug, Error)]
pub enum ReflexError {
    #[error("reflex source error: {0}")]
    Backend(String),
}

/// One firing of a reflex. The `trigger` is a stable identifier the
/// reflex picks for itself (e.g. `"fs:/etc/hosts"`, `"cron:nightly"`)
/// — it ends up in `intent::Provenance::Reflex { trigger, … }` so
/// audit and recall can correlate firings to their source.
///
/// `payload` is opaque JSON; each reflex defines its own shape so
/// consumers (the signal pipeline, downstream classifiers) can pull
/// what they need without a shared schema across every trigger
/// flavor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReflexEvent {
    /// Stable trigger identifier. Examples: `"fs:/etc/hosts"`,
    /// `"cron:nightly_cleanup"`, `"sys:battery_below_20"`.
    pub trigger: String,
    /// Opaque per-reflex payload. Convention: an object with at
    /// least one named field describing what changed.
    pub payload: serde_json::Value,
    /// When the trigger condition was observed.
    pub ts: DateTime<Utc>,
}

impl ReflexEvent {
    pub fn new(trigger: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            trigger: trigger.into(),
            payload,
            ts: Utc::now(),
        }
    }
}

/// Stream produced by [`ReflexSource::subscribe`]. Boxed so different
/// concrete `Stream` impls can coexist behind a trait object.
pub type ReflexStream = Pin<Box<dyn Stream<Item = ReflexEvent> + Send + 'static>>;

/// Trait every reflex source implements.
///
/// `subscribe` takes `Arc<Self>` (not `&self`) so the returned
/// stream can outlive any one borrow — most real sources will spawn
/// a background watcher whose lifetime is tied to the source's
/// `Arc`. Sources that don't need shared ownership (`NoopReflex`)
/// just ignore the captured `Arc`.
#[async_trait::async_trait]
pub trait ReflexSource: Send + Sync {
    /// Stable name for logs, telemetry, and audit correlation.
    fn name(&self) -> &str;

    /// Open a subscription. Implementations should return promptly —
    /// stream the work, don't block the call. Errors during stream
    /// setup return `Err`; per-event failures should surface as
    /// payload data inside `ReflexEvent` so the subscription
    /// survives blips.
    async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError>;
}
