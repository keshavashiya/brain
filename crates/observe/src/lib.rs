//! # Brain Observability Bus
//!
//! Single source of truth for everything the user might want to see Brain do.
//!
//! - [`BrainEvent`] formalises every consequential moment in the signal lifecycle
//!   (received, classified, routed, executed, audited, budget-crossed, etc.).
//! - [`Observer`] is the trait every consequential code path publishes through.
//! - [`BroadcastObserver`] is the default in-process fanout implementation.
//! - [`Redactor`] scrubs vault-marked secrets from any event payload before publication.
//!
//! The crate is intentionally self-contained — it owns no I/O or persistence;
//! wiring into `crates/signal` and `crates/audit` happens at the processor seam.

pub mod event;
pub mod observer;
pub mod redact;
pub mod sampling;

pub use event::{
    BrainEvent, IntentSummary, OutcomeSummary, PrincipalSummary, SignalSummary, ToolRouteSummary,
};
pub use observer::{BroadcastObserver, ObserveError, Observer, DEFAULT_BROADCAST_CAPACITY};
pub use redact::{Redactor, SENTINEL_PREFIX, SENTINEL_SUFFIX};
pub use sampling::LogSampler;
