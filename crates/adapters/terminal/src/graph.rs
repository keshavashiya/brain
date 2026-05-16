//! Optional graph-mirror hook for the Terminal Bridge.
//!
//! When wired with a `TerminalGraphSink`, every session lifecycle
//! emits two graph events: a `record_open` at session creation that
//! returns the node handles needed to wire the close edge, and a
//! `record_close` that lands the final `terminal_event(close)` node
//! and the second `causal_produced` edge.
//!
//! Defining the trait here (rather than pulling in `hippocampus`)
//! keeps this crate's dep surface storage-free. The concrete
//! `EpisodicGraph`-backed impl lives in `brainos-signal` where both
//! crates are already in scope.

use async_trait::async_trait;
use identity::Principal;

/// Node ids produced by [`TerminalGraphSink::record_open`]. Kept on
/// the [`crate::SessionRegistry`] so the matching `record_close` can
/// wire its edge back to the open event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalGraphHandles {
    /// `node_kind = "tool_call"`, body describes the
    /// `terminal.open` invocation.
    pub tool_call_node_id: String,
    /// `node_kind = "terminal_event"`, body describes the open-side
    /// event for this `session_id`. Acts as the parent of the close
    /// event so traversal can reconstruct the full lifecycle.
    pub open_event_node_id: String,
}

/// Errors a [`TerminalGraphSink`] impl can surface.
#[derive(Debug, thiserror::Error)]
pub enum MirrorError {
    #[error("graph mirror error: {0}")]
    Backend(String),
}

/// The bridge's hook into a graph store. Failures are caller-visible
/// because a graph-write failure during a terminal lifecycle is a
/// data-loss event the operator wants to see — `TerminalSvc` logs at
/// `warn!` and continues so the PTY itself keeps working.
#[async_trait]
pub trait TerminalGraphSink: Send + Sync {
    async fn record_open(
        &self,
        session_id: &str,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        principal: Option<&Principal>,
    ) -> Result<TerminalGraphHandles, MirrorError>;

    async fn record_close(
        &self,
        handles: &TerminalGraphHandles,
        session_id: &str,
        exit_code: i32,
        was_killed: bool,
    ) -> Result<(), MirrorError>;
}
