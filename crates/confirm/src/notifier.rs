//! Approval delivery hook.
//!
//! The confirmation engine writes pending approval rows to SQLite and
//! polls for a response. By itself it has no way to *tell* the user that
//! an approval is waiting — that's the channel layer's job. To avoid a
//! direct `confirm → channel` crate coupling (which would invert the
//! current dependency direction), the engine speaks to an
//! [`ApprovalNotifier`] trait. A concrete implementation that fans out
//! through `channel::ChannelDispatcher` lives in the `signal` crate.
//!
//! The notifier is fire-and-forget from the engine's perspective: it is
//! invoked once per approval request, errors are logged but don't fail
//! the request, and timing-out is still the engine's responsibility (the
//! notifier may have failed to deliver, but the engine times out the
//! same way it always did).

use async_trait::async_trait;

use crate::nonce::ApprovalSpec;

/// Delivers approval prompts to the user. The engine calls this once
/// when a pending approval is created. Implementations should be
/// non-blocking past their internal send budget — the engine awaits the
/// future, but a slow implementation directly delays auto-approval
/// auditing.
#[async_trait]
pub trait ApprovalNotifier: Send + Sync {
    /// Push the approval prompt out. Errors are logged and dropped — the
    /// engine never aborts a request just because notification failed.
    async fn notify(&self, spec: &ApprovalSpec) -> Result<(), NotifyError>;
}

/// Surface error type for notifier implementations. Kept generic so the
/// channel crate and other backends can share the same trait.
#[derive(Debug, thiserror::Error)]
pub enum NotifyError {
    #[error("delivery failed: {0}")]
    Delivery(String),
}
