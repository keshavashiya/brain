//! Per-category dispatch traits — Issue 111.
//!
//! The top-level dispatch in [`crate::SignalProcessor::prepare`] used to
//! be a 37-arm match over [`thalamus::Intent`]. After Issue 149 grouped
//! the variants into 7 categories, that match becomes a 7-arm dispatch
//! on `intent.category()` plus a small (≤16-arm) per-category match
//! inside each sibling module.
//!
//! Each category exposes one trait — [`InspectionHandler`],
//! [`MemoryHandler`], etc. — that owns the dispatch logic for that
//! category's variants. The [`IntentHandler`] super-trait bundles them
//! so a single bound on the processor is enough to express "can
//! dispatch every intent variant". Adding a new category grows the
//! super-trait bounds — every call site fails to compile until the new
//! sub-trait is implemented for the processor.
//!
//! Per-category match arms are exhaustive over the variants
//! [`thalamus::Intent::category`] routes to them; the catch-all is
//! [`unreachable!`] because hitting it means the category routing is
//! out of sync with the per-category dispatch (a bug in either the
//! `Intent::category` map or in a sibling dispatch).

use uuid::Uuid;

use crate::types::*;

/// Per-call values that every category dispatcher needs. Bundled so
/// dispatcher signatures don't grow linearly with what a single handler
/// happens to want. The borrowed lifetime ties to the surrounding
/// `prepare()` call.
pub(crate) struct HandlerContext<'a> {
    pub signal_id: Uuid,
    pub signal: &'a Signal,
    pub importance: f32,
    pub conversation_history: Option<&'a [cortex::llm::Message]>,
    pub procedure_context: &'a [String],
    pub progress: Option<&'a tokio::sync::mpsc::Sender<&'static str>>,
}

/// Closure type the pipeline uses to prepend queued nudges to a final
/// response. `prepare()` builds one of these once from drained
/// notifications and passes it through to every category dispatcher
/// that may produce a `Complete` result.
pub(crate) type NudgeFn<'a> = dyn Fn(SignalResponse) -> SignalResponse + Send + Sync + 'a;

#[async_trait::async_trait]
pub(crate) trait InspectionHandler {
    async fn dispatch_inspection(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

#[async_trait::async_trait]
pub(crate) trait MemoryHandler {
    async fn dispatch_memory(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

#[async_trait::async_trait]
pub(crate) trait ActionHandler {
    async fn dispatch_action(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

#[async_trait::async_trait]
pub(crate) trait LifecycleHandler {
    async fn dispatch_lifecycle(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

#[async_trait::async_trait]
pub(crate) trait GovernanceHandler {
    async fn dispatch_governance(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

#[async_trait::async_trait]
pub(crate) trait CapabilityHandler {
    async fn dispatch_capability(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

#[async_trait::async_trait]
pub(crate) trait ConversationHandler {
    async fn dispatch_conversation(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError>;
}

/// Bundles all seven per-category handler traits. A single bound on the
/// processor type is enough to express "can dispatch every intent
/// variant". Adding a new category grows this list — every site that
/// requires `IntentHandler` fails to compile until the new sub-trait is
/// implemented for the processor.
///
/// Currently unused as a bound (the call site in `pipeline.rs` imports
/// the sub-traits directly so each method resolves), but kept as the
/// canonical "fully wired processor" contract for future test mocks and
/// feature-flagged variants.
#[allow(dead_code)]
pub(crate) trait IntentHandler:
    InspectionHandler
    + MemoryHandler
    + ActionHandler
    + LifecycleHandler
    + GovernanceHandler
    + CapabilityHandler
    + ConversationHandler
{
}

impl<T> IntentHandler for T where
    T: InspectionHandler
        + MemoryHandler
        + ActionHandler
        + LifecycleHandler
        + GovernanceHandler
        + CapabilityHandler
        + ConversationHandler
{
}
