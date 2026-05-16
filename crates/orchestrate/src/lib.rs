//! # Brain Task Orchestrator
//!
//! Decomposes user requests into executable task plans, tracks state through
//! a dependency DAG, coordinates execution with approval gates at the right
//! moments, and synthesizes outcomes into user-facing summaries.
//!
//! This is the "prefrontal cortex" of Brain OS — the planning and
//! coordination layer that turns "build a feature" into a sequence of
//! researched, planned, implemented, tested, and delivered steps.

mod actions;
mod aggregation;
pub mod decompose;
pub mod graph;
pub mod orchestrator;
pub mod state;
pub mod step;
pub mod synthesize;

pub use decompose::{DecompositionContext, DecompositionError, LlmDecomposer, TaskDecomposer};
pub use graph::{GraphError, RollbackAction, TaskGraph};
pub use orchestrator::{OrchestrateError, TaskOrchestrator};
pub use state::{StepOutcome, StepState, TaskCounts, TaskPhase, TaskState};
pub use step::{StepAction, TaskStep};
pub use synthesize::{format_plan_for_approval, summarize_task};
