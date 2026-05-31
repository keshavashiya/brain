//! Centralized LLM prompts for the orchestrator.
//!
//! The long planner prompts are loaded from sibling `.txt` files via
//! [`include_str!`] so they're auditable and versionable without burying
//! ~80-line strings in control-flow code; the short per-action prompts live
//! here as named constants for the same reason. Keeping every orchestrator
//! prompt in one module means there is one place to audit "what do we tell
//! the model," and the invariant tests below guard against the prompt drifting
//! away from the code it describes.

/// System prompt for the task decomposer ([`crate::decompose`]).
pub(crate) const DECOMPOSE_SYSTEM: &str = include_str!("prompts/decompose_system.txt");

/// System prompt for the plan-repair / replan path ([`crate::decompose`]).
pub(crate) const REPAIR_SYSTEM: &str = include_str!("prompts/repair_system.txt");

/// System prompt for a `Research` step ([`crate::actions`]).
pub(crate) const RESEARCH_SYSTEM: &str =
    "You are a research assistant. Answer the user's research query \
     concisely with concrete facts and references where possible. \
     If the query asks about a specific codebase or filesystem path, \
     state plainly that you do not have direct access and ask the \
     orchestrator to delegate the work.";

/// System prompt for a `Review` step ([`crate::actions`]).
pub(crate) const REVIEW_SYSTEM: &str = "You are a code/report reviewer. Critique the artifact for \
     completeness, correctness, and obvious gaps. Surface concrete \
     issues, not platitudes. Keep the critique under 200 words.";

#[cfg(test)]
mod tests {
    use super::*;

    /// The decomposer prompt must describe every `StepAction` the planner can
    /// emit. If a new variant is added to `crate::step::StepAction`, this
    /// reminds us to teach the planner about it instead of silently shipping a
    /// prompt that can never produce that step.
    #[test]
    fn decompose_prompt_covers_every_action_type() {
        // Wire names the planner emits in the `action_type` field, one per
        // StepAction variant (Research/Plan/Implement/Execute/Test/Shell/
        // Review/Notify).
        for action in [
            "research",
            "plan",
            "implement",
            "execute",
            "test",
            "shell",
            "review",
            "notify",
        ] {
            assert!(
                DECOMPOSE_SYSTEM.contains(&format!("\"{action}\"")),
                "decompose prompt never mentions action_type \"{action}\"",
            );
        }
    }

    #[test]
    fn prompts_are_nonempty_and_json_only_where_required() {
        assert!(!DECOMPOSE_SYSTEM.trim().is_empty());
        assert!(!REPAIR_SYSTEM.trim().is_empty());
        // Both planner prompts must keep their "JSON only" contract — the
        // parser depends on it.
        assert!(DECOMPOSE_SYSTEM.contains("ONLY valid JSON"));
        assert!(REPAIR_SYSTEM.contains("ONLY valid JSON"));
        assert!(!RESEARCH_SYSTEM.trim().is_empty());
        assert!(!REVIEW_SYSTEM.trim().is_empty());
    }
}
