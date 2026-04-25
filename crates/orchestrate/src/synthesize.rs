//! Outcome synthesizer — aggregates step outcomes into user-facing summaries.

use crate::state::{StepState, TaskCounts, TaskState};

/// Generate a human-readable summary of the task's current state.
pub fn summarize_task(state: &TaskState) -> String {
    let counts = state.counts();
    let total = counts.total();

    match state.phase {
        crate::state::TaskPhase::Planning => {
            format!("Planning: \"{}\" — {total} steps identified", state.request)
        }
        crate::state::TaskPhase::AwaitingApproval => {
            format!(
                "Awaiting approval: \"{}\" — {total} steps in plan",
                state.request
            )
        }
        crate::state::TaskPhase::Executing => format_executing(&state.request, &counts, total),
        crate::state::TaskPhase::Completed => {
            format_completed(&state.request, state, &counts, total)
        }
        crate::state::TaskPhase::Cancelled => {
            format!("Cancelled: \"{}\"", state.request)
        }
    }
}

fn format_executing(request: &str, counts: &TaskCounts, total: usize) -> String {
    let mut parts = Vec::new();
    if counts.completed > 0 {
        parts.push(format!("{} done", counts.completed));
    }
    if counts.running > 0 {
        parts.push(format!("{} running", counts.running));
    }
    if counts.awaiting > 0 {
        parts.push(format!("{} awaiting approval", counts.awaiting));
    }
    if counts.failed > 0 {
        parts.push(format!("{} failed", counts.failed));
    }
    let progress = parts.join(", ");
    format!("Executing: \"{request}\" — {progress} (of {total} steps)")
}

fn format_completed(request: &str, state: &TaskState, counts: &TaskCounts, total: usize) -> String {
    let mut lines = Vec::new();

    if state.all_succeeded() {
        lines.push(format!(
            "Completed: \"{request}\" — all {total} steps succeeded"
        ));
    } else {
        lines.push(format!(
            "Completed: \"{request}\" — {}/{total} succeeded, {} failed, {} skipped",
            counts.completed, counts.failed, counts.skipped
        ));
    }

    // List failures with their error messages
    for (step_id, step_state) in &state.step_states {
        if let StepState::Failed { error, .. } = step_state {
            if let Some(step) = state.graph.steps.get(step_id) {
                lines.push(format!("  FAILED: {} — {}", step.description, error));
            }
        }
    }

    lines.join("\n")
}

/// Format the task plan for user review before execution.
pub fn format_plan_for_approval(state: &TaskState) -> String {
    let order = state.graph.topological_order();
    let mut lines = Vec::new();

    lines.push(format!("Task plan for: \"{}\"", state.request));
    lines.push(format!("{} steps:", order.len()));
    lines.push(String::new());

    for (i, step_id) in order.iter().enumerate() {
        if let Some(step) = state.graph.steps.get(step_id) {
            let tier_marker = match step.tier {
                audit::ActionTier::Read => "",
                audit::ActionTier::Write => " [write]",
                audit::ActionTier::Execute => " [exec]",
                audit::ActionTier::Destructive => " [DESTRUCTIVE — requires approval]",
                audit::ActionTier::External => " [EXTERNAL — requires approval]",
            };

            let deps = if step.depends_on.is_empty() {
                String::new()
            } else {
                let dep_indices: Vec<String> = step
                    .depends_on
                    .iter()
                    .filter_map(|dep_id| {
                        order
                            .iter()
                            .position(|id| id == dep_id)
                            .map(|pos| format!("#{}", pos + 1))
                    })
                    .collect();
                format!(" (after {})", dep_indices.join(", "))
            };

            lines.push(format!(
                "  {}. {}{}{}",
                i + 1,
                step.description,
                tier_marker,
                deps
            ));
        }
    }

    lines.push(String::new());
    lines.push(format!(
        "Approve this plan? Reply 'approve {id}' or 'reject {id}'.",
        id = state.id
    ));

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TaskGraph;
    use crate::state::TaskState;
    use crate::step::{StepAction, TaskStep};

    fn test_state() -> TaskState {
        let steps = vec![
            TaskStep {
                id: "s1".to_string(),
                description: "Research".to_string(),
                action: StepAction::Research {
                    query: "test".to_string(),
                },
                depends_on: vec![],
                tier: audit::ActionTier::Read,
                estimated_tokens: 0,
            },
            TaskStep {
                id: "s2".to_string(),
                description: "Implement".to_string(),
                action: StepAction::Implement {
                    spec: "spec".to_string(),
                    agent: "claude-code".to_string(),
                },
                depends_on: vec!["s1".to_string()],
                tier: audit::ActionTier::Execute,
                estimated_tokens: 1000,
            },
        ];
        let graph = TaskGraph::from_steps(steps).unwrap();
        TaskState::new("t1".to_string(), "build a feature".to_string(), graph)
    }

    #[test]
    fn test_summarize_planning() {
        let state = test_state();
        let summary = summarize_task(&state);
        assert!(summary.contains("Planning"));
        assert!(summary.contains("2 steps"));
    }

    #[test]
    fn test_format_plan() {
        let state = test_state();
        let plan = format_plan_for_approval(&state);
        assert!(plan.contains("Research"));
        assert!(plan.contains("Implement"));
        assert!(plan.contains("[exec]"));
    }

    #[test]
    fn test_format_plan_embeds_task_id() {
        let state = test_state();
        let plan = format_plan_for_approval(&state);
        // The prompt must literally include the task ID so the user can copy
        // it back as `approve <id>` / `reject <id>`.
        assert!(
            plan.contains("approve t1") && plan.contains("reject t1"),
            "approval prompt missing literal task id, got:\n{plan}"
        );
        // The old `<nonce>` placeholder must be gone — it was the bug.
        assert!(
            !plan.contains("<nonce>"),
            "literal `<nonce>` placeholder still present"
        );
    }
}
