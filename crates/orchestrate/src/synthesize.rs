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
    if counts.skipped > 0 {
        parts.push(format!("{} skipped", counts.skipped));
    }
    if counts.cancelled > 0 {
        parts.push(format!("{} cancelled", counts.cancelled));
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
        // Include cancelled in the headline so a 5-of-5 plan never shows
        // "2/5 succeeded, 2 failed, 0 skipped" with one step silently
        // dropped from the count.
        let mut headline = format!(
            "Completed: \"{request}\" — {}/{total} succeeded, {} failed, {} skipped",
            counts.completed, counts.failed, counts.skipped
        );
        if counts.cancelled > 0 {
            headline.push_str(&format!(", {} cancelled", counts.cancelled));
        }
        lines.push(headline);
    }

    // Walk steps in topological (display) order so summaries don't come
    // back in HashMap iteration order.
    let order = state.graph.topological_order();
    let mut had_results = false;
    for (i, step_id) in order.iter().enumerate() {
        let Some(step) = state.graph.steps.get(step_id) else {
            continue;
        };
        match state.step_states.get(step_id) {
            Some(StepState::Completed { outcome, .. }) => {
                had_results = true;
                let summary = if outcome.summary.trim().is_empty() {
                    "(no output)".to_string()
                } else {
                    outcome.summary.clone()
                };
                lines.push(format!("  {}. ✓ {} — {summary}", i + 1, step.description));
            }
            Some(StepState::Failed { error, .. }) => {
                had_results = true;
                lines.push(format!("  {}. ✗ {} — {error}", i + 1, step.description));
            }
            Some(StepState::Skipped { reason }) => {
                had_results = true;
                lines.push(format!(
                    "  {}. — {} (skipped: {reason})",
                    i + 1,
                    step.description
                ));
            }
            Some(StepState::Cancelled) => {
                had_results = true;
                lines.push(format!("  {}. — {} (cancelled)", i + 1, step.description));
            }
            _ => {}
        }
    }

    // Stitch any free-text artifacts the user is likely to want — the
    // `Plan` step's `output` field and `Research`/`Review` LLM responses
    // already land in `outcome.stdout`. Surface the longest one as the
    // task's primary deliverable when present, so the chat answer is the
    // actual report rather than a status line.
    if had_results {
        if let Some(report) = pick_primary_artifact(state, &order) {
            lines.push(String::new());
            lines.push("─── Result ───".to_string());
            lines.push(report);
        }
    }

    lines.join("\n")
}

/// Pick the single most useful free-text artifact from the task's
/// successful steps. Heuristic: longest non-empty `stdout` from any
/// `Completed` step, since `Research`/`Review`/`Plan` all land their
/// output there and that's almost always the thing the user asked for.
fn pick_primary_artifact(state: &TaskState, order: &[String]) -> Option<String> {
    let mut best: Option<&str> = None;
    for step_id in order {
        if let Some(StepState::Completed { outcome, .. }) = state.step_states.get(step_id) {
            let s = outcome.stdout.trim();
            if s.is_empty() {
                continue;
            }
            match best {
                Some(b) if s.len() <= b.len() => {}
                _ => best = Some(s),
            }
        }
    }
    best.map(|s| s.to_string())
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
            // Only flag tiers that materially change the user's risk
            // exposure. Read/Write/Execute are routine; show them
            // unmarked. Destructive/External *must* be visible because
            // they imply a separate per-step confirmation later.
            let tier_marker = match step.tier {
                audit::ActionTier::Read | audit::ActionTier::Write | audit::ActionTier::Execute => {
                    ""
                }
                audit::ActionTier::Destructive => " (destructive — extra confirmation)",
                audit::ActionTier::External => " (external call — extra confirmation)",
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
    lines.push(
        "Reply `approve` to run it, or `reject` to discard.\n\
         (You can also reply `approve <id>` if multiple plans are pending — id: "
            .to_string()
            + &state.id
            + ")",
    );

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
        // Routine `Execute` tier no longer renders a `[exec]` tag — only
        // destructive/external tiers get a marker now.
        assert!(!plan.contains("[exec]"));
        // Bare `approve` shortcut must be advertised.
        assert!(plan.contains("`approve`") && plan.contains("`reject`"));
    }

    #[test]
    fn test_format_plan_embeds_task_id() {
        let state = test_state();
        let plan = format_plan_for_approval(&state);
        // The plan still surfaces the task id in case multiple plans are
        // pending and the user needs to disambiguate via `approve <id>`.
        assert!(
            plan.contains("t1"),
            "approval prompt missing literal task id, got:\n{plan}"
        );
        // The old `<nonce>` placeholder must be gone — it was the bug.
        assert!(
            !plan.contains("<nonce>"),
            "literal `<nonce>` placeholder still present"
        );
    }

    #[test]
    fn test_format_plan_marks_destructive_tier() {
        let steps = vec![TaskStep {
            id: "rm".to_string(),
            description: "drop tables".to_string(),
            action: StepAction::Plan {
                output: "DROP TABLE x".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Destructive,
            estimated_tokens: 0,
        }];
        let graph = TaskGraph::from_steps(steps).unwrap();
        let state = TaskState::new("t1".to_string(), "drop".to_string(), graph);
        let plan = format_plan_for_approval(&state);
        assert!(
            plan.contains("destructive"),
            "destructive tier must be flagged, got:\n{plan}"
        );
    }
}
