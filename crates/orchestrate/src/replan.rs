//! Best-effort corrective replanning after a step failure for
//! [`TaskOrchestrator`]. Bounded by `MAX_REPLAN_ATTEMPTS`.

use tokio_util::sync::CancellationToken;

use crate::orchestrator::{TaskOrchestrator, MAX_REPLAN_ATTEMPTS};
use crate::state::StepState;

impl TaskOrchestrator {
    /// Best-effort corrective replan after a step failure. Asks the
    /// decomposer for a fresh sub-plan given the original goal +
    /// what's already succeeded + the actual error, then splices the
    /// new steps into the graph so the execution loop picks them up
    /// next iteration. Bounded by `MAX_REPLAN_ATTEMPTS`.
    pub(crate) async fn try_replan_after_failure(
        &self,
        task_id: &str,
        failed_step_id: &str,
        failed_step_description: &str,
        error: &str,
        token: &CancellationToken,
    ) {
        // PR-6b: don't burn an LLM call on a task that just got
        // cancelled. The execute loop's next iteration will see the
        // cancellation and break anyway, but skipping the replan saves
        // the round-trip.
        if token.is_cancelled() {
            return;
        }
        // Snapshot the fields we need under a short read lock.
        let (request, completed, attempts) = {
            let tasks = self.tasks.read().await;
            let task = match tasks.get(task_id) {
                Some(t) => t,
                None => return,
            };
            if task.replan_attempts >= MAX_REPLAN_ATTEMPTS {
                tracing::info!(
                    task_id = %task_id,
                    attempts = task.replan_attempts,
                    "replan budget exhausted; leaving plan in failed state"
                );
                return;
            }
            // Stdout per completed step, capped so a single noisy step
            // can't dominate the prompt. The replan LLM uses these to
            // ground its next step in the real data prior steps produced.
            const PER_STEP_OUTPUT_LIMIT: usize = 1500;
            let completed: Vec<crate::decompose::CompletedStepRecap> = task
                .graph
                .topological_order()
                .into_iter()
                .filter_map(|id| {
                    let state = task.step_states.get(&id)?;
                    let StepState::Completed { outcome, .. } = state else {
                        return None;
                    };
                    let step = task.graph.steps.get(&id)?;
                    let trimmed = outcome.stdout.trim();
                    let excerpt = if trimmed.len() > PER_STEP_OUTPUT_LIMIT {
                        let head = &trimmed[..PER_STEP_OUTPUT_LIMIT];
                        format!("{head}\n…[truncated]")
                    } else {
                        trimmed.to_string()
                    };
                    Some(crate::decompose::CompletedStepRecap {
                        description: step.description.clone(),
                        output_excerpt: excerpt,
                    })
                })
                .collect();
            (task.request.clone(), completed, task.replan_attempts)
        };

        let context = crate::decompose::DecompositionContext {
            available_tools: self.available_tools.clone(),
            // Surface the real delegate roster so a repair plan picks an
            // agent that exists, and an unknown one is rejected before the
            // replanned step is spliced in. Empty when no registry is wired.
            available_agents: self.agents.as_ref().map(|r| r.list()).unwrap_or_default(),
            ..Default::default()
        };
        let repair = crate::decompose::RepairContext {
            original_request: request,
            failed_step: failed_step_description.to_string(),
            error: error.to_string(),
            completed,
        };

        tracing::info!(
            task_id = %task_id,
            failed_step_id = %failed_step_id,
            attempt = attempts + 1,
            max = MAX_REPLAN_ATTEMPTS,
            "attempting replan after step failure"
        );

        let replan_call = self.decomposer.replan_after_failure(repair, context);
        let new_steps = tokio::select! {
            biased;
            _ = token.cancelled() => {
                tracing::info!(task_id = %task_id, "replan aborted by cancellation");
                return;
            }
            r = replan_call => match r {
                Ok(steps) if !steps.is_empty() => steps,
                Ok(_) => {
                    tracing::info!(task_id = %task_id, "replan returned empty plan; skipping");
                    return;
                }
                Err(e) => {
                    tracing::warn!(task_id = %task_id, error = %e, "replan failed; leaving plan as-is");
                    return;
                }
            },
        };

        // Splice the new steps in. Each new step's depends_on already
        // references its sibling new steps via UUIDs from build_task_step
        // (via the sequential-fallback in replan_after_failure), so the
        // first new step has no deps and runs immediately on the next
        // execute() loop iteration.
        let mut tasks = self.tasks.write().await;
        let task = match tasks.get_mut(task_id) {
            Some(t) => t,
            None => return,
        };

        let new_ids: Vec<String> = new_steps.iter().map(|s| s.id.clone()).collect();
        match task.graph.add_steps(new_steps) {
            Ok(()) => {
                for id in &new_ids {
                    task.step_states
                        .insert(id.clone(), crate::state::StepState::Pending);
                }
                task.replan_attempts += 1;
                tracing::info!(
                    task_id = %task_id,
                    spliced = new_ids.len(),
                    total_attempts = task.replan_attempts,
                    "replan succeeded; new steps spliced into graph"
                );
            }
            Err(e) => {
                tracing::warn!(task_id = %task_id, error = %e, "splicing replan steps failed");
            }
        }
    }
}
