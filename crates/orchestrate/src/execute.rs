//! Task execution loop for [`TaskOrchestrator`].
//!
//! Holds `execute` (drive a planned task to a terminal phase) and the
//! per-step `execute_step` dispatcher. Split out of `orchestrator.rs` to
//! keep the construction/state-machine core small; the per-`StepAction`
//! handlers live in `crate::actions`.

use std::collections::HashSet;

use chrono::Utc;
use tokio_util::sync::CancellationToken;

use crate::orchestrator::{OrchestrateError, TaskOrchestrator};
use crate::state::{StepOutcome, StepState, TaskPhase};
use crate::step::StepAction;
use crate::synthesize;

impl TaskOrchestrator {
    /// Execute a previously planned task (after user approval).
    pub async fn execute(&self, task_id: &str) -> Result<String, OrchestrateError> {
        // Confirm the task exists before any state work so a wrong
        // task_id never produces a phantom transition event.
        {
            let tasks = self.tasks.read().await;
            if !tasks.contains_key(task_id) {
                return Err(OrchestrateError::TaskNotFound(task_id.to_string()));
            }
        }
        // PR-6b: clone the task's cancellation token up-front. Every
        // checkpoint below — top of loop, per-step dispatch, the per-
        // action future, the confirmation wait, the replan LLM call —
        // races against `token.cancelled()` so a `cancel()` call mid-
        // step aborts within one polling cycle.
        let token = self.cancel_token_for(task_id).await;
        if token.is_cancelled() {
            // Cancel fired before execute() even started; honor it.
            return Ok(synthesize::summarize_task(
                self.tasks
                    .read()
                    .await
                    .get(task_id)
                    .expect("invariant: task_id is present (checked above)"),
            ));
        }
        self.transition_phase(task_id, TaskPhase::Executing).await;

        tracing::info!(task_id = %task_id, "Starting task execution");

        // Execute steps in topological order, respecting dependencies.
        //
        // `ready_steps` is computed against the *succeeded* set, not the
        // terminal set — a failed step must NOT unblock its dependents.
        // Failure cascades are handled below by marking dependents
        // `Skipped` so the loop still terminates without busy-looping.
        loop {
            if token.is_cancelled() {
                tracing::info!(task_id = %task_id, "execute loop observed cancellation");
                break;
            }
            let ready_steps = {
                let tasks = self.tasks.read().await;
                let task = tasks
                    .get(task_id)
                    .expect("invariant: task inserted by plan(); only state changes after");

                if task.is_complete() {
                    break;
                }

                let succeeded: HashSet<String> = task
                    .step_states
                    .iter()
                    .filter(|(_, s)| s.is_success())
                    .map(|(id, _)| id.clone())
                    .collect();
                // `ready_steps` only checks dep-satisfaction — it does
                // NOT exclude steps that are already terminal. Without
                // this filter a Failed step (which is not in `succeeded`
                // and has no missing deps) would be picked as "ready"
                // again on the next iteration, re-running the failure
                // and re-triggering the replan loop. Only steps whose
                // current state is Pending may be (re)scheduled.
                task.graph
                    .ready_steps(&succeeded)
                    .into_iter()
                    .filter(|id| {
                        matches!(
                            task.step_states.get(id),
                            Some(StepState::Pending) | Some(StepState::Ready)
                        )
                    })
                    .collect::<Vec<_>>()
            };

            if ready_steps.is_empty() {
                // No ready steps but not complete — some steps must be blocked
                // (running or awaiting confirmation). Break to avoid busy-loop.
                break;
            }

            // Execute ready steps (sequentially for now; parallel in future)
            for step_id in &ready_steps {
                if token.is_cancelled() {
                    break;
                }
                self.execute_step(task_id, step_id, &token).await?;
            }
        }

        // Generate summary
        let tasks = self.tasks.read().await;
        let task = tasks
            .get(task_id)
            .expect("invariant: task inserted by plan() and never removed");
        let summary = synthesize::summarize_task(task);

        Ok(summary)
    }

    /// Execute a single step.
    async fn execute_step(
        &self,
        task_id: &str,
        step_id: &str,
        token: &CancellationToken,
    ) -> Result<(), OrchestrateError> {
        // Pre-flight: if cancellation already fired (e.g. between the
        // outer loop's check and us entering this fn), mark the step
        // cancelled and bail without touching the action handlers.
        if token.is_cancelled() {
            self.mark_step_cancelled(task_id, step_id).await;
            return Ok(());
        }
        let (action, tier, description) = {
            let tasks = self.tasks.read().await;
            let task = tasks
                .get(task_id)
                .expect("invariant: task_id always corresponds to a planned task");
            let step = task
                .graph
                .steps
                .get(step_id)
                .expect("invariant: step_id sourced from task.graph.ready_steps()");
            (step.action.clone(), step.tier, step.description.clone())
        };

        // Mark as running
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .expect("invariant: task_id always corresponds to a planned task");
            task.set_step_state(
                step_id,
                StepState::Running {
                    started_at: Utc::now(),
                },
            );
        }

        tracing::info!(task_id = %task_id, step_id = %step_id, step = %description, "Executing step");

        // Check confirmation for destructive/external tiers
        if tier.requires_confirmation() {
            if let Some(confirm) = &self.confirm {
                let spec = confirm::ApprovalSpec::new(&description, tier);
                let nonce = spec.nonce.clone();

                // Mark as awaiting confirmation
                {
                    let mut tasks = self.tasks.write().await;
                    let task = tasks
                        .get_mut(task_id)
                        .expect("invariant: task_id always corresponds to a planned task");
                    task.set_step_state(
                        step_id,
                        StepState::AwaitingConfirmation {
                            nonce: nonce.clone(),
                            since: Utc::now(),
                        },
                    );
                }

                // PR-6b: race the confirmation wait against the task
                // token. A `cancel()` mid-prompt aborts the wait so the
                // step doesn't block forever on a confirmation that will
                // never come.
                let confirm_outcome = tokio::select! {
                    biased;
                    _ = token.cancelled() => {
                        self.mark_step_cancelled(task_id, step_id).await;
                        return Ok(());
                    }
                    r = confirm.request(spec) => r,
                };
                match confirm_outcome {
                    Ok(confirm::ApprovalOutcome::Approved) => {
                        tracing::info!(step = %description, "Step approved");
                    }
                    Ok(outcome) => {
                        let reason = format!("Approval denied: {outcome:?}");
                        let mut tasks = self.tasks.write().await;
                        let task = tasks
                            .get_mut(task_id)
                            .expect("invariant: task_id always corresponds to a planned task");
                        task.set_step_state(step_id, StepState::Cancelled);
                        tracing::info!(step = %description, reason = %reason, "Step cancelled");
                        return Ok(());
                    }
                    Err(e) => {
                        let mut tasks = self.tasks.write().await;
                        let task = tasks
                            .get_mut(task_id)
                            .expect("invariant: task_id always corresponds to a planned task");
                        task.set_step_state(
                            step_id,
                            StepState::Failed {
                                error: format!("Confirmation error: {e}"),
                                retryable: true,
                                failed_at: Utc::now(),
                            },
                        );
                        return Ok(());
                    }
                }
            }
        }

        // Execute the action. PR-6b: race against `token.cancelled()`
        // so an in-flight sandbox/LLM/delegate call aborts mid-flight.
        // Dropping the action future is cancel-safe — none of the
        // handlers hold mutable global state past an await.
        let result = tokio::select! {
            biased;
            _ = token.cancelled() => {
                self.mark_step_cancelled(task_id, step_id).await;
                return Ok(());
            }
            r = async { match &action {
            StepAction::Execute { command, workdir } | StepAction::Test { command, workdir } => {
                self.execute_sandbox_step(command, workdir).await
            }
            StepAction::Shell { command, workdir } => {
                self.execute_shell_step(command, workdir).await
            }
            StepAction::Research { query } => self.execute_research_step(query).await,
            StepAction::Plan { output } => {
                // A `Plan` step that carries no output is effectively a
                // no-op — the LLM emitted a step the executor cannot
                // perform but marked it `plan` so it would silently
                // succeed. Treat that as an honest failure so the user
                // sees that nothing happened, instead of a "succeeded"
                // count that masks an empty result.
                let trimmed = output.trim();
                if trimmed.is_empty() {
                    Err(format!(
                        "Plan step '{description}' had no output to produce — \
                         the planner did not specify what this step should write. \
                         Re-plan with concrete steps (research/execute/implement)."
                    ))
                } else {
                    Ok(StepOutcome {
                        stdout: output.clone(),
                        stderr: String::new(),
                        exit_code: None,
                        artifacts: vec![],
                        summary: summarize_first_line(trimmed),
                    })
                }
            }
            StepAction::Implement { spec, agent } => {
                self.delegate_implement_step(spec, agent).await
            }
            StepAction::Review { artifact } => self.execute_review_step(artifact).await,
            StepAction::Notify { channel, message } => {
                self.execute_notify_step(channel, message).await
            }
        } } => r,
        };

        // Update step state
        let mut tasks = self.tasks.write().await;
        let task = tasks
            .get_mut(task_id)
            .expect("invariant: task_id always corresponds to a planned task");

        match result {
            Ok(outcome) => {
                // Record in audit trail
                if let Some(audit) = &self.audit {
                    let entry = audit::AuditEntry::new(
                        &description,
                        "step executed",
                        &outcome.summary,
                        tier,
                    )
                    .with_source("orchestrator")
                    .with_execution(
                        outcome.stdout.clone(),
                        outcome.stderr.clone(),
                        outcome.exit_code.unwrap_or(0),
                        0, // duration tracked elsewhere
                    );
                    if let Err(e) = audit.record(entry).await {
                        tracing::warn!("Failed to audit step outcome: {e}");
                    }
                }

                task.set_step_state(
                    step_id,
                    StepState::Completed {
                        outcome,
                        completed_at: Utc::now(),
                    },
                );
            }
            Err(error) => {
                // Mirror the success-path audit write so failed steps
                // are recorded in the audit trail too — otherwise a
                // sandbox exit-1 disappears from history once we lifted
                // it out of the Ok arm.
                if let Some(audit) = &self.audit {
                    let entry = audit::AuditEntry::new(&description, "step failed", &error, tier)
                        .with_source("orchestrator")
                        .with_outcome(audit::AuditOutcome::Failure);
                    if let Err(e) = audit.record(entry).await {
                        tracing::warn!("Failed to audit step failure: {e}");
                    }
                }

                task.set_step_state(
                    step_id,
                    StepState::Failed {
                        error: error.clone(),
                        retryable: true,
                        failed_at: Utc::now(),
                    },
                );

                // Mark all transitive dependents `Skipped` so the loop
                // terminates and the user sees an honest status instead
                // of cascading attempts against missing inputs.
                let dependents = task.graph.transitive_dependents(step_id);
                let reason = format!("dependency {step_id} failed");
                for dep_id in dependents {
                    if let Some(state) = task.step_states.get(&dep_id) {
                        if !state.is_terminal() {
                            task.set_step_state(
                                &dep_id,
                                StepState::Skipped {
                                    reason: reason.clone(),
                                },
                            );
                        }
                    }
                }

                // Drop the write lock before the (potentially slow) LLM
                // replan call below. We still own a snapshot of the
                // fields the replan needs.
                drop(tasks);

                // Try to repair the plan if we still have replan budget.
                // Best-effort: a replan failure leaves the task in the
                // standard "failed step + skipped dependents" state.
                self.try_replan_after_failure(task_id, step_id, &description, &error, token)
                    .await;

                // Re-check completion + drive the canonical
                // Reconciling → (Completed | Failed) shape under the
                // state-machine helper.
                let (done, all_succeeded) = {
                    let tasks = self.tasks.read().await;
                    let task = tasks
                        .get(task_id)
                        .expect("invariant: task_id always corresponds to a planned task");
                    (task.is_complete(), task.all_succeeded())
                };
                if done {
                    self.transition_phase(task_id, TaskPhase::Reconciling).await;
                    let terminal = if all_succeeded {
                        TaskPhase::Completed
                    } else {
                        TaskPhase::Failed
                    };
                    self.transition_phase(task_id, terminal).await;
                    tracing::info!(task_id = %task_id, terminal = %terminal.as_str(), "Task complete");
                }
                return Ok(());
            }
        }

        // Drop the write lock before the I/O-bound transition_phase
        // calls below — they take their own lock for the brief in-mem
        // flip and we don't want to hold the executor's lock through
        // the audit-row write and observer publish.
        drop(tasks);
        let (done, all_succeeded) = {
            let tasks = self.tasks.read().await;
            let task = tasks
                .get(task_id)
                .expect("invariant: task_id always corresponds to a planned task");
            (task.is_complete(), task.all_succeeded())
        };
        if done {
            self.transition_phase(task_id, TaskPhase::Reconciling).await;
            let terminal = if all_succeeded {
                TaskPhase::Completed
            } else {
                TaskPhase::Failed
            };
            self.transition_phase(task_id, terminal).await;
            tracing::info!(task_id = %task_id, terminal = %terminal.as_str(), "Task complete");
        }

        Ok(())
    }
}

/// First non-empty line of `s` truncated to 160 chars — used for short
/// step summaries surfaced in the user-facing task report.
fn summarize_first_line(s: &str) -> String {
    let line = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("Plan produced");
    if line.chars().count() > 160 {
        let truncated: String = line.chars().take(157).collect();
        format!("{truncated}…")
    } else {
        line.to_string()
    }
}
