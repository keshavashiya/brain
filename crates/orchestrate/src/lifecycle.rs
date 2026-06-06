//! Task lifecycle for [`TaskOrchestrator`]: planning entry, read queries,
//! cancellation, and the canonical phase-transition state machine.

use chrono::Utc;
use tokio_util::sync::CancellationToken;

use crate::decompose::DecompositionContext;
use crate::graph::TaskGraph;
use crate::orchestrator::{OrchestrateError, TaskOrchestrator};
use crate::state::{StepState, TaskPhase, TaskState};
use crate::synthesize;

impl TaskOrchestrator {
    /// Decompose a user request into a task plan.
    /// Returns the task ID and a formatted plan for user review.
    pub async fn plan(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<(String, String), OrchestrateError> {
        tracing::info!(request = %request, "Decomposing task");

        let steps = self.decomposer.decompose(request, context).await?;
        let graph = TaskGraph::from_steps(steps)?;

        let task_id = uuid::Uuid::new_v4().to_string();
        let task_state = TaskState::new(task_id.clone(), request.to_string(), graph);

        let plan_text = synthesize::format_plan_for_approval(&task_state);

        // Record in audit trail
        if let Some(audit) = &self.audit {
            let entry = audit::AuditEntry::new(
                request,
                "decomposed into task plan",
                &plan_text,
                audit::ActionTier::Read,
            )
            .with_source("orchestrator");
            if let Err(e) = audit.record(entry).await {
                tracing::warn!("Failed to audit task plan: {e}");
            }
        }

        self.tasks.write().await.insert(task_id.clone(), task_state);
        self.cancel_tokens
            .write()
            .await
            .insert(task_id.clone(), CancellationToken::new());

        // State-machine: emit the initial `planning` entry, then
        // transition to AwaitingApproval. Both events are visible to the
        // observer and persisted to `task_states` (if a pool is wired).
        self.record_initial_planning(&task_id).await;
        self.transition_phase(&task_id, TaskPhase::AwaitingApproval)
            .await;

        tracing::info!(task_id = %task_id, "Task plan created");
        Ok((task_id, plan_text))
    }
    /// Get the current state of a task.
    pub async fn get_task(&self, task_id: &str) -> Option<TaskState> {
        self.tasks.read().await.get(task_id).cloned()
    }

    /// Return task IDs currently in the `AwaitingApproval` phase. Used by
    /// the signal pipeline to resolve bare `approve` / `reject` (no id)
    /// to the single pending plan when there's exactly one.
    pub async fn pending_approvals(&self) -> Vec<String> {
        self.tasks
            .read()
            .await
            .iter()
            .filter(|(_, t)| t.phase == TaskPhase::AwaitingApproval)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// List all active tasks.
    pub async fn list_tasks(&self) -> Vec<(String, String, TaskPhase)> {
        self.tasks
            .read()
            .await
            .iter()
            .map(|(id, t)| (id.clone(), t.request.clone(), t.phase))
            .collect()
    }

    /// Cancel a task. Flips all non-terminal step states to `Cancelled`,
    /// transitions the task phase to `Cancelled`, and (PR-6b) fires the
    /// per-task [`CancellationToken`] so any in-flight step future
    /// observing the token aborts within one polling cycle — without
    /// PR-6b, cancellation would have to wait for the current step to
    /// finish on its own.
    pub async fn cancel(&self, task_id: &str) -> Result<(), OrchestrateError> {
        {
            let mut tasks = self.tasks.write().await;
            let task = tasks
                .get_mut(task_id)
                .ok_or_else(|| OrchestrateError::TaskNotFound(task_id.to_string()))?;
            for (_, state) in task.step_states.iter_mut() {
                if !state.is_terminal() {
                    *state = StepState::Cancelled;
                }
            }
        }
        self.transition_phase(task_id, TaskPhase::Cancelled).await;
        // Fire the cancellation token AFTER state has already been
        // flipped to Cancelled — that way a select-loser that races to
        // overwrite step state with Cancelled is a no-op, not a write
        // that could clobber a Completed/Failed transition that
        // legitimately landed first.
        if let Some(t) = self.cancel_tokens.read().await.get(task_id) {
            t.cancel();
        }
        Ok(())
    }

    /// State-machine helper. The single canonical mutator of
    /// [`TaskState::phase`]: takes the write lock just long enough to
    /// flip the in-memory field, then releases it before doing
    /// I/O-bound work (audit row write + observer publish). Idempotent
    /// for terminal transitions — if a task is already in a terminal
    /// phase, the helper is a no-op so cancel-then-complete races stay
    /// well-defined.
    pub(crate) async fn transition_phase(&self, task_id: &str, to: TaskPhase) {
        // Read prior phase + write the new one under one lock. The
        // bound block guarantees the guard drops before the async I/O
        // below so other handlers aren't blocked on the disk write.
        let from = {
            let mut tasks = self.tasks.write().await;
            let task = match tasks.get_mut(task_id) {
                Some(t) => t,
                None => return,
            };
            if task.phase.is_terminal() && task.phase != to {
                // Already done — refuse to flip out of a terminal
                // state so a late completion doesn't overwrite a
                // cancellation.
                tracing::debug!(
                    task_id = %task_id,
                    from = %task.phase.as_str(),
                    to = %to.as_str(),
                    "ignoring transition out of terminal state"
                );
                return;
            }
            if task.phase == to {
                return;
            }
            let from = task.phase;
            task.phase = to;
            if to.is_terminal() {
                task.completed_at = Some(Utc::now());
            }
            from
        };

        // Audit table append (best-effort — a write failure is logged
        // and we proceed so the in-memory phase update isn't undone).
        if let Some(pool) = &self.state_pool {
            let task_id_owned = task_id.to_string();
            let state_str = to.as_str();
            let res = pool.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO task_states (task_id, state) VALUES (?1, ?2)",
                    rusqlite::params![task_id_owned, state_str],
                )?;
                Ok(())
            });
            if let Err(e) = res {
                tracing::warn!(
                    task_id = %task_id,
                    state = %to.as_str(),
                    error = %e,
                    "task_states row append failed"
                );
            }
        }

        // Observer publish (best-effort, same rationale).
        if let Some(observer) = &self.observer {
            let event = observe::BrainEvent::TaskStateChange {
                id: uuid::Uuid::new_v4(),
                task_id: task_id.to_string(),
                from: from.as_str().to_string(),
                to: to.as_str().to_string(),
                ts: Utc::now(),
            };
            let _ = observer.publish(event).await;
        }

        tracing::info!(
            task_id = %task_id,
            from = %from.as_str(),
            to = %to.as_str(),
            "task phase transition"
        );
    }

    /// Convenience: emit the initial Planning transition (`from = "none"`).
    /// Called from [`plan`] right after the task is inserted into the
    /// active map, so the audit table records the task's birth before
    /// any subsequent state moves.
    pub(crate) async fn record_initial_planning(&self, task_id: &str) {
        if let Some(pool) = &self.state_pool {
            let task_id_owned = task_id.to_string();
            let res = pool.with_conn(|conn| {
                conn.execute(
                    "INSERT INTO task_states (task_id, state) VALUES (?1, 'planning')",
                    rusqlite::params![task_id_owned],
                )?;
                Ok(())
            });
            if let Err(e) = res {
                tracing::warn!(
                    task_id = %task_id,
                    error = %e,
                    "initial planning state append failed"
                );
            }
        }
        if let Some(observer) = &self.observer {
            let event = observe::BrainEvent::TaskStateChange {
                id: uuid::Uuid::new_v4(),
                task_id: task_id.to_string(),
                from: "none".into(),
                to: "planning".into(),
                ts: Utc::now(),
            };
            let _ = observer.publish(event).await;
        }
    }
}
