//! CLI commands for task orchestration.

use anyhow::Result;
use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum TaskAction {
    /// Decompose a request into a task plan.
    ///
    /// The LLM breaks down your request into discrete steps with
    /// dependencies, action tiers, and a DAG execution graph.
    ///
    /// Examples:
    ///   brain task plan "add a /health endpoint to the HTTP adapter"
    ///   brain task plan "refactor the embedding pipeline to support batching"
    Plan {
        /// The request to decompose into a task plan
        request: String,
    },

    /// Execute a previously planned task (after reviewing the plan).
    ///
    /// Steps run in topological order respecting dependencies.
    /// Destructive / external steps pause for approval.
    ///
    /// Examples:
    ///   brain task execute <task_id>
    Execute {
        /// Task ID returned by `brain task plan`
        task_id: String,
    },

    /// List all active tasks.
    List,

    /// Show detailed status of a task.
    ///
    /// Examples:
    ///   brain task status <task_id>
    Status {
        /// Task ID
        task_id: String,
    },

    /// Cancel a task (non-terminal steps become cancelled).
    ///
    /// Examples:
    ///   brain task cancel <task_id>
    Cancel {
        /// Task ID
        task_id: String,
    },
}

pub(crate) async fn cmd_task(config: &brain_core::BrainConfig, action: TaskAction) -> Result<()> {
    let processor = crate::bootstrap::build_processor(config).await?;

    let orchestrator = processor
        .orchestrator()
        .ok_or_else(|| {
            anyhow::anyhow!("Task orchestrator not available. Is it wired in bootstrap?")
        })?
        .clone();

    match action {
        TaskAction::Plan { request } => {
            println!("Decomposing request...\n");
            let context = orchestrate::DecompositionContext::default();
            let (task_id, plan_text) = orchestrator
                .plan(&request, context)
                .await
                .map_err(|e| anyhow::anyhow!("Task planning failed: {e}"))?;

            println!("{plan_text}");
            println!();
            println!("Task ID: {task_id}");
            println!();
            println!("To execute:  brain task execute {task_id}");
            println!("To cancel:   brain task cancel {task_id}");
        }

        TaskAction::Execute { task_id } => {
            // Verify the task exists and is in the right phase
            let task = orchestrator
                .get_task(&task_id)
                .await
                .ok_or_else(|| anyhow::anyhow!("Task not found: {task_id}"))?;

            match task.phase {
                orchestrate::TaskPhase::AwaitingApproval => {}
                orchestrate::TaskPhase::Executing => {
                    anyhow::bail!("Task {task_id} is already executing.");
                }
                orchestrate::TaskPhase::Completed => {
                    anyhow::bail!("Task {task_id} is already completed.");
                }
                orchestrate::TaskPhase::Cancelled => {
                    anyhow::bail!("Task {task_id} was cancelled.");
                }
                orchestrate::TaskPhase::Planning => {
                    anyhow::bail!("Task {task_id} is still being planned.");
                }
            }

            println!("Executing task {task_id}...\n");
            let summary = orchestrator
                .execute(&task_id)
                .await
                .map_err(|e| anyhow::anyhow!("Task execution failed: {e}"))?;

            println!("{summary}");
        }

        TaskAction::List => {
            let tasks = orchestrator.list_tasks().await;
            if tasks.is_empty() {
                println!("No active tasks.");
                return Ok(());
            }

            println!("{:<36} {:<20} Request", "Task ID", "Phase");
            println!("{}", "-".repeat(80));

            for (id, request, phase) in &tasks {
                let phase_str = format_phase(*phase);
                let req_display = if request.len() > 40 {
                    format!("{}...", &request[..37])
                } else {
                    request.clone()
                };
                println!("{:<36} {:<20} {}", id, phase_str, req_display);
            }
        }

        TaskAction::Status { task_id } => {
            let task = orchestrator
                .get_task(&task_id)
                .await
                .ok_or_else(|| anyhow::anyhow!("Task not found: {task_id}"))?;

            println!("Task: {}", task.id);
            println!("Request: \"{}\"", task.request);
            println!("Phase: {}", format_phase(task.phase));
            println!("Created: {}", task.created_at.format("%Y-%m-%d %H:%M:%S"));
            if let Some(completed) = task.completed_at {
                println!("Completed: {}", completed.format("%Y-%m-%d %H:%M:%S"));
            }

            let counts = task.counts();
            println!();
            println!(
                "Steps: {} total — {} completed, {} failed, {} pending, {} running, {} cancelled",
                counts.total(),
                counts.completed,
                counts.failed,
                counts.pending,
                counts.running,
                counts.cancelled,
            );

            // Show steps in topological order
            let order = task.graph.topological_order();
            println!();
            for (i, step_id) in order.iter().enumerate() {
                if let Some(step) = task.graph.steps.get(step_id) {
                    let state = task
                        .step_states
                        .get(step_id)
                        .map(format_step_state)
                        .unwrap_or_else(|| "unknown".to_string());
                    println!(
                        "  {}. {} [{}] — {}",
                        i + 1,
                        step.description,
                        state,
                        step.tier
                    );
                }
            }
        }

        TaskAction::Cancel { task_id } => {
            orchestrator
                .cancel(&task_id)
                .await
                .map_err(|e| anyhow::anyhow!("Cancel failed: {e}"))?;
            println!("Task {task_id} cancelled.");
        }
    }

    Ok(())
}

fn format_phase(phase: orchestrate::TaskPhase) -> &'static str {
    match phase {
        orchestrate::TaskPhase::Planning => "planning",
        orchestrate::TaskPhase::AwaitingApproval => "awaiting approval",
        orchestrate::TaskPhase::Executing => "executing",
        orchestrate::TaskPhase::Completed => "completed",
        orchestrate::TaskPhase::Cancelled => "cancelled",
    }
}

fn format_step_state(state: &orchestrate::StepState) -> String {
    match state {
        orchestrate::StepState::Pending => "pending".to_string(),
        orchestrate::StepState::Ready => "ready".to_string(),
        orchestrate::StepState::Running { .. } => "running".to_string(),
        orchestrate::StepState::AwaitingConfirmation { .. } => "awaiting confirmation".to_string(),
        orchestrate::StepState::Completed { .. } => "completed".to_string(),
        orchestrate::StepState::Failed { error, .. } => format!("failed: {error}"),
        orchestrate::StepState::Skipped { reason } => format!("skipped: {reason}"),
        orchestrate::StepState::Cancelled => "cancelled".to_string(),
    }
}
