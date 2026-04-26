//! Step-action handlers for [`TaskOrchestrator`].
//!
//! Each [`crate::step::StepAction`] variant has a corresponding executor
//! method on `TaskOrchestrator`. Splitting them out of `orchestrator.rs`
//! keeps the lifecycle/state-machine code there and groups the
//! per-action dispatch logic here. All methods follow the same pattern:
//! degrade gracefully (return a synthetic outcome) when the relevant
//! handle is not attached, so a partially-wired orchestrator still
//! finishes plans rather than hard-failing.

use crate::orchestrator::TaskOrchestrator;
use crate::state::StepOutcome;

impl TaskOrchestrator {
    /// Run a `Research` step against the configured LLM. If no LLM is
    /// attached, degrade to a no-op string outcome so a partially-wired
    /// orchestrator still finishes the plan instead of failing.
    pub(super) async fn execute_research_step(&self, query: &str) -> Result<StepOutcome, String> {
        let Some(llm) = self.llm.as_ref() else {
            return Ok(StepOutcome {
                stdout: format!("Research query: {query}"),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Researched (no LLM attached): {query}"),
            });
        };

        let messages = vec![
            cortex::llm::Message {
                role: cortex::llm::Role::System,
                content: "You are a research assistant. Answer the user's research query \
                          concisely with concrete facts and references where possible. \
                          If the query asks about a specific codebase or filesystem path, \
                          state plainly that you do not have direct access and ask the \
                          orchestrator to delegate the work."
                    .to_string(),
            },
            cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: query.to_string(),
            },
        ];

        match llm.generate(&messages).await {
            Ok(resp) => Ok(StepOutcome {
                stdout: resp.content.clone(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: summary_first_line(&resp.content, &format!("Research: {query}")),
            }),
            Err(e) => Err(format!("LLM research failed: {e}")),
        }
    }

    /// Run a `Review` step — asks the LLM to critique the named artifact.
    /// Falls back to a no-op outcome if no LLM is attached.
    pub(super) async fn execute_review_step(&self, artifact: &str) -> Result<StepOutcome, String> {
        let Some(llm) = self.llm.as_ref() else {
            return Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![artifact.to_string()],
                summary: format!("Review requested (no LLM attached): {artifact}"),
            });
        };

        let messages = vec![
            cortex::llm::Message {
                role: cortex::llm::Role::System,
                content: "You are a code/report reviewer. Critique the artifact for \
                          completeness, correctness, and obvious gaps. Surface concrete \
                          issues, not platitudes. Keep the critique under 200 words."
                    .to_string(),
            },
            cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: format!("Review this artifact: {artifact}"),
            },
        ];

        match llm.generate(&messages).await {
            Ok(resp) => Ok(StepOutcome {
                stdout: resp.content.clone(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![artifact.to_string()],
                summary: summary_first_line(&resp.content, &format!("Reviewed: {artifact}")),
            }),
            Err(e) => Err(format!("LLM review failed: {e}")),
        }
    }

    /// Deliver a `Notify` step through the channel dispatcher. The
    /// `channel` field of the step is a soft preference — the dispatcher
    /// falls back to learned preferences and initiation channel if the
    /// requested channel is unavailable.
    pub(super) async fn execute_notify_step(
        &self,
        channel: &str,
        message: &str,
    ) -> Result<StepOutcome, String> {
        let Some(dispatcher) = self.dispatcher.as_ref() else {
            return Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Notify (no dispatcher attached): {channel}: {message}"),
            });
        };

        let mut intent = channel::DeliveryIntent::new(
            message,
            channel::DeliveryCategory::Report,
            channel::UrgencyLevel::Normal,
        );
        if !channel.is_empty() && channel != "default" {
            intent = intent.with_preferred(channel);
        }

        match dispatcher.dispatch(intent).await {
            Ok(receipt) => Ok(StepOutcome {
                stdout: format!("Delivered via {} ({})", receipt.channel_id, receipt.reason),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Notified via {}: {message}", receipt.channel_id),
            }),
            Err(e) => Err(format!("Notify delivery failed: {e}")),
        }
    }

    /// Execute a command in the sandbox.
    pub(super) async fn execute_sandbox_step(
        &self,
        command: &str,
        workdir: &std::path::Path,
    ) -> Result<StepOutcome, String> {
        let sandbox = match self.sandbox.as_ref() {
            Some(s) => s,
            None => {
                return Err("Sandbox not available".to_string());
            }
        };

        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty command".to_string());
        }

        let cmd = sandbox::SandboxCommand::new(
            parts[0],
            parts[1..].iter().map(|s| s.to_string()).collect(),
        )
        .with_workdir(workdir.to_path_buf());

        match sandbox.run(cmd).await {
            Ok(outcome) => Ok(StepOutcome {
                stdout: outcome.stdout,
                stderr: outcome.stderr,
                exit_code: Some(outcome.exit_code),
                artifacts: vec![],
                summary: if outcome.exit_code == 0 {
                    format!("Command succeeded: {command}")
                } else {
                    format!("Command failed (exit {}): {command}", outcome.exit_code)
                },
            }),
            Err(e) => Err(format!("Sandbox execution failed: {e}")),
        }
    }

    /// Hand the step off to a registered [`delegate::AgentDelegate`].
    /// Failures are run through the configured escalation policy — a
    /// primary hang or launch failure transparently falls over to the
    /// declared fallback chain; anything the chain can't recover becomes
    /// a human escalation recorded as a failed step outcome.
    pub(super) async fn delegate_implement_step(
        &self,
        spec: &str,
        agent: &str,
    ) -> Result<StepOutcome, String> {
        let registry = self
            .agents
            .as_ref()
            .ok_or_else(|| "Agent registry not attached to orchestrator".to_string())?;

        let primary = registry
            .get(agent)
            .map_err(|e| format!("Delegate '{agent}' unavailable: {e}"))?;

        let task_spec = self.build_delegate_task_spec(spec).await;
        let task = delegate::AgentTask::new(task_spec);
        let outcome = delegate::run_with_escalation(
            primary,
            registry.as_ref(),
            task,
            &self.delegation_policy,
        )
        .await;

        match outcome {
            delegate::EscalationOutcome::Succeeded(result) => {
                self.record_delegate_episode(agent, spec, &result, None)
                    .await;
                Ok(StepOutcome {
                    stdout: result.stdout,
                    stderr: result.stderr,
                    exit_code: result.exit_code,
                    artifacts: result
                        .artifacts
                        .iter()
                        .map(|a| a.reference.clone())
                        .collect(),
                    summary: format!("{agent}: {}", result.summary),
                })
            }
            delegate::EscalationOutcome::Recovered { via, result } => {
                self.record_delegate_episode(agent, spec, &result, Some(&via))
                    .await;
                Ok(StepOutcome {
                    stdout: result.stdout,
                    stderr: result.stderr,
                    exit_code: result.exit_code,
                    artifacts: result
                        .artifacts
                        .iter()
                        .map(|a| a.reference.clone())
                        .collect(),
                    summary: format!("{agent} failed; recovered via {via}: {}", result.summary),
                })
            }
            delegate::EscalationOutcome::EscalateToHuman { reason } => Err(reason),
        }
    }
}

/// First non-empty line of `s`, truncated to 160 chars; falls back to
/// `default_label` when the LLM returned only whitespace.
pub(crate) fn summary_first_line(s: &str, default_label: &str) -> String {
    let line = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if line.is_empty() {
        return default_label.to_string();
    }
    if line.chars().count() > 160 {
        let truncated: String = line.chars().take(157).collect();
        format!("{truncated}…")
    } else {
        line.to_string()
    }
}
