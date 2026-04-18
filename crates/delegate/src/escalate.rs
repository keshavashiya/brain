//! Escalation — turn a delegate failure into a typed decision the
//! orchestrator can act on (retry once, try a fallback agent, escalate
//! to the human).

use std::sync::Arc;

use crate::registry::AgentRegistry;
use crate::traits::{AgentDelegate, AgentError, AgentResult, AgentTask, AgentTaskStatus};

/// What should happen after a delegate call returns.
#[derive(Debug, Clone)]
pub enum EscalationOutcome {
    /// The delegate succeeded — carry the result upstream.
    Succeeded(AgentResult),
    /// The delegate failed; the orchestrator should pause the task and
    /// surface `reason` to the user via the channel router.
    EscalateToHuman { reason: String },
    /// The delegate failed on a retryable error; a fallback delegate
    /// took over and succeeded. `via` is the fallback's name.
    Recovered { via: String, result: AgentResult },
}

impl EscalationOutcome {
    pub fn succeeded(&self) -> bool {
        matches!(
            self,
            EscalationOutcome::Succeeded(_) | EscalationOutcome::Recovered { .. }
        )
    }

    /// Unwrap to an `AgentResult` when the outcome represents a finished
    /// run (either on the primary or a fallback). Returns `None` for
    /// `EscalateToHuman`.
    pub fn into_result(self) -> Option<AgentResult> {
        match self {
            EscalationOutcome::Succeeded(r) => Some(r),
            EscalationOutcome::Recovered { result, .. } => Some(result),
            EscalationOutcome::EscalateToHuman { .. } => None,
        }
    }
}

/// Caller-supplied policy controlling how failures are handled.
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Ordered fallback delegate names tried after the primary fails on
    /// a retryable error. Empty means "escalate immediately".
    pub fallbacks: Vec<String>,
    /// Whether timeouts should trigger fallback attempts. Defaults to
    /// `true` — a hung primary is often recoverable via a different
    /// agent. Set `false` for tasks where retry cost is prohibitive.
    pub retry_on_timeout: bool,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            fallbacks: Vec::new(),
            retry_on_timeout: true,
        }
    }
}

impl EscalationPolicy {
    pub fn with_fallbacks<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.fallbacks = names.into_iter().map(Into::into).collect();
        self
    }
}

fn is_retryable(err: &AgentError) -> bool {
    matches!(
        err,
        AgentError::Timeout { .. }
            | AgentError::Launch(_)
            | AgentError::Io(_)
            | AgentError::NonZeroExit { .. }
    )
}

/// Run a delegate against the registry, honouring `policy`. `AgentResult`s
/// with non-success status are counted as failures (e.g. subprocess
/// delegates surface exit != 0 via `Failed` rather than `Err`).
pub async fn run_with_escalation(
    primary: Arc<dyn AgentDelegate>,
    registry: &AgentRegistry,
    task: AgentTask,
    policy: &EscalationPolicy,
) -> EscalationOutcome {
    let primary_name = primary.name().to_string();
    let primary_err = match primary.delegate(task.clone()).await {
        Ok(result) if result.status.is_success() => {
            return EscalationOutcome::Succeeded(result);
        }
        Ok(result) => format!(
            "delegate '{primary_name}' returned status {:?}: {}",
            result.status, result.summary
        ),
        Err(e) => {
            if !is_retryable(&e)
                || (matches!(e, AgentError::Timeout { .. }) && !policy.retry_on_timeout)
            {
                return EscalationOutcome::EscalateToHuman {
                    reason: format!("delegate '{primary_name}' failed: {e}"),
                };
            }
            format!("delegate '{primary_name}' failed: {e}")
        }
    };

    for fallback in &policy.fallbacks {
        if fallback == &primary_name {
            continue;
        }
        let candidate = match registry.get(fallback) {
            Ok(c) => c,
            Err(_) => {
                tracing::warn!(fallback = %fallback, "Fallback delegate not registered");
                continue;
            }
        };
        match candidate.delegate(task.clone()).await {
            Ok(result) if result.status == AgentTaskStatus::Succeeded => {
                tracing::info!(
                    primary = %primary_name,
                    fallback = %fallback,
                    "Delegate recovered via fallback"
                );
                return EscalationOutcome::Recovered {
                    via: fallback.clone(),
                    result,
                };
            }
            Ok(r) => tracing::warn!(
                fallback = %fallback,
                status = ?r.status,
                "Fallback delegate returned non-success status"
            ),
            Err(e) => tracing::warn!(fallback = %fallback, error = %e, "Fallback delegate failed"),
        }
    }

    EscalationOutcome::EscalateToHuman {
        reason: format!("{primary_err}; no fallback recovered"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{AgentCapabilities, AgentTaskStatus};
    use async_trait::async_trait;
    use chrono::Utc;

    /// Test delegate that always returns the configured status.
    struct FixedAgent {
        name: String,
        succeed: bool,
    }

    #[async_trait]
    impl AgentDelegate for FixedAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn capabilities(&self) -> AgentCapabilities {
            AgentCapabilities::default()
        }
        async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
            let now = Utc::now();
            Ok(AgentResult {
                task_id: task.id,
                status: if self.succeed {
                    AgentTaskStatus::Succeeded
                } else {
                    AgentTaskStatus::Failed
                },
                summary: format!("{} ran", self.name),
                artifacts: vec![],
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(if self.succeed { 0 } else { 1 }),
                started_at: now,
                completed_at: now,
            })
        }
    }

    /// Test delegate that always errors with the configured variant.
    struct ErroringAgent {
        name: String,
        err_kind: ErrKind,
    }
    enum ErrKind {
        Timeout,
        Launch,
        Parse,
    }

    #[async_trait]
    impl AgentDelegate for ErroringAgent {
        fn name(&self) -> &str {
            &self.name
        }
        async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
            Err(match self.err_kind {
                ErrKind::Timeout => AgentError::Timeout {
                    task_id: task.id,
                    secs: 1,
                },
                ErrKind::Launch => AgentError::Launch("boom".to_string()),
                ErrKind::Parse => AgentError::Parse("bad output".to_string()),
            })
        }
    }

    fn task() -> AgentTask {
        AgentTask::new("do a thing")
    }

    #[tokio::test]
    async fn success_returns_succeeded() {
        let agent: Arc<dyn AgentDelegate> = Arc::new(FixedAgent {
            name: "a".to_string(),
            succeed: true,
        });
        let reg = AgentRegistry::new();
        let out = run_with_escalation(agent, &reg, task(), &EscalationPolicy::default()).await;
        assert!(matches!(out, EscalationOutcome::Succeeded(_)));
    }

    #[tokio::test]
    async fn non_retryable_error_escalates_immediately() {
        let agent: Arc<dyn AgentDelegate> = Arc::new(ErroringAgent {
            name: "a".to_string(),
            err_kind: ErrKind::Parse,
        });
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(FixedAgent {
            name: "b".to_string(),
            succeed: true,
        }));
        let policy = EscalationPolicy::default().with_fallbacks(["b"]);
        let out = run_with_escalation(agent, &reg, task(), &policy).await;
        // Parse error isn't retryable — fallback should NOT be attempted.
        assert!(matches!(out, EscalationOutcome::EscalateToHuman { .. }));
    }

    #[tokio::test]
    async fn retryable_error_tries_fallback() {
        let primary: Arc<dyn AgentDelegate> = Arc::new(ErroringAgent {
            name: "primary".to_string(),
            err_kind: ErrKind::Launch,
        });
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(FixedAgent {
            name: "fallback".to_string(),
            succeed: true,
        }));
        let policy = EscalationPolicy::default().with_fallbacks(["fallback"]);
        let out = run_with_escalation(primary, &reg, task(), &policy).await;
        assert!(matches!(out, EscalationOutcome::Recovered { via, .. } if via == "fallback"));
    }

    #[tokio::test]
    async fn timeout_respects_retry_flag() {
        let primary: Arc<dyn AgentDelegate> = Arc::new(ErroringAgent {
            name: "primary".to_string(),
            err_kind: ErrKind::Timeout,
        });
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(FixedAgent {
            name: "fallback".to_string(),
            succeed: true,
        }));
        let mut policy = EscalationPolicy::default().with_fallbacks(["fallback"]);
        policy.retry_on_timeout = false;
        let out = run_with_escalation(primary, &reg, task(), &policy).await;
        assert!(matches!(out, EscalationOutcome::EscalateToHuman { .. }));
    }

    #[tokio::test]
    async fn all_fallbacks_fail_escalates() {
        let primary: Arc<dyn AgentDelegate> = Arc::new(ErroringAgent {
            name: "primary".to_string(),
            err_kind: ErrKind::Launch,
        });
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(ErroringAgent {
            name: "also-broken".to_string(),
            err_kind: ErrKind::Launch,
        }));
        let policy = EscalationPolicy::default().with_fallbacks(["also-broken"]);
        let out = run_with_escalation(primary, &reg, task(), &policy).await;
        assert!(matches!(out, EscalationOutcome::EscalateToHuman { .. }));
    }

    #[tokio::test]
    async fn non_success_status_counts_as_failure_and_triggers_fallback() {
        let primary: Arc<dyn AgentDelegate> = Arc::new(FixedAgent {
            name: "primary".to_string(),
            succeed: false,
        });
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(FixedAgent {
            name: "fallback".to_string(),
            succeed: true,
        }));
        // A primary that reports non-success via `status` (rather than
        // an AgentError) is treated the same as a retryable failure —
        // fallbacks are tried in order.
        let policy = EscalationPolicy::default().with_fallbacks(["fallback"]);
        let out = run_with_escalation(primary, &reg, task(), &policy).await;
        assert!(matches!(out, EscalationOutcome::Recovered { via, .. } if via == "fallback"));
    }
}
