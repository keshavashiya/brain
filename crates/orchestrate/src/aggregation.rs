//! Result aggregation + context injection across orchestrated steps.
//!
//! Two responsibilities live here:
//!
//! 1. **Context injection** — before delegating to a specialist agent,
//!    build a richer task spec by pulling recent episodic memory and
//!    packing it into the prompt under a soft byte budget.
//!
//! 2. **Result aggregation** — after a delegate returns, write a
//!    structured episode capturing the delegated work so future runs
//!    can recall it. Failures of the delegate flow are still recorded
//!    via the audit trail by `execute_step`; this module only writes
//!    successful (or recovered) outcomes into episodic memory.
//!
//! Both helpers are no-ops if the relevant store isn't attached, so a
//! partially-wired orchestrator still completes plans cleanly.

use crate::orchestrator::TaskOrchestrator;

/// Soft byte budget for context injection into a delegate task spec.
/// ~2KB ≈ 500 tokens; older episodes drop first.
pub(crate) const DELEGATE_CONTEXT_BUDGET_BYTES: usize = 2048;
/// Cap on episodes considered for context injection.
pub(crate) const DELEGATE_CONTEXT_RECENT_LIMIT: usize = 10;
/// Importance score for orchestrator-recorded delegate episodes.
/// Mid-range so consolidation can prune them if unreinforced.
pub(crate) const DELEGATE_EPISODE_IMPORTANCE: f64 = 0.6;

impl TaskOrchestrator {
    /// Build the task spec for a delegate, optionally enriched with
    /// relevant facts from semantic memory under a soft token budget.
    /// When no context source is wired, returns the spec unchanged so the
    /// delegate behaves as before.
    pub(super) async fn build_delegate_task_spec(&self, spec: &str) -> String {
        let Some(episodic) = self.episodic.as_ref() else {
            return spec.to_string();
        };

        let recent = episodic
            .recent(DELEGATE_CONTEXT_RECENT_LIMIT, Some("personal"))
            .unwrap_or_default();
        if recent.is_empty() {
            return spec.to_string();
        }

        let mut bullets: Vec<String> = Vec::new();
        let mut budget = DELEGATE_CONTEXT_BUDGET_BYTES;
        for ep in recent.iter() {
            let line = format!("- [{}] {}: {}", ep.timestamp, ep.role, ep.content.trim());
            if line.len() > budget {
                break;
            }
            budget -= line.len();
            bullets.push(line);
        }

        if bullets.is_empty() {
            return spec.to_string();
        }

        format!(
            "{spec}\n\n## Recent context (auto-injected by orchestrator)\n{}",
            bullets.join("\n")
        )
    }

    /// Record a delegate completion in episodic memory so the
    /// orchestration history is queryable across sessions.
    pub(super) async fn record_delegate_episode(
        &self,
        agent: &str,
        spec: &str,
        result: &delegate::AgentResult,
        recovered_via: Option<&str>,
    ) {
        let Some(episodic) = self.episodic.as_ref() else {
            return;
        };
        let session_id = format!("orchestrator-{}", chrono::Utc::now().format("%Y%m%d"));
        if let Err(e) = episodic.ensure_session(&session_id, "orchestrator") {
            tracing::warn!("episodic ensure_session failed: {e}");
            return;
        }
        let header = match recovered_via {
            Some(via) => format!(
                "[delegate {agent}→{via}] {}",
                spec.lines().next().unwrap_or(spec)
            ),
            None => format!("[delegate {agent}] {}", spec.lines().next().unwrap_or(spec)),
        };
        let artifact_refs: Vec<String> = result
            .artifacts
            .iter()
            .map(|a| a.reference.clone())
            .collect();
        let body = format!(
            "{header}\nsummary: {}\nartifacts: {}",
            result.summary,
            if artifact_refs.is_empty() {
                "(none)".to_string()
            } else {
                artifact_refs.join(", ")
            }
        );
        if let Err(e) = episodic.store_episode(
            &session_id,
            "agent",
            &body,
            DELEGATE_EPISODE_IMPORTANCE,
            Some("personal"),
            Some(agent),
        ) {
            tracing::warn!("episodic store_episode failed: {e}");
        }
    }
}
