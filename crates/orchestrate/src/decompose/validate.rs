//! Post-parse plan repair + validation shared by the initial decompose
//! path and the replan-on-failure path. Keeping these in one place means a
//! bad LLM response can't slip an unrunnable plan past one path that the
//! other would have rejected — both flows apply the identical sequential
//! backstop, the identical parse-time gating, and the identical
//! `RawStep` → `TaskStep` finalization.

use std::collections::HashSet;

use uuid::Uuid;

use crate::step::TaskStep;

use super::parse::{build_task_step, RawStep};
use super::{DecompositionContext, DecompositionError};

/// `action_type`s whose steps obviously consume earlier output and so
/// should default to depending on the previous step when the LLM left
/// `depends_on` empty. A Research step is deliberately excluded so the
/// backstop never adds a spurious edge that blocks legitimate parallel
/// research.
fn consumes_prior(kind: &str) -> bool {
    matches!(
        kind,
        "shell" | "execute" | "test" | "review" | "notify" | "implement"
    )
}

/// Sequential-default backstop. The system prompt asks the LLM to chain
/// inherently sequential plans, but model output is unreliable — we've
/// seen six steps come back with `depends_on: []` for what is obviously
/// "scan → write → run → verify → review → notify".
///
/// Two cases to repair:
///   A. *No* step has any deps → chain the whole plan.
///   B. The first step has no deps (legitimate) but later steps that ALSO
///      have no deps are mid-plan — they should depend on the previous
///      step. We only force this for steps whose `action_type` obviously
///      consumes earlier output (see [`consumes_prior`]).
pub(super) fn apply_sequential_fallback(raw_steps: &mut [RawStep]) {
    if raw_steps.len() <= 1 {
        return;
    }
    let none_have_deps = raw_steps.iter().all(|s| s.depends_on.is_empty());
    if none_have_deps {
        for (i, step) in raw_steps.iter_mut().enumerate().skip(1) {
            step.depends_on = vec![i - 1];
        }
    } else {
        for (i, step) in raw_steps.iter_mut().enumerate().skip(1) {
            if step.depends_on.is_empty() && consumes_prior(&step.action_type) {
                step.depends_on = vec![i - 1];
            }
        }
    }
}

/// Guarantee the dependency edges can only form a DAG. `depends_on`
/// indices are 0-based positions into the plan, and the decomposer emits
/// steps in intended execution order — so a legitimate dependency always
/// points to an *earlier* step. Any index that is self-referential
/// (`dep == i`), points forward (`dep > i`), or is out of range can only
/// introduce a cycle (which `TaskGraph::from_steps` rejects wholesale with
/// "Cycle detected in task graph", failing the entire plan) or a dangling
/// edge. We drop those edges and dedup the rest, turning a malformed plan
/// into a runnable one; [`apply_sequential_fallback`] then re-links any
/// step this leaves dependency-less. Runs *before* the fallback for that
/// reason.
pub(super) fn sanitize_dependencies(raw_steps: &mut [RawStep]) {
    for (i, step) in raw_steps.iter_mut().enumerate() {
        let mut seen = HashSet::new();
        // `dep < i` subsumes the out-of-range check (i < len) and rejects
        // both self and forward references; `seen.insert` dedups.
        step.depends_on.retain(|&dep| dep < i && seen.insert(dep));
    }
}

/// Reject steps the executor can't possibly run, at plan time, so the
/// user sees the failure before approving rather than five seconds into
/// execution. Checks, per `action_type`:
///   - `shell`: non-empty command (goes through `sh -c`, so pipes /
///     redirects / `$VAR` are fine).
///   - `execute` / `test`: non-empty command, parseable as argv (see
///     [`crate::actions::parse_sandbox_command`]), and — when the caller
///     supplied an allowlist — a binary on that allowlist.
///   - `implement`: an explicitly-named agent must be in the registry
///     roster when the caller supplied one. An omitted/`default` agent is
///     resolved later by the orchestrator.
///
/// An empty `available_tools` / `available_agents` means "no allowlist /
/// no registry wired" → that check is skipped.
pub(super) fn validate_steps(
    raw_steps: &[RawStep],
    context: &DecompositionContext,
) -> Result<(), DecompositionError> {
    let allowed: Option<HashSet<&str>> = if context.available_tools.is_empty() {
        None
    } else {
        Some(context.available_tools.iter().map(String::as_str).collect())
    };
    let allowed_agents: Option<HashSet<&str>> = if context.available_agents.is_empty() {
        None
    } else {
        Some(
            context
                .available_agents
                .iter()
                .map(String::as_str)
                .collect(),
        )
    };

    for (i, step) in raw_steps.iter().enumerate() {
        match step.action_type.as_str() {
            "shell" => {
                let cmd = step.command.as_deref().unwrap_or("").trim();
                if cmd.is_empty() {
                    return Err(DecompositionError::Parse(format!(
                        "step {} ({:?}) is action_type=shell but has no `command`",
                        i + 1,
                        step.description,
                    )));
                }
            }
            "execute" | "test" => {
                let cmd = step.command.as_deref().unwrap_or("").trim();
                if cmd.is_empty() {
                    return Err(DecompositionError::Parse(format!(
                        "step {} ({:?}) is action_type={} but has no `command` — \
                         the LLM produced an unrunnable plan",
                        i + 1,
                        step.description,
                        step.action_type,
                    )));
                }
                let parsed = crate::actions::parse_sandbox_command(cmd).map_err(|why| {
                    DecompositionError::Parse(format!(
                        "step {} ({:?}) has an unrunnable command {:?}: {} \
                         (use action_type=\"shell\" if you need pipes/redirects/$VAR)",
                        i + 1,
                        step.description,
                        cmd,
                        why,
                    ))
                })?;
                // Allowlist check applies only to argv mode; shell mode
                // delegates binary lookup to the system shell.
                if let Some(allowed) = &allowed {
                    if let Some(binary) = parsed.argv.first() {
                        let basename = std::path::Path::new(binary)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(binary);
                        if !allowed.contains(basename) {
                            return Err(DecompositionError::Parse(format!(
                                "step {} ({:?}) calls `{}` which is not on the sandbox allowlist. \
                                 Allowed binaries: {}. \
                                 Either re-plan using only allowed tools, switch to \
                                 action_type=\"shell\", or add `{}` to `security.exec_allowlist`.",
                                i + 1,
                                step.description,
                                basename,
                                context.available_tools.join(", "),
                                basename,
                            )));
                        }
                    }
                }
            }
            "implement" => {
                if let Some(allowed) = &allowed_agents {
                    let named = step.agent.as_deref().map(str::trim).unwrap_or("");
                    if !named.is_empty() && named != "default" && !allowed.contains(named) {
                        let mut available: Vec<&str> = allowed.iter().copied().collect();
                        available.sort_unstable();
                        return Err(DecompositionError::Parse(format!(
                            "step {} ({:?}) delegates to agent `{}` which is not registered. \
                             Available agents: {}. \
                             Re-plan using one of those, or install/configure `{}`.",
                            i + 1,
                            step.description,
                            named,
                            available.join(", "),
                            named,
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Assign UUIDs and convert raw steps to `TaskStep`s. `depends_on`
/// indices are 0-based positions into the plan; the UUID table lets
/// [`build_task_step`] resolve them to ids.
pub(super) fn finalize(raw_steps: Vec<RawStep>) -> Vec<TaskStep> {
    let ids: Vec<String> = raw_steps
        .iter()
        .map(|_| Uuid::new_v4().to_string())
        .collect();
    raw_steps
        .into_iter()
        .enumerate()
        .map(|(i, raw)| build_task_step(i, raw, &ids))
        .collect()
}
