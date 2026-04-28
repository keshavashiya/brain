//! LLM-based task decomposition + procedural memory validation.
//!
//! Pipeline: user request → LLM generates candidate steps (JSON) →
//! cerebellum validates against known patterns → tier assignment → output.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use thiserror::Error;
use uuid::Uuid;

use crate::step::{StepAction, TaskStep};

#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("LLM error: {0}")]
    Llm(#[from] cortex::llm::LlmError),
    #[error("Failed to parse LLM output: {0}")]
    Parse(String),
    #[error("Empty plan — LLM produced no steps")]
    EmptyPlan,
}

/// Context passed to the decomposer to inform the LLM.
#[derive(Debug, Default)]
pub struct DecompositionContext {
    /// Known procedures from cerebellum (matched by trigger).
    pub known_procedures: Vec<String>,
    /// Available tools/commands.
    pub available_tools: Vec<String>,
    /// Relevant facts from semantic memory.
    pub relevant_facts: Vec<String>,
    /// Credential scopes available in the vault (tool names, not values).
    pub available_credentials: Vec<String>,
}

/// Context for a replan-on-failure call. Built by the orchestrator from
/// the original task state when a step fails and we want the LLM to
/// produce a corrective sub-plan.
#[derive(Debug, Clone)]
pub struct RepairContext {
    /// The user's original request.
    pub original_request: String,
    /// One-line description of the failed step.
    pub failed_step: String,
    /// The actual error returned by the failed step.
    pub error: String,
    /// Brief notes on what already succeeded so the new plan can build
    /// on it (descriptions only, not full outputs).
    pub completed: Vec<String>,
}

/// Decompose a user request into executable task steps.
#[async_trait]
pub trait TaskDecomposer: Send + Sync {
    async fn decompose(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError>;

    /// Attempt to replan after a step failed. Returns a fresh sub-plan
    /// to splice into the graph in place of the failed work. Default
    /// implementation declines (returns `EmptyPlan`) so trait impls
    /// without LLM access don't accidentally succeed with no steps.
    async fn replan_after_failure(
        &self,
        _repair: RepairContext,
        _context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        Err(DecompositionError::EmptyPlan)
    }
}

/// LLM-based task decomposer.
pub struct LlmDecomposer {
    llm: Arc<dyn cortex::LlmProvider>,
}

impl LlmDecomposer {
    pub fn new(llm: Arc<dyn cortex::LlmProvider>) -> Self {
        Self { llm }
    }
}

/// Raw step as parsed from LLM JSON output.
///
/// Every nullable field uses `deserialize_with = "null_to_default"` so a
/// JSON `null` (which the LLM frequently emits) deserializes the same as
/// a missing field. Without this the entire plan parse fails with
/// `invalid type: null, expected sequence` when the LLM helpfully
/// includes `"depends_on": null`.
#[derive(Debug, Deserialize)]
struct RawStep {
    #[serde(default, deserialize_with = "lenient_required_string")]
    description: String,
    #[serde(default, deserialize_with = "lenient_required_string")]
    action_type: String,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    command: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    query: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    spec: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    agent: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    artifact: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    channel: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    message: Option<String>,
    #[serde(default, deserialize_with = "lenient_usize_vec")]
    depends_on: Vec<usize>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    tier: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    estimated_tokens: Option<u64>,
}

/// Deserialize `null` as `T::default()`. The LLM emits `null` for empty
/// lists/strings/numbers regularly; without this every such field
/// crashes the whole plan parse.
fn null_to_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    let opt = Option::<T>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// Lenient string deserializer for `Option<String>` fields. Accepts:
///   - `null` / missing → `None`
///   - empty string → `None` (so a stray `""` doesn't silently override
///     a default like `"default"`)
///   - any string → `Some(s)`
///   - integer / float / bool → coerced to its `to_string()` form
///
/// Returning `None` instead of failing is the right behavior: the LLM
/// occasionally emits `"command": 0` or `"query": null`. A parse failure
/// here used to discard the entire plan; instead we let the field be
/// empty and let the per-action validation in `decompose_impl` /
/// `replan_after_failure` produce a precise "step N has no command"
/// error message that points at the actual problem step.
fn lenient_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Option<String>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string, integer, float, bool, or null")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            })
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(if v.is_empty() { None } else { Some(v) })
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(self)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
    }
    deserializer.deserialize_any(V)
}

/// Lenient `String` deserializer for required string fields
/// (`description`, `action_type`). Same coercion rules as
/// `lenient_optional_string` but produces an empty string instead of
/// `None`, deferring the "missing required field" complaint to the
/// per-step validator which has more context to give a useful error.
fn lenient_required_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(lenient_optional_string(deserializer)?.unwrap_or_default())
}

/// Lenient `Vec<usize>` deserializer for `depends_on`. The LLM
/// sometimes emits a bare integer (`"depends_on": 1`) or `null` instead
/// of an array. Coerce single ints to a one-element vec, null/missing
/// to an empty vec, and accept normal arrays as-is. Anything we can't
/// interpret yields an empty vec — the worst case is the step has no
/// dependencies, which the orchestrator's sequential-fallback logic
/// repairs at planning time.
fn lenient_usize_vec<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Vec<usize>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("array of indices, single index, or null")
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(self)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(vec![v as usize])
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            if v < 0 {
                Ok(Vec::new())
            } else {
                Ok(vec![v as usize])
            }
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                if let Some(n) = elem.as_u64() {
                    out.push(n as usize);
                } else if let Some(n) = elem.as_i64() {
                    if n >= 0 {
                        out.push(n as usize);
                    }
                }
                // anything else (string, null, object) is silently dropped
            }
            Ok(out)
        }
    }
    deserializer.deserialize_any(V)
}

const DECOMPOSE_SYSTEM_PROMPT: &str = r#"You are a task planner for Brain OS. Given a user request, decompose it into executable steps.

Each step must be independently executable. Steps must have clear dependencies.

Output a JSON array of step objects with these fields:
- "description": human-readable description of the step
- "action_type": one of "research", "plan", "implement", "shell", "execute", "test", "review", "notify"
- "command": command string (for shell/execute/test action types)
- "query": search query (for research action type)
- "spec": implementation specification (for implement action type)
- "agent": which agent to use (for implement, e.g. "claude-code", "qwen")
- "artifact": what to review (for review action type)
- "channel": notification channel (for notify action type)
- "message": notification message (for notify action type)
- "depends_on": array of step indices (0-based) this step depends on
- "tier": action tier — "read", "write", "execute", "destructive", "external"
- "estimated_tokens": estimated LLM tokens needed (0 for non-LLM steps)

Execution-mode rules — pick "shell" by default:
- "shell" — wrapped in `sh -c`. Pipes, redirects (`> file`, `< file`),
  $VAR expansion, glob, quoted args, and PATH-resolved binaries (cargo,
  python, brew-installed tools, etc.) ALL work. Use this for anything
  beyond a single binary with literal args.
- "execute"/"test" — direct argv, no shell. ONLY use these when the
  command is one binary plus literal arguments (e.g. `ls /tmp`,
  `git status`). The first token must be on the per-binary allowlist.
  No metacharacters of any kind.
When in doubt, use "shell" — the safety surface is the same (rlimits,
sandbox-exec, timeout, forbidden_commands) and you avoid a whole class
of "unrunnable command" rejections.

Dependency rules:
- Default to sequential dependencies — step N depends on step N-1 unless you can clearly justify true parallelism (e.g., two independent research queries).
- A plan that reads as a chain ("scan → write → run → verify → review → notify") MUST be encoded as a chain in `depends_on`. Do not produce a flat list of independent steps for inherently sequential work.
- An `execute`/`test` step that runs a script MUST depend on the `implement` step that produces that script. Never depend only on a `plan` step — `plan` only emits text, it does NOT create files on disk. If you need a file to exist, the producing step must be `implement` (or an `execute` step that writes the file with a real shell command).
- `notify` is always last in the chain and depends on every prior result-producing step it summarizes.

Command rules:
- `command` MUST be a non-empty string. For "shell" steps it's the full
  shell line (anything `sh -c` can run). For "execute"/"test" it's an
  argv-style string (binary plus literal args, no metacharacters).
- For "execute"/"test" the FIRST token MUST be on the "Available sandbox
  binaries" list when one is provided. For "shell" the wrapped command
  may call any binary on the daemon's PATH — but if the task
  fundamentally requires a tool that isn't available (`docker`, `aws`,
  `act`, etc.), do NOT plan installation or pretend it's there: return a
  single "notify" step asking the user to install or allowlist the tool.
- Never plan `brew install`, `apt install`, `pip install`,
  `npm install -g`, or `cargo install` to set up a tool the orchestrator
  will need — those side-effects belong to the user, not to a plan.

Tier rules:
- "read": queries memory, reads files, surfaces information
- "write": stores facts, edits files, modifies local state
- "execute": runs sandboxed commands, builds/tests code
- "destructive": deletes data, force-pushes, drops tables, irrevocable file deletion
- "external": calls third-party APIs, deploys to remote services, posts to public platforms (NOT internal user notifications — those are "read")
- Prefer reversible actions where possible.

Notify rules:
- Internal notifications to the user (telling the user a task is done, surfacing results) are "read" tier — they are output, not external API calls.
- Reserve "external" for genuine third-party calls (Slack webhook to a public channel, email send via an SMTP API, etc.).

Keep the plan practical and minimal — no unnecessary steps.

Return ONLY valid JSON (an array of objects). No markdown, no explanations."#;

impl LlmDecomposer {
    async fn decompose_impl(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        let mut user_prompt = format!("Decompose this request into steps:\n\n\"{request}\"");

        if !context.known_procedures.is_empty() {
            user_prompt.push_str("\n\nKnown procedures for similar tasks:\n");
            for proc in &context.known_procedures {
                user_prompt.push_str(&format!("- {proc}\n"));
            }
        }
        if !context.relevant_facts.is_empty() {
            user_prompt.push_str("\n\nRelevant project context:\n");
            for fact in &context.relevant_facts {
                user_prompt.push_str(&format!("- {fact}\n"));
            }
        }
        if !context.available_tools.is_empty() {
            user_prompt.push_str(
                "\n\nAvailable sandbox binaries (every `execute`/`test` step MUST start with one of these — see system rules):\n  ",
            );
            user_prompt.push_str(&context.available_tools.join(", "));
        }

        let messages = vec![
            cortex::llm::Message {
                role: cortex::llm::Role::System,
                content: DECOMPOSE_SYSTEM_PROMPT.to_string(),
            },
            cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: user_prompt,
            },
        ];

        let response = self.llm.generate(&messages).await?;
        let mut raw_steps = parse_steps(&response.content)?;

        if raw_steps.is_empty() {
            return Err(DecompositionError::EmptyPlan);
        }

        // Reject execute/test steps the sandbox can't possibly run, so
        // the user sees the failure at planning time instead of a
        // mysterious "step failed" five seconds into execution. The
        // sandbox runs argv directly — see actions::parse_sandbox_command
        // for the full list of unsupported shell metacharacters.
        let allowed: Option<std::collections::HashSet<&str>> = if context.available_tools.is_empty()
        {
            None
        } else {
            Some(context.available_tools.iter().map(String::as_str).collect())
        };

        for (i, step) in raw_steps.iter().enumerate() {
            match step.action_type.as_str() {
                "shell" => {
                    // Shell steps go through `sh -c` — pipes, redirects,
                    // $VAR, PATH lookup all work. The only parse-time
                    // requirement is a non-empty command.
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
                    // Allowlist check applies only to argv mode; shell
                    // mode delegates binary lookup to the system shell.
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
                _ => {}
            }
        }

        // Sequential-default backstop. The system prompt asks the LLM to
        // chain inherently sequential plans, but model output is unreliable
        // — we've seen six steps come back with `depends_on: []` for what
        // is obviously "scan → write → run → verify → review → notify".
        //
        // Two cases to repair:
        //   A. *No* step has any deps → chain the whole plan.
        //   B. The first step has no deps (legitimate) but later steps
        //      that ALSO have no deps are mid-plan — they should depend
        //      on the previous step. We only force this for steps whose
        //      action_type is one that obviously consumes earlier output
        //      ("execute", "test", "review", "notify"). Adding spurious
        //      edges to a Research step would block legitimate parallel
        //      research.
        let consumes_prior = |kind: &str| {
            matches!(
                kind,
                "shell" | "execute" | "test" | "review" | "notify" | "implement"
            )
        };
        if raw_steps.len() > 1 {
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

        // Assign UUIDs and convert raw steps to TaskSteps.
        // deps reference 0-based indices → resolve to UUIDs.
        let ids: Vec<String> = raw_steps
            .iter()
            .map(|_| Uuid::new_v4().to_string())
            .collect();

        let steps: Vec<TaskStep> = raw_steps
            .into_iter()
            .enumerate()
            .map(|(i, raw)| build_task_step(i, raw, &ids))
            .collect();

        Ok(steps)
    }
}

const REPAIR_SYSTEM_PROMPT: &str = r#"You are a task planner repairing a failed plan for Brain OS.

The user originally asked for something. The orchestrator decomposed it into steps and started executing. One step failed. Your job: produce a fresh JSON array of replacement steps that achieves the original goal given what already succeeded and what just went wrong.

Rules:
- Output ONLY a JSON array of step objects, same schema as the original decomposer (description, action_type, command, depends_on, tier, ...).
- Do NOT repeat work that already succeeded.
- Do NOT retry the failed step verbatim — pick a different approach. If the error suggests a missing tool or capability, prefer "shell" action_type or pick a different binary. If the underlying need cannot be met without something the sandbox can't do, return a single "notify" step that explains the blocker honestly.
- Keep the plan minimal. 1–4 steps is plenty for most repairs; never more than 6.
- Sequential dependencies (step N depends on step N-1) unless you can clearly justify parallelism.
- All execution-mode rules from the main planner still apply (shell vs execute, allowlist, no install steps, etc.).

Return ONLY valid JSON. No markdown, no explanation."#;

impl LlmDecomposer {
    async fn replan_inner(
        &self,
        repair: &RepairContext,
        context: &DecompositionContext,
    ) -> Result<Vec<RawStep>, DecompositionError> {
        let mut user_prompt = format!(
            "Original request:\n  {}\n\nWhat already succeeded (do NOT redo):\n",
            repair.original_request
        );
        if repair.completed.is_empty() {
            user_prompt.push_str("  (nothing yet)\n");
        } else {
            for line in &repair.completed {
                user_prompt.push_str(&format!("  - {line}\n"));
            }
        }
        user_prompt.push_str(&format!(
            "\nFailed step:\n  {}\n\nActual error:\n  {}\n",
            repair.failed_step, repair.error,
        ));
        if !context.available_tools.is_empty() {
            user_prompt.push_str(
                "\nAvailable sandbox binaries (for execute/test action_type — shell mode bypasses this):\n  ",
            );
            user_prompt.push_str(&context.available_tools.join(", "));
        }

        let messages = vec![
            cortex::llm::Message {
                role: cortex::llm::Role::System,
                content: REPAIR_SYSTEM_PROMPT.to_string(),
            },
            cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: user_prompt,
            },
        ];

        let response = self.llm.generate(&messages).await?;
        parse_steps(&response.content)
    }
}

#[async_trait]
impl TaskDecomposer for LlmDecomposer {
    async fn replan_after_failure(
        &self,
        repair: RepairContext,
        context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        let mut raw_steps = self.replan_inner(&repair, &context).await?;
        if raw_steps.is_empty() {
            return Err(DecompositionError::EmptyPlan);
        }

        // Re-use the same sequential-fallback + parse-time validation
        // path as the main decompose flow so a bad LLM response can't
        // create a worse plan than the one we just failed on.
        let consumes_prior = |kind: &str| {
            matches!(
                kind,
                "shell" | "execute" | "test" | "review" | "notify" | "implement"
            )
        };
        if raw_steps.len() > 1 {
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

        let allowed: Option<std::collections::HashSet<&str>> = if context.available_tools.is_empty()
        {
            None
        } else {
            Some(context.available_tools.iter().map(String::as_str).collect())
        };

        for (i, step) in raw_steps.iter().enumerate() {
            match step.action_type.as_str() {
                "shell" => {
                    let cmd = step.command.as_deref().unwrap_or("").trim();
                    if cmd.is_empty() {
                        return Err(DecompositionError::Parse(format!(
                            "replan step {} ({:?}) is action_type=shell but has no command",
                            i + 1,
                            step.description
                        )));
                    }
                }
                "execute" | "test" => {
                    let cmd = step.command.as_deref().unwrap_or("").trim();
                    if cmd.is_empty() {
                        return Err(DecompositionError::Parse(format!(
                            "replan step {} ({:?}) is action_type={} but has no command",
                            i + 1,
                            step.description,
                            step.action_type
                        )));
                    }
                    let parsed = crate::actions::parse_sandbox_command(cmd).map_err(|why| {
                        DecompositionError::Parse(format!(
                            "replan step {} ({:?}) has unrunnable command {:?}: {}",
                            i + 1,
                            step.description,
                            cmd,
                            why
                        ))
                    })?;
                    if let Some(allowed) = &allowed {
                        if let Some(binary) = parsed.argv.first() {
                            let basename = std::path::Path::new(binary)
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or(binary);
                            if !allowed.contains(basename) {
                                return Err(DecompositionError::Parse(format!(
                                    "replan step {} calls `{basename}` which is not on the sandbox allowlist",
                                    i + 1
                                )));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let ids: Vec<String> = raw_steps
            .iter()
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let steps: Vec<TaskStep> = raw_steps
            .into_iter()
            .enumerate()
            .map(|(i, raw)| build_task_step(i, raw, &ids))
            .collect();
        Ok(steps)
    }

    async fn decompose(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        self.decompose_impl(request, context).await
    }
}

/// Convert one raw LLM step into a `TaskStep` using the populated UUID
/// table so `depends_on` indices resolve to ids. Lifted out of the
/// original `decompose` impl so the replan path can share it.
fn build_task_step(i: usize, raw: RawStep, ids: &[String]) -> TaskStep {
    let depends_on: Vec<String> = raw
        .depends_on
        .iter()
        .filter_map(|&idx| ids.get(idx).cloned())
        .collect();

    let action = match raw.action_type.as_str() {
        "research" => StepAction::Research {
            query: raw.query.unwrap_or_else(|| raw.description.clone()),
        },
        "plan" => StepAction::Plan {
            output: raw.spec.unwrap_or_default(),
        },
        "implement" => StepAction::Implement {
            spec: raw.spec.unwrap_or_else(|| raw.description.clone()),
            agent: raw.agent.unwrap_or_else(|| "default".to_string()),
        },
        "execute" => StepAction::Execute {
            command: raw.command.unwrap_or_default(),
            workdir: std::env::current_dir().unwrap_or_default(),
        },
        "test" => StepAction::Test {
            command: raw.command.unwrap_or_else(|| "cargo test".to_string()),
            workdir: std::env::current_dir().unwrap_or_default(),
        },
        "shell" => StepAction::Shell {
            command: raw.command.unwrap_or_default(),
            workdir: std::env::current_dir().unwrap_or_default(),
        },
        "review" => StepAction::Review {
            artifact: raw.artifact.unwrap_or_else(|| raw.description.clone()),
        },
        "notify" => StepAction::Notify {
            channel: raw.channel.unwrap_or_else(|| "default".to_string()),
            message: raw.message.unwrap_or_else(|| raw.description.clone()),
        },
        _ => StepAction::Plan {
            output: raw.description.clone(),
        },
    };

    let tier = match raw.tier.as_deref() {
        Some("read") => audit::ActionTier::Read,
        Some("write") => audit::ActionTier::Write,
        Some("destructive") => audit::ActionTier::Destructive,
        Some("external") => audit::ActionTier::External,
        _ => audit::ActionTier::Execute,
    };

    let tier = match (&action, tier) {
        (StepAction::Notify { .. }, audit::ActionTier::External) => audit::ActionTier::Read,
        (_, t) => t,
    };

    TaskStep {
        id: ids[i].clone(),
        description: raw.description,
        action,
        depends_on,
        tier,
        estimated_tokens: raw.estimated_tokens.unwrap_or(0),
    }
}

/// Parse LLM JSON output into raw step structs.
fn parse_steps(raw: &str) -> Result<Vec<RawStep>, DecompositionError> {
    // Try to extract JSON array from potentially markdown-wrapped output.
    let trimmed = raw.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    serde_json::from_str(json_str).map_err(|e| DecompositionError::Parse(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_steps_basic() {
        let json = r#"[
            {
                "description": "Research existing patterns",
                "action_type": "research",
                "query": "CSV export patterns",
                "depends_on": [],
                "tier": "read"
            },
            {
                "description": "Implement CSV endpoint",
                "action_type": "implement",
                "spec": "Add /api/export/csv endpoint",
                "agent": "claude-code",
                "depends_on": [0],
                "tier": "execute"
            }
        ]"#;

        let steps = parse_steps(json).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].action_type, "research");
        assert_eq!(steps[1].depends_on, vec![0]);
    }

    #[test]
    fn test_parse_steps_tolerates_null_fields() {
        // The LLM regularly emits `null` for fields it has no value for.
        // Without lenient deserialization the entire plan parse fails on
        // the first null and the user sees `invalid type: null, expected
        // sequence` instead of a runnable plan.
        let json = r#"[
            {
                "description": "do thing",
                "action_type": "shell",
                "command": "echo hi",
                "depends_on": null,
                "tier": null,
                "estimated_tokens": null,
                "spec": null
            }
        ]"#;
        let steps = parse_steps(json).expect("null fields should be lenient");
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].action_type, "shell");
        assert!(steps[0].depends_on.is_empty());
        assert!(steps[0].tier.is_none());
    }

    #[test]
    fn test_parse_steps_tolerates_integer_tier() {
        // Some LLMs emit tier as an integer code instead of a string.
        // The lenient deserializer coerces it to its string form;
        // downstream tier matching falls through to the safe Execute
        // default for any unrecognized tier name.
        let json = r#"[
            {"description": "x", "action_type": "shell", "command": "true", "tier": 1}
        ]"#;
        let steps = parse_steps(json).expect("integer tier should not break parse");
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].tier.as_deref(), Some("1"));
    }

    #[test]
    fn test_parse_steps_tolerates_integer_string_fields() {
        // The Groq replan path was observed emitting numeric values for
        // string fields (`"command": 0`, `"query": 0`) — see
        // brain.log:623, 653, 676, 889. The previous deserializer
        // failed the entire plan with `invalid type: integer 0,
        // expected a string`, masking the actual blocker. Coerce ints
        // to their string form so the per-step validator can then
        // reject the malformed step with a precise message.
        let json = r#"[
            {"description": "noisy step", "action_type": "shell", "command": 0, "query": 1, "spec": 2.5, "tier": "read"}
        ]"#;
        let steps = parse_steps(json).expect("integer string-field values should not break parse");
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].command.as_deref(), Some("0"));
        assert_eq!(steps[0].query.as_deref(), Some("1"));
        assert_eq!(steps[0].spec.as_deref(), Some("2.5"));
    }

    #[test]
    fn test_parse_steps_tolerates_integer_depends_on() {
        // Same family of LLM glitch — `depends_on` arrives as a bare
        // integer instead of an array (`invalid type: integer 0,
        // expected a sequence` in brain.log:889). Wrap a single index
        // into a one-element vec so the graph builder gets the right
        // shape.
        let json = r#"[
            {"description": "first", "action_type": "shell", "command": "true", "depends_on": []},
            {"description": "second", "action_type": "shell", "command": "true", "depends_on": 0}
        ]"#;
        let steps = parse_steps(json).expect("integer depends_on should not break parse");
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].depends_on, vec![0]);
    }

    #[test]
    fn test_parse_steps_tolerates_empty_string_fields() {
        // The lenient deserializer treats an empty string as None so a
        // stray `""` doesn't override a meaningful default later in the
        // pipeline (e.g. the "default" channel fallback for Notify).
        let json = r#"[
            {"description": "x", "action_type": "notify", "channel": "", "message": "hello", "depends_on": []}
        ]"#;
        let steps = parse_steps(json).unwrap();
        assert_eq!(steps.len(), 1);
        assert!(steps[0].channel.is_none());
        assert_eq!(steps[0].message.as_deref(), Some("hello"));
    }

    #[test]
    fn test_parse_steps_markdown_wrapped() {
        let json = r#"```json
[{"description": "Do something", "action_type": "plan", "depends_on": []}]
```"#;

        let steps = parse_steps(json).unwrap();
        assert_eq!(steps.len(), 1);
    }

    #[tokio::test]
    async fn rejects_execute_step_with_empty_command() {
        use cortex::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk};
        use futures::Stream;
        use std::pin::Pin;

        struct EmptyCmdLlm;
        #[async_trait]
        impl LlmProvider for EmptyCmdLlm {
            async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
                Ok(Response {
                    content: r#"[
                        {"description": "run the script", "action_type": "execute", "command": "", "depends_on": []}
                    ]"#.to_string(),
                    usage: None,
                })
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unimplemented!()
            }
            async fn health_check(&self) -> bool {
                true
            }
            fn name(&self) -> &str {
                "test"
            }
            fn model(&self) -> &str {
                "test-model"
            }
            async fn list_models(&self) -> Result<Vec<String>, LlmError> {
                Ok(vec!["test-model".into()])
            }
        }

        let llm = std::sync::Arc::new(EmptyCmdLlm);
        let decomposer = LlmDecomposer::new(llm);
        let err = decomposer
            .decompose("anything", DecompositionContext::default())
            .await
            .unwrap_err();
        assert!(
            matches!(err, DecompositionError::Parse(_)),
            "expected parse-time rejection, got {err:?}"
        );
    }

    #[tokio::test]
    async fn rejects_execute_step_outside_sandbox_allowlist() {
        // Regression for the user's `act` / `brew` plan: when the
        // caller supplies an allowlist via DecompositionContext,
        // execute steps that call binaries outside it must be
        // rejected at decompose time, not at sandbox time.
        use cortex::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk};
        use futures::Stream;
        use std::pin::Pin;

        struct ActLlm;
        #[async_trait]
        impl LlmProvider for ActLlm {
            async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
                Ok(Response {
                    content: r#"[
                        {"description": "check act installed", "action_type": "execute", "command": "which act", "depends_on": []}
                    ]"#.to_string(),
                    usage: None,
                })
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unimplemented!()
            }
            async fn health_check(&self) -> bool {
                true
            }
            fn name(&self) -> &str {
                "test"
            }
            fn model(&self) -> &str {
                "test-model"
            }
            async fn list_models(&self) -> Result<Vec<String>, LlmError> {
                Ok(vec!["test-model".into()])
            }
        }

        let llm = std::sync::Arc::new(ActLlm);
        let decomposer = LlmDecomposer::new(llm);
        let ctx = DecompositionContext {
            available_tools: vec!["ls".into(), "grep".into(), "cargo".into()],
            ..Default::default()
        };
        let err = decomposer.decompose("anything", ctx).await.unwrap_err();
        match err {
            DecompositionError::Parse(msg) => {
                assert!(
                    msg.contains("which") && msg.contains("not on the sandbox allowlist"),
                    "expected allowlist-rejection message, got: {msg}"
                );
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_execute_step_with_pipeline() {
        use cortex::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk};
        use futures::Stream;
        use std::pin::Pin;

        struct PipeLlm;
        #[async_trait]
        impl LlmProvider for PipeLlm {
            async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
                Ok(Response {
                    content: r#"[
                        {"description": "pipeline step", "action_type": "execute", "command": "ls | grep foo", "depends_on": []}
                    ]"#.to_string(),
                    usage: None,
                })
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unimplemented!()
            }
            async fn health_check(&self) -> bool {
                true
            }
            fn name(&self) -> &str {
                "test"
            }
            fn model(&self) -> &str {
                "test-model"
            }
            async fn list_models(&self) -> Result<Vec<String>, LlmError> {
                Ok(vec!["test-model".into()])
            }
        }

        let llm = std::sync::Arc::new(PipeLlm);
        let decomposer = LlmDecomposer::new(llm);
        let err = decomposer
            .decompose("anything", DecompositionContext::default())
            .await
            .unwrap_err();
        assert!(
            matches!(err, DecompositionError::Parse(_)),
            "expected parse-time rejection of pipeline, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_sequential_fallback_links_dependencyless_plans() {
        use cortex::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk};
        use futures::Stream;
        use std::pin::Pin;

        struct FlatPlanLlm;
        #[async_trait]
        impl LlmProvider for FlatPlanLlm {
            async fn generate(&self, _messages: &[Message]) -> Result<Response, LlmError> {
                Ok(Response {
                    content: r#"[
                        {"description": "scan dir", "action_type": "research", "depends_on": []},
                        {"description": "write script", "action_type": "implement", "depends_on": []},
                        {"description": "run script", "action_type": "execute", "command": "echo hi", "depends_on": []},
                        {"description": "notify user", "action_type": "notify", "depends_on": []}
                    ]"#.to_string(),
                    usage: None,
                })
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unimplemented!()
            }
            async fn health_check(&self) -> bool {
                true
            }
            fn name(&self) -> &str {
                "test"
            }
            fn model(&self) -> &str {
                "test-model"
            }
            async fn list_models(&self) -> Result<Vec<String>, LlmError> {
                Ok(vec!["test-model".into()])
            }
        }

        let llm = std::sync::Arc::new(FlatPlanLlm);
        let decomposer = LlmDecomposer::new(llm);
        let steps = decomposer
            .decompose("do something", DecompositionContext::default())
            .await
            .unwrap();

        assert_eq!(steps.len(), 4);
        // First step has no deps; rest are linked to predecessor.
        assert!(steps[0].depends_on.is_empty());
        assert_eq!(steps[1].depends_on, vec![steps[0].id.clone()]);
        assert_eq!(steps[2].depends_on, vec![steps[1].id.clone()]);
        assert_eq!(steps[3].depends_on, vec![steps[2].id.clone()]);
    }
}
