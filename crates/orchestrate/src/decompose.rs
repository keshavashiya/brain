//! LLM-based task decomposition + procedural memory validation.
//!
//! Pipeline: user request → LLM generates candidate steps (JSON) →
//! cerebellum validates against known patterns → tier assignment → output.

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::step::TaskStep;

mod parse;
mod validate;
use parse::{parse_steps, RawStep};

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
    /// Sandbox binary allowlist. `execute`/`test` steps must start with
    /// one of these; the planner surfaces it and the validation pass
    /// rejects argv steps that name anything else.
    pub available_tools: Vec<String>,
    /// Relevant facts from semantic memory.
    pub relevant_facts: Vec<String>,
    /// Credential scopes available in the vault (tool names, not values).
    pub available_credentials: Vec<String>,
    /// Names of delegate agents the registry can actually dispatch to
    /// (from [`delegate::AgentRegistry::list`]). When non-empty, an
    /// `implement` step that names an agent outside this set is rejected
    /// at plan time instead of failing once execution reaches it.
    pub available_agents: Vec<String>,
    /// Live capability manifest summary lines — native backends, mounted
    /// MCP server actions, the terminal — so the planner composes against
    /// faculties that actually exist instead of inventing them. Advisory:
    /// surfaced in the prompt but not used as a hard reject gate (mapping a
    /// free-text step to a manifest tool is fuzzy; a false reject is worse
    /// than letting execution-time gating handle the edge).
    pub available_capabilities: Vec<String>,
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
    /// What already succeeded — description **and** a stdout excerpt so
    /// the LLM can ground the next step in the data those steps actually
    /// produced (instead of inventing intermediate file names).
    pub completed: Vec<CompletedStepRecap>,
}

/// One completed-step recap fed back into the replan prompt.
#[derive(Debug, Clone)]
pub struct CompletedStepRecap {
    pub description: String,
    /// Trimmed stdout from the step. The orchestrator caps length so a
    /// single noisy step can't crowd out the rest of the prompt.
    pub output_excerpt: String,
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
        if !context.available_capabilities.is_empty() {
            user_prompt.push_str(
                "\n\nLive kernel capabilities (faculties wired right now — compose against these, do not invent others):\n",
            );
            for cap in &context.available_capabilities {
                user_prompt.push_str(&format!("- {cap}\n"));
            }
        }
        if !context.available_agents.is_empty() {
            user_prompt.push_str(
                "\n\nDelegate agents available for `implement` steps (the `agent` field MUST be exactly one of these):\n  ",
            );
            user_prompt.push_str(&context.available_agents.join(", "));
        }

        let messages = vec![
            cortex::llm::Message::system(crate::prompts::DECOMPOSE_SYSTEM),
            cortex::llm::Message::user(user_prompt),
        ];

        let response = self.llm.generate(&messages).await?;
        let mut raw_steps = parse_steps(&response.content)?;

        if raw_steps.is_empty() {
            return Err(DecompositionError::EmptyPlan);
        }

        validate::validate_steps(&raw_steps, &context)?;
        validate::apply_sequential_fallback(&mut raw_steps);
        Ok(validate::finalize(raw_steps))
    }
}

impl LlmDecomposer {
    async fn replan_inner(
        &self,
        repair: &RepairContext,
        context: &DecompositionContext,
    ) -> Result<Vec<RawStep>, DecompositionError> {
        let mut user_prompt = format!(
            "Original request:\n  {}\n\nWhat already succeeded (do NOT redo). Each entry includes the actual stdout the step produced — base your next step on this real data, do not invent intermediate files:\n",
            repair.original_request
        );
        if repair.completed.is_empty() {
            user_prompt.push_str("  (nothing yet)\n");
        } else {
            for recap in &repair.completed {
                user_prompt.push_str(&format!("  - {}\n", recap.description));
                let excerpt = recap.output_excerpt.trim();
                if excerpt.is_empty() {
                    user_prompt.push_str("    (no stdout)\n");
                } else {
                    user_prompt.push_str("    stdout:\n");
                    for line in excerpt.lines() {
                        user_prompt.push_str(&format!("      {line}\n"));
                    }
                }
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
        if !context.available_agents.is_empty() {
            user_prompt.push_str(
                "\nDelegate agents available for `implement` steps (the `agent` field MUST be one of these):\n  ",
            );
            user_prompt.push_str(&context.available_agents.join(", "));
        }

        let messages = vec![
            cortex::llm::Message::system(crate::prompts::REPAIR_SYSTEM),
            cortex::llm::Message::user(user_prompt),
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

        // Re-use the exact same validation + sequential-fallback path as
        // the main decompose flow so a bad LLM response can't create a
        // worse plan than the one we just failed on.
        validate::validate_steps(&raw_steps, &context)?;
        validate::apply_sequential_fallback(&mut raw_steps);
        Ok(validate::finalize(raw_steps))
    }

    async fn decompose(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        self.decompose_impl(request, context).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::StepAction;

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
                Ok(Response::text(
                    r#"[
                        {"description": "run the script", "action_type": "execute", "command": "", "depends_on": []}
                    ]"#,
                    None,
                ))
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unreachable!("mock provider: the decomposer never streams")
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
                Ok(Response::text(
                    r#"[
                        {"description": "check act installed", "action_type": "execute", "command": "which act", "depends_on": []}
                    ]"#,
                    None,
                ))
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unreachable!("mock provider: the decomposer never streams")
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
                Ok(Response::text(
                    r#"[
                        {"description": "pipeline step", "action_type": "execute", "command": "ls | grep foo", "depends_on": []}
                    ]"#,
                    None,
                ))
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unreachable!("mock provider: the decomposer never streams")
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
                Ok(Response::text(
                    r#"[
                        {"description": "scan dir", "action_type": "research", "depends_on": []},
                        {"description": "write script", "action_type": "implement", "depends_on": []},
                        {"description": "run script", "action_type": "execute", "command": "echo hi", "depends_on": []},
                        {"description": "notify user", "action_type": "notify", "depends_on": []}
                    ]"#,
                    None,
                ))
            }
            async fn generate_stream(
                &self,
                _messages: &[Message],
            ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>
            {
                unreachable!("mock provider: the decomposer never streams")
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

    /// Minimal LLM stub that always returns one canned plan, for the
    /// agent-validation tests below.
    struct CannedLlm(&'static str);
    #[async_trait]
    impl cortex::llm::LlmProvider for CannedLlm {
        async fn generate(
            &self,
            _messages: &[cortex::llm::Message],
        ) -> Result<cortex::llm::Response, cortex::llm::LlmError> {
            Ok(cortex::llm::Response::text(self.0, None))
        }
        async fn generate_stream(
            &self,
            _messages: &[cortex::llm::Message],
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<
                            Item = Result<cortex::llm::ResponseChunk, cortex::llm::LlmError>,
                        > + Send,
                >,
            >,
            cortex::llm::LlmError,
        > {
            unreachable!("mock provider: the decomposer never streams")
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
        async fn list_models(&self) -> Result<Vec<String>, cortex::llm::LlmError> {
            Ok(vec!["test-model".into()])
        }
    }

    const IMPLEMENT_WITH_GHOST_AGENT: &str = r#"[
        {"description": "do the work", "action_type": "implement", "spec": "build it", "agent": "ghost-agent", "depends_on": []}
    ]"#;

    #[tokio::test]
    async fn rejects_implement_step_with_unregistered_agent() {
        // The caller supplies the live agent roster; an `implement` step
        // naming an agent outside it must fail at plan time, not five
        // steps into execution.
        let llm = std::sync::Arc::new(CannedLlm(IMPLEMENT_WITH_GHOST_AGENT));
        let decomposer = LlmDecomposer::new(llm);
        let ctx = DecompositionContext {
            available_agents: vec!["claude-code".into(), "qwen".into()],
            ..Default::default()
        };
        let err = decomposer.decompose("anything", ctx).await.unwrap_err();
        match err {
            DecompositionError::Parse(msg) => {
                assert!(
                    msg.contains("ghost-agent") && msg.contains("not registered"),
                    "expected agent-rejection message, got: {msg}"
                );
                assert!(
                    msg.contains("claude-code") && msg.contains("qwen"),
                    "rejection should list available agents, got: {msg}"
                );
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn accepts_implement_step_with_registered_agent() {
        let llm = std::sync::Arc::new(CannedLlm(
            r#"[{"description": "do the work", "action_type": "implement", "spec": "build it", "agent": "claude-code", "depends_on": []}]"#,
        ));
        let decomposer = LlmDecomposer::new(llm);
        let ctx = DecompositionContext {
            available_agents: vec!["claude-code".into(), "qwen".into()],
            ..Default::default()
        };
        let steps = decomposer.decompose("anything", ctx).await.unwrap();
        assert_eq!(steps.len(), 1);
        assert!(matches!(
            &steps[0].action,
            StepAction::Implement { agent, .. } if agent == "claude-code"
        ));
    }

    #[tokio::test]
    async fn skips_agent_validation_when_roster_unknown() {
        // No registry wired (empty roster) ⇒ no validation; the planner
        // keeps its prior behavior and the step is built as-is.
        let llm = std::sync::Arc::new(CannedLlm(IMPLEMENT_WITH_GHOST_AGENT));
        let decomposer = LlmDecomposer::new(llm);
        let steps = decomposer
            .decompose("anything", DecompositionContext::default())
            .await
            .unwrap();
        assert_eq!(steps.len(), 1);
        assert!(matches!(
            &steps[0].action,
            StepAction::Implement { agent, .. } if agent == "ghost-agent"
        ));
    }
}
