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

/// Decompose a user request into executable task steps.
#[async_trait]
pub trait TaskDecomposer: Send + Sync {
    async fn decompose(
        &self,
        request: &str,
        context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError>;
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
#[derive(Debug, Deserialize)]
struct RawStep {
    description: String,
    #[serde(default)]
    action_type: String,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    spec: Option<String>,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    artifact: Option<String>,
    #[serde(default)]
    channel: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    depends_on: Vec<usize>,
    #[serde(default)]
    tier: Option<String>,
    #[serde(default)]
    estimated_tokens: Option<u64>,
}

const DECOMPOSE_SYSTEM_PROMPT: &str = r#"You are a task planner for Brain OS. Given a user request, decompose it into executable steps.

Each step must be independently executable. Steps must have clear dependencies.

Output a JSON array of step objects with these fields:
- "description": human-readable description of the step
- "action_type": one of "research", "plan", "implement", "execute", "test", "review", "notify"
- "command": shell command (for execute/test action types)
- "query": search query (for research action type)
- "spec": implementation specification (for implement action type)
- "agent": which agent to use (for implement, e.g. "claude-code", "qwen")
- "artifact": what to review (for review action type)
- "channel": notification channel (for notify action type)
- "message": notification message (for notify action type)
- "depends_on": array of step indices (0-based) this step depends on
- "tier": action tier — "read", "write", "execute", "destructive", "external"
- "estimated_tokens": estimated LLM tokens needed (0 for non-LLM steps)

Dependency rules:
- Default to sequential dependencies — step N depends on step N-1 unless you can clearly justify true parallelism (e.g., two independent research queries).
- A plan that reads as a chain ("scan → write → run → verify → review → notify") MUST be encoded as a chain in `depends_on`. Do not produce a flat list of independent steps for inherently sequential work.

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

#[async_trait]
impl TaskDecomposer for LlmDecomposer {
    async fn decompose(
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
            user_prompt.push_str("\n\nAvailable tools: ");
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

        // Sequential-default fallback: if the LLM left every step's
        // depends_on empty across a multi-step plan, link them as a chain
        // so they don't all fire in parallel. The prompt asks for this
        // explicitly, but model output isn't always reliable.
        let none_have_deps = raw_steps.iter().all(|s| s.depends_on.is_empty());
        if raw_steps.len() > 1 && none_have_deps {
            for (i, step) in raw_steps.iter_mut().enumerate().skip(1) {
                step.depends_on = vec![i - 1];
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
            .map(|(i, raw)| {
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

                TaskStep {
                    id: ids[i].clone(),
                    description: raw.description,
                    action,
                    depends_on,
                    tier,
                    estimated_tokens: raw.estimated_tokens.unwrap_or(0),
                }
            })
            .collect();

        Ok(steps)
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
    fn test_parse_steps_markdown_wrapped() {
        let json = r#"```json
[{"description": "Do something", "action_type": "plan", "depends_on": []}]
```"#;

        let steps = parse_steps(json).unwrap();
        assert_eq!(steps.len(), 1);
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
