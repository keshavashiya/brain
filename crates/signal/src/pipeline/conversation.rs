//! Conversation-category intent handler: free-form chat (the catch-all
//! classification). Returns [`PipelineResult::LlmReady`] so the caller
//! (`SignalProcessor::process`) can run LLM generation — for agent callers
//! it returns [`PipelineResult::Complete`] with structured memory context
//! instead.
//!
//! Variant: [`thalamus::Intent::Chat`].

use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

impl SignalProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_chat(
        &self,
        signal_id: Uuid,
        signal: &Signal,
        content: String,
        importance: f32,
        conversation_history: Option<&[cortex::llm::Message]>,
        procedure_context: &[String],
        prepend_nudges: &impl Fn(SignalResponse) -> SignalResponse,
        progress: Option<&tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let top_k = self.config.memory.semantic.max_results as usize;
        if let Some(tx) = progress {
            let _ = tx.try_send("searching…");
        }
        let query_vector = self.embed_text(&content).await;
        let (memories, facts_used, episodes_used) = self
            .do_recall(&content, query_vector, top_k, Some(&signal.namespace))
            .await;

        // Reuse caller-supplied session or create a new one
        let session_id = if let Some(ref sid) = signal.session_id {
            // Ensure the session row exists so FK constraints on episodes never fail.
            // This handles the case where a client reuses a session_id from a
            // previous daemon run that was cleared.
            self.episodic
                .ensure_session(sid, &signal.channel)
                .map_err(|e| SignalError::Storage(e.to_string()))?;
            sid.clone()
        } else {
            self.episodic
                .create_session(&signal.channel)
                .map_err(|e| SignalError::Storage(e.to_string()))?
        };

        self.episodic
            .store_episode(
                &session_id,
                "user",
                &signal.content,
                importance as f64,
                Some(&signal.namespace),
                signal.agent.as_deref(),
            )
            .map_err(|e| SignalError::Storage(e.to_string()))?;

        // Agent callers get structured memory context
        if signal.agent.is_some() {
            let response_text = if memories.is_empty() {
                format!(
                    "Stored episode. No relevant memories found for: {}",
                    content
                )
            } else {
                let mem_lines: String = memories
                    .iter()
                    .map(|m| format!("[{:?}] {}", m.source, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("Stored episode. Relevant memories:\n{}", mem_lines)
            };

            let resp = prepend_nudges(SignalResponse {
                signal_id,
                status: ResponseStatus::Ok,
                response: ResponseContent::Text(response_text),
                memory_context: MemoryContext {
                    facts_used,
                    episodes_used,
                },
                session_id: Some(session_id.clone()),
            });
            return Ok(PipelineResult::Complete(resp));
        }

        let proc_history: Vec<cortex::llm::Message> = procedure_context
            .iter()
            .map(|step| cortex::llm::Message {
                role: cortex::llm::Role::User,
                content: format!("[procedure step] {step}"),
            })
            .collect();
        let history = conversation_history.unwrap_or(&proc_history);
        let namespace_is_empty = self.list_facts(Some(&signal.namespace)).is_empty()
            && self.recent_episodes(1, Some(&signal.namespace)).is_empty();
        let addendum = if namespace_is_empty {
            Some(cortex::context::ONBOARDING_ADDENDUM)
        } else {
            None
        };

        // Path-attachment grounding: if the user named one or more
        // local paths in `content`, read each on their behalf and pass
        // the snapshots to the assembler. Replaces the old
        // `Intent::ProjectInspect` branch that bypassed SOUL entirely.
        let attachments = crate::attachment::build_chat_attachments(
            &content,
            &self.config.security.allowed_paths,
        );
        if !attachments.is_empty() {
            tracing::debug!(
                attached = attachments.attached.len(),
                skipped = attachments.skipped.len(),
                "chat turn carries path attachments"
            );
        }

        let messages = self.context_assembler.assemble_full(
            &content,
            &memories,
            history,
            addendum,
            &attachments.attached,
            &attachments.skipped,
        );

        Ok(PipelineResult::LlmReady {
            signal_id,
            messages,
            memory_context: MemoryContext {
                facts_used,
                episodes_used,
            },
            session_id: Some(session_id),
            user_content: content,
            namespace: signal.namespace.clone(),
            agent: signal.agent.clone(),
        })
    }
}
