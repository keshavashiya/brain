//! BrainSession — CLI-specific wrapper around SignalProcessor.

use std::sync::Arc;

/// Result of context preparation (Phase 0).
pub(crate) enum PrepareResult {
    /// Thalamus dispatched an action — response is ready.
    ActionResult(String),
    /// Messages are assembled and ready for the LLM.
    LlmReady(Vec<cortex::llm::Message>),
}

pub(crate) struct BrainSession {
    /// Central signal processor — owns all stores (episodic, semantic, embedder, recall)
    /// plus the action dispatcher.
    pub(crate) processor: signal::SignalProcessor,
    /// LLM provider Arc — kept separately so streaming doesn't conflict with &mut self.
    pub(crate) llm: Arc<dyn cortex::llm::LlmProvider>,
    pub(crate) namespace: String,
    pub(crate) conversation_history: Vec<cortex::llm::Message>,
    pub(crate) session_id: String,
}

impl BrainSession {
    pub(crate) async fn new(config: &brain_core::BrainConfig) -> anyhow::Result<Self> {
        let processor = crate::bootstrap::build_processor(config).await?;

        let llm = processor.llm_arc();

        let session_id = processor.episodic().create_session("cli")?;

        Ok(Self {
            processor,
            llm,
            namespace: "personal".to_string(),
            conversation_history: Vec::new(),
            session_id,
        })
    }

    pub(crate) fn db(&self) -> &storage::SqlitePool {
        self.processor.episodic().pool()
    }

    /// Total active semantic facts.
    pub(crate) fn semantic_fact_count(&self) -> i64 {
        self.processor
            .episodic()
            .pool()
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT COUNT(*) FROM semantic_facts WHERE superseded_by IS NULL",
                    [],
                    |row| row.get(0),
                )
                .map_err(Into::into)
            })
            .unwrap_or(0)
    }

    pub(crate) fn episode_count(&self) -> i64 {
        self.processor
            .episodic()
            .pool()
            .with_conn(|conn| {
                conn.query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))
                    .map_err(Into::into)
            })
            .unwrap_or(0)
    }

    /// True while the user has fewer than 5 facts (onboarding phase).
    fn is_onboarding(&self) -> bool {
        self.semantic_fact_count() < 5
    }

    /// Record the onboarding greeting in episodic memory and conversation history.
    pub(crate) fn record_onboarding_greeting(&mut self) {
        let _ = self.processor.episodic().store_episode(
            &self.session_id,
            "assistant",
            cortex::context::ONBOARDING_GREETING,
            0.6,
            Some(&self.namespace),
            None,
        );
        self.conversation_history.push(cortex::llm::Message {
            role: cortex::llm::Role::Assistant,
            content: cortex::context::ONBOARDING_GREETING.to_string(),
        });
    }

    pub(crate) fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Prepare context by delegating to the unified SignalProcessor pipeline.
    pub(crate) async fn prepare_context(&mut self, message: &str) -> anyhow::Result<PrepareResult> {
        self.processor.set_action_namespace(&self.namespace);

        let signal = signal::Signal::new(signal::SignalSource::Cli, "cli", "user", message)
            .with_namespace(self.namespace.clone())
            .with_agent("cli");

        match self
            .processor
            .prepare(&signal, Some(&self.conversation_history))
            .await?
        {
            signal::PipelineResult::Complete(resp) => Ok(PrepareResult::ActionResult(
                signal::response_to_text(&resp.response),
            )),
            signal::PipelineResult::LlmReady {
                mut messages,
                session_id,
                ..
            } => {
                if let Some(sid) = session_id {
                    self.session_id = sid;
                }

                if self.is_onboarding() {
                    if let Some(sys) = messages.first_mut() {
                        sys.content.push_str(cortex::context::ONBOARDING_ADDENDUM);
                    }
                }

                Ok(PrepareResult::LlmReady(messages))
            }
        }
    }

    /// Store assistant response in episodic memory and update conversation history.
    pub(crate) fn finalize_response(
        &mut self,
        user_message: &str,
        assistant_content: &str,
    ) -> anyhow::Result<()> {
        use cortex::llm::{Message, Role};

        self.processor.finalize_streaming(
            &self.session_id,
            assistant_content,
            &self.namespace,
            None,
        )?;

        self.conversation_history.push(Message {
            role: Role::User,
            content: user_message.to_string(),
        });
        self.conversation_history.push(Message {
            role: Role::Assistant,
            content: assistant_content.to_string(),
        });

        if self.conversation_history.len() > 40 {
            self.conversation_history = self.conversation_history.split_off(20);
        }

        Ok(())
    }

    /// Convenience wrapper: prepare_context → generate → finalize (non-streaming).
    #[allow(dead_code)]
    pub(crate) async fn process_message(&mut self, message: &str) -> anyhow::Result<String> {
        match self.prepare_context(message).await? {
            PrepareResult::ActionResult(text) => Ok(text),
            PrepareResult::LlmReady(messages) => {
                let response = self.llm.generate(&messages).await?;
                self.finalize_response(message, &response.content)?;
                Ok(response.content)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_brain_session_schedule_intent_dispatches_and_persists() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.scheduling.enabled = true;
        config.actions.web_search.enabled = false;
        config.actions.messaging.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        session.namespace = "work".to_string();

        let result = session
            .prepare_context("remind me to ship release notes")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("schedule_task ok"));
                assert!(text.contains("namespace=work"));
            }
            _ => panic!("expected action dispatch result"),
        }

        let db = storage::SqlitePool::open(&config.sqlite_path()).unwrap();
        let intents = db.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(intents.len(), 1);
    }

    #[tokio::test]
    async fn test_brain_session_web_search_custom_without_endpoint_returns_explicit_error() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = brain_core::config::WebSearchProvider::Custom;
        config.actions.web_search.endpoint.clear();
        config.actions.messaging.enabled = false;
        config.actions.scheduling.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        let result = session
            .prepare_context("search for rust async")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("backend not configured"));
            }
            _ => panic!("expected action dispatch result"),
        }
    }

    #[tokio::test]
    async fn test_brain_session_tavily_without_api_key_returns_explicit_error() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = brain_core::config::WebSearchProvider::Tavily;
        config.actions.web_search.api_key.clear();
        config.actions.messaging.enabled = false;
        config.actions.scheduling.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        let result = session
            .prepare_context("search for rust async")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("backend not configured"));
            }
            _ => panic!("expected action dispatch result"),
        }
    }

    #[tokio::test]
    async fn test_brain_session_send_message_enabled_without_channel_mapping_explicit_error() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.messaging.enabled = true;
        config.actions.messaging.channels.clear();
        config.actions.web_search.enabled = false;
        config.actions.scheduling.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        let result = session
            .prepare_context("send via ops to alice saying deploy now")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("backend not configured"));
            }
            _ => panic!("expected action dispatch result"),
        }
    }
}
