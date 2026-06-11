//! Streaming finalization for SignalProcessor.

use crate::SignalError;
use crate::SignalProcessor;

impl SignalProcessor {
    /// Store the assistant response in episodic memory after streaming completes.
    ///
    /// Call this after streaming LLM generation finishes to persist the
    /// assistant turn in episodic memory. The `session_id` comes from the
    /// `PipelineResult::LlmReady` variant. Ensures the session row exists
    /// first to avoid FK constraint violations. Async because the episode
    /// runs through the memory-writer attestation gate (standing-approval
    /// lookup) like every other agent-attributed write.
    pub async fn finalize_streaming(
        &self,
        session_id: &str,
        assistant_content: &str,
        namespace: &str,
        agent: Option<&str>,
    ) -> Result<(), SignalError> {
        self.memory
            .episodic
            .ensure_session(session_id, "streaming")
            .map_err(|e| SignalError::Storage(e.to_string()))?;

        let episode_id = self
            .memory
            .episodic
            .store_episode(
                session_id,
                "assistant",
                assistant_content,
                0.5,
                Some(namespace),
                agent,
            )
            .map_err(|e| SignalError::Storage(e.to_string()))?;
        self.quarantine_episode_if_unattested(&episode_id, agent)
            .await;
        Ok(())
    }
}
