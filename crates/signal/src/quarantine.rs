//! Memory-writer quarantine: writes from agents nobody vouched for land
//! in a reviewable holding state instead of live memory.
//!
//! An agent is **attested** as a memory writer when the user vouched for
//! it somewhere: a `memory.trust.agents` entry (config), an API key
//! bound to its identity (config), or a standing `memory.write` approval
//! (runtime, granted via `/memory-approve`, revocable via
//! `/approval-revoke`). Anything else writing under a claimed agent id
//! gets its rows quarantined — stored and auditable, but excluded from
//! recall, search, listings, and consolidation until reviewed. The
//! quarantine fails closed *visibly*: counts surface in `/grants` and
//! the capability digest.

use crate::SignalProcessor;

/// Per-agent quarantine totals for the review surfaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantineCount {
    pub agent: String,
    pub facts: i64,
    pub episodes: i64,
}

impl SignalProcessor {
    /// True when `agent` is vouched for as a memory writer. User-origin
    /// writes (`None`) are always attested.
    pub(crate) async fn memory_writer_attested(&self, agent: Option<&str>) -> bool {
        let Some(agent) = agent else {
            return true;
        };
        if self.config.memory.trust.agents.contains_key(agent) {
            return true;
        }
        if self
            .config
            .access
            .api_keys
            .iter()
            .any(|k| k.agent_id.as_deref() == Some(agent))
        {
            return true;
        }
        if let Some(store) = &self.safety.standing_approvals {
            if store
                .is_granted(&confirm::GrantKey::new(agent, "memory", "write"))
                .await
                .unwrap_or(false)
            {
                return true;
            }
        }
        false
    }

    /// Quarantine a just-stored episode when its writer is unattested.
    pub(crate) async fn quarantine_episode_if_unattested(
        &self,
        episode_id: &str,
        agent: Option<&str>,
    ) {
        if self.memory_writer_attested(agent).await {
            return;
        }
        let agent = agent.expect("unattested implies an agent id");
        if let Err(e) = self.memory.episodic.quarantine_episode(episode_id, agent) {
            tracing::warn!(episode_id, agent, "failed to quarantine episode: {e}");
        } else {
            tracing::info!(
                episode_id,
                agent,
                "memory quarantine: episode from unattested writer held for review"
            );
        }
    }

    /// Quarantine a just-stored fact when its writer is unattested.
    pub(crate) async fn quarantine_fact_if_unattested(&self, fact_id: &str, agent: Option<&str>) {
        if self.memory_writer_attested(agent).await {
            return;
        }
        let agent = agent.expect("unattested implies an agent id");
        let Some(semantic) = &self.memory.semantic else {
            return;
        };
        if let Err(e) = semantic.quarantine_fact(fact_id, agent) {
            tracing::warn!(fact_id, agent, "failed to quarantine fact: {e}");
        } else {
            tracing::info!(
                fact_id,
                agent,
                "memory quarantine: fact from unattested writer held for review"
            );
        }
    }

    /// Quarantined memory totals per agent, merged across facts and
    /// episodes — the data behind the `/grants` section and the digest
    /// line.
    pub(crate) fn quarantined_memory_counts(&self) -> Vec<QuarantineCount> {
        let mut by_agent: std::collections::BTreeMap<String, (i64, i64)> = Default::default();
        if let Some(semantic) = &self.memory.semantic {
            for (agent, n) in semantic.quarantined_fact_counts().unwrap_or_default() {
                by_agent.entry(agent).or_default().0 += n;
            }
        }
        for (agent, n) in self
            .memory
            .episodic
            .quarantined_episode_counts()
            .unwrap_or_default()
        {
            by_agent.entry(agent).or_default().1 += n;
        }
        by_agent
            .into_iter()
            .map(|(agent, (facts, episodes))| QuarantineCount {
                agent,
                facts,
                episodes,
            })
            .collect()
    }

    /// Approve `agent` as a memory writer: record a standing
    /// `memory.write` approval (so future writes land live, revocable
    /// via `/approval-revoke`) and release everything it has in
    /// quarantine. Returns (released facts, released episodes,
    /// standing-approval id).
    pub(crate) async fn approve_memory_writer(
        &self,
        agent: &str,
    ) -> Result<(usize, usize, String), crate::SignalError> {
        let store = self.safety.standing_approvals.as_ref().ok_or_else(|| {
            crate::SignalError::Processing("Standing-approval store is not wired".to_string())
        })?;
        let key = confirm::GrantKey::new(agent, "memory", "write");
        let grant_id = store
            .grant(&key, Some("memory writer approved via /memory-approve"))
            .await
            .map_err(|e| crate::SignalError::Processing(format!("grant failed: {e}")))?;

        let facts = match &self.memory.semantic {
            Some(semantic) => semantic
                .release_quarantined_facts(agent)
                .map_err(|e| crate::SignalError::Storage(e.to_string()))?,
            None => 0,
        };
        let episodes = self
            .memory
            .episodic
            .release_quarantined_episodes(agent)
            .map_err(|e| crate::SignalError::Storage(e.to_string()))?;
        tracing::info!(
            agent,
            facts,
            episodes,
            grant_id,
            "memory quarantine: writer approved, memories released"
        );
        Ok((facts, episodes, grant_id))
    }
}
