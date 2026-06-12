//! Task-tier routing over the provider pool (`llm.tiers`).
//!
//! Three tiers, three chains: `fast` carries cheap kernel chores
//! (classification fallback, importance, compaction, web synthesis),
//! `deep` carries quality-sensitive generation (chat, decomposition),
//! `balanced` carries everything unrouted. Each configured tier is its
//! own [`FailoverProvider`] built from the named `llm.providers[]`
//! entries in order; an unconfigured tier shares the default startup
//! chain, so a config without `tiers` behaves exactly as before.
//!
//! Resolution fails closed: a tier name that matches no provider entry
//! is a startup error, never a silent fallback — a typo in
//! `tiers.fast: ["local"]` must not quietly reroute work that was meant
//! to stay on this machine onto a remote chain.

use std::sync::Arc;

use super::failover::FailoverProvider;
use super::{provider_config_from_entry, synthesise_entries, LlmError, LlmProvider};

/// Which chain a piece of work rides. See module docs for the cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskTier {
    Fast,
    Balanced,
    Deep,
}

impl TaskTier {
    /// Stable lowercase label — used as the `tier:<name>` budget-ledger
    /// key and in log lines.
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskTier::Fast => "fast",
            TaskTier::Balanced => "balanced",
            TaskTier::Deep => "deep",
        }
    }
}

impl std::fmt::Display for TaskTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The three resolved tier chains. Tiers left empty in config alias the
/// same default chain `Arc`, so pointer equality tells a renderer (or
/// test) whether a tier was actually routed somewhere else.
pub struct LlmTiers {
    pub fast: Arc<dyn LlmProvider>,
    pub balanced: Arc<dyn LlmProvider>,
    pub deep: Arc<dyn LlmProvider>,
}

impl std::fmt::Debug for LlmTiers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmTiers")
            .field("fast", &self.fast.name())
            .field("balanced", &self.balanced.name())
            .field("deep", &self.deep.name())
            .finish()
    }
}

impl LlmTiers {
    /// Every tier on one chain — the zero-config shape.
    pub fn uniform(chain: Arc<dyn LlmProvider>) -> Self {
        Self {
            fast: chain.clone(),
            balanced: chain.clone(),
            deep: chain,
        }
    }

    pub fn get(&self, tier: TaskTier) -> &Arc<dyn LlmProvider> {
        match tier {
            TaskTier::Fast => &self.fast,
            TaskTier::Balanced => &self.balanced,
            TaskTier::Deep => &self.deep,
        }
    }

    /// True when `tier` rides its own configured chain rather than the
    /// default one (`default_chain` being the chain passed to
    /// [`build_tier_chains`]).
    pub fn is_routed(&self, tier: TaskTier, default_chain: &Arc<dyn LlmProvider>) -> bool {
        !Arc::ptr_eq(self.get(tier), default_chain)
    }
}

/// Resolve `llm.tiers` against the provider pool into per-tier chains.
///
/// No network probing: tier chains use each entry's configured `model`
/// directly and rely on failover for outages, unlike the startup-probed
/// default chain. Unknown names and tiers whose every entry fails to
/// construct return an error (fail closed; see module docs).
pub fn build_tier_chains(
    llm: &brain::LlmConfig,
    default_chain: Arc<dyn LlmProvider>,
) -> Result<LlmTiers, LlmError> {
    if llm.tiers.is_unset() {
        return Ok(LlmTiers::uniform(default_chain));
    }
    let entries = synthesise_entries(llm);
    let chain_for = |tier: TaskTier, names: &[String]| -> Result<Arc<dyn LlmProvider>, LlmError> {
        if names.is_empty() {
            return Ok(default_chain.clone());
        }
        let mut providers: Vec<Box<dyn LlmProvider>> = Vec::with_capacity(names.len());
        for name in names {
            let Some(entry) = entries.iter().find(|e| &e.name == name) else {
                return Err(LlmError::ProviderUnavailable(format!(
                    "llm.tiers.{tier} names provider `{name}`, which matches no llm.providers[] entry — \
                     fix the name; a tier is never silently rerouted"
                )));
            };
            let cfg =
                provider_config_from_entry(entry, llm.temperature, llm.max_tokens as i32, None);
            match super::create_provider(&cfg) {
                Ok(p) => providers.push(p),
                Err(e) => {
                    tracing::warn!(tier = %tier, provider = %name, error = %e, "tier provider construction failed — skipping");
                }
            }
        }
        if providers.is_empty() {
            return Err(LlmError::ProviderUnavailable(format!(
                "llm.tiers.{tier} resolved to zero working providers"
            )));
        }
        tracing::info!(tier = %tier, chain = ?names, "LLM tier chain configured");
        Ok(Arc::new(FailoverProvider::new(providers)))
    };

    Ok(LlmTiers {
        fast: chain_for(TaskTier::Fast, &llm.tiers.fast)?,
        balanced: chain_for(TaskTier::Balanced, &llm.tiers.balanced)?,
        deep: chain_for(TaskTier::Deep, &llm.tiers.deep)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::Stream;
    use std::pin::Pin;

    struct NullProvider;

    #[async_trait::async_trait]
    impl LlmProvider for NullProvider {
        async fn generate(
            &self,
            _messages: &[super::super::Message],
        ) -> Result<super::super::Response, LlmError> {
            Ok(super::super::Response::text("null", None))
        }
        async fn generate_stream(
            &self,
            _messages: &[super::super::Message],
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<super::super::ResponseChunk, LlmError>> + Send>>,
            LlmError,
        > {
            unimplemented!()
        }
        async fn health_check(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "null"
        }
        fn model(&self) -> &str {
            "null"
        }
        async fn list_models(&self) -> Result<Vec<String>, LlmError> {
            Ok(vec![])
        }
    }

    fn cfg_with_providers(names: &[&str]) -> brain::LlmConfig {
        let mut cfg = brain::BrainConfig::default().llm;
        cfg.providers = names
            .iter()
            .map(|n| brain::ProviderEntry {
                name: n.to_string(),
                kind: "openai_compat".to_string(),
                base_url: "http://127.0.0.1:9".to_string(),
                api_key: String::new(),
                api_key_file: None,
                model: format!("{n}-model"),
                preferred_models: Vec::new(),
            })
            .collect();
        cfg
    }

    fn default_chain() -> Arc<dyn LlmProvider> {
        Arc::new(NullProvider)
    }

    #[test]
    fn unset_tiers_alias_the_default_chain() {
        let cfg = cfg_with_providers(&["a", "b"]);
        let dc = default_chain();
        let tiers = build_tier_chains(&cfg, dc.clone()).unwrap();
        for tier in [TaskTier::Fast, TaskTier::Balanced, TaskTier::Deep] {
            assert!(
                Arc::ptr_eq(tiers.get(tier), &dc),
                "{tier} must alias default"
            );
            assert!(!tiers.is_routed(tier, &dc));
        }
    }

    #[test]
    fn named_tier_builds_its_own_chain_and_others_stay_default() {
        let mut cfg = cfg_with_providers(&["local", "cloud"]);
        cfg.tiers.fast = vec!["local".to_string()];
        let dc = default_chain();
        let tiers = build_tier_chains(&cfg, dc.clone()).unwrap();
        assert!(tiers.is_routed(TaskTier::Fast, &dc));
        assert_eq!(
            tiers.fast.model(),
            "local-model",
            "chain head is the named entry"
        );
        assert!(!tiers.is_routed(TaskTier::Balanced, &dc));
        assert!(!tiers.is_routed(TaskTier::Deep, &dc));
    }

    #[test]
    fn tier_chain_preserves_configured_order() {
        let mut cfg = cfg_with_providers(&["local", "cloud"]);
        cfg.tiers.deep = vec!["cloud".to_string(), "local".to_string()];
        let tiers = build_tier_chains(&cfg, default_chain()).unwrap();
        // FailoverProvider reports its first (primary) member.
        assert_eq!(tiers.deep.model(), "cloud-model");
    }

    #[test]
    fn unknown_tier_name_fails_closed() {
        let mut cfg = cfg_with_providers(&["local"]);
        cfg.tiers.fast = vec!["loacl".to_string()]; // typo
        let err = build_tier_chains(&cfg, default_chain()).unwrap_err();
        assert!(
            err.to_string().contains("loacl"),
            "error must name the bad entry: {err}"
        );
    }

    #[test]
    fn legacy_single_provider_config_resolves_by_synthesised_name() {
        // providers[] empty → the legacy fields synthesise one entry named
        // "default"; a tier may reference it.
        let mut cfg = brain::BrainConfig::default().llm;
        cfg.providers.clear();
        cfg.tiers.fast = vec!["default".to_string()];
        let dc = default_chain();
        let tiers = build_tier_chains(&cfg, dc.clone()).unwrap();
        assert!(tiers.is_routed(TaskTier::Fast, &dc));
    }
}
