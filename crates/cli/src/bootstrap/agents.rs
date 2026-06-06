//! Agent delegation registry construction: `$PATH` auto-discovery plus
//! manual `agents.delegates[]` entries.

/// Build the agent delegation registry.
///
/// Two population paths compose:
/// 1. **Auto-discovery** — `$PATH` scan + version probe for known CLI
///    agents using the fingerprints in `delegate::default_fingerprints`.
///    Skipped when `agents.auto_discovery = false`.
/// 2. **Manual `agents.delegates[]` entries** — advanced/custom agents
///    that aren't fingerprinted. These always run and overwrite any
///    auto-discovered entry on name collision.
pub(super) async fn build_agent_registry(
    config: &brain::BrainConfig,
) -> anyhow::Result<delegate::AgentRegistry> {
    let mut registry = delegate::AgentRegistry::new();

    let overrides = delegate::DelegateOverrides {
        auto_discovery: config.agents.auto_discovery,
        overrides: config
            .agents
            .discovery_overrides
            .iter()
            .map(|(id, ov)| {
                (
                    id.clone(),
                    delegate::AgentOverride {
                        binary: ov.binary.as_ref().map(std::path::PathBuf::from),
                        disabled: ov.disabled,
                        capabilities: ov.capabilities.as_ref().map(|c| {
                            delegate::AgentCapabilities {
                                tags: c.tags.clone(),
                                languages: c.languages.clone(),
                                max_concurrency: c.max_concurrency,
                                needs_network: c.needs_network,
                            }
                        }),
                        args: ov.args.clone(),
                        prompt_via_stdin: ov.prompt_via_stdin,
                    },
                )
            })
            .collect(),
        custom: Vec::new(),
    };

    if overrides.auto_discovery {
        let discovery = delegate::DelegateDiscovery::new();
        let discovered = discovery.discover().await;
        tracing::info!(found = discovered.len(), "Agent discovery scan complete");
        for d in &discovered {
            tracing::debug!(
                agent = %d.agent_id,
                path = %d.path.display(),
                version = ?d.version,
                status = ?d.status,
                "Discovered candidate"
            );
        }
        registry.populate_from_discovery(discovered, &overrides);
    } else {
        tracing::info!("Agent auto-discovery disabled by config");
    }

    for entry in &config.agents.delegates {
        match entry.kind.as_str() {
            "subprocess" => {
                if entry.binary.is_empty() {
                    anyhow::bail!(
                        "agents.delegates[{}]: `subprocess` kind requires a non-empty `binary`",
                        entry.name
                    );
                }
                let spec = delegate::CustomAgentSpec {
                    id: entry.name.clone(),
                    binary: std::path::PathBuf::from(&entry.binary),
                    args: entry.args.clone(),
                    prompt_via_stdin: entry.prompt_via_stdin,
                    capabilities: delegate::AgentCapabilities {
                        tags: entry.tags.clone(),
                        languages: Vec::new(),
                        max_concurrency: 1,
                        needs_network: true,
                    },
                    workdir: entry.workdir.as_ref().map(std::path::PathBuf::from),
                    alias: entry.alias.clone(),
                };
                registry.register_subprocess_spec(&spec, delegate::AgentSource::Manual);
            }
            other => {
                tracing::warn!(
                    kind = %other,
                    name = %entry.name,
                    "Unknown agent kind — skipping (use `subprocess` or rely on auto-discovery)"
                );
            }
        }
    }

    // Surface misconfigured fallbacks at boot so the first delegation
    // failure doesn't become the discovery event.
    for fb in &config.agents.fallbacks {
        if !registry.contains(fb) {
            tracing::warn!(
                fallback = %fb,
                "agents.fallbacks references an unknown agent — nothing will catch a retryable failure for this name"
            );
        }
    }

    Ok(registry)
}
