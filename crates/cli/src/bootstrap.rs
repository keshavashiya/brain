//! Shared bootstrap — single source of truth for building a fully-wired SignalProcessor.
//!
//! Used by `brain serve`, `brain chat`, and `brain mcp` to eliminate
//! bootstrap duplication and ensure consistent backend wiring.

use std::sync::Arc;

#[cfg(feature = "encryption")]
use crate::encryption::resolve_encryptor;
use crate::encryption::resolve_llm_api_key;
use backends::*;

/// Build a fully-wired `SignalProcessor` from config.
///
/// This is the canonical bootstrap path. All CLI commands that need a processor
/// should call this instead of wiring backends individually.
///
/// What gets wired:
/// - Encryption (if enabled)
/// - Action dispatcher with memory backend
/// - Web search backend (searxng / tavily / custom)
/// - Scheduling backend
/// - Messaging backend (webhooks)
/// - Phase 1 safety infrastructure (audit, confirm, budget, sandbox)
///
/// What is NOT wired here (caller-specific):
/// - Notification router (only needed for `brain serve`)
/// - Background tasks (consolidation, proactivity — only `brain serve`)
/// - Credential vault (wired separately via `wire_vault`)
pub async fn build_processor(
    config: &brain_core::BrainConfig,
) -> anyhow::Result<signal::SignalProcessor> {
    #[cfg(feature = "encryption")]
    let mut processor = {
        let encryptor = resolve_encryptor(config)?;
        signal::SignalProcessor::new_with_encryptor(config.clone(), encryptor).await?
    };
    #[cfg(not(feature = "encryption"))]
    let mut processor = signal::SignalProcessor::new(config.clone()).await?;

    let action_dispatcher = build_action_dispatcher(config, &processor)?;
    processor = processor.with_action_dispatcher(action_dispatcher);

    // ── Phase 1: Safety infrastructure ──────────────────────────────────
    processor = wire_safety_infrastructure(processor, config).await?;

    Ok(processor)
}

/// Wire Phase 1 safety infrastructure into the processor.
///
/// Components: audit trail, confirmation engine, cost budget, sandbox executor.
/// All share the same SQLite pool as the episodic store for simplicity.
/// The credential vault is NOT wired here — it requires passphrase input
/// and is wired on demand (e.g. `brain vault` / `brain auth` commands).
async fn wire_safety_infrastructure(
    processor: signal::SignalProcessor,
    config: &brain_core::BrainConfig,
) -> anyhow::Result<signal::SignalProcessor> {
    let db = processor.episodic().pool().clone();

    // Audit trail — always wired (foundation for all other Phase 1 components)
    let audit_trail = audit::SqliteAuditTrail::new(db.clone());
    audit_trail
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Audit trail table init failed: {e}"))?;
    let audit_trail: Arc<dyn audit::AuditTrail> = Arc::new(audit_trail);
    tracing::info!("Audit trail wired");

    // Channel preference store + router + dispatcher — built before the
    // confirmation engine so the engine can attach the notifier hook that
    // pushes approval prompts out to the user. Transports register with
    // the dispatcher later (in `serve.rs::wire_preset_transports`).
    let pref_store = channel::SqlitePreferenceStore::new(db.clone());
    pref_store
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Channel preference table init failed: {e}"))?;
    let preferences: Arc<dyn channel::ChannelPreferenceStore> = Arc::new(pref_store);
    let router: Arc<dyn channel::ChannelRouter> =
        Arc::new(channel::DefaultChannelRouter::new(preferences.clone()));
    let dispatcher = Arc::new(channel::ChannelDispatcher::new(router.clone()));

    // Confirmation engine — always wired, with notifier hook so approval
    // prompts actually reach the user instead of deadlocking on timeout.
    let approval_notifier: Arc<dyn confirm::ApprovalNotifier> =
        Arc::new(signal::ChannelApprovalNotifier::new(dispatcher.clone()));
    let confirm_engine =
        confirm::SqliteConfirmationEngine::new(db.clone()).with_notifier(approval_notifier);
    confirm_engine
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Confirmation engine table init failed: {e}"))?;
    let confirm_engine: Arc<dyn confirm::ConfirmationEngine> = Arc::new(confirm_engine);
    tracing::info!("Confirmation engine wired (with channel-backed approval notifier)");

    // Cost budget — always wired, with audit coupling
    let budget_policy = budget::BudgetPolicy::default();
    let sqlite_budget = budget::SqliteBudget::new(db.clone(), budget_policy);
    sqlite_budget
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Cost budget table init failed: {e}"))?;
    let sqlite_budget = sqlite_budget.with_audit(audit_trail.clone());
    let cost_budget: Arc<dyn budget::CostBudget> = Arc::new(sqlite_budget);
    tracing::info!("Cost budget wired (with audit coupling)");

    // Sandbox executor — isolated invocation with rlimits, allowlist, and
    // platform layers (macOS sandbox-exec / Linux namespaces).
    let exec_timeout = std::time::Duration::from_secs(config.security.exec_timeout_seconds as u64);
    let sandbox =
        sandbox::IsolatedSandbox::new(config.security.exec_allowlist.clone(), exec_timeout)
            .with_allowed_paths(vec![
                std::path::PathBuf::from(&config.brain.data_dir),
                std::env::current_dir().unwrap_or_default(),
            ]);
    let sandbox_executor: Arc<dyn sandbox::SandboxExecutor> = Arc::new(sandbox);
    tracing::info!(
        allowlist_size = config.security.exec_allowlist.len(),
        timeout_s = config.security.exec_timeout_seconds,
        "Sandbox executor wired (isolated: rlimits + allowlist)"
    );

    let processor = processor
        .with_audit_trail(audit_trail.clone())
        .with_confirmation_engine(confirm_engine.clone())
        .with_cost_budget(cost_budget)
        .with_sandbox_executor(sandbox_executor.clone());

    // ── Agent registry (Phase 3) ────────────────────────────────────────
    // Built before the orchestrator so `Implement` steps can dispatch to
    // registered specialist agents. Discovery scans `$PATH` for known
    // CLI agents at boot; manual `agents.delegates[]` entries still
    // work alongside (last-write-wins on name collisions).
    let agent_registry = build_agent_registry(config).await?;
    let agent_registry_arc = Arc::new(agent_registry);
    if !agent_registry_arc.is_empty() {
        tracing::info!(
            agents = ?agent_registry_arc.list(),
            "Agent delegation registry wired"
        );
    } else {
        tracing::info!("Agent delegation registry empty — Implement steps will require config");
    }

    // Task orchestrator — wired with the LLM provider for decomposition
    let decomposer: Arc<dyn orchestrate::TaskDecomposer> =
        Arc::new(orchestrate::LlmDecomposer::new(processor.llm_arc()));
    let escalation_policy = delegate::EscalationPolicy {
        fallbacks: config.agents.fallbacks.clone(),
        retry_on_timeout: config.agents.retry_on_timeout,
    };
    let orchestrator = orchestrate::TaskOrchestrator::new(decomposer)
        .with_audit(audit_trail)
        .with_confirmation(confirm_engine.clone())
        .with_sandbox(sandbox_executor)
        .with_agents(agent_registry_arc.clone())
        .with_channel_dispatcher(dispatcher.clone())
        .with_llm(processor.llm_arc())
        .with_episodic(Arc::new(hippocampus::EpisodicStore::new(db.clone())))
        .with_delegation_policy(escalation_policy)
        // Cache the sandbox allowlist so the replan-on-failure loop can
        // include it in its corrective LLM call.
        .with_available_tools(config.security.exec_allowlist.clone());
    let processor = processor
        .with_orchestrator(Arc::new(orchestrator))
        .with_agent_registry(agent_registry_arc);
    tracing::info!("Task orchestrator wired");

    // ── Channel intelligence — bind the pieces we built above ──────────
    let correlator = Arc::new(channel::ConfirmationCorrelator::new(confirm_engine));
    let processor = processor
        .with_channel_preferences(preferences)
        .with_channel_router(router)
        .with_confirmation_correlator(correlator)
        .with_channel_dispatcher(dispatcher);
    tracing::info!("Channel intelligence wired (router + dispatcher + preferences + correlator)");

    Ok(processor)
}

/// Build the agent delegation registry.
///
/// Two population paths compose:
/// 1. **Auto-discovery** — `$PATH` scan + version probe for known CLI
///    agents using the fingerprints in `delegate::default_fingerprints`.
///    Skipped when `agents.auto_discovery = false`.
/// 2. **Manual `agents.delegates[]` entries** — advanced/custom agents
///    that aren't fingerprinted. These always run and overwrite any
///    auto-discovered entry on name collision.
async fn build_agent_registry(
    config: &brain_core::BrainConfig,
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
                        capabilities: None,
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
        let capabilities = delegate::AgentCapabilities {
            tags: entry.tags.clone(),
            languages: Vec::new(),
            max_concurrency: 1,
            needs_network: true,
        };

        let workdir = entry.workdir.as_ref().map(std::path::PathBuf::from);

        match entry.kind.as_str() {
            "subprocess" => {
                if entry.binary.is_empty() {
                    anyhow::bail!(
                        "agents.delegates[{}]: `subprocess` kind requires a non-empty `binary`",
                        entry.name
                    );
                }
                let mut sub_cfg = delegate::SubprocessAgentConfig::new(&entry.name, &entry.binary)
                    .with_args(entry.args.clone())
                    .with_capabilities(capabilities)
                    .with_prompt_via_stdin(entry.prompt_via_stdin);
                if let Some(dir) = workdir {
                    sub_cfg = sub_cfg.with_workdir(dir);
                }
                let binary_path = std::path::PathBuf::from(&entry.binary);
                let d: Arc<dyn delegate::AgentDelegate> =
                    Arc::new(delegate::SubprocessAgentDelegate::new(sub_cfg));
                registry.register_manual(d, binary_path, None);
            }
            other => {
                tracing::warn!(
                    kind = %other,
                    name = %entry.name,
                    "Unknown agent kind — skipping (use `subprocess` or rely on auto-discovery)"
                );
                continue;
            }
        }

        if let Some(alias) = &entry.alias {
            registry.alias(alias.clone(), entry.name.clone());
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

/// Build the action dispatcher with all configured backends.
fn build_action_dispatcher(
    config: &brain_core::BrainConfig,
    processor: &signal::SignalProcessor,
) -> anyhow::Result<cortex::actions::ActionDispatcher> {
    let embedding_dim = processor.embedding_dim();
    let llm_api_key = resolve_llm_api_key(config);
    let embedder = Arc::new(tokio::sync::Mutex::new(
        hippocampus::Embedder::from_config(
            &config.llm.provider,
            &config.llm.base_url,
            &config.embedding.model,
            &llm_api_key,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create embedding provider: {e}"))?,
    ));

    let action_backend = Arc::new(DefaultMemoryBackend {
        semantic: processor.semantic().cloned(),
        embedder,
        embedding_dim,
    });

    let action_config = cortex::actions::ActionConfig {
        command_allowlist: config.security.exec_allowlist.clone(),
        command_timeout_secs: config.security.exec_timeout_seconds as u64,
        enable_web_search: config.actions.web_search.enabled,
        enable_scheduling: config.actions.scheduling.enabled,
        enable_channel_send: config.actions.messaging.enabled,
        web_search_top_k: config.actions.web_search.default_top_k,
    };

    let mut dispatcher =
        cortex::actions::ActionDispatcher::with_memory_backend(action_config, action_backend);
    dispatcher.set_namespace("personal");

    // ── URL fetch backend ────────────────────────────────────────────────
    // Always wired when web search is enabled so user-provided links can
    // be fetched alongside the search query. Cheap, no config, no key.
    if config.actions.web_search.enabled {
        let res = &config.actions.resilience;
        match BasicUrlFetcher::new_with_metrics(res, Some(processor.metrics().clone())) {
            Ok(fetcher) => {
                tracing::info!("URL fetch backend configured");
                dispatcher = dispatcher.with_url_fetch_backend(Arc::new(fetcher));
            }
            Err(e) => tracing::warn!("URL fetch backend init failed: {e}"),
        }
    }

    // ── Web search backend ───────────────────────────────────────────────
    if config.actions.web_search.enabled {
        let ws = &config.actions.web_search;
        let timeout = ws.timeout_ms;
        let endpoint = ws.endpoint.trim();
        let res = &config.actions.resilience;
        let metrics = Some(processor.metrics().clone());

        let backend_result: Result<
            Option<Arc<dyn cortex::actions::WebSearchBackend>>,
            anyhow::Error,
        > = match ws.provider {
            brain_core::config::WebSearchProvider::DuckDuckGo => {
                DuckDuckGoSearchBackend::new_with_metrics(timeout, res, metrics.clone())
                    .map(|b| Some(Arc::new(b) as _))
            }
            brain_core::config::WebSearchProvider::Searxng => {
                let ep = if endpoint.is_empty() {
                    "http://localhost:8888"
                } else {
                    endpoint
                };
                SearxngSearchBackend::new_with_metrics(ep, timeout, res, metrics.clone())
                    .map(|b| Some(Arc::new(b) as _))
            }
            brain_core::config::WebSearchProvider::Tavily => {
                let api_key = ws.api_key.trim();
                if api_key.is_empty() {
                    tracing::warn!(
                            "actions.web_search.provider=tavily but api_key is empty; backend not configured"
                        );
                    Ok(None)
                } else {
                    let ep = if endpoint.is_empty() {
                        "https://api.tavily.com"
                    } else {
                        endpoint
                    };
                    TavilySearchBackend::new_with_metrics(
                        ep,
                        api_key,
                        timeout,
                        res,
                        metrics.clone(),
                    )
                    .map(|b| Some(Arc::new(b) as _))
                }
            }
            brain_core::config::WebSearchProvider::Custom => {
                if endpoint.is_empty() {
                    tracing::warn!(
                            "actions.web_search.provider=custom but endpoint is empty; backend not configured"
                        );
                    Ok(None)
                } else {
                    CustomSearchBackend::new_with_metrics(endpoint, timeout, res, metrics.clone())
                        .map(|b| Some(Arc::new(b) as _))
                }
            }
        };

        match backend_result {
            Ok(Some(backend)) => {
                tracing::info!(
                    provider = %serde_json::to_string(&ws.provider).unwrap_or_default().trim_matches('"'),
                    "Web search backend configured"
                );
                dispatcher = dispatcher.with_web_search_backend(backend);
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("Web search backend init failed: {e}"),
        }
    }

    // ── Scheduling backend ───────────────────────────────────────────────
    if config.actions.scheduling.enabled {
        let backend = DefaultSchedulingBackend {
            db: processor.episodic().pool().clone(),
            mode: config.actions.scheduling.mode.clone(),
        };
        dispatcher = dispatcher.with_scheduling_backend(Arc::new(backend));
    }

    // ── Messaging backend ────────────────────────────────────────────────
    if config.actions.messaging.enabled {
        if config.actions.messaging.channels.is_empty() {
            // Only warn when neither the legacy webhook channels nor the
            // newer preset-driven transports are configured — otherwise
            // delivery still works through `channel.transports[]`.
            if config.channel.transports.is_empty() {
                tracing::warn!(
                    "actions.messaging.enabled=true but no channel webhook mappings are configured"
                );
            }
        } else {
            let res = &config.actions.resilience;
            match WebhookMessageBackend::new_with_metrics(
                &config.actions.messaging.channels,
                config.actions.messaging.timeout_ms,
                res,
                Some(processor.metrics().clone()),
            ) {
                Ok(backend) => {
                    tracing::info!("Message backend configured");
                    dispatcher = dispatcher.with_message_backend(Arc::new(backend));
                }
                Err(e) => tracing::warn!("Message backend init failed: {e}"),
            }
        }
    }

    Ok(dispatcher)
}

/// Check if a Brain daemon is already running by probing its health endpoint.
///
/// Returns the base URL (e.g. `http://127.0.0.1:19789`) if the daemon is alive.
pub async fn detect_running_daemon(config: &brain_core::BrainConfig) -> Option<String> {
    let host = &config.adapters.http.host;
    let port = config.adapters.http.port;
    let base_url = format!("http://{host}:{port}");
    let health_url = format!("{base_url}/health");

    let client = reqwest::Client::builder()
        .timeout(brain_core::timeouts::HEALTH_CHECK)
        .build()
        .ok()?;

    match client.get(&health_url).send().await {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!(url = %base_url, "Detected running Brain daemon");
            Some(base_url)
        }
        Ok(resp) => {
            tracing::debug!(status = %resp.status(), "Daemon health check returned non-success");
            None
        }
        Err(e) => {
            tracing::debug!(error = %e, "Daemon health check failed");
            None
        }
    }
}

/// Require a running Brain daemon, returning its base URL or a clear error.
///
/// Retries a few times to handle the case where the daemon is still booting.
/// This is the canonical way for CLI commands to ensure they don't create
/// their own SignalProcessor (which would cause RuVector lock contention
/// and memory isolation).
pub async fn require_daemon(config: &brain_core::BrainConfig) -> anyhow::Result<String> {
    let max_attempts = 4;
    for attempt in 0..max_attempts {
        if let Some(url) = detect_running_daemon(config).await {
            return Ok(url);
        }
        if attempt < max_attempts - 1 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }

    let port = config.adapters.http.port;
    anyhow::bail!(
        "No running Brain daemon detected (expected at http://127.0.0.1:{port}).\n\
         Run `brain start` to wake the daemon first.\n\
         All CLI commands require a running daemon to ensure a single shared SignalProcessor."
    )
}

/// Proxy MCP stdio through a running daemon's MCP HTTP transport.
///
/// Reads JSON-RPC lines from stdin, forwards each as an HTTP POST to the
/// daemon's MCP endpoint, and writes the response to stdout. This ensures
/// that the daemon's single SignalProcessor handles all requests — no
/// ruvector lock contention, no memory isolation.
pub async fn proxy_mcp_stdio(
    mcp_url: &str,
    config: &brain_core::BrainConfig,
) -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let client = reqwest::Client::builder()
        .timeout(brain_core::timeouts::DAEMON_SETUP)
        .build()?;

    // Resolve API key for the x-api-key header.
    let api_key = std::env::var("BRAIN_API_KEY").unwrap_or_default();
    let api_key = if api_key.is_empty() {
        config
            .access
            .api_keys
            .first()
            .map(|k| k.key.clone())
            .unwrap_or_default()
    } else {
        api_key
    };

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // EOF
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Forward the raw JSON-RPC request to the daemon's MCP HTTP endpoint.
        let resp = client
            .post(mcp_url)
            .header("Content-Type", "application/json")
            .header("x-api-key", &api_key)
            .body(trimmed.to_string())
            .send()
            .await;

        match resp {
            Ok(r) => {
                // 204 No Content = notification ack — nothing to forward to the
                // stdio client (JSON-RPC spec: no response for notifications).
                if r.status() == reqwest::StatusCode::NO_CONTENT {
                    continue;
                }
                let body = r.text().await.unwrap_or_default();
                if !body.is_empty() {
                    stdout.write_all(body.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
            }
            Err(e) => {
                // Connection error — daemon may have stopped. Return a JSON-RPC error.
                let err_resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32603,
                        "message": format!("Daemon proxy error: {e}")
                    }
                });
                let json = serde_json::to_string(&err_resp)?;
                stdout.write_all(json.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal config producer that disables auto-discovery and provides
    // one manual subprocess delegate — keeps the test offline and tight.
    fn cfg_with_manual_delegate() -> brain_core::BrainConfig {
        let mut c = brain_core::BrainConfig::default();
        c.agents.auto_discovery = false;
        c.agents.delegates = vec![brain_core::config::AgentEntry {
            name: "hand-wired".to_string(),
            kind: "subprocess".to_string(),
            alias: None,
            binary: "/usr/bin/true".to_string(),
            args: vec![],
            workdir: None,
            prompt_via_stdin: true,
            tags: vec!["test".to_string()],
        }];
        c.agents.fallbacks = vec!["hand-wired".to_string(), "ghost".to_string()];
        c
    }

    #[tokio::test]
    async fn build_agent_registry_records_manual_entry_in_status() {
        let cfg = cfg_with_manual_delegate();
        let reg = build_agent_registry(&cfg).await.expect("registry builds");
        assert!(reg.contains("hand-wired"), "manual delegate not registered");
        match reg.agent_status("hand-wired") {
            Some(delegate::RegistryAgentStatus::Registered { source, .. }) => {
                assert_eq!(*source, delegate::AgentSource::Manual);
            }
            other => panic!("expected Manual Registered, got {other:?}"),
        }
        // Fallback "ghost" is bogus — known_agents should not gain a phantom
        // entry from the validator pass (validator only warns).
        assert!(reg.agent_status("ghost").is_none());
    }

    #[tokio::test]
    async fn build_agent_registry_empty_when_fully_disabled() {
        let mut cfg = brain_core::BrainConfig::default();
        cfg.agents.auto_discovery = false;
        let reg = build_agent_registry(&cfg).await.expect("registry builds");
        assert!(reg.is_empty());
        assert!(reg.known_agents().is_empty());
    }
}
