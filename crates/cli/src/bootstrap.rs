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
    processor = wire_safety_infrastructure(processor, config)?;

    Ok(processor)
}

/// Wire Phase 1 safety infrastructure into the processor.
///
/// Components: audit trail, confirmation engine, cost budget, sandbox executor.
/// All share the same SQLite pool as the episodic store for simplicity.
/// The credential vault is NOT wired here — it requires passphrase input
/// and is wired on demand (e.g. `brain vault` / `brain auth` commands).
fn wire_safety_infrastructure(
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

    // Confirmation engine — always wired
    let confirm_engine = confirm::SqliteConfirmationEngine::new(db.clone());
    confirm_engine
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Confirmation engine table init failed: {e}"))?;
    let confirm_engine: Arc<dyn confirm::ConfirmationEngine> = Arc::new(confirm_engine);
    tracing::info!("Confirmation engine wired");

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
    let sandbox = sandbox::IsolatedSandbox::new(
        config.security.exec_allowlist.clone(),
        exec_timeout,
    )
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
    // registered specialist agents. When `agents.delegates` is empty,
    // `StepAction::Implement` will fail with a clear error — that's the
    // desired behaviour until an agent is configured.
    let agent_registry = build_agent_registry(config)?;
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
        .with_delegation_policy(escalation_policy);
    let processor = processor
        .with_orchestrator(Arc::new(orchestrator))
        .with_agent_registry(agent_registry_arc);
    tracing::info!("Task orchestrator wired");

    // ── Channel intelligence — always wired ─────────────────────────────
    let pref_store = channel::SqlitePreferenceStore::new(db.clone());
    pref_store
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Channel preference table init failed: {e}"))?;
    let preferences: Arc<dyn channel::ChannelPreferenceStore> = Arc::new(pref_store);
    let router: Arc<dyn channel::ChannelRouter> =
        Arc::new(channel::DefaultChannelRouter::new(preferences.clone()));
    let correlator = Arc::new(channel::ConfirmationCorrelator::new(confirm_engine));
    let processor = processor
        .with_channel_preferences(preferences)
        .with_channel_router(router)
        .with_confirmation_correlator(correlator);
    tracing::info!("Channel intelligence wired (router + preferences + correlator)");

    Ok(processor)
}

/// Build the agent delegation registry from `config.agents.delegates`.
///
/// Every entry maps to a concrete [`delegate::AgentDelegate`] implementation
/// based on `kind`. Unknown kinds are warned about and skipped — missing
/// required fields (e.g. `binary` for a `subprocess` entry) are fatal so
/// bad config surfaces at boot instead of mid-delegation.
fn build_agent_registry(
    config: &brain_core::BrainConfig,
) -> anyhow::Result<delegate::AgentRegistry> {
    let mut registry = delegate::AgentRegistry::new();

    for entry in &config.agents.delegates {
        let capabilities = delegate::AgentCapabilities {
            tags: entry.tags.clone(),
            languages: Vec::new(),
            max_concurrency: 1,
            needs_network: true,
        };

        let workdir = entry.workdir.as_ref().map(std::path::PathBuf::from);

        match entry.kind.as_str() {
            "claude_code" => {
                let binary = if entry.binary.is_empty() {
                    "claude".to_string()
                } else {
                    entry.binary.clone()
                };
                let cfg = delegate::ClaudeCodeConfig {
                    name: entry.name.clone(),
                    binary,
                    extra_args: entry.args.clone(),
                    workdir,
                    capabilities,
                };
                let d: Arc<dyn delegate::AgentDelegate> =
                    Arc::new(delegate::ClaudeCodeDelegate::new(cfg));
                registry.register(d);
            }
            "subprocess" => {
                if entry.binary.is_empty() {
                    anyhow::bail!(
                        "agents.delegates[{}]: `subprocess` kind requires a non-empty `binary`",
                        entry.name
                    );
                }
                let mut sub_cfg =
                    delegate::SubprocessAgentConfig::new(&entry.name, &entry.binary)
                        .with_args(entry.args.clone())
                        .with_capabilities(capabilities)
                        .with_prompt_via_stdin(entry.prompt_via_stdin);
                if let Some(dir) = workdir {
                    sub_cfg = sub_cfg.with_workdir(dir);
                }
                let d: Arc<dyn delegate::AgentDelegate> =
                    Arc::new(delegate::SubprocessAgentDelegate::new(sub_cfg));
                registry.register(d);
            }
            other => {
                tracing::warn!(
                    kind = %other,
                    name = %entry.name,
                    "Unknown agent kind — skipping"
                );
                continue;
            }
        }

        if let Some(alias) = &entry.alias {
            registry.alias(alias.clone(), entry.name.clone());
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
            tracing::warn!(
                "actions.messaging.enabled=true but no channel webhook mappings are configured"
            );
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
