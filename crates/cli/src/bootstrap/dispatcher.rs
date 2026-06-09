//! Action dispatcher construction: the memory backend plus the optional
//! URL-fetch, web-search, scheduling, and messaging backends gated by config.

use std::sync::Arc;

use crate::encryption::resolve_llm_api_key;
use backends::*;

/// Build the action dispatcher with all configured backends.
pub(super) fn build_action_dispatcher(
    config: &brain::BrainConfig,
    processor: &signal::SignalProcessor,
) -> anyhow::Result<cortex::actions::ActionDispatcher> {
    let embedding_dim = processor.embedding_dim();
    let llm_api_key = resolve_llm_api_key(config)?;
    // `Embedder::from_config` still keys off `llm.provider`. The field is
    // #[deprecated] (Issue 40) but load-bearing here until embedder
    // selection learns to read `providers[]`; the explicit read is
    // suppressed at the call site rather than forced through a wrapper.
    let embedder = {
        #[allow(deprecated)]
        let embed_provider = config.llm.provider.clone();
        #[allow(deprecated)]
        let embed_base = config.llm.base_url.clone();
        Arc::new(tokio::sync::Mutex::new(
            hippocampus::Embedder::from_config(
                &embed_provider,
                &embed_base,
                &config.embedding.model,
                &llm_api_key,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create embedding provider: {e}"))?,
        ))
    };

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
            brain::config::WebSearchProvider::DuckDuckGo => {
                DuckDuckGoSearchBackend::new_with_metrics(timeout, res, metrics.clone())
                    .map(|b| Some(Arc::new(b) as _))
                    .map_err(anyhow::Error::from)
            }
            brain::config::WebSearchProvider::Searxng => {
                let ep = if endpoint.is_empty() {
                    "http://localhost:8888"
                } else {
                    endpoint
                };
                SearxngSearchBackend::new_with_metrics(ep, timeout, res, metrics.clone())
                    .map(|b| Some(Arc::new(b) as _))
                    .map_err(anyhow::Error::from)
            }
            brain::config::WebSearchProvider::Tavily => {
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
                    .map_err(anyhow::Error::from)
                }
            }
            brain::config::WebSearchProvider::Custom => {
                if endpoint.is_empty() {
                    tracing::warn!(
                            "actions.web_search.provider=custom but endpoint is empty; backend not configured"
                        );
                    Ok(None)
                } else {
                    CustomSearchBackend::new_with_metrics(endpoint, timeout, res, metrics.clone())
                        .map(|b| Some(Arc::new(b) as _))
                        .map_err(anyhow::Error::from)
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

    // ── Network diagnostics backend ──────────────────────────────────────
    // Always wired: read-only probes (check/trace/cert) with no config gate
    // and no API key. Pure std/tokio + rustls; egress consent is enforced at
    // the External tier when dispatched.
    dispatcher = dispatcher.with_net_diagnostics_backend(Arc::new(NetDiagnostics));

    // ── Security audit backend ───────────────────────────────────────────
    // Always wired: a pure, offline audit over a snapshot of the loaded
    // config. Read-only (Read tier), no network, no key.
    dispatcher = dispatcher
        .with_security_audit_backend(Arc::new(ConfigSecurityAuditor::new(config.clone())));

    Ok(dispatcher)
}
