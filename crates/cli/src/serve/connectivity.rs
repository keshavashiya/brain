//! Connectivity probing — the writer side of [`brain::Connectivity`].
//!
//! The serve loop spawns one bounded probe task (see
//! `background::spawn_connectivity_probe`) that TCP-connects the derived
//! target set each round and folds the result into the kernel's
//! `Online / Degraded / Offline` view. This module owns the pieces that
//! loop composes:
//!
//! * [`probe_targets`] — derive the `host:port` set from the
//!   **already-configured remote LLM provider endpoints** (or the explicit
//!   `monitoring.connectivity.targets` override), so probing never adds an
//!   egress destination the user didn't already opt into. A fully-local
//!   install derives an empty set and spawns no loop.
//! * [`probe_round`] / [`state_for`] / [`detail_for`] — one round of
//!   reachability checks and its fold into a [`ConnectivityState`].
//! * [`advisory`] — the proactive-notification body for a transition.

use std::time::Duration;

use brain::ConnectivityState;

/// Derive the probe target set as `host:port` strings.
///
/// Explicit `monitoring.connectivity.targets` entries win verbatim. Otherwise
/// the set is the configured **remote** (non-loopback) LLM provider endpoints:
/// each `llm.providers[]` entry's `base_url` (or its preset's), falling back
/// to the legacy single-provider fields when `providers[]` is empty — the same
/// resolution order the provider factory uses. Loopback endpoints are skipped
/// (their reachability says nothing about the network), and duplicates
/// collapse so two providers behind one gateway probe it once.
pub(crate) fn probe_targets(config: &brain::BrainConfig) -> Vec<String> {
    let cfg = &config.monitoring.connectivity;
    if !cfg.targets.is_empty() {
        let mut out = Vec::new();
        for t in &cfg.targets {
            let t = t.trim();
            if !t.is_empty() && !out.iter().any(|o| o == t) {
                out.push(t.to_string());
            }
        }
        return out;
    }

    let mut urls: Vec<String> = Vec::new();
    if config.llm.providers.is_empty() {
        // Legacy single-provider shape — still honoured by the factory, so
        // honour it here too.
        #[allow(deprecated)]
        urls.push(resolve_base_url(&config.llm.provider, &config.llm.base_url));
    } else {
        for entry in &config.llm.providers {
            urls.push(resolve_base_url(&entry.kind, &entry.base_url));
        }
    }

    let mut out = Vec::new();
    for url in &urls {
        if url.is_empty() || brain::url_is_loopback(url) {
            continue;
        }
        if let Some(hp) = host_port(url) {
            if !out.iter().any(|o| o == &hp) {
                out.push(hp);
            }
        }
    }
    out
}

/// An entry's effective endpoint: explicit `base_url` wins, else the preset
/// registered under its `kind`/provider name, else nothing (e.g. `ollama`
/// with no `base_url` defaults to loopback — not a connectivity signal).
fn resolve_base_url(kind: &str, explicit: &str) -> String {
    if !explicit.is_empty() {
        explicit.to_string()
    } else {
        cortex::presets::resolve(kind)
            .map(|p| p.base_url.to_string())
            .unwrap_or_default()
    }
}

/// `host:port` of a URL's authority, defaulting the port from the scheme
/// (80 for explicit `http://`, 443 otherwise — provider endpoints are HTTPS
/// unless stated). Bracket-aware for IPv6 literals.
fn host_port(url: &str) -> Option<String> {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next()?;
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    if host_port.is_empty() {
        return None;
    }
    let has_port = if host_port.starts_with('[') {
        host_port
            .rsplit_once(']')
            .is_some_and(|(_, rest)| rest.starts_with(':'))
    } else {
        host_port.contains(':')
    };
    if has_port {
        Some(host_port.to_string())
    } else {
        let default_port = if url.starts_with("http://") { 80 } else { 443 };
        Some(format!("{host_port}:{default_port}"))
    }
}

/// One probe round: TCP-connect every target, return how many accepted
/// before the timeout. Sequential — the target set is small (one entry per
/// distinct provider gateway) and a dead network fails fast, so a round
/// comfortably fits inside the probe interval.
pub(crate) async fn probe_round(targets: &[String], timeout: Duration) -> usize {
    let mut reachable = 0;
    for target in targets {
        let attempt = tokio::time::timeout(timeout, tokio::net::TcpStream::connect(target)).await;
        if matches!(attempt, Ok(Ok(_))) {
            reachable += 1;
        }
    }
    reachable
}

/// Fold a round's tally into a state: everything up → `Online`, a strict
/// subset → `Degraded`, nothing → `Offline`.
pub(crate) fn state_for(reachable: usize, total: usize) -> ConnectivityState {
    if reachable == total {
        ConnectivityState::Online
    } else if reachable > 0 {
        ConnectivityState::Degraded
    } else {
        ConnectivityState::Offline
    }
}

/// Human-readable cause for a transition, e.g. `"2 of 3 endpoints
/// unreachable"`. Counts, not hostnames — the event is for orientation; the
/// per-endpoint story belongs to the service-health monitors.
pub(crate) fn detail_for(reachable: usize, total: usize) -> String {
    if reachable == total {
        "all endpoints reachable".to_string()
    } else {
        format!("{} of {total} endpoints unreachable", total - reachable)
    }
}

/// Body of the proactive notification for one connectivity transition.
/// Says what stopped working *and what still works* — the same honest
/// degradation the pipeline's offline web-search reply gives in-chat.
pub(crate) fn advisory(state: ConnectivityState, detail: &str) -> String {
    match state {
        ConnectivityState::Online => {
            "Network connectivity restored — remote models and web search are available again."
                .to_string()
        }
        ConnectivityState::Degraded => format!(
            "Network connectivity is degraded ({detail}). Some remote models and tools may \
             fail; local models and memory are unaffected."
        ),
        ConnectivityState::Offline => format!(
            "Network is unreachable ({detail}). Running fully local: web search and remote \
             models are unavailable, but local models and stored memory still work."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_port_defaults_the_port_from_the_scheme() {
        assert_eq!(
            host_port("https://api.openai.com/v1").as_deref(),
            Some("api.openai.com:443")
        );
        assert_eq!(
            host_port("http://gateway.lan/v1").as_deref(),
            Some("gateway.lan:80")
        );
        assert_eq!(
            host_port("https://host.example:8443/v1").as_deref(),
            Some("host.example:8443")
        );
        assert_eq!(
            host_port("https://[2001:db8::1]/v1").as_deref(),
            Some("[2001:db8::1]:443")
        );
        assert_eq!(
            host_port("https://[2001:db8::1]:9000/v1").as_deref(),
            Some("[2001:db8::1]:9000")
        );
    }

    #[test]
    fn fully_local_default_config_derives_no_targets() {
        let config = brain::BrainConfig::default();
        assert!(
            probe_targets(&config).is_empty(),
            "default (loopback ollama) install must not spawn a probe loop"
        );
    }

    #[test]
    fn targets_come_from_remote_providers_skipping_loopback_and_duplicates() {
        let mut config = brain::BrainConfig::default();
        config.llm.providers = vec![
            brain::config::ProviderEntry {
                name: "local".into(),
                kind: "ollama".into(),
                base_url: "http://127.0.0.1:11434".into(),
                api_key: String::new(),
                api_key_file: None,
                model: "qwen3".into(),
                preferred_models: vec![],
            },
            brain::config::ProviderEntry {
                name: "hosted".into(),
                kind: "openai_compat".into(),
                base_url: "https://api.example.com/v1".into(),
                api_key: String::new(),
                api_key_file: None,
                model: "gpt".into(),
                preferred_models: vec![],
            },
            brain::config::ProviderEntry {
                name: "hosted-twin".into(),
                kind: "openai_compat".into(),
                base_url: "https://api.example.com/v1".into(),
                api_key: String::new(),
                api_key_file: None,
                model: "gpt-mini".into(),
                preferred_models: vec![],
            },
            // Preset with no explicit base_url resolves through the registry.
            brain::config::ProviderEntry {
                name: "router".into(),
                kind: "openrouter".into(),
                base_url: String::new(),
                api_key: String::new(),
                api_key_file: None,
                model: "auto".into(),
                preferred_models: vec![],
            },
        ];
        assert_eq!(
            probe_targets(&config),
            vec![
                "api.example.com:443".to_string(),
                "openrouter.ai:443".to_string()
            ]
        );
    }

    #[test]
    fn explicit_targets_override_derivation() {
        let mut config = brain::BrainConfig::default();
        config.llm.providers = vec![brain::config::ProviderEntry {
            name: "hosted".into(),
            kind: "openai_compat".into(),
            base_url: "https://api.example.com/v1".into(),
            api_key: String::new(),
            api_key_file: None,
            model: "gpt".into(),
            preferred_models: vec![],
        }];
        config.monitoring.connectivity.targets =
            vec!["probe.example:443".into(), " probe.example:443 ".into()];
        assert_eq!(
            probe_targets(&config),
            vec!["probe.example:443".to_string()]
        );
    }

    #[test]
    fn state_folds_all_some_none() {
        assert_eq!(state_for(3, 3), ConnectivityState::Online);
        assert_eq!(state_for(1, 3), ConnectivityState::Degraded);
        assert_eq!(state_for(0, 3), ConnectivityState::Offline);
    }

    #[test]
    fn detail_counts_the_unreachable() {
        assert_eq!(detail_for(3, 3), "all endpoints reachable");
        assert_eq!(detail_for(1, 3), "2 of 3 endpoints unreachable");
    }
}
