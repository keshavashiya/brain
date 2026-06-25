//! Capability discovery nudge.
//!
//! On a slow cadence the daemon asks the signal layer for capabilities the user
//! has never used (authored, user-facing faculties with no recorded fitness),
//! and surfaces one as a gentle "did you know Brain can…" proactive nudge — so a
//! capability the user never knew about doesn't stay invisible. Each capability
//! is suggested at most once per process; the cadence and bookkeeping live here,
//! the manifest×fitness query lives in `signal::discovery`.
//!
//! Gated like every other nudge by the runtime proactivity toggle, and delivered
//! at priority 1 (a habit-style suggestion, not a health alert) through the same
//! notification router.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Spawn the discovery loop. A no-op when discovery is disabled. The first scan
/// lands one interval after boot, not at startup, so a fresh daemon doesn't
/// greet the user with a suggestion before they've done anything.
pub(super) fn spawn_capability_discovery(
    processor: Arc<signal::SignalProcessor>,
    cfg: brain::config::DiscoveryConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    if !cfg.enabled {
        return;
    }
    let runtime_toggle = processor.proactivity_enabled();
    let p = processor.clone();
    set.spawn(async move {
        // Suggested capabilities, so each is surfaced at most once this run.
        let mut suggested: HashSet<String> = HashSet::new();
        let period = tokio::time::Duration::from_secs(cfg.interval_hours.max(1) * 3600);
        let mut ticker = tokio::time::interval(period);
        // Skip the immediate first tick — the first nudge lands one interval in.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Respect the live proactivity toggle, the same gate the habit and
            // open-loop nudges honour.
            if !runtime_toggle.load(Ordering::SeqCst) {
                continue;
            }
            // Surface the first untried capability we haven't already suggested.
            let Some(cap) = p
                .untried_capabilities()
                .await
                .into_iter()
                .find(|c| !suggested.contains(&c.tool_id))
            else {
                continue;
            };
            suggested.insert(cap.tool_id.clone());
            tracing::info!(
                tool_id = %cap.tool_id,
                "Capability discovery: suggesting an unused capability"
            );
            if let Some(router) = p.notification_router() {
                router
                    .deliver(signal::notification::ProactiveNotification {
                        content: discovery_message(&cap),
                        triggered_by: format!("capability_discovery:{}", cap.tool_id),
                        priority: 1,
                        agent: None,
                    })
                    .await;
            }
        }
    });
    tracing::info!(
        interval_hours = cfg.interval_hours,
        "Capability discovery scheduled"
    );
}

/// The nudge body — names the capability, why it's useful, and (when authored)
/// how to invoke it.
fn discovery_message(cap: &signal::discovery::UntriedCapability) -> String {
    let mut msg = format!(
        "💡 You have a capability you haven't used yet — `{}`: {}.",
        cap.verb, cap.when_to_use
    );
    if let Some(example) = &cap.example {
        msg.push_str(&format!(" For example: {example}"));
    }
    msg
}

// ── MCP server discovery ─────────────────────────────────────────────────────

/// Spawn the MCP-config-scan loop. On a slow cadence it reads *other* MCP
/// clients' config files on this machine (Claude Desktop, Cursor, Windsurf, …),
/// finds servers Brain hasn't mounted, and proposes one as a copy-paste
/// `/mcp-mount` nudge. The scan is read-only and local; mounting stays a
/// consented user action with its own egress scopes.
pub(super) fn spawn_mcp_discovery(
    processor: Arc<signal::SignalProcessor>,
    cfg: brain::config::DiscoveryConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let runtime_toggle = processor.proactivity_enabled();
    let p = processor.clone();
    set.spawn(async move {
        let mut suggested: HashSet<String> = HashSet::new();
        let period = tokio::time::Duration::from_secs(cfg.interval_hours.max(1) * 3600);
        let mut ticker = tokio::time::interval(period);
        ticker.tick().await; // first scan lands one interval after boot
        loop {
            ticker.tick().await;
            if !runtime_toggle.load(Ordering::SeqCst) {
                continue;
            }
            let discovered = scan_external_mcp_configs();
            if discovered.is_empty() {
                continue;
            }
            // Skip servers Brain already runs (by name) and ones already
            // suggested this run.
            let mounted: HashSet<String> = match p.mcp_host() {
                Some(host) => host
                    .list_servers()
                    .await
                    .into_iter()
                    .map(|s| s.name)
                    .collect(),
                None => HashSet::new(),
            };
            let Some((tool, server)) = select_new_servers(discovered, |name| {
                mounted.contains(name) || suggested.contains(name)
            })
            .into_iter()
            .next() else {
                continue;
            };
            suggested.insert(server.name.clone());
            tracing::info!(
                server = %server.name,
                tool = %tool,
                "MCP discovery: proposing an unmounted server from another tool's config"
            );
            if let Some(router) = p.notification_router() {
                router
                    .deliver(signal::notification::ProactiveNotification {
                        content: mcp_discovery_message(&tool, &server),
                        triggered_by: format!("mcp_discovery:{}", server.name),
                        priority: 1,
                        agent: None,
                    })
                    .await;
            }
        }
    });
    tracing::info!(
        interval_hours = cfg.interval_hours,
        "MCP server discovery scheduled"
    );
}

/// Read every known external MCP config file present on this machine and return
/// the servers each declares, tagged with the tool they came from. A missing or
/// unreadable file is simply skipped — discovery is best-effort.
fn scan_external_mcp_configs() -> Vec<(String, mcphost::DiscoveredServer)> {
    let mut out = Vec::new();
    for (tool, path) in known_mcp_config_paths() {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for server in mcphost::parse_mcp_servers(&text) {
            out.push((tool.to_string(), server));
        }
    }
    out
}

/// Well-known MCP client config locations for this platform. Each entry is the
/// tool's display name and the path to its config file; only those that exist
/// are read.
fn known_mcp_config_paths() -> Vec<(&'static str, std::path::PathBuf)> {
    let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
        return Vec::new();
    };
    let join = |rel: &str| home.join(rel);
    #[cfg(target_os = "macos")]
    let entries = vec![
        (
            "Claude Desktop",
            join("Library/Application Support/Claude/claude_desktop_config.json"),
        ),
        ("Cursor", join(".cursor/mcp.json")),
        ("Windsurf", join(".codeium/windsurf/mcp_config.json")),
    ];
    #[cfg(not(target_os = "macos"))]
    let entries = vec![
        (
            "Claude Desktop",
            join(".config/Claude/claude_desktop_config.json"),
        ),
        ("Cursor", join(".cursor/mcp.json")),
        ("Windsurf", join(".codeium/windsurf/mcp_config.json")),
    ];
    entries
}

/// Filter discovered servers to the ones Brain doesn't already know — pure over
/// the `is_known` predicate (mounted names ∪ already-suggested) so the selection
/// rule is unit-testable without a host or filesystem.
fn select_new_servers(
    discovered: Vec<(String, mcphost::DiscoveredServer)>,
    is_known: impl Fn(&str) -> bool,
) -> Vec<(String, mcphost::DiscoveredServer)> {
    discovered
        .into_iter()
        .filter(|(_, s)| !is_known(&s.name))
        .collect()
}

/// The MCP-discovery nudge body: what was found, where, and the consented
/// command to adopt it.
fn mcp_discovery_message(tool: &str, server: &mcphost::DiscoveredServer) -> String {
    format!(
        "🔌 {tool} has an MCP server `{}` that Brain hasn't mounted. \
         To add it (review its egress scopes first): {}",
        server.name,
        server.mount_hint()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcphost::{DiscoveredServer, DiscoveredTransport};
    use signal::discovery::UntriedCapability;

    fn server(name: &str) -> DiscoveredServer {
        DiscoveredServer {
            name: name.to_string(),
            transport: DiscoveredTransport::Stdio {
                command: "run".into(),
                args: vec![],
            },
        }
    }

    #[test]
    fn select_new_skips_mounted_and_suggested() {
        let discovered = vec![
            ("Cursor".to_string(), server("filesystem")),
            ("Cursor".to_string(), server("github")),
            ("Claude Desktop".to_string(), server("already-mounted")),
        ];
        // `already-mounted` is known; the other two are new.
        let new = select_new_servers(discovered, |name| name == "already-mounted");
        assert_eq!(new.len(), 2);
        assert!(new.iter().all(|(_, s)| s.name != "already-mounted"));
    }

    #[test]
    fn mcp_message_names_tool_and_carries_mount_command() {
        let s = DiscoveredServer {
            name: "github".into(),
            transport: DiscoveredTransport::Remote {
                url: "https://mcp/sse".into(),
            },
        };
        let msg = mcp_discovery_message("Cursor", &s);
        assert!(msg.contains("Cursor"), "{msg}");
        assert!(msg.contains("github"), "{msg}");
        assert!(
            msg.contains("/mcp-mount github streamable_http https://mcp/sse"),
            "{msg}"
        );
    }

    #[test]
    fn message_names_capability_and_reason() {
        let cap = UntriedCapability {
            tool_id: "net.check".into(),
            verb: "net.check".into(),
            when_to_use: "test whether a host is reachable".into(),
            example: Some("is github.com reachable?".into()),
        };
        let msg = discovery_message(&cap);
        assert!(msg.contains("net.check"), "{msg}");
        assert!(msg.contains("test whether a host is reachable"), "{msg}");
        assert!(msg.contains("is github.com reachable?"), "{msg}");
    }

    #[test]
    fn message_without_example_omits_the_example_clause() {
        let cap = UntriedCapability {
            tool_id: "x.y".into(),
            verb: "x.y".into(),
            when_to_use: "do a thing".into(),
            example: None,
        };
        let msg = discovery_message(&cap);
        assert!(msg.contains("x.y") && msg.contains("do a thing"));
        assert!(!msg.contains("For example"), "{msg}");
    }
}
