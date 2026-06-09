//! Register the kernel's *native* capabilities into the shared
//! [`intent::ToolRegistry`].
//!
//! Before this, only mounted MCP servers populated the registry (the MCP
//! host auto-registers on `mount`). The kernel's own built-in tools —
//! the action-dispatcher backends (memory, web, scheduling, messaging)
//! and the terminal bridge — were invisible to the one manifest that
//! feeds both the internal reasoner (the SOUL capability digest) and
//! external clients (`tools/list`). This module closes that gap: at boot
//! we seed the registry with a [`intent::ToolDescriptor`] for every
//! native capability that is actually wired this run, enriched with
//! reasoner-facing [`intent::ToolUsage`] guidance.
//!
//! **Awareness, not permission.** Registering a descriptor only makes the
//! capability *describable*; execution still flows through the
//! consent/audit/breaker path. The `usage.tier` mirrors the verb's
//! conservative [`intent::verbs`] tier hint so a reader sees the gate it
//! will hit.

use std::sync::Arc;

use intent::{
    verbs, BackendId, ToolAnnotations, ToolDescriptor, ToolRegistry, ToolSource, ToolUsage, Verb,
};

/// Which native capabilities are live this boot. Mirrors the same config
/// flags `build_action_dispatcher` keys off, plus the always-on faculties
/// (memory, filesystem read, terminal bridge) that bootstrap wires
/// unconditionally.
#[derive(Debug, Clone, Copy)]
pub struct WiredNatives {
    /// Semantic memory store/delete — always wired.
    pub memory: bool,
    /// Outbound HTTP (web search + URL fetch) — `actions.web_search.enabled`.
    pub web: bool,
    /// Scheduled-intent create/cancel — `actions.scheduling.enabled`.
    pub scheduling: bool,
    /// Outbound messaging via the channel dispatcher — `actions.messaging.enabled`.
    pub messaging: bool,
    /// Terminal bridge + sandboxed one-shot exec — always wired.
    pub terminal: bool,
    /// Path-scoped filesystem read — always wired.
    pub fs_read: bool,
    /// Read-only network diagnostics (`net.check`/`trace`/`cert`) — always
    /// wired; they neither search nor fetch, so they don't track web_search.
    pub net_diag: bool,
    /// Read-only security-posture audit (`security.audit`) — always wired;
    /// pure config inspection, no I/O.
    pub security_audit: bool,
    /// Read-only log pattern analysis (`logs.analyze`) — always wired; reads
    /// the daemon's own (or the OS) log offline.
    pub logs: bool,
    /// System-baseline capture/diff/list (`baseline.*`) — always wired; local
    /// snapshot files, no network.
    pub baseline: bool,
}

impl WiredNatives {
    /// Derive the live set from the runtime config. `memory`, `terminal`,
    /// and `fs_read` are unconditional in `build_processor`; the rest
    /// track the `actions.*` toggles `build_action_dispatcher` reads.
    pub fn from_config(config: &brain::BrainConfig) -> Self {
        Self {
            memory: true,
            web: config.actions.web_search.enabled,
            scheduling: config.actions.scheduling.enabled,
            messaging: config.actions.messaging.enabled,
            terminal: true,
            fs_read: true,
            net_diag: true,
            security_audit: true,
            logs: true,
            baseline: true,
        }
    }
}

/// Build a native [`ToolDescriptor`] for `(ns, action)`, pulling the
/// description + tier from the canonical [`verbs`] vocabulary and layering
/// the supplied usage guidance on top. The `tier` is filled from the verb
/// hint when the caller leaves it unset.
fn native(
    ns: &str,
    action: &str,
    source: ToolSource,
    annotations: ToolAnnotations,
    mut usage: ToolUsage,
) -> ToolDescriptor {
    let spec = verbs::lookup(ns, action);
    let description = spec.map(|s| s.summary.to_string()).unwrap_or_default();
    if usage.tier.is_none() {
        usage.tier = spec.map(|s| s.tier_hint.as_str().to_string());
    }
    ToolDescriptor {
        // `native:` prefix mirrors the `mcp:{server}:{tool}` id shape so
        // ids stay source-discriminable at a glance.
        tool_id: format!("native:{ns}.{action}"),
        source,
        verb: Verb::new(ns, action),
        description,
        input_schema: serde_json::json!({ "type": "object" }),
        output_schema: None,
        capabilities: vec![format!("{ns}.{action}")],
        annotations,
        usage,
        embedding: None,
    }
}

fn read_only() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: true,
        destructive_hint: false,
        idempotent_hint: true,
    }
}

fn destructive() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: false,
        destructive_hint: true,
        idempotent_hint: false,
    }
}

fn usage(
    when_to_use: &str,
    when_not_to: &str,
    preconditions: &[&str],
    cost: &str,
    example: &str,
) -> ToolUsage {
    ToolUsage {
        when_to_use: Some(when_to_use.to_string()),
        when_not_to: Some(when_not_to.to_string()),
        preconditions: preconditions.iter().map(|s| s.to_string()).collect(),
        cost: Some(cost.to_string()),
        example: Some(example.to_string()),
        tier: None, // filled from the verb hint in `native`
    }
}

/// Assemble every native descriptor that should be registered given the
/// live wiring. Pure (no I/O) so it can be unit-tested.
pub fn native_descriptors(w: WiredNatives) -> Vec<ToolDescriptor> {
    let mem = || ToolSource::NativeBackend {
        backend: BackendId::new("memory"),
    };
    let net = || ToolSource::NativeBackend {
        backend: BackendId::new("net"),
    };
    let sched = || ToolSource::NativeBackend {
        backend: BackendId::new("scheduling"),
    };
    let msg = || ToolSource::NativeBackend {
        backend: BackendId::new("messaging"),
    };
    let fs = || ToolSource::NativeBackend {
        backend: BackendId::new("fs"),
    };
    let security = || ToolSource::NativeBackend {
        backend: BackendId::new("security"),
    };
    let logs = || ToolSource::NativeBackend {
        backend: BackendId::new("logs"),
    };
    let baseline = || ToolSource::NativeBackend {
        backend: BackendId::new("baseline"),
    };

    let mut out = Vec::new();

    if w.memory {
        out.push(native(
            "memory",
            "store",
            mem(),
            ToolAnnotations::default(),
            usage(
                "The user states a durable fact about themselves, their world, projects, or preferences that should survive the session.",
                "Transient chit-chat, or content already captured as an episodic turn.",
                &["A subject-predicate-object triple can be extracted from the statement."],
                "free / local SQLite + embedding",
                "\"Remember that my deploy script lives in ops/deploy.sh\"",
            ),
        ));
        out.push(native(
            "memory",
            "delete",
            mem(),
            destructive(),
            usage(
                "The user asks to forget or correct a previously stored fact.",
                "When unsure which facts match — deletion is irreversible.",
                &["A matching subject/predicate is known."],
                "free / local",
                "\"Forget what I said about the old API key\"",
            ),
        ));
    }

    if w.web {
        out.push(native(
            "net",
            "http",
            net(),
            read_only(),
            usage(
                "The answer needs fresh, external, or post-training-cutoff information, or the user references a URL to read.",
                "The answer is in memory, the conversation, or general knowledge.",
                &["actions.web_search.enabled = true", "Network egress is permitted."],
                "network call (latency + possible API quota)",
                "\"What's the latest release of ripgrep?\"",
            ),
        ));
    }

    if w.scheduling {
        out.push(native(
            "schedule",
            "create",
            sched(),
            ToolAnnotations::default(),
            usage(
                "The user wants something to happen later or on a recurring cadence.",
                "For one-shot actions to run right now.",
                &["actions.scheduling.enabled = true"],
                "free / local row",
                "\"Remind me to rotate the certs every 90 days\"",
            ),
        ));
        out.push(native(
            "schedule",
            "cancel",
            sched(),
            ToolAnnotations::default(),
            usage(
                "The user wants to stop a previously scheduled item.",
                "When the schedule id / description is unknown.",
                &["actions.scheduling.enabled = true"],
                "free / local row",
                "\"Cancel the nightly backup reminder\"",
            ),
        ));
    }

    if w.messaging {
        out.push(native(
            "notify",
            "send",
            msg(),
            ToolAnnotations::default(),
            usage(
                "The user asks to send a message/notification out through a configured channel.",
                "For replying inside the current conversation — just answer.",
                &[
                    "actions.messaging.enabled = true",
                    "A channel transport is configured.",
                ],
                "network call (external delivery)",
                "\"Ping the ops webhook when the job finishes\"",
            ),
        ));
    }

    if w.terminal {
        out.push(native(
            "terminal",
            "open",
            ToolSource::Terminal,
            ToolAnnotations::default(),
            usage(
                "A task needs an interactive, stateful shell session (multiple commands sharing cwd/env).",
                "For a single one-shot command — prefer shell.exec.",
                &["Terminal bridge wired."],
                "spawns a PTY process",
                "\"Open a shell in the repo so we can poke around\"",
            ),
        ));
        out.push(native(
            "terminal",
            "close",
            ToolSource::Terminal,
            ToolAnnotations::default(),
            usage(
                "Tear down a PTY session opened earlier.",
                "When no session id is known.",
                &["An open session id exists."],
                "free",
                "\"Close that shell session\"",
            ),
        ));
        out.push(native(
            "shell",
            "exec",
            ToolSource::Terminal,
            ToolAnnotations::default(),
            usage(
                "Run a single sandboxed command and capture its output.",
                "For multi-step interactive work — use terminal.open.",
                &["The command is on security.exec_allowlist."],
                "spawns a sandboxed subprocess",
                "\"Run `git status` in the project\"",
            ),
        ));
    }

    if w.fs_read {
        out.push(native(
            "fs",
            "read",
            fs(),
            read_only(),
            usage(
                "Read a file or directory the user referenced, to ground the answer in what is actually on disk.",
                "For paths outside security.allowed_paths, or binary/huge files.",
                &["Path is inside security.allowed_paths."],
                "free / local read",
                "\"What's in ./README.md?\"",
            ),
        ));
    }

    if w.net_diag {
        out.push(native(
            "net",
            "check",
            net(),
            read_only(),
            usage(
                "The user asks whether a host/endpoint is reachable, or to diagnose a connectivity problem.",
                "For fetching a page's contents — use net.http (web search/fetch).",
                &["Network egress is permitted."],
                "network call (DNS + a single TCP connect)",
                "\"Can you reach api.github.com:443?\"",
            ),
        ));
        out.push(native(
            "net",
            "trace",
            net(),
            read_only(),
            usage(
                "The user wants to see the network path/hops to a host, e.g. to locate where traffic stalls.",
                "When only reachability matters — net.check is faster.",
                &["Network egress is permitted.", "`traceroute` is available (Unix)."],
                "network call (spawns a bounded traceroute child process)",
                "\"Trace the route to example.com\"",
            ),
        ));
        out.push(native(
            "net",
            "cert",
            net(),
            read_only(),
            usage(
                "The user asks about a site's TLS certificate — expiry, issuer, or SANs.",
                "For plain reachability — use net.check.",
                &["Network egress is permitted."],
                "network call (a single TLS handshake)",
                "\"When does the cert for example.com expire?\"",
            ),
        ));
    }

    if w.security_audit {
        out.push(native(
            "security",
            "audit",
            security(),
            read_only(),
            usage(
                "The user asks about the security posture, hardening, exposure, or whether the configuration is safe.",
                "For changing settings — this only reports; it never edits config.",
                &["None — offline config inspection."],
                "free / local config inspection",
                "\"Audit my security setup\" / \"Is anything exposed?\"",
            ),
        ));
    }

    if w.logs {
        out.push(native(
            "logs",
            "analyze",
            logs(),
            read_only(),
            usage(
                "The user asks what's been going wrong, to scan recent logs, or to surface recurring errors/warnings.",
                "For reading a specific file's contents — use fs.read.",
                &["The daemon has written logs (or the OS log is readable)."],
                "free / local log read (offline; optional LLM narration)",
                "\"Any recurring errors in the logs lately?\"",
            ),
        ));
    }

    if w.baseline {
        out.push(native(
            "baseline",
            "capture",
            baseline(),
            ToolAnnotations::default(),
            usage(
                "The user wants to record the current system state so future drift can be detected.",
                "For comparing against an existing snapshot — use baseline.diff.",
                &["None — offline config + capability snapshot."],
                "free / local snapshot file",
                "\"Capture a baseline of the current setup\"",
            ),
        ));
        out.push(native(
            "baseline",
            "diff",
            baseline(),
            read_only(),
            usage(
                "The user asks what has changed since a baseline, or to detect configuration drift.",
                "When no baseline has been captured yet — capture one first.",
                &["At least one baseline has been captured."],
                "free / local snapshot comparison",
                "\"What's drifted since my last baseline?\"",
            ),
        ));
        out.push(native(
            "baseline",
            "list",
            baseline(),
            read_only(),
            usage(
                "The user wants to see which baseline snapshots have been stored.",
                "For the contents of a specific snapshot — use baseline.diff.",
                &["None."],
                "free / local directory listing",
                "\"List the baselines I've captured\"",
            ),
        ));
    }

    out
}

/// Flatten the live native descriptors into the [`backends::CapabilitySummary`]
/// inventory the baseline backend snapshots. The descriptor list is the single
/// source of which capabilities are live this run; passing it in keeps the
/// baseline core a pure function of its inputs (no reach-back into the binary).
pub fn capability_inventory(config: &brain::BrainConfig) -> Vec<backends::CapabilitySummary> {
    native_descriptors(WiredNatives::from_config(config))
        .into_iter()
        .map(|d| backends::CapabilitySummary {
            namespace: d.verb.namespace,
            action: d.verb.action,
            tier: d.usage.tier.unwrap_or_else(|| "unknown".to_string()),
        })
        .collect()
}

/// Register every live native capability into `registry`. Idempotent at
/// the registry level (re-registering the same `tool_id` overwrites).
/// Failures are logged, not fatal — a missing native descriptor degrades
/// the manifest, it doesn't break boot.
pub async fn register_native_capabilities(
    registry: &Arc<dyn ToolRegistry>,
    config: &brain::BrainConfig,
) {
    let descriptors = native_descriptors(WiredNatives::from_config(config));
    let count = descriptors.len();
    for d in descriptors {
        let id = d.tool_id.clone();
        if let Err(e) = registry.register(d).await {
            tracing::warn!(tool_id = %id, error = %e, "native capability registration failed");
        }
    }
    tracing::info!(
        count,
        "Native capabilities registered into the tool registry"
    );
}

/// `brain capabilities` — print the live capability manifest from the
/// running daemon. Drives the `Intent::ListCapabilities` inspection path
/// over the same one-shot WebSocket client `brain chat "<msg>"` uses, so
/// the listing reflects runtime state (e.g. MCP servers mounted since
/// boot), not just this process's static view.
pub async fn cmd_capabilities(config: &brain::BrainConfig) -> anyhow::Result<()> {
    crate::chat::command_over_chat(config, "/capabilities").await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_wired() -> WiredNatives {
        WiredNatives {
            memory: true,
            web: true,
            scheduling: true,
            messaging: true,
            terminal: true,
            fs_read: true,
            net_diag: true,
            security_audit: true,
            logs: true,
            baseline: true,
        }
    }

    #[test]
    fn descriptors_carry_verb_description_and_tier() {
        let ds = native_descriptors(all_wired());
        let store = ds
            .iter()
            .find(|d| d.verb == Verb::new("memory", "store"))
            .expect("memory.store present");
        // Description comes from the canonical verb vocabulary.
        assert_eq!(
            store.description,
            "Store a subject-predicate-object fact in semantic memory."
        );
        // Tier is stamped from the verb hint.
        assert_eq!(store.usage.tier.as_deref(), Some("write"));
        assert!(store.usage.when_to_use.is_some());
        assert_eq!(store.tool_id, "native:memory.store");
    }

    #[test]
    fn destructive_verb_is_flagged_and_tiered() {
        let ds = native_descriptors(all_wired());
        let del = ds
            .iter()
            .find(|d| d.verb == Verb::new("memory", "delete"))
            .unwrap();
        assert!(del.annotations.destructive_hint);
        assert_eq!(del.usage.tier.as_deref(), Some("destructive"));
    }

    #[test]
    fn disabled_backends_are_omitted() {
        let w = WiredNatives {
            memory: true,
            web: false,
            scheduling: false,
            messaging: false,
            terminal: true,
            fs_read: true,
            net_diag: true,
            security_audit: true,
            logs: true,
            baseline: true,
        };
        let ds = native_descriptors(w);
        // The egress verb (net.http) is gated by web search; diagnostics are not.
        assert!(!ds.iter().any(|d| d.verb == Verb::new("net", "http")));
        assert!(!ds.iter().any(|d| d.verb.namespace == "schedule"));
        assert!(!ds.iter().any(|d| d.verb.namespace == "notify"));
        // Always-on + terminal still present.
        assert!(ds.iter().any(|d| d.verb == Verb::new("memory", "store")));
        assert!(ds.iter().any(|d| d.verb == Verb::new("shell", "exec")));
        assert!(ds.iter().any(|d| d.verb == Verb::new("net", "check")));
    }

    #[test]
    fn terminal_sourced_tools_use_terminal_source() {
        let ds = native_descriptors(all_wired());
        for action in ["open", "close"] {
            let d = ds
                .iter()
                .find(|d| d.verb == Verb::new("terminal", action))
                .unwrap();
            assert!(matches!(d.source, ToolSource::Terminal));
        }
        let sh = ds
            .iter()
            .find(|d| d.verb == Verb::new("shell", "exec"))
            .unwrap();
        assert!(matches!(sh.source, ToolSource::Terminal));
    }
}
