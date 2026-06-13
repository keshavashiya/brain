//! Wire the kernel's native capabilities into the shared
//! [`intent::ToolRegistry`] at boot.
//!
//! The capability *declarations* live with the backends that implement them
//! (see [`backends::capabilities`]); each backend module owns a `capabilities()`
//! that returns the descriptors it is wired for this run. This module is the
//! composition root's *wiring*: it unions the backend-declared manifest with the
//! handful of capabilities whose logic the binary itself owns — the terminal
//! bridge, sandboxed one-shot exec, and path-scoped filesystem read — and
//! registers the result through the same [`intent::ToolRegistry::register`] path
//! the MCP host uses on `mount`. Native backends, mounted MCP servers, and
//! (future) skill packs are therefore peers feeding one registry.
//!
//! **Awareness, not permission.** Registering a descriptor only makes the
//! capability *describable*; execution still flows through the
//! consent/audit/breaker path. The `usage.tier` mirrors the verb's conservative
//! [`intent::verbs`] tier hint so a reader sees the gate it will hit.

use std::sync::Arc;

use backends::capabilities::{backend, native, read_only, usage};
use intent::{ToolAnnotations, ToolDescriptor, ToolRegistry, ToolSource};

/// Descriptors for the capabilities whose *logic* is wired by the composition
/// root rather than living in the `backends` crate: the terminal bridge and
/// sandboxed one-shot exec (`ToolSource::Terminal`), and path-scoped filesystem
/// read. All are unconditional — `build_processor` wires the terminal bridge and
/// fs backend on every boot.
fn composition_root_capabilities() -> Vec<ToolDescriptor> {
    vec![
        native(
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
        ),
        native(
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
        ),
        native(
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
        ),
        native(
            "fs",
            "read",
            backend("fs"),
            read_only(),
            usage(
                "Read a file or directory the user referenced, to ground the answer in what is actually on disk.",
                "For paths outside security.allowed_paths, or binary/huge files.",
                &["Path is inside security.allowed_paths."],
                "free / local read",
                "\"What's in ./README.md?\"",
            ),
        ),
    ]
}

/// The full live native manifest for this run: the capabilities each backend
/// declares (gated by their own config), unioned with the
/// composition-root-owned ones. Pure (no I/O) so it can be unit-tested and so
/// the baseline backend can snapshot it.
pub fn native_descriptors(config: &brain::BrainConfig) -> Vec<ToolDescriptor> {
    let mut out = backends::native_capabilities(config);
    out.extend(composition_root_capabilities());
    out
}

/// Flatten the live native manifest into the [`backends::CapabilitySummary`]
/// inventory the baseline backend snapshots — keeping the baseline core a pure
/// function of its inputs (no reach-back into the binary).
pub fn capability_inventory(config: &brain::BrainConfig) -> Vec<backends::CapabilitySummary> {
    native_descriptors(config)
        .into_iter()
        .map(|d| backends::CapabilitySummary {
            namespace: d.verb.namespace,
            action: d.verb.action,
            tier: d.usage.tier.unwrap_or_else(|| "unknown".to_string()),
        })
        .collect()
}

/// Register every live native capability into `registry`. Idempotent at the
/// registry level (re-registering the same `tool_id` overwrites). Failures are
/// logged, not fatal — a missing native descriptor degrades the manifest, it
/// doesn't break boot.
pub async fn register_native_capabilities(
    registry: &Arc<dyn ToolRegistry>,
    config: &brain::BrainConfig,
    embedder: Option<&signal::capability_embed::CapabilityEmbedder>,
) {
    let mut descriptors = native_descriptors(config);
    // Semantic capability retrieval: stamp each descriptor's embedding
    // from its text projection before registration, so the router / tool-loop
    // / index can score by cosine. Best-effort — a degraded install with no
    // embedder leaves `embedding` unset and ranking stays lexical-only.
    if let Some(embedder) = embedder {
        embedder.embed_descriptors(&mut descriptors).await;
    }
    let count = descriptors.len();
    let embedded = descriptors.iter().filter(|d| d.embedding.is_some()).count();
    for d in descriptors {
        let id = d.tool_id.clone();
        if let Err(e) = registry.register(d).await {
            tracing::warn!(tool_id = %id, error = %e, "native capability registration failed");
        }
    }
    tracing::info!(
        count,
        embedded,
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
    use intent::Verb;

    /// A config with every gated backend enabled, for exercising the full set.
    fn all_enabled() -> brain::BrainConfig {
        let mut c = brain::BrainConfig::default();
        c.actions.web_search.enabled = true;
        c.actions.scheduling.enabled = true;
        c.actions.messaging.enabled = true;
        c
    }

    #[test]
    fn descriptors_carry_verb_description_and_tier() {
        let ds = native_descriptors(&all_enabled());
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
        let ds = native_descriptors(&all_enabled());
        let del = ds
            .iter()
            .find(|d| d.verb == Verb::new("memory", "delete"))
            .unwrap();
        assert!(del.annotations.destructive_hint);
        assert_eq!(del.usage.tier.as_deref(), Some("destructive"));
    }

    #[test]
    fn disabled_backends_are_omitted() {
        let mut c = brain::BrainConfig::default();
        c.actions.web_search.enabled = false;
        c.actions.scheduling.enabled = false;
        c.actions.messaging.enabled = false;
        let ds = native_descriptors(&c);
        // The egress verb (net.http) is gated by web search; diagnostics are not.
        assert!(!ds.iter().any(|d| d.verb == Verb::new("net", "http")));
        assert!(!ds.iter().any(|d| d.verb.namespace == "schedule"));
        assert!(!ds.iter().any(|d| d.verb.namespace == "notify"));
        // Always-on + terminal still present.
        assert!(ds.iter().any(|d| d.verb == Verb::new("memory", "store")));
        assert!(ds.iter().any(|d| d.verb == Verb::new("shell", "exec")));
        assert!(ds.iter().any(|d| d.verb == Verb::new("net", "check")));
        assert!(ds.iter().any(|d| d.verb == Verb::new("logs", "analyze")));
        assert!(ds
            .iter()
            .any(|d| d.verb == Verb::new("baseline", "capture")));
    }

    #[test]
    fn terminal_sourced_tools_use_terminal_source() {
        let ds = native_descriptors(&all_enabled());
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
