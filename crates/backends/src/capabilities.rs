//! The declarative capability-registration substrate.
//!
//! Every native capability the kernel exposes is *declared* as data next to the
//! backend logic that implements it: each backend module owns a
//! `pub fn capabilities(&BrainConfig) -> Vec<ToolDescriptor>` that returns the
//! descriptors it is wired for this run (applying its own config gate). The
//! [`native_capabilities`] registrar here is the single collection point — it
//! concatenates every backend's declaration into the one manifest that feeds
//! both the internal reasoner (the SOUL capability digest) and external clients
//! (`tools/list`). The composition root registers the result through the same
//! [`intent::ToolRegistry::register`] path the MCP host uses on mount, so native
//! backends, mounted MCP servers, and (future) skill packs are all peers.
//!
//! **Authoring lives with the logic, not in the binary.** Before this, the
//! native catalog was a hand-built `Vec` inside the `cli` crate; adding a
//! capability meant editing the binary. Now adding a capability touches only the
//! backend that owns it. Capabilities whose *logic* is not in this crate (the
//! terminal bridge, sandboxed exec, path-scoped filesystem read) are declared by
//! the composition root that wires them, using the same [`native`] builder.
//!
//! **Awareness, not permission.** A descriptor only makes a capability
//! *describable*; execution still flows through the consent/audit/breaker path.
//! The `usage.tier` mirrors the verb's conservative [`intent::verbs`] tier hint
//! so a reader sees the gate it will hit.

use brain::BrainConfig;
use intent::{verbs, BackendId, ToolAnnotations, ToolDescriptor, ToolSource, ToolUsage, Verb};

/// Build a native [`ToolDescriptor`] for `(ns, action)`, pulling the description
/// and tier from the canonical [`verbs`] vocabulary and layering the supplied
/// usage guidance on top. The `tier` is filled from the verb hint when the
/// caller leaves it unset. Shared by every backend declaration and by the
/// composition root for the capabilities it owns directly.
pub fn native(
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
        // `native:` prefix mirrors the `mcp:{server}:{tool}` id shape so ids
        // stay source-discriminable at a glance.
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

/// A [`ToolSource::NativeBackend`] for the named backend. Convenience for
/// declarations so each module names its own backend id once.
pub fn backend(name: &str) -> ToolSource {
    ToolSource::NativeBackend {
        backend: BackendId::new(name),
    }
}

/// Read-only, idempotent annotations.
pub fn read_only() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: true,
        destructive_hint: false,
        idempotent_hint: true,
    }
}

/// Destructive, non-idempotent annotations.
pub fn destructive() -> ToolAnnotations {
    ToolAnnotations {
        read_only_hint: false,
        destructive_hint: true,
        idempotent_hint: false,
    }
}

/// Assemble reasoner-facing [`ToolUsage`] guidance. `tier` is left unset and
/// filled from the verb hint by [`native`].
pub fn usage(
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
        tier: None,
    }
}

/// The registrar: collect every native capability this crate's backends declare
/// for the given config into one manifest. Each backend applies its own gate, so
/// disabled backends contribute nothing. Pure (no I/O) so it can be unit-tested
/// and so the composition root can both register it and snapshot it (baseline).
pub fn native_capabilities(config: &BrainConfig) -> Vec<ToolDescriptor> {
    let mut out = Vec::new();
    out.extend(crate::memory::capabilities(config));
    out.extend(crate::search::capabilities(config));
    out.extend(crate::scheduling::capabilities(config));
    out.extend(crate::messaging::capabilities(config));
    out.extend(crate::net::capabilities(config));
    out.extend(crate::security::capabilities(config));
    out.extend(crate::logs::capabilities(config));
    out.extend(crate::baseline::capabilities(config));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn all_enabled() -> BrainConfig {
        let mut c = BrainConfig::default();
        c.actions.web_search.enabled = true;
        c.actions.scheduling.enabled = true;
        c.actions.messaging.enabled = true;
        c
    }

    fn has(ds: &[ToolDescriptor], ns: &str, action: &str) -> bool {
        ds.iter().any(|d| d.verb == Verb::new(ns, action))
    }

    /// Each always-on backend must declare at least its headline capability, so
    /// a backend that ships logic but forgets to declare a descriptor is caught
    /// here rather than going silently un-advertised.
    #[test]
    fn every_always_on_backend_declares_its_capability() {
        let ds = native_capabilities(&all_enabled());
        for (ns, action) in [
            ("memory", "store"),
            ("memory", "delete"),
            ("net", "check"),
            ("net", "trace"),
            ("net", "cert"),
            ("security", "audit"),
            ("logs", "analyze"),
            ("baseline", "capture"),
            ("baseline", "diff"),
            ("baseline", "list"),
        ] {
            assert!(has(&ds, ns, action), "{ns}.{action} not declared");
        }
    }

    /// A backend whose config gate is off contributes nothing to the manifest.
    #[test]
    fn config_gates_drop_disabled_backends() {
        let mut c = BrainConfig::default();
        c.actions.web_search.enabled = false;
        c.actions.scheduling.enabled = false;
        c.actions.messaging.enabled = false;
        let ds = native_capabilities(&c);
        assert!(
            !has(&ds, "net", "http"),
            "web search disabled → no net.http"
        );
        assert!(!ds.iter().any(|d| d.verb.namespace == "schedule"));
        assert!(!ds.iter().any(|d| d.verb.namespace == "notify"));
        // Always-on diagnostics + memory survive the gate.
        assert!(has(&ds, "net", "check"));
        assert!(has(&ds, "memory", "store"));
    }

    /// Tool ids are unique across the assembled manifest — a collision would
    /// silently overwrite a capability on registration.
    #[test]
    fn declared_tool_ids_are_unique() {
        let ds = native_capabilities(&all_enabled());
        let mut ids: Vec<&str> = ds.iter().map(|d| d.tool_id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate tool_id in native manifest");
    }

    /// Round-trip: every declared descriptor registers through the same
    /// [`ToolRegistry::register`] path the MCP host uses, and is then
    /// retrievable from the manifest by id.
    #[tokio::test]
    async fn declared_capabilities_register_and_are_retrievable() {
        let ds = native_capabilities(&all_enabled());
        let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
        for d in &ds {
            registry.register(d.clone()).await.unwrap();
        }
        assert_eq!(registry.list().await.len(), ds.len());
        for d in &ds {
            assert!(
                registry.get(&d.tool_id).await.is_some(),
                "{} not retrievable after registration",
                d.tool_id
            );
        }
    }
}
