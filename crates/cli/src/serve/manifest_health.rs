//! Manifest-health computation — the writer side of [`brain::ManifestHealth`].
//!
//! The serve loop spawns one bounded sweep task (see
//! `background::spawn_manifest_health_sweep`) that, each round, probes the
//! subsystems registered capabilities depend on and stamps every tool's
//! runtime health. This module owns the pure pieces that loop composes, kept
//! here (not inline in the loop) so they are unit-testable without a runtime:
//!
//! * [`Dependency`] / [`capability_dependency`] — what external subsystem a
//!   capability needs to function (the embedding model, the network, or
//!   nothing). Derived from the descriptor's source + verb; documented as the
//!   one place to extend when a new dependency-bearing backend verb lands.
//! * [`health_for`] — fold a capability's dependency + the live subsystem
//!   readings (embedder reachable? network online? breaker open?) into a
//!   [`CapabilityHealth`].
//!
//! The breaker is checked first and unconditionally — an open breaker means the
//! capability fails fast regardless of why — then dependency reachability.

use brain::{CapabilityHealth, ConnectivityState};
use intent::{ToolDescriptor, ToolSource};

/// The external subsystem a capability needs to function this turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dependency {
    /// Needs the embedding model (semantic memory writes embed their content).
    Embedding,
    /// Needs outbound network reachability (web search, raw HTTP).
    Network,
    /// No external subsystem beyond the always-local stores — only an open
    /// circuit breaker can degrade it.
    None,
}

/// What subsystem `tool` depends on for health purposes.
///
/// Derived from the descriptor's source + verb namespace/action:
/// * native semantic-memory writes (`memory.store`, `memory.import`) embed
///   their content, so they need the embedding model;
/// * the network namespaces (`web.*`, `net.*`) need outbound connectivity;
/// * everything else — local memory reads/deletes, the terminal, filesystem,
///   and MCP tools (whose reachability is the host's quarantine concern, not
///   the kernel's internet view) — has no external dependency here.
///
/// This is the single extension point: a new backend verb that depends on a
/// probed subsystem adds its arm here.
pub(crate) fn capability_dependency(tool: &ToolDescriptor) -> Dependency {
    match &tool.source {
        // An MCP server's reachability is tracked by the host (mount / refresh
        // / quarantine), not by the kernel's internet-connectivity view — a
        // local stdio server works fine while the internet is down. Leave it
        // dependency-free here so connectivity loss doesn't falsely degrade it.
        ToolSource::McpServer { .. } => Dependency::None,
        ToolSource::NativeBackend { .. } | ToolSource::Terminal => {
            let (ns, action) = (tool.verb.namespace.as_str(), tool.verb.action.as_str());
            match ns {
                "memory" if matches!(action, "store" | "import") => Dependency::Embedding,
                "web" | "net" => Dependency::Network,
                _ => Dependency::None,
            }
        }
    }
}

/// Fold a capability's dependency and the live subsystem readings into its
/// health. `breaker_open` wins unconditionally (the call fails fast); otherwise
/// an unreachable dependency degrades it; otherwise it is verified.
pub(crate) fn health_for(
    dependency: Dependency,
    embedder_ok: bool,
    connectivity: ConnectivityState,
    breaker_open: bool,
) -> CapabilityHealth {
    if breaker_open {
        return CapabilityHealth::BreakerOpen;
    }
    match dependency {
        Dependency::Embedding if !embedder_ok => CapabilityHealth::Degraded {
            reason: "local embedding model unreachable".to_string(),
        },
        Dependency::Network if connectivity == ConnectivityState::Offline => {
            CapabilityHealth::Degraded {
                reason: "network unreachable".to_string(),
            }
        }
        _ => CapabilityHealth::Verified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use intent::{BackendId, Verb};

    fn native(ns: &str, action: &str) -> ToolDescriptor {
        ToolDescriptor {
            tool_id: format!("{ns}.{action}"),
            source: ToolSource::NativeBackend {
                backend: BackendId::new(ns),
            },
            verb: Verb::new(ns, action),
            description: String::new(),
            input_schema: serde_json::json!({"type": "object"}),
            output_schema: None,
            capabilities: Vec::new(),
            annotations: Default::default(),
            usage: Default::default(),
            embedding: None,
        }
    }

    fn mcp(server: &str, action: &str) -> ToolDescriptor {
        let mut d = native("mcp", action);
        d.source = ToolSource::McpServer {
            server: server.to_string(),
        };
        d
    }

    #[test]
    fn dependency_derivation() {
        assert_eq!(
            capability_dependency(&native("memory", "store")),
            Dependency::Embedding
        );
        assert_eq!(
            capability_dependency(&native("memory", "import")),
            Dependency::Embedding
        );
        // Local removal doesn't embed.
        assert_eq!(
            capability_dependency(&native("memory", "delete")),
            Dependency::None
        );
        assert_eq!(
            capability_dependency(&native("web", "search")),
            Dependency::Network
        );
        assert_eq!(
            capability_dependency(&native("net", "http")),
            Dependency::Network
        );
        // MCP tools are never connectivity-degraded (server reachability is the
        // host's concern); local faculties have no external dependency.
        assert_eq!(capability_dependency(&mcp("fs", "read")), Dependency::None);
        assert_eq!(
            capability_dependency(&native("terminal", "open")),
            Dependency::None
        );
    }

    #[test]
    fn breaker_open_wins_over_everything() {
        // Even a healthy dependency: an open breaker fails the call fast.
        assert_eq!(
            health_for(Dependency::None, true, ConnectivityState::Online, true),
            CapabilityHealth::BreakerOpen
        );
        assert_eq!(
            health_for(Dependency::Embedding, true, ConnectivityState::Online, true),
            CapabilityHealth::BreakerOpen
        );
    }

    #[test]
    fn embedding_dependent_degrades_when_embedder_down() {
        // The DoD: embedder unreachable → embedding-dependent capability degraded.
        let h = health_for(
            Dependency::Embedding,
            false,
            ConnectivityState::Online,
            false,
        );
        assert_eq!(h.as_str(), "degraded");
        assert_eq!(
            h.reason().as_deref(),
            Some("local embedding model unreachable")
        );
        // A non-embedding capability is unaffected by the embedder being down.
        assert_eq!(
            health_for(Dependency::None, false, ConnectivityState::Online, false),
            CapabilityHealth::Verified
        );
    }

    #[test]
    fn network_dependent_degrades_only_when_offline() {
        assert_eq!(
            health_for(Dependency::Network, true, ConnectivityState::Offline, false),
            CapabilityHealth::Degraded {
                reason: "network unreachable".to_string()
            }
        );
        // Degraded (some endpoints up) is not full offline — don't cry wolf.
        assert_eq!(
            health_for(
                Dependency::Network,
                true,
                ConnectivityState::Degraded,
                false
            ),
            CapabilityHealth::Verified
        );
        assert_eq!(
            health_for(Dependency::Network, true, ConnectivityState::Online, false),
            CapabilityHealth::Verified
        );
    }
}
