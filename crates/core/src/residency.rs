//! Namespace data-residency policy.
//!
//! A namespace marked `local_only` physically cannot reach a non-local
//! provider: recall results from it are excluded from prompts bound for
//! remote LLMs, its content is never embedded by a remote embedder, and
//! exports mark it. The policy is enforced at each egress point, not
//! configured by convention.

use serde::{Deserialize, Serialize};

/// Where content from a namespace may travel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    /// No restriction — content may be sent to any configured provider.
    #[default]
    Any,
    /// Content never leaves this machine: excluded from remote-bound
    /// prompts, embedded only by a loopback embedder (deterministic
    /// fallback otherwise), marked in exports.
    LocalOnly,
}

impl Residency {
    pub fn is_local_only(self) -> bool {
        matches!(self, Residency::LocalOnly)
    }
}

/// Per-namespace policy block under `memory.namespaces.<name>`.
///
/// An entry also governs its `name/…` sub-namespaces unless a more
/// specific entry exists — the same hierarchy rule recall uses for
/// namespace scoping.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamespaceConfig {
    #[serde(default)]
    pub residency: Residency,
}

/// Resolve a namespace against per-namespace entries: exact match first,
/// then each `/`-truncated ancestor (`a/b/c` → `a/b` → `a`), so
/// sub-namespaces inherit unless they declare their own policy — the
/// same hierarchy rule recall uses for namespace scoping.
pub fn resolve_residency<F>(namespace: &str, lookup: F) -> Residency
where
    F: Fn(&str) -> Option<Residency>,
{
    let mut scope = namespace;
    loop {
        if let Some(r) = lookup(scope) {
            return r;
        }
        match scope.rsplit_once('/') {
            Some((parent, _)) => scope = parent,
            None => return Residency::default(),
        }
    }
}

/// Compiled residency policy — a config-free copy of
/// `memory.namespaces.*.residency` that can be handed to subsystems
/// (action backends, graph sinks) that never see `BrainConfig`.
#[derive(Debug, Clone, Default)]
pub struct ResidencyPolicy {
    entries: std::collections::HashMap<String, Residency>,
}

impl ResidencyPolicy {
    pub fn new(entries: std::collections::HashMap<String, Residency>) -> Self {
        Self { entries }
    }

    /// True when no namespace declares a policy — the zero-config fast
    /// path: every gate is a no-op.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn residency_of(&self, namespace: &str) -> Residency {
        resolve_residency(namespace, |s| self.entries.get(s).copied())
    }

    pub fn is_local_only(&self, namespace: &str) -> bool {
        self.residency_of(namespace).is_local_only()
    }
}

/// True when `url`'s host is this machine (loopback). This is the
/// locality test for providers: `localhost`, `127.0.0.0/8`, and IPv6
/// `::1` count; LAN addresses do **not** — `local_only` promises the
/// content never leaves the machine, not the subnet.
pub fn url_is_loopback(url: &str) -> bool {
    let Some(host) = host_of(url) else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => false,
    }
}

/// Extract the host portion of a URL without pulling in a URL crate:
/// strip the scheme, cut at the first `/`, drop credentials, then strip
/// the port (bracket-aware for IPv6 literals).
fn host_of(url: &str) -> Option<&str> {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next()?;
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    if host_port.is_empty() {
        return None;
    }
    if let Some(stripped) = host_port.strip_prefix('[') {
        // IPv6 literal: [::1]:8080
        return stripped.split_once(']').map(|(h, _)| h);
    }
    Some(
        host_port
            .rsplit_once(':')
            .map(|(h, _)| h)
            .unwrap_or(host_port),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_hosts_are_local() {
        for url in [
            "http://localhost:11434",
            "http://127.0.0.1:8080/v1",
            "http://LOCALHOST/api",
            "http://[::1]:1234/v1",
            "https://127.0.0.99",
        ] {
            assert!(url_is_loopback(url), "{url} should be loopback");
        }
    }

    #[test]
    fn remote_hosts_are_not_local() {
        for url in [
            "https://api.openai.com/v1",
            "http://192.168.1.10:11434", // LAN is not "this machine"
            "https://generativelanguage.googleapis.com",
            "http://10.0.0.5",
            "",
        ] {
            assert!(!url_is_loopback(url), "{url} should not be loopback");
        }
    }

    #[test]
    fn residency_default_is_any() {
        assert_eq!(Residency::default(), Residency::Any);
        let ns: NamespaceConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(ns.residency, Residency::Any);
    }

    #[test]
    fn residency_parses_snake_case() {
        let ns: NamespaceConfig = serde_yaml::from_str("residency: local_only").unwrap();
        assert!(ns.residency.is_local_only());
    }

    #[test]
    fn sub_namespaces_inherit_and_can_override() {
        let policy = ResidencyPolicy::new(
            [
                ("private".to_string(), Residency::LocalOnly),
                ("private/share".to_string(), Residency::Any),
            ]
            .into(),
        );
        assert!(policy.is_local_only("private"));
        assert!(policy.is_local_only("private/health"));
        assert!(policy.is_local_only("private/health/labs"));
        assert!(
            !policy.is_local_only("private/share"),
            "child override wins"
        );
        assert!(!policy.is_local_only("private/share/x"));
        assert!(
            !policy.is_local_only("personal"),
            "undeclared defaults to any"
        );
        // `privateer` must not match the `private` entry (segment, not prefix).
        assert!(!policy.is_local_only("privateer"));
    }
}
