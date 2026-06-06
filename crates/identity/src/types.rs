//! Core identity types.

use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable identifier for the human owner of the Brain instance. Single-user
/// by design (per `docs/ROADMAP.md` § "What Is NOT on the Roadmap").
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct UserId(pub String);

impl fmt::Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for UserId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for UserId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Opaque agent identifier. Examples: `"claude-code"`, `"cursor"`,
/// `"terminal:zsh"`, `"reflex:fs"`, `"mcp:stdio"`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AgentId(pub String);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for AgentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for AgentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Authorization tier — used both as the principal's authorization level and
/// as the tier required by an action. Ordered so `>=` answers "does this
/// principal's tier satisfy the action's required tier?".
///
/// Per-tier policy (`requires_confirmation`, `default_timeout`) mirrors
/// VISION §6 (human in the loop): each tier maps to a distinct approval
/// ceremony. `Read`/`Write`/`Execute` are auto-approved (with audit
/// recording); `Destructive` and `External` block on explicit human
/// approval.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Read-only — never requires confirmation.
    Read,
    /// Write — implicit confirmation, user can undo.
    Write,
    /// Execute — sandboxed, reversible.
    Execute,
    /// Destructive — explicit approval required.
    Destructive,
    /// External — explicit approval + credential audit.
    External,
}

impl Tier {
    /// Whether this tier requires explicit confirmation before execution.
    pub fn requires_confirmation(self) -> bool {
        matches!(self, Tier::Destructive | Tier::External)
    }

    /// Default approval timeout for this tier.
    ///
    /// These are kept humane (tens of seconds, not minutes): an interactive
    /// user answers in seconds, and the non-interactive CLI now fast-fails an
    /// unanswerable gate (W1) instead of blocking to the deadline, so a long
    /// server timeout only ever punished people. Destructive gets the most
    /// breathing room because an irreversible action deserves a beat of
    /// consideration; Read the least because it barely warrants a pause.
    pub fn default_timeout(self) -> std::time::Duration {
        match self {
            Tier::Read => std::time::Duration::from_secs(30),
            Tier::Write => std::time::Duration::from_secs(45),
            Tier::Execute => std::time::Duration::from_secs(60),
            Tier::Destructive => std::time::Duration::from_secs(90),
            Tier::External => std::time::Duration::from_secs(60),
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::Read => "read",
            Tier::Write => "write",
            Tier::Execute => "execute",
            Tier::Destructive => "destructive",
            Tier::External => "external",
        })
    }
}

/// Who is asking. Threaded through `Signal` so every downstream component
/// (audit, confirmation, capability index) can read it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Principal {
    pub user_id: UserId,
    pub agent_id: AgentId,
    /// Dotted-string capability scopes the principal holds. Wildcards like
    /// `fs.*` match every action under `fs`. Empty list = no permissions.
    pub scopes: Vec<String>,
    pub tier: Tier,
}

impl Principal {
    /// Returns `true` iff `verb_ns.verb_action` is covered by any scope
    /// in this principal's scope list. Scope strings support exact match
    /// (`"shell.exec"`) and namespace wildcards (`"fs.*"`).
    pub fn has_scope(&self, verb_ns: &str, verb_action: &str) -> bool {
        let target = format!("{verb_ns}.{verb_action}");
        for scope in &self.scopes {
            if scope == &target {
                return true;
            }
            // Namespace wildcard: "fs.*" matches "fs.read", "fs.write", etc.
            if let Some(prefix) = scope.strip_suffix(".*") {
                if verb_ns == prefix {
                    return true;
                }
            }
            // Global wildcard: "*" matches everything.
            if scope == "*" {
                return true;
            }
        }
        false
    }
}

/// How a [`ModifierConstraint`] compares an allow-list entry against the
/// actual modifier value pulled from an [`AuthorizationRequest`].
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    /// Value must equal an allow entry exactly. The default — the strictest
    /// and the only safe choice for opaque identifiers (server names, ids).
    #[default]
    Exact,
    /// Value must start with an allow entry. For path-like or prefix-shaped
    /// modifiers. (The built-in `path_allowlist` uses a stricter
    /// segment-boundary prefix; this is the plain `starts_with` form.)
    Prefix,
    /// Host-suffix match (case-insensitive): an entry `example.com` permits
    /// `example.com` and any `*.example.com` subdomain. A leading `*.` on the
    /// entry is accepted and ignored, so `*.example.com` behaves identically.
    HostSuffix,
}

/// A per-principal constraint on one modifier of one verb (or verb namespace).
///
/// This is the general form of the older `path_allowlist`: that list is the
/// built-in path constraint; `constraints` lets a principal scope *any*
/// modifier — `net.http` `host`, `shell.exec` `command`, `mcp.mount` `name`,
/// and so on. It is the enforcement substrate for capability-scoped Skill
/// Packs: a pack granted `net.http` to `api.github.com` cannot reach
/// `evil.com`, enforced at signal-entry rather than at execution.
///
/// Semantics (fail-closed): when a principal carries a constraint whose
/// [`Self::applies_to`] matches the request's verb, the named `modifier` MUST
/// be present in the request and its value MUST be [`Self::permits`]ted —
/// otherwise the check denies. An empty `allow` list denies everything for
/// that `(verb, modifier)` pair.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModifierConstraint {
    /// Verb this governs: an exact `"net.http"`, a namespace wildcard
    /// `"net.*"`, or the global `"*"`.
    pub verb: String,
    /// Modifier key read from the request (e.g. `"host"`, `"command"`).
    pub modifier: String,
    #[serde(default)]
    pub match_kind: MatchKind,
    /// Permitted values. Empty = deny everything for this `(verb, modifier)`.
    #[serde(default)]
    pub allow: Vec<String>,
}

impl ModifierConstraint {
    /// Does this constraint govern `verb_ns.verb_action`? Supports exact
    /// match, a `"ns.*"` namespace wildcard, and the global `"*"`.
    pub fn applies_to(&self, verb_ns: &str, verb_action: &str) -> bool {
        if self.verb == "*" {
            return true;
        }
        if let Some(prefix) = self.verb.strip_suffix(".*") {
            return verb_ns == prefix;
        }
        let mut target = String::with_capacity(verb_ns.len() + 1 + verb_action.len());
        target.push_str(verb_ns);
        target.push('.');
        target.push_str(verb_action);
        self.verb == target
    }

    /// Is `value` permitted by this constraint's allow-list under its
    /// [`MatchKind`]? An empty allow-list permits nothing.
    pub fn permits(&self, value: &str) -> bool {
        self.allow.iter().any(|entry| match self.match_kind {
            MatchKind::Exact => entry == value,
            MatchKind::Prefix => value.starts_with(entry.as_str()),
            MatchKind::HostSuffix => host_suffix_match(entry, value),
        })
    }
}

/// `entry` permits `value` when `value` is the host itself or a subdomain of
/// it. A leading `*.` on the entry is stripped first so `example.com` and
/// `*.example.com` behave identically. Case-insensitive.
fn host_suffix_match(entry: &str, value: &str) -> bool {
    let base = entry.trim_start_matches("*.").to_ascii_lowercase();
    if base.is_empty() {
        return false;
    }
    let value = value.to_ascii_lowercase();
    value == base || value.ends_with(&format!(".{base}"))
}

/// Identifier the adapter passes when resolving a principal from auth context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentHint {
    /// API-key-style identification (HTTP / WS / gRPC).
    AgentId(AgentId),
    /// Unknown caller — the store decides whether to materialise a default
    /// or refuse.
    Anonymous,
}

/// Authorization context for `IdentityStore::check`. Carries the verb plus
/// a free-form `modifiers` JSON object so scope checks can read fields like
/// `path` / `cwd`. The intent-routing `IntentToken` carries the same fields
/// and reduces to this struct at the call site.
#[derive(Clone, Debug)]
pub struct AuthorizationRequest {
    pub verb_ns: String,
    pub verb_action: String,
    pub modifiers: serde_json::Value,
}

impl AuthorizationRequest {
    pub fn new(verb_ns: impl Into<String>, verb_action: impl Into<String>) -> Self {
        Self {
            verb_ns: verb_ns.into(),
            verb_action: verb_action.into(),
            modifiers: serde_json::Value::Null,
        }
    }

    pub fn with_modifiers(mut self, modifiers: serde_json::Value) -> Self {
        self.modifiers = modifiers;
        self
    }

    /// Read a string from `modifiers[key]`; returns `None` if absent or
    /// not a string. Used by path-scope check paths.
    pub fn modifier_str(&self, key: &str) -> Option<&str> {
        self.modifiers.get(key)?.as_str()
    }
}

/// Result of an authorization check.
///
/// `EscalateToUser` is the default stance: missing scope is never a silent
/// fail. The `ConfirmationEngine` shows the user a prompt with the carried
/// reason; the user can approve once, grant a standing approval, or reject.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckOutcome {
    Allow,
    EscalateToUser { reason: String },
    Deny { reason: String },
}

#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("unknown agent: {0}")]
    UnknownAgent(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Async trait every adapter holds an `Arc<dyn IdentityStore>` to. The
/// default implementation is [`crate::ConfigIdentityStore`]; production
/// deployments can wrap it with caching, rate limiting, or signed-token
/// validators without changing the trait.
#[async_trait]
pub trait IdentityStore: Send + Sync {
    /// Resolve a `Principal` from an adapter-supplied hint. Returns
    /// `Err(UnknownAgent)` if the hint cannot be mapped — the caller
    /// decides whether to escalate or refuse.
    async fn principal_for(&self, agent_hint: &AgentHint) -> Result<Principal, IdentityError>;

    /// Authorize an action. The `required` tier names the minimum tier the
    /// principal must hold *and* the verb in `req` must be covered by one
    /// of `p.scopes`. Path-scope checks read `req.modifier_str("path")` /
    /// `req.modifier_str("cwd")`.
    async fn check(
        &self,
        p: &Principal,
        req: &AuthorizationRequest,
        required: Tier,
    ) -> CheckOutcome;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_is_strictly_ordered() {
        assert!(Tier::Read < Tier::Write);
        assert!(Tier::Write < Tier::Execute);
        assert!(Tier::Execute < Tier::Destructive);
        assert!(Tier::Destructive < Tier::External);
    }

    #[test]
    fn tier_satisfies_via_ge() {
        // An Execute-tier principal satisfies a Write requirement.
        assert!(Tier::Execute >= Tier::Write);
        // A Read-tier principal does NOT satisfy Execute.
        assert!(Tier::Read < Tier::Execute);
    }

    #[test]
    fn requires_confirmation_only_destructive_external() {
        assert!(!Tier::Read.requires_confirmation());
        assert!(!Tier::Write.requires_confirmation());
        assert!(!Tier::Execute.requires_confirmation());
        assert!(Tier::Destructive.requires_confirmation());
        assert!(Tier::External.requires_confirmation());
    }

    #[test]
    fn default_timeout_increases_with_tier() {
        assert!(Tier::Read.default_timeout() < Tier::Execute.default_timeout());
        assert!(Tier::Execute.default_timeout() < Tier::Destructive.default_timeout());
    }

    #[test]
    fn external_timeout_is_shorter_than_destructive() {
        // External lives in chat; a 5-min deadlock there feels broken.
        assert!(Tier::External.default_timeout() < Tier::Destructive.default_timeout());
    }

    #[test]
    fn display_matches_serde_repr() {
        assert_eq!(Tier::Destructive.to_string(), "destructive");
        let json = serde_json::to_string(&Tier::Destructive).unwrap();
        assert_eq!(json, "\"destructive\"");
    }

    #[test]
    fn has_scope_exact_match() {
        let p = Principal {
            user_id: "k".into(),
            agent_id: "claude-code".into(),
            scopes: vec!["shell.exec".into()],
            tier: Tier::Execute,
        };
        assert!(p.has_scope("shell", "exec"));
        assert!(!p.has_scope("shell", "kill"));
        assert!(!p.has_scope("fs", "read"));
    }

    #[test]
    fn has_scope_namespace_wildcard() {
        let p = Principal {
            user_id: "k".into(),
            agent_id: "claude-code".into(),
            scopes: vec!["fs.*".into()],
            tier: Tier::Write,
        };
        assert!(p.has_scope("fs", "read"));
        assert!(p.has_scope("fs", "write"));
        assert!(!p.has_scope("shell", "exec"));
    }

    #[test]
    fn has_scope_global_wildcard() {
        let p = Principal {
            user_id: "k".into(),
            agent_id: "root".into(),
            scopes: vec!["*".into()],
            tier: Tier::External,
        };
        assert!(p.has_scope("shell", "exec"));
        assert!(p.has_scope("fs", "read"));
        assert!(p.has_scope("net", "http"));
    }

    #[test]
    fn auth_request_modifier_str_reads_paths() {
        let req = AuthorizationRequest::new("fs", "read")
            .with_modifiers(serde_json::json!({ "path": "/tmp/x", "limit": 10 }));
        assert_eq!(req.modifier_str("path"), Some("/tmp/x"));
        assert_eq!(req.modifier_str("limit"), None); // number, not string
        assert_eq!(req.modifier_str("missing"), None);
    }

    fn constraint(
        verb: &str,
        modifier: &str,
        match_kind: MatchKind,
        allow: &[&str],
    ) -> ModifierConstraint {
        ModifierConstraint {
            verb: verb.into(),
            modifier: modifier.into(),
            match_kind,
            allow: allow.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn constraint_applies_to_exact_namespace_and_global() {
        let exact = constraint("net.http", "host", MatchKind::HostSuffix, &[]);
        assert!(exact.applies_to("net", "http"));
        assert!(!exact.applies_to("net", "connect"));
        assert!(!exact.applies_to("fs", "read"));

        let ns = constraint("net.*", "host", MatchKind::HostSuffix, &[]);
        assert!(ns.applies_to("net", "http"));
        assert!(ns.applies_to("net", "connect"));
        assert!(!ns.applies_to("fs", "read"));

        let global = constraint("*", "host", MatchKind::HostSuffix, &[]);
        assert!(global.applies_to("anything", "at-all"));
    }

    #[test]
    fn constraint_exact_match() {
        let c = constraint(
            "mcp.mount",
            "name",
            MatchKind::Exact,
            &["github", "filesystem"],
        );
        assert!(c.permits("github"));
        assert!(!c.permits("git")); // not a prefix match
        assert!(!c.permits("evil"));
    }

    #[test]
    fn constraint_prefix_match() {
        let c = constraint(
            "shell.exec",
            "command",
            MatchKind::Prefix,
            &["git ", "cargo "],
        );
        assert!(c.permits("git status"));
        assert!(c.permits("cargo build"));
        assert!(!c.permits("rm -rf /"));
    }

    #[test]
    fn constraint_host_suffix_match() {
        let c = constraint(
            "net.http",
            "host",
            MatchKind::HostSuffix,
            &["github.com", "*.internal.example"],
        );
        assert!(c.permits("github.com")); // bare host
        assert!(c.permits("api.github.com")); // subdomain
        assert!(c.permits("svc.internal.example")); // leading *. on entry ignored
        assert!(c.permits("API.GitHub.com")); // case-insensitive
        assert!(!c.permits("github.com.evil.com")); // suffix-boundary respected
        assert!(!c.permits("evil.com"));
    }

    #[test]
    fn constraint_empty_allow_denies_everything() {
        let c = constraint("net.http", "host", MatchKind::HostSuffix, &[]);
        assert!(!c.permits("github.com"));
        assert!(!c.permits(""));
    }
}
