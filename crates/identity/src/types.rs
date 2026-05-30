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
}
