//! `ConfigIdentityStore` — default identity store backed by
//! `~/.brain/config.yaml`.
//!
//! The YAML shape is:
//!
//! ```yaml
//! identity:
//!   user_id: keshav
//!   principals:
//!     - agent_id: claude-code
//!       scopes: [fs.*, shell.exec, net.http]
//!       tier: execute
//!       path_allowlist:        # optional, for fs.* / memory.* scopes
//!         - /Users/keshav/Developer
//!         - /tmp
//!     - agent_id: cursor
//!       scopes: [fs.read]
//!       tier: read
//! ```
//!
//! Behaviour:
//! - `principal_for(AgentId(x))` looks up `x` in `principals` and returns
//!   the materialised `Principal`. Returns `Err(UnknownAgent)` on miss.
//! - `principal_for(Anonymous)` returns `Err(UnknownAgent("anonymous"))` —
//!   adapters that receive unauthenticated traffic decide whether to refuse
//!   or fall back to a configured anonymous principal.
//! - `check` enforces three rules in order:
//!   1. `principal.tier >= required` — otherwise `EscalateToUser`.
//!   2. `principal.has_scope(verb_ns, verb_action)` — otherwise `EscalateToUser`.
//!   3. If the verb is path-scoped (`fs.*`, `memory.*`, or `shell.exec` with
//!      a `cwd` modifier), the requested path must be covered by the
//!      principal's `path_allowlist` — otherwise `Deny` (paths are a
//!      stricter boundary than verb scopes; an unauthorised path is not
//!      "ask the user", it's "no").

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::types::{
    AgentHint, AgentId, AuthorizationRequest, CheckOutcome, IdentityError, IdentityStore,
    Principal, Tier, UserId,
};

/// Top-level YAML structure under the `identity:` key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityConfig {
    pub user_id: String,
    #[serde(default)]
    pub principals: Vec<PrincipalConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrincipalConfig {
    pub agent_id: String,
    #[serde(default)]
    pub scopes: Vec<String>,
    pub tier: Tier,
    /// Path prefixes the principal may read/write. Empty list = no
    /// path-scoped operations allowed. Matched as prefixes against the
    /// canonical (absolute) path in `req.modifiers["path"]` / `["cwd"]`.
    #[serde(default)]
    pub path_allowlist: Vec<String>,
}

/// In-memory store materialised from an [`IdentityConfig`].
pub struct ConfigIdentityStore {
    user_id: UserId,
    by_agent: HashMap<AgentId, StoredPrincipal>,
}

struct StoredPrincipal {
    principal: Principal,
    path_allowlist: Vec<String>,
}

impl ConfigIdentityStore {
    /// Construct from a parsed `IdentityConfig`.
    pub fn from_config(cfg: IdentityConfig) -> Self {
        let user_id = UserId(cfg.user_id);
        let mut by_agent = HashMap::with_capacity(cfg.principals.len());
        for entry in cfg.principals {
            let agent_id = AgentId(entry.agent_id.clone());
            let principal = Principal {
                user_id: user_id.clone(),
                agent_id: agent_id.clone(),
                scopes: entry.scopes,
                tier: entry.tier,
            };
            by_agent.insert(
                agent_id,
                StoredPrincipal {
                    principal,
                    path_allowlist: entry.path_allowlist,
                },
            );
        }
        Self { user_id, by_agent }
    }

    /// Construct from a YAML file path. Reads the file and parses just the
    /// `identity:` section.
    pub fn from_yaml_path(path: impl AsRef<Path>) -> Result<Arc<Self>, IdentityError> {
        let text = std::fs::read_to_string(path)?;
        let root: serde_yaml::Value = serde_yaml::from_str(&text)?;
        let identity = root
            .get("identity")
            .ok_or_else(|| IdentityError::Config("missing `identity:` section".into()))?;
        let cfg: IdentityConfig = serde_yaml::from_value(identity.clone())?;
        Ok(Arc::new(Self::from_config(cfg)))
    }

    pub fn user_id(&self) -> &UserId {
        &self.user_id
    }

    /// `true` if this verb+request pair must have its path checked against
    /// the principal's `path_allowlist`. The rule:
    ///
    /// - `fs.*` ALWAYS requires a `path` modifier — every fs verb touches
    ///   a filesystem location.
    /// - `memory.import` / `memory.export` require a `path` modifier — these
    ///   are the file-bound memory verbs. Other `memory.*` operations
    ///   (`store`, `delete`, `recall`) act on the embedded semantic memory,
    ///   not a filesystem path, so they are NOT path-scoped.
    /// - `shell.exec` only requires path-scoping when `cwd` is explicitly
    ///   set (the sandbox executor handles the default-cwd case).
    /// - Every other verb is not path-scoped.
    fn needs_path_check(req: &AuthorizationRequest) -> bool {
        if req.verb_ns == "fs" {
            return true;
        }
        if req.verb_ns == "memory" && matches!(req.verb_action.as_str(), "import" | "export") {
            return true;
        }
        if req.verb_ns == "shell" && req.verb_action == "exec" && req.modifier_str("cwd").is_some()
        {
            return true;
        }
        false
    }

    /// Returns the path (if any) the request claims access to. Inspects
    /// `path` first, then `cwd`.
    fn requested_path(req: &AuthorizationRequest) -> Option<String> {
        req.modifier_str("path")
            .or_else(|| req.modifier_str("cwd"))
            .map(|s| s.to_string())
    }

    fn path_is_covered(path: &str, allowlist: &[String]) -> bool {
        if allowlist.is_empty() {
            return false;
        }
        // Match by prefix on a normalised string. Does not do full
        // canonicalisation (we'd need to `std::fs::canonicalize` which
        // requires the path to exist) — instead we strip a trailing `/`
        // from allowlist entries and require the path to start with the
        // allowlist entry followed by `/` or end-of-string.
        for entry in allowlist {
            let trimmed = entry.trim_end_matches('/');
            if path == trimmed {
                return true;
            }
            let with_sep = format!("{trimmed}/");
            if path.starts_with(&with_sep) {
                return true;
            }
        }
        false
    }
}

#[async_trait]
impl IdentityStore for ConfigIdentityStore {
    async fn principal_for(&self, agent_hint: &AgentHint) -> Result<Principal, IdentityError> {
        match agent_hint {
            AgentHint::AgentId(id) => self
                .by_agent
                .get(id)
                .map(|sp| sp.principal.clone())
                .ok_or_else(|| IdentityError::UnknownAgent(id.0.clone())),
            AgentHint::Anonymous => Err(IdentityError::UnknownAgent("anonymous".into())),
        }
    }

    async fn check(
        &self,
        p: &Principal,
        req: &AuthorizationRequest,
        required: Tier,
    ) -> CheckOutcome {
        // 1. Tier check.
        if p.tier < required {
            return CheckOutcome::EscalateToUser {
                reason: format!(
                    "agent_id={} tier={} below required {}",
                    p.agent_id, p.tier, required
                ),
            };
        }

        // 2. Scope check.
        if !p.has_scope(&req.verb_ns, &req.verb_action) {
            return CheckOutcome::EscalateToUser {
                reason: format!(
                    "agent_id={} missing scope {}.{}",
                    p.agent_id, req.verb_ns, req.verb_action
                ),
            };
        }

        // 3. Path-scope check (if applicable).
        if Self::needs_path_check(req) {
            let Some(path) = Self::requested_path(req) else {
                return CheckOutcome::Deny {
                    reason: format!(
                        "{}.{} requires a `path` or `cwd` modifier",
                        req.verb_ns, req.verb_action
                    ),
                };
            };
            let allowlist = self
                .by_agent
                .get(&p.agent_id)
                .map(|sp| sp.path_allowlist.as_slice())
                .unwrap_or(&[]);
            if !Self::path_is_covered(&path, allowlist) {
                return CheckOutcome::Deny {
                    reason: format!(
                        "agent_id={} cannot access {} (not in path_allowlist)",
                        p.agent_id, path
                    ),
                };
            }
        }

        CheckOutcome::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store_with(cfg_yaml: &str) -> ConfigIdentityStore {
        let cfg: IdentityConfig = serde_yaml::from_str(cfg_yaml).unwrap();
        ConfigIdentityStore::from_config(cfg)
    }

    #[tokio::test]
    async fn principal_for_returns_configured_agent() {
        let s = store_with(
            r#"
            user_id: keshav
            principals:
              - agent_id: claude-code
                scopes: [shell.exec]
                tier: execute
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        assert_eq!(p.agent_id, AgentId("claude-code".into()));
        assert_eq!(p.user_id, UserId("keshav".into()));
        assert_eq!(p.tier, Tier::Execute);
    }

    #[tokio::test]
    async fn principal_for_unknown_agent_fails() {
        let s = store_with(
            r#"
            user_id: k
            principals: []
            "#,
        );
        let err = s
            .principal_for(&AgentHint::AgentId("ghost".into()))
            .await
            .unwrap_err();
        assert!(matches!(err, IdentityError::UnknownAgent(ref a) if a == "ghost"));
    }

    #[tokio::test]
    async fn principal_for_anonymous_fails_closed() {
        let s = store_with(
            r#"
            user_id: k
            principals: []
            "#,
        );
        assert!(s.principal_for(&AgentHint::Anonymous).await.is_err());
    }

    #[tokio::test]
    async fn check_allows_with_sufficient_tier_and_scope() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: claude-code
                scopes: [shell.exec]
                tier: execute
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("shell", "exec");
        assert_eq!(s.check(&p, &req, Tier::Execute).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn check_escalates_on_insufficient_tier() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: cursor
                scopes: [shell.exec]
                tier: read
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("cursor".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("shell", "exec");
        match s.check(&p, &req, Tier::Execute).await {
            CheckOutcome::EscalateToUser { reason } => {
                assert!(reason.contains("tier=read"), "reason: {reason}");
                assert!(reason.contains("required execute"), "reason: {reason}");
            }
            other => panic!("expected EscalateToUser, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn check_escalates_on_missing_scope() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: cursor
                scopes: [fs.read]
                tier: execute
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("cursor".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("shell", "exec");
        match s.check(&p, &req, Tier::Execute).await {
            CheckOutcome::EscalateToUser { reason } => {
                assert!(
                    reason.contains("missing scope shell.exec"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected EscalateToUser, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn check_denies_path_outside_allowlist() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: claude-code
                scopes: [fs.read]
                tier: read
                path_allowlist: [/Users/keshav/Developer]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("fs", "read")
            .with_modifiers(json!({ "path": "/etc/passwd" }));
        match s.check(&p, &req, Tier::Read).await {
            CheckOutcome::Deny { reason } => {
                assert!(reason.contains("/etc/passwd"), "reason: {reason}");
                assert!(reason.contains("path_allowlist"), "reason: {reason}");
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn check_allows_path_inside_allowlist() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: claude-code
                scopes: [fs.read]
                tier: read
                path_allowlist: [/Users/keshav/Developer]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("fs", "read")
            .with_modifiers(json!({ "path": "/Users/keshav/Developer/brain/src/lib.rs" }));
        assert_eq!(s.check(&p, &req, Tier::Read).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn check_denies_path_scoped_call_without_path_modifier() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: claude-code
                scopes: [fs.read]
                tier: read
                path_allowlist: [/tmp]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("fs", "read"); // no modifier
        match s.check(&p, &req, Tier::Read).await {
            CheckOutcome::Deny { reason } => {
                assert!(
                    reason.contains("requires a `path` or `cwd` modifier"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn shell_exec_path_scoped_via_cwd_modifier() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: claude-code
                scopes: [shell.exec]
                tier: execute
                path_allowlist: [/Users/keshav/Developer]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        let inside = AuthorizationRequest::new("shell", "exec")
            .with_modifiers(json!({ "cwd": "/Users/keshav/Developer/brain" }));
        assert_eq!(
            s.check(&p, &inside, Tier::Execute).await,
            CheckOutcome::Allow
        );

        let outside = AuthorizationRequest::new("shell", "exec")
            .with_modifiers(json!({ "cwd": "/private/etc" }));
        match s.check(&p, &outside, Tier::Execute).await {
            CheckOutcome::Deny { .. } => {} // expected
            other => panic!("expected Deny for outside cwd, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn path_match_does_not_falsely_extend() {
        // "/Users/keshav" allowlist must NOT match "/Users/keshav-evil/foo".
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: a
                scopes: [fs.read]
                tier: read
                path_allowlist: [/Users/keshav]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("a".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("fs", "read")
            .with_modifiers(json!({ "path": "/Users/keshav-evil/foo" }));
        match s.check(&p, &req, Tier::Read).await {
            CheckOutcome::Deny { .. } => {} // expected
            other => panic!("expected Deny, got {other:?}"),
        }
    }
}
