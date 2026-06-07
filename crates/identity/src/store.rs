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
//! - `check` enforces four rules in order:
//!   1. `principal.tier >= required` — otherwise `EscalateToUser`.
//!   2. `principal.has_scope(verb_ns, verb_action)` — otherwise `EscalateToUser`.
//!   3. If the verb is path-scoped (`fs.*`, `memory.*`, or `shell.exec` with
//!      a `cwd` modifier), the requested path must be covered by the
//!      principal's `path_allowlist` — otherwise `Deny` (paths are a
//!      stricter boundary than verb scopes; an unauthorised path is not
//!      "ask the user", it's "no").
//!   4. For every per-principal [`ModifierConstraint`] whose verb matches the
//!      request, the named modifier must be present and permitted — otherwise
//!      `Deny`. This is the general form of rule 3: it scopes any modifier
//!      (`net.http` `host`, `shell.exec` `command`, `mcp.mount` `name`), and
//!      is the enforcement substrate for capability-scoped Skill Packs. Like
//!      paths, an unauthorised modifier value is a hard `Deny`, not a prompt.
//!      Constraints are opt-in: a principal with none behaves exactly as
//!      before. Note a constrained verb that arrives without its modifier is
//!      denied (fail-closed) — e.g. a `net.http` `host` constraint denies the
//!      plain `WebSearch` path, which carries no host, so host constraints are
//!      meant for principals on the capability/agent (`ToolCall`) surface.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::types::{
    AgentHint, AgentId, AuthorizationRequest, CheckOutcome, IdentityError, IdentityStore,
    ModifierConstraint, Principal, Tier, UserId,
};

/// Top-level YAML structure under the `identity:` key.
///
/// `Default` yields an empty store (`user_id = ""`, no principals).
/// Adapters that resolve a `Principal` will find none and skip the
/// identity gate — equivalent to pre-wiring behavior. This lets a
/// fresh install run without an `identity:` block.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IdentityConfig {
    #[serde(default)]
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
    /// General per-`(verb, modifier)` allowlists (the path-allowlist's
    /// generalisation — see [`ModifierConstraint`]). Empty = unconstrained
    /// beyond tier/scope/path. Opt-in: principals without constraints behave
    /// exactly as before.
    #[serde(default)]
    pub constraints: Vec<ModifierConstraint>,
}

/// In-memory store materialised from an [`IdentityConfig`].
pub struct ConfigIdentityStore {
    user_id: UserId,
    by_agent: HashMap<AgentId, StoredPrincipal>,
}

struct StoredPrincipal {
    principal: Principal,
    path_allowlist: Vec<String>,
    constraints: Vec<ModifierConstraint>,
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
                    constraints: entry.constraints,
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

        // 4. General modifier constraints (the path-allowlist's generalisation).
        let constraints = self
            .by_agent
            .get(&p.agent_id)
            .map(|sp| sp.constraints.as_slice())
            .unwrap_or(&[]);
        for c in constraints {
            if !c.applies_to(&req.verb_ns, &req.verb_action) {
                continue;
            }
            let Some(value) = req.modifier_str(&c.modifier) else {
                return CheckOutcome::Deny {
                    reason: format!(
                        "agent_id={} {}.{} requires modifier `{}` (constrained)",
                        p.agent_id, req.verb_ns, req.verb_action, c.modifier
                    ),
                };
            };
            if !c.permits(value) {
                return CheckOutcome::Deny {
                    reason: format!(
                        "agent_id={} {}.{} modifier {}={} not permitted",
                        p.agent_id, req.verb_ns, req.verb_action, c.modifier, value
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
    async fn memory_import_export_is_path_scoped_but_store_is_not() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: a
                scopes: [memory.import, memory.export, memory.store]
                tier: write
                path_allowlist: [/tmp]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("a".into()))
            .await
            .unwrap();

        // memory.import without a path → Deny (path-scoped)
        let import_no_path = AuthorizationRequest::new("memory", "import");
        assert!(matches!(
            s.check(&p, &import_no_path, Tier::Write).await,
            CheckOutcome::Deny { .. }
        ));

        // memory.export inside allowlist → Allow
        let export_ok = AuthorizationRequest::new("memory", "export")
            .with_modifiers(json!({ "path": "/tmp/out.json" }));
        assert_eq!(
            s.check(&p, &export_ok, Tier::Write).await,
            CheckOutcome::Allow
        );

        // memory.store is NOT path-scoped — no modifier required.
        let store_call = AuthorizationRequest::new("memory", "store");
        assert_eq!(
            s.check(&p, &store_call, Tier::Write).await,
            CheckOutcome::Allow
        );
    }

    #[tokio::test]
    async fn shell_exec_without_cwd_is_not_path_scoped() {
        // The sandbox executor handles the default-cwd case, so
        // `shell.exec` only path-checks when `cwd` is explicit.
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: a
                scopes: [shell.exec]
                tier: execute
                path_allowlist: [/tmp]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("a".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("shell", "exec");
        assert_eq!(s.check(&p, &req, Tier::Execute).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn path_allowlist_entry_with_trailing_slash_matches() {
        // `path_is_covered` strips trailing `/` on allowlist entries
        // so `/tmp/` and `/tmp` behave identically.
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: a
                scopes: [fs.read]
                tier: read
                path_allowlist: ["/tmp/"]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("a".into()))
            .await
            .unwrap();
        let req =
            AuthorizationRequest::new("fs", "read").with_modifiers(json!({ "path": "/tmp/x" }));
        assert_eq!(s.check(&p, &req, Tier::Read).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn path_exactly_equal_to_allowlist_entry_matches() {
        // Asking for `/tmp` when `/tmp` is in the allowlist must Allow,
        // not require a trailing component.
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: a
                scopes: [fs.read]
                tier: read
                path_allowlist: ["/tmp"]
            "#,
        );
        let p = s
            .principal_for(&AgentHint::AgentId("a".into()))
            .await
            .unwrap();
        let req = AuthorizationRequest::new("fs", "read").with_modifiers(json!({ "path": "/tmp" }));
        assert_eq!(s.check(&p, &req, Tier::Read).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn empty_path_allowlist_denies_all_path_scoped_access() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: a
                scopes: [fs.read]
                tier: read
            "#, // no path_allowlist → default empty
        );
        let p = s
            .principal_for(&AgentHint::AgentId("a".into()))
            .await
            .unwrap();
        let req =
            AuthorizationRequest::new("fs", "read").with_modifiers(json!({ "path": "/tmp/x" }));
        match s.check(&p, &req, Tier::Read).await {
            CheckOutcome::Deny { .. } => {}
            other => panic!("expected Deny for empty allowlist, got {other:?}"),
        }
    }

    #[test]
    fn identity_config_default_is_empty() {
        let cfg = IdentityConfig::default();
        assert!(cfg.user_id.is_empty());
        assert!(cfg.principals.is_empty());
        let store = ConfigIdentityStore::from_config(cfg);
        assert_eq!(store.user_id(), &UserId(String::new()));
    }

    #[tokio::test]
    async fn from_yaml_path_round_trips_principals() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(
            tmp,
            r#"
identity:
  user_id: keshav
  principals:
    - agent_id: claude-code
      scopes: [fs.read, shell.exec]
      tier: execute
      path_allowlist: [/tmp]
"#
        )
        .unwrap();
        let store = ConfigIdentityStore::from_yaml_path(tmp.path()).unwrap();
        let p = store
            .principal_for(&AgentHint::AgentId("claude-code".into()))
            .await
            .unwrap();
        assert_eq!(p.user_id, UserId("keshav".into()));
        assert_eq!(p.tier, Tier::Execute);
        assert!(p.has_scope("shell", "exec"));
    }

    #[test]
    fn from_yaml_path_missing_identity_section_errors() {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, "brain:\n  data_dir: /tmp\n").unwrap();
        let result = ConfigIdentityStore::from_yaml_path(tmp.path());
        match result {
            Err(IdentityError::Config(msg)) => assert!(msg.contains("identity")),
            Err(other) => panic!("expected Config error, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok(store)"),
        }
    }

    #[test]
    fn from_yaml_path_io_error_propagates() {
        let result = ConfigIdentityStore::from_yaml_path("/nonexistent/path/cfg.yaml");
        assert!(matches!(result, Err(IdentityError::Io(_))));
    }

    #[test]
    fn identity_error_display_strings() {
        let unk = IdentityError::UnknownAgent("ghost".into());
        assert!(unk.to_string().contains("ghost"));
        let cfg = IdentityError::Config("missing".into());
        assert!(cfg.to_string().contains("missing"));
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

    // --- Issue 159: general modifier constraints -------------------------

    /// A principal with a `net.http` host allowlist. Used by the constraint
    /// tests below; the `net.http` tier is External so checks pass `External`.
    fn net_host_store() -> ConfigIdentityStore {
        store_with(
            r#"
            user_id: k
            principals:
              - agent_id: agent
                scopes: [net.http]
                tier: external
                constraints:
                  - verb: net.http
                    modifier: host
                    match_kind: host_suffix
                    allow: [github.com, api.openai.com]
            "#,
        )
    }

    async fn agent_of(s: &ConfigIdentityStore) -> Principal {
        s.principal_for(&AgentHint::AgentId("agent".into()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn constraint_permits_allowed_host() {
        let s = net_host_store();
        let p = agent_of(&s).await;
        let req = AuthorizationRequest::new("net", "http")
            .with_modifiers(json!({ "host": "api.github.com" }));
        assert_eq!(s.check(&p, &req, Tier::External).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn constraint_denies_disallowed_host() {
        let s = net_host_store();
        let p = agent_of(&s).await;
        let req =
            AuthorizationRequest::new("net", "http").with_modifiers(json!({ "host": "evil.com" }));
        match s.check(&p, &req, Tier::External).await {
            CheckOutcome::Deny { reason } => assert!(reason.contains("host=evil.com")),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn constrained_verb_without_modifier_is_denied() {
        // A `net.http` host constraint denies a request that carries no host —
        // e.g. the plain WebSearch path (fail-closed; documented in the module
        // header). Constraints are intended for the capability/agent surface.
        let s = net_host_store();
        let p = agent_of(&s).await;
        let req = AuthorizationRequest::new("net", "http"); // no modifiers
        match s.check(&p, &req, Tier::External).await {
            CheckOutcome::Deny { reason } => assert!(reason.contains("requires modifier `host`")),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unconstrained_verb_passes_through() {
        // The host constraint governs net.http only; an fs.read on this
        // principal is unaffected by it (no constraint applies).
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: agent
                scopes: [net.http, fs.read]
                tier: external
                path_allowlist: [/tmp]
                constraints:
                  - verb: net.http
                    modifier: host
                    match_kind: host_suffix
                    allow: [github.com]
            "#,
        );
        let p = agent_of(&s).await;
        let req =
            AuthorizationRequest::new("fs", "read").with_modifiers(json!({ "path": "/tmp/x" }));
        assert_eq!(s.check(&p, &req, Tier::Read).await, CheckOutcome::Allow);
    }

    #[tokio::test]
    async fn empty_allow_constraint_denies() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: agent
                scopes: [mcp.mount]
                tier: external
                constraints:
                  - verb: mcp.mount
                    modifier: name
                    allow: []
            "#,
        );
        let p = agent_of(&s).await;
        let req =
            AuthorizationRequest::new("mcp", "mount").with_modifiers(json!({ "name": "anything" }));
        match s.check(&p, &req, Tier::External).await {
            CheckOutcome::Deny { .. } => {} // empty allow = deny everything
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn namespace_wildcard_constraint_applies_to_all_actions() {
        let s = store_with(
            r#"
            user_id: k
            principals:
              - agent_id: agent
                scopes: [net.http]
                tier: external
                constraints:
                  - verb: net.*
                    modifier: host
                    match_kind: host_suffix
                    allow: [github.com]
            "#,
        );
        let p = agent_of(&s).await;
        let req =
            AuthorizationRequest::new("net", "http").with_modifiers(json!({ "host": "evil.com" }));
        match s.check(&p, &req, Tier::External).await {
            CheckOutcome::Deny { .. } => {} // net.* wildcard covers net.http
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    // ── Property tests: path_is_covered ───────────────────────────────
    //
    // The path allowlist is the filesystem access boundary. The risk it
    // must rule out is segment-boundary prefix confusion: an allowlist of
    // "/home/u" must cover "/home/u/x" but never the sibling "/home/user".
    // These assert that boundary, plus empty-list fail-closed, trailing-
    // slash normalization, and monotonicity, for arbitrary paths.

    use proptest::prelude::*;

    fn covered(path: &str, allow: &[String]) -> bool {
        ConfigIdentityStore::path_is_covered(path, allow)
    }

    /// Absolute paths of 1–4 lowercase segments, no trailing slash, no `..`.
    fn abs_path() -> impl Strategy<Value = String> {
        proptest::collection::vec("[a-z]{1,5}", 1..4)
            .prop_map(|segs| format!("/{}", segs.join("/")))
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// Fail closed: an empty allowlist never covers anything.
        #[test]
        fn empty_allowlist_denies_all(path in abs_path()) {
            prop_assert!(!covered(&path, &[]));
        }

        /// An entry covers itself and any descendant beneath a `/` boundary.
        #[test]
        fn entry_covers_itself_and_descendants(entry in abs_path(), child in "[a-z]{1,5}") {
            let allow = [entry.clone()];
            let descendant = format!("{entry}/{child}");
            prop_assert!(covered(&entry, &allow), "entry didn't cover itself: {}", entry);
            prop_assert!(covered(&descendant, &allow), "didn't cover descendant: {}", descendant);
        }

        /// Segment-boundary safety: extending the final segment (no `/`)
        /// yields a sibling that must NOT be covered ("/a/b" ⊉ "/a/bc").
        #[test]
        fn sibling_sharing_prefix_is_not_covered(entry in abs_path(), tail in "[a-z]{1,5}") {
            let allow = [entry.clone()];
            let sibling = format!("{entry}{tail}");
            prop_assert!(!covered(&sibling, &allow), "covered a sibling: {}", sibling);
        }

        /// A trailing slash on an allowlist entry is normalized away — it
        /// decides coverage identically to the same entry without one.
        #[test]
        fn trailing_slash_entry_is_equivalent(entry in abs_path(), child in "[a-z]{1,5}") {
            let with = vec![format!("{entry}/")];
            let without = vec![entry.clone()];
            for p in [entry.clone(), format!("{entry}/{child}"), format!("{entry}{child}")] {
                prop_assert_eq!(covered(&p, &with), covered(&p, &without), "mismatch for {}", p);
            }
        }

        /// Monotonic: adding more entries never revokes existing coverage.
        #[test]
        fn adding_entries_preserves_coverage(
            entry in abs_path(),
            child in "[a-z]{1,5}",
            extra in proptest::collection::vec(abs_path(), 0..3),
        ) {
            let path = format!("{entry}/{child}");
            prop_assume!(covered(&path, std::slice::from_ref(&entry)));
            let mut bigger = extra;
            bigger.push(entry);
            prop_assert!(covered(&path, &bigger));
        }
    }
}
