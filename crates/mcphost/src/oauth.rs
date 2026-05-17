//! OAuth 2.1 + PKCE for HTTP MCP transports.
//!
//! Brain delegates the OAuth state machine itself to `rmcp::transport::auth`
//! (which implements the MCP-spec 2025-11-25 flow: PRM discovery → AS metadata
//! → CIMD/DCR → PKCE S256 → token exchange with RFC 8707 `resource`). This
//! module provides:
//!
//! 1. A [`VaultCredentialStore`] bridging rmcp's [`CredentialStore`] trait to
//!    `brainos-vault::CredentialVault`, so tokens persist across restarts in
//!    the OS keychain or the encrypted-file backend.
//! 2. Helpers that build a ready-to-mount [`AuthorizationManager`] from the
//!    persisted credentials for a given server, validating the access
//!    token's `aud` claim against the configured RFC 8707 resource at
//!    mount time (CVE-2025-6514 / confused-deputy mitigation).
//!
//! The interactive bootstrap (loopback HTTP server + browser launch to drive
//! the auth-code grant) is intentionally out of scope here — that's a CLI-
//! level concern. Once tokens land in the vault, [`AuthorizationManager`]
//! refreshes them automatically.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use observe::{BrainEvent, Observer};
use rmcp::transport::auth::{AuthError, AuthorizationManager, CredentialStore, StoredCredentials};
use tracing::warn;
use uuid::Uuid;
use vault::{CredentialValue, CredentialVault, InjectionShape, VaultError};

use crate::aud_check::{validate_token_aud, AudCheckOutcome};
use crate::error::McpHostError;

/// Vault tool namespace used for OAuth credentials. Keying is
/// `(tool = "mcphost.oauth", key = server_name)`.
pub const VAULT_TOOL: &str = "mcphost.oauth";

/// Vault-backed [`CredentialStore`] for a single MCP server. Pure
/// passthrough over [`CredentialVault`]; the audience-claim check
/// lives in [`manager_from_vault`] so it can use rmcp's
/// `AuthorizationManager::get_access_token` public API rather than
/// reaching into `oauth2` internals from inside the [`CredentialStore`]
/// implementation.
pub struct VaultCredentialStore {
    vault: Arc<dyn CredentialVault>,
    server: String,
}

impl VaultCredentialStore {
    pub fn new(vault: Arc<dyn CredentialVault>, server: impl Into<String>) -> Self {
        Self {
            vault,
            server: server.into(),
        }
    }
}

#[async_trait]
impl CredentialStore for VaultCredentialStore {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        match self.vault.get(VAULT_TOOL, &self.server).await {
            Ok(injected) => {
                let raw = injected.value.as_str();
                let parsed: StoredCredentials = serde_json::from_str(raw)
                    .map_err(|e| AuthError::InternalError(format!("vault decode: {e}")))?;
                Ok(Some(parsed))
            }
            Err(VaultError::NotFound { .. }) => Ok(None),
            Err(e) => Err(AuthError::InternalError(format!("vault load: {e}"))),
        }
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let json = serde_json::to_string(&credentials)
            .map_err(|e| AuthError::InternalError(format!("vault encode: {e}")))?;
        self.vault
            .store(
                VAULT_TOOL,
                &self.server,
                CredentialValue::new(json),
                InjectionShape::Header {
                    name: "Authorization".into(),
                },
            )
            .await
            .map_err(|e| AuthError::InternalError(format!("vault save: {e}")))
    }

    async fn clear(&self) -> Result<(), AuthError> {
        match self.vault.delete(VAULT_TOOL, &self.server).await {
            Ok(()) | Err(VaultError::NotFound { .. }) => Ok(()),
            Err(e) => Err(AuthError::InternalError(format!("vault clear: {e}"))),
        }
    }
}

/// Build an [`AuthorizationManager`] for `base_url` (the MCP server
/// URL), wired to a vault-backed credential store and pre-loaded with
/// any persisted tokens. Caller passes the resulting manager to
/// [`rmcp::transport::auth::AuthClient::new`].
///
/// `expected_resource` is the RFC 8707 resource indicator the server
/// was configured with — typically the same string as `base_url` for
/// vanilla MCP deployments, but kept separate so deployments where the
/// AS issues tokens with a distinct resource id can still validate.
///
/// `observer`, when wired, receives a `BrainEvent::Error { source:
/// "mcphost.oauth" }` whenever the persisted token's JWT `aud` claim
/// does not include `expected_resource`. The mount fails closed in
/// that case — see [`AudCheckOutcome::Mismatch`].
///
/// Returns `Err(McpHostError::Auth("no persisted credentials …"))` when
/// the vault has nothing for this server — callers should run the
/// interactive bootstrap (out of scope here) before retrying the
/// mount.
pub async fn manager_from_vault(
    base_url: &str,
    server: &str,
    expected_resource: &str,
    vault: Arc<dyn CredentialVault>,
    observer: Option<Arc<dyn Observer>>,
) -> Result<AuthorizationManager, McpHostError> {
    let mut manager = AuthorizationManager::new(base_url)
        .await
        .map_err(|e| McpHostError::Auth(format!("authorization manager: {e}")))?;
    manager.set_credential_store(VaultCredentialStore::new(vault, server));
    let initialized = manager
        .initialize_from_store()
        .await
        .map_err(|e| McpHostError::Auth(format!("initialize from vault: {e}")))?;
    if !initialized {
        return Err(McpHostError::Auth(format!(
            "no persisted OAuth credentials for server '{server}' — run the auth bootstrap first"
        )));
    }

    // Audience-claim check. The token is now loaded into the manager;
    // pull it back out and validate `aud` against the configured
    // resource. JWT with a mismatched `aud` fails closed; opaque
    // tokens and missing-aud JWTs pass through (logged for visibility).
    let access_token = manager
        .get_access_token()
        .await
        .map_err(|e| McpHostError::Auth(format!("read persisted access token: {e}")))?;

    match validate_token_aud(&access_token, expected_resource) {
        AudCheckOutcome::Match => {}
        AudCheckOutcome::OpaqueToken => {
            // Common case — opaque OAuth 2.1 access tokens. Resource
            // indicator already enforced at request time by the AS.
        }
        AudCheckOutcome::MissingAud => {
            warn!(
                server = %server,
                "OAuth JWT access token has no `aud` claim — RFC 9068 says it should, allowing through",
            );
        }
        AudCheckOutcome::Mismatch { found, expected: _ } => {
            emit_aud_mismatch(server, expected_resource, &found, observer.as_ref()).await;
            return Err(McpHostError::Auth(format!(
                "OAuth aud mismatch for server '{server}': token aud {found:?} does not include configured resource '{expected_resource}' (CVE-2025-6514 / confused-deputy mitigation)"
            )));
        }
    }

    Ok(manager)
}

async fn emit_aud_mismatch(
    server: &str,
    expected_resource: &str,
    found: &[String],
    observer: Option<&Arc<dyn Observer>>,
) {
    let message = format!(
        "OAuth aud mismatch for server '{server}': token aud={found:?} expected '{expected_resource}' (CVE-2025-6514 / confused-deputy mitigation)"
    );
    warn!(server = %server, "{message}");
    if let Some(observer) = observer {
        let _ = observer
            .publish(BrainEvent::Error {
                id: Uuid::new_v4(),
                source: "mcphost.oauth".to_string(),
                message,
                ts: Utc::now(),
            })
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::transport::auth::StoredCredentials;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use vault::{BackendKind, CredentialMetadata, CredentialVault, InjectedCredential, VaultError};

    /// In-memory `CredentialVault` for unit tests — does not touch the OS
    /// keychain or filesystem.
    #[derive(Default)]
    struct MemVault {
        items: Mutex<HashMap<(String, String), (CredentialValue, InjectionShape)>>,
    }

    #[async_trait]
    impl CredentialVault for MemVault {
        async fn store(
            &self,
            tool: &str,
            key: &str,
            value: CredentialValue,
            shape: InjectionShape,
        ) -> Result<(), VaultError> {
            self.items
                .lock()
                .unwrap()
                .insert((tool.into(), key.into()), (value, shape));
            Ok(())
        }
        async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError> {
            let guard = self.items.lock().unwrap();
            let (value, shape) =
                guard
                    .get(&(tool.into(), key.into()))
                    .ok_or_else(|| VaultError::NotFound {
                        tool: tool.into(),
                        key: key.into(),
                    })?;
            Ok(InjectedCredential {
                shape: shape.clone(),
                value: value.clone(),
            })
        }
        async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError> {
            self.items
                .lock()
                .unwrap()
                .remove(&(tool.into(), key.into()));
            Ok(())
        }
        async fn list(&self, _tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError> {
            Ok(Vec::new())
        }
        fn backend_kind(&self) -> BackendKind {
            BackendKind::File
        }
    }

    #[tokio::test]
    async fn vault_store_round_trip() {
        let vault: Arc<dyn CredentialVault> = Arc::new(MemVault::default());
        let store = VaultCredentialStore::new(vault, "fs");

        assert!(store.load().await.unwrap().is_none());

        let creds = StoredCredentials::new("client-abc".into(), None, vec![], None);
        store.save(creds).await.unwrap();
        let loaded = store.load().await.unwrap().expect("must be present");
        assert_eq!(loaded.client_id, "client-abc");

        store.clear().await.unwrap();
        assert!(store.load().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn vault_clear_when_missing_ok() {
        let vault: Arc<dyn CredentialVault> = Arc::new(MemVault::default());
        let store = VaultCredentialStore::new(vault, "missing");
        // Idempotent: clearing a non-existent entry is fine.
        store.clear().await.unwrap();
    }
}
