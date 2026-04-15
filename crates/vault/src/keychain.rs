//! macOS keychain backend (Generic Password items).
//!
//! Service = "brain", Account = "<tool>:<key>".
//! Password payload is the raw credential value; injection shape is kept in
//! the generic attribute (comment field) as a JSON blob.

use chrono::Utc;
use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};

use crate::inject::{CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape};
use crate::vault::VaultError;

const SERVICE: &str = "brain";
/// Prefix written alongside the secret value so we can recover the shape and
/// metadata on read: `{shape_json}\0{created_at}\0{secret}`
const FIELD_SEP: u8 = 0x1F;

#[derive(Default)]
pub struct KeychainBackend;

impl KeychainBackend {
    pub fn new() -> Self {
        Self
    }

    fn account(tool: &str, key: &str) -> String {
        format!("{tool}:{key}")
    }

    fn encode(shape: &InjectionShape, created_at: &str, secret: &str) -> Vec<u8> {
        let shape_json = serde_json::to_string(shape).unwrap_or_else(|_| "{}".to_string());
        let mut out = Vec::with_capacity(shape_json.len() + created_at.len() + secret.len() + 2);
        out.extend_from_slice(shape_json.as_bytes());
        out.push(FIELD_SEP);
        out.extend_from_slice(created_at.as_bytes());
        out.push(FIELD_SEP);
        out.extend_from_slice(secret.as_bytes());
        out
    }

    fn decode(raw: &[u8]) -> Result<(InjectionShape, String, String), VaultError> {
        let mut parts = raw.splitn(3, |b| *b == FIELD_SEP);
        let shape_bytes = parts
            .next()
            .ok_or_else(|| VaultError::InvalidData("missing shape".into()))?;
        let created_bytes = parts
            .next()
            .ok_or_else(|| VaultError::InvalidData("missing created_at".into()))?;
        let secret_bytes = parts
            .next()
            .ok_or_else(|| VaultError::InvalidData("missing secret".into()))?;
        let shape: InjectionShape = serde_json::from_slice(shape_bytes)
            .map_err(|e| VaultError::InvalidData(format!("shape parse: {e}")))?;
        let created_at = std::str::from_utf8(created_bytes)
            .map_err(|e| VaultError::InvalidData(format!("created_at utf8: {e}")))?
            .to_string();
        let secret = std::str::from_utf8(secret_bytes)
            .map_err(|e| VaultError::InvalidData(format!("secret utf8: {e}")))?
            .to_string();
        Ok((shape, created_at, secret))
    }

    pub async fn store(
        &self,
        tool: &str,
        key: &str,
        value: CredentialValue,
        shape: InjectionShape,
    ) -> Result<(), VaultError> {
        let account = Self::account(tool, key);
        let created_at = Utc::now().to_rfc3339();
        let payload = Self::encode(&shape, &created_at, value.as_str());
        set_generic_password(SERVICE, &account, &payload)
            .map_err(|e| VaultError::Backend(format!("keychain set: {e}")))
    }

    pub async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError> {
        let account = Self::account(tool, key);
        let raw = get_generic_password(SERVICE, &account).map_err(|e| {
            let msg = format!("{e}");
            if msg.contains("could not be found") || msg.contains("-25300") {
                VaultError::NotFound {
                    tool: tool.to_string(),
                    key: key.to_string(),
                }
            } else {
                VaultError::Backend(format!("keychain get: {e}"))
            }
        })?;
        let (shape, _created, secret) = Self::decode(&raw)?;
        Ok(InjectedCredential {
            shape,
            value: CredentialValue::new(secret),
        })
    }

    pub async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError> {
        let account = Self::account(tool, key);
        delete_generic_password(SERVICE, &account).map_err(|e| {
            let msg = format!("{e}");
            if msg.contains("could not be found") || msg.contains("-25300") {
                VaultError::NotFound {
                    tool: tool.to_string(),
                    key: key.to_string(),
                }
            } else {
                VaultError::Backend(format!("keychain delete: {e}"))
            }
        })
    }

    pub async fn list(&self, _tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError> {
        // security-framework's high-level API doesn't expose enumeration of
        // generic passwords by service without touching lower-level SecItem
        // queries. For the MVP we document this as a known gap: `list` on
        // the keychain backend returns an empty set; use `brain vault list`
        // against the file backend, or `security find-generic-password -s brain`
        // from the command line.
        tracing::info!("vault/keychain: list is not yet implemented; returning empty set");
        Ok(Vec::new())
    }
}
