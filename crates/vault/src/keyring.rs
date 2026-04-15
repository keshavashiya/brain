//! Linux `secret-service` backend (GNOME Keyring / KDE Wallet).
//!
//! Collection = default collection; items are identified by attributes
//! `{brain="1", tool=<tool>, key=<key>}` and labelled `brain:<tool>:<key>`.

use std::collections::HashMap;

use chrono::Utc;
use secret_service::{EncryptionType, SecretService};

use crate::inject::{CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape};
use crate::vault::VaultError;

#[derive(Default)]
pub struct SecretServiceBackend;

impl SecretServiceBackend {
    pub fn new() -> Self {
        Self
    }

    fn label(tool: &str, key: &str) -> String {
        format!("brain:{tool}:{key}")
    }

    fn attrs<'a>(tool: &'a str, key: &'a str) -> HashMap<&'a str, &'a str> {
        let mut m = HashMap::with_capacity(3);
        m.insert("brain", "1");
        m.insert("tool", tool);
        m.insert("key", key);
        m
    }

    async fn connect(&self) -> Result<SecretService<'_>, VaultError> {
        SecretService::connect(EncryptionType::Dh)
            .await
            .map_err(|e| VaultError::BackendUnavailable(format!("secret-service: {e}")))
    }

    fn encode_shape(shape: &InjectionShape) -> String {
        serde_json::to_string(shape).unwrap_or_else(|_| "{}".to_string())
    }

    fn decode_shape(raw: &str) -> Result<InjectionShape, VaultError> {
        serde_json::from_str(raw).map_err(|e| VaultError::InvalidData(format!("shape parse: {e}")))
    }

    pub async fn store(
        &self,
        tool: &str,
        key: &str,
        value: CredentialValue,
        shape: InjectionShape,
    ) -> Result<(), VaultError> {
        let ss = self.connect().await?;
        let collection = ss
            .get_default_collection()
            .await
            .map_err(|e| VaultError::Backend(format!("get collection: {e}")))?;

        let shape_json = Self::encode_shape(&shape);
        let created_at = Utc::now().to_rfc3339();
        let mut attrs = Self::attrs(tool, key);
        attrs.insert("shape", shape_json.as_str());
        attrs.insert("created_at", created_at.as_str());

        collection
            .create_item(
                &Self::label(tool, key),
                attrs,
                value.as_str().as_bytes(),
                true,
                "text/plain",
            )
            .await
            .map_err(|e| VaultError::Backend(format!("create_item: {e}")))?;
        Ok(())
    }

    pub async fn get(&self, tool: &str, key: &str) -> Result<InjectedCredential, VaultError> {
        let ss = self.connect().await?;
        let attrs = Self::attrs(tool, key);
        let items = ss
            .search_items(attrs)
            .await
            .map_err(|e| VaultError::Backend(format!("search: {e}")))?;

        let item = items
            .unlocked
            .into_iter()
            .chain(items.locked.into_iter())
            .next()
            .ok_or_else(|| VaultError::NotFound {
                tool: tool.to_string(),
                key: key.to_string(),
            })?;

        item.unlock()
            .await
            .map_err(|e| VaultError::Backend(format!("unlock: {e}")))?;

        let secret = item
            .get_secret()
            .await
            .map_err(|e| VaultError::Backend(format!("get_secret: {e}")))?;
        let value =
            String::from_utf8(secret).map_err(|e| VaultError::InvalidData(format!("utf8: {e}")))?;

        let item_attrs = item
            .get_attributes()
            .await
            .map_err(|e| VaultError::Backend(format!("get_attributes: {e}")))?;
        let shape = item_attrs
            .get("shape")
            .map(|s| Self::decode_shape(s))
            .transpose()?
            .ok_or_else(|| VaultError::InvalidData("missing shape attribute".into()))?;

        Ok(InjectedCredential {
            shape,
            value: CredentialValue::new(value),
        })
    }

    pub async fn delete(&self, tool: &str, key: &str) -> Result<(), VaultError> {
        let ss = self.connect().await?;
        let attrs = Self::attrs(tool, key);
        let items = ss
            .search_items(attrs)
            .await
            .map_err(|e| VaultError::Backend(format!("search: {e}")))?;

        let item = items
            .unlocked
            .into_iter()
            .chain(items.locked.into_iter())
            .next()
            .ok_or_else(|| VaultError::NotFound {
                tool: tool.to_string(),
                key: key.to_string(),
            })?;

        item.delete()
            .await
            .map_err(|e| VaultError::Backend(format!("delete: {e}")))
    }

    pub async fn list(&self, tool: Option<&str>) -> Result<Vec<CredentialMetadata>, VaultError> {
        let ss = self.connect().await?;
        let mut attrs = HashMap::new();
        attrs.insert("brain", "1");
        if let Some(t) = tool {
            attrs.insert("tool", t);
        }

        let items = ss
            .search_items(attrs)
            .await
            .map_err(|e| VaultError::Backend(format!("search: {e}")))?;

        let mut out = Vec::new();
        for item in items.unlocked.into_iter().chain(items.locked.into_iter()) {
            let item_attrs = match item.get_attributes().await {
                Ok(m) => m,
                Err(err) => {
                    tracing::warn!(error = %err, "vault/secret-service: skip item with no attrs");
                    continue;
                }
            };
            let tool_name = item_attrs.get("tool").cloned().unwrap_or_default();
            let key_name = item_attrs.get("key").cloned().unwrap_or_default();
            let created_at = item_attrs.get("created_at").cloned().unwrap_or_default();
            let shape = item_attrs
                .get("shape")
                .map(|s| Self::decode_shape(s))
                .transpose()
                .ok()
                .flatten()
                .unwrap_or(InjectionShape::EnvVar {
                    name: "UNKNOWN".to_string(),
                });
            out.push(CredentialMetadata {
                tool: tool_name,
                key: key_name,
                backend: "secret-service".to_string(),
                created_at,
                last_used_at: None,
                shape,
            });
        }
        Ok(out)
    }
}
