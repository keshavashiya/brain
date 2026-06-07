//! Injection shapes and credential value wrapper.
//!
//! The vault never returns raw `String` values directly — callers receive an
//! `InjectedCredential` that describes *how* the credential should be applied
//! (env var, HTTP header, positional arg). This keeps credential handling
//! at the site of use and avoids accidentally logging the value.

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

/// How a credential should be injected into the consuming call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "shape")]
pub enum InjectionShape {
    /// Set as environment variable.
    EnvVar { name: String },
    /// Set as HTTP header.
    Header { name: String },
    /// Substitute as positional argument at index.
    Arg { position: usize },
}

/// A credential value. Zeroized on drop. `Debug` does not print the value.
#[derive(Clone, Zeroize)]
#[zeroize(drop)]
pub struct CredentialValue(String);

impl CredentialValue {
    pub fn new(value: String) -> Self {
        Self(value)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for CredentialValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CredentialValue(<redacted, {} bytes>)", self.0.len())
    }
}

impl From<String> for CredentialValue {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// A credential prepared for injection. Carries the value plus its shape.
#[derive(Debug, Clone)]
pub struct InjectedCredential {
    pub shape: InjectionShape,
    pub value: CredentialValue,
}

impl InjectedCredential {
    pub fn env(name: impl Into<String>, value: impl Into<CredentialValue>) -> Self {
        Self {
            shape: InjectionShape::EnvVar { name: name.into() },
            value: value.into(),
        }
    }

    pub fn header(name: impl Into<String>, value: impl Into<CredentialValue>) -> Self {
        Self {
            shape: InjectionShape::Header { name: name.into() },
            value: value.into(),
        }
    }

    pub fn arg(position: usize, value: impl Into<CredentialValue>) -> Self {
        Self {
            shape: InjectionShape::Arg { position },
            value: value.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn injection_shape() -> impl Strategy<Value = InjectionShape> {
        prop_oneof![
            ".*".prop_map(|name| InjectionShape::EnvVar { name }),
            ".*".prop_map(|name| InjectionShape::Header { name }),
            any::<usize>().prop_map(|position| InjectionShape::Arg { position }),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// Security invariant: a credential's secret value must never appear
        /// in its Debug rendering — only a byte count. Stated as an exact
        /// equality so it can't leak even a prefix.
        #[test]
        fn debug_never_prints_secret(s in ".*") {
            let cred = CredentialValue::new(s.clone());
            prop_assert_eq!(
                format!("{cred:?}"),
                format!("CredentialValue(<redacted, {} bytes>)", s.len())
            );
            // len()/is_empty() track the wrapped value (byte length).
            prop_assert_eq!(cred.len(), s.len());
            prop_assert_eq!(cred.is_empty(), s.is_empty());
        }

        /// InjectionShape is persisted as JSON (the `.meta` sidecar and the
        /// keychain/secret-service payloads). Serializing then deserializing
        /// must round-trip exactly, including the internal `shape` tag, for
        /// any field contents.
        #[test]
        fn injection_shape_json_round_trips(shape in injection_shape()) {
            let json = serde_json::to_string(&shape).unwrap();
            let back: InjectionShape = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(back, shape);
        }
    }
}

/// Metadata returned by `list` / `get` (without the value). Safe to log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CredentialMetadata {
    pub tool: String,
    pub key: String,
    pub backend: String, // "keychain" | "secret-service" | "file"
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub shape: InjectionShape,
}
