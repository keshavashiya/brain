//! Qwen OAuth provider — device-code flow + OpenAI-compatible completions.
//!
//! Authenticates via `chat.qwen.ai` OAuth2 device-code flow, stores tokens
//! in the credential vault, and proxies LLM calls to the DashScope
//! OpenAI-compatible endpoint.

use std::pin::Pin;
use std::sync::Arc;

use chrono::{Duration, Utc};
use futures::Stream;
use tokio::sync::RwLock;

use vault::{CredentialValue, CredentialVault, InjectionShape};

use crate::llm::{LlmError, LlmProvider, Message, Response, ResponseChunk};
use crate::oauth::{
    generate_code_challenge, generate_code_verifier, AuthChallenge, OAuthProvider, PollResult,
    TokenSet,
};
use crate::OpenAiProvider;

const DEVICE_CODE_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/device/code";
const TOKEN_URL: &str = "https://chat.qwen.ai/api/v1/oauth2/token";
const CLIENT_ID: &str = "f0304373b74a44d2b584a3fb70ca9e56";
const SCOPE: &str = "openid profile email model.completion";
const GRANT_TYPE_DEVICE: &str = "urn:ietf:params:oauth:grant-type:device_code";
const DEFAULT_ENDPOINT: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

const VAULT_TOOL: &str = "qwen-oauth";
const VAULT_KEY: &str = "credentials";

pub struct QwenOAuthProvider {
    http: reqwest::Client,
    vault: Arc<dyn CredentialVault>,
    model: String,
    temperature: f64,
    max_tokens: Option<i32>,
    cached: RwLock<Option<TokenSet>>,
}

impl QwenOAuthProvider {
    pub fn new(
        vault: Arc<dyn CredentialVault>,
        model: &str,
        temperature: f64,
        max_tokens: Option<i32>,
    ) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(brain_core::timeouts::LLM_GENERATE)
            .build()
            .map_err(|e| LlmError::ProviderUnavailable(format!("HTTP client: {e}")))?;
        Ok(Self {
            http,
            vault,
            model: model.into(),
            temperature,
            max_tokens,
            cached: RwLock::new(None),
        })
    }

    fn endpoint(resource_url: Option<&str>) -> String {
        let base = resource_url.unwrap_or(DEFAULT_ENDPOINT);
        let normalized = if base.starts_with("http") {
            base.to_string()
        } else {
            format!("https://{base}")
        };
        if normalized.ends_with("/v1") {
            normalized
        } else {
            format!("{normalized}/v1")
        }
    }

    async fn get_or_refresh_token(&self) -> Result<(String, String), LlmError> {
        {
            let guard = self.cached.read().await;
            if let Some(ts) = guard.as_ref() {
                if !ts.is_expired() {
                    return Ok((
                        ts.access_token.clone(),
                        Self::endpoint(ts.resource_url.as_deref()),
                    ));
                }
            }
        }
        let stored = self.load_tokens().await?;
        if let Some(ts) = stored {
            if !ts.is_expired() {
                *self.cached.write().await = Some(ts.clone());
                return Ok((
                    ts.access_token.clone(),
                    Self::endpoint(ts.resource_url.as_deref()),
                ));
            }
            if ts.refresh_token.is_some() {
                let refreshed = self.refresh_tokens().await?;
                return Ok((
                    refreshed.access_token.clone(),
                    Self::endpoint(refreshed.resource_url.as_deref()),
                ));
            }
        }
        Err(LlmError::ProviderUnavailable(
            "No valid Qwen OAuth token — run `brain auth qwen` to authenticate".into(),
        ))
    }

    async fn force_refresh(&self) -> Result<(String, String), LlmError> {
        let refreshed = self.refresh_tokens().await?;
        Ok((
            refreshed.access_token.clone(),
            Self::endpoint(refreshed.resource_url.as_deref()),
        ))
    }

    fn make_inner(&self, endpoint: &str, token: &str) -> Result<OpenAiProvider, LlmError> {
        OpenAiProvider::new(
            endpoint,
            Some(token),
            &self.model,
            self.temperature,
            self.max_tokens,
        )
    }
}

#[async_trait::async_trait]
impl OAuthProvider for QwenOAuthProvider {
    fn provider_name(&self) -> &str {
        "qwen-oauth"
    }

    async fn begin_device_auth(&self) -> Result<AuthChallenge, LlmError> {
        let verifier = generate_code_verifier();
        let challenge = generate_code_challenge(&verifier);

        let params = [
            ("client_id", CLIENT_ID),
            ("scope", SCOPE),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
        ];

        let resp = self
            .http
            .post(DEVICE_CODE_URL)
            .header("Accept", "application/json")
            .form(&params)
            .send()
            .await
            .map_err(LlmError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status,
                message: format!("device auth failed: {body}"),
            });
        }

        #[derive(serde::Deserialize)]
        struct DeviceResp {
            device_code: String,
            user_code: String,
            verification_uri: String,
            #[serde(default)]
            verification_uri_complete: String,
            expires_in: u64,
        }

        let dr: DeviceResp = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidFormat(format!("device auth json: {e}")))?;

        // Stash verifier alongside device_code so poll can use it.
        // We encode it into the device_code field as `{code}:{verifier}`.
        Ok(AuthChallenge {
            verification_uri: dr.verification_uri,
            verification_uri_complete: dr.verification_uri_complete,
            user_code: dr.user_code,
            device_code: format!("{}:{}", dr.device_code, verifier),
            expires_in: dr.expires_in,
        })
    }

    async fn poll_device_token(
        &self,
        device_code: &str,
        code_verifier: &str,
    ) -> Result<PollResult, LlmError> {
        let params = [
            ("grant_type", GRANT_TYPE_DEVICE),
            ("client_id", CLIENT_ID),
            ("device_code", device_code),
            ("code_verifier", code_verifier),
        ];

        let resp = self
            .http
            .post(TOKEN_URL)
            .header("Accept", "application/json")
            .form(&params)
            .send()
            .await
            .map_err(LlmError::Http)?;

        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();

        if status == 400 && body.contains("authorization_pending") {
            return Ok(PollResult::Pending);
        }
        if status == 429 || body.contains("slow_down") {
            return Ok(PollResult::SlowDown);
        }
        if status >= 400 {
            return Err(LlmError::Api {
                status,
                message: format!("token poll: {body}"),
            });
        }

        #[derive(serde::Deserialize)]
        struct TokenResp {
            access_token: Option<String>,
            refresh_token: Option<String>,
            expires_in: Option<i64>,
            resource_url: Option<String>,
        }

        let tr: TokenResp = serde_json::from_str(&body)
            .map_err(|e| LlmError::InvalidFormat(format!("token json: {e}")))?;

        match tr.access_token {
            Some(at) if !at.is_empty() => {
                let ts = TokenSet {
                    access_token: at,
                    refresh_token: tr.refresh_token,
                    expiry_date: tr.expires_in.map(|s| Utc::now() + Duration::seconds(s)),
                    resource_url: tr.resource_url,
                };
                self.save_tokens(&ts).await?;
                Ok(PollResult::Ready(ts))
            }
            _ => Ok(PollResult::Pending),
        }
    }

    async fn refresh_tokens(&self) -> Result<TokenSet, LlmError> {
        let stored = self
            .load_tokens()
            .await?
            .ok_or_else(|| LlmError::ProviderUnavailable("no stored tokens".into()))?;
        let refresh = stored.refresh_token.as_deref().ok_or_else(|| {
            LlmError::ProviderUnavailable(
                "no refresh token — run `brain auth qwen` to re-authenticate".into(),
            )
        })?;

        let params = [
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh),
            ("client_id", CLIENT_ID),
        ];

        let resp = self
            .http
            .post(TOKEN_URL)
            .header("Accept", "application/json")
            .form(&params)
            .send()
            .await
            .map_err(LlmError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            if status == 400 || status == 401 {
                self.clear_tokens().await.ok();
                return Err(LlmError::ProviderUnavailable(
                    "refresh token expired — run `brain auth qwen` to re-authenticate".into(),
                ));
            }
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status,
                message: format!("refresh: {body}"),
            });
        }

        #[derive(serde::Deserialize)]
        struct RefreshResp {
            access_token: String,
            expires_in: Option<i64>,
            refresh_token: Option<String>,
            resource_url: Option<String>,
        }

        let rr: RefreshResp = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidFormat(format!("refresh json: {e}")))?;

        let ts = TokenSet {
            access_token: rr.access_token,
            refresh_token: rr.refresh_token.or(stored.refresh_token),
            expiry_date: rr.expires_in.map(|s| Utc::now() + Duration::seconds(s)),
            resource_url: rr.resource_url.or(stored.resource_url),
        };
        self.save_tokens(&ts).await?;
        Ok(ts)
    }

    async fn load_tokens(&self) -> Result<Option<TokenSet>, LlmError> {
        match self.vault.get(VAULT_TOOL, VAULT_KEY).await {
            Ok(cred) => {
                let ts: TokenSet = serde_json::from_str(cred.value.as_str())
                    .map_err(|e| LlmError::InvalidFormat(format!("vault token parse: {e}")))?;
                Ok(Some(ts))
            }
            Err(vault::VaultError::NotFound { .. }) => Ok(None),
            Err(e) => Err(LlmError::ProviderUnavailable(format!("vault read: {e}"))),
        }
    }

    async fn save_tokens(&self, tokens: &TokenSet) -> Result<(), LlmError> {
        let json = serde_json::to_string(tokens)
            .map_err(|e| LlmError::InvalidFormat(format!("serialize tokens: {e}")))?;
        self.vault
            .store(
                VAULT_TOOL,
                VAULT_KEY,
                CredentialValue::new(json),
                InjectionShape::EnvVar {
                    name: "QWEN_OAUTH_CREDENTIALS".into(),
                },
            )
            .await
            .map_err(|e| LlmError::ProviderUnavailable(format!("vault write: {e}")))
    }

    async fn clear_tokens(&self) -> Result<(), LlmError> {
        self.vault
            .delete(VAULT_TOOL, VAULT_KEY)
            .await
            .map_err(|e| LlmError::ProviderUnavailable(format!("vault delete: {e}")))
    }
}

#[async_trait::async_trait]
impl LlmProvider for QwenOAuthProvider {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let (token, endpoint) = self.get_or_refresh_token().await?;
        let inner = self.make_inner(&endpoint, &token)?;
        match inner.generate(messages).await {
            Err(LlmError::Api { status: 401, .. }) => {
                let (token, endpoint) = self.force_refresh().await?;
                let inner = self.make_inner(&endpoint, &token)?;
                inner.generate(messages).await
            }
            other => other,
        }
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        let (token, endpoint) = self.get_or_refresh_token().await?;
        let inner = self.make_inner(&endpoint, &token)?;
        inner.generate_stream(messages).await
    }

    async fn health_check(&self) -> bool {
        self.get_or_refresh_token().await.is_ok()
    }

    fn name(&self) -> &str {
        "qwen-oauth"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_defaults() {
        assert_eq!(
            QwenOAuthProvider::endpoint(None),
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        );
    }

    #[test]
    fn endpoint_normalizes_protocol() {
        assert_eq!(
            QwenOAuthProvider::endpoint(Some("dashscope.aliyuncs.com/compatible-mode")),
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        );
    }

    #[test]
    fn endpoint_preserves_existing_v1() {
        let url = "https://custom.host.com/v1";
        assert_eq!(QwenOAuthProvider::endpoint(Some(url)), url);
    }
}
