//! CLI commands for OAuth provider authentication.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Subcommand;
use cortex::oauth::{OAuthProvider, PollResult};
use cortex::qwen::QwenOAuthProvider;
use vault::vault::{resolve_backend, DefaultVault};
use vault::VaultConfig;

#[derive(Subcommand)]
pub(crate) enum AuthAction {
    /// Authenticate with a provider via device-code flow.
    Login {
        /// Provider name (e.g. qwen).
        provider: String,
        /// Model to use with this provider.
        #[arg(long, default_value = "qwen3-coder")]
        model: String,
    },

    /// Show token status for a provider.
    Status {
        /// Provider name.
        provider: String,
    },

    /// Clear stored tokens for a provider.
    Logout {
        /// Provider name.
        provider: String,
    },
}

pub(crate) async fn cmd_auth(_config: &brain_core::BrainConfig, action: AuthAction) -> Result<()> {
    match action {
        AuthAction::Login { provider, model } => cmd_login(&provider, &model).await,
        AuthAction::Status { provider } => cmd_status(&provider).await,
        AuthAction::Logout { provider } => cmd_logout(&provider).await,
    }
}

fn build_qwen(vault: Arc<dyn vault::CredentialVault>) -> Result<QwenOAuthProvider> {
    QwenOAuthProvider::new(vault, "qwen3-coder", 0.7, Some(4096)).map_err(anyhow::Error::from)
}

fn make_vault() -> Result<Arc<DefaultVault>> {
    let config = VaultConfig::default();
    let backend = resolve_backend(&config)?;
    Ok(Arc::new(DefaultVault::new(backend)))
}

async fn cmd_login(provider: &str, _model: &str) -> Result<()> {
    match provider {
        "qwen" => {}
        other => return Err(anyhow!("unknown provider `{other}` — supported: qwen")),
    }

    let vault = make_vault()?;
    let qwen = build_qwen(vault)?;

    eprintln!("Starting Qwen device-code authentication...\n");

    let challenge = qwen
        .begin_device_auth()
        .await
        .map_err(anyhow::Error::from)?;

    let (device_code, code_verifier) = challenge
        .device_code
        .split_once(':')
        .map(|(d, v)| (d.to_string(), v.to_string()))
        .ok_or_else(|| anyhow!("malformed device_code bundle"))?;

    eprintln!("Open this URL in your browser:");
    eprintln!();
    if !challenge.verification_uri_complete.is_empty() {
        eprintln!("  {}", challenge.verification_uri_complete);
    } else {
        eprintln!("  {}", challenge.verification_uri);
        eprintln!("  Code: {}", challenge.user_code);
    }
    eprintln!();
    eprintln!(
        "Waiting for authorization (expires in {}s)...",
        challenge.expires_in
    );

    let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(challenge.expires_in);

    loop {
        interval.tick().await;
        if tokio::time::Instant::now() >= deadline {
            return Err(anyhow!("device authorization timed out"));
        }

        match qwen
            .poll_device_token(&device_code, &code_verifier)
            .await
            .map_err(anyhow::Error::from)?
        {
            PollResult::Pending => {
                eprint!(".");
                continue;
            }
            PollResult::SlowDown => {
                interval = tokio::time::interval(std::time::Duration::from_secs(10));
                continue;
            }
            PollResult::Ready(ts) => {
                eprintln!();
                let endpoint = ts.resource_url.as_deref().unwrap_or("(default)");
                eprintln!("Authenticated. Endpoint: {endpoint}");
                eprintln!("Tokens stored in vault (qwen-oauth:credentials).");
                return Ok(());
            }
        }
    }
}

async fn cmd_status(provider: &str) -> Result<()> {
    match provider {
        "qwen" => {}
        other => return Err(anyhow!("unknown provider `{other}` — supported: qwen")),
    }

    let vault = make_vault()?;
    let qwen = build_qwen(vault)?;

    match qwen.load_tokens().await.map_err(anyhow::Error::from)? {
        Some(ts) => {
            println!("Provider:  qwen-oauth");
            println!(
                "Endpoint:  {}",
                ts.resource_url.as_deref().unwrap_or("(default dashscope)")
            );
            if let Some(exp) = ts.expiry_date {
                let remaining = exp - chrono::Utc::now();
                if remaining.num_seconds() > 0 {
                    println!("Expires:   in {}s", remaining.num_seconds());
                } else {
                    println!("Expires:   EXPIRED (refresh token may still work)");
                }
            } else {
                println!("Expires:   unknown");
            }
            println!(
                "Refresh:   {}",
                if ts.refresh_token.is_some() {
                    "present"
                } else {
                    "none"
                }
            );
        }
        None => {
            eprintln!("No tokens stored for qwen. Run `brain auth login qwen` to authenticate.");
        }
    }
    Ok(())
}

async fn cmd_logout(provider: &str) -> Result<()> {
    match provider {
        "qwen" => {}
        other => return Err(anyhow!("unknown provider `{other}` — supported: qwen")),
    }

    let vault = make_vault()?;
    let qwen = build_qwen(vault)?;

    match qwen.clear_tokens().await {
        Ok(()) => println!("Cleared Qwen OAuth tokens."),
        Err(e) => eprintln!("No tokens to clear (or error: {e})"),
    }
    Ok(())
}
