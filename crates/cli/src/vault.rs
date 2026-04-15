//! CLI commands for the credential vault.

use anyhow::{anyhow, Result};
use clap::Subcommand;
use std::io::{BufRead, IsTerminal};

use vault::{
    resolve_backend, BackendSelection, CredentialValue, CredentialVault, DefaultVault,
    InjectionShape, VaultBackend, VaultConfig, VaultError,
};

#[derive(Subcommand)]
pub(crate) enum VaultAction {
    /// Initialize the vault (picks backend, sets up fallback verifier).
    Init {
        /// Force file backend even if OS keychain is available.
        #[arg(long)]
        file: bool,
        /// Path to a file containing the fallback passphrase (optional).
        #[arg(long)]
        passphrase_file: Option<String>,
    },

    /// Store a credential. Value is read from stdin (never argv, never logged).
    Set {
        tool: String,
        key: String,
        /// Injection shape: env:<NAME>, header:<NAME>, arg:<POS>
        #[arg(long, default_value = "env")]
        shape: String,
        /// Env var / header name (required for env and header shapes unless
        /// shape is given as e.g. `env:MY_VAR`).
        #[arg(long)]
        name: Option<String>,
        /// Arg position (required for arg shape unless shape is e.g. `arg:0`).
        #[arg(long)]
        position: Option<usize>,
    },

    /// Show metadata for a credential. `--reveal` prints the value.
    Get {
        tool: String,
        key: String,
        #[arg(long)]
        reveal: bool,
    },

    /// Delete a credential.
    Delete { tool: String, key: String },

    /// List credential metadata (no values).
    List {
        /// Filter by tool name.
        #[arg(long)]
        tool: Option<String>,
    },

    /// Show which backend is active and where the vault lives.
    Status,
}

pub(crate) async fn cmd_vault(
    _config: &brain_core::BrainConfig,
    action: VaultAction,
) -> Result<()> {
    // For now, vault config is default-constructed. Wiring it into
    // BrainConfig is a follow-up; in this slice the env var and default
    // ~/.brain/vault path are sufficient.
    match action {
        VaultAction::Init {
            file,
            passphrase_file,
        } => cmd_init(file, passphrase_file).await,
        VaultAction::Set {
            tool,
            key,
            shape,
            name,
            position,
        } => cmd_set(&tool, &key, &shape, name.as_deref(), position).await,
        VaultAction::Get { tool, key, reveal } => cmd_get(&tool, &key, reveal).await,
        VaultAction::Delete { tool, key } => cmd_delete(&tool, &key).await,
        VaultAction::List { tool } => cmd_list(tool.as_deref()).await,
        VaultAction::Status => cmd_status().await,
    }
}

fn build_vault(config: VaultConfig) -> Result<DefaultVault> {
    let backend = resolve_backend(&config).map_err(anyhow::Error::from)?;
    Ok(DefaultVault::new(backend))
}

async fn cmd_init(file: bool, passphrase_file: Option<String>) -> Result<()> {
    let config = VaultConfig {
        backend: if file {
            BackendSelection::File
        } else {
            BackendSelection::Auto
        },
        passphrase_file: passphrase_file.map(std::path::PathBuf::from),
        ..Default::default()
    };
    let backend = resolve_backend(&config)?;
    match &backend {
        VaultBackend::File(fb) => {
            fb.init().await?;
            println!("Vault initialized: file backend at ~/.brain/vault");
        }
        #[cfg(target_os = "macos")]
        VaultBackend::Keychain(_) => {
            println!("Vault backend: macOS keychain (service=\"brain\"). No init required.");
        }
        #[cfg(target_os = "linux")]
        VaultBackend::SecretService(_) => {
            println!(
                "Vault backend: secret-service (GNOME Keyring / KDE Wallet). No init required."
            );
        }
    }
    Ok(())
}

async fn cmd_set(
    tool: &str,
    key: &str,
    shape_spec: &str,
    name: Option<&str>,
    position: Option<usize>,
) -> Result<()> {
    let shape = parse_shape(shape_spec, name, position)?;
    let value = read_secret_from_stdin(tool, key)?;

    let vault = build_vault(VaultConfig::default())?;
    vault
        .store(tool, key, CredentialValue::new(value), shape)
        .await
        .map_err(anyhow::Error::from)?;
    println!("Stored {tool}:{key} ({})", vault.backend_kind());
    Ok(())
}

async fn cmd_get(tool: &str, key: &str, reveal: bool) -> Result<()> {
    let vault = build_vault(VaultConfig::default())?;
    let injected = match vault.get(tool, key).await {
        Ok(i) => i,
        Err(VaultError::NotFound { .. }) => {
            eprintln!("Not found: {tool}:{key}");
            std::process::exit(1);
        }
        Err(e) => return Err(anyhow::Error::from(e)),
    };
    println!("tool:    {tool}");
    println!("key:     {key}");
    println!("shape:   {}", shape_to_str(&injected.shape));
    println!("backend: {}", vault.backend_kind());
    if reveal {
        println!("value:   {}", injected.value.as_str());
    } else {
        println!("value:   <hidden — pass --reveal to print>");
    }
    Ok(())
}

async fn cmd_delete(tool: &str, key: &str) -> Result<()> {
    let vault = build_vault(VaultConfig::default())?;
    match vault.delete(tool, key).await {
        Ok(()) => {
            println!("Deleted {tool}:{key}");
            Ok(())
        }
        Err(VaultError::NotFound { .. }) => {
            eprintln!("Not found: {tool}:{key}");
            std::process::exit(1);
        }
        Err(e) => Err(anyhow::Error::from(e)),
    }
}

async fn cmd_list(tool: Option<&str>) -> Result<()> {
    let vault = build_vault(VaultConfig::default())?;
    let entries = vault.list(tool).await?;
    if entries.is_empty() {
        println!("No credentials stored.");
        return Ok(());
    }
    println!(
        "{:<16} {:<20} {:<16} {:<25} SHAPE",
        "TOOL", "KEY", "BACKEND", "CREATED"
    );
    for m in entries {
        println!(
            "{:<16} {:<20} {:<16} {:<25} {}",
            m.tool,
            m.key,
            m.backend,
            m.created_at,
            shape_to_str(&m.shape)
        );
    }
    Ok(())
}

async fn cmd_status() -> Result<()> {
    let config = VaultConfig::default();
    let backend = resolve_backend(&config)?;
    let vault = DefaultVault::new(backend);
    println!("Backend:    {}", vault.backend_kind());
    let passphrase_source = if std::env::var("BRAIN_VAULT_PASSPHRASE").is_ok() {
        "env:BRAIN_VAULT_PASSPHRASE"
    } else if config.passphrase_file.is_some() {
        "file"
    } else {
        "tty-prompt"
    };
    println!("Passphrase: {passphrase_source}");
    let count = vault.list(None).await.map(|v| v.len()).unwrap_or(0);
    println!("Entries:    {count}");
    Ok(())
}

fn parse_shape(spec: &str, name: Option<&str>, position: Option<usize>) -> Result<InjectionShape> {
    let (kind, rest) = match spec.split_once(':') {
        Some((k, r)) => (k, Some(r)),
        None => (spec, None),
    };
    match kind.to_lowercase().as_str() {
        "env" => {
            let n = rest
                .map(str::to_string)
                .or_else(|| name.map(str::to_string))
                .ok_or_else(|| anyhow!("env shape requires --name <NAME> or env:<NAME>"))?;
            Ok(InjectionShape::EnvVar { name: n })
        }
        "header" => {
            let n = rest
                .map(str::to_string)
                .or_else(|| name.map(str::to_string))
                .ok_or_else(|| anyhow!("header shape requires --name <NAME> or header:<NAME>"))?;
            Ok(InjectionShape::Header { name: n })
        }
        "arg" => {
            let p = rest
                .and_then(|s| s.parse::<usize>().ok())
                .or(position)
                .ok_or_else(|| anyhow!("arg shape requires --position <N> or arg:<N>"))?;
            Ok(InjectionShape::Arg { position: p })
        }
        other => Err(anyhow!("unknown shape `{other}` — use env, header, or arg")),
    }
}

fn shape_to_str(shape: &InjectionShape) -> String {
    match shape {
        InjectionShape::EnvVar { name } => format!("env:{name}"),
        InjectionShape::Header { name } => format!("header:{name}"),
        InjectionShape::Arg { position } => format!("arg:{position}"),
    }
}

fn read_secret_from_stdin(tool: &str, key: &str) -> Result<String> {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        // Prompt with no-echo.
        let prompt = format!("Value for {tool}:{key}: ");
        rpassword::prompt_password(&prompt).map_err(anyhow::Error::from)
    } else {
        // Piped input — read a single line.
        let mut handle = stdin.lock();
        let mut buf = String::new();
        handle.read_line(&mut buf)?;
        let trimmed = buf.trim_end_matches(['\n', '\r']).to_string();
        if trimmed.is_empty() {
            return Err(anyhow!("no value provided on stdin"));
        }
        Ok(trimmed)
    }
}
