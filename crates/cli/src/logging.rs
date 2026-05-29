//! Tracing subscriber installation, driven by the `[logging]` config policy.
//!
//! Routing:
//! - **Long-running services** (`serve`, `mcp`) log through a rotating file
//!   appender at `~/.brain/logs/brain.log` (rotation per `logging.rotation`).
//!   `mcp` *must* keep stdout clean — it's the JSON-RPC channel — and a file
//!   sink trivially satisfies that. `serve` is normally daemonised, so the
//!   file is the canonical place its logs land (with rotation, unlike the old
//!   shell-redirect that grew unbounded).
//! - **Interactive / one-shot commands** (`init`, `status`, `chat`, …) log to
//!   stderr so human-readable stdout stays clean.
//!
//! Filter precedence: `RUST_LOG` wins if set; otherwise the base level comes
//! from `logging.level` for services (or `--verbose`), and `warn` for one-shot
//! commands, with per-subsystem overrides from `logging.targets` layered on.

use brain::{BrainConfig, LogFormat, LogRotation};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

use crate::{Cli, Commands};

/// Install the global tracing subscriber. Returns the appender's
/// [`WorkerGuard`] when logging to a file — the caller must hold it for the
/// process lifetime so buffered lines flush on shutdown.
pub(crate) fn init(cli: &Cli, config: &BrainConfig) -> Option<WorkerGuard> {
    let is_service = matches!(cli.command, Commands::Serve { .. } | Commands::Mcp);
    let filter = build_filter(cli, config, is_service);
    let format = config.logging.format;

    if is_service {
        let dir = config.data_dir().join("logs");
        let appender = match config.logging.rotation {
            LogRotation::Daily => rolling::daily(&dir, "brain.log"),
            LogRotation::Hourly => rolling::hourly(&dir, "brain.log"),
            LogRotation::Never => rolling::never(&dir, "brain.log"),
        };
        let (writer, guard) = tracing_appender::non_blocking(appender);
        install(filter, format, BoxMakeWriter::new(writer), false);
        Some(guard)
    } else {
        install(filter, format, BoxMakeWriter::new(std::io::stderr), true);
        None
    }
}

/// Build the `EnvFilter`. `RUST_LOG` short-circuits everything; otherwise the
/// base `brain` level + per-target overrides are assembled from config.
fn build_filter(cli: &Cli, config: &BrainConfig, is_service: bool) -> EnvFilter {
    if let Ok(from_env) = EnvFilter::try_from_default_env() {
        return from_env;
    }

    // Services (and -v) honour the configured base level; one-shot commands
    // stay at warn so INFO never leaks into stdout-adjacent UX.
    let base = if is_service || cli.verbose {
        format!("brain={}", config.logging.level)
    } else {
        "warn".to_string()
    };
    let mut filter = EnvFilter::new(base);
    for (target, level) in &config.logging.targets {
        match format!("{target}={level}").parse() {
            Ok(directive) => filter = filter.add_directive(directive),
            Err(e) => eprintln!("warning: ignoring invalid logging.targets[{target}]={level}: {e}"),
        }
    }
    filter
}

/// Finalise the subscriber for a concrete writer + format. ANSI is on only for
/// the stderr (interactive) sink; a file never gets escape codes.
fn install(filter: EnvFilter, format: LogFormat, writer: BoxMakeWriter, ansi: bool) {
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(writer)
        .with_ansi(ansi);
    let fmt_layer = match format {
        LogFormat::Json => fmt_layer.json().boxed(),
        LogFormat::Pretty => fmt_layer.boxed(),
    };
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .init();
}
