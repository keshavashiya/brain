//! REPL signals (slash-commands) recognized inside interactive chat.
//!
//! [`SIGNALS`] is the single source of truth: the banner, the `/help` listing,
//! the unknown-signal hint, and the product-self-model docs are all rendered
//! from it, so adding a signal here surfaces it everywhere automatically.

/// A REPL signal (slash-command) recognized inside the interactive chat.
pub(super) struct Signal {
    /// Canonical name, e.g. `/status`.
    pub(super) name: &'static str,
    /// Alternate spellings that invoke the same action.
    pub(super) aliases: &'static [&'static str],
    /// One-line description shown by `/help`.
    pub(super) summary: &'static str,
}

pub(super) const SIGNALS: &[Signal] = &[
    Signal {
        name: "/help",
        aliases: &["/?"],
        summary: "list available signals",
    },
    Signal {
        name: "/status",
        aliases: &[],
        summary: "show cortex, memory, and synapse status",
    },
    Signal {
        name: "/clear",
        aliases: &[],
        summary: "start a fresh conversation",
    },
    Signal {
        name: "/quit",
        aliases: &["/exit", "/q"],
        summary: "go dormant and exit chat",
    },
];

/// The REPL signals rendered as product-self-model docs, so the SOUL's
/// grounding lists the real in-chat commands (`/help`, `/status`, …) instead of
/// inventing plausible ones like `/msg`. [`SIGNALS`] stays the single source of
/// truth — the clap-walked [`crate::command_catalog::build`] is its CLI
/// counterpart.
pub(crate) fn signal_catalog() -> Vec<selfmodel::SignalDoc> {
    SIGNALS
        .iter()
        .map(|s| selfmodel::SignalDoc {
            name: s.name.to_string(),
            summary: s.summary.to_string(),
        })
        .collect()
}

/// Space-separated list of canonical signal names, for the banner and the
/// unknown-signal hint.
pub(super) fn signals_line() -> String {
    SIGNALS
        .iter()
        .map(|s| s.name)
        .collect::<Vec<_>>()
        .join("  ")
}

/// Multi-line `/help` body: each signal with its aliases and summary.
pub(super) fn signals_help() -> String {
    let width = SIGNALS.iter().map(|s| s.name.len()).max().unwrap_or(0);
    SIGNALS
        .iter()
        .map(|s| {
            let aliases = if s.aliases.is_empty() {
                String::new()
            } else {
                format!(" ({})", s.aliases.join(", "))
            };
            format!(
                "  {:<width$}  {}{}",
                s.name,
                s.summary,
                aliases,
                width = width
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
