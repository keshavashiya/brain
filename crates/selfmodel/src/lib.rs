//! Product self-model — Brain's grounded knowledge of *itself*.
//!
//! The resident reasoner (the SOUL) already receives a live capability digest
//! of the *external* tools it mediates (MCP servers, native backends, agents).
//! It had no grounded account of **Brain-the-product**: its own CLI commands,
//! config schema, or platform-agnostic policy. Lacking that, it confabulated a
//! plausible-looking surface — inventing config keys (`channels.telegram.token`)
//! and commands (`brain send`) that do not exist.
//!
//! This module assembles a [`ProductSelfModel`] from three code-derived sources
//! and renders a retrieval-scored grounding block the chat prompt injects as an
//! authoritative "About Brain" section:
//!
//! - **Commands** — the real CLI surface, walked from the clap definition by the
//!   binary crate and handed in as [`CommandDoc`]s (clap stays the single source
//!   of truth, so a new subcommand self-registers).
//! - **Config schema** — sliced from the embedded `default.yaml` (handed in by
//!   the caller via `BrainConfig::default_config_content()`), the same file
//!   `brain init` writes. Its
//!   comments carry the human descriptions and the real
//!   `actions.messaging.channels { url, body, headers }` webhook shape, so a new
//!   config key self-registers there too.
//! - **Policy facts** — a small curated list of architecture/policy invariants
//!   that aren't mechanically derivable (e.g. "no native Telegram transport").

pub mod host;
pub use host::{DiskClass, DiskInfo, GpuInfo, HostModel, ModelFit};

/// One CLI command, derived from the clap definition by the binary crate.
/// Trusted, structural fields only (name, one-line summary, arg names) — the
/// same restraint the capability digest applies to external tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandDoc {
    /// Subcommand name as typed, e.g. `chat`.
    pub name: String,
    /// First line of the command's clap `about`/doc-comment.
    pub summary: String,
    /// Argument identifiers (positionals + options), for shape hints.
    pub args: Vec<String>,
}

/// One in-chat REPL signal (slash-command), derived from the binary crate's
/// `SIGNALS` table. The CLI subcommands ([`CommandDoc`]) are the *outer*
/// surface (`brain …`); these are the *inner* surface available while a chat
/// session is open (`/status`, `/clear`, …). Modeling them stops the SOUL
/// inventing plausible-but-nonexistent signals like `/msg`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignalDoc {
    /// Canonical signal name including the leading slash, e.g. `/status`.
    pub name: String,
    /// One-line description.
    pub summary: String,
}

/// A top-level slice of the embedded default config, anchored on one of the
/// `# ── … ──` banner comments. Body is the verbatim YAML + comments — the
/// comments are the descriptions, so they ride along unmodified.
#[derive(Debug, Clone)]
struct ConfigSection {
    /// Human label from the banner, e.g. `Actions`.
    label: String,
    /// First top-level key in the slice, e.g. `actions`.
    top_key: String,
    /// Verbatim YAML + comments for the slice.
    body: String,
}

/// Curated policy/architecture invariants the SOUL must not contradict or
/// fabricate around. Hand-maintained on purpose: architecture *intent* isn't
/// mechanically derivable from the code. Keep this terse.
const POLICY_FACTS: &[&str] = &[
    "Brain has NO native Telegram, Slack, Discord, or Email transports. To send a \
     message, configure a generic webhook under `actions.messaging.channels` (each \
     entry is `{ url, body, headers }`) or a long-lived WebSocket gateway under \
     `channel.relays`. There is no `channels.telegram.token` / `chat_id` / `parse_mode` \
     schema — platform specifics live inside the `url`/`body` you provide.",
    "Once a channel is configured, you CAN send: a natural request like \
     \"send via <channel> to <recipient>: <message>\" routes as a gated messaging \
     action (External tier — it asks for confirmation, then delivers over the \
     configured webhook/relay). This is an action, not a CLI command or a `/`-signal; \
     if no channel is configured, say so and point to `actions.messaging.channels`.",
    "Brain is local-first: it runs entirely on the user's machine; memories and \
     credentials never leave it.",
    "Config lives at `~/.brain/config.yaml`. Any key can be overridden by an \
     environment variable named `BRAIN_<SECTION>__<KEY>` (e.g. `BRAIN_LLM__API_KEY`).",
    "Brain is not a cloud service or a generic chatbot wrapper — it is a local \
     cognitive/memory engine driven by the fixed set of CLI commands listed above.",
    "There is no `brain restart`, `brain reload`, or `brain send` command — restart \
     by running `brain stop` then `brain start`. Compose any multi-step operation \
     from the commands and in-chat signals listed above; never invent a new verb.",
];

/// Commands the lifecycle [`POLICY_FACTS`] entry explicitly says do **not**
/// exist. The binary crate cross-checks these against the live clap catalog
/// (`cli::selfmodel`) so a future rename/addition can't leave a stale negation
/// here — the self-model must never both deny and list the same command (F4).
pub const DENIED_COMMANDS: &[&str] = &["restart", "reload", "send"];

/// Commands the lifecycle [`POLICY_FACTS`] entry points users *to* as the real
/// path (e.g. "restart = `brain stop` then `brain start`"). These MUST exist in
/// the clap catalog, or the remediation advice is itself a fabrication.
pub const AFFIRMED_COMMANDS: &[&str] = &["stop", "start"];

/// Per-section cap on rendered config body, so a comment-heavy section can't
/// dominate the prompt. Generous enough to keep small sections intact.
const MAX_SECTION_CHARS: usize = 1600;

/// Brain's grounded self-knowledge, injected into the SOUL prompt as the
/// authoritative "About Brain" section. Built once at bootstrap.
#[derive(Debug, Clone)]
pub struct ProductSelfModel {
    commands: Vec<CommandDoc>,
    signals: Vec<SignalDoc>,
    config_sections: Vec<ConfigSection>,
    policy_facts: &'static [&'static str],
}

impl ProductSelfModel {
    /// Build from the binary crate's code-derived catalogs: the clap-walked CLI
    /// `commands` and the `SIGNALS`-table-walked in-chat `signals`. The
    /// `default_config` is the embedded `default.yaml` text (the same source
    /// `brain init` writes, via `BrainConfig::default_config_content()`); its
    /// commented sections become the config-schema grounding. Policy facts are
    /// sourced internally from this crate's curated [`POLICY_FACTS`].
    pub fn new(commands: Vec<CommandDoc>, signals: Vec<SignalDoc>, default_config: &str) -> Self {
        Self {
            commands,
            signals,
            config_sections: parse_config_sections(default_config),
            policy_facts: POLICY_FACTS,
        }
    }

    /// Render the authoritative grounding block for one chat turn.
    ///
    /// The command list and policy facts are always included (small, bounded);
    /// config sections are retrieval-scored against `query` and only the top
    /// `k` *matching* sections are injected — an unrelated turn (e.g. "hi")
    /// pulls in no config noise. The header instructs the reasoner to prefer
    /// this over general knowledge for any claim about Brain's own surface.
    pub fn render_grounding(&self, query: &str, k: usize) -> String {
        let mut out = String::from(
            "About Brain (authoritative self-knowledge — prefer this over general \
             knowledge for any claim about Brain's own commands, config, or features; \
             if something isn't here, say so rather than inventing it):\n",
        );

        if !self.commands.is_empty() {
            out.push_str("\nCLI commands (the complete set — there are no others):\n");
            for cmd in &self.commands {
                let args = if cmd.args.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", cmd.args.join(" "))
                };
                if cmd.summary.is_empty() {
                    out.push_str(&format!("- brain {}{}\n", cmd.name, args));
                } else {
                    out.push_str(&format!("- brain {}{} — {}\n", cmd.name, args, cmd.summary));
                }
            }
        }

        if !self.signals.is_empty() {
            out.push_str(
                "\nIn-chat signals, typed inside a `brain chat` session (the complete set — \
                 there are no others; none of these send a message — messaging is a gated \
                 channel action, see Policy below):\n",
            );
            for sig in &self.signals {
                if sig.summary.is_empty() {
                    out.push_str(&format!("- {}\n", sig.name));
                } else {
                    out.push_str(&format!("- {} — {}\n", sig.name, sig.summary));
                }
            }
        }

        let sections = self.top_config_sections(query, k);
        if !sections.is_empty() {
            out.push_str(
                "\nRelevant config (from the real schema in ~/.brain/config.yaml — these \
                 keys and shapes are exact; do not invent others):\n",
            );
            for section in sections {
                out.push_str(&format!(
                    "\n[{} — `{}:`]\n{}\n",
                    section.label,
                    section.top_key,
                    cap_body(&section.body),
                ));
            }
        }

        if !self.policy_facts.is_empty() {
            out.push_str("\nPolicy & invariants:\n");
            for fact in self.policy_facts {
                out.push_str(&format!("- {fact}\n"));
            }
        }

        out
    }

    /// Score config sections by keyword overlap with `query`, keeping the best
    /// `k` with a positive score (sorted by score, then label for stability).
    fn top_config_sections(&self, query: &str, k: usize) -> Vec<&ConfigSection> {
        if k == 0 {
            return Vec::new();
        }
        let terms = tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, &ConfigSection)> = self
            .config_sections
            .iter()
            .map(|s| (score_section(s, &terms), s))
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.label.cmp(&b.1.label)));
        scored.truncate(k);
        scored.into_iter().map(|(_, s)| s).collect()
    }
}

/// Split the embedded config into top-level sections anchored on the
/// `# ── … ──` banner comments. The leading file-header comment (before the
/// first keyed section) is dropped — it carries no schema.
fn parse_config_sections(yaml: &str) -> Vec<ConfigSection> {
    let mut sections = Vec::new();
    let mut label: Option<String> = None;
    let mut body = String::new();

    let flush = |sections: &mut Vec<ConfigSection>, label: &Option<String>, body: &str| {
        if let Some(top_key) = first_top_level_key(body) {
            sections.push(ConfigSection {
                label: label.clone().unwrap_or_else(|| top_key.clone()),
                top_key,
                body: body.trim_matches('\n').to_string(),
            });
        }
    };

    for line in yaml.lines() {
        if let Some(banner) = banner_label(line) {
            // A new banner closes the section in progress.
            flush(&mut sections, &label, &body);
            label = Some(banner);
            body.clear();
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    flush(&mut sections, &label, &body);
    sections
}

/// If `line` is a `# ── Label ──…` banner, return the trimmed label.
fn banner_label(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('#')?.trim_start();
    if !rest.starts_with('─') {
        return None;
    }
    let label = rest.trim_matches(|c| c == '─' || c == ' ');
    if label.is_empty() {
        None
    } else {
        Some(label.to_string())
    }
}

/// First unindented `key:` in a body chunk (the section's top-level config key).
fn first_top_level_key(body: &str) -> Option<String> {
    body.lines().find_map(|line| {
        if line.starts_with(|c: char| c.is_ascii_lowercase()) {
            let key = line.split(':').next()?.trim();
            if !key.is_empty() && key.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                return Some(key.to_string());
            }
        }
        None
    })
}

/// Truncate a section body to [`MAX_SECTION_CHARS`] on a line boundary.
fn cap_body(body: &str) -> String {
    if body.len() <= MAX_SECTION_CHARS {
        return body.to_string();
    }
    let mut out = String::with_capacity(MAX_SECTION_CHARS + 16);
    for line in body.lines() {
        if out.len() + line.len() + 1 > MAX_SECTION_CHARS {
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("# … (truncated)");
    out
}

// Lowercase alphanumeric terms — same tokenization the tool-loop ranker uses.
use synapse::tokenize;

/// Count of distinct query terms that appear as a *whole word* in the section's
/// label + key + body. Whole-word (not substring) so a short query term like
/// "hi" can't spuriously match "higher".
fn score_section(section: &ConfigSection, terms: &[String]) -> usize {
    let haystack: std::collections::HashSet<String> = tokenize(&format!(
        "{} {} {}",
        section.label, section.top_key, section.body
    ))
    .into_iter()
    .collect();
    terms
        .iter()
        .filter(|t| haystack.contains(t.as_str()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ProductSelfModel {
        ProductSelfModel::new(
            vec![
                CommandDoc {
                    name: "chat".to_string(),
                    summary: "interactive chat session".to_string(),
                    args: vec!["message".to_string()],
                },
                CommandDoc {
                    name: "status".to_string(),
                    summary: "show system vitals".to_string(),
                    args: vec![],
                },
            ],
            vec![
                SignalDoc {
                    name: "/status".to_string(),
                    summary: "show cortex, memory, and synapse status".to_string(),
                },
                SignalDoc {
                    name: "/quit".to_string(),
                    summary: "go dormant and exit chat".to_string(),
                },
            ],
            // Exercise against the real shipped defaults so the config-schema
            // grounding tests guard the actual `default.yaml` surface.
            brain::BrainConfig::default_config_content(),
        )
    }

    #[test]
    fn parses_real_sections_from_default_config() {
        let sections = parse_config_sections(brain::BrainConfig::default_config_content());
        let keys: Vec<&str> = sections.iter().map(|s| s.top_key.as_str()).collect();
        assert!(
            keys.contains(&"actions"),
            "actions section present: {keys:?}"
        );
        assert!(keys.contains(&"security"));
        assert!(keys.contains(&"llm"));
    }

    #[test]
    fn telegram_query_surfaces_messaging_webhook_shape() {
        let grounding = model().render_grounding("how do I configure telegram in config.yaml", 2);
        // Real webhook shape is present…
        assert!(grounding.contains("actions"));
        assert!(grounding.contains("channels"));
        assert!(grounding.contains("url"));
        // …and the fabricated keys from the transcript are NOT introduced as schema.
        assert!(!grounding.contains("parse_mode:"));
        assert!(!grounding.contains("chat_id:"));
        // Policy fact pins the no-native-transport rule.
        assert!(grounding.contains("NO native Telegram"));
    }

    #[test]
    fn commands_are_listed_and_phantoms_absent() {
        let grounding = model().render_grounding("what commands can I run", 1);
        assert!(grounding.contains("- brain chat"));
        assert!(grounding.contains("- brain status"));
        // Phantoms must not appear as *available* commands (rendered bullets).
        // The lifecycle policy fact may name them in a negation, so we check the
        // bullet form rather than a blanket substring.
        assert!(!grounding.contains("- brain send"));
        assert!(!grounding.contains("- brain reload"));
        assert!(!grounding.contains("- brain restart"));
        // …and that policy fact pins the no-restart invariant the SOUL
        // fabricated in the end-user transcript.
        assert!(grounding.contains("no `brain restart`"));
    }

    #[test]
    fn lifecycle_policy_prose_matches_structured_command_lists() {
        // The structured DENIED/AFFIRMED command lists are the machine-checkable
        // half of the hand-written lifecycle POLICY_FACT. Keep the two in sync:
        // every command named in the constants must appear (as `brain <name>`)
        // in some policy fact, so the binary crate's catalog cross-check (F4)
        // is guarding the same names the prose actually claims.
        let policy = POLICY_FACTS.join("\n");
        for cmd in DENIED_COMMANDS {
            assert!(
                policy.contains(&format!("brain {cmd}")),
                "DENIED_COMMANDS lists `{cmd}` but no POLICY_FACT mentions `brain {cmd}`",
            );
        }
        for cmd in AFFIRMED_COMMANDS {
            assert!(
                policy.contains(&format!("brain {cmd}")),
                "AFFIRMED_COMMANDS lists `{cmd}` but no POLICY_FACT mentions `brain {cmd}`",
            );
        }
        // The two lists must be disjoint — a command can't be both denied and
        // pointed-to as the real path.
        for d in DENIED_COMMANDS {
            assert!(
                !AFFIRMED_COMMANDS.contains(d),
                "`{d}` is in both DENIED_COMMANDS and AFFIRMED_COMMANDS",
            );
        }
    }

    #[test]
    fn in_chat_signals_are_listed_and_phantoms_absent() {
        let grounding = model().render_grounding("how do I send a message in chat", 1);
        // Real signals from the SIGNALS table are grounded…
        assert!(grounding.contains("/status"));
        assert!(grounding.contains("/quit"));
        assert!(grounding.contains("In-chat signals"));
        // …and the phantom signal from the transcript is not introduced.
        assert!(!grounding.contains("/msg"));
    }

    #[test]
    fn unrelated_query_injects_no_config_sections() {
        let grounding = model().render_grounding("hi there", 3);
        // Commands + policy still render; no config slice for a chit-chat turn.
        assert!(grounding.contains("CLI commands"));
        assert!(!grounding.contains("Relevant config"));
    }

    #[test]
    fn banner_label_parses_and_ignores_plain_comments() {
        assert_eq!(
            banner_label("# ── Actions ────────────────"),
            Some("Actions".to_string())
        );
        assert_eq!(banner_label("# just a comment"), None);
        assert_eq!(banner_label("actions:"), None);
    }
}
