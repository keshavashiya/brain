//! Log pattern analysis — the executor behind `brain logs analyze` and the
//! `logs.analyze` native capability.
//!
//! Two-stage, truthful-by-construction:
//!
//! 1. **Deterministic pattern pass** (always runs, works offline). Recent log
//!    lines are read from one of two sources, each line is classified by level
//!    and *normalised* into a signature — volatile tokens (timestamps, UUIDs,
//!    hex, paths, numbers) are collapsed to placeholders — and identical
//!    signatures are counted. The result is a digest: per-level totals plus the
//!    top recurring patterns with their counts. This is the core; it never
//!    invents anything, it only counts what is there. [`analyze`] returns this
//!    digest, and it is the entire `logs.analyze` capability — narration is the
//!    reasoner's job, not the capability's.
//!
//! 2. **Optional LLM narration** ([`narrate`], used by the CLI). The
//!    deterministic digest — not the raw log — is handed to the LLM, which
//!    renders a short plain-language summary. Because it only ever sees the
//!    digest, it cannot fabricate log lines or causes.
//!
//! Sources: [`LogSource::Brain`] (default) reads the daemon's own rotated logs in
//! `~/.brain/logs/`; [`LogSource::System`] reads OS logs (`log show` on macOS,
//! sandboxed `journalctl` on Linux).
//!
//! Both the daemon's pretty format (`2026-… INFO target: msg`) and its JSON
//! format (`logging.format: json`) are parsed: a JSON line's `timestamp`,
//! `level`, and `fields.message` are read directly; otherwise the leading
//! RFC3339 token + level keyword are used.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use brain::BrainConfig;
use chrono::{DateTime, Utc};

/// Where to read logs from. A plain enum (no `clap` derive) so the backends
/// crate stays free of the CLI's arg-parsing dependency; the CLI maps its own
/// `--source` value onto this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogSource {
    /// The daemon's own rotated logs in `~/.brain/logs/`.
    Brain,
    /// OS logs: `log show` (macOS) / sandboxed `journalctl` (Linux).
    System,
}

/// Analyse recent logs into the deterministic digest. Reads from `source`,
/// applies the `since` window, caps at `lines`, groups by normalised signature,
/// and renders the per-level totals + top recurring patterns. No LLM, no
/// network — this is the whole `logs.analyze` capability.
pub async fn analyze(
    config: &BrainConfig,
    source: LogSource,
    since: &str,
    lines: usize,
) -> Result<String> {
    // Validate the window up front so a bad `since` fails before we read.
    let window = parse_since(since)?;
    let lines = lines.max(1);

    // The brain source filters by timestamp here (its lines carry one); the
    // system source is already filtered by the OS tool, so we don't re-cut it.
    let (raw, source_label, cutoff) = match source {
        LogSource::Brain => {
            let (raw, label) = read_brain_log(config, lines)?;
            (raw, label, Some(Utc::now() - window))
        }
        LogSource::System => {
            let (raw, label) = read_system_log(since, lines).await?;
            (raw, label, None)
        }
    };

    let window_label = match source {
        LogSource::Brain => format!("last {since}, \u{2264}{lines} lines"),
        LogSource::System => format!("last {since}"),
    };

    let digest = analyze_lines(&raw, cutoff, &source_label, &window_label);
    Ok(digest.render())
}

// ─── Sources ────────────────────────────────────────────────────────────────

/// Maximum number of rotated log files to read. The newest files are kept; the
/// per-line `since` filter then cuts precisely, so this is just a memory bound
/// for an unexpectedly long history.
const MAX_LOG_FILES: usize = 8;

/// Read the daemon's own logs. `tracing_appender` rotates daily/hourly into
/// `brain.log.<date>[-<hh>]` (or a single `brain.log` when rotation is off), so
/// we gather every such file, read the newest [`MAX_LOG_FILES`], concatenate
/// oldest→newest, and tail to `lines`.
fn read_brain_log(config: &BrainConfig, lines: usize) -> Result<(Vec<String>, String)> {
    let dir = config.data_dir().join("logs");
    let files = resolve_log_files(&dir);
    if files.is_empty() {
        anyhow::bail!(
            "no daemon logs in {} — the daemon may never have run (start it with `brain start`)",
            dir.display()
        );
    }

    let mut all: Vec<String> = Vec::new();
    for path in &files {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading daemon log at {}", path.display()))?;
        all.extend(content.lines().map(str::to_string));
    }

    let label = if files.len() == 1 {
        files[0].display().to_string()
    } else {
        format!("{} ({} rotated files)", dir.display(), files.len())
    };
    Ok((tail_n(all, lines), label))
}

/// Enumerate the daemon's log files in a directory, newest last (so a final
/// concat is chronological). Matches `brain.log` and `brain.log.<suffix>`,
/// excluding the `brain.stderr.log` capture file. The date/datetime suffix
/// sorts chronologically, so a lexicographic sort by filename is correct.
fn resolve_log_files(dir: &PathBuf) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n == "brain.log" || n.starts_with("brain.log."))
        })
        .collect();
    files.sort();
    // Keep the newest MAX_LOG_FILES (sort is ascending, so that's the tail).
    if files.len() > MAX_LOG_FILES {
        files.drain(0..files.len() - MAX_LOG_FILES);
    }
    files
}

/// Read the platform's system log and return its most recent lines.
///
/// On Linux, `journalctl` runs through the exec sandbox (bounded, isolated).
/// On macOS, Apple's `log` tool *refuses to run inside `sandbox-exec`* ("Cannot
/// run while sandboxed"), so the fixed, read-only `log show` command runs via a
/// timeout-bounded child process instead — the only way to make the system
/// source work there.
async fn read_system_log(since: &str, lines: usize) -> Result<(Vec<String>, String)> {
    let (binary, args, label) = system_log_command(since)?;
    let stdout = run_system_log(&binary, &args).await?;
    let all: Vec<String> = stdout.lines().map(str::to_string).collect();
    Ok((tail_n(all, lines), label))
}

/// Run a system-log command and return its stdout. macOS bypasses the sandbox
/// (see [`read_system_log`]); Linux routes through `IsolatedSandbox`.
#[cfg(target_os = "macos")]
async fn run_system_log(binary: &str, args: &[String]) -> Result<String> {
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::process::Command::new(binary).args(args).output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("`{binary}` timed out after 30s"))?
    .with_context(|| format!("running `{binary}`"))?;

    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        anyhow::bail!(
            "`{binary}` failed{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(not(target_os = "macos"))]
async fn run_system_log(binary: &str, args: &[String]) -> Result<String> {
    use sandbox::SandboxExecutor;

    let timeout = Duration::from_secs(30);
    let command =
        sandbox::SandboxCommand::new(binary.to_string(), args.to_vec()).with_timeout(timeout);
    let executor = sandbox::IsolatedSandbox::new(vec![binary.to_string()], timeout);

    let outcome = executor
        .run(command)
        .await
        .with_context(|| format!("running `{binary}` via the exec sandbox"))?;

    if outcome.exit_code != 0 {
        let detail = outcome.stderr.trim();
        anyhow::bail!(
            "`{binary}` exited {}{}",
            outcome.exit_code,
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        );
    }
    Ok(outcome.stdout)
}

#[cfg(target_os = "macos")]
fn system_log_command(since: &str) -> Result<(String, Vec<String>, String)> {
    // `log show --last` accepts the same `30m`/`1h`/`2d` shape we validate.
    let _ = parse_since(since)?;
    Ok((
        "log".to_string(),
        vec![
            "show".to_string(),
            "--style".to_string(),
            "syslog".to_string(),
            "--last".to_string(),
            since.to_string(),
        ],
        format!("system log (`log show --last {since}`)"),
    ))
}

#[cfg(target_os = "linux")]
fn system_log_command(since: &str) -> Result<(String, Vec<String>, String)> {
    let ago = linux_since(since)?;
    Ok((
        "journalctl".to_string(),
        vec!["--no-pager".to_string(), "--since".to_string(), ago.clone()],
        format!("system journal (`journalctl --since \"{ago}\"`)"),
    ))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn system_log_command(_since: &str) -> Result<(String, Vec<String>, String)> {
    anyhow::bail!("the system log source is only supported on macOS and Linux")
}

/// Translate `<n>{m|h|d}` into a systemd `--since` phrase, e.g. `1h` →
/// `"1 hour ago"`.
#[cfg(any(target_os = "linux", test))]
fn linux_since(since: &str) -> Result<String> {
    let (amount, unit) = parse_since_parts(since)?;
    let noun = match unit {
        'm' => "minute",
        'h' => "hour",
        _ => "day",
    };
    let plural = if amount == 1 { "" } else { "s" };
    Ok(format!("{amount} {noun}{plural} ago"))
}

// ─── Line parsing (pretty + JSON) ─────────────────────────────────────────────

/// Log severity, ordered most-severe-first so the derived `Ord` sorts the way a
/// reader expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
    Other,
}

impl Level {
    fn label(self) -> &'static str {
        match self {
            Level::Error => "ERROR",
            Level::Warn => "WARN",
            Level::Info => "INFO",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
            Level::Other => "other",
        }
    }

    /// Map a level keyword (any case) to a [`Level`], or `None` if unrecognised.
    fn from_keyword(word: &str) -> Option<Level> {
        match word.to_ascii_uppercase().as_str() {
            "ERROR" | "ERR" | "FAULT" | "CRITICAL" | "FATAL" => Some(Level::Error),
            "WARN" | "WARNING" => Some(Level::Warn),
            "INFO" | "NOTICE" => Some(Level::Info),
            "DEBUG" => Some(Level::Debug),
            "TRACE" => Some(Level::Trace),
            _ => None,
        }
    }
}

/// One parsed log line: an optional timestamp (for `since` filtering), a level,
/// and a normalised signature (for recurrence grouping).
struct ParsedLine {
    ts: Option<DateTime<Utc>>,
    level: Level,
    signature: String,
}

/// Parse one raw line, handling both the daemon's JSON format and its
/// pretty/syslog text format.
fn parse_line(line: &str) -> ParsedLine {
    let trimmed = line.trim_start();
    if trimmed.starts_with('{') {
        if let Some(parsed) = parse_json_line(trimmed) {
            return parsed;
        }
    }
    parse_text_line(line)
}

/// Parse a structured JSON log line (`logging.format: json`). Reads `timestamp`,
/// `level`, and the `fields.message` (falling back to a top-level `message`).
fn parse_json_line(line: &str) -> Option<ParsedLine> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let obj = value.as_object()?;

    let ts = obj
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|t| t.with_timezone(&Utc));

    let level = obj
        .get("level")
        .and_then(|v| v.as_str())
        .and_then(Level::from_keyword)
        .unwrap_or(Level::Other);

    let message = obj
        .get("fields")
        .and_then(|f| f.get("message"))
        .or_else(|| obj.get("message"))
        .and_then(|v| v.as_str())
        .unwrap_or(line);

    // Group by target + message so the same event from the same module clusters.
    let target = obj.get("target").and_then(|v| v.as_str()).unwrap_or("");
    let basis = if target.is_empty() {
        message.to_string()
    } else {
        format!("{target}: {message}")
    };

    Some(ParsedLine {
        ts,
        level,
        signature: normalize_signature(&basis),
    })
}

/// Parse a pretty/syslog text line. A leading RFC3339 token is taken as the
/// timestamp and a following level keyword as the level; both are stripped from
/// the signature so it reads as `target: message` (matching the JSON path) and
/// doesn't repeat the level already shown beside the count. When the leading
/// tokens aren't in that shape (e.g. `log show` syslog style), the level falls
/// back to a keyword scan and the whole line normalises into the signature.
fn parse_text_line(line: &str) -> ParsedLine {
    let mut rest = line.trim_start();

    let mut ts = None;
    if let Some((first, tail)) = split_first_token(rest) {
        if let Ok(t) = DateTime::parse_from_rfc3339(first) {
            ts = Some(t.with_timezone(&Utc));
            rest = tail;
        }
    }

    let level = match split_first_token(rest) {
        Some((first, tail)) if Level::from_keyword(first).is_some() => {
            rest = tail;
            Level::from_keyword(first).unwrap()
        }
        // Level wasn't where we expected it; scan the whole line instead.
        _ => classify_text_level(line),
    };

    ParsedLine {
        ts,
        level,
        signature: normalize_signature(rest),
    }
}

/// Split off the first whitespace-delimited token, returning `(token, rest)`
/// with `rest` left-trimmed. `None` when the input is all whitespace.
fn split_first_token(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    match s.find(char::is_whitespace) {
        Some(i) => Some((&s[..i], s[i..].trim_start())),
        None => Some((s, "")),
    }
}

/// Heuristic level classification for text lines: the first level keyword found
/// as a standalone alphabetic token.
fn classify_text_level(line: &str) -> Level {
    for token in line.split(|c: char| !c.is_ascii_alphabetic()) {
        if let Some(level) = Level::from_keyword(token) {
            return level;
        }
    }
    Level::Other
}

// ─── Normalisation ────────────────────────────────────────────────────────────

/// Signatures longer than this are truncated so one giant line can't dominate.
const MAX_SIGNATURE_LEN: usize = 160;

/// Collapse a line into a stable signature by replacing volatile tokens with
/// placeholders, so structurally-identical messages group together.
fn normalize_signature(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for (i, token) in line.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&normalize_token(token));
    }
    if out.chars().count() > MAX_SIGNATURE_LEN {
        out = out.chars().take(MAX_SIGNATURE_LEN).collect();
        out.push('\u{2026}');
    }
    out
}

fn normalize_token(token: &str) -> String {
    if looks_like_uuid(token) {
        return "<uuid>".to_string();
    }
    if let Some(rest) = token.strip_prefix("0x") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_hexdigit()) {
            return "<hex>".to_string();
        }
    }
    if token.starts_with('/') && token.len() > 1 {
        return "<path>".to_string();
    }
    if is_timestamp(token) {
        return "<ts>".to_string();
    }
    collapse_digit_runs(token)
}

fn looks_like_uuid(token: &str) -> bool {
    // 8-4-4-4-12 hex with dashes. Strip common trailing punctuation first.
    let t = token.trim_end_matches([',', ';', ')', ']', '"', '\'']);
    let parts: Vec<&str> = t.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let lens = [8, 4, 4, 4, 12];
    parts
        .iter()
        .zip(lens)
        .all(|(p, n)| p.len() == n && p.chars().all(|c| c.is_ascii_hexdigit()))
}

/// A token carrying a date/time: has a digit and a date/time separator. Catches
/// `2026-06-09T10:11:12.345Z`, `10:11:12`, and `2026-06-09` alike.
fn is_timestamp(token: &str) -> bool {
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_sep = token.contains(':') || (token.contains('-') && token.contains('T'));
    has_digit && has_sep
}

/// Replace each run of ASCII digits with `<n>`, so `tool-42` → `tool-<n>` and
/// `count=1500` → `count=<n>`.
fn collapse_digit_runs(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    let mut in_run = false;
    for c in token.chars() {
        if c.is_ascii_digit() {
            if !in_run {
                out.push_str("<n>");
                in_run = true;
            }
        } else {
            out.push(c);
            in_run = false;
        }
    }
    out
}

/// Keep only the last `n` elements of `v` (the most recent log lines).
fn tail_n(mut v: Vec<String>, n: usize) -> Vec<String> {
    if v.len() > n {
        v.drain(0..v.len() - n);
    }
    v
}

// ─── Deterministic digest ─────────────────────────────────────────────────────

/// One recurring signature with its level and occurrence count.
struct PatternHit {
    level: Level,
    signature: String,
    count: usize,
}

/// The deterministic result of analysing a batch of log lines.
struct Digest {
    source_label: String,
    window_label: String,
    total_lines: usize,
    level_counts: BTreeMap<Level, usize>,
    /// Recurring signatures (count ≥ 2), most frequent first.
    patterns: Vec<PatternHit>,
}

/// The maximum number of recurring patterns to surface.
const TOP_PATTERNS: usize = 10;

/// Analyse a batch of raw log lines into a [`Digest`]. When `cutoff` is set,
/// lines whose parsed timestamp is older are dropped (lines without a timestamp
/// are kept — we never drop what we can't date).
fn analyze_lines(
    lines: &[String],
    cutoff: Option<DateTime<Utc>>,
    source_label: &str,
    window_label: &str,
) -> Digest {
    let mut level_counts: BTreeMap<Level, usize> = BTreeMap::new();
    // Group by signature, remembering the level of the first occurrence.
    let mut groups: BTreeMap<String, (Level, usize)> = BTreeMap::new();

    let mut total = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let parsed = parse_line(line);
        if let (Some(cut), Some(ts)) = (cutoff, parsed.ts) {
            if ts < cut {
                continue;
            }
        }
        total += 1;
        *level_counts.entry(parsed.level).or_insert(0) += 1;
        let entry = groups.entry(parsed.signature).or_insert((parsed.level, 0));
        entry.1 += 1;
    }

    let mut patterns: Vec<PatternHit> = groups
        .into_iter()
        .filter(|(_, (_, count))| *count >= 2)
        .map(|(signature, (level, count))| PatternHit {
            level,
            signature,
            count,
        })
        .collect();
    // Most frequent first; ties broken by severity, then signature for stability.
    patterns.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then(a.level.cmp(&b.level))
            .then(a.signature.cmp(&b.signature))
    });
    patterns.truncate(TOP_PATTERNS);

    Digest {
        source_label: source_label.to_string(),
        window_label: window_label.to_string(),
        total_lines: total,
        level_counts,
        patterns,
    }
}

impl Digest {
    fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Analyzed {} line{} from {} ({})\n",
            self.total_lines,
            if self.total_lines == 1 { "" } else { "s" },
            self.source_label,
            self.window_label,
        ));

        // Per-level totals, severe first, only the levels actually seen.
        let parts: Vec<String> = [
            Level::Error,
            Level::Warn,
            Level::Info,
            Level::Debug,
            Level::Trace,
            Level::Other,
        ]
        .into_iter()
        .filter_map(|lvl| {
            self.level_counts
                .get(&lvl)
                .map(|c| format!("{}: {c}", lvl.label()))
        })
        .collect();
        if parts.is_empty() {
            s.push_str("  (no log lines in this window)");
            return s;
        }
        s.push_str(&format!("  {}\n", parts.join("   ")));

        if self.patterns.is_empty() {
            s.push_str("\nNo recurring patterns — every line was distinct after normalization.");
        } else {
            s.push_str("\nTop recurring patterns:\n");
            for p in &self.patterns {
                s.push_str(&format!(
                    "  \u{00d7}{:<4} [{}] {}\n",
                    p.count,
                    p.level.label(),
                    p.signature
                ));
            }
            // Trim the trailing newline for a clean join with the summary.
            s.pop();
        }
        s
    }
}

// ─── Optional LLM narration (CLI only) ─────────────────────────────────────────

/// Narrate a rendered digest in plain language. The prompt is strictly
/// grounded: the model is told to summarise only what the digest contains and
/// never to invent log lines or causes. Returns an error (not a panic) when no
/// provider is reachable, so the caller can degrade to the digest alone.
///
/// This is the optional CLI layer on top of [`analyze`]; the `logs.analyze`
/// capability returns the digest only and lets the reasoner narrate.
pub async fn narrate(config: &BrainConfig, digest: &str) -> Result<String> {
    let provider = cortex::llm::select_provider(&config.llm)
        .await
        .map_err(|e| anyhow::anyhow!("no LLM provider reachable: {e}"))?;

    let system = "You are summarizing a log-analysis digest for an operator. \
        Describe ONLY what the digest below contains — the level counts and the \
        recurring patterns with their occurrence counts. Do NOT invent log lines, \
        error messages, root causes, or fixes that are not present in the digest. \
        If the digest shows no errors or warnings, say the logs look healthy. \
        Keep it to 2–4 short sentences, plain and factual.";

    let user = format!("Log-analysis digest:\n\n{digest}");

    let messages = [
        cortex::llm::Message::system(system),
        cortex::llm::Message::user(user),
    ];
    let response = provider
        .generate(&messages)
        .await
        .map_err(|e| anyhow::anyhow!("provider call failed: {e}"))?;

    let text = response.content.trim().to_string();
    if text.is_empty() {
        anyhow::bail!("provider returned an empty summary");
    }
    Ok(text)
}

// ─── Since parsing ────────────────────────────────────────────────────────────

/// Parse a `<n>{m|h|d}` window into `(amount, unit)`.
fn parse_since_parts(since: &str) -> Result<(u64, char)> {
    let since = since.trim();
    let unit = since
        .chars()
        .last()
        .filter(|c| matches!(c, 'm' | 'h' | 'd'))
        .ok_or_else(|| {
            anyhow::anyhow!("invalid since '{since}': expected <n>{{m|h|d}}, e.g. 30m, 1h, 2d")
        })?;
    let amount: u64 = since[..since.len() - 1].parse().map_err(|_| {
        anyhow::anyhow!(
            "invalid since '{since}': '{}' is not a number",
            &since[..since.len() - 1]
        )
    })?;
    if amount == 0 {
        anyhow::bail!("invalid since '{since}': the amount must be greater than zero");
    }
    Ok((amount, unit))
}

/// Parse a `<n>{m|h|d}` window into a `chrono::Duration`.
fn parse_since(since: &str) -> Result<chrono::Duration> {
    let (amount, unit) = parse_since_parts(since)?;
    let amount = amount as i64;
    Ok(match unit {
        'm' => chrono::Duration::minutes(amount),
        'h' => chrono::Duration::hours(amount),
        _ => chrono::Duration::days(amount),
    })
}

// ─── cortex backend wiring ──────────────────────────────────────────────────

/// The [`cortex::actions::LogAnalysisBackend`] implementation. Holds a snapshot
/// of the config taken at boot and runs the deterministic [`analyze`] on each
/// dispatch, returning the digest. The chat tool-loop dispatches `logs.analyze`
/// through this; the CLI calls [`analyze`]/[`narrate`] directly.
pub struct LogAnalysis {
    config: BrainConfig,
}

impl LogAnalysis {
    pub fn new(config: BrainConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl cortex::actions::LogAnalysisBackend for LogAnalysis {
    async fn analyze(
        &self,
        system: bool,
        since: &str,
        lines: usize,
    ) -> Result<String, cortex::actions::ActionError> {
        let source = if system {
            LogSource::System
        } else {
            LogSource::Brain
        };
        analyze(&self.config, source, since, lines)
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_volatile_tokens() {
        assert_eq!(normalize_token("0xdeadbeef"), "<hex>");
        assert_eq!(normalize_token("/Users/x/.brain/db.sqlite"), "<path>");
        assert_eq!(
            normalize_token("550e8400-e29b-41d4-a716-446655440000"),
            "<uuid>"
        );
        assert_eq!(normalize_token("2026-06-09T10:11:12.345Z"), "<ts>");
        assert_eq!(normalize_token("10:11:12"), "<ts>");
        assert_eq!(normalize_token("tool-42"), "tool-<n>");
        assert_eq!(normalize_token("count=1500"), "count=<n>");
        assert_eq!(normalize_token("provider"), "provider");
    }

    #[test]
    fn uuid_detection_tolerates_trailing_punctuation() {
        assert!(looks_like_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(looks_like_uuid("550e8400-e29b-41d4-a716-446655440000,"));
        assert!(!looks_like_uuid("not-a-uuid"));
        assert!(!looks_like_uuid("550e8400-e29b-41d4-a716")); // only 4 groups
    }

    #[test]
    fn parses_real_pretty_line() {
        // The exact shape the daemon writes (verified against ~/.brain/logs).
        let line = "2026-06-09T08:39:16.314126Z  INFO brainos_storage::sqlite: \
                    SQLite database opened at /Users/x/.brain/db/brain.db (pool size 8)";
        let p = parse_line(line);
        assert_eq!(p.level, Level::Info);
        assert!(p.ts.is_some(), "leading RFC3339 token should parse");
        // The ts + level tokens are stripped; the signature reads as target: msg.
        assert!(p.signature.starts_with("brainos_storage::sqlite:"));
        assert!(!p.signature.contains("<ts>"));
        assert!(!p.signature.contains("INFO"));
        assert!(p.signature.contains("<path>"));
        assert!(p.signature.contains("pool size <n>"));
    }

    #[test]
    fn parses_json_line() {
        let line = r#"{"timestamp":"2026-06-09T08:39:16.314126Z","level":"WARN","fields":{"message":"breaker open for tool abc-1"},"target":"brainos_resilience"}"#;
        let p = parse_line(line);
        assert_eq!(p.level, Level::Warn);
        assert!(p.ts.is_some());
        // Signature is built from target + message, not the JSON scaffolding.
        assert!(p.signature.contains("brainos_resilience"));
        assert!(p.signature.contains("breaker open for tool"));
        assert!(p.signature.contains("<n>"), "the id should be normalized");
        assert!(!p.signature.contains("timestamp"));
    }

    #[test]
    fn json_lines_with_volatile_ids_group_together() {
        let lines: Vec<String> = (1..=3)
            .map(|i| {
                format!(
                    r#"{{"timestamp":"2026-06-09T10:00:0{i}Z","level":"ERROR","fields":{{"message":"breaker open for tool t-{i}"}},"target":"res"}}"#
                )
            })
            .collect();
        let digest = analyze_lines(&lines, None, "test", "all");
        assert_eq!(digest.level_counts.get(&Level::Error), Some(&3));
        assert_eq!(digest.patterns.len(), 1);
        assert_eq!(digest.patterns[0].count, 3);
    }

    #[test]
    fn analyze_groups_structurally_identical_lines() {
        let lines: Vec<String> = vec![
            "2026-06-09T10:00:00Z ERROR circuit breaker open for tool abc-1".into(),
            "2026-06-09T10:00:01Z ERROR circuit breaker open for tool abc-2".into(),
            "2026-06-09T10:00:02Z ERROR circuit breaker open for tool abc-3".into(),
            "2026-06-09T10:00:03Z INFO  signal received from user".into(),
            "2026-06-09T10:00:04Z INFO  consolidation complete".into(),
        ];
        let digest = analyze_lines(&lines, None, "test", "all");

        assert_eq!(digest.total_lines, 5);
        assert_eq!(digest.level_counts.get(&Level::Error), Some(&3));
        assert_eq!(digest.level_counts.get(&Level::Info), Some(&2));

        assert_eq!(digest.patterns.len(), 1);
        let top = &digest.patterns[0];
        assert_eq!(top.count, 3);
        assert_eq!(top.level, Level::Error);
        assert!(top.signature.contains("circuit breaker open for tool"));
        assert!(
            top.signature.contains("<n>"),
            "tool id should be normalized"
        );
    }

    #[test]
    fn analyze_applies_the_since_cutoff() {
        let cutoff = Utc::now() - chrono::Duration::hours(1);
        let fresh = format!("{} ERROR fresh boom", Utc::now().to_rfc3339());
        let lines: Vec<String> = vec![
            "2000-01-01T00:00:00Z ERROR ancient boom".into(),
            fresh,
            "no timestamp but kept".into(),
        ];
        let digest = analyze_lines(&lines, Some(cutoff), "test", "1h");
        // Ancient line dropped; fresh + undatable kept.
        assert_eq!(digest.total_lines, 2);
    }

    #[test]
    fn analyze_reports_no_recurrence_when_all_distinct() {
        let lines: Vec<String> = vec![
            "alpha event one".into(),
            "beta event two".into(),
            "gamma event three".into(),
        ];
        let digest = analyze_lines(&lines, None, "test", "all");
        assert!(digest.patterns.is_empty());
        assert!(digest.render().contains("No recurring patterns"));
    }

    #[test]
    fn resolve_log_files_picks_rotated_files_newest_last() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        for name in [
            "brain.log.2026-06-07",
            "brain.log.2026-06-09",
            "brain.log.2026-06-08",
            "brain.stderr.log", // must be excluded
            "unrelated.txt",    // must be excluded
        ] {
            std::fs::write(p.join(name), "x").unwrap();
        }
        let files = resolve_log_files(&p.to_path_buf());
        let names: Vec<String> = files
            .iter()
            .map(|f| f.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "brain.log.2026-06-07",
                "brain.log.2026-06-08",
                "brain.log.2026-06-09",
            ],
            "rotated files only, sorted chronologically (newest last)"
        );
    }

    #[test]
    fn resolve_log_files_handles_unrotated_single_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("brain.log"), "x").unwrap();
        let files = resolve_log_files(&dir.path().to_path_buf());
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("brain.log"));
    }

    #[test]
    fn tail_n_keeps_the_most_recent() {
        let v: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        assert_eq!(tail_n(v.clone(), 2), vec!["3".to_string(), "4".to_string()]);
        assert_eq!(tail_n(v.clone(), 10), v);
    }

    #[test]
    fn parse_since_accepts_units_and_rejects_garbage() {
        assert_eq!(parse_since("30m").unwrap(), chrono::Duration::minutes(30));
        assert_eq!(parse_since("2h").unwrap(), chrono::Duration::hours(2));
        assert_eq!(parse_since("1d").unwrap(), chrono::Duration::days(1));
        assert!(parse_since("5").is_err()); // no unit
        assert!(parse_since("h").is_err()); // no amount
        assert!(parse_since("0h").is_err()); // zero
        assert!(parse_since("3w").is_err()); // unsupported unit
    }

    #[test]
    fn linux_since_phrases_are_systemd_shaped() {
        assert_eq!(linux_since("1h").unwrap(), "1 hour ago");
        assert_eq!(linux_since("30m").unwrap(), "30 minutes ago");
        assert_eq!(linux_since("2d").unwrap(), "2 days ago");
    }
}
