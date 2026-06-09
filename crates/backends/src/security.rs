//! Security-posture audit — the executor behind `brain security audit` and the
//! `security.audit` native capability (Issue 140).
//!
//! A deterministic, offline pass over the *loaded config* that emits
//! severity-ranked findings about the running posture — network exposure,
//! authentication, egress, secret handling, the execution sandbox surface, and
//! at-rest encryption. Truthful by construction: it reads only `config`, never
//! the network or an LLM, so every finding is a literal consequence of what the
//! config declares.
//!
//! The same pure [`audit`] function backs both surfaces: the CLI calls it
//! directly and prints [`render`]; [`ConfigSecurityAuditor`] wraps it to satisfy
//! the [`cortex::actions::SecurityAuditBackend`] trait the chat tool-loop
//! dispatches `security.audit` through.

use brain::BrainConfig;
use serde::Serialize;

/// Finding severity, ordered most-to-least serious. Derived `Ord` follows
/// declaration order, so sorting ascending puts the worst findings first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl Severity {
    fn label(self) -> &'static str {
        match self {
            Severity::Critical => "CRITICAL",
            Severity::High => "HIGH",
            Severity::Medium => "MEDIUM",
            Severity::Low => "LOW",
            Severity::Info => "INFO",
        }
    }
}

/// One security finding — a stable `id`, a one-line `title`, the `detail`
/// explaining why it matters here, and a concrete `remediation`.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub severity: Severity,
    pub id: &'static str,
    pub title: String,
    pub detail: String,
    pub remediation: String,
}

// ─── the audit ──────────────────────────────────────────────────────────────

/// Inspect `config` and produce the security findings, sorted worst-first.
/// Pure — no I/O, no LLM — so it is deterministic and unit-testable.
pub fn audit(config: &BrainConfig) -> Vec<Finding> {
    let mut f = Vec::new();
    let mut add = |severity, id, title: String, detail: String, remediation: String| {
        f.push(Finding {
            severity,
            id,
            title,
            detail,
            remediation,
        });
    };

    let http = &config.adapters.http;
    let a = &config.adapters;
    let any_listener = http.enabled || a.ws.enabled || a.mcp.enabled || a.grpc.enabled;

    // ── Network exposure ──────────────────────────────────────────────────
    if http.enabled && !is_loopback_host(&http.host) {
        add(
            Severity::High,
            "adapter-exposed",
            format!(
                "HTTP adapter bound to a non-loopback address ({})",
                http.host
            ),
            "The HTTP API is reachable beyond localhost, so anyone who can route \
             to this host can reach it."
                .to_string(),
            "Bind `adapters.http.host` to 127.0.0.1 unless remote access is \
             intended; if it is, ensure API keys, rate limiting, and TLS \
             termination are all in place."
                .to_string(),
        );
    }

    // ── Authentication ────────────────────────────────────────────────────
    if any_listener && config.access.api_keys.is_empty() {
        let exposed = http.enabled && !is_loopback_host(&http.host);
        add(
            if exposed {
                Severity::High
            } else {
                Severity::Medium
            },
            "no-api-keys",
            "Adapters are enabled with no API keys configured".to_string(),
            "With `access.api_keys` empty the adapters fail closed (clients are \
             denied), so the surface is locked rather than open — but it is also \
             unusable, and a future key added without other controls would be the \
             only thing standing between the network and the API."
                .to_string(),
            "Add scoped entries to `access.api_keys` (least privilege: prefer \
             `read`/`write` over `admin`) before exposing any adapter."
                .to_string(),
        );
    }
    for key in &config.access.api_keys {
        if key.key.trim().is_empty() {
            add(
                Severity::High,
                "empty-api-key",
                format!("API key '{}' has an empty secret", key.name),
                "An empty key string authenticates trivially.".to_string(),
                "Set a long, random secret (or remove the entry).".to_string(),
            );
        } else if key.key.trim().len() < 16 {
            add(
                Severity::Medium,
                "weak-api-key",
                format!(
                    "API key '{}' is short ({} chars)",
                    key.name,
                    key.key.trim().len()
                ),
                "Short keys are guessable and weak against brute force.".to_string(),
                "Use at least 32 random characters.".to_string(),
            );
        }
        if key.has_permission("admin") {
            add(
                Severity::Info,
                "admin-api-key",
                format!("API key '{}' grants the `admin` scope", key.name),
                "`admin` implicitly grants every scope (read/write/export).".to_string(),
                "Grant the narrowest scopes a client actually needs.".to_string(),
            );
        }
    }
    if any_listener && !config.access.rate_limit.enabled {
        add(
            Severity::Medium,
            "rate-limit-off",
            "Per-client rate limiting is disabled while adapters are enabled".to_string(),
            "Without rate limiting an abusive or compromised client can hammer the \
             API (brute force, resource exhaustion) unthrottled."
                .to_string(),
            "Set `access.rate_limit.enabled = true`.".to_string(),
        );
    }
    if http.enabled && http.cors {
        add(
            Severity::Medium,
            "cors-enabled",
            "CORS is enabled on the HTTP adapter".to_string(),
            "Permissive CORS lets a browser page on any origin call the API with \
             the visitor's ambient credentials."
                .to_string(),
            "Disable `adapters.http.cors` unless a trusted browser front-end needs \
             it; if it does, scope the allowed origins."
                .to_string(),
        );
    }
    if http.enabled && !is_loopback_host(&http.host) && !http.sse_redact_previews {
        add(
            Severity::Low,
            "sse-previews-exposed",
            "SSE event previews are not redacted on an exposed HTTP adapter".to_string(),
            "An observer with only `read` scope sees full LLM responses and \
             notification bodies on the `/v1/events` stream."
                .to_string(),
            "Set `adapters.http.sse_redact_previews = true` for shared deployments.".to_string(),
        );
    }

    // ── At-rest encryption ────────────────────────────────────────────────
    if !config.encryption.enabled {
        add(
            Severity::Medium,
            "encryption-disabled",
            "At-rest encryption is disabled".to_string(),
            "Stored memory (facts, episodes) is written to disk unencrypted; anyone \
             who can read `~/.brain` can read everything Brain knows."
                .to_string(),
            "Set `encryption.enabled = true` to encrypt the store at rest.".to_string(),
        );
    }

    // ── Secret handling ───────────────────────────────────────────────────
    secret_findings(config, &mut add);

    // ── Execution sandbox surface ─────────────────────────────────────────
    exec_findings(config, &mut add);

    // ── Filesystem read surface ───────────────────────────────────────────
    for p in &config.security.allowed_paths {
        if let Some(what) = path_is_broad(p) {
            add(
                Severity::High,
                "broad-allowed-path",
                format!("Filesystem read is allowed to touch {what} ('{p}')"),
                "Read-only file grounding can be pointed at sensitive locations far \
                 outside the user's project space."
                    .to_string(),
                "Restrict `security.allowed_paths` to the specific project roots \
                 Brain should read (e.g. ~/code)."
                    .to_string(),
            );
        }
    }

    // ── Egress / capability surface (informational) ──────────────────────
    if a.terminal.enabled {
        add(
            Severity::Low,
            "terminal-bridge",
            "The terminal bridge is enabled".to_string(),
            "AI agents can open interactive PTY sessions and run commands through \
             the bridge."
                .to_string(),
            "Disable `adapters.terminal.enabled` if PTY access is not needed.".to_string(),
        );
    }
    if config.actions.web_search.enabled {
        add(
            Severity::Info,
            "web-egress",
            "Outbound web search/fetch is enabled".to_string(),
            "Brain can make outbound HTTP requests on the user's behalf.".to_string(),
            "Leave enabled if external lookups are wanted; disable \
             `actions.web_search.enabled` to keep Brain fully offline."
                .to_string(),
        );
    }
    if config.actions.messaging.enabled {
        add(
            Severity::Info,
            "messaging-egress",
            "Outbound messaging is enabled".to_string(),
            "Brain can deliver messages to configured external channels.".to_string(),
            "Disable `actions.messaging.enabled` to prevent outbound delivery.".to_string(),
        );
    }

    f.sort_by(|x, y| x.severity.cmp(&y.severity).then(x.id.cmp(y.id)));
    f
}

/// Findings about secrets stored inline in YAML (which is typically backed up,
/// version-controlled, and replicated) rather than in a chmod-0600 file.
fn secret_findings(
    config: &BrainConfig,
    add: &mut impl FnMut(Severity, &'static str, String, String, String),
) {
    // Legacy single-provider key.
    #[allow(deprecated)]
    let legacy_key = config.llm.api_key.trim();
    #[allow(deprecated)]
    let legacy_has_file = config.llm.api_key_file.is_some();
    if !legacy_key.is_empty() && !legacy_has_file {
        add(
            Severity::High,
            "plaintext-llm-key",
            "`llm.api_key` is stored in plaintext in the config".to_string(),
            "The YAML config is commonly backed up and replicated, carrying the \
             secret with it."
                .to_string(),
            "Move it to `llm.api_key_file` (a chmod-0600 file) or the credential \
             vault."
                .to_string(),
        );
    }
    for p in &config.llm.providers {
        if !p.api_key.trim().is_empty() && p.api_key_file.is_none() {
            add(
                Severity::High,
                "plaintext-provider-key",
                format!("Provider '{}' stores its api_key in plaintext", p.name),
                "The YAML config is commonly backed up and replicated, carrying the \
                 secret with it."
                    .to_string(),
                format!(
                    "Set `llm.providers[].api_key_file` for '{}' instead.",
                    p.name
                ),
            );
        }
    }
    if !config.actions.web_search.api_key.trim().is_empty() {
        add(
            Severity::Medium,
            "plaintext-search-key",
            "`actions.web_search.api_key` is stored in plaintext".to_string(),
            "The search-provider key lives in the backed-up config.".to_string(),
            "Provide it via the `BRAIN_*` environment instead of YAML where \
             possible."
                .to_string(),
        );
    }
    for r in &config.channel.relays {
        if !r.api_key.trim().is_empty() {
            add(
                Severity::Medium,
                "plaintext-relay-key",
                format!("Relay '{}' stores its api_key in plaintext", r.label),
                "The relay bearer token lives in the backed-up config.".to_string(),
                "Avoid committing the config, or rotate the token if it has been.".to_string(),
            );
        }
    }
}

/// Findings about the command-execution sandbox surface — which binaries the
/// allowlist permits.
fn exec_findings(
    config: &BrainConfig,
    add: &mut impl FnMut(Severity, &'static str, String, String, String),
) {
    let mut shell = Vec::new();
    let mut arbitrary = Vec::new();
    for entry in &config.security.exec_allowlist {
        let name = command_basename(entry);
        if SHELL_OR_PRIV.contains(&name) {
            shell.push(entry.clone());
        } else if ARBITRARY_CODE.contains(&name) {
            arbitrary.push(entry.clone());
        }
    }
    if !shell.is_empty() {
        add(
            Severity::High,
            "exec-shell-allowed",
            format!(
                "Exec allowlist permits a shell / privilege tool: {}",
                shell.join(", ")
            ),
            "A shell (or sudo/su) on the allowlist enables the shell-wrapped \
             execution tier, where the per-binary allowlist is bypassed for the \
             wrapped command — effectively arbitrary command execution (rlimits, \
             network deny, and the forbidden list still apply)."
                .to_string(),
            "Remove shell/privilege entries from `security.exec_allowlist` unless \
             the shell tier is genuinely required."
                .to_string(),
        );
    }
    if !arbitrary.is_empty() {
        add(
            Severity::Medium,
            "exec-interpreter-allowed",
            format!(
                "Exec allowlist permits interpreters/tools that can run arbitrary code: {}",
                arbitrary.join(", ")
            ),
            "Language interpreters and tools like find/awk/xargs can execute \
             arbitrary code or spawn child processes, widening the sandbox surface."
                .to_string(),
            "Drop entries the assistant does not need from \
             `security.exec_allowlist`."
                .to_string(),
        );
    }
}

// ─── classification tables + helpers ─────────────────────────────────────────

/// Shells and privilege-escalation tools — their presence sidesteps the
/// per-binary allowlist via the shell-wrapped execution tier.
const SHELL_OR_PRIV: &[&str] = &[
    "sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "eval", "exec", "sudo", "su", "doas",
];

/// Interpreters and tools that can run arbitrary code or spawn children.
const ARBITRARY_CODE: &[&str] = &[
    "python",
    "python3",
    "ruby",
    "perl",
    "node",
    "npm",
    "npx",
    "php",
    "lua",
    "rscript",
    "osascript",
    "env",
    "xargs",
    "awk",
    "gawk",
    "sed",
    "find",
];

/// The bare command name of an allowlist entry, lowercased — so `/bin/SH` and
/// `sh` both match the tables.
fn command_basename(entry: &str) -> &str {
    entry.trim().rsplit(['/', '\\']).next().unwrap_or("").trim()
}

/// True when `host` is a loopback / not-network-exposed bind address.
fn is_loopback_host(host: &str) -> bool {
    matches!(
        host.trim(),
        "" | "localhost" | "127.0.0.1" | "::1" | "[::1]"
    )
}

/// If `p` resolves to an over-broad root, return a human description of it;
/// otherwise `None`. `$HOME`/`~` are treated as the home root.
fn path_is_broad(p: &str) -> Option<&'static str> {
    let t = p.trim().trim_end_matches('/');
    let t = if t.is_empty() { "/" } else { t };
    if t.contains("..") {
        return Some("a path with `..` traversal");
    }
    match t {
        "" | "~" | "$HOME" => Some("the entire home directory"),
        "/" => Some("the filesystem root"),
        "/etc" => Some("system configuration (/etc)"),
        "/Users" | "/home" => Some("all user home directories"),
        "/root" => Some("the root account's home"),
        "/var" | "/usr" | "/private" | "/System" => Some("a system directory"),
        _ => None,
    }
}

// ─── render ───────────────────────────────────────────────────────────────

/// Render the findings as the human-readable report both surfaces print.
pub fn render(findings: &[Finding]) -> String {
    if findings.is_empty() {
        return "Security audit — no findings. Posture looks clean.\n".to_string();
    }
    let count = |s: Severity| findings.iter().filter(|x| x.severity == s).count();
    let mut out = format!(
        "Security audit — {} finding{} ({} critical, {} high, {} medium, {} low, {} info)\n\n",
        findings.len(),
        if findings.len() == 1 { "" } else { "s" },
        count(Severity::Critical),
        count(Severity::High),
        count(Severity::Medium),
        count(Severity::Low),
        count(Severity::Info),
    );
    for fnd in findings {
        out.push_str(&format!("  [{:<8}] {}\n", fnd.severity.label(), fnd.title));
        out.push_str(&format!("             {}\n", fnd.detail));
        out.push_str(&format!("             → {}\n\n", fnd.remediation));
    }
    out
}

// ─── cortex backend wiring ──────────────────────────────────────────────────

/// The [`cortex::actions::SecurityAuditBackend`] implementation. Holds a
/// snapshot of the config taken at boot and re-runs the pure [`audit`] on each
/// dispatch, returning the rendered report. The chat tool-loop dispatches
/// `security.audit` through this; the CLI calls [`audit`]/[`render`] directly.
pub struct ConfigSecurityAuditor {
    config: BrainConfig,
}

impl ConfigSecurityAuditor {
    pub fn new(config: BrainConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl cortex::actions::SecurityAuditBackend for ConfigSecurityAuditor {
    async fn audit(&self) -> Result<String, cortex::actions::ActionError> {
        Ok(render(&audit(&self.config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_config() -> BrainConfig {
        let mut c = BrainConfig::default();
        // A deliberately tightened posture so the baseline is "no findings",
        // letting each test reintroduce exactly one issue.
        c.encryption.enabled = true;
        c.adapters.http.host = "127.0.0.1".to_string();
        c.adapters.http.cors = false;
        c.adapters.http.enabled = true;
        c.adapters.ws.enabled = false;
        c.adapters.mcp.enabled = false;
        c.adapters.grpc.enabled = false;
        c.adapters.terminal.enabled = false;
        c.access.rate_limit.enabled = true;
        c.access.api_keys = vec![brain::config::ApiKeyConfig {
            key: "0123456789abcdef0123456789abcdef".to_string(),
            name: "local".to_string(),
            permissions: vec!["read".to_string()],
            agent_id: None,
        }];
        c.security.exec_allowlist = vec!["ls".to_string(), "git".to_string()];
        c.security.allowed_paths = vec!["~/code".to_string()];
        c.actions.web_search.enabled = false;
        c.actions.web_search.api_key = String::new();
        c.actions.messaging.enabled = false;
        #[allow(deprecated)]
        {
            c.llm.api_key = String::new();
        }
        c.llm.providers = vec![];
        c.channel.relays = vec![];
        c
    }

    fn ids(findings: &[Finding]) -> Vec<&'static str> {
        findings.iter().map(|f| f.id).collect()
    }

    #[test]
    fn clean_config_has_no_findings() {
        let findings = audit(&clean_config());
        assert!(
            findings.is_empty(),
            "unexpected findings: {:?}",
            ids(&findings)
        );
        assert!(render(&findings).contains("no findings"));
    }

    #[test]
    fn default_config_flags_expected_posture() {
        // The shipped defaults are convenient, not hardened: at-rest encryption
        // off, CORS on, and `sh` on the exec allowlist. The audit surfaces them.
        let findings = audit(&BrainConfig::default());
        let got = ids(&findings);
        assert!(got.contains(&"encryption-disabled"), "{got:?}");
        assert!(got.contains(&"cors-enabled"), "{got:?}");
        assert!(got.contains(&"exec-shell-allowed"), "{got:?}");
    }

    #[test]
    fn non_loopback_bind_is_high() {
        let mut c = clean_config();
        c.adapters.http.host = "0.0.0.0".to_string();
        let findings = audit(&c);
        let exposed = findings.iter().find(|f| f.id == "adapter-exposed").unwrap();
        assert_eq!(exposed.severity, Severity::High);
        // ::1 and localhost stay clean.
        c.adapters.http.host = "::1".to_string();
        assert!(!ids(&audit(&c)).contains(&"adapter-exposed"));
    }

    #[test]
    fn missing_keys_escalate_when_exposed() {
        let mut c = clean_config();
        c.access.api_keys.clear();
        // Loopback + no keys → Medium.
        let medium = audit(&c)
            .into_iter()
            .find(|f| f.id == "no-api-keys")
            .unwrap();
        assert_eq!(medium.severity, Severity::Medium);
        // Exposed + no keys → High.
        c.adapters.http.host = "0.0.0.0".to_string();
        let high = audit(&c)
            .into_iter()
            .find(|f| f.id == "no-api-keys")
            .unwrap();
        assert_eq!(high.severity, Severity::High);
    }

    #[test]
    fn weak_and_empty_keys_flagged() {
        let mut c = clean_config();
        c.access.api_keys[0].key = "short".to_string();
        assert!(ids(&audit(&c)).contains(&"weak-api-key"));
        c.access.api_keys[0].key = "  ".to_string();
        assert!(ids(&audit(&c)).contains(&"empty-api-key"));
    }

    #[test]
    fn plaintext_secrets_flagged() {
        let mut c = clean_config();
        c.llm.providers = vec![brain::config::ProviderEntry {
            name: "groq".to_string(),
            kind: "groq".to_string(),
            base_url: String::new(),
            api_key: "sk-secret-token".to_string(),
            api_key_file: None,
            model: "llama".to_string(),
            preferred_models: vec![],
        }];
        assert!(ids(&audit(&c)).contains(&"plaintext-provider-key"));
        // A file-backed key is fine.
        c.llm.providers[0].api_key_file = Some("/run/secrets/groq".into());
        assert!(!ids(&audit(&c)).contains(&"plaintext-provider-key"));
    }

    #[test]
    fn risky_exec_entries_classified() {
        let mut c = clean_config();
        c.security.exec_allowlist = vec!["ls".into(), "/bin/bash".into(), "python3".into()];
        let findings = audit(&c);
        let got = ids(&findings);
        assert!(got.contains(&"exec-shell-allowed"), "{got:?}");
        assert!(got.contains(&"exec-interpreter-allowed"), "{got:?}");
        // `/bin/bash` matches by basename.
        let shell = findings
            .iter()
            .find(|f| f.id == "exec-shell-allowed")
            .unwrap();
        assert!(shell.title.contains("/bin/bash"));
    }

    #[test]
    fn broad_allowed_paths_flagged() {
        let mut c = clean_config();
        c.security.allowed_paths = vec!["/".into()];
        assert!(ids(&audit(&c)).contains(&"broad-allowed-path"));
        c.security.allowed_paths = vec!["~".into()];
        assert!(ids(&audit(&c)).contains(&"broad-allowed-path"));
        c.security.allowed_paths = vec!["~/projects/../../etc".into()];
        assert!(ids(&audit(&c)).contains(&"broad-allowed-path"));
    }

    #[test]
    fn findings_sorted_worst_first() {
        // High (exposed) should precede Medium (cors) should precede Info.
        let mut c = clean_config();
        c.adapters.http.host = "0.0.0.0".to_string();
        c.adapters.http.cors = true;
        c.actions.web_search.enabled = true;
        let findings = audit(&c);
        let severities: Vec<Severity> = findings.iter().map(|f| f.severity).collect();
        let mut sorted = severities.clone();
        sorted.sort();
        assert_eq!(severities, sorted, "findings must be worst-first");
    }

    #[test]
    fn is_loopback_classification() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("::1"));
        assert!(is_loopback_host(""));
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
    }
}
