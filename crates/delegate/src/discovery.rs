//! Agent auto-discovery — scan `$PATH` for known CLI agents and probe
//! their versions in parallel, so the registry can populate itself
//! without user configuration.
//!
//! Discovery is data-only: it produces a [`DiscoveredBinary`] list.
//! Turning those into [`AgentDelegate`] instances happens in the
//! registry layer, which can also merge user overrides on top.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tokio::task::JoinSet;
use tokio::time::timeout;

use crate::traits::AgentCapabilities;

/// Per-binary version probe deadline. 3s gives Node-based CLIs
/// (gemini-cli, qwen-code, opencode) enough time to cold-start and
/// print --version; faster CLIs still return well under the cap.
pub const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_millis(3000);

/// How a discovered agent should be wired into a runnable delegate.
///
/// Defaults here are best-effort from public CLI conventions — users
/// with unusual flags can override args/stdin through config.
#[derive(Debug, Clone)]
pub struct InvocationTemplate {
    /// Args passed to the binary. `{prompt}` and `{task_id}` are
    /// substituted at spawn time.
    pub args: &'static [&'static str],
    /// Whether the rendered prompt is written to the child's stdin
    /// instead of being templated into `args`.
    pub prompt_via_stdin: bool,
}

/// A static description of one known agent family. The discovery pass
/// tries each `binary_names` entry on `$PATH`; the first hit is probed
/// and the result becomes a [`DiscoveredBinary`].
#[derive(Debug, Clone)]
pub struct AgentFingerprint {
    /// Canonical id used by the registry (e.g. `"aider"`).
    pub id: &'static str,
    /// Candidate binary names, in priority order.
    pub binary_names: &'static [&'static str],
    /// Args passed to the binary to extract a version banner. Reused as
    /// the default `version_args` on the resulting delegate.
    pub version_args: &'static [&'static str],
    /// Default capabilities attached to a discovered agent. The
    /// registry is free to override these from config.
    pub capabilities: AgentCapabilities,
    /// Default invocation shape used when the registry constructs a
    /// runnable delegate from this fingerprint.
    pub invocation: InvocationTemplate,
}

/// Why a candidate binary isn't usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryStatus {
    /// Binary found on `$PATH` and the version probe succeeded.
    Available,
    /// Binary was located but the probe failed (timeout, non-zero exit,
    /// missing runtime deps). The reason is operator-facing.
    Unavailable(String),
}

/// Outcome of discovery for one fingerprinted agent.
#[derive(Debug, Clone)]
pub struct DiscoveredBinary {
    pub agent_id: String,
    /// The name actually found on `$PATH` (e.g. when the fingerprint
    /// listed several aliases for the same binary, the one that matched).
    pub binary_name: String,
    pub path: PathBuf,
    /// First non-empty line of the version probe's stdout, trimmed.
    pub version: Option<String>,
    pub status: DiscoveryStatus,
    pub capabilities: AgentCapabilities,
    pub invocation: InvocationTemplate,
    /// Args used to probe the binary's version. Reused as the default
    /// `version_args` on the resulting delegate's health probe.
    pub version_args: Vec<String>,
}

impl DiscoveredBinary {
    pub fn is_available(&self) -> bool {
        matches!(self.status, DiscoveryStatus::Available)
    }
}

/// Walks directories (usually from `$PATH`) looking for executable
/// files by name. Caches the directory list so repeated lookups
/// during a single discovery pass don't re-read the env var.
#[derive(Debug, Clone)]
pub struct PathScanner {
    dirs: Vec<PathBuf>,
}

impl PathScanner {
    /// Build a scanner from the current process `$PATH`. Empty on
    /// platforms where `PATH` is unset.
    pub fn from_env() -> Self {
        let dirs = std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default();
        Self { dirs }
    }

    pub fn from_dirs(dirs: Vec<PathBuf>) -> Self {
        Self { dirs }
    }

    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// First executable entry matching `name`, walking directories in
    /// `$PATH` order. Skips unreadable directories silently.
    pub fn find_first(&self, name: &str) -> Option<PathBuf> {
        for dir in &self.dirs {
            let candidate = dir.join(name);
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
        None
    }
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(m) => m.is_file() && m.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.is_file())
        .unwrap_or(false)
}

/// Top-level discovery coordinator. Holds the fingerprint table and
/// probe budget; one instance can be reused across SIGHUP refreshes.
#[derive(Debug, Clone)]
pub struct DelegateDiscovery {
    scanner: PathScanner,
    fingerprints: Vec<AgentFingerprint>,
    probe_timeout: Duration,
}

impl DelegateDiscovery {
    /// Default: scan `$PATH`, use the built-in fingerprints, 500ms probe
    /// budget per binary.
    pub fn new() -> Self {
        Self {
            scanner: PathScanner::from_env(),
            fingerprints: default_fingerprints(),
            probe_timeout: DEFAULT_PROBE_TIMEOUT,
        }
    }

    pub fn with_scanner(mut self, scanner: PathScanner) -> Self {
        self.scanner = scanner;
        self
    }

    pub fn with_fingerprints(mut self, fps: Vec<AgentFingerprint>) -> Self {
        self.fingerprints = fps;
        self
    }

    pub fn with_probe_timeout(mut self, d: Duration) -> Self {
        self.probe_timeout = d;
        self
    }

    pub fn fingerprints(&self) -> &[AgentFingerprint] {
        &self.fingerprints
    }

    /// Run the discovery pass. Returns one entry per fingerprint whose
    /// binary was found on `$PATH` — fingerprints with no hit are
    /// omitted (callers treat absence as "not installed"). Unavailable
    /// entries *are* included so the operator can see "installed but
    /// broken" in doctor reports.
    pub async fn discover(&self) -> Vec<DiscoveredBinary> {
        let mut found: Vec<(AgentFingerprint, String, PathBuf)> = Vec::new();
        let mut seen_paths: HashSet<PathBuf> = HashSet::new();

        for fp in &self.fingerprints {
            for name in fp.binary_names {
                if let Some(path) = self.scanner.find_first(name) {
                    // Same binary matched twice (e.g. `claude` and
                    // `claude-code` both pointing at the same file) —
                    // keep the first hit only.
                    if seen_paths.insert(path.clone()) {
                        found.push((fp.clone(), (*name).to_string(), path));
                    }
                    break;
                }
            }
        }

        let mut set: JoinSet<DiscoveredBinary> = JoinSet::new();
        let probe_timeout = self.probe_timeout;
        for (fp, binary_name, path) in found {
            let version_args: Vec<String> = fp.version_args.iter().map(|s| s.to_string()).collect();
            set.spawn(async move {
                let (status, version) = probe(&path, &version_args, probe_timeout).await;
                DiscoveredBinary {
                    agent_id: fp.id.to_string(),
                    binary_name,
                    path,
                    version,
                    status,
                    capabilities: fp.capabilities,
                    invocation: fp.invocation,
                    version_args,
                }
            });
        }

        let mut out = Vec::new();
        while let Some(res) = set.join_next().await {
            match res {
                Ok(b) => out.push(b),
                Err(e) => tracing::warn!(error = %e, "discovery probe task panicked"),
            }
        }
        out.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));
        out
    }
}

impl Default for DelegateDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

async fn probe(
    path: &Path,
    version_args: &[String],
    budget: Duration,
) -> (DiscoveryStatus, Option<String>) {
    let fut = Command::new(path)
        .args(version_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output();

    match timeout(budget, fut).await {
        Ok(Ok(out)) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let v = text
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .map(|l| l.to_string());
            (DiscoveryStatus::Available, v)
        }
        Ok(Ok(out)) => {
            let code = out.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = stderr.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
            let reason = if msg.is_empty() {
                format!("version probe exited {code}")
            } else {
                format!("version probe exited {code}: {}", msg.trim())
            };
            (DiscoveryStatus::Unavailable(reason), None)
        }
        Ok(Err(e)) => (
            DiscoveryStatus::Unavailable(format!("probe spawn failed: {e}")),
            None,
        ),
        Err(_) => (
            DiscoveryStatus::Unavailable(format!("probe timed out after {budget:?}")),
            None,
        ),
    }
}

/// Built-in fingerprints shipped with Brain. Extend this list when a new
/// CLI agent becomes common enough to warrant zero-config wiring.
pub fn default_fingerprints() -> Vec<AgentFingerprint> {
    vec![
        AgentFingerprint {
            id: "claude-code",
            binary_names: &["claude", "claude-code"],
            version_args: &["--version"],
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string(), "plan".to_string()],
                languages: vec!["rust".to_string(), "typescript".to_string()],
                max_concurrency: 1,
                needs_network: true,
            },
            // `-p -` reads the prompt from stdin.
            invocation: InvocationTemplate {
                args: &["-p", "-"],
                prompt_via_stdin: true,
            },
        },
        AgentFingerprint {
            id: "codex",
            binary_names: &["codex", "codex-cli"],
            version_args: &["--version"],
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string()],
                languages: vec![],
                max_concurrency: 1,
                needs_network: true,
            },
            // `codex exec` reads the prompt from stdin.
            invocation: InvocationTemplate {
                args: &["exec", "-"],
                prompt_via_stdin: true,
            },
        },
        AgentFingerprint {
            id: "aider",
            binary_names: &["aider"],
            version_args: &["--version"],
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string()],
                languages: vec![],
                max_concurrency: 1,
                needs_network: true,
            },
            // aider takes its prompt via `--message`, runs non-interactive with `--yes`.
            invocation: InvocationTemplate {
                args: &["--yes", "--no-stream", "--message", "{prompt}"],
                prompt_via_stdin: false,
            },
        },
        AgentFingerprint {
            id: "qwen-code",
            binary_names: &["qwen", "qwen-code"],
            version_args: &["--version"],
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string()],
                languages: vec![],
                max_concurrency: 1,
                needs_network: true,
            },
            // qwen-code follows the `-p -` convention for piped prompts.
            invocation: InvocationTemplate {
                args: &["-p", "-"],
                prompt_via_stdin: true,
            },
        },
        AgentFingerprint {
            id: "gemini-cli",
            binary_names: &["gemini"],
            version_args: &["--version"],
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string()],
                languages: vec![],
                max_concurrency: 1,
                needs_network: true,
            },
            invocation: InvocationTemplate {
                args: &["-p", "-"],
                prompt_via_stdin: true,
            },
        },
        AgentFingerprint {
            id: "opencode",
            binary_names: &["opencode"],
            version_args: &["--version"],
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string()],
                languages: vec![],
                max_concurrency: 1,
                needs_network: true,
            },
            invocation: InvocationTemplate {
                args: &["run", "-"],
                prompt_via_stdin: true,
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_scanner_finds_executable() {
        // Use a directory with a known layout via tempfile so the test
        // doesn't depend on host `$PATH`.
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("fakeagent");
        std::fs::write(&bin, "#!/bin/sh\necho ok\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut p = std::fs::metadata(&bin).unwrap().permissions();
            p.set_mode(0o755);
            std::fs::set_permissions(&bin, p).unwrap();
        }
        let scanner = PathScanner::from_dirs(vec![dir.path().to_path_buf()]);
        assert_eq!(scanner.find_first("fakeagent"), Some(bin));
        assert!(scanner.find_first("nope").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn path_scanner_skips_non_executable_files() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("not-exec");
        std::fs::write(&bin, "hi").unwrap();
        let scanner = PathScanner::from_dirs(vec![dir.path().to_path_buf()]);
        assert!(scanner.find_first("not-exec").is_none());
    }

    #[test]
    fn default_fingerprints_are_non_empty_and_unique() {
        let fps = default_fingerprints();
        assert!(!fps.is_empty());
        let ids: HashSet<_> = fps.iter().map(|f| f.id).collect();
        assert_eq!(ids.len(), fps.len(), "fingerprint ids must be unique");
        for fp in &fps {
            assert!(!fp.binary_names.is_empty(), "{} has no binary names", fp.id);
            assert!(!fp.version_args.is_empty(), "{} has no version args", fp.id);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn discover_reports_available_for_working_probe() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("faux-claude");
        std::fs::write(&bin, "#!/bin/sh\necho 'faux-claude 1.2.3'\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&bin).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&bin, p).unwrap();

        let fp = AgentFingerprint {
            id: "faux",
            binary_names: &["faux-claude"],
            version_args: &["--version"],
            capabilities: AgentCapabilities::default(),
            invocation: InvocationTemplate {
                args: &[],
                prompt_via_stdin: true,
            },
        };
        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_fingerprints(vec![fp])
            .with_probe_timeout(Duration::from_secs(2));
        let results = discovery.discover().await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id, "faux");
        assert_eq!(results[0].status, DiscoveryStatus::Available);
        assert_eq!(results[0].version.as_deref(), Some("faux-claude 1.2.3"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn discover_marks_failing_probe_unavailable() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("broken-agent");
        std::fs::write(&bin, "#!/bin/sh\necho 'bad happened' 1>&2\nexit 2\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&bin).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&bin, p).unwrap();

        let fp = AgentFingerprint {
            id: "broken",
            binary_names: &["broken-agent"],
            version_args: &["--version"],
            capabilities: AgentCapabilities::default(),
            invocation: InvocationTemplate {
                args: &[],
                prompt_via_stdin: true,
            },
        };
        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_fingerprints(vec![fp])
            .with_probe_timeout(Duration::from_secs(2));
        let results = discovery.discover().await;
        assert_eq!(results.len(), 1);
        match &results[0].status {
            DiscoveryStatus::Unavailable(reason) => {
                assert!(reason.contains("exited 2"), "reason was {reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
        assert!(results[0].version.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn discover_times_out_hung_probe() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("hang-agent");
        std::fs::write(&bin, "#!/bin/sh\nsleep 5\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&bin).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&bin, p).unwrap();

        let fp = AgentFingerprint {
            id: "hang",
            binary_names: &["hang-agent"],
            version_args: &["--version"],
            capabilities: AgentCapabilities::default(),
            invocation: InvocationTemplate {
                args: &[],
                prompt_via_stdin: true,
            },
        };
        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_fingerprints(vec![fp])
            .with_probe_timeout(Duration::from_millis(100));
        let results = discovery.discover().await;
        assert_eq!(results.len(), 1);
        match &results[0].status {
            DiscoveryStatus::Unavailable(reason) => {
                assert!(reason.contains("timed out"), "reason was {reason}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn discover_dedupes_when_two_fingerprints_hit_same_file() {
        // Set up two names pointing at the same executable; only the first
        // fingerprint that matches should register it.
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("shared");
        std::fs::write(&bin, "#!/bin/sh\necho 'shared 1.0'\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&bin).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&bin, p).unwrap();

        let fp_a = AgentFingerprint {
            id: "alpha",
            binary_names: &["shared"],
            version_args: &["--version"],
            capabilities: AgentCapabilities::default(),
            invocation: InvocationTemplate {
                args: &[],
                prompt_via_stdin: true,
            },
        };
        let fp_b = AgentFingerprint {
            id: "beta",
            binary_names: &["shared"],
            version_args: &["--version"],
            capabilities: AgentCapabilities::default(),
            invocation: InvocationTemplate {
                args: &[],
                prompt_via_stdin: true,
            },
        };
        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_fingerprints(vec![fp_a, fp_b])
            .with_probe_timeout(Duration::from_secs(2));
        let results = discovery.discover().await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id, "alpha");
    }
}
