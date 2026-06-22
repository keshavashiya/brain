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

use crate::definition::AgentDefinition;

/// Per-binary version probe deadline. 3s gives Node-based CLIs
/// (gemini, qwen, opencode) enough time to cold-start and print
/// --version; faster CLIs still return well under the cap.
pub const DEFAULT_PROBE_TIMEOUT: Duration = Duration::from_millis(3000);

/// Why a candidate binary isn't usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryStatus {
    /// Binary found on `$PATH` and the version probe succeeded.
    Available,
    /// Binary was located but the probe failed (timeout, non-zero exit,
    /// missing runtime deps). The reason is operator-facing.
    Unavailable(String),
}

/// Outcome of discovery for one agent definition.
#[derive(Debug, Clone)]
pub struct DiscoveredBinary {
    /// The full definition that matched — carries id, args, capabilities,
    /// version_args and alias forward to the registry.
    pub definition: AgentDefinition,
    /// The name actually found on `$PATH` (e.g. when the definition listed
    /// several aliases for the same binary, the one that matched).
    pub binary_name: String,
    pub path: PathBuf,
    /// First non-empty line of the version probe's stdout, trimmed.
    pub version: Option<String>,
    pub status: DiscoveryStatus,
}

impl DiscoveredBinary {
    pub fn is_available(&self) -> bool {
        matches!(self.status, DiscoveryStatus::Available)
    }

    /// Canonical id of the matched definition.
    pub fn agent_id(&self) -> &str {
        &self.definition.id
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

/// Top-level discovery coordinator. Holds the agent-definition table and
/// probe budget; one instance can be reused across SIGHUP refreshes.
#[derive(Debug, Clone)]
pub struct DelegateDiscovery {
    scanner: PathScanner,
    definitions: Vec<AgentDefinition>,
    probe_timeout: Duration,
}

impl DelegateDiscovery {
    /// Default: scan `$PATH`, use the embedded built-in definitions, with
    /// the default per-binary probe budget. Use [`with_definitions`] to
    /// supply seeds merged with user `<data_dir>/agents/` overrides.
    ///
    /// [`with_definitions`]: Self::with_definitions
    pub fn new() -> Self {
        Self {
            scanner: PathScanner::from_env(),
            definitions: crate::definition::embedded_definitions(),
            probe_timeout: DEFAULT_PROBE_TIMEOUT,
        }
    }

    pub fn with_scanner(mut self, scanner: PathScanner) -> Self {
        self.scanner = scanner;
        self
    }

    pub fn with_definitions(mut self, defs: Vec<AgentDefinition>) -> Self {
        self.definitions = defs;
        self
    }

    pub fn with_probe_timeout(mut self, d: Duration) -> Self {
        self.probe_timeout = d;
        self
    }

    pub fn definitions(&self) -> &[AgentDefinition] {
        &self.definitions
    }

    /// Run the discovery pass. Returns one entry per *discoverable*
    /// definition whose binary was found on `$PATH` — definitions with no
    /// hit are omitted (callers treat absence as "not installed").
    /// Explicit-binary definitions are skipped here (they're registered
    /// directly). Unavailable entries *are* included so the operator can
    /// see "installed but broken" in doctor reports.
    pub async fn discover(&self) -> Vec<DiscoveredBinary> {
        let mut found: Vec<(AgentDefinition, String, PathBuf)> = Vec::new();
        let mut seen_paths: HashSet<PathBuf> = HashSet::new();

        for def in &self.definitions {
            if !def.is_discoverable() {
                continue;
            }
            for name in &def.binary_names {
                if let Some(path) = self.scanner.find_first(name) {
                    // Same binary matched twice (e.g. `claude` and
                    // `claude-code` both pointing at the same file) —
                    // keep the first hit only.
                    if seen_paths.insert(path.clone()) {
                        found.push((def.clone(), name.clone(), path));
                    }
                    break;
                }
            }
        }

        let mut set: JoinSet<DiscoveredBinary> = JoinSet::new();
        let probe_timeout = self.probe_timeout;
        for (definition, binary_name, path) in found {
            let version_args = definition.version_args.clone();
            set.spawn(async move {
                let (status, version) = probe(&path, &version_args, probe_timeout).await;
                DiscoveredBinary {
                    definition,
                    binary_name,
                    path,
                    version,
                    status,
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
        out.sort_by(|a, b| a.definition.id.cmp(&b.definition.id));
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
    // Linux can return ETXTBSY (errno 26, "Text file busy") when execing a
    // file whose write/close just happened — kernel-side state hasn't fully
    // propagated. Hits in CI right after `std::fs::write` and in real
    // life right after a fresh `cargo install`. A handful of short retries
    // smooths it out without changing observable behavior elsewhere.
    let mut attempt = 0u32;
    loop {
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
                return (DiscoveryStatus::Available, v);
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
                return (DiscoveryStatus::Unavailable(reason), None);
            }
            Ok(Err(e)) if e.raw_os_error() == Some(26) && attempt < 4 => {
                tokio::time::sleep(Duration::from_millis(20 * (attempt as u64 + 1))).await;
                attempt += 1;
                continue;
            }
            Ok(Err(e)) => {
                return (
                    DiscoveryStatus::Unavailable(format!("probe spawn failed: {e}")),
                    None,
                );
            }
            Err(_) => {
                return (
                    DiscoveryStatus::Unavailable(format!("probe timed out after {budget:?}")),
                    None,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal discoverable definition for tests.
    fn def(id: &str, binary_names: &[&str]) -> AgentDefinition {
        AgentDefinition {
            id: id.to_string(),
            alias: None,
            binary_names: binary_names.iter().map(|s| s.to_string()).collect(),
            binary: None,
            version_args: vec!["--version".to_string()],
            args: Vec::new(),
            prompt_via_stdin: true,
            capabilities: crate::traits::AgentCapabilities::default(),
            workdir: None,
        }
    }

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
    fn default_definitions_are_non_empty_and_unique() {
        let defs = DelegateDiscovery::new().definitions().to_vec();
        assert!(!defs.is_empty());
        let ids: HashSet<_> = defs.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids.len(), defs.len(), "definition ids must be unique");
        for d in &defs {
            d.validate().unwrap_or_else(|e| panic!("{}: {e}", d.id));
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

        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_definitions(vec![def("faux", &["faux-claude"])])
            .with_probe_timeout(Duration::from_secs(5));
        let results = discovery.discover().await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id(), "faux");
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

        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_definitions(vec![def("broken", &["broken-agent"])])
            .with_probe_timeout(Duration::from_secs(5));
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

        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_definitions(vec![def("hang", &["hang-agent"])])
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
    async fn discover_dedupes_when_two_definitions_hit_same_file() {
        // Set up two names pointing at the same executable; only the first
        // definition that matches should register it.
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("shared");
        std::fs::write(&bin, "#!/bin/sh\necho 'shared 1.0'\n").unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(&bin).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(&bin, p).unwrap();

        let discovery = DelegateDiscovery::new()
            .with_scanner(PathScanner::from_dirs(vec![dir.path().to_path_buf()]))
            .with_definitions(vec![def("alpha", &["shared"]), def("beta", &["shared"])])
            .with_probe_timeout(Duration::from_secs(5));
        let results = discovery.discover().await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].agent_id(), "alpha");
    }
}
