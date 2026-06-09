//! Action dispatch — tool execution.
//!
//! Dispatches tool calls from LLM: command execution (sandboxed),
//! web search, scheduling, memory operations, and message sending.

use std::sync::Arc;

use thiserror::Error;

mod validation;

#[cfg(test)]
mod tests;

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from action execution.
#[derive(Debug, Error)]
pub enum ActionError {
    #[error("Command not allowed: {0}")]
    CommandNotAllowed(String),

    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Timeout")]
    Timeout,

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ─── Action Types ───────────────────────────────────────────────────────────

/// Available actions/tools.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Execute a shell command (sandboxed).
    ExecuteCommand { command: String, args: Vec<String> },
    /// Search the web.
    WebSearch { query: String },
    /// Schedule a task.
    ScheduleTask {
        description: String,
        cron: Option<String>,
    },
    /// Store a fact in semantic memory.
    StoreFact {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Recall from memory.
    Recall { query: String },
    /// Send a message to an external endpoint (via protocol adapters).
    SendMessage {
        channel: String,
        recipient: String,
        content: String,
    },
    /// Run a read-only network diagnostic probe against a target host.
    NetDiagnostic { probe: NetProbe, target: String },
    /// Run a read-only audit of the security posture.
    SecurityAudit,
    /// Analyse recent logs for recurring patterns (read-only). `system` selects
    /// the OS log source instead of the daemon's own log.
    AnalyzeLogs {
        system: bool,
        since: String,
        lines: usize,
    },
    /// Capture a new system-baseline snapshot (local write).
    BaselineCapture { label: Option<String> },
    /// Diff two baselines, or the latest baseline against live state (read-only).
    BaselineDiff { from: Option<u32>, to: Option<u32> },
    /// List stored baseline snapshots (read-only).
    BaselineList,
}

/// Which network diagnostic to run. Backs the `net.check` / `net.trace` /
/// `net.cert` native capabilities (Issue 139).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetProbe {
    /// DNS resolution + a timed TCP connect to a host[:port].
    Check,
    /// `traceroute` to a host (Unix; privileged child process).
    Trace,
    /// Inspect the TLS certificate chain a host presents.
    Cert,
}

/// Result of an action execution.
#[derive(Debug, Clone)]
pub struct ActionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

/// Normalized memory fact used by action backends.
#[derive(Debug, Clone)]
pub struct MemoryFact {
    pub namespace: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Optional backend that provides real memory read/write operations.
#[async_trait::async_trait]
pub trait MemoryBackend: Send + Sync {
    async fn store_fact(
        &self,
        namespace: &str,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String, ActionError>;

    async fn recall(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<MemoryFact>, ActionError>;
}

/// Structured web-search hit returned by WebSearchBackend.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Optional backend for web search actions.
#[async_trait::async_trait]
pub trait WebSearchBackend: Send + Sync {
    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>, ActionError>;
}

/// Result of fetching a single URL: cleaned, bounded text content the
/// LLM can be given as grounding. `text` is plain text — HTML tags and
/// scripts have been stripped by the backend.
#[derive(Debug, Clone)]
pub struct FetchedPage {
    pub url: String,
    pub title: String,
    pub text: String,
}

/// Optional backend for fetching the body of a URL the user (or an
/// upstream search hit) handed us. Kept separate from `WebSearchBackend`
/// so a deployment can have search without fetch (or vice versa) and so
/// the two contracts can evolve independently.
#[async_trait::async_trait]
pub trait UrlFetchBackend: Send + Sync {
    /// Fetch a single URL. The backend is responsible for timeouts, body
    /// size caps, and HTML-to-text reduction so the returned page is
    /// safe to pass straight into an LLM context window.
    async fn fetch(&self, url: &str) -> Result<FetchedPage, ActionError>;
}

/// Structured scheduling outcome returned by SchedulingBackend.
#[derive(Debug, Clone)]
pub struct ScheduleOutcome {
    pub schedule_id: String,
    pub status: String,
}

/// Optional backend for scheduling actions.
#[async_trait::async_trait]
pub trait SchedulingBackend: Send + Sync {
    async fn schedule(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
    ) -> Result<ScheduleOutcome, ActionError>;
}

/// Structured message-delivery outcome returned by MessageBackend.
#[derive(Debug, Clone)]
pub struct MessageOutcome {
    pub delivery_id: String,
    pub status: String,
}

/// Optional backend for outbound message actions.
#[async_trait::async_trait]
pub trait MessageBackend: Send + Sync {
    async fn send(
        &self,
        channel: &str,
        recipient: &str,
        content: &str,
        namespace: &str,
    ) -> Result<MessageOutcome, ActionError>;
}

/// Optional backend for read-only network diagnostics (Issue 139). Each method
/// runs one probe against `target` and returns a human/LLM-legible report. Kept
/// as its own trait — separate from the egress-oriented `WebSearchBackend` —
/// because diagnostics neither search nor fetch: they resolve, connect, trace,
/// and inspect, and stay wired even when web search is disabled.
#[async_trait::async_trait]
pub trait NetDiagnosticsBackend: Send + Sync {
    /// DNS resolution + timed TCP connect to `target` (host[:port] or URL).
    async fn check(&self, target: &str) -> Result<String, ActionError>;
    /// Trace the network route to `target`.
    async fn trace(&self, target: &str) -> Result<String, ActionError>;
    /// Inspect the TLS certificate chain `target` presents.
    async fn cert(&self, target: &str) -> Result<String, ActionError>;
}

/// Optional backend for a read-only security-posture audit (Issue 140).
/// Inspects the loaded config and returns a rendered, severity-ranked report.
/// No network, no LLM — a deterministic consequence of the configuration.
#[async_trait::async_trait]
pub trait SecurityAuditBackend: Send + Sync {
    async fn audit(&self) -> Result<String, ActionError>;
}

/// Optional backend for read-only log pattern analysis. Reads recent
/// log lines (daemon log, or the OS log when `system`), groups them into
/// recurring signatures, and returns the deterministic digest. Narration by the
/// reasoner is a separate concern — the capability only returns counts.
#[async_trait::async_trait]
pub trait LogAnalysisBackend: Send + Sync {
    async fn analyze(&self, system: bool, since: &str, lines: usize)
        -> Result<String, ActionError>;
}

/// Optional backend for system-baseline capture + drift detection.
/// Each method returns a rendered report. `capture` is a local write (a snapshot
/// file); `diff` and `list` are read-only.
#[async_trait::async_trait]
pub trait BaselineBackend: Send + Sync {
    async fn capture(&self, label: Option<&str>) -> Result<String, ActionError>;
    async fn diff(&self, from: Option<u32>, to: Option<u32>) -> Result<String, ActionError>;
    async fn list(&self) -> Result<String, ActionError>;
}

impl ActionResult {
    /// Create a successful result.
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
        }
    }
}

// ─── Action Dispatcher ──────────────────────────────────────────────────────

/// Configuration for action execution.
#[derive(Debug, Clone)]
pub struct ActionConfig {
    /// Allowed commands for execution.
    pub command_allowlist: Vec<String>,
    /// Timeout for command execution (seconds).
    pub command_timeout_secs: u64,
    /// Enable web search.
    pub enable_web_search: bool,
    /// Enable scheduling.
    pub enable_scheduling: bool,
    /// Enable channel sends.
    pub enable_channel_send: bool,
    /// Default number of hits to request from the web search backend.
    pub web_search_top_k: usize,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            command_allowlist: vec![
                "ls".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "git".to_string(),
                "cargo".to_string(),
                "rustc".to_string(),
                "pwd".to_string(),
            ],
            command_timeout_secs: 30,
            enable_web_search: true,
            enable_scheduling: false,
            enable_channel_send: false,
            web_search_top_k: 5,
        }
    }
}

/// Dispatches actions/tools.
pub struct ActionDispatcher {
    config: ActionConfig,
    memory_backend: Option<Arc<dyn MemoryBackend>>,
    web_search_backend: Option<Arc<dyn WebSearchBackend>>,
    url_fetch_backend: Option<Arc<dyn UrlFetchBackend>>,
    scheduling_backend: Option<Arc<dyn SchedulingBackend>>,
    message_backend: Option<Arc<dyn MessageBackend>>,
    net_diagnostics_backend: Option<Arc<dyn NetDiagnosticsBackend>>,
    security_audit_backend: Option<Arc<dyn SecurityAuditBackend>>,
    log_analysis_backend: Option<Arc<dyn LogAnalysisBackend>>,
    baseline_backend: Option<Arc<dyn BaselineBackend>>,
    /// Sandbox executor that backs `Action::ExecuteCommand` (Issue 121).
    /// When unset the action refuses with an explicit error rather than
    /// silently shelling out via raw `tokio::process::Command`.
    sandbox_executor: Option<Arc<dyn sandbox::SandboxExecutor>>,
    namespace: String,
}

impl ActionDispatcher {
    /// Create a new dispatcher.
    pub fn new(config: ActionConfig) -> Self {
        Self {
            config,
            memory_backend: None,
            web_search_backend: None,
            url_fetch_backend: None,
            scheduling_backend: None,
            message_backend: None,
            net_diagnostics_backend: None,
            security_audit_backend: None,
            log_analysis_backend: None,
            baseline_backend: None,
            sandbox_executor: None,
            namespace: "personal".to_string(),
        }
    }

    /// Create a new dispatcher with a memory backend attached.
    pub fn with_memory_backend(
        config: ActionConfig,
        memory_backend: Arc<dyn MemoryBackend>,
    ) -> Self {
        Self::new(config).with_memory(memory_backend)
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ActionConfig::default())
    }

    /// Attach a memory backend.
    pub fn with_memory(mut self, memory_backend: Arc<dyn MemoryBackend>) -> Self {
        self.memory_backend = Some(memory_backend);
        self
    }

    /// Attach a web-search backend.
    pub fn with_web_search_backend(mut self, backend: Arc<dyn WebSearchBackend>) -> Self {
        self.web_search_backend = Some(backend);
        self
    }

    /// Attach a URL-fetch backend so user-provided links can be enriched
    /// inline with web-search results. Optional — without it, URLs in the
    /// query are still surfaced as part of the search query string but
    /// not fetched.
    pub fn with_url_fetch_backend(mut self, backend: Arc<dyn UrlFetchBackend>) -> Self {
        self.url_fetch_backend = Some(backend);
        self
    }

    /// Attach a scheduling backend.
    pub fn with_scheduling_backend(mut self, backend: Arc<dyn SchedulingBackend>) -> Self {
        self.scheduling_backend = Some(backend);
        self
    }

    /// Attach a message backend.
    pub fn with_message_backend(mut self, backend: Arc<dyn MessageBackend>) -> Self {
        self.message_backend = Some(backend);
        self
    }

    /// Attach a network-diagnostics backend (`net.check`/`trace`/`cert`).
    pub fn with_net_diagnostics_backend(mut self, backend: Arc<dyn NetDiagnosticsBackend>) -> Self {
        self.net_diagnostics_backend = Some(backend);
        self
    }

    /// Attach a security-audit backend (`security.audit`).
    pub fn with_security_audit_backend(mut self, backend: Arc<dyn SecurityAuditBackend>) -> Self {
        self.security_audit_backend = Some(backend);
        self
    }

    /// Attach a log-analysis backend (`logs.analyze`).
    pub fn with_log_analysis_backend(mut self, backend: Arc<dyn LogAnalysisBackend>) -> Self {
        self.log_analysis_backend = Some(backend);
        self
    }

    /// Attach a baseline backend (`baseline.capture`/`diff`/`list`).
    pub fn with_baseline_backend(mut self, backend: Arc<dyn BaselineBackend>) -> Self {
        self.baseline_backend = Some(backend);
        self
    }

    /// Attach the sandbox executor used by `Action::ExecuteCommand`.
    /// Without one wired, the action returns an explicit error instead
    /// of executing — this is the production hardening from Issue 121.
    pub fn with_sandbox_executor(mut self, executor: Arc<dyn sandbox::SandboxExecutor>) -> Self {
        self.sandbox_executor = Some(executor);
        self
    }

    /// Set the default namespace used by action backends.
    pub fn set_namespace(&mut self, namespace: impl Into<String>) {
        self.namespace = namespace.into();
    }

    fn active_namespace(&self) -> &str {
        let trimmed = self.namespace.trim();
        if trimmed.is_empty() {
            "personal"
        } else {
            trimmed
        }
    }

    /// Execute an action.
    pub async fn dispatch(&self, action: &Action) -> ActionResult {
        match action {
            Action::ExecuteCommand { command, args } => self.execute_command(command, args).await,
            Action::WebSearch { query } => self.web_search(query).await,
            Action::ScheduleTask { description, cron } => {
                self.schedule_task(description, cron.as_deref()).await
            }
            Action::StoreFact {
                subject,
                predicate,
                object,
            } => self.store_fact(subject, predicate, object).await,
            Action::Recall { query } => self.recall(query).await,
            Action::SendMessage {
                channel,
                recipient,
                content,
            } => self.send_message(channel, recipient, content).await,
            Action::NetDiagnostic { probe, target } => self.net_diagnostic(*probe, target).await,
            Action::SecurityAudit => self.security_audit().await,
            Action::AnalyzeLogs {
                system,
                since,
                lines,
            } => self.analyze_logs(*system, since, *lines).await,
            Action::BaselineCapture { label } => self.baseline_capture(label.as_deref()).await,
            Action::BaselineDiff { from, to } => self.baseline_diff(*from, *to).await,
            Action::BaselineList => self.baseline_list().await,
        }
    }

    /// Analyse recent logs through the wired [`LogAnalysisBackend`]. Without one
    /// configured this returns an explicit failure rather than silently nothing.
    async fn analyze_logs(&self, system: bool, since: &str, lines: usize) -> ActionResult {
        let Some(backend) = self.log_analysis_backend.as_ref() else {
            return ActionResult::failure("log-analysis backend not configured in this deployment");
        };
        match backend.analyze(system, since, lines).await {
            Ok(report) => ActionResult::success(report),
            Err(e) => ActionResult::failure(e.to_string()),
        }
    }

    /// Capture a baseline snapshot through the wired [`BaselineBackend`].
    async fn baseline_capture(&self, label: Option<&str>) -> ActionResult {
        let Some(backend) = self.baseline_backend.as_ref() else {
            return ActionResult::failure("baseline backend not configured in this deployment");
        };
        match backend.capture(label).await {
            Ok(report) => ActionResult::success(report),
            Err(e) => ActionResult::failure(e.to_string()),
        }
    }

    /// Diff baselines through the wired [`BaselineBackend`].
    async fn baseline_diff(&self, from: Option<u32>, to: Option<u32>) -> ActionResult {
        let Some(backend) = self.baseline_backend.as_ref() else {
            return ActionResult::failure("baseline backend not configured in this deployment");
        };
        match backend.diff(from, to).await {
            Ok(report) => ActionResult::success(report),
            Err(e) => ActionResult::failure(e.to_string()),
        }
    }

    /// List baselines through the wired [`BaselineBackend`].
    async fn baseline_list(&self) -> ActionResult {
        let Some(backend) = self.baseline_backend.as_ref() else {
            return ActionResult::failure("baseline backend not configured in this deployment");
        };
        match backend.list().await {
            Ok(report) => ActionResult::success(report),
            Err(e) => ActionResult::failure(e.to_string()),
        }
    }

    /// Run the security-posture audit through the wired
    /// [`SecurityAuditBackend`]. Without one configured this returns an
    /// explicit failure rather than silently doing nothing.
    async fn security_audit(&self) -> ActionResult {
        let Some(backend) = self.security_audit_backend.as_ref() else {
            return ActionResult::failure(
                "security audit backend not configured in this deployment",
            );
        };
        match backend.audit().await {
            Ok(report) => ActionResult::success(report),
            Err(e) => ActionResult::failure(e.to_string()),
        }
    }

    /// Run a read-only network diagnostic through the wired
    /// [`NetDiagnosticsBackend`]. Without one configured this returns an
    /// explicit failure rather than silently doing nothing.
    async fn net_diagnostic(&self, probe: NetProbe, target: &str) -> ActionResult {
        let Some(backend) = self.net_diagnostics_backend.as_ref() else {
            return ActionResult::failure(
                "network diagnostics backend not configured in this deployment",
            );
        };
        let target = target.trim();
        if target.is_empty() {
            return ActionResult::failure("net diagnostic needs a target host");
        }
        let result = match probe {
            NetProbe::Check => backend.check(target).await,
            NetProbe::Trace => backend.trace(target).await,
            NetProbe::Cert => backend.cert(target).await,
        };
        match result {
            Ok(report) => ActionResult::success(report),
            Err(e) => ActionResult::failure(e.to_string()),
        }
    }

    /// Execute a sandboxed command (Issue 121).
    ///
    /// Two layers of defense:
    /// 1. Dispatcher-level allowlist + argument deny-list (cheap, runs
    ///    before we touch the sandbox).
    /// 2. The wired [`sandbox::SandboxExecutor`] which enforces rlimits,
    ///    platform isolation (macOS Seatbelt / Linux namespaces), and a
    ///    second binary allowlist. Without a sandbox wired we refuse —
    ///    the previous raw `tokio::process::Command` path is gone, so a
    ///    misconfigured dispatcher can no longer shell out unbounded.
    async fn execute_command(&self, command: &str, args: &[String]) -> ActionResult {
        if !self.config.command_allowlist.iter().any(|c| c == command) {
            return ActionResult::failure(format!("Command '{command}' is not in the allowlist"));
        }

        if let Err(reason) = validation::validate_args(command, args) {
            return ActionResult::failure(format!("Blocked: {}", reason));
        }

        let Some(executor) = self.sandbox_executor.as_ref() else {
            tracing::warn!(
                command,
                "execute_command refused — no sandbox executor wired"
            );
            return ActionResult::failure(
                "Sandbox not configured — refusing to execute commands without isolation",
            );
        };

        let timeout = std::time::Duration::from_secs(self.config.command_timeout_secs);
        let sandbox_command = sandbox::SandboxCommand::new(command, args.to_vec())
            .with_workdir(std::env::current_dir().unwrap_or_default())
            .with_timeout(timeout);

        match executor.run(sandbox_command).await {
            Ok(outcome) => {
                if outcome.exit_code == 0 {
                    ActionResult::success(outcome.stdout)
                } else {
                    ActionResult::failure(format!(
                        "Exit code: {}\nstderr: {}",
                        outcome.exit_code, outcome.stderr
                    ))
                }
            }
            Err(sandbox::SandboxError::Timeout(d)) => {
                ActionResult::failure(format!("Command timed out after {:?}", d))
            }
            Err(sandbox::SandboxError::Forbidden(reason)) => {
                ActionResult::failure(format!("Blocked by sandbox: {reason}"))
            }
            Err(sandbox::SandboxError::PathNotAllowed(p)) => {
                ActionResult::failure(format!("Blocked by sandbox (path not allowed): {p}"))
            }
            Err(e) => ActionResult::failure(format!("Sandbox execution failed: {e}")),
        }
    }

    /// Search the web. If the query contains URLs, fetch their bodies
    /// in parallel and append them as a `Linked sources:` block so the
    /// downstream LLM can ground its answer in what the user actually
    /// pasted, not just what the search engine surfaced.
    async fn web_search(&self, query: &str) -> ActionResult {
        if !self.config.enable_web_search {
            return ActionResult::failure("Web search is disabled by config");
        }
        let Some(backend) = &self.web_search_backend else {
            return ActionResult::failure("Web search backend not configured");
        };
        let top_k = self.config.web_search_top_k.max(1);
        let urls = extract_urls(query);

        // Strip the URLs out of the search query so we send the engine
        // the actual semantic question, not a wall of links it will
        // tokenize into noise. If nothing else remains, fall back to
        // searching for the first URL's hostname (which usually still
        // returns the canonical landing page).
        let cleaned = strip_urls(query);
        let search_query = if cleaned.trim().is_empty() {
            urls.first()
                .and_then(|u| url_hostname(u))
                .unwrap_or_else(|| query.to_string())
        } else {
            cleaned
        };

        let search_future = backend.search(&search_query, top_k);
        let fetch_future = self.fetch_urls(&urls);
        let (search_result, fetched) = tokio::join!(search_future, fetch_future);

        let mut out = String::new();
        match search_result {
            Ok(hits) if hits.is_empty() => {
                out.push_str(&format!(
                    "web_search ok query=\"{}\" top_k={} hits=0\n",
                    search_query, top_k
                ));
            }
            Ok(hits) => {
                let lines = hits
                    .iter()
                    .enumerate()
                    .map(|(i, hit)| {
                        format!("{}. {} ({}) - {}", i + 1, hit.title, hit.url, hit.snippet)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                out.push_str(&format!(
                    "web_search ok query=\"{}\" top_k={} hits={}\n{}\n",
                    search_query,
                    top_k,
                    hits.len(),
                    lines
                ));
            }
            Err(e) => {
                // Search failure is not fatal if we managed to fetch the
                // user's pasted URLs — the LLM can still answer from
                // those. Surface the search error inline so the caller
                // can see what happened.
                out.push_str(&format!("web_search error: {e}\n"));
                if fetched.is_empty() {
                    return ActionResult::failure(format!("Web search failed: {e}"));
                }
            }
        }

        if !fetched.is_empty() {
            out.push_str("\nLinked sources (fetched directly):\n");
            for (i, page) in fetched.iter().enumerate() {
                out.push_str(&format!(
                    "--- [{}] {} ({})\n{}\n\n",
                    i + 1,
                    page.title,
                    page.url,
                    page.text
                ));
            }
        }

        ActionResult::success(out.trim_end().to_string())
    }

    /// Fetch up to `MAX_FETCH_URLS` URLs in parallel using the configured
    /// fetch backend. Returns successfully fetched pages only — failures
    /// are logged and dropped so a single bad URL doesn't block the rest.
    async fn fetch_urls(&self, urls: &[String]) -> Vec<FetchedPage> {
        const MAX_FETCH_URLS: usize = 4;
        let Some(fetcher) = &self.url_fetch_backend else {
            return Vec::new();
        };
        if urls.is_empty() {
            return Vec::new();
        }
        let to_fetch: Vec<String> = urls.iter().take(MAX_FETCH_URLS).cloned().collect();
        let futures = to_fetch.into_iter().map(|u| {
            let fetcher = fetcher.clone();
            async move {
                match fetcher.fetch(&u).await {
                    Ok(page) => Some(page),
                    Err(e) => {
                        tracing::warn!(url = %u, error = %e, "URL fetch failed");
                        None
                    }
                }
            }
        });
        futures::future::join_all(futures)
            .await
            .into_iter()
            .flatten()
            .collect()
    }

    /// Schedule a task.
    async fn schedule_task(&self, description: &str, cron: Option<&str>) -> ActionResult {
        if !self.config.enable_scheduling {
            return ActionResult::failure("Scheduling is disabled by config");
        }
        let Some(backend) = &self.scheduling_backend else {
            return ActionResult::failure("Scheduling backend not configured");
        };
        let namespace = self.active_namespace();
        match backend.schedule(description, cron, namespace).await {
            Ok(outcome) => ActionResult::success(format!(
                "schedule_task ok id={} status={} namespace={} cron={} description=\"{}\"",
                outcome.schedule_id,
                outcome.status,
                namespace,
                cron.unwrap_or("none"),
                description
            )),
            Err(e) => ActionResult::failure(format!("Schedule task failed: {e}")),
        }
    }

    /// Store a fact in semantic memory.
    async fn store_fact(&self, subject: &str, predicate: &str, object: &str) -> ActionResult {
        let Some(memory) = &self.memory_backend else {
            return ActionResult::failure("Memory backend not available");
        };
        let namespace = self.active_namespace();

        match memory
            .store_fact(namespace, "action", subject, predicate, object)
            .await
        {
            Ok(id) => ActionResult::success(format!(
                "Fact stored [{}] [{}]: {} {} {}",
                id, namespace, subject, predicate, object
            )),
            Err(e) => ActionResult::failure(format!("Failed to store fact: {e}")),
        }
    }

    /// Recall from memory.
    async fn recall(&self, query: &str) -> ActionResult {
        let Some(memory) = &self.memory_backend else {
            return ActionResult::failure("Memory backend not available");
        };
        let namespace = self.active_namespace();

        match memory.recall(query, 10, Some(namespace)).await {
            Ok(results) if results.is_empty() => ActionResult::success("No matching facts found."),
            Ok(results) => {
                let lines = results
                    .iter()
                    .map(|r| {
                        format!(
                            "[{}] {} {} {} (confidence: {:.2})",
                            r.namespace, r.subject, r.predicate, r.object, r.confidence
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ActionResult::success(format!("Found {} fact(s):\n{}", results.len(), lines))
            }
            Err(e) => ActionResult::failure(format!("Recall failed: {e}")),
        }
    }

    /// Send a message via channel.
    async fn send_message(&self, channel: &str, recipient: &str, content: &str) -> ActionResult {
        if !self.config.enable_channel_send {
            return ActionResult::failure("Channel sending is disabled by config");
        }
        let Some(backend) = &self.message_backend else {
            return ActionResult::failure("Message backend not configured");
        };
        let namespace = self.active_namespace();
        match backend.send(channel, recipient, content, namespace).await {
            Ok(outcome) => ActionResult::success(format!(
                "send_message ok id={} status={} channel={} recipient={} namespace={}",
                outcome.delivery_id, outcome.status, channel, recipient, namespace
            )),
            Err(e) => ActionResult::failure(format!("Send message failed: {e}")),
        }
    }
}

/// Extract `http(s)://` URLs from a free-form text. Strips trailing
/// punctuation that's almost certainly not part of the URL (`.`, `,`,
/// `)`, `]`, `}`, `;`, `'`, `"`).
pub(crate) fn extract_urls(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in text.split(|c: char| c.is_whitespace() || c == '<' || c == '>') {
        let t = token.trim();
        if !(t.starts_with("http://") || t.starts_with("https://")) {
            continue;
        }
        let cleaned = t.trim_end_matches(|c: char| {
            matches!(
                c,
                '.' | ',' | ')' | ']' | '}' | ';' | '\'' | '"' | '!' | '?'
            )
        });
        if cleaned.len() > "https://".len() && !out.iter().any(|u: &String| u == cleaned) {
            out.push(cleaned.to_string());
        }
    }
    out
}

/// Remove `http(s)://...` tokens from `text` so a query passed to a
/// search engine isn't dominated by the link wall.
pub(crate) fn strip_urls(text: &str) -> String {
    text.split_whitespace()
        .filter(|t| !t.starts_with("http://") && !t.starts_with("https://"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Best-effort hostname extraction (no `url` crate dependency). Used as
/// a fallback search query when the user pasted only links and no
/// surrounding question.
pub(crate) fn url_hostname(url: &str) -> Option<String> {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let host = after_scheme.split('/').next().unwrap_or(after_scheme);
    let host = host.split('@').next_back().unwrap_or(host);
    let host = host.split(':').next().unwrap_or(host);
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}
