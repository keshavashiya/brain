//! Step-action handlers for [`TaskOrchestrator`].
//!
//! Each [`crate::step::StepAction`] variant has a corresponding executor
//! method on `TaskOrchestrator`. Splitting them out of `orchestrator.rs`
//! keeps the lifecycle/state-machine code there and groups the
//! per-action dispatch logic here. All methods follow the same pattern:
//! degrade gracefully (return a synthetic outcome) when the relevant
//! handle is not attached, so a partially-wired orchestrator still
//! finishes plans rather than hard-failing.

use crate::orchestrator::TaskOrchestrator;
use crate::state::StepOutcome;

impl TaskOrchestrator {
    /// Run a `Research` step against the configured LLM. If no LLM is
    /// attached, degrade to a no-op string outcome so a partially-wired
    /// orchestrator still finishes the plan instead of failing.
    pub(super) async fn execute_research_step(&self, query: &str) -> Result<StepOutcome, String> {
        let Some(llm) = self.llm.as_ref() else {
            return Ok(StepOutcome {
                stdout: format!("Research query: {query}"),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Researched (no LLM attached): {query}"),
            });
        };

        let messages = vec![
            cortex::llm::Message::system(crate::prompts::RESEARCH_SYSTEM),
            cortex::llm::Message::user(query.to_string()),
        ];

        match llm.generate(&messages).await {
            Ok(resp) => Ok(StepOutcome {
                stdout: resp.content.clone(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: summary_first_line(&resp.content, &format!("Research: {query}")),
            }),
            Err(e) => Err(format!("LLM research failed: {e}")),
        }
    }

    /// Run a `Review` step — asks the LLM to critique the named artifact.
    /// Falls back to a no-op outcome if no LLM is attached.
    pub(super) async fn execute_review_step(&self, artifact: &str) -> Result<StepOutcome, String> {
        let Some(llm) = self.llm.as_ref() else {
            return Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![artifact.to_string()],
                summary: format!("Review requested (no LLM attached): {artifact}"),
            });
        };

        let messages = vec![
            cortex::llm::Message::system(crate::prompts::REVIEW_SYSTEM),
            cortex::llm::Message::user(format!("Review this artifact: {artifact}")),
        ];

        match llm.generate(&messages).await {
            Ok(resp) => Ok(StepOutcome {
                stdout: resp.content.clone(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![artifact.to_string()],
                summary: summary_first_line(&resp.content, &format!("Reviewed: {artifact}")),
            }),
            Err(e) => Err(format!("LLM review failed: {e}")),
        }
    }

    /// Deliver a `Notify` step through the channel dispatcher. The
    /// `channel` field of the step is a soft preference — the dispatcher
    /// falls back to learned preferences and initiation channel if the
    /// requested channel is unavailable.
    pub(super) async fn execute_notify_step(
        &self,
        channel: &str,
        message: &str,
    ) -> Result<StepOutcome, String> {
        let Some(dispatcher) = self.dispatcher.as_ref() else {
            return Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Notify (no dispatcher attached): {channel}: {message}"),
            });
        };

        let mut intent = channel::DeliveryIntent::new(
            message,
            channel::DeliveryCategory::Report,
            channel::UrgencyLevel::Normal,
        );
        if !channel.is_empty() && channel != "default" {
            intent = intent.with_preferred(channel);
        }

        match dispatcher.dispatch(intent).await {
            Ok(receipt) => Ok(StepOutcome {
                stdout: format!("Delivered via {} ({})", receipt.channel_id, receipt.reason),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!("Notified via {}: {message}", receipt.channel_id),
            }),
            // No transports configured / no channel matches the intent — surface
            // the message via the orchestrator's task-completion summary instead
            // of failing the step. Replan-on-failure produces Notify steps as
            // its honest "I cannot do this" path; if delivery itself failed
            // here we'd recurse into more Notify steps and exhaust the replan
            // budget, leaving the user with a cascade of "delivery failed"
            // entries hiding the actual blocker message.
            Err(channel::ChannelError::NoChannelAvailable(_, _)) => Ok(StepOutcome {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: None,
                artifacts: vec![],
                summary: format!(
                    "Notify (no external channel — included in this report): {message}"
                ),
            }),
            Err(e) => Err(format!("Notify delivery failed: {e}")),
        }
    }

    /// Execute a command in the sandbox.
    pub(super) async fn execute_sandbox_step(
        &self,
        command: &str,
        workdir: &std::path::Path,
    ) -> Result<StepOutcome, String> {
        let sandbox = match self.sandbox.as_ref() {
            Some(s) => s,
            None => {
                return Err("Sandbox not available".to_string());
            }
        };

        // The sandbox runs argv directly — no shell. Parse with quote
        // handling and detect a trailing `> path` / `>> path` redirect so
        // commands like `find … > /tmp/file` actually capture their output
        // instead of passing `>` as a literal argument to the binary.
        let parsed = parse_sandbox_command(command)?;
        let (binary, args) = parsed
            .argv
            .split_first()
            .ok_or_else(|| "Empty command".to_string())?;

        let cmd =
            sandbox::SandboxCommand::new(binary, args.to_vec()).with_workdir(workdir.to_path_buf());

        match sandbox.run(cmd).await {
            Err(e) => Err(humanize_sandbox_error(binary, &e)),
            Ok(outcome) if outcome.exit_code != 0 => {
                // A non-zero exit is a real failure — surface it so the
                // orchestrator marks the step Failed (and cascade-skips
                // dependents) instead of recording it as Completed with
                // a misleading "succeeded" status. The audit trail picks
                // up the structured failure from the Err path in
                // `execute_step`.
                let stderr_tail: String = outcome
                    .stderr
                    .lines()
                    .rev()
                    .take(5)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut msg = format!("Command failed (exit {}): {command}", outcome.exit_code);
                if !stderr_tail.trim().is_empty() {
                    msg.push_str("\nstderr: ");
                    msg.push_str(&stderr_tail);
                }
                Err(msg)
            }
            Ok(outcome) => {
                // exit_code == 0 — apply any trailing redirect, then
                // record the success.
                let mut artifacts = Vec::new();
                if let Some((path, append)) = &parsed.redirect {
                    let write_result = if *append {
                        use std::io::Write as _;
                        std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                            .and_then(|mut f| f.write_all(outcome.stdout.as_bytes()))
                    } else {
                        std::fs::write(path, outcome.stdout.as_bytes())
                    };
                    if let Err(e) = write_result {
                        return Err(format!("Redirect to {} failed: {e}", path.display()));
                    }
                    artifacts.push(path.to_string_lossy().into_owned());
                }

                Ok(StepOutcome {
                    stdout: outcome.stdout,
                    stderr: outcome.stderr,
                    exit_code: Some(outcome.exit_code),
                    artifacts,
                    summary: format!("Command succeeded: {command}"),
                })
            }
        }
    }

    /// Run a shell-wrapped command in the sandbox. Unlike
    /// [`execute_sandbox_step`], the command is passed verbatim to
    /// `sh -c` so the system shell handles pipes, redirects, escaping,
    /// $VAR expansion, and PATH lookup. The per-binary allowlist is
    /// bypassed for the wrapped command — see [`sandbox::SandboxCommand::shell`]
    /// for the safety story.
    pub(super) async fn execute_shell_step(
        &self,
        command: &str,
        workdir: &std::path::Path,
    ) -> Result<StepOutcome, String> {
        let sandbox = match self.sandbox.as_ref() {
            Some(s) => s,
            None => return Err("Sandbox not available".to_string()),
        };

        let trimmed = command.trim();
        if trimmed.is_empty() {
            return Err("Empty shell command".to_string());
        }

        let cmd = sandbox::SandboxCommand::shell(trimmed).with_workdir(workdir.to_path_buf());

        match sandbox.run(cmd).await {
            Err(e) => Err(humanize_sandbox_error("sh", &e)),
            Ok(outcome) if outcome.exit_code != 0 => {
                let stderr_tail: String = outcome
                    .stderr
                    .lines()
                    .rev()
                    .take(5)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut msg = format!(
                    "Shell command failed (exit {}): {trimmed}",
                    outcome.exit_code
                );
                if !stderr_tail.trim().is_empty() {
                    msg.push_str("\nstderr: ");
                    msg.push_str(&stderr_tail);
                }
                Err(msg)
            }
            Ok(outcome) => Ok(StepOutcome {
                stdout: outcome.stdout,
                stderr: outcome.stderr,
                exit_code: Some(outcome.exit_code),
                artifacts: vec![],
                summary: format!("Shell command succeeded: {trimmed}"),
            }),
        }
    }

    /// Hand the step off to a registered [`delegate::AgentDelegate`].
    /// Failures are run through the configured escalation policy — a
    /// primary hang or launch failure transparently falls over to the
    /// declared fallback chain; anything the chain can't recover becomes
    /// a human escalation recorded as a failed step outcome.
    pub(super) async fn delegate_implement_step(
        &self,
        spec: &str,
        agent: &str,
    ) -> Result<StepOutcome, String> {
        let registry = self.agents.as_ref().ok_or_else(|| {
            "No specialist agents are configured — install one (e.g. claude-code, \
             aider, codex) on your PATH and it will be picked up on the next boot."
                .to_string()
        })?;

        let primary = registry.get(agent).map_err(|_| {
            let known = registry.list();
            if known.is_empty() {
                format!(
                    "Specialist agent '{agent}' isn't available — no agents are \
                     currently registered. Install one on your PATH or pick a \
                     different planner."
                )
            } else {
                format!(
                    "Specialist agent '{agent}' isn't available. Available agents: {}.",
                    known.join(", ")
                )
            }
        })?;

        let task_spec = self.build_delegate_task_spec(spec).await;
        let task = delegate::AgentTask::new(task_spec);
        let outcome = delegate::run_with_escalation(
            primary,
            registry.as_ref(),
            task,
            &self.delegation_policy,
        )
        .await;

        match outcome {
            delegate::EscalationOutcome::Succeeded(result) => {
                self.record_delegate_episode(agent, spec, &result, None)
                    .await;
                Ok(StepOutcome {
                    stdout: result.stdout,
                    stderr: result.stderr,
                    exit_code: result.exit_code,
                    artifacts: result
                        .artifacts
                        .iter()
                        .map(|a| a.reference.clone())
                        .collect(),
                    summary: format!("{agent}: {}", result.summary),
                })
            }
            delegate::EscalationOutcome::Recovered { via, result } => {
                self.record_delegate_episode(agent, spec, &result, Some(&via))
                    .await;
                Ok(StepOutcome {
                    stdout: result.stdout,
                    stderr: result.stderr,
                    exit_code: result.exit_code,
                    artifacts: result
                        .artifacts
                        .iter()
                        .map(|a| a.reference.clone())
                        .collect(),
                    summary: format!("{agent} failed; recovered via {via}: {}", result.summary),
                })
            }
            delegate::EscalationOutcome::EscalateToHuman { reason } => Err(reason),
        }
    }
}

/// Parsed result of a shell-style command string we run in the sandbox.
#[derive(Debug)]
pub(crate) struct ParsedCommand {
    pub argv: Vec<String>,
    /// `Some((path, append))` if a trailing `> path` / `>> path` redirect
    /// was extracted. The orchestrator captures stdout to that path after
    /// the sandbox returns — the sandbox itself only sees argv.
    pub redirect: Option<(std::path::PathBuf, bool)>,
}

/// Tokenize a command string with single/double-quote handling, then
/// split off a trailing `> path` / `>> path` redirect. Reject any other
/// shell metacharacter — pipes, sequencing, backgrounding, command
/// substitution — because the sandbox executes argv directly with no
/// shell to interpret them. A clear error here beats silently passing
/// `|` as an argument to `find`.
pub(crate) fn parse_sandbox_command(command: &str) -> Result<ParsedCommand, String> {
    let tokens = tokenize_shell(command)?;
    if tokens.is_empty() {
        return Err("Empty command".to_string());
    }

    for tok in &tokens {
        // Reject metacharacters that would need a real shell. `>`/`>>` are
        // handled below as a structured redirect, not by passing through.
        if tok == "|" || tok == "||" || tok == "&&" || tok == ";" || tok == "&" || tok == "<" {
            return Err(format!(
                "Shell metacharacter {tok:?} not supported — sandbox runs argv directly. \
                 Split this into multiple steps or capture output via the trailing `> path` form."
            ));
        }
        if tok.contains('`') || tok.contains("$(") {
            return Err(format!(
                "Shell substitution in {tok:?} not supported — sandbox runs argv directly."
            ));
        }
    }

    // Strip a trailing redirect: `… > path` or `… >> path`.
    let mut tokens = tokens;
    let mut redirect = None;
    if tokens.len() >= 3 {
        let last = tokens.len() - 1;
        let op = &tokens[last - 1];
        if op == ">" || op == ">>" {
            let append = op == ">>";
            let path = std::path::PathBuf::from(tokens.remove(last));
            tokens.pop(); // remove the operator token
            redirect = Some((path, append));
        }
    }
    // Reject any remaining redirect operators in non-trailing positions.
    if tokens.iter().any(|t| t == ">" || t == ">>") {
        return Err(
            "Multiple or non-trailing `>` redirects are not supported — sandbox runs argv directly."
                .to_string(),
        );
    }

    Ok(ParsedCommand {
        argv: tokens,
        redirect,
    })
}

/// Minimal POSIX-style tokenizer: whitespace splits, single/double quotes
/// group, backslash escapes the next char (outside single quotes). Enough
/// to handle the shapes the LLM decomposer produces; not a full shell.
fn tokenize_shell(input: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
            }
            '"' if !in_single => {
                in_double = !in_double;
            }
            '\\' if !in_single => {
                if let Some(next) = chars.next() {
                    cur.push(next);
                }
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            // Emit `>` / `>>` / `|` / `||` / `&` / `&&` / `;` / `<` as their own tokens
            // so the redirect/metacharacter checks above can spot them
            // even when the LLM doesn't space-separate them.
            c @ ('>' | '<' | '|' | '&' | ';') if !in_single && !in_double => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                let mut op = String::from(c);
                if let Some(&peek) = chars.peek() {
                    if (c == '>' && peek == '>')
                        || (c == '|' && peek == '|')
                        || (c == '&' && peek == '&')
                    {
                        op.push(peek);
                        chars.next();
                    }
                }
                out.push(op);
            }
            _ => cur.push(c),
        }
    }
    if in_single || in_double {
        return Err("Unterminated quoted string".to_string());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

/// Translate a `SandboxError` into a one-liner the chat user can act on.
/// Strips the internal "binary X not in allowlist" / "workdir Y not in
/// allowlist" framing in favour of plain language plus the actionable
/// config key. The full structured error still hits the audit trail.
pub(crate) fn humanize_sandbox_error(binary: &str, e: &sandbox::SandboxError) -> String {
    use sandbox::SandboxError;
    match e {
        SandboxError::Forbidden(msg) => {
            if msg.contains("not in allowlist") {
                format!(
                    "Cannot run `{binary}` here — it isn't on the sandbox allowlist. \
                     Add it under `security.exec_allowlist` if this command is safe to run."
                )
            } else if msg.contains("explicitly forbidden") {
                format!("`{binary}` is explicitly forbidden by config and cannot run.")
            } else {
                format!("`{binary}` was blocked: {msg}")
            }
        }
        SandboxError::PathNotAllowed(_) => format!(
            "Working directory for `{binary}` isn't allowed. Add the path under \
             `security.allowed_paths` to permit it."
        ),
        SandboxError::Timeout(_) => format!("`{binary}` was killed for taking too long."),
        other => format!("`{binary}` failed to run: {other}"),
    }
}

/// First non-empty line of `s`, truncated to 160 chars; falls back to
/// `default_label` when the LLM returned only whitespace.
pub(crate) fn summary_first_line(s: &str, default_label: &str) -> String {
    let line = s
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if line.is_empty() {
        return default_label.to_string();
    }
    if line.chars().count() > 160 {
        let truncated: String = line.chars().take(157).collect();
        format!("{truncated}…")
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn parses_simple_argv() {
        let p = parse_sandbox_command("find /tmp -type f").unwrap();
        assert_eq!(p.argv, vec!["find", "/tmp", "-type", "f"]);
        assert!(p.redirect.is_none());
    }

    #[test]
    fn extracts_trailing_redirect() {
        let p = parse_sandbox_command("find /tmp -type f > /tmp/list.txt").unwrap();
        assert_eq!(p.argv, vec!["find", "/tmp", "-type", "f"]);
        let (path, append) = p.redirect.unwrap();
        assert_eq!(path, std::path::PathBuf::from("/tmp/list.txt"));
        assert!(!append);
    }

    #[test]
    fn extracts_trailing_append_redirect() {
        let p = parse_sandbox_command("ls /etc >> /tmp/log").unwrap();
        let (_, append) = p.redirect.unwrap();
        assert!(append);
    }

    #[test]
    fn handles_unspaced_redirect() {
        let p = parse_sandbox_command("find /tmp -type f >/tmp/list.txt").unwrap();
        assert_eq!(p.argv, vec!["find", "/tmp", "-type", "f"]);
        assert_eq!(
            p.redirect.unwrap().0,
            std::path::PathBuf::from("/tmp/list.txt")
        );
    }

    #[test]
    fn rejects_pipe() {
        let err = parse_sandbox_command("ls | grep foo").unwrap_err();
        assert!(err.contains("Shell metacharacter"));
    }

    #[test]
    fn rejects_command_substitution() {
        let err = parse_sandbox_command("echo $(pwd)").unwrap_err();
        assert!(err.contains("Shell substitution"));
    }

    #[test]
    fn respects_quotes() {
        let p = parse_sandbox_command(r#"grep "hello world" file.txt"#).unwrap();
        assert_eq!(p.argv, vec!["grep", "hello world", "file.txt"]);
    }

    #[test]
    fn redirect_inside_quotes_is_argument_not_operator() {
        let p = parse_sandbox_command(r#"echo "a > b""#).unwrap();
        assert_eq!(p.argv, vec!["echo", "a > b"]);
        assert!(p.redirect.is_none());
    }

    #[test]
    fn rejects_redirect_in_middle() {
        let err = parse_sandbox_command("foo > bar baz").unwrap_err();
        assert!(err.contains("non-trailing"));
    }
}
