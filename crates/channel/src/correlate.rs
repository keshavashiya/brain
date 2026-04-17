//! Cross-channel confirmation correlation.
//!
//! The [`ConfirmationCorrelator`] parses user messages arriving on *any*
//! channel (CLI, HTTP, WS, relay gateway) and — if the message is an
//! approve/reject command with a nonce — forwards the decision to the
//! [`confirm::ConfirmationEngine`]. That lets a user request an approval
//! via Telegram and respond via Slack: the nonce carries correlation,
//! not the transport.
//!
//! Grammar accepted (all case-insensitive, punctuation tolerant):
//! - `approve <nonce>` · `yes <nonce>` · `ok <nonce>` · `confirm <nonce>`
//! - `/approve <nonce>` (Telegram-style slash commands)
//! - `reject <nonce> [reason...]` · `no <nonce> [reason...]`
//!   · `deny <nonce> [reason...]` · `decline <nonce> [reason...]`
//! - Any of the above where the verb appears *after* the nonce, e.g.
//!   `<nonce> approve` — useful for channels that require the verb last.
//!
//! The nonce itself is recognised by its UUID-v4 layout (`8-4-4-4-12` hex).
//! That avoids false positives from free-form text and makes the parser
//! resilient to quoting, brackets, and other channel-specific garnish.

use std::sync::Arc;

use confirm::{ApprovalDecision, ApprovalStatus, ConfirmError, ConfirmationEngine};

use crate::error::ChannelError;

/// A parsed approval command, before it has been dispatched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrelatedCommand {
    Approve {
        nonce: String,
    },
    Reject {
        nonce: String,
        reason: Option<String>,
    },
}

impl CorrelatedCommand {
    pub fn nonce(&self) -> &str {
        match self {
            Self::Approve { nonce } | Self::Reject { nonce, .. } => nonce,
        }
    }

    fn into_decision(self) -> (String, ApprovalDecision) {
        match self {
            Self::Approve { nonce } => (nonce, ApprovalDecision::Approve),
            Self::Reject {
                nonce,
                reason: None,
            } => (nonce, ApprovalDecision::Reject),
            Self::Reject {
                nonce,
                reason: Some(r),
            } => (nonce, ApprovalDecision::RejectWithReason(r)),
        }
    }
}

/// Outcome of trying to correlate an inbound message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrelationOutcome {
    /// Text did not parse as an approval command.
    NoMatch,
    /// Parsed, but the nonce is not a pending approval (typo, already resolved, etc.).
    UnknownNonce { nonce: String },
    /// Parsed, but the approval is already resolved — we do not re-resolve.
    AlreadyResolved { nonce: String },
    /// Parsed and successfully forwarded to the confirmation engine.
    Applied {
        nonce: String,
        approved: bool,
        reason: Option<String>,
    },
    /// Parsed but the confirmation engine rejected the respond() call
    /// (race with timeout, storage error, etc.).
    EngineError { nonce: String, error: String },
}

/// Parses inbound messages and applies recognized approval commands.
pub struct ConfirmationCorrelator {
    engine: Arc<dyn ConfirmationEngine>,
}

impl ConfirmationCorrelator {
    pub fn new(engine: Arc<dyn ConfirmationEngine>) -> Self {
        Self { engine }
    }

    /// Try to parse `text` and, if it looks like an approval command and
    /// the nonce is pending, forward the decision to the engine.
    pub async fn process(&self, text: &str) -> Result<CorrelationOutcome, ChannelError> {
        let Some(cmd) = parse_command(text) else {
            return Ok(CorrelationOutcome::NoMatch);
        };

        let nonce = cmd.nonce().to_string();

        // Probe the engine — avoid a respond() call if the nonce is unknown
        // or already resolved. This keeps the audit trail clean and gives
        // callers a clear outcome to report back to the user.
        match self.engine.status(&nonce).await {
            Ok(ApprovalStatus::Pending { .. }) => {}
            Ok(ApprovalStatus::Resolved { .. }) => {
                return Ok(CorrelationOutcome::AlreadyResolved { nonce });
            }
            Err(ConfirmError::NotFound(_)) => {
                return Ok(CorrelationOutcome::UnknownNonce { nonce });
            }
            Err(e) => return Err(ChannelError::Confirm(e)),
        }

        let (nonce, decision) = cmd.into_decision();
        let approved = matches!(decision, ApprovalDecision::Approve);
        let reason = match &decision {
            ApprovalDecision::RejectWithReason(r) => Some(r.clone()),
            _ => None,
        };

        match self.engine.respond(&nonce, decision).await {
            Ok(()) => Ok(CorrelationOutcome::Applied {
                nonce,
                approved,
                reason,
            }),
            Err(e) => Ok(CorrelationOutcome::EngineError {
                nonce,
                error: e.to_string(),
            }),
        }
    }
}

// ─── Parser ─────────────────────────────────────────────────────────────────

const APPROVE_VERBS: &[&str] = &["approve", "yes", "ok", "okay", "confirm", "accept", "allow"];
const REJECT_VERBS: &[&str] = &["reject", "no", "deny", "decline", "cancel", "abort"];

fn parse_command(text: &str) -> Option<CorrelatedCommand> {
    // Canonical form: strip punctuation characters that usually surround
    // commands (`/approve`, `"approve"`, `[nonce]`, etc.), lowercase,
    // whitespace-tokenize.
    let cleaned: String = text
        .chars()
        .map(|c| match c {
            '/' | ':' | '"' | '\'' | '[' | ']' | '(' | ')' | ',' | '.' | '!' | '?' | '>' => ' ',
            c => c,
        })
        .collect();
    let lower = cleaned.to_ascii_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }

    // Locate (verb_position, kind) — prefer the first verb found.
    let mut verb_pos: Option<usize> = None;
    let mut approve: bool = false;
    for (i, tok) in tokens.iter().enumerate() {
        if APPROVE_VERBS.contains(tok) {
            verb_pos = Some(i);
            approve = true;
            break;
        }
        if REJECT_VERBS.contains(tok) {
            verb_pos = Some(i);
            approve = false;
            break;
        }
    }
    let verb_pos = verb_pos?;

    // Locate the nonce — first token matching UUID layout.
    let nonce_pos = tokens.iter().position(|t| looks_like_uuid_v4(t))?;
    let nonce = tokens[nonce_pos].to_string();

    if approve {
        return Some(CorrelatedCommand::Approve { nonce });
    }

    // Rejection — reason is whatever trailing tokens come after BOTH verb
    // and nonce, joined. Any dedicated `--reason` style flag token is
    // skipped so CLI-flavored inputs still parse cleanly.
    let last_anchor = verb_pos.max(nonce_pos);
    let reason_tokens: Vec<&str> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            if i > last_anchor && *t != "--reason" && *t != "reason" && *t != "-" {
                Some(*t)
            } else {
                None
            }
        })
        .collect();

    let reason = if reason_tokens.is_empty() {
        None
    } else {
        Some(reason_tokens.join(" "))
    };

    Some(CorrelatedCommand::Reject { nonce, reason })
}

/// Cheap check for UUID-v4 layout: `8-4-4-4-12` hex characters.
/// We accept v1..v5 — the engine only cares that it matches a stored nonce.
fn looks_like_uuid_v4(tok: &str) -> bool {
    if tok.len() != 36 {
        return false;
    }
    let bytes = tok.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        let is_hyphen_pos = matches!(i, 8 | 13 | 18 | 23);
        if is_hyphen_pos {
            if *b != b'-' {
                return false;
            }
        } else if !b.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use chrono::Utc;
    use confirm::{ApprovalOutcome, ApprovalSpec};
    use std::sync::Mutex;

    // ── Parser tests (no engine needed) ─────────────────────────────────

    const NONCE: &str = "550e8400-e29b-41d4-a716-446655440000";

    #[test]
    fn approve_simple() {
        let cmd = parse_command(&format!("approve {NONCE}")).unwrap();
        assert_eq!(
            cmd,
            CorrelatedCommand::Approve {
                nonce: NONCE.into()
            }
        );
    }

    #[test]
    fn approve_slash_command() {
        let cmd = parse_command(&format!("/approve {NONCE}")).unwrap();
        assert_eq!(
            cmd,
            CorrelatedCommand::Approve {
                nonce: NONCE.into()
            }
        );
    }

    #[test]
    fn approve_yes_uppercase() {
        let cmd = parse_command(&format!("YES {NONCE}")).unwrap();
        assert_eq!(
            cmd,
            CorrelatedCommand::Approve {
                nonce: NONCE.into()
            }
        );
    }

    #[test]
    fn approve_verb_after_nonce() {
        let cmd = parse_command(&format!("{NONCE} approve")).unwrap();
        assert_eq!(
            cmd,
            CorrelatedCommand::Approve {
                nonce: NONCE.into()
            }
        );
    }

    #[test]
    fn reject_with_reason() {
        let cmd = parse_command(&format!("reject {NONCE} too risky right now")).unwrap();
        match cmd {
            CorrelatedCommand::Reject { nonce, reason } => {
                assert_eq!(nonce, NONCE);
                assert_eq!(reason.as_deref(), Some("too risky right now"));
            }
            _ => panic!("expected Reject"),
        }
    }

    #[test]
    fn reject_no_reason() {
        let cmd = parse_command(&format!("no {NONCE}")).unwrap();
        assert_eq!(
            cmd,
            CorrelatedCommand::Reject {
                nonce: NONCE.into(),
                reason: None
            }
        );
    }

    #[test]
    fn reject_with_explicit_reason_flag() {
        let cmd = parse_command(&format!("reject {NONCE} --reason pipeline broken")).unwrap();
        match cmd {
            CorrelatedCommand::Reject { reason, .. } => {
                assert_eq!(reason.as_deref(), Some("pipeline broken"));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn punctuation_tolerant() {
        let cmd = parse_command(&format!("\"approve\", [{NONCE}].")).unwrap();
        assert_eq!(
            cmd,
            CorrelatedCommand::Approve {
                nonce: NONCE.into()
            }
        );
    }

    #[test]
    fn no_verb_no_match() {
        assert!(parse_command(&format!("hey there {NONCE}")).is_none());
    }

    #[test]
    fn no_nonce_no_match() {
        assert!(parse_command("approve").is_none());
    }

    #[test]
    fn not_a_uuid_no_match() {
        assert!(parse_command("approve abc123").is_none());
    }

    #[test]
    fn uuid_detector_rejects_wrong_length() {
        assert!(!looks_like_uuid_v4("short"));
        assert!(!looks_like_uuid_v4(&"a".repeat(36)));
    }

    #[test]
    fn uuid_detector_rejects_non_hex() {
        assert!(!looks_like_uuid_v4("zzze8400-e29b-41d4-a716-446655440000"));
    }

    // ── Engine integration (with mock engine) ───────────────────────────

    #[derive(Default)]
    struct MockEngine {
        resolved: Mutex<Vec<(String, ApprovalDecision)>>,
        status_override: Mutex<Option<ApprovalStatus>>,
        force_not_found: Mutex<bool>,
    }

    impl MockEngine {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        fn set_already_resolved(&self) {
            *self.status_override.lock().unwrap() = Some(ApprovalStatus::Resolved {
                outcome: ApprovalOutcome::Approved,
                resolved_at: Utc::now(),
            });
        }

        fn set_not_found(&self) {
            *self.force_not_found.lock().unwrap() = true;
        }

        fn responses(&self) -> Vec<(String, ApprovalDecision)> {
            self.resolved.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ConfirmationEngine for MockEngine {
        async fn request(&self, _spec: ApprovalSpec) -> Result<ApprovalOutcome, ConfirmError> {
            unimplemented!()
        }
        async fn respond(
            &self,
            nonce: &str,
            decision: ApprovalDecision,
        ) -> Result<(), ConfirmError> {
            self.resolved
                .lock()
                .unwrap()
                .push((nonce.to_string(), decision));
            Ok(())
        }
        async fn status(&self, _nonce: &str) -> Result<ApprovalStatus, ConfirmError> {
            if *self.force_not_found.lock().unwrap() {
                return Err(ConfirmError::NotFound("nope".into()));
            }
            if let Some(s) = self.status_override.lock().unwrap().clone() {
                return Ok(s);
            }
            Ok(ApprovalStatus::Pending { since: Utc::now() })
        }
        async fn pending(&self) -> Result<Vec<ApprovalSpec>, ConfirmError> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn applies_approve() {
        let engine = MockEngine::new();
        let correlator = ConfirmationCorrelator::new(engine.clone() as Arc<dyn ConfirmationEngine>);
        let outcome = correlator
            .process(&format!("approve {NONCE}"))
            .await
            .unwrap();
        match outcome {
            CorrelationOutcome::Applied { approved, .. } => assert!(approved),
            o => panic!("unexpected {o:?}"),
        }
        let r = engine.responses();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, NONCE);
        assert!(matches!(r[0].1, ApprovalDecision::Approve));
    }

    #[tokio::test]
    async fn applies_reject_with_reason() {
        let engine = MockEngine::new();
        let correlator = ConfirmationCorrelator::new(engine.clone() as Arc<dyn ConfirmationEngine>);
        let outcome = correlator
            .process(&format!("reject {NONCE} budget exceeded"))
            .await
            .unwrap();
        match outcome {
            CorrelationOutcome::Applied {
                approved, reason, ..
            } => {
                assert!(!approved);
                assert_eq!(reason.as_deref(), Some("budget exceeded"));
            }
            o => panic!("unexpected {o:?}"),
        }
    }

    #[tokio::test]
    async fn no_match_returns_no_match() {
        let engine = MockEngine::new();
        let correlator = ConfirmationCorrelator::new(engine as Arc<dyn ConfirmationEngine>);
        let outcome = correlator.process("just chatting here").await.unwrap();
        assert_eq!(outcome, CorrelationOutcome::NoMatch);
    }

    #[tokio::test]
    async fn unknown_nonce_reported() {
        let engine = MockEngine::new();
        engine.set_not_found();
        let correlator = ConfirmationCorrelator::new(engine as Arc<dyn ConfirmationEngine>);
        let outcome = correlator
            .process(&format!("approve {NONCE}"))
            .await
            .unwrap();
        match outcome {
            CorrelationOutcome::UnknownNonce { nonce } => assert_eq!(nonce, NONCE),
            o => panic!("unexpected {o:?}"),
        }
    }

    #[tokio::test]
    async fn already_resolved_short_circuits() {
        let engine = MockEngine::new();
        engine.set_already_resolved();
        let correlator = ConfirmationCorrelator::new(engine.clone() as Arc<dyn ConfirmationEngine>);
        let outcome = correlator
            .process(&format!("approve {NONCE}"))
            .await
            .unwrap();
        match outcome {
            CorrelationOutcome::AlreadyResolved { nonce } => assert_eq!(nonce, NONCE),
            o => panic!("unexpected {o:?}"),
        }
        // Engine should NOT have been called for respond().
        assert!(engine.responses().is_empty());
    }
}
