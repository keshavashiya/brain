//! Governance-category intent handlers: approvals, audit, channel
//! prefs, proactivity — the meta-configuration surface.
//!
//! Variants: [`thalamus::Intent::RespondToApproval`],
//! [`thalamus::Intent::RevokeStandingApproval`],
//! [`thalamus::Intent::PruneAudit`],
//! [`thalamus::Intent::SetChannelPreference`],
//! [`thalamus::Intent::SetProactivity`].

use uuid::Uuid;

use super::dispatch::{GovernanceHandler, HandlerContext, NudgeFn};
use crate::types::*;
use crate::SignalProcessor;

#[async_trait::async_trait]
impl GovernanceHandler for SignalProcessor {
    async fn dispatch_governance(
        &self,
        ctx: HandlerContext<'_>,
        intent: thalamus::Intent,
        prepend_nudges: &NudgeFn<'_>,
    ) -> Result<PipelineResult, SignalError> {
        match intent {
            thalamus::Intent::RespondToApproval { nonce, decision } => {
                self.handle_respond_to_approval(ctx.signal_id, nonce, decision, prepend_nudges)
                    .await
            }
            thalamus::Intent::RevokeStandingApproval { id } => {
                self.handle_revoke_standing_approval(ctx.signal_id, id, prepend_nudges)
                    .await
            }
            thalamus::Intent::PruneAudit { older_than } => {
                self.handle_prune_audit(ctx.signal_id, older_than, prepend_nudges)
                    .await
            }
            thalamus::Intent::SetChannelPreference {
                channel,
                category,
                weight,
                pinned,
            } => {
                self.handle_set_channel_preference(
                    ctx.signal_id,
                    channel,
                    category,
                    weight,
                    pinned,
                    prepend_nudges,
                )
                .await
            }
            thalamus::Intent::SetProactivity { enabled, until } => {
                self.handle_set_proactivity(ctx.signal_id, enabled, until, prepend_nudges)
                    .await
            }
            other => unreachable!(
                "non-governance variant routed to dispatch_governance: {other:?} \
                 (Intent::category() / dispatch table out of sync)"
            ),
        }
    }
}

impl SignalProcessor {
    pub(super) async fn handle_respond_to_approval(
        &self,
        signal_id: Uuid,
        nonce: String,
        decision: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let approved = decision.to_lowercase().contains("approve");

        // Plan-level approval: an ID matching a task in AwaitingApproval
        // phase is a phase-transition request, not a confirm-engine nonce.
        // Approving kicks off execution; rejecting cancels the plan.
        //
        // Resolution order:
        //   1. If the user typed an explicit nonce, try it first.
        //   2. Otherwise (or if it doesn't match a pending plan) look at
        //      `pending_approvals()`. If exactly one plan is pending,
        //      route the bare yes/no to it. Multiple pending → ask the
        //      user to disambiguate. Zero pending → fall through to the
        //      confirm-engine path (per-step approvals).
        if let Some(orch) = &self.orchestrator {
            let mut resolved: Option<String> = None;
            if !nonce.is_empty() {
                if let Some(task) = orch.get_task(&nonce).await {
                    if task.phase == orchestrate::TaskPhase::AwaitingApproval {
                        resolved = Some(nonce.clone());
                    }
                }
            }
            if resolved.is_none() {
                let pending = orch.pending_approvals().await;
                match pending.len() {
                    1 => resolved = Some(pending[0].clone()),
                    n if n > 1 && nonce.is_empty() => {
                        let message = format!(
                            "{n} plans are awaiting approval. Reply `approve <id>` or \
                             `reject <id>` to choose one. Pending: {}",
                            pending.join(", ")
                        );
                        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
                        return Ok(PipelineResult::Complete(resp));
                    }
                    _ => {}
                }
            }

            if let Some(plan_id) = resolved {
                let message = if approved {
                    match orch.execute(&plan_id).await {
                        Ok(summary) => format!("Plan approved.\n\n{summary}"),
                        Err(e) => {
                            format!("Plan approved but execution failed: {e}")
                        }
                    }
                } else {
                    match orch.cancel(&plan_id).await {
                        Ok(_) => "Plan rejected and cancelled.".to_string(),
                        Err(e) => format!("Failed to cancel plan: {e}"),
                    }
                };
                let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
                return Ok(PipelineResult::Complete(resp));
            }
        }

        // Per-step approval: resolve via the confirm engine.
        let message = match &self.confirmation_engine {
            Some(engine) => {
                let dec = if approved {
                    confirm::ApprovalDecision::Approve
                } else {
                    confirm::ApprovalDecision::Reject
                };
                match engine.respond(&nonce, dec).await {
                    Ok(_) => {
                        if approved {
                            format!("Approval {nonce} accepted. Execution resumed.")
                        } else {
                            format!("Approval {nonce} rejected. Action cancelled.")
                        }
                    }
                    // The user replied after the nonce already settled
                    // (timed_out / approved / rejected / NotFound). This
                    // is almost always benign: the chat client buffered
                    // the keystroke during a previous in-flight signal
                    // and flushed it slightly late. Surfacing
                    // "Approval already resolved" as a Brain: error
                    // just confuses the user, so we swallow it quietly
                    // with no body so the renderer skips it.
                    Err(confirm::ConfirmError::AlreadyResolved(_))
                    | Err(confirm::ConfirmError::NotFound(_)) => {
                        let resp = prepend_nudges(SignalResponse::ok(signal_id, String::new()));
                        return Ok(PipelineResult::Complete(resp));
                    }
                    Err(e) => format!("Failed to respond to {nonce}: {e}"),
                }
            }
            None => "Confirmation engine is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_revoke_standing_approval(
        &self,
        signal_id: Uuid,
        id: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.standing_approvals {
            Some(store) => match store.revoke(&id).await {
                Ok(true) => format!("Revoked standing approval `{id}`."),
                Ok(false) => format!("Standing approval `{id}` not found or already revoked."),
                Err(e) => format!("Failed to revoke `{id}`: {e}"),
            },
            None => "Standing-approval store is not wired.".to_string(),
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_prune_audit(
        &self,
        signal_id: Uuid,
        older_than: String,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match &self.audit_trail {
            Some(audit) => match parse_human_duration(&older_than) {
                Ok(duration) => match audit.prune(duration).await {
                    Ok(n) => format!("Pruned {n} entries older than {older_than}"),
                    Err(e) => format!("Failed to prune audit: {e}"),
                },
                Err(e) => format!(
                    "Couldn't parse duration {older_than:?}: {e}. \
                     Try forms like 24h, 7d, 4w, 1y."
                ),
            },
            None => "Audit trail is not wired.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_set_channel_preference(
        &self,
        signal_id: Uuid,
        channel_id: String,
        category: String,
        weight: f32,
        pinned: bool,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let message = match (channel::DeliveryCategory::parse(&category), &self.channel_preferences) {
            (None, _) => format!(
                "Unknown delivery category: {category}. Try: confirm, nudge, report, response, alert.",
            ),
            (_, None) => "Channel preference store not wired in this build.".to_string(),
            (Some(cat), Some(store)) => match store
                .upsert_preference("personal", cat, &channel_id, weight, pinned)
                .await
            {
                Ok(_) => {
                    if weight <= 0.0 && !pinned {
                        format!("Cleared preference for {channel_id} on {category}.")
                    } else {
                        format!(
                            "Set preference: {channel_id} for {category} → weight {:.2}{}.",
                            weight,
                            if pinned { " (pinned)" } else { "" }
                        )
                    }
                }
                Err(e) => format!("Failed to update preference: {e}"),
            },
        };
        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }

    pub(super) async fn handle_set_proactivity(
        &self,
        signal_id: Uuid,
        enabled: bool,
        until: Option<String>,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        if let Some(window) = until.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let message = format!(
                "Time-bounded proactivity pauses (`for {window}`) aren't supported yet — \
                 v0.4.0 only honours plain `enable nudges` / `disable nudges`. \
                 Re-issue without the duration suffix."
            );
            let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
            return Ok(PipelineResult::Complete(resp));
        }

        let previous = self
            .proactivity_enabled
            .swap(enabled, std::sync::atomic::Ordering::SeqCst);
        let startup_enabled = self.config.proactivity.enabled;
        let message = match (previous, enabled) {
            (false, true) if !startup_enabled => {
                "Proactivity flag set to enabled, but the background habit and \
                 open-loop tasks weren't spawned at startup (config had \
                 `proactivity.enabled: false`). Set it `true` in your config and \
                 restart to actually start generating nudges."
                    .to_string()
            }
            (false, true) => "Proactivity enabled. Nudges resume on the next tick.".to_string(),
            (true, false) => "Proactivity disabled. Background habit / open-loop tasks will \
                 skip generation on the next tick. Set `proactivity.enabled: false` \
                 in your config to keep it off across restarts."
                .to_string(),
            (true, true) => "Proactivity already enabled.".to_string(),
            (false, false) => "Proactivity already disabled.".to_string(),
        };

        let resp = prepend_nudges(SignalResponse::ok(signal_id, message));
        Ok(PipelineResult::Complete(resp))
    }
}

/// Parse a short human duration like `24h`, `7d`, `4w`, `2y` into a
/// `chrono::Duration`. Used by intents that take a `older_than` field
/// from the user. Trailing whitespace is tolerated; case-insensitive
/// on the unit suffix; the numeric prefix must be a positive integer.
///
/// Supported units (single letter):
/// - `m` minutes (rarely useful for retention, kept for symmetry)
/// - `h` hours
/// - `d` days
/// - `w` weeks (7 days)
/// - `y` years (365 days — non-leap approximation, fine for prune
///   thresholds where ±1 day doesn't matter)
fn parse_human_duration(input: &str) -> Result<chrono::Duration, String> {
    let s = input.trim();
    if s.is_empty() {
        return Err("empty duration".into());
    }
    let bytes = s.as_bytes();
    let unit = bytes[bytes.len() - 1].to_ascii_lowercase() as char;
    if !matches!(unit, 'm' | 'h' | 'd' | 'w' | 'y') {
        return Err(format!("unknown unit {unit:?}"));
    }
    let n_str = &s[..s.len() - 1];
    let n: i64 = n_str
        .parse()
        .map_err(|_| format!("not a non-negative integer: {n_str:?}"))?;
    if n <= 0 {
        return Err(format!("duration must be positive, got {n}"));
    }
    let dur = match unit {
        'm' => chrono::Duration::try_minutes(n),
        'h' => chrono::Duration::try_hours(n),
        'd' => chrono::Duration::try_days(n),
        'w' => chrono::Duration::try_weeks(n),
        'y' => chrono::Duration::try_days(n.saturating_mul(365)),
        _ => unreachable!(),
    };
    dur.ok_or_else(|| format!("duration out of range: {n}{unit}"))
}

#[cfg(test)]
mod duration_parse_tests {
    use super::parse_human_duration;

    #[test]
    fn parses_common_forms() {
        assert_eq!(
            parse_human_duration("24h").unwrap(),
            chrono::Duration::try_hours(24).unwrap()
        );
        assert_eq!(
            parse_human_duration("7d").unwrap(),
            chrono::Duration::try_days(7).unwrap()
        );
        assert_eq!(
            parse_human_duration("4w").unwrap(),
            chrono::Duration::try_weeks(4).unwrap()
        );
        assert_eq!(
            parse_human_duration("1y").unwrap(),
            chrono::Duration::try_days(365).unwrap()
        );
        assert_eq!(
            parse_human_duration("30m").unwrap(),
            chrono::Duration::try_minutes(30).unwrap()
        );
    }

    #[test]
    fn ignores_trailing_whitespace_and_unit_case() {
        assert_eq!(
            parse_human_duration("30D ").unwrap(),
            chrono::Duration::try_days(30).unwrap()
        );
        assert_eq!(
            parse_human_duration("12H").unwrap(),
            chrono::Duration::try_hours(12).unwrap()
        );
    }

    #[test]
    fn rejects_zero_negative_and_garbage() {
        assert!(parse_human_duration("0d").is_err());
        assert!(parse_human_duration("-5d").is_err());
        assert!(parse_human_duration("").is_err());
        assert!(parse_human_duration("30").is_err());
        assert!(parse_human_duration("30x").is_err());
        assert!(parse_human_duration("abc").is_err());
    }
}

#[cfg(test)]
mod proactivity_tests {
    use crate::types::{PipelineResult, SignalResponse};
    use crate::SignalProcessor;
    use std::sync::atomic::Ordering;
    use uuid::Uuid;

    async fn make_processor() -> SignalProcessor {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        // Pin proactivity disabled — these tests exercise the disabled→
        // enabled toggle path and must not inherit whichever value the
        // shipped default carries (Issue 36 made the YAML default true).
        config.proactivity.enabled = false;
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);
        processor
    }

    fn body_of(result: PipelineResult) -> String {
        match result {
            PipelineResult::Complete(resp) => match resp.response {
                crate::types::ResponseContent::Text(t) => t,
                other => panic!("expected Text response, got {other:?}"),
            },
            _ => panic!("expected PipelineResult::Complete"),
        }
    }

    #[tokio::test]
    async fn toggle_flips_runtime_flag_and_is_visible_in_status() {
        let processor = make_processor().await;
        // Default config has proactivity disabled, so the runtime flag starts false.
        assert!(!processor.proactivity_enabled.load(Ordering::SeqCst));

        let enable = processor
            .handle_set_proactivity(Uuid::new_v4(), true, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(enable);
        // Default config has proactivity disabled at startup, so the response
        // should warn that background tasks weren't spawned.
        assert!(body.contains("weren't spawned at startup"), "got: {body:?}");
        assert!(processor.proactivity_enabled.load(Ordering::SeqCst));

        let status = processor
            .handle_proactivity_status(Uuid::new_v4(), &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(status);
        assert!(
            body.contains("Runtime toggle: enabled"),
            "status missing runtime label: {body:?}"
        );
        assert!(
            body.contains("toggled this session"),
            "status missing drift marker: {body:?}"
        );

        let disable = processor
            .handle_set_proactivity(Uuid::new_v4(), false, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(disable);
        assert!(body.contains("Proactivity disabled"), "got: {body:?}");
        assert!(!processor.proactivity_enabled.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn repeat_toggle_reports_already_state() {
        let processor = make_processor().await;
        // Already disabled by default.
        let result = processor
            .handle_set_proactivity(Uuid::new_v4(), false, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("already disabled"),
            "expected idempotent ack: {body:?}"
        );

        processor.proactivity_enabled.store(true, Ordering::SeqCst);
        let result = processor
            .handle_set_proactivity(Uuid::new_v4(), true, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("already enabled"),
            "expected idempotent ack: {body:?}"
        );
    }

    #[tokio::test]
    async fn enable_when_startup_was_enabled_promises_next_tick() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.proactivity.enabled = true;
        let processor = SignalProcessor::new(config).await.unwrap();
        std::mem::forget(temp);

        // Flip off then back on to land in the (false, true) branch with
        // startup_enabled = true.
        processor.proactivity_enabled.store(false, Ordering::SeqCst);
        let result = processor
            .handle_set_proactivity(Uuid::new_v4(), true, None, &|r: SignalResponse| r)
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("Nudges resume on the next tick"),
            "got: {body:?}"
        );
    }

    #[tokio::test]
    async fn until_window_is_rejected_without_mutating_flag() {
        let processor = make_processor().await;
        let before = processor.proactivity_enabled.load(Ordering::SeqCst);
        let result = processor
            .handle_set_proactivity(
                Uuid::new_v4(),
                false,
                Some("2h".to_string()),
                &|r: SignalResponse| r,
            )
            .await
            .unwrap();
        let body = body_of(result);
        assert!(
            body.contains("aren't supported yet") && body.contains("2h"),
            "got: {body:?}"
        );
        assert_eq!(
            before,
            processor.proactivity_enabled.load(Ordering::SeqCst),
            "rejected request must not flip the flag"
        );
    }
}
