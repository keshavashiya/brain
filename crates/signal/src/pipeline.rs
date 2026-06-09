//! Pipeline root for [`SignalProcessor`].
//!
//! Holds orchestration only:
//! - [`SignalProcessor::process`] — the public entry point.
//! - [`SignalProcessor::prepare`] — dispatch by classified intent.
//! - [`SignalProcessor::classify_and_store_facts`] / `match_procedures` —
//!   per-call helpers used during preparation.
//! - [`SignalProcessor::enforce_identity`] /
//!   [`SignalProcessor::confirmation_gate`] — the two pre-handler gates.
//! - [`SignalProcessor::publish_signal_received`] /
//!   [`SignalProcessor::publish_event`] — observability publishes.
//!
//! Per-intent handlers live in the per-category sibling modules — see
//! `pipeline/{inspection,memory,action,lifecycle,governance,capability,conversation}.rs`.
//! Cross-cutting infra is in `pipeline/{cancel,observe,paths}.rs`. The
//! category cut mirrors [`thalamus::IntentCategory`] so dispatch, trait
//! definitions (Issue 111), and tier resolution (Issue 112) can share one
//! cut of the enum.

use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

mod action;
mod cancel;
mod capability;
mod conversation;
mod dispatch;
mod governance;
mod inspection;
mod lifecycle;
mod memory;
mod observe;
mod paths;
mod toolloop;

use dispatch::{
    ActionHandler, CapabilityHandler, ConversationHandler, GovernanceHandler, HandlerContext,
    InspectionHandler, LifecycleHandler, MemoryHandler,
};

pub(crate) use dispatch::IntentAuthorizer;

// `crate::attachment` imports these by their historic
// `crate::pipeline::<name>` paths — keep them re-exported so the sandbox
// gate API doesn't churn while we move code around.
pub(crate) use paths::{
    build_directory_snapshot, build_file_snapshot, expand_user_path, extract_path_tokens,
    friendly_io_error, path_under_any_root, resolve_allowed_roots,
};

use cancel::CancelGuard;

impl SignalProcessor {
    /// Process a signal through the full Brain pipeline.
    ///
    /// Delegates to `prepare()` for pipeline work, then handles LLM generation
    /// for intents that require it (Chat, Recall).
    ///
    /// Routes by intent:
    /// - `StoreFact`  → Amygdala importance → Hippocampus semantic store → confirmation
    /// - `Recall`     → Hippocampus hybrid search → Cortex context assembly → LLM response
    /// - `Chat`       → Hippocampus context → Cortex LLM → Hippocampus episode store
    /// - `Forget`     → search + delete matching facts
    /// - `SystemStatus` → memory counts
    /// - Action intents → ActionDispatcher
    #[tracing::instrument(
        name = "signal.process",
        skip(self, signal),
        fields(
            // `correlation_id` is the signal's id; every log line emitted while
            // processing this signal — and every BrainEvent on the bus — shares
            // it, so one turn can be reconstructed end-to-end (`brain tail
            // --correlation <id>`). Kept alongside `signal_id` for back-compat
            // with existing log filters.
            correlation_id = %signal.id,
            signal_id = %signal.id,
            source = ?signal.source,
            namespace = %signal.namespace
        )
    )]
    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError> {
        // Register a cancellation notify for this signal id. The guard removes
        // it on drop so abort/error paths don't leak entries.
        let signal_id = signal.id;
        let cancel = self.register_cancel(signal_id).await;
        let _cancel_guard = CancelGuard {
            processor: self,
            signal_id,
        };

        self.publish_signal_received(&signal).await;

        // Both prepare() and the LLM-generation branch below are wrapped in a
        // `tokio::select!` against the registered cancel notify so an
        // `Intent::CancelSignal` for this id aborts whichever phase is
        // running. Issue 97: previously only the LlmReady branch was
        // cancel-aware, so a cancel mid-`handle_action` (WebSearch grounding
        // LLM) or mid-`handle_decompose` (orchestrator.plan) had to wait for
        // the await to return on its own.
        let prepared = tokio::select! {
            biased;
            _ = cancel.notified() => {
                return Ok(self.cancelled_response(signal_id, &signal).await);
            }
            r = self.prepare(&signal, None, None) => r?,
        };
        match prepared {
            PipelineResult::Complete(resp) => {
                self.publish_event(&signal, &resp);
                Ok(resp)
            }
            PipelineResult::LlmReady {
                signal_id,
                messages,
                memory_context,
                session_id,
                namespace,
                agent,
                ..
            } => {
                let provider_name = self.llm.name().to_string();
                let gate = crate::budget_guard::check_llm_input(
                    self.cost_budget(),
                    &provider_name,
                    &messages,
                )
                .await;
                let estimated_input = match gate {
                    crate::budget_guard::BudgetGate::Blocked { message } => {
                        let resp = SignalResponse {
                            signal_id,
                            status: ResponseStatus::Ok,
                            response: ResponseContent::Text(message),
                            memory_context,
                            session_id,
                        };
                        self.publish_event(&signal, &resp);
                        return Ok(resp);
                    }
                    crate::budget_guard::BudgetGate::Proceed {
                        estimated_input_tokens,
                    } => estimated_input_tokens,
                };

                let llm_resp = tokio::select! {
                    biased;
                    _ = cancel.notified() => {
                        return Ok(self.cancelled_response(signal_id, &signal).await);
                    }
                    r = self.run_chat_turn(&signal, signal_id, messages) => r?,
                };

                crate::budget_guard::record_llm_usage(
                    self.cost_budget(),
                    &provider_name,
                    llm_resp.usage.as_ref(),
                    estimated_input,
                )
                .await;

                // Store assistant episode for Chat/Recall
                if let Some(sid) = &session_id {
                    self.memory
                        .episodic
                        .store_episode(
                            sid,
                            "assistant",
                            &llm_resp.content,
                            0.5,
                            Some(&namespace),
                            agent.as_deref(),
                        )
                        .map_err(|e| SignalError::Storage(e.to_string()))?;
                }

                let resp = SignalResponse {
                    signal_id,
                    status: ResponseStatus::Ok,
                    response: ResponseContent::Text(llm_resp.content),
                    memory_context,
                    session_id,
                };
                self.publish_event(&signal, &resp);
                Ok(resp)
            }
        }
    }

    /// Prepare the Pipeline up to (but not including) LLM generation.
    ///
    /// Returns either a complete response (for StoreFact, Forget, SystemStatus,
    /// Actions) or assembled LLM messages (for Chat, Recall). The caller can
    /// then choose streaming vs batch LLM generation.
    ///
    /// If `conversation_history` is provided, it is used instead of an empty
    /// history when assembling context (useful for CLI which manages its own).
    pub async fn prepare(
        &self,
        signal: &Signal,
        conversation_history: Option<&[cortex::llm::Message]>,
        progress: Option<tokio::sync::mpsc::Sender<&'static str>>,
    ) -> Result<PipelineResult, SignalError> {
        let signal_id = signal.id;

        let loaded_history = self.hydrate_history(signal, conversation_history);
        let conversation_history: Option<&[cortex::llm::Message]> =
            conversation_history.or(loaded_history.as_deref());

        let pending_notifications = self.drain_pending_notifications();
        let importance = self.importance.score(&signal.content);
        let classification = self
            .classify_and_store_facts(signal, conversation_history)
            .await;
        let procedure_context = self.match_procedures(&signal.content);

        // Identity gate. Runs after classification (so we know the intent)
        // and before any handler executes. Three short-circuit outcomes:
        // EscalateToUser → text response pointing to the pending approval;
        // Deny → error response; otherwise proceed.
        if let Some(early) = self
            .enforce_identity(signal, signal_id, &classification.intent)
            .await
        {
            return Ok(PipelineResult::Complete(early));
        }

        // Inline confirmation gate. Identity says "this principal may
        // attempt this verb"; the confirmation gate says "but for tiers
        // ≥ Destructive, a human (or a standing approval) must consent."
        // Reflex firings travel the same path — they have no way to
        // bypass this checkpoint.
        if let Some(early) = self
            .confirmation_gate(signal, signal_id, &classification.intent)
            .await
        {
            return Ok(PipelineResult::Complete(early));
        }

        // Closure: prepend queued notifications as nudges to a final
        // response. Captured by reference and threaded through every
        // category dispatcher that may produce a `Complete` result.
        let prepend_nudges = move |mut resp: SignalResponse| -> SignalResponse {
            if !pending_notifications.is_empty() {
                // Dedupe by content: the same nudge can land in the outbox
                // more than once (e.g. re-enqueued by proactivity), and two
                // identical `[nudge]` lines in one reply reads as a bug.
                let mut seen = std::collections::HashSet::new();
                let nudge_text: String = pending_notifications
                    .iter()
                    .filter(|n| seen.insert(n.content.as_str()))
                    .map(|n| format!("[nudge] {}", n.content))
                    .collect::<Vec<_>>()
                    .join("\n");
                if let ResponseContent::Text(ref text) = resp.response {
                    resp.response = ResponseContent::Text(format!("{nudge_text}\n\n{text}"));
                }
            }
            resp
        };
        let prepend_nudges: &dispatch::NudgeFn<'_> = &prepend_nudges;

        let ctx = HandlerContext {
            signal_id,
            signal,
            importance,
            conversation_history,
            procedure_context: &procedure_context,
            progress: progress.as_ref(),
        };
        let intent = classification.intent;

        // 7-arm dispatch on category — each sibling module owns the
        // small (≤16-arm) match over the variants in its category.
        // Issue 111: replaces the 37-arm match that used to live here.
        match intent.category() {
            thalamus::IntentCategory::Inspection => {
                self.dispatch_inspection(ctx, intent, prepend_nudges).await
            }
            thalamus::IntentCategory::Memory => {
                self.dispatch_memory(ctx, intent, prepend_nudges).await
            }
            thalamus::IntentCategory::Action => {
                self.dispatch_action(ctx, intent, prepend_nudges).await
            }
            thalamus::IntentCategory::Lifecycle => {
                self.dispatch_lifecycle(ctx, intent, prepend_nudges).await
            }
            thalamus::IntentCategory::Governance => {
                self.dispatch_governance(ctx, intent, prepend_nudges).await
            }
            thalamus::IntentCategory::Capability => {
                self.dispatch_capability(ctx, intent, prepend_nudges).await
            }
            thalamus::IntentCategory::Conversation => {
                self.dispatch_conversation(ctx, intent, prepend_nudges)
                    .await
            }
        }
    }

    // ── prepare() helpers ───────────────────────────────────────────────────

    /// Hydrate conversation history from session episodes when the caller
    /// didn't pass any. Returns `None` if the caller supplied history or
    /// no session lookup is possible. Bounded so the token budget stays
    /// predictable.
    fn hydrate_history(
        &self,
        signal: &Signal,
        caller_history: Option<&[cortex::llm::Message]>,
    ) -> Option<Vec<cortex::llm::Message>> {
        const SESSION_HISTORY_LIMIT: usize = 20;
        match caller_history {
            Some(_) => None,
            None => signal
                .session_id
                .as_deref()
                .map(|sid| self.load_session_messages(sid, SESSION_HISTORY_LIMIT))
                .filter(|h| !h.is_empty()),
        }
    }

    /// Drain up to 10 pending proactive notifications from the outbox so
    /// the response can prepend them as nudges.
    fn drain_pending_notifications(&self) -> Vec<storage::sqlite::Notification> {
        match &self.channels.notification_router {
            Some(router) => router.drain_pending(10),
            None => Vec::new(),
        }
    }

    /// Run intent classification, log the outcome, and persist any facts
    /// the classifier extracted on the side. When `history` is supplied,
    /// the LLM fallback uses it to disambiguate follow-up replies from
    /// new biographical claims.
    async fn classify_and_store_facts(
        &self,
        signal: &Signal,
        history: Option<&[cortex::llm::Message]>,
    ) -> thalamus::Classification {
        let history = history.unwrap_or(&[]);
        // Feed the live capability manifest into the classifier so it shares
        // the same view of available tools the SOUL and external clients see.
        // Lightweight — registry list only, no fitness query; only the
        // LLM-fallback branch consults it.
        let capability_summary = self.planner_capabilities().await.join("\n");
        let capabilities = (!capability_summary.is_empty()).then_some(capability_summary.as_str());
        let classification = self
            .classifier
            .classify_with_context(&signal.content, history, capabilities)
            .await;
        self.metrics.inc_intent_classification();
        if matches!(classification.method, thalamus::ClassificationMethod::Llm) {
            self.metrics.inc_intent_llm_fallback();
        }

        tracing::info!(
            signal_id = %signal.id,
            source = ?signal.source,
            intent = ?classification.intent,
            importance = self.importance.score(&signal.content),
            method = ?classification.method,
            extracted_facts = classification.extracted_facts.len(),
            "Signal classified"
        );

        if let Some(observer) = &self.observability.observer {
            let intent_summary = observe::intent_summary_of(&classification.intent);
            let ev = ::observe::BrainEvent::IntentClassified {
                id: signal.id,
                intent: intent_summary,
                confidence: classification.confidence as f32,
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }

        if !classification.extracted_facts.is_empty() {
            let facts_to_store: Vec<_> = classification
                .extracted_facts
                .iter()
                .map(|f| crate::exchange::FactToStore {
                    subject: f.subject.clone(),
                    predicate: f.predicate.clone(),
                    object: f.object.clone(),
                })
                .collect();

            let (stored, errors) = self
                .store_facts_batch(
                    &signal.namespace,
                    "extracted",
                    &facts_to_store,
                    signal.agent.as_deref(),
                )
                .await;

            for id in &stored {
                tracing::info!("Extracted fact stored: {id}");
            }
            for (text, e) in errors {
                tracing::warn!("Failed to store extracted fact ({text}): {e}");
            }
        }

        classification
    }

    /// Match user content against stored procedures and surface the
    /// concatenated step list as additional context. Procedure-matcher
    /// errors are non-fatal — we log and degrade to no procedures.
    fn match_procedures(&self, content: &str) -> Vec<String> {
        match self.procedures.match_trigger(content) {
            Ok(procs) if !procs.is_empty() => {
                tracing::debug!(
                    count = procs.len(),
                    "Procedure(s) matched — injecting steps into context"
                );
                let mut steps: Vec<String> = Vec::new();
                for proc in &procs {
                    if let Err(e) = self.procedures.record_execution(&proc.id) {
                        tracing::warn!(
                            procedure_id = %proc.id,
                            "Failed to persist procedure execution count: {e}"
                        );
                    }
                    steps.extend(proc.steps.clone());
                }
                steps
            }
            Ok(_) => Vec::new(),
            Err(e) => {
                tracing::warn!("Procedure match failed (non-fatal): {e}");
                Vec::new()
            }
        }
    }

    /// Publish a `BrainEvent::SignalReceived` to the observability bus if one
    /// is configured. Silent no-op when no observer or no subscribers are
    /// attached — observability must never block the pipeline.
    ///
    /// All string fields are passed through [`observe::Redactor`] first so a
    /// vault-marked secret embedded in `Signal.content` cannot leak onto the
    /// bus.
    pub async fn publish_signal_received(&self, signal: &Signal) {
        let Some(observer) = &self.observability.observer else {
            return;
        };
        const PREVIEW_BYTES: usize = 256;
        let preview = if signal.content.len() <= PREVIEW_BYTES {
            signal.content.clone()
        } else {
            let mut end = PREVIEW_BYTES;
            while end > 0 && !signal.content.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &signal.content[..end])
        };
        let redactor = ::observe::Redactor::new();
        let scrub = |s: String| -> String {
            let mut v = serde_json::Value::String(s);
            redactor.redact(&mut v);
            v.as_str().map(|s| s.to_string()).unwrap_or_default()
        };
        let ev = ::observe::BrainEvent::SignalReceived {
            id: signal.id,
            signal: ::observe::SignalSummary {
                source: format!("{:?}", signal.source).to_lowercase(),
                channel: scrub(signal.channel.clone()),
                sender: scrub(signal.sender.clone()),
                namespace: scrub(signal.namespace.clone()),
                content_preview: scrub(preview),
            },
            ts: chrono::Utc::now(),
        };
        // BusClosed (no subscribers) is informational, not fatal.
        let _ = observer.publish(ev).await;
    }

    pub(super) fn publish_event(&self, signal: &Signal, response: &SignalResponse) {
        let event = SignalProcessedEvent {
            signal_id: response.signal_id,
            source: signal.source.clone(),
            channel: signal.channel.clone(),
            sender: signal.sender.clone(),
            namespace: signal.namespace.clone(),
            status: response.status.clone(),
            response: response_to_text(&response.response),
            facts_used: response.memory_context.facts_used,
            episodes_used: response.memory_context.episodes_used,
            timestamp: chrono::Utc::now(),
        };
        let _ = self.observability.events_tx.send(event);
    }

    // ── Identity + confirmation gates ────────────────────────────────────

    /// Run the configured `IdentityStore::check` against the classified
    /// intent. Returns `Some(resp)` if the pipeline should short-circuit
    /// with that response, `None` to proceed.
    ///
    /// Skipped when no identity store is wired, when the signal carries no
    /// principal, or when the intent is unguarded (chat/inspection).
    pub(super) async fn enforce_identity(
        &self,
        signal: &Signal,
        signal_id: Uuid,
        intent: &thalamus::Intent,
    ) -> Option<SignalResponse> {
        let store = self.identity_store.as_ref()?;
        let principal = signal.principal.as_ref()?;
        let (req, required) = crate::authz::intent_to_auth(intent)?;

        match store.check(principal, &req, required).await {
            identity::CheckOutcome::Allow => None,
            identity::CheckOutcome::EscalateToUser { reason } => {
                if let Some(observer) = &self.observability.observer {
                    let ev = ::observe::BrainEvent::ConfirmationRequested {
                        id: signal_id,
                        nonce: signal_id.to_string(),
                        reason: reason.clone(),
                        ts: chrono::Utc::now(),
                    };
                    let _ = observer.publish(ev).await;
                }
                Some(SignalResponse::ok(
                    signal_id,
                    format!(
                        "Approval required: {reason}. Awaiting your decision \
                         (Live tab → cancel/approve, or `respond` from chat)."
                    ),
                ))
            }
            identity::CheckOutcome::Deny { reason } => {
                if let Some(observer) = &self.observability.observer {
                    let ev = ::observe::BrainEvent::Error {
                        id: signal_id,
                        source: "identity.deny".into(),
                        message: reason.clone(),
                        ts: chrono::Utc::now(),
                    };
                    let _ = observer.publish(ev).await;
                }
                Some(SignalResponse::error(signal_id, reason))
            }
        }
    }

    /// Inline confirmation gate. Runs after the identity check. When a
    /// confirmation engine is wired and the intent lands a tier that
    /// `requires_confirmation`, builds an [`ApprovalSpec`] with the
    /// principal-bound [`GrantKey`] (so [`StandingApprovalStore`] matches
    /// bypass any user prompt) and blocks on `engine.request`.
    ///
    /// Returns `Some(resp)` to short-circuit:
    /// - `ApprovalOutcome::Approved` → `None` (proceed to dispatch)
    /// - `Rejected` / `TimedOut` / `Aborted` → text response explaining
    ///   the outcome
    /// - `engine.request` error → error response
    ///
    /// Skipped when no engine is wired, the intent is unguarded, or the
    /// tier doesn't require confirmation. This is the single inline
    /// checkpoint that enforces the cardinal rule: every action that
    /// reaches a Destructive/External tier passes through the same gate
    /// regardless of provenance (user typing, LLM, reflex firing).
    pub(super) async fn confirmation_gate(
        &self,
        signal: &Signal,
        signal_id: Uuid,
        intent: &thalamus::Intent,
    ) -> Option<SignalResponse> {
        // identity::Tier and brain::security::ActionTier carry
        // identical variants; the converter keeps the cross-crate
        // boundary explicit rather than relying on shared serialization.
        fn convert_tier(t: identity::Tier) -> brain::security::ActionTier {
            use brain::security::ActionTier;
            match t {
                identity::Tier::Read => ActionTier::Read,
                identity::Tier::Write => ActionTier::Write,
                identity::Tier::Execute => ActionTier::Execute,
                identity::Tier::Destructive => ActionTier::Destructive,
                identity::Tier::External => ActionTier::External,
            }
        }

        let engine = self.safety.confirmation_engine.as_ref()?;
        let (req, identity_tier) = crate::authz::intent_to_auth(intent)?;
        let action_tier = convert_tier(identity_tier);
        if !action_tier.requires_confirmation() {
            return None;
        }

        let description = format!("{}.{}", req.verb_ns, req.verb_action);
        let timeout = self
            .safety
            .confirmation_timeout
            .unwrap_or_else(|| action_tier.default_timeout());
        let mut spec =
            confirm::ApprovalSpec::new(description.clone(), action_tier).with_timeout(timeout);
        if let Some(principal) = &signal.principal {
            spec = spec.with_grant_key(confirm::GrantKey::new(
                principal.agent_id.0.clone(),
                &req.verb_ns,
                &req.verb_action,
            ));
        }
        let nonce = spec.nonce.clone();

        if let Some(observer) = &self.observability.observer {
            let ev = ::observe::BrainEvent::ConfirmationRequested {
                id: signal_id,
                nonce: nonce.clone(),
                reason: description.clone(),
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }

        let result = engine.request(spec).await;

        if let Some(observer) = &self.observability.observer {
            let decision = match &result {
                Ok(confirm::ApprovalOutcome::Approved) => "approved",
                Ok(confirm::ApprovalOutcome::Rejected { .. }) => "rejected",
                Ok(confirm::ApprovalOutcome::TimedOut) => "timed_out",
                Ok(confirm::ApprovalOutcome::Aborted { .. }) => "aborted",
                Err(_) => "error",
            };
            let ev = ::observe::BrainEvent::ConfirmationResolved {
                id: signal_id,
                nonce,
                decision: decision.to_string(),
                ts: chrono::Utc::now(),
            };
            let _ = observer.publish(ev).await;
        }

        match result {
            Ok(confirm::ApprovalOutcome::Approved) => None,
            Ok(confirm::ApprovalOutcome::Rejected { reason }) => Some(SignalResponse::error(
                signal_id,
                format!("Approval rejected for `{description}`: {reason}"),
            )),
            Ok(confirm::ApprovalOutcome::TimedOut) => Some(SignalResponse::error(
                signal_id,
                format!("Approval timed out waiting for confirmation on `{description}`"),
            )),
            Ok(confirm::ApprovalOutcome::Aborted { reason }) => Some(SignalResponse::error(
                signal_id,
                format!("Approval aborted for `{description}`: {reason}"),
            )),
            Err(e) => Some(SignalResponse::error(
                signal_id,
                format!("Confirmation engine error on `{description}`: {e}"),
            )),
        }
    }
}
