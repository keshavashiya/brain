//! Safety infrastructure wiring: audit trail, channel preferences/router,
//! standing approvals, confirmation engine, cost budget, agent registry, and
//! the task orchestrator — all sharing the processor's SQLite pool.

use std::sync::Arc;

use confirm::StandingApprovalStore;

use super::agents::build_agent_registry;

/// Wire safety infrastructure into the processor.
///
/// Components: audit trail, confirmation engine, cost budget, sandbox executor.
/// All share the same SQLite pool as the episodic store for simplicity.
/// The credential vault is NOT wired here — it requires passphrase input
/// and is wired on demand (e.g. `brain vault` / `brain auth` commands).
pub(super) async fn wire_safety_infrastructure(
    processor: signal::SignalProcessor,
    config: &brain::BrainConfig,
    sandbox_executor: Arc<dyn sandbox::SandboxExecutor>,
) -> anyhow::Result<signal::SignalProcessor> {
    let db = processor.episodic().pool().clone();

    // Audit trail — always wired (foundation for everything below).
    // The observer is attached so every committed audit row publishes a
    // `BrainEvent::AuditAppended` onto the shared bus — that's what HTTP
    // `/v1/events`, the WS adapter, and gRPC SSE subscribers consume.
    // Without this, only `SignalReceived` reaches subscribers.
    let mut audit_trail = audit::SqliteAuditTrail::new(db.clone());
    if let Some(obs) = processor.observer() {
        audit_trail = audit_trail.with_observer(obs.clone());
    }
    // At-rest encryption of the sensitive audit columns, keyed off the install
    // identity key. Best-effort: if the key can't be loaded we log and fall
    // back to plaintext rather than refusing to wire the audit trail (which
    // every safety component below depends on).
    if config.security.audit_encryption {
        let key_path = config.data_dir().join("identity.key");
        match identity::IdentityKey::load_or_create(&key_path) {
            Ok(key) => {
                audit_trail =
                    audit_trail.with_encryptor(audit::SqliteAuditTrail::encryptor_for(&key));
                tracing::info!("Audit trail at-rest encryption enabled (identity-keyed)");
            }
            Err(e) => tracing::error!(
                error = %e,
                path = %key_path.display(),
                "failed to load identity key; audit trail will store plaintext"
            ),
        }
    }
    audit_trail
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Audit trail table init failed: {e}"))?;
    let audit_trail: Arc<dyn audit::AuditTrail> = Arc::new(audit_trail);
    tracing::info!("Audit trail wired");

    // Channel preference store + router + dispatcher — built before the
    // confirmation engine so the engine can attach the notifier hook that
    // pushes approval prompts out to the user. Transports register with
    // the dispatcher later (in `serve.rs::wire_preset_transports`).
    let pref_store = channel::SqlitePreferenceStore::new(db.clone());
    pref_store
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Channel preference table init failed: {e}"))?;
    let preferences: Arc<dyn channel::ChannelPreferenceStore> = Arc::new(pref_store);
    let router: Arc<dyn channel::ChannelRouter> =
        Arc::new(channel::DefaultChannelRouter::new(preferences.clone()));
    let dispatcher = Arc::new(channel::ChannelDispatcher::new(router.clone()));

    // Standing-approval store — same DB as the confirm engine. Migration
    // v21 creates the table; we populate any YAML-declared grants here
    // (idempotent: skip rows already active under the same triple, so
    // restarts don't pile up duplicate grants).
    let standing_concrete = confirm::SqliteStandingApprovals::new(db.clone());
    for decl in &config.confirm.standing_approvals {
        let key = confirm::GrantKey::new(&decl.agent_id, &decl.verb_ns, &decl.verb_action);
        match standing_concrete.is_granted(&key).await {
            Ok(true) => {
                tracing::debug!(
                    agent = %decl.agent_id,
                    verb_ns = %decl.verb_ns,
                    verb_action = %decl.verb_action,
                    "standing approval already active; skipping config grant"
                );
            }
            Ok(false) => match standing_concrete.grant(&key, decl.note.as_deref()).await {
                Ok(id) => tracing::info!(
                    id = %id,
                    agent = %decl.agent_id,
                    verb_ns = %decl.verb_ns,
                    verb_action = %decl.verb_action,
                    "standing approval granted from config"
                ),
                Err(e) => tracing::warn!(
                    agent = %decl.agent_id,
                    verb_ns = %decl.verb_ns,
                    verb_action = %decl.verb_action,
                    error = %e,
                    "config-declared standing approval failed to insert"
                ),
            },
            Err(e) => tracing::warn!(
                agent = %decl.agent_id,
                error = %e,
                "standing-approval lookup failed during config load"
            ),
        }
    }
    let standing_store: Arc<dyn confirm::StandingApprovalStore> = Arc::new(standing_concrete);

    // Confirmation engine — always wired, with notifier hook so approval
    // prompts actually reach the user instead of deadlocking on timeout.
    // The standing-approval store is wired here so `request()` can
    // bypass the prompt for pre-granted (agent, verb) tuples.
    let approval_notifier: Arc<dyn confirm::ApprovalNotifier> =
        Arc::new(signal::ChannelApprovalNotifier::new(dispatcher.clone()));
    let confirm_engine = confirm::SqliteConfirmationEngine::new(db.clone())
        .with_notifier(approval_notifier)
        .with_standing_approvals(standing_store.clone());
    confirm_engine
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Confirmation engine table init failed: {e}"))?;
    let confirm_engine: Arc<dyn confirm::ConfirmationEngine> = Arc::new(confirm_engine);
    tracing::info!("Confirmation engine wired (notifier + standing approvals)");

    // Cost budget — always wired, with audit coupling
    let budget_policy = budget::BudgetPolicy::default();
    let sqlite_budget = budget::SqliteBudget::new(db.clone(), budget_policy);
    sqlite_budget
        .ensure_tables()
        .map_err(|e| anyhow::anyhow!("Cost budget table init failed: {e}"))?;
    let sqlite_budget = sqlite_budget.with_audit(audit_trail.clone());
    let cost_budget: Arc<dyn budget::CostBudget> = Arc::new(sqlite_budget);
    tracing::info!("Cost budget wired (with audit coupling)");

    let processor = processor
        .with_audit_trail(audit_trail.clone())
        .with_confirmation_engine(confirm_engine.clone())
        .with_standing_approvals(standing_store.clone())
        .with_cost_budget(cost_budget)
        .with_sandbox_executor(sandbox_executor.clone());

    // ── Agent registry ──────────────────────────────────────────────────
    // Built before the orchestrator so `Implement` steps can dispatch to
    // registered specialist agents. Discovery scans `$PATH` for known
    // CLI agents at boot; manual `agents.delegates[]` entries still
    // work alongside (last-write-wins on name collisions).
    let agent_registry = build_agent_registry(config).await?;
    let agent_registry_arc = Arc::new(agent_registry);
    if !agent_registry_arc.is_empty() {
        tracing::info!(
            agents = ?agent_registry_arc.list(),
            "Agent delegation registry wired"
        );
    } else {
        tracing::info!("Agent delegation registry empty — Implement steps will require config");
    }

    // Task orchestrator — wired with the LLM provider for decomposition
    let decomposer: Arc<dyn orchestrate::TaskDecomposer> =
        Arc::new(orchestrate::LlmDecomposer::new(processor.llm_arc()));
    let escalation_policy = delegate::EscalationPolicy {
        fallbacks: config.agents.fallbacks.clone(),
        retry_on_timeout: config.agents.retry_on_timeout,
    };
    let orchestrator = orchestrate::TaskOrchestrator::new(decomposer)
        .with_audit(audit_trail)
        .with_confirmation(confirm_engine.clone())
        .with_sandbox(sandbox_executor)
        .with_agents(agent_registry_arc.clone())
        .with_channel_dispatcher(dispatcher.clone())
        .with_llm(processor.llm_arc())
        .with_episodic(Arc::new(hippocampus::EpisodicStore::new(db.clone())))
        .with_delegation_policy(escalation_policy)
        // Run independent ready steps concurrently per wave (the graph is a
        // DAG); 1 keeps execution sequential.
        .with_max_parallel_steps(config.actions.max_parallel_steps)
        // Cache the sandbox allowlist so the replan-on-failure loop can
        // include it in its corrective LLM call.
        .with_available_tools(config.security.exec_allowlist.clone());
    let processor = processor
        .with_orchestrator(Arc::new(orchestrator))
        .with_agent_registry(agent_registry_arc);
    tracing::info!("Task orchestrator wired");

    // ── Channel intelligence — bind the pieces we built above ──────────
    let correlator = Arc::new(channel::ConfirmationCorrelator::new(confirm_engine));
    let processor = processor
        .with_channel_preferences(preferences)
        .with_channel_router(router)
        .with_confirmation_correlator(correlator)
        .with_channel_dispatcher(dispatcher);
    tracing::info!("Channel intelligence wired (router + dispatcher + preferences + correlator)");

    Ok(processor)
}
