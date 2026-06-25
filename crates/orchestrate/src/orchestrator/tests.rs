use std::sync::Arc;

use super::TaskOrchestrator;
use crate::decompose::{DecompositionContext, DecompositionError, TaskDecomposer};
use crate::state::{StepState, TaskPhase};
use crate::step::{StepAction, TaskStep};

/// A mock decomposer that returns a fixed set of steps.
struct MockDecomposer {
    steps: Vec<TaskStep>,
}

#[async_trait::async_trait]
impl TaskDecomposer for MockDecomposer {
    async fn decompose(
        &self,
        _request: &str,
        _context: DecompositionContext,
    ) -> Result<Vec<TaskStep>, DecompositionError> {
        Ok(self.steps.clone())
    }
}

fn test_steps() -> Vec<TaskStep> {
    vec![
        TaskStep {
            id: "s1".to_string(),
            description: "Research".to_string(),
            action: StepAction::Research {
                query: "test".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        },
        TaskStep {
            id: "s2".to_string(),
            description: "Test".to_string(),
            action: StepAction::Execute {
                command: "echo hello".to_string(),
                workdir: "/tmp".into(),
            },
            depends_on: vec!["s1".to_string()],
            tier: audit::ActionTier::Execute,
            estimated_tokens: 0,
        },
    ]
}

// ── State-machine acceptance ────────────────────────────────────────

#[tokio::test]
async fn phase6_state_machine_emits_canonical_transitions_and_persists_rows() {
    use observe::{BrainEvent, BroadcastObserver, Observer};

    let pool = storage::SqlitePool::open_memory().unwrap();
    let observer_arc = BroadcastObserver::new();
    // Subscribe BEFORE the orchestrator runs so the broadcast
    // channel has a live receiver — without a subscriber,
    // `BroadcastObserver::publish` returns `Err(BusClosed)` and
    // the orchestrator's best-effort send swallows it.
    let mut rx = observer_arc.subscribe();
    let observer: Arc<dyn Observer> = observer_arc.clone();

    // One no-op Plan step is the cheapest path through execute()
    // that exercises is_complete() + all_succeeded() → Reconciling
    // → Completed without dragging in the sandbox / LLM / agent
    // registry. Plan-with-non-empty-output succeeds cleanly.
    let decomposer = Arc::new(MockDecomposer {
        steps: vec![TaskStep {
            id: "s1".to_string(),
            description: "no-op step".to_string(),
            action: StepAction::Plan {
                output: "did nothing observable".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }],
    });
    let orchestrator = TaskOrchestrator::new(decomposer)
        .with_observer(observer)
        .with_state_pool(pool.clone());

    let (task_id, _plan) = orchestrator
        .plan("phase6 smoke", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    // Drain observed transitions.
    let mut transitions: Vec<(String, String)> = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        if let BrainEvent::TaskStateChange { from, to, .. } = ev {
            transitions.push((from, to));
        }
    }

    let expected: Vec<(&str, &str)> = vec![
        ("none", "planning"),
        ("planning", "awaiting_approval"),
        ("awaiting_approval", "executing"),
        ("executing", "reconciling"),
        ("reconciling", "completed"),
    ];
    let observed: Vec<(&str, &str)> = transitions
        .iter()
        .map(|(f, t)| (f.as_str(), t.as_str()))
        .collect();
    assert_eq!(observed, expected, "transition sequence mismatch");

    // The `task_states` audit table should mirror the events,
    // newest-last. ORDER BY id ASC preserves insertion order even
    // if two transitions land in the same wall-clock second.
    let states_in_db = pool
        .with_conn(|conn| {
            let mut stmt =
                conn.prepare("SELECT state FROM task_states WHERE task_id = ?1 ORDER BY id ASC")?;
            let states: Vec<String> = stmt
                .query_map([&task_id], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(states)
        })
        .unwrap();
    assert_eq!(
        states_in_db,
        vec![
            "planning".to_string(),
            "awaiting_approval".to_string(),
            "executing".to_string(),
            "reconciling".to_string(),
            "completed".to_string(),
        ],
        "task_states audit rows must mirror the emitted events"
    );

    // Final in-memory phase agrees with the table.
    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(task.phase, TaskPhase::Completed);
    assert!(task.completed_at.is_some());
}

#[tokio::test]
async fn phase6_failed_step_lands_in_failed_terminal_state() {
    // Empty-output Plan step is the deterministic failure path —
    // the executor treats it as "honest failure" so we don't need
    // to mock a flaky sandbox.
    use observe::{BroadcastObserver, Observer};

    let pool = storage::SqlitePool::open_memory().unwrap();
    let observer_arc = BroadcastObserver::new();
    let _rx = observer_arc.subscribe();
    let observer: Arc<dyn Observer> = observer_arc.clone();

    let decomposer = Arc::new(MockDecomposer {
        steps: vec![TaskStep {
            id: "s1".to_string(),
            description: "failing step".to_string(),
            action: StepAction::Plan {
                output: String::new(), // empty → fail
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }],
    });
    let orchestrator = TaskOrchestrator::new(decomposer)
        .with_observer(observer)
        .with_state_pool(pool.clone());

    let (task_id, _) = orchestrator
        .plan("phase6 fail", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(
        task.phase,
        TaskPhase::Failed,
        "task with a failed step must land in Failed, not Completed"
    );

    // Sanity: the final row in task_states is `failed`.
    let last_state: String = pool
        .with_conn(|conn| {
            conn.query_row(
                "SELECT state FROM task_states WHERE task_id = ?1 ORDER BY id DESC LIMIT 1",
                [&task_id],
                |r| r.get(0),
            )
            .map_err(Into::into)
        })
        .unwrap();
    assert_eq!(last_state, "failed");
}

#[tokio::test]
async fn phase6_terminal_transitions_are_idempotent() {
    // After Cancelled, a stray Completed transition must not flip
    // the phase back. Protects against a slow-completing step that
    // returns after the user has already cancelled.
    let decomposer = Arc::new(MockDecomposer {
        steps: vec![TaskStep {
            id: "s1".to_string(),
            description: "any".to_string(),
            action: StepAction::Plan {
                output: "ok".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }],
    });
    let orchestrator = TaskOrchestrator::new(decomposer);
    let (task_id, _) = orchestrator
        .plan(
            "phase6 cancel-then-late-completion",
            DecompositionContext::default(),
        )
        .await
        .unwrap();
    orchestrator.cancel(&task_id).await.unwrap();
    orchestrator
        .transition_phase(&task_id, TaskPhase::Completed)
        .await;
    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(
        task.phase,
        TaskPhase::Cancelled,
        "late Completed transition must not overwrite Cancelled"
    );
}

// ── PR-6b: CancellationToken propagation ────────────────────────────

#[tokio::test]
async fn pr6b_cancel_aborts_in_flight_step_within_one_polling_cycle() {
    // A step that would otherwise sleep for an hour must abort
    // promptly once `cancel()` fires. Acceptance criterion:
    // execute() returns within a bounded wall-clock window
    // after cancel() — we use 2s to give CI breathing room.
    use async_trait::async_trait;
    use chrono::Utc;
    use delegate::{
        AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
        AgentTaskStatus,
    };
    use std::time::{Duration, Instant};

    struct SlowAgent;
    #[async_trait]
    impl AgentDelegate for SlowAgent {
        fn name(&self) -> &str {
            "slow"
        }
        fn capabilities(&self) -> AgentCapabilities {
            AgentCapabilities::default()
        }
        async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
            // Sleep for an hour — well beyond any reasonable test
            // timeout. Drop-on-cancel from tokio::select! must
            // interrupt this so execute() can return.
            tokio::time::sleep(Duration::from_secs(3600)).await;
            let now = Utc::now();
            Ok(AgentResult {
                task_id: task.id,
                status: AgentTaskStatus::Succeeded,
                summary: "unreachable".to_string(),
                artifacts: vec![],
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(0),
                started_at: now,
                completed_at: now,
            })
        }
    }

    let mut registry = AgentRegistry::new();
    registry.register(Arc::new(SlowAgent));
    let registry = Arc::new(registry);

    let decomposer = Arc::new(MockDecomposer {
        steps: vec![TaskStep {
            id: "slow".to_string(),
            description: "long-running step".to_string(),
            action: StepAction::Implement {
                spec: "do nothing forever".to_string(),
                agent: "slow".to_string(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        }],
    });
    let orchestrator = Arc::new(TaskOrchestrator::new(decomposer).with_agents(registry));

    let (task_id, _) = orchestrator
        .plan("pr6b mid-step cancel", DecompositionContext::default())
        .await
        .unwrap();

    let exec_orch = orchestrator.clone();
    let exec_task_id = task_id.clone();
    let exec_handle = tokio::spawn(async move { exec_orch.execute(&exec_task_id).await.unwrap() });

    // Let execute() actually enter the slow step before we cancel —
    // otherwise the cancel could race ahead of the spawn and just
    // hit the outer-loop pre-check, which wouldn't exercise the
    // mid-step abort path we're testing.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let cancel_at = Instant::now();
    orchestrator.cancel(&task_id).await.unwrap();

    // execute() must return shortly after cancel — well under the
    // 3600s the SlowAgent would otherwise sleep for.
    let _summary = tokio::time::timeout(Duration::from_secs(2), exec_handle)
        .await
        .expect("execute() must return within 2s of cancel; did the token thread through?")
        .expect("execute task panicked");
    let elapsed = cancel_at.elapsed();
    assert!(
            elapsed < Duration::from_secs(2),
            "execute() returned but took {elapsed:?} after cancel — cancellation should be near-instant"
        );

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(
        task.phase,
        TaskPhase::Cancelled,
        "task must land in Cancelled after mid-step cancel"
    );
    // The slow step itself should be Cancelled, not lingering in
    // Running. cancel() flips state, and the select-loser's
    // mark_step_cancelled overwrite is a no-op — either way the
    // observed state is Cancelled.
    assert!(
        matches!(task.step_states.get("slow"), Some(StepState::Cancelled)),
        "in-flight step must be Cancelled, got {:?}",
        task.step_states.get("slow")
    );
}

#[tokio::test]
async fn pr6b_cancel_before_execute_exits_without_running_steps() {
    // If cancel() fires after plan() but before execute() starts,
    // execute() should observe the cancellation and exit without
    // touching any step handlers. The task lands Cancelled.
    let decomposer = Arc::new(MockDecomposer {
        steps: test_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer);

    let (task_id, _) = orchestrator
        .plan("pr6b pre-execute cancel", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.cancel(&task_id).await.unwrap();

    let _summary = orchestrator.execute(&task_id).await.unwrap();
    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(task.phase, TaskPhase::Cancelled);
    // No step should have advanced past Cancelled — set by cancel()
    // before execute() ran.
    for (id, state) in &task.step_states {
        assert!(
            matches!(state, StepState::Cancelled),
            "step {id} should be Cancelled, got {state:?}"
        );
    }
}

#[tokio::test]
async fn pr6b_cancel_token_fires_when_cancel_called() {
    // Lower-level invariant: cancel() must actually fire the
    // per-task token so future select! consumers (e.g. an external
    // bridge that wires its own cancel-aware future) observe it.
    let decomposer = Arc::new(MockDecomposer {
        steps: test_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer);
    let (task_id, _) = orchestrator
        .plan("pr6b token fires", DecompositionContext::default())
        .await
        .unwrap();
    let token = orchestrator.cancel_token_for(&task_id).await;
    assert!(!token.is_cancelled(), "token must start uncancelled");
    orchestrator.cancel(&task_id).await.unwrap();
    assert!(
        token.is_cancelled(),
        "cancel() must fire the per-task token"
    );
}

#[tokio::test]
async fn test_plan_creates_task() {
    let decomposer = Arc::new(MockDecomposer {
        steps: test_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer);

    let (task_id, plan_text) = orchestrator
        .plan("build something", DecompositionContext::default())
        .await
        .unwrap();

    assert!(!task_id.is_empty());
    assert!(plan_text.contains("Research"));
    assert!(plan_text.contains("Test"));

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(task.phase, TaskPhase::AwaitingApproval);
}

#[tokio::test]
async fn test_execute_runs_steps() {
    let sandbox = Arc::new(sandbox::StubSandbox::new());
    let decomposer = Arc::new(MockDecomposer {
        steps: test_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer).with_sandbox(sandbox);

    let (task_id, _) = orchestrator
        .plan("build something", DecompositionContext::default())
        .await
        .unwrap();

    let summary = orchestrator.execute(&task_id).await.unwrap();
    assert!(summary.contains("Completed"));

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(task.phase, TaskPhase::Completed);
    assert!(task.all_succeeded());
}

#[tokio::test]
async fn test_implement_step_dispatches_through_registry() {
    use async_trait::async_trait;
    use chrono::Utc;
    use delegate::{
        AgentCapabilities, AgentDelegate, AgentError, AgentRegistry, AgentResult, AgentTask,
        AgentTaskStatus,
    };

    struct StubAgent;

    #[async_trait]
    impl AgentDelegate for StubAgent {
        fn name(&self) -> &str {
            "stub"
        }
        fn capabilities(&self) -> AgentCapabilities {
            AgentCapabilities::default()
        }
        async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
            let now = Utc::now();
            Ok(AgentResult {
                task_id: task.id,
                status: AgentTaskStatus::Succeeded,
                summary: format!("stubbed: {}", task.description),
                artifacts: vec![],
                stdout: "ok".to_string(),
                stderr: String::new(),
                exit_code: Some(0),
                started_at: now,
                completed_at: now,
            })
        }
    }

    let mut registry = AgentRegistry::new();
    registry.register(Arc::new(StubAgent));
    let registry = Arc::new(registry);

    let implement_step = TaskStep {
        id: "impl".to_string(),
        description: "Implement feature".to_string(),
        action: StepAction::Implement {
            spec: "write a README".to_string(),
            agent: "stub".to_string(),
        },
        depends_on: vec![],
        tier: audit::ActionTier::Write,
        estimated_tokens: 0,
    };
    let decomposer = Arc::new(MockDecomposer {
        steps: vec![implement_step],
    });
    let orchestrator = TaskOrchestrator::new(decomposer).with_agents(registry);

    let (task_id, _) = orchestrator
        .plan("build it", DecompositionContext::default())
        .await
        .unwrap();
    let summary = orchestrator.execute(&task_id).await.unwrap();
    assert!(summary.contains("Completed"));

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert!(task.all_succeeded());
    let step = task.step_states.get("impl").unwrap();
    match step {
        StepState::Completed { outcome, .. } => {
            assert!(outcome.summary.contains("stub"));
            assert!(outcome.summary.contains("write a README"));
        }
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn test_implement_step_without_registry_fails() {
    let implement_step = TaskStep {
        id: "impl".to_string(),
        description: "Implement feature".to_string(),
        action: StepAction::Implement {
            spec: "do the thing".to_string(),
            agent: "ghost".to_string(),
        },
        depends_on: vec![],
        tier: audit::ActionTier::Write,
        estimated_tokens: 0,
    };
    let decomposer = Arc::new(MockDecomposer {
        steps: vec![implement_step],
    });
    let orchestrator = TaskOrchestrator::new(decomposer);

    let (task_id, _) = orchestrator
        .plan("build it", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    let task = orchestrator.get_task(&task_id).await.unwrap();
    let step = task.step_states.get("impl").unwrap();
    assert!(
        matches!(step, StepState::Failed { .. }),
        "expected Failed without registry, got {step:?}"
    );
}

#[tokio::test]
async fn failed_step_skips_dependents_instead_of_running_them() {
    // Regression: previously `is_terminal()` was used to decide which
    // deps were satisfied, so a Failed step unblocked its dependents
    // and they ran against missing inputs. Now they should be Skipped.
    let steps = vec![
        TaskStep {
            id: "s1".to_string(),
            description: "fail".to_string(),
            action: StepAction::Implement {
                spec: "won't matter".to_string(),
                agent: "missing".to_string(), // no registry → fails
            },
            depends_on: vec![],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        },
        TaskStep {
            id: "s2".to_string(),
            description: "depends on s1".to_string(),
            action: StepAction::Plan {
                output: "should not run".to_string(),
            },
            depends_on: vec!["s1".to_string()],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        },
        TaskStep {
            id: "s3".to_string(),
            description: "depends on s2".to_string(),
            action: StepAction::Plan {
                output: "should not run".to_string(),
            },
            depends_on: vec!["s2".to_string()],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        },
    ];
    let decomposer = Arc::new(MockDecomposer { steps });
    let orchestrator = TaskOrchestrator::new(decomposer);

    let (task_id, _) = orchestrator
        .plan("anything", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert!(matches!(
        task.step_states.get("s1"),
        Some(StepState::Failed { .. })
    ));
    assert!(
        matches!(task.step_states.get("s2"), Some(StepState::Skipped { .. })),
        "s2 should be Skipped after s1 failed, got {:?}",
        task.step_states.get("s2")
    );
    assert!(
        matches!(task.step_states.get("s3"), Some(StepState::Skipped { .. })),
        "s3 should be transitively Skipped, got {:?}",
        task.step_states.get("s3")
    );
    // State-machine: a task with any non-succeeded step
    // lands in `Failed`, not `Completed`.
    assert_eq!(task.phase, TaskPhase::Failed);
}

#[tokio::test]
async fn nonzero_exit_marks_step_failed_and_skips_dependents() {
    // Regression for the daemon RCA: a sandbox command that returns
    // exit_code != 0 used to be recorded as `Completed` because the
    // executor returned `Ok(StepOutcome { exit_code: Some(1), .. })`.
    // It must now be marked Failed so dependents cascade-skip.
    let sandbox = Arc::new(sandbox::StubSandbox::new());
    let steps = vec![
        TaskStep {
            id: "fail".to_string(),
            description: "always-fail command".to_string(),
            action: StepAction::Execute {
                command: "false".to_string(),
                workdir: "/tmp".into(),
            },
            depends_on: vec![],
            tier: audit::ActionTier::Execute,
            estimated_tokens: 0,
        },
        TaskStep {
            id: "after".to_string(),
            description: "should be skipped".to_string(),
            action: StepAction::Plan {
                output: "must not run".to_string(),
            },
            depends_on: vec!["fail".to_string()],
            tier: audit::ActionTier::Read,
            estimated_tokens: 0,
        },
    ];
    let decomposer = Arc::new(MockDecomposer { steps });
    let orchestrator = TaskOrchestrator::new(decomposer).with_sandbox(sandbox);

    let (task_id, _) = orchestrator
        .plan("anything", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    let task = orchestrator.get_task(&task_id).await.unwrap();
    let fail = task.step_states.get("fail").unwrap();
    assert!(
        matches!(fail, StepState::Failed { .. }),
        "non-zero exit must mark step Failed, got {fail:?}"
    );
    let after = task.step_states.get("after").unwrap();
    assert!(
        matches!(after, StepState::Skipped { .. }),
        "dependent must be Skipped, got {after:?}"
    );
}

#[tokio::test]
async fn replan_on_failure_splices_corrective_steps() {
    // After a step fails, the orchestrator should call the
    // decomposer's replan_after_failure hook and splice the
    // returned steps into the graph so they execute next.
    use crate::decompose::RepairContext;

    struct ReplanDecomposer {
        initial: Vec<TaskStep>,
        replan_called: std::sync::atomic::AtomicUsize,
        replan_steps: Vec<TaskStep>,
    }

    #[async_trait::async_trait]
    impl TaskDecomposer for ReplanDecomposer {
        async fn decompose(
            &self,
            _request: &str,
            _context: DecompositionContext,
        ) -> Result<Vec<TaskStep>, crate::decompose::DecompositionError> {
            Ok(self.initial.clone())
        }
        async fn replan_after_failure(
            &self,
            _repair: RepairContext,
            _context: DecompositionContext,
        ) -> Result<Vec<TaskStep>, crate::decompose::DecompositionError> {
            self.replan_called
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.replan_steps.clone())
        }
    }

    // Initial plan: one step that always fails.
    let initial = vec![TaskStep {
        id: "fail".to_string(),
        description: "missing-agent step".to_string(),
        action: StepAction::Implement {
            spec: "doomed".to_string(),
            agent: "ghost".to_string(),
        },
        depends_on: vec![],
        tier: audit::ActionTier::Read,
        estimated_tokens: 0,
    }];
    // Replan plan: a single Plan step that always succeeds.
    let replan_steps = vec![TaskStep {
        id: "replan-1".to_string(),
        description: "corrective step".to_string(),
        action: StepAction::Plan {
            output: "fixed it".to_string(),
        },
        depends_on: vec![],
        tier: audit::ActionTier::Read,
        estimated_tokens: 0,
    }];

    let decomposer = Arc::new(ReplanDecomposer {
        initial,
        replan_called: std::sync::atomic::AtomicUsize::new(0),
        replan_steps: replan_steps.clone(),
    });
    let decomposer_handle = decomposer.clone();
    let orchestrator = TaskOrchestrator::new(decomposer);

    let (task_id, _) = orchestrator
        .plan("anything", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    assert_eq!(
        decomposer_handle
            .replan_called
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "decomposer.replan_after_failure must be invoked exactly once"
    );

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(
        task.replan_attempts, 1,
        "task.replan_attempts must increment after a successful splice"
    );
    // The original step stays Failed; the replanned step succeeds.
    assert!(matches!(
        task.step_states.get("fail"),
        Some(StepState::Failed { .. })
    ));
    assert!(matches!(
        task.step_states.get("replan-1"),
        Some(StepState::Completed { .. })
    ));
    // Mixed-outcome task lands in `Failed` — the
    // original failure is recorded even though the replan
    // succeeded. (The user can still see the replanned-step
    // success in the per-step states.)
    assert_eq!(task.phase, TaskPhase::Failed);
}

#[tokio::test]
async fn test_cancel_task() {
    let decomposer = Arc::new(MockDecomposer {
        steps: test_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer);

    let (task_id, _) = orchestrator
        .plan("build something", DecompositionContext::default())
        .await
        .unwrap();

    orchestrator.cancel(&task_id).await.unwrap();

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert_eq!(task.phase, TaskPhase::Cancelled);
}

#[tokio::test]
async fn notify_with_no_channels_is_soft_success() {
    // When the dispatcher has no transports registered, the router
    // returns NoChannelAvailable. The orchestrator must NOT fail the
    // step — replan-on-failure produces Notify steps as its honest
    // "I cannot do this" path, and a hard failure here recurses into
    // more Notify steps until the replan budget is exhausted (see
    // brain.log:1036–1043 for the user-visible cascade).
    let db = storage::SqlitePool::open_memory().unwrap();
    let prefs = Arc::new(channel::SqlitePreferenceStore::new(db));
    prefs.ensure_tables().unwrap();
    let router: Arc<dyn channel::ChannelRouter> =
        Arc::new(channel::DefaultChannelRouter::new(prefs));
    let dispatcher = Arc::new(channel::ChannelDispatcher::new(router));

    let decomposer = Arc::new(MockDecomposer {
        steps: test_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer).with_channel_dispatcher(dispatcher);

    let outcome = orchestrator
        .execute_notify_step("default", "PDF cannot be parsed: pdftotext missing")
        .await
        .expect("notify must not fail when no channels are configured");
    assert!(outcome.summary.contains("no external channel"));
    assert!(outcome.summary.contains("pdftotext missing"));
}

// ── Parallel wave execution ─────────────────────────────────────────

/// A `Research`-step LLM stub that measures how many `generate` calls are
/// in flight at once. Each call bumps an in-flight counter, records the
/// running peak, sleeps briefly so concurrent calls actually overlap in
/// time, then decrements. The recorded peak is the observable proof of
/// (non-)parallel execution.
struct ConcurrencyProbeLlm {
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
    peak: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl cortex::llm::LlmProvider for ConcurrencyProbeLlm {
    async fn generate(
        &self,
        _messages: &[cortex::llm::Message],
    ) -> Result<cortex::llm::Response, cortex::llm::LlmError> {
        use std::sync::atomic::Ordering;
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(cortex::llm::Response::text("done", None))
    }
    async fn generate_stream(
        &self,
        _messages: &[cortex::llm::Message],
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures::Stream<
                        Item = Result<cortex::llm::ResponseChunk, cortex::llm::LlmError>,
                    > + Send,
            >,
        >,
        cortex::llm::LlmError,
    > {
        unreachable!("probe provider: research steps never stream")
    }
    async fn health_check(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "probe"
    }
    fn model(&self) -> &str {
        "probe-model"
    }
    async fn list_models(&self) -> Result<Vec<String>, cortex::llm::LlmError> {
        Ok(vec!["probe-model".into()])
    }
}

/// A diamond DAG: `a → {b, c} → d`. `b` and `c` are independent and become
/// ready in the same wave, so they are the steps that may overlap.
fn diamond_research_steps() -> Vec<TaskStep> {
    let research = |id: &str, deps: &[&str]| TaskStep {
        id: id.to_string(),
        description: format!("research {id}"),
        action: StepAction::Research {
            query: id.to_string(),
        },
        depends_on: deps.iter().map(|d| d.to_string()).collect(),
        tier: audit::ActionTier::Read,
        estimated_tokens: 0,
    };
    vec![
        research("a", &[]),
        research("b", &["a"]),
        research("c", &["a"]),
        research("d", &["b", "c"]),
    ]
}

async fn run_diamond_with_parallelism(max_parallel: usize) -> (usize, TaskPhase) {
    use std::sync::atomic::Ordering;
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let llm = Arc::new(ConcurrencyProbeLlm {
        in_flight: in_flight.clone(),
        peak: peak.clone(),
    });
    let decomposer = Arc::new(MockDecomposer {
        steps: diamond_research_steps(),
    });
    let orchestrator = TaskOrchestrator::new(decomposer)
        .with_llm(llm)
        .with_max_parallel_steps(max_parallel);
    let (task_id, _plan) = orchestrator
        .plan("diamond", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();
    let phase = orchestrator.get_task(&task_id).await.unwrap().phase;
    (peak.load(Ordering::SeqCst), phase)
}

#[tokio::test]
async fn parallel_wave_runs_independent_steps_concurrently() {
    // With headroom, the diamond's two independent middle steps run in the
    // same wave → at least two generate calls overlap in time.
    let (peak, phase) = run_diamond_with_parallelism(4).await;
    assert!(
        peak >= 2,
        "independent ready steps should overlap; observed peak {peak}"
    );
    assert_eq!(phase, TaskPhase::Completed, "all four steps should succeed");
}

#[tokio::test]
async fn max_parallel_one_keeps_execution_sequential() {
    // The same diamond with a width-1 limit must never overlap two steps,
    // proving the knob fully serialises execution (the pre-refactor default).
    let (peak, phase) = run_diamond_with_parallelism(1).await;
    assert_eq!(peak, 1, "max_parallel_steps=1 must never overlap steps");
    assert_eq!(phase, TaskPhase::Completed);
}

#[tokio::test]
async fn failure_in_a_parallel_wave_skips_dependents_but_lets_siblings_finish() {
    // Diamond a → {b, c} → d, but b is a no-output Plan step (deterministic
    // failure, no LLM needed). In the {b, c} wave, b fails while c still
    // completes; d depends on the failed b and is skipped. The default
    // decomposer declines to replan, so the failure stands.
    let plan = |id: &str, output: &str, deps: &[&str]| TaskStep {
        id: id.to_string(),
        description: format!("plan {id}"),
        action: StepAction::Plan {
            output: output.to_string(),
        },
        depends_on: deps.iter().map(|d| d.to_string()).collect(),
        tier: audit::ActionTier::Read,
        estimated_tokens: 0,
    };
    let steps = vec![
        plan("a", "root ok", &[]),
        plan("b", "", &["a"]),           // empty output → deterministic failure
        plan("c", "sibling ok", &["a"]), // independent of b, must still complete
        plan("d", "leaf", &["b", "c"]),  // depends on failed b → skipped
    ];
    let decomposer = Arc::new(MockDecomposer { steps });
    let orchestrator = TaskOrchestrator::new(decomposer).with_max_parallel_steps(4);
    let (task_id, _plan) = orchestrator
        .plan("failure wave", DecompositionContext::default())
        .await
        .unwrap();
    orchestrator.execute(&task_id).await.unwrap();

    let task = orchestrator.get_task(&task_id).await.unwrap();
    assert!(
        matches!(task.step_states.get("a"), Some(StepState::Completed { .. })),
        "root should complete"
    );
    assert!(
        matches!(task.step_states.get("b"), Some(StepState::Failed { .. })),
        "the empty Plan step should fail"
    );
    assert!(
        matches!(task.step_states.get("c"), Some(StepState::Completed { .. })),
        "the independent sibling should still complete despite b failing"
    );
    assert!(
        matches!(task.step_states.get("d"), Some(StepState::Skipped { .. })),
        "the dependent of the failed step should be skipped"
    );
    assert_eq!(
        task.phase,
        TaskPhase::Failed,
        "task with a failed step ends Failed"
    );
}
