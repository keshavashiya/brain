//! Acceptance suite — memory **consolidation** end-to-end.
//!
//! The per-outcome unit tests in `hippocampus::consolidation` cover prune,
//! keep, and promote in isolation. This suite exercises a single realistic
//! consolidation pass over a *mixed* episodic store and asserts all three
//! outcomes coexist in one `ConsolidationReport`:
//!
//! - a trivial, low-importance episode is **pruned**,
//! - an important, recent episode **survives**, and
//! - a frequently reinforced episode becomes a **promotion candidate**.
//!
//! Because `forgetting_curve(importance, ~0h, decay) ≈ importance`, a single
//! `prune_threshold = 0.5` cleanly separates the trivial episode (0.01) from
//! the important (1.0) and reinforced (0.8) ones, while `promotion_threshold
//! = 3` catches the thrice-reinforced episode — so one config drives every
//! branch.

use hippocampus::{ConsolidationConfig, Consolidator, EpisodicStore};
use storage::SqlitePool;

#[test]
fn single_pass_prunes_keeps_and_promotes() {
    let db = SqlitePool::open_memory().unwrap();
    let store = EpisodicStore::new(db);
    let session = store.create_session("consolidation-acceptance").unwrap();

    // (1) Trivial chatter — low importance, should be pruned.
    store
        .store_episode(&session, "user", "ok thanks", 0.01, None, None)
        .unwrap();

    // (2) Critical fact — high importance, must survive.
    store
        .store_episode(
            &session,
            "user",
            "critical: the prod database lives in us-east-1",
            1.0,
            Some("work"),
            None,
        )
        .unwrap();

    // (3) A recurring preference — reinforced past the promotion threshold.
    // `store_episode` returns the new episode's id directly, so we reinforce
    // exactly this one (don't rely on session-history ordering).
    let recurring_id = store
        .store_episode(
            &session,
            "user",
            "I prefer Rust for systems work",
            0.8,
            Some("work"),
            None,
        )
        .unwrap();
    store.reinforce(&recurring_id).unwrap();
    store.reinforce(&recurring_id).unwrap();
    store.reinforce(&recurring_id).unwrap();

    assert_eq!(store.count().unwrap(), 3, "all three episodes stored");

    // One consolidation pass with a config that triggers every branch.
    let consolidator = Consolidator::new(ConsolidationConfig {
        prune_threshold: 0.5,
        promotion_threshold: 3,
        ..Default::default()
    });
    let report = consolidator.consolidate(&store).unwrap();

    // (1) The trivial episode was pruned …
    assert_eq!(
        report.episodes_pruned, 1,
        "exactly the one low-importance episode should be pruned"
    );
    // … leaving the important + reinforced episodes behind.
    assert_eq!(
        report.episodes_remaining, 2,
        "the important and reinforced episodes must survive"
    );
    assert_eq!(store.count().unwrap(), 2, "store reflects the prune");

    // (3) The reinforced episode is flagged for promotion to semantic memory.
    assert!(
        report.episodes_promoted >= 1,
        "the thrice-reinforced episode should be a promotion candidate, report: {report:?}"
    );
    assert!(
        report
            .promotion_candidates
            .iter()
            .any(|c| c.episode_id == recurring_id),
        "the promotion candidate set must name the reinforced episode"
    );
}
