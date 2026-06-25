//! Learned answer-quality fitness — the conversational complement to
//! [`CapabilityFitnessStore`](crate::CapabilityFitnessStore).
//!
//! The tool store learns whether *tools* succeed. This store learns whether
//! *answers* helped: after each chat turn the kernel judges the user's
//! follow-up (a satisfied "thanks" vs an immediate rephrase or correction) and
//! reinforces a per-`(task-kind, model)` **success/failure mass**, decayed
//! under the same forgetting curve so stale evidence fades.
//!
//! The signal layer then biases tier selection with it: when the deep tier's
//! model measurably underperforms a cheaper tier *that has its own evidence*
//! for a task kind, that kind's turns route to the cheaper tier. So a
//! deliberately-degraded model loses routing share for the kinds it does badly
//! at — without ever escaping the configured tiers.
//!
//! Keyed by `(kind, model)` where `model` is `"provider/model"` exactly as the
//! L2 telemetry (`BrainEvent::TurnCompleted`) reports it, so a record joins
//! directly against the served chain.
//!
//! Backed by the shared SQLite pool (table `answer_fitness`, migration v26).
//! Decay is **lazy** — computed on every read/write from the elapsed time since
//! `last_used_at` — so there is no background sweeper.

use chrono::Utc;
use rusqlite::OptionalExtension;
use storage::SqlitePool;

use crate::fitness::{decay, hours_between};
use crate::CerebellumError;

/// In-band judgement of how a prior answer landed, scored from the user's
/// follow-up turn. Mapped to reinforcement mass by [`AnswerFitnessStore::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerOutcome {
    /// An explicit correction of the prior answer — the strongest negative.
    /// Recorded as failure, same weight as `Fail` (the distinction is for
    /// logging/telemetry, not the curve).
    Correction,
    /// An immediate rephrase of the same ask, or other dissatisfaction — the
    /// prior answer did not land. Failure mass.
    Fail,
    /// A passive acceptance / new sub-topic that builds on the answer. A mild
    /// positive. Success mass `+1`.
    Success,
    /// Explicit gratitude or praise ("thanks", "perfect", "that worked"). The
    /// strongest positive. Success mass `+2`.
    Gold,
    /// No usable signal (unrelated new topic, ambiguous). Records nothing.
    None,
}

impl AnswerOutcome {
    /// Reinforcement masses `(success_delta, failure_delta)` for this outcome.
    fn masses(self) -> (f64, f64) {
        match self {
            AnswerOutcome::Gold => (2.0, 0.0),
            AnswerOutcome::Success => (1.0, 0.0),
            AnswerOutcome::Fail | AnswerOutcome::Correction => (0.0, 1.0),
            AnswerOutcome::None => (0.0, 0.0),
        }
    }
}

/// Decayed answer-quality snapshot for one `(kind, model)` pair. `success` /
/// `failure` are decayed reinforcement masses (not raw tallies); `uses` is the
/// undecayed lifetime count of judged turns; `ratio` is
/// `success / (success + failure)`.
#[derive(Debug, Clone)]
pub struct AnswerQuality {
    pub kind: String,
    pub model: String,
    pub success: f32,
    pub failure: f32,
    pub uses: i64,
    pub ratio: f32,
}

/// Per-`(kind, model)` learned answer-quality record. See module docs.
/// Cheap to clone (the pool is an `Arc` internally), so a follow-up judge can
/// be handed an owned handle to record from a spawned task.
#[derive(Clone)]
pub struct AnswerFitnessStore {
    db: SqlitePool,
    enabled: bool,
    half_life_hours: f64,
}

impl AnswerFitnessStore {
    /// Create an answer-fitness store. A non-positive `half_life_hours` falls
    /// back to [`crate::fitness::DEFAULT_HALF_LIFE_HOURS`]. When `enabled` is
    /// false the store is inert: `record` is a no-op and `quality` returns
    /// `None` (the table is still created, so toggling the flag on later just
    /// starts accumulating).
    pub fn new(db: SqlitePool, enabled: bool, half_life_hours: f64) -> Self {
        Self {
            db,
            enabled,
            half_life_hours: if half_life_hours > 0.0 {
                half_life_hours
            } else {
                crate::fitness::DEFAULT_HALF_LIFE_HOURS
            },
        }
    }

    /// Whether answer-quality learning is on for this deployment.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Ensure the `answer_fitness` table exists (idempotent). Mirrors
    /// [`crate::CapabilityFitnessStore::ensure_tables`] so a caller can build
    /// the store on a not-yet-migrated pool without knowing migration state.
    pub fn ensure_tables(&self) -> Result<(), CerebellumError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS answer_fitness (
                    kind         TEXT    NOT NULL,
                    model        TEXT    NOT NULL,
                    success_mass REAL    NOT NULL DEFAULT 0,
                    failure_mass REAL    NOT NULL DEFAULT 0,
                    uses         INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (kind, model)
                );",
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Reinforce `(kind, model)` with one judged outcome. Decays the existing
    /// masses to now, adds the outcome's masses, and bumps `uses`. No-op when
    /// the store is disabled or the outcome carries no signal
    /// ([`AnswerOutcome::None`]).
    pub fn record(
        &self,
        kind: &str,
        model: &str,
        outcome: AnswerOutcome,
    ) -> Result<(), CerebellumError> {
        if !self.enabled || outcome == AnswerOutcome::None {
            return Ok(());
        }
        let (ds, df) = outcome.masses();
        let now = Utc::now();
        let half_life = self.half_life_hours;
        self.db.with_conn(|conn| {
            let existing: Option<(f64, f64, i64, String)> = conn
                .query_row(
                    "SELECT success_mass, failure_mass, uses, last_used_at
                     FROM answer_fitness WHERE kind = ?1 AND model = ?2",
                    rusqlite::params![kind, model],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .optional()?;
            let (mut s, mut f, uses) = match existing {
                Some((s, f, uses, last)) => {
                    let hours = hours_between(&last, now);
                    (decay(s, hours, half_life), decay(f, hours, half_life), uses)
                }
                None => (0.0, 0.0, 0),
            };
            s += ds;
            f += df;
            conn.execute(
                "INSERT INTO answer_fitness
                     (kind, model, success_mass, failure_mass, uses, last_used_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(kind, model) DO UPDATE SET
                     success_mass = excluded.success_mass,
                     failure_mass = excluded.failure_mass,
                     uses         = excluded.uses,
                     last_used_at = excluded.last_used_at",
                rusqlite::params![kind, model, s, f, uses + 1, now.to_rfc3339()],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Decayed answer quality for one `(kind, model)`, or `None` if never
    /// recorded (or the store is disabled).
    pub fn quality(
        &self,
        kind: &str,
        model: &str,
    ) -> Result<Option<AnswerQuality>, CerebellumError> {
        if !self.enabled {
            return Ok(None);
        }
        let now = Utc::now();
        let half_life = self.half_life_hours;
        let row = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT success_mass, failure_mass, uses, last_used_at
                 FROM answer_fitness WHERE kind = ?1 AND model = ?2",
                rusqlite::params![kind, model],
                |r| {
                    Ok((
                        r.get::<_, f64>(0)?,
                        r.get::<_, f64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                },
            )
            .optional()
            .map_err(Into::into)
        })?;
        Ok(row.map(|(s, f, uses, last)| {
            let hours = hours_between(&last, now);
            decayed_quality(kind, model, s, f, uses, hours, half_life)
        }))
    }
}

/// Build a decayed [`AnswerQuality`] from stored masses + elapsed hours.
#[allow(clippy::too_many_arguments)]
fn decayed_quality(
    kind: &str,
    model: &str,
    s: f64,
    f: f64,
    uses: i64,
    hours: f64,
    half_life: f64,
) -> AnswerQuality {
    let s = decay(s, hours, half_life);
    let f = decay(f, hours, half_life);
    let total = s + f;
    let ratio = if total > 0.0 { (s / total) as f32 } else { 0.0 };
    AnswerQuality {
        kind: kind.to_string(),
        model: model.to_string(),
        success: s as f32,
        failure: f as f32,
        uses,
        ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::DEFAULT_HALF_LIFE_HOURS;

    fn test_store() -> AnswerFitnessStore {
        let pool = SqlitePool::open_memory().unwrap();
        let store = AnswerFitnessStore::new(pool, true, DEFAULT_HALF_LIFE_HOURS);
        store.ensure_tables().unwrap();
        store
    }

    #[test]
    fn gold_outweighs_success_and_fail_is_failure() {
        let store = test_store();
        store
            .record("coding", "ollama/llama3", AnswerOutcome::Gold)
            .unwrap();
        store
            .record("coding", "ollama/llama3", AnswerOutcome::Success)
            .unwrap();
        store
            .record("coding", "ollama/llama3", AnswerOutcome::Fail)
            .unwrap();
        let q = store.quality("coding", "ollama/llama3").unwrap().unwrap();
        assert_eq!(q.uses, 3);
        // gold(2) + success(1) = 3 success, 1 failure → ratio 0.75.
        assert!((q.success - 3.0).abs() < 0.01, "{}", q.success);
        assert!((q.failure - 1.0).abs() < 0.01, "{}", q.failure);
        assert!((q.ratio - 0.75).abs() < 0.01, "{}", q.ratio);
    }

    #[test]
    fn correction_records_as_failure() {
        let store = test_store();
        store
            .record("reasoning", "m", AnswerOutcome::Correction)
            .unwrap();
        let q = store.quality("reasoning", "m").unwrap().unwrap();
        assert_eq!(q.uses, 1);
        assert!((q.failure - 1.0).abs() < 0.01);
        assert_eq!(q.ratio, 0.0);
    }

    #[test]
    fn none_outcome_records_nothing() {
        let store = test_store();
        store.record("chitchat", "m", AnswerOutcome::None).unwrap();
        assert!(store.quality("chitchat", "m").unwrap().is_none());
    }

    #[test]
    fn keys_are_independent_per_kind_and_model() {
        let store = test_store();
        store
            .record("coding", "fast-m", AnswerOutcome::Fail)
            .unwrap();
        store
            .record("coding", "deep-m", AnswerOutcome::Gold)
            .unwrap();
        store
            .record("factual-qa", "deep-m", AnswerOutcome::Fail)
            .unwrap();
        assert_eq!(
            store.quality("coding", "fast-m").unwrap().unwrap().ratio,
            0.0
        );
        assert_eq!(
            store.quality("coding", "deep-m").unwrap().unwrap().ratio,
            1.0
        );
        assert_eq!(
            store
                .quality("factual-qa", "deep-m")
                .unwrap()
                .unwrap()
                .ratio,
            0.0
        );
    }

    #[test]
    fn quality_absent_for_unrecorded_pair() {
        let store = test_store();
        assert!(store.quality("coding", "ghost").unwrap().is_none());
    }

    #[test]
    fn disabled_store_records_nothing_and_reads_nothing() {
        let pool = SqlitePool::open_memory().unwrap();
        let store = AnswerFitnessStore::new(pool, false, DEFAULT_HALF_LIFE_HOURS);
        store.ensure_tables().unwrap();
        store.record("coding", "m", AnswerOutcome::Gold).unwrap();
        assert!(store.quality("coding", "m").unwrap().is_none());
    }

    // ── Property tests ────────────────────────────────────────────────
    //
    // The store reuses cerebellum's proven `decay`/`hours_between` (covered by
    // the fitness suite), so the load-bearing property here is the
    // outcome→mass mapping and the resulting ratio: a positive outcome can
    // only raise the success ratio, a negative can only lower it, and the
    // ratio always stays in `[0, 1]`.
    use proptest::prelude::*;

    fn outcome_strategy() -> impl Strategy<Value = AnswerOutcome> {
        prop_oneof![
            Just(AnswerOutcome::Gold),
            Just(AnswerOutcome::Success),
            Just(AnswerOutcome::Fail),
            Just(AnswerOutcome::Correction),
            Just(AnswerOutcome::None),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

        /// Whatever sequence of outcomes is recorded, the decayed ratio stays
        /// in `[0, 1]` and `uses` counts exactly the signal-bearing outcomes.
        #[test]
        fn ratio_bounded_and_uses_counts_signal(outcomes in prop::collection::vec(outcome_strategy(), 0..40)) {
            let store = test_store();
            let mut signal_count = 0i64;
            for o in &outcomes {
                store.record("k", "m", *o).unwrap();
                if *o != AnswerOutcome::None {
                    signal_count += 1;
                }
            }
            match store.quality("k", "m").unwrap() {
                Some(q) => {
                    prop_assert!((0.0..=1.0).contains(&q.ratio));
                    prop_assert_eq!(q.uses, signal_count);
                }
                None => prop_assert_eq!(signal_count, 0),
            }
        }
    }
}
