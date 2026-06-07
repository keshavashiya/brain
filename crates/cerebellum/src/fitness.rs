//! Learned capability fitness — the *learned* half of the capability
//! self-model.
//!
//! `intent::ToolUsage` is *declared* guidance (hand-authored: when to use,
//! cost, tier). This store records what actually happened: after each tool
//! dispatch the kernel reinforces a per-tool **success/failure mass**, which
//! decays under the forgetting curve so stale wins fade. The signal layer
//! then (a) uses it as a bounded tie-breaker when ranking the tools it
//! advertises to the chat model and (b) surfaces a "tools you've used
//! successfully" line in the SOUL capability digest.
//!
//! Keyed by `tool_id` exactly as `intent::ToolDescriptor.tool_id`
//! (`mcp:{server}:{tool}` or `native:{ns}.{action}`), so the learned record
//! joins directly against the live capability manifest.
//!
//! Backed by the shared SQLite pool (table `capability_fitness`, migration
//! v24). Decay is **lazy** — computed on every read/write from the elapsed
//! time since `last_used_at` — so there is no background sweeper.

use chrono::{DateTime, Utc};
use rusqlite::OptionalExtension;
use storage::SqlitePool;

use crate::CerebellumError;

/// Minimum lifetime invocations before a tool is "proven" enough to surface
/// in the digest or earn a ranking nudge. Below this we lack the evidence to
/// claim reliability. Behavioral tuning (not a deployment knob), so it lives
/// here as a named constant rather than in config.
pub const MIN_USES_TO_SURFACE: i64 = 3;

/// Minimum decayed success ratio for a tool to count as proven. A tool that
/// fails as often as it succeeds is not something to recommend or boost.
pub const MIN_RATIO_TO_SURFACE: f32 = 0.6;

/// Fallback decay half-life (hours) when config supplies a non-positive
/// value. 30 days — long enough that a genuinely useful tool stays "proven"
/// across weeks of intermittent use, short enough that a one-off win from
/// months ago doesn't linger.
pub const DEFAULT_HALF_LIFE_HOURS: f64 = 30.0 * 24.0;

/// Decayed fitness snapshot for one tool. `success`/`failure` are decayed
/// reinforcement masses (not raw tallies); `uses` is the undecayed lifetime
/// invocation count; `ratio` is `success / (success + failure)`.
#[derive(Debug, Clone)]
pub struct Fitness {
    pub tool_id: String,
    pub success: f32,
    pub failure: f32,
    pub uses: i64,
    pub ratio: f32,
}

/// Per-tool learned success/failure record. See module docs.
pub struct CapabilityFitnessStore {
    db: SqlitePool,
    enabled: bool,
    half_life_hours: f64,
}

impl CapabilityFitnessStore {
    /// Create a fitness store. A non-positive `half_life_hours` falls back to
    /// [`DEFAULT_HALF_LIFE_HOURS`]. When `enabled` is false the store is inert:
    /// `record` is a no-op and `proven_tools` returns empty (the table is still
    /// created so toggling the flag on later just starts accumulating).
    pub fn new(db: SqlitePool, enabled: bool, half_life_hours: f64) -> Self {
        Self {
            db,
            enabled,
            half_life_hours: if half_life_hours > 0.0 {
                half_life_hours
            } else {
                DEFAULT_HALF_LIFE_HOURS
            },
        }
    }

    /// Whether learning is on for this deployment.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Ensure the `capability_fitness` table exists (idempotent).
    ///
    /// The storage migration layer (`SqlitePool::migrate`) already creates it
    /// (v24); this mirrors [`crate::ProcedureStore::ensure_tables`] so a caller
    /// can construct the store on a not-yet-migrated pool without knowing
    /// migration state.
    pub fn ensure_tables(&self) -> Result<(), CerebellumError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS capability_fitness (
                    tool_id      TEXT PRIMARY KEY,
                    success_mass REAL    NOT NULL DEFAULT 0,
                    failure_mass REAL    NOT NULL DEFAULT 0,
                    uses         INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );",
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Reinforce `tool_id` with one outcome. Decays the existing masses to now,
    /// then adds `1.0` to the success or failure mass and bumps `uses`. No-op
    /// when the store is disabled.
    pub fn record(&self, tool_id: &str, success: bool) -> Result<(), CerebellumError> {
        if !self.enabled {
            return Ok(());
        }
        let now = Utc::now();
        let half_life = self.half_life_hours;
        self.db.with_conn(|conn| {
            let existing: Option<(f64, f64, i64, String)> = conn
                .query_row(
                    "SELECT success_mass, failure_mass, uses, last_used_at
                     FROM capability_fitness WHERE tool_id = ?1",
                    [tool_id],
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
            if success {
                s += 1.0;
            } else {
                f += 1.0;
            }
            conn.execute(
                "INSERT INTO capability_fitness
                     (tool_id, success_mass, failure_mass, uses, last_used_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(tool_id) DO UPDATE SET
                     success_mass = excluded.success_mass,
                     failure_mass = excluded.failure_mass,
                     uses         = excluded.uses,
                     last_used_at = excluded.last_used_at",
                rusqlite::params![tool_id, s, f, uses + 1, now.to_rfc3339()],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Decayed fitness for a single tool, or `None` if never recorded.
    pub fn fitness(&self, tool_id: &str) -> Result<Option<Fitness>, CerebellumError> {
        let now = Utc::now();
        let half_life = self.half_life_hours;
        let row = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT success_mass, failure_mass, uses, last_used_at
                 FROM capability_fitness WHERE tool_id = ?1",
                [tool_id],
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
            decayed_fitness(tool_id, s, f, uses, hours, half_life)
        }))
    }

    /// Tools whose decayed record clears the proven bar (`uses >= min_uses`,
    /// `ratio >= min_ratio`, non-zero decayed success), best first by decayed
    /// success mass then tool_id. Drives both the ranking nudge and the digest
    /// line. Empty when disabled.
    pub fn proven_tools(
        &self,
        min_uses: i64,
        min_ratio: f32,
        limit: usize,
    ) -> Result<Vec<Fitness>, CerebellumError> {
        if !self.enabled || limit == 0 {
            return Ok(Vec::new());
        }
        let now = Utc::now();
        let half_life = self.half_life_hours;
        let rows = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT tool_id, success_mass, failure_mass, uses, last_used_at
                 FROM capability_fitness WHERE uses >= ?1",
            )?;
            let rows = stmt
                .query_map([min_uses], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, f64>(1)?,
                        r.get::<_, f64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;
        let mut out: Vec<Fitness> = rows
            .into_iter()
            .map(|(id, s, f, uses, last)| {
                let hours = hours_between(&last, now);
                decayed_fitness(&id, s, f, uses, hours, half_life)
            })
            .filter(|fit| fit.success > 0.0 && fit.ratio >= min_ratio)
            .collect();
        out.sort_by(|a, b| {
            b.success
                .partial_cmp(&a.success)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.tool_id.cmp(&b.tool_id))
        });
        out.truncate(limit);
        Ok(out)
    }
}

/// Exponential forgetting-curve decay — the same form as
/// `hippocampus::search::forgetting_curve` (`value * e^(-λ·Δt)`), here in the
/// half-life parameterization `mass * 2^(-Δt / half_life)`. Kept as a local
/// pure fn so cerebellum carries no dependency on hippocampus.
pub(crate) fn decay(mass: f64, hours_elapsed: f64, half_life_hours: f64) -> f64 {
    if mass <= 0.0 || hours_elapsed <= 0.0 || half_life_hours <= 0.0 {
        return mass.max(0.0);
    }
    mass * (-std::f64::consts::LN_2 * hours_elapsed / half_life_hours).exp()
}

/// Build a decayed [`Fitness`] from stored masses + elapsed hours.
fn decayed_fitness(
    tool_id: &str,
    s: f64,
    f: f64,
    uses: i64,
    hours: f64,
    half_life: f64,
) -> Fitness {
    let s = decay(s, hours, half_life);
    let f = decay(f, hours, half_life);
    let total = s + f;
    let ratio = if total > 0.0 { (s / total) as f32 } else { 0.0 };
    Fitness {
        tool_id: tool_id.to_string(),
        success: s as f32,
        failure: f as f32,
        uses,
        ratio,
    }
}

/// Whole hours between an RFC3339 `last` timestamp and `now` (never negative).
/// A row whose timestamp doesn't parse is treated as just-used (no decay) —
/// the only writer ([`CapabilityFitnessStore::record`]) always stores RFC3339,
/// so this only guards against hand-edited rows.
fn hours_between(last: &str, now: DateTime<Utc>) -> f64 {
    match DateTime::parse_from_rfc3339(last) {
        Ok(t) => {
            let secs = (now - t.with_timezone(&Utc)).num_seconds();
            (secs as f64 / 3600.0).max(0.0)
        }
        Err(_) => 0.0,
    }
}

/// Bounded ranking nudge for a proven tool, in `[0.0, 0.99]`.
///
/// Scales the decayed success ratio by a saturating function of decayed
/// success mass (more proven wins → closer to the full ratio). Clamped below
/// `1.0` so it can only ever break ties among tools with **equal** integer
/// keyword overlap — never overtake a tool that matched one more query term.
pub fn fitness_bonus(f: &Fitness) -> f32 {
    if f.uses <= 0 || f.success <= 0.0 {
        return 0.0;
    }
    // Soft saturation: ~0.5 at one win, ~0.8 at four, asymptotes to 1.0.
    let saturation = f.success / (f.success + 1.0);
    (f.ratio * saturation).clamp(0.0, 0.99)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> CapabilityFitnessStore {
        let pool = SqlitePool::open_memory().unwrap();
        let store = CapabilityFitnessStore::new(pool, true, DEFAULT_HALF_LIFE_HOURS);
        store.ensure_tables().unwrap();
        store
    }

    #[test]
    fn record_success_then_failure_accumulates_mass() {
        let store = test_store();
        store.record("native:net.http", true).unwrap();
        store.record("native:net.http", true).unwrap();
        store.record("native:net.http", false).unwrap();
        let fit = store.fitness("native:net.http").unwrap().unwrap();
        assert_eq!(fit.uses, 3);
        // Decay over the ~instant elapsed is negligible: ~2 success, ~1 failure.
        assert!((fit.success - 2.0).abs() < 0.01, "{}", fit.success);
        assert!((fit.failure - 1.0).abs() < 0.01, "{}", fit.failure);
        assert!((fit.ratio - 2.0 / 3.0).abs() < 0.01, "{}", fit.ratio);
    }

    #[test]
    fn fitness_absent_for_unrecorded_tool() {
        let store = test_store();
        assert!(store.fitness("mcp:ghost:noop").unwrap().is_none());
    }

    #[test]
    fn disabled_store_records_nothing_and_lists_nothing() {
        let pool = SqlitePool::open_memory().unwrap();
        let store = CapabilityFitnessStore::new(pool, false, DEFAULT_HALF_LIFE_HOURS);
        store.ensure_tables().unwrap();
        store.record("native:net.http", true).unwrap();
        assert!(store.fitness("native:net.http").unwrap().is_none());
        assert!(store
            .proven_tools(MIN_USES_TO_SURFACE, MIN_RATIO_TO_SURFACE, 8)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn proven_tools_applies_use_and_ratio_thresholds() {
        let store = test_store();
        // Reliable + enough uses → proven.
        for _ in 0..4 {
            store.record("mcp:web:search", true).unwrap();
        }
        // Enough uses but mostly failing → not proven.
        store.record("native:notify.send", true).unwrap();
        for _ in 0..4 {
            store.record("native:notify.send", false).unwrap();
        }
        // Reliable but too few uses → not proven.
        store.record("native:memory.store", true).unwrap();

        let proven = store
            .proven_tools(MIN_USES_TO_SURFACE, MIN_RATIO_TO_SURFACE, 8)
            .unwrap();
        let ids: Vec<&str> = proven.iter().map(|f| f.tool_id.as_str()).collect();
        assert_eq!(ids, vec!["mcp:web:search"], "only the reliable, used tool");
    }

    #[test]
    fn proven_tools_orders_by_decayed_success_then_id() {
        let store = test_store();
        for _ in 0..3 {
            store.record("mcp:b:tool", true).unwrap();
        }
        for _ in 0..6 {
            store.record("mcp:a:tool", true).unwrap();
        }
        let proven = store
            .proven_tools(MIN_USES_TO_SURFACE, MIN_RATIO_TO_SURFACE, 8)
            .unwrap();
        // a has more wins → first despite later id.
        assert_eq!(proven[0].tool_id, "mcp:a:tool");
        assert_eq!(proven[1].tool_id, "mcp:b:tool");
    }

    #[test]
    fn proven_tools_limit_caps_results() {
        let store = test_store();
        for tool in ["mcp:a:t", "mcp:b:t", "mcp:c:t"] {
            for _ in 0..3 {
                store.record(tool, true).unwrap();
            }
        }
        assert_eq!(
            store
                .proven_tools(MIN_USES_TO_SURFACE, MIN_RATIO_TO_SURFACE, 2)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn decay_halves_mass_after_one_half_life() {
        let hl = 100.0;
        assert!((decay(1.0, hl, hl) - 0.5).abs() < 1e-9);
        assert!((decay(1.0, 2.0 * hl, hl) - 0.25).abs() < 1e-9);
        // Non-positive inputs are inert.
        assert_eq!(decay(1.0, 0.0, hl), 1.0);
        assert_eq!(decay(0.0, hl, hl), 0.0);
    }

    #[test]
    fn hours_between_is_nonnegative_and_parse_tolerant() {
        let now = Utc::now();
        let past = (now - chrono::Duration::hours(5)).to_rfc3339();
        assert!((hours_between(&past, now) - 5.0).abs() < 0.01);
        // Future timestamp clamps to 0 (no negative decay).
        let future = (now + chrono::Duration::hours(5)).to_rfc3339();
        assert_eq!(hours_between(&future, now), 0.0);
        // Unparseable → 0.
        assert_eq!(hours_between("not-a-date", now), 0.0);
    }

    #[test]
    fn fitness_bonus_is_bounded_and_monotone() {
        let mk = |s: f32, f: f32, uses: i64| Fitness {
            tool_id: "x".into(),
            success: s,
            failure: f,
            uses,
            ratio: if s + f > 0.0 { s / (s + f) } else { 0.0 },
        };
        // Never reaches 1.0 even for a perfect, heavily-used tool.
        let strong = fitness_bonus(&mk(50.0, 0.0, 50));
        assert!(strong < 1.0 && strong > 0.9, "{strong}");
        // A 50/50 tool earns much less than a reliable one with equal uses.
        let mixed = fitness_bonus(&mk(5.0, 5.0, 10));
        let reliable = fitness_bonus(&mk(10.0, 0.0, 10));
        assert!(mixed < reliable, "{mixed} < {reliable}");
        // No evidence → no bonus.
        assert_eq!(fitness_bonus(&mk(0.0, 0.0, 0)), 0.0);
    }

    // ── Property tests ────────────────────────────────────────────────
    //
    // Two pure functions carry correctness load. `decay` is the forgetting
    // curve: it must never invent mass (grow a value) or go negative, and it
    // must fade monotonically with elapsed time. `fitness_bonus` is a learned
    // ranking nudge whose whole safety story is the `[0, 0.99]` bound — it may
    // only break ties among tools with equal keyword overlap, never overtake a
    // tool that matched one more query term. If that ceiling ever reached 1.0
    // the learned signal could hijack capability selection.
    use proptest::prelude::*;

    fn fitness_from(s: f32, f: f32, uses: i64) -> Fitness {
        let total = s + f;
        Fitness {
            tool_id: "x".into(),
            success: s,
            failure: f,
            uses,
            ratio: if total > 0.0 { s / total } else { 0.0 },
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// Decay never invents mass or goes negative: for a non-negative
        /// input the result stays in `[0, mass]`, whatever the elapsed time
        /// or half-life (including the non-positive guard cases).
        #[test]
        fn decay_stays_within_zero_and_input(
            mass in 0.0f64..1e6,
            hours in -10.0f64..1e6,
            half_life in -10.0f64..1e6,
        ) {
            let d = decay(mass, hours, half_life);
            prop_assert!(d >= 0.0);
            prop_assert!(d <= mass);
        }

        /// More elapsed time never retains more mass — decay is monotone
        /// non-increasing in the hours elapsed.
        #[test]
        fn decay_is_monotone_in_elapsed(
            mass in 0.01f64..1e4,
            half_life in 0.1f64..1e4,
            h1 in 0.0f64..1e5,
            h2 in 0.0f64..1e5,
        ) {
            let (lo, hi) = if h1 <= h2 { (h1, h2) } else { (h2, h1) };
            let near = decay(mass, lo, half_life);
            let far = decay(mass, hi, half_life);
            prop_assert!(near + 1e-9 >= far);
        }

        /// A longer half-life retains more mass for the same elapsed time —
        /// decay is monotone non-decreasing in the half-life.
        #[test]
        fn decay_is_monotone_in_half_life(
            mass in 0.01f64..1e4,
            hours in 0.1f64..1e4,
            hl1 in 0.1f64..1e4,
            hl2 in 0.1f64..1e4,
        ) {
            let (lo, hi) = if hl1 <= hl2 { (hl1, hl2) } else { (hl2, hl1) };
            let shorter = decay(mass, hours, lo);
            let longer = decay(mass, hours, hi);
            prop_assert!(longer + 1e-9 >= shorter);
        }

        /// One half-life halves the mass; n half-lives scale by `2^-n`
        /// (the curve's defining property).
        #[test]
        fn decay_halves_each_half_life(
            mass in 0.1f64..1e3,
            half_life in 0.1f64..1e3,
            n in 0u32..6,
        ) {
            let elapsed = half_life * n as f64;
            let got = decay(mass, elapsed, half_life);
            let expected = mass * 0.5f64.powi(n as i32);
            prop_assert!((got - expected).abs() <= expected * 1e-6 + 1e-12);
        }

        /// A non-positive elapsed time leaves the mass untouched (the guard
        /// branch): nothing decays into the past.
        #[test]
        fn decay_is_inert_for_nonpositive_elapsed(
            mass in 0.0f64..1e6,
            hours in -1e6f64..=0.0,
            half_life in 0.1f64..1e4,
        ) {
            prop_assert_eq!(decay(mass, hours, half_life), mass);
        }

        /// The learned ranking nudge is always within `[0, 0.99]` — for any
        /// fitness record, however heavily and reliably used. This is the
        /// invariant that keeps a learned boost from ever outranking a real
        /// keyword match.
        #[test]
        fn fitness_bonus_never_reaches_one(
            s in 0.0f32..1e4,
            f in 0.0f32..1e4,
            uses in 0i64..100_000,
        ) {
            let b = fitness_bonus(&fitness_from(s, f, uses));
            prop_assert!((0.0..=0.99).contains(&b));
        }

        /// No evidence, no nudge: a tool with no recorded uses or no decayed
        /// success earns exactly zero.
        #[test]
        fn fitness_bonus_zero_without_evidence(s in 0.0f32..1e4, f in 0.0f32..1e4) {
            // uses == 0 (unused) ...
            prop_assert_eq!(fitness_bonus(&fitness_from(s, f, 0)), 0.0);
            // ... or zero success mass (only failures recorded).
            prop_assert_eq!(fitness_bonus(&fitness_from(0.0, f, 5)), 0.0);
        }

        /// The nudge never exceeds the raw success ratio it is built from
        /// (the saturation factor is `<= 1`), so a flaky tool can't be
        /// boosted above its observed reliability.
        #[test]
        fn fitness_bonus_never_exceeds_ratio(
            s in 0.0f32..1e4,
            f in 0.0f32..1e4,
            uses in 0i64..100_000,
        ) {
            let fit = fitness_from(s, f, uses);
            prop_assert!(fitness_bonus(&fit) <= fit.ratio + f32::EPSILON);
        }

        /// More proven wins (failure held at zero, so ratio stays 1.0) only
        /// ever raise the nudge — the saturation term is monotone in success
        /// mass.
        #[test]
        fn fitness_bonus_monotone_in_success(a in 0.01f32..1e4, b in 0.01f32..1e4) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let weak = fitness_bonus(&fitness_from(lo, 0.0, 10));
            let strong = fitness_bonus(&fitness_from(hi, 0.0, 10));
            prop_assert!(strong + f32::EPSILON >= weak);
        }

        /// Elapsed-hours is never negative and recovers a known past offset:
        /// a timestamp N hours ago reads back as ~N, a future one clamps to 0.
        #[test]
        fn hours_between_nonnegative_and_recovers_offset(offset in 0i64..100_000) {
            let now = Utc::now();
            let past = (now - chrono::Duration::hours(offset)).to_rfc3339();
            let got = hours_between(&past, now);
            prop_assert!(got >= 0.0);
            prop_assert!((got - offset as f64).abs() <= 1.0);

            let future = (now + chrono::Duration::hours(offset)).to_rfc3339();
            prop_assert_eq!(hours_between(&future, now), 0.0);
        }
    }
}
