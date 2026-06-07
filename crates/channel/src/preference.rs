//! Channel preferences — learned and explicit.
//!
//! Preferences are scored per `(namespace, category, channel_id)`. A higher
//! weight means the router should try this channel earlier for deliveries
//! in that category. Weights are learned by observing which channels the
//! user actually responds on, and can be pinned by explicit configuration.
//!
//! Design notes
//! - Two tables: `channel_preferences` (rolled-up scores) and
//!   `channel_interactions` (raw deliver/response events, useful for
//!   retrospective inspection and richer learning later).
//! - Weight update uses a bounded EMA so the score stays in `[0.0, 1.0]`
//!   and recent behavior is prioritized without wiping prior signal.
//! - A preference row with `pinned = 1` represents an explicit user choice
//!   and is never overwritten by learned updates.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use storage::SqlitePool;

use crate::error::ChannelError;
use crate::types::DeliveryCategory;

/// EMA smoothing factor — higher = faster adaptation, lower = more stable.
const EMA_ALPHA: f32 = 0.25;

/// Positive signal: user responded within the attention window.
const SIGNAL_RESPONDED: f32 = 1.0;
/// Neutral signal: delivered but no response before window expired.
/// (We decay rather than penalize hard; a single miss shouldn't reset a
/// preference that's been reliable for weeks.)
const SIGNAL_NO_RESPONSE: f32 = 0.2;
/// Negative signal: delivery itself failed (transport error).
const SIGNAL_DELIVERY_FAIL: f32 = 0.0;

/// One exponential-moving-average step on a learned weight: nudge the prior
/// weight toward the new `signal` by [`EMA_ALPHA`]. For a `prev` and `signal`
/// both in `[0.0, 1.0]` this is a convex combination, so the result stays in
/// range and never overshoots either endpoint — the caller still clamps as a
/// guard against an out-of-range stored weight.
fn ema_step(prev: f32, signal: f32) -> f32 {
    (1.0 - EMA_ALPHA) * prev + EMA_ALPHA * signal
}

/// One learned preference row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelPreference {
    pub namespace: String,
    pub category: DeliveryCategory,
    pub channel_id: String,
    /// Learned weight in `[0.0, 1.0]`. Higher = prefer this channel.
    pub weight: f32,
    /// Total deliveries recorded for this (ns, category, channel).
    pub response_count: u32,
    /// Successful responses recorded (subset of response_count).
    pub success_count: u32,
    /// Set to true when the user explicitly pinned the preference —
    /// learned updates do not overwrite pinned rows.
    pub pinned: bool,
    pub last_updated: DateTime<Utc>,
}

/// One recorded deliver/response event. Fed into
/// [`ChannelPreferenceStore::record_interaction`] to update weights.
#[derive(Debug, Clone)]
pub struct RecordedInteraction {
    pub namespace: String,
    pub category: DeliveryCategory,
    pub channel_id: String,
    pub delivered_at: DateTime<Utc>,
    /// Populated if the user responded — regardless of approve/reject.
    pub responded_at: Option<DateTime<Utc>>,
    /// Whether the transport delivery itself succeeded (distinct from
    /// whether the user responded).
    pub delivered_ok: bool,
}

impl RecordedInteraction {
    /// Classify this interaction into a numeric signal for the EMA update.
    fn signal(&self) -> f32 {
        if !self.delivered_ok {
            SIGNAL_DELIVERY_FAIL
        } else if self.responded_at.is_some() {
            SIGNAL_RESPONDED
        } else {
            SIGNAL_NO_RESPONSE
        }
    }

    fn response_ms(&self) -> Option<i64> {
        self.responded_at
            .map(|r| (r - self.delivered_at).num_milliseconds().max(0))
    }
}

/// Storage trait for channel preferences.
#[async_trait]
pub trait ChannelPreferenceStore: Send + Sync {
    /// Update preference weights based on a deliver/response event.
    async fn record_interaction(
        &self,
        interaction: RecordedInteraction,
    ) -> Result<(), ChannelError>;

    /// Fetch preferences for a `(namespace, category)` pair, ordered by
    /// weight descending. Unpinned rows with `weight < min_weight` are
    /// filtered out so the router doesn't waste attempts on decayed
    /// channels.
    async fn get_preferences(
        &self,
        namespace: &str,
        category: DeliveryCategory,
        min_weight: f32,
    ) -> Result<Vec<ChannelPreference>, ChannelError>;

    /// Explicitly set (and pin) a preference. Used by `brain channel prefer ...`.
    async fn upsert_preference(
        &self,
        namespace: &str,
        category: DeliveryCategory,
        channel_id: &str,
        weight: f32,
        pinned: bool,
    ) -> Result<(), ChannelError>;

    /// List every preference in a namespace (for CLI inspection).
    async fn list_all(&self, namespace: &str) -> Result<Vec<ChannelPreference>, ChannelError>;

    /// Remove a preference row (user override reset).
    async fn delete(
        &self,
        namespace: &str,
        category: DeliveryCategory,
        channel_id: &str,
    ) -> Result<bool, ChannelError>;
}

/// SQLite-backed preference store.
pub struct SqlitePreferenceStore {
    db: SqlitePool,
}

impl SqlitePreferenceStore {
    pub fn new(db: SqlitePool) -> Self {
        Self { db }
    }

    /// Create tables if missing. Safe to call repeatedly.
    pub fn ensure_tables(&self) -> Result<(), ChannelError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS channel_preferences (
                    namespace      TEXT NOT NULL,
                    category       TEXT NOT NULL,
                    channel_id     TEXT NOT NULL,
                    weight         REAL NOT NULL,
                    response_count INTEGER NOT NULL DEFAULT 0,
                    success_count  INTEGER NOT NULL DEFAULT 0,
                    pinned         INTEGER NOT NULL DEFAULT 0,
                    last_updated   TEXT NOT NULL,
                    PRIMARY KEY (namespace, category, channel_id)
                );
                CREATE INDEX IF NOT EXISTS idx_channel_prefs_lookup
                    ON channel_preferences(namespace, category, weight DESC);

                CREATE TABLE IF NOT EXISTS channel_interactions (
                    id           TEXT PRIMARY KEY,
                    namespace    TEXT NOT NULL,
                    category     TEXT NOT NULL,
                    channel_id   TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    responded_at TEXT,
                    delivered_ok INTEGER NOT NULL,
                    response_ms  INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_channel_interactions_ns
                    ON channel_interactions(namespace, category, delivered_at DESC);
                "#,
            )?;
            Ok(())
        })?;
        Ok(())
    }
}

#[async_trait]
impl ChannelPreferenceStore for SqlitePreferenceStore {
    async fn record_interaction(
        &self,
        interaction: RecordedInteraction,
    ) -> Result<(), ChannelError> {
        let signal = interaction.signal();
        let response_ms = interaction.response_ms();
        let now = Utc::now().to_rfc3339();
        let id = uuid::Uuid::new_v4().to_string();
        let cat_str = interaction.category.as_str().to_string();

        self.db.with_conn(move |conn| {
            // Log the raw interaction.
            conn.execute(
                r#"INSERT INTO channel_interactions
                   (id, namespace, category, channel_id, delivered_at, responded_at, delivered_ok, response_ms)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
                params![
                    id,
                    interaction.namespace,
                    cat_str,
                    interaction.channel_id,
                    interaction.delivered_at.to_rfc3339(),
                    interaction.responded_at.map(|t| t.to_rfc3339()),
                    if interaction.delivered_ok { 1i32 } else { 0 },
                    response_ms,
                ],
            )?;

            // Upsert the rolled-up preference — but never overwrite a pinned row's weight.
            let existing: Option<(f32, u32, u32, i32)> = conn
                .query_row(
                    r#"SELECT weight, response_count, success_count, pinned
                       FROM channel_preferences
                       WHERE namespace = ?1 AND category = ?2 AND channel_id = ?3"#,
                    params![interaction.namespace, cat_str, interaction.channel_id],
                    |row| {
                        Ok((
                            row.get::<_, f64>(0)? as f32,
                            row.get::<_, i64>(1)? as u32,
                            row.get::<_, i64>(2)? as u32,
                            row.get::<_, i32>(3)?,
                        ))
                    },
                )
                .ok();

            let (new_weight, new_resp_count, new_succ_count, pinned) = match existing {
                Some((w, rc, sc, p)) => {
                    let weight = if p == 1 {
                        // Pinned — keep user-set weight, just update counters.
                        w
                    } else {
                        ema_step(w, signal)
                    };
                    let weight = weight.clamp(0.0, 1.0);
                    let rc = rc + 1;
                    let sc = if interaction.responded_at.is_some() {
                        sc + 1
                    } else {
                        sc
                    };
                    (weight, rc, sc, p)
                }
                None => {
                    // First observation — seed with the signal value.
                    let weight = signal.clamp(0.0, 1.0);
                    let sc = if interaction.responded_at.is_some() { 1 } else { 0 };
                    (weight, 1u32, sc, 0)
                }
            };

            conn.execute(
                r#"INSERT INTO channel_preferences
                     (namespace, category, channel_id, weight, response_count, success_count, pinned, last_updated)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                   ON CONFLICT(namespace, category, channel_id) DO UPDATE SET
                     weight = excluded.weight,
                     response_count = excluded.response_count,
                     success_count = excluded.success_count,
                     last_updated = excluded.last_updated"#,
                params![
                    interaction.namespace,
                    cat_str,
                    interaction.channel_id,
                    new_weight as f64,
                    new_resp_count as i64,
                    new_succ_count as i64,
                    pinned,
                    now,
                ],
            )?;
            Ok(())
        })?;

        Ok(())
    }

    async fn get_preferences(
        &self,
        namespace: &str,
        category: DeliveryCategory,
        min_weight: f32,
    ) -> Result<Vec<ChannelPreference>, ChannelError> {
        let cat_str = category.as_str().to_string();
        let ns = namespace.to_string();
        let min = min_weight as f64;

        let rows: Vec<ChannelPreference> = self.db.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                r#"SELECT namespace, category, channel_id, weight, response_count,
                          success_count, pinned, last_updated
                   FROM channel_preferences
                   WHERE namespace = ?1 AND category = ?2
                     AND (pinned = 1 OR weight >= ?3)
                   ORDER BY pinned DESC, weight DESC, last_updated DESC"#,
            )?;

            let iter = stmt.query_map(params![ns, cat_str, min], |row| {
                Ok(ChannelPreference {
                    namespace: row.get(0)?,
                    category: DeliveryCategory::parse(&row.get::<_, String>(1)?)
                        .unwrap_or(DeliveryCategory::Response),
                    channel_id: row.get(2)?,
                    weight: row.get::<_, f64>(3)? as f32,
                    response_count: row.get::<_, i64>(4)? as u32,
                    success_count: row.get::<_, i64>(5)? as u32,
                    pinned: row.get::<_, i32>(6)? == 1,
                    last_updated: row
                        .get::<_, String>(7)?
                        .parse::<DateTime<Utc>>()
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?;

            let mut out = Vec::new();
            for r in iter {
                out.push(r?);
            }
            Ok(out)
        })?;

        Ok(rows)
    }

    async fn upsert_preference(
        &self,
        namespace: &str,
        category: DeliveryCategory,
        channel_id: &str,
        weight: f32,
        pinned: bool,
    ) -> Result<(), ChannelError> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(ChannelError::InvalidWeight(weight));
        }
        let ns = namespace.to_string();
        let ch = channel_id.to_string();
        let cat_str = category.as_str().to_string();
        let w = weight as f64;
        let now = Utc::now().to_rfc3339();

        self.db.with_conn(move |conn| {
            conn.execute(
                r#"INSERT INTO channel_preferences
                     (namespace, category, channel_id, weight, response_count, success_count, pinned, last_updated)
                     VALUES (?1, ?2, ?3, ?4, 0, 0, ?5, ?6)
                   ON CONFLICT(namespace, category, channel_id) DO UPDATE SET
                     weight = excluded.weight,
                     pinned = excluded.pinned,
                     last_updated = excluded.last_updated"#,
                params![
                    ns,
                    cat_str,
                    ch,
                    w,
                    if pinned { 1i32 } else { 0 },
                    now,
                ],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    async fn list_all(&self, namespace: &str) -> Result<Vec<ChannelPreference>, ChannelError> {
        let ns = namespace.to_string();
        let rows: Vec<ChannelPreference> = self.db.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                r#"SELECT namespace, category, channel_id, weight, response_count,
                          success_count, pinned, last_updated
                   FROM channel_preferences
                   WHERE namespace = ?1
                   ORDER BY category, pinned DESC, weight DESC"#,
            )?;
            let iter = stmt.query_map(params![ns], |row| {
                Ok(ChannelPreference {
                    namespace: row.get(0)?,
                    category: DeliveryCategory::parse(&row.get::<_, String>(1)?)
                        .unwrap_or(DeliveryCategory::Response),
                    channel_id: row.get(2)?,
                    weight: row.get::<_, f64>(3)? as f32,
                    response_count: row.get::<_, i64>(4)? as u32,
                    success_count: row.get::<_, i64>(5)? as u32,
                    pinned: row.get::<_, i32>(6)? == 1,
                    last_updated: row
                        .get::<_, String>(7)?
                        .parse::<DateTime<Utc>>()
                        .unwrap_or_else(|_| Utc::now()),
                })
            })?;
            let mut out = Vec::new();
            for r in iter {
                out.push(r?);
            }
            Ok(out)
        })?;
        Ok(rows)
    }

    async fn delete(
        &self,
        namespace: &str,
        category: DeliveryCategory,
        channel_id: &str,
    ) -> Result<bool, ChannelError> {
        let ns = namespace.to_string();
        let ch = channel_id.to_string();
        let cat_str = category.as_str().to_string();
        let n = self.db.with_conn(move |conn| {
            let changed = conn.execute(
                r#"DELETE FROM channel_preferences
                   WHERE namespace = ?1 AND category = ?2 AND channel_id = ?3"#,
                params![ns, cat_str, ch],
            )?;
            Ok(changed)
        })?;
        Ok(n > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    async fn mk_store() -> SqlitePreferenceStore {
        let db = SqlitePool::open_memory().unwrap();
        let store = SqlitePreferenceStore::new(db);
        store.ensure_tables().unwrap();
        store
    }

    fn delivered_ok_response(channel: &str, delay_s: i64) -> RecordedInteraction {
        let delivered = Utc::now();
        let responded = delivered + Duration::seconds(delay_s);
        RecordedInteraction {
            namespace: "personal".into(),
            category: DeliveryCategory::Confirm,
            channel_id: channel.into(),
            delivered_at: delivered,
            responded_at: Some(responded),
            delivered_ok: true,
        }
    }

    fn delivered_no_response(channel: &str) -> RecordedInteraction {
        RecordedInteraction {
            namespace: "personal".into(),
            category: DeliveryCategory::Confirm,
            channel_id: channel.into(),
            delivered_at: Utc::now(),
            responded_at: None,
            delivered_ok: true,
        }
    }

    #[tokio::test]
    async fn first_interaction_seeds_weight() {
        let store = mk_store().await;
        store
            .record_interaction(delivered_ok_response("telegram", 5))
            .await
            .unwrap();
        let prefs = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.0)
            .await
            .unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].channel_id, "telegram");
        assert!((prefs[0].weight - 1.0).abs() < 1e-6);
        assert_eq!(prefs[0].response_count, 1);
        assert_eq!(prefs[0].success_count, 1);
    }

    #[tokio::test]
    async fn ema_decays_on_no_response() {
        let store = mk_store().await;
        // First interaction: good response
        store
            .record_interaction(delivered_ok_response("telegram", 3))
            .await
            .unwrap();
        let before = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.0)
            .await
            .unwrap()[0]
            .weight;

        // Second: delivered but no response
        store
            .record_interaction(delivered_no_response("telegram"))
            .await
            .unwrap();
        let after = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.0)
            .await
            .unwrap()[0]
            .weight;

        assert!(after < before, "weight should decay after a miss");
        assert!(after > 0.0);
    }

    #[tokio::test]
    async fn pinned_weight_not_overwritten() {
        let store = mk_store().await;
        store
            .upsert_preference("personal", DeliveryCategory::Confirm, "telegram", 0.9, true)
            .await
            .unwrap();

        // Unresponded interactions would normally decay weight — pinned ignores them.
        for _ in 0..5 {
            store
                .record_interaction(delivered_no_response("telegram"))
                .await
                .unwrap();
        }

        let prefs = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.0)
            .await
            .unwrap();
        assert_eq!(prefs.len(), 1);
        assert!((prefs[0].weight - 0.9).abs() < 1e-6);
        assert!(prefs[0].pinned);
        assert_eq!(prefs[0].response_count, 5);
    }

    #[tokio::test]
    async fn invalid_weight_rejected() {
        let store = mk_store().await;
        let err = store
            .upsert_preference("personal", DeliveryCategory::Confirm, "x", 1.5, false)
            .await
            .unwrap_err();
        assert!(matches!(err, ChannelError::InvalidWeight(_)));
    }

    #[tokio::test]
    async fn min_weight_filter_skips_unpinned() {
        let store = mk_store().await;
        store
            .upsert_preference("personal", DeliveryCategory::Confirm, "low", 0.1, false)
            .await
            .unwrap();
        store
            .upsert_preference("personal", DeliveryCategory::Confirm, "high", 0.8, false)
            .await
            .unwrap();
        let prefs = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.5)
            .await
            .unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].channel_id, "high");
    }

    #[tokio::test]
    async fn min_weight_filter_keeps_pinned_even_if_low() {
        let store = mk_store().await;
        store
            .upsert_preference(
                "personal",
                DeliveryCategory::Confirm,
                "pinned-low",
                0.05,
                true,
            )
            .await
            .unwrap();
        let prefs = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.5)
            .await
            .unwrap();
        assert_eq!(prefs.len(), 1);
        assert!(prefs[0].pinned);
    }

    #[tokio::test]
    async fn delete_removes_row() {
        let store = mk_store().await;
        store
            .upsert_preference(
                "personal",
                DeliveryCategory::Confirm,
                "telegram",
                0.7,
                false,
            )
            .await
            .unwrap();
        let removed = store
            .delete("personal", DeliveryCategory::Confirm, "telegram")
            .await
            .unwrap();
        assert!(removed);
        let prefs = store
            .get_preferences("personal", DeliveryCategory::Confirm, 0.0)
            .await
            .unwrap();
        assert!(prefs.is_empty());
    }

    #[tokio::test]
    async fn list_all_groups_by_category() {
        let store = mk_store().await;
        store
            .upsert_preference("personal", DeliveryCategory::Confirm, "telegram", 0.9, true)
            .await
            .unwrap();
        store
            .upsert_preference("personal", DeliveryCategory::Nudge, "desktop", 0.5, false)
            .await
            .unwrap();

        let all = store.list_all("personal").await.unwrap();
        assert_eq!(all.len(), 2);
        // Sorted by category asc then weight desc
    }

    // ── Property tests ────────────────────────────────────────────────
    //
    // The learned channel weight must stay a valid `[0, 1]` preference under
    // any stream of interactions: a single EMA step is a convex blend that
    // can't overshoot, moves toward the observed signal, and converges to a
    // repeated one. And every interaction classifies to a signal already in
    // range, so the blend's validity precondition always holds.
    use proptest::prelude::*;

    fn interaction(delivered_ok: bool, responded: bool) -> RecordedInteraction {
        let t = Utc::now();
        RecordedInteraction {
            namespace: "personal".into(),
            category: DeliveryCategory::Nudge,
            channel_id: "c".into(),
            delivered_at: t,
            responded_at: responded.then(|| t + Duration::seconds(1)),
            delivered_ok,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// One EMA step on a valid weight with a valid signal lands within
        /// the closed span of the two — it never overshoots either endpoint,
        /// so the learned weight can never leave `[0, 1]`.
        #[test]
        fn ema_step_stays_within_endpoints(prev in 0.0f32..=1.0, signal in 0.0f32..=1.0) {
            let next = ema_step(prev, signal);
            let lo = prev.min(signal);
            let hi = prev.max(signal);
            prop_assert!(next >= lo - f32::EPSILON);
            prop_assert!(next <= hi + f32::EPSILON);
            prop_assert!((0.0..=1.0).contains(&next));
        }

        /// Each step moves the weight toward the signal (or holds it): the
        /// distance to the signal never grows.
        #[test]
        fn ema_step_contracts_toward_signal(prev in 0.0f32..=1.0, signal in 0.0f32..=1.0) {
            let next = ema_step(prev, signal);
            prop_assert!((next - signal).abs() <= (prev - signal).abs() + f32::EPSILON);
        }

        /// A repeated identical signal is a fixed point — once the weight
        /// equals the signal, the step leaves it there.
        #[test]
        fn ema_step_fixed_point_at_equal(x in 0.0f32..=1.0) {
            prop_assert!((ema_step(x, x) - x).abs() <= f32::EPSILON);
        }

        /// The step is monotone in the signal: a stronger positive signal
        /// never produces a lower updated weight (same prior).
        #[test]
        fn ema_step_monotone_in_signal(prev in 0.0f32..=1.0, s1 in 0.0f32..=1.0, s2 in 0.0f32..=1.0) {
            let (lo, hi) = if s1 <= s2 { (s1, s2) } else { (s2, s1) };
            prop_assert!(ema_step(prev, hi) + f32::EPSILON >= ema_step(prev, lo));
        }

        /// Every interaction classifies to a signal in `[0, 1]` — the
        /// precondition the EMA blend relies on. Responded outranks
        /// no-response outranks delivery-failure.
        #[test]
        fn interaction_signal_is_an_ordered_valid_weight(ok in any::<bool>(), responded in any::<bool>()) {
            let sig = interaction(ok, responded).signal();
            prop_assert!((0.0..=1.0).contains(&sig));
            // A failed delivery is the weakest signal; a response the strongest.
            if !ok {
                prop_assert_eq!(sig, SIGNAL_DELIVERY_FAIL);
            } else if responded {
                prop_assert_eq!(sig, SIGNAL_RESPONDED);
            } else {
                prop_assert_eq!(sig, SIGNAL_NO_RESPONSE);
            }
        }
    }
}
