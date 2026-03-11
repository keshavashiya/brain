//! Memory consolidation pipeline — "sleep-like" memory optimization.
//!
//! Runs periodically to:
//! 1. Prune low-importance episodes that have decayed past threshold
//! 2. Promote frequently reinforced episodes to semantic facts
//! 3. Update statistics for memory health monitoring

use crate::episodic::{EpisodicError, EpisodicStore};
use crate::search::forgetting_curve;
use thiserror::Error;

/// Errors from the consolidation pipeline.
#[derive(Debug, Error)]
pub enum ConsolidationError {
    #[error("Episodic error: {0}")]
    Episodic(#[from] EpisodicError),

    #[error("Storage error: {0}")]
    Storage(#[from] storage::sqlite::SqliteError),
}

/// Configuration for the consolidation pipeline.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum retention score to keep an episode (0.0–1.0).
    pub prune_threshold: f64,
    /// Decay rate for forgetting curve calculation.
    pub decay_rate: f64,
    /// Minimum reinforcement count to promote to semantic memory.
    pub promotion_threshold: i32,
    /// Maximum number of episodes to prune per run.
    pub max_prune_per_run: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            prune_threshold: 0.05,
            decay_rate: 0.01,
            promotion_threshold: 3,
            max_prune_per_run: 100,
        }
    }
}

/// Result of a consolidation run.
#[derive(Debug, Clone)]
pub struct ConsolidationReport {
    /// Number of episodes pruned.
    pub episodes_pruned: usize,
    /// Number of episodes promoted to semantic facts.
    pub episodes_promoted: usize,
    /// Total episodes remaining.
    pub episodes_remaining: i64,
    /// Concrete episodes eligible for semantic promotion.
    pub promotion_candidates: Vec<PromotionCandidate>,
}

/// Episode metadata needed for deterministic semantic promotion.
#[derive(Debug, Clone)]
pub struct PromotionCandidate {
    pub episode_id: String,
    pub namespace: String,
    pub content: String,
    pub importance: f64,
    pub reinforcement_count: i32,
}

/// The consolidation engine manages memory lifecycle.
pub struct Consolidator {
    config: ConsolidationConfig,
}

impl Consolidator {
    pub fn new(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ConsolidationConfig::default())
    }

    /// Run consolidation: prune decayed episodes and identify promotion candidates.
    ///
    /// Returns a report of what was done.
    pub fn consolidate(
        &self,
        episodic: &EpisodicStore,
    ) -> Result<ConsolidationReport, ConsolidationError> {
        let mut pruned = 0;
        let mut promotion_candidates = Vec::new();

        // Get all episodes sorted by importance (lowest first)
        let db = episodic.pool();
        let pool = episodic.pool();
        let candidates = db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT rowid, id, namespace, content, importance, decay_rate, reinforcement_count,
                        COALESCE(last_accessed, timestamp) as last_access_time
                 FROM episodes
                 ORDER BY importance ASC
                 LIMIT ?1",
            )?;

            let rows = stmt
                .query_map([self.config.max_prune_per_run as i64 * 2], |row| {
                    let raw_content: String = row.get(3)?;
                    Ok(ConsolidationCandidate {
                        row_id: row.get(0)?,
                        id: row.get(1)?,
                        namespace: row.get(2)?,
                        content: pool.decrypt_content(&raw_content),
                        importance: row.get(4)?,
                        decay_rate: row.get(5)?,
                        reinforcement_count: row.get(6)?,
                        last_accessed: row.get::<_, String>(7)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            Ok(rows)
        })?;

        let now = chrono::Utc::now();

        for candidate in &candidates {
            if pruned >= self.config.max_prune_per_run {
                break;
            }

            // Calculate hours since last access
            let hours = parse_hours_since(&candidate.last_accessed, &now);

            // Calculate retention
            let retention = forgetting_curve(candidate.importance, hours, candidate.decay_rate);

            if retention < self.config.prune_threshold {
                // Prune this episode
                db.with_conn(|conn| {
                    conn.execute("DELETE FROM episodes WHERE id = ?1", [&candidate.id])?;
                    conn.execute(
                        "DELETE FROM episodes_fts WHERE rowid = ?1",
                        [candidate.row_id],
                    )?;
                    Ok(())
                })?;
                pruned += 1;
            }

            // Check for promotion candidates
            if candidate.reinforcement_count >= self.config.promotion_threshold {
                promotion_candidates.push(PromotionCandidate {
                    episode_id: candidate.id.clone(),
                    namespace: candidate.namespace.clone(),
                    content: candidate.content.clone(),
                    importance: candidate.importance,
                    reinforcement_count: candidate.reinforcement_count,
                });
            }
        }

        let remaining = episodic.count().map_err(ConsolidationError::Episodic)?;

        Ok(ConsolidationReport {
            episodes_pruned: pruned,
            episodes_promoted: promotion_candidates.len(),
            episodes_remaining: remaining,
            promotion_candidates,
        })
    }
}

/// Internal candidate for consolidation evaluation.
#[derive(Debug)]
struct ConsolidationCandidate {
    row_id: i64,
    id: String,
    namespace: String,
    content: String,
    importance: f64,
    decay_rate: f64,
    reinforcement_count: i32,
    last_accessed: String,
}

/// Parse hours since a timestamp string (RFC3339 or SQLite datetime format).
fn parse_hours_since(timestamp_str: &str, now: &chrono::DateTime<chrono::Utc>) -> f64 {
    // Try RFC3339 first, then SQLite datetime format
    if let Ok(ts) = chrono::DateTime::parse_from_rfc3339(timestamp_str) {
        let duration = *now - ts.with_timezone(&chrono::Utc);
        return duration.num_seconds() as f64 / 3600.0;
    }
    if let Ok(ts) = chrono::NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S") {
        let utc_ts = ts.and_utc();
        let duration = *now - utc_ts;
        return duration.num_seconds() as f64 / 3600.0;
    }
    // Default to 24 hours if parsing fails
    24.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hours_since_rfc3339() {
        let now = chrono::Utc::now();
        let one_hour_ago = (now - chrono::Duration::hours(1)).to_rfc3339();
        let hours = parse_hours_since(&one_hour_ago, &now);
        assert!((hours - 1.0).abs() < 0.1, "Expected ~1.0 hour, got {hours}");
    }

    #[test]
    fn test_parse_hours_since_sqlite_format() {
        let now = chrono::Utc::now();
        let two_hours_ago = (now - chrono::Duration::hours(2))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let hours = parse_hours_since(&two_hours_ago, &now);
        assert!(
            (hours - 2.0).abs() < 0.1,
            "Expected ~2.0 hours, got {hours}"
        );
    }

    #[test]
    fn test_parse_hours_since_invalid() {
        let now = chrono::Utc::now();
        let hours = parse_hours_since("not-a-date", &now);
        assert_eq!(hours, 24.0);
    }

    #[test]
    fn test_consolidation_config_defaults() {
        let config = ConsolidationConfig::default();
        assert!((config.prune_threshold - 0.05).abs() < 1e-6);
        assert!((config.decay_rate - 0.01).abs() < 1e-6);
        assert_eq!(config.promotion_threshold, 3);
        assert_eq!(config.max_prune_per_run, 100);
    }

    #[test]
    fn test_consolidation_prune() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let store = EpisodicStore::new(db);

        let session_id = store.create_session("test").unwrap();

        // Store a low-importance episode
        store
            .store_episode(&session_id, "user", "trivial message", 0.01, None, None)
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);

        // Run consolidation — episode should be pruned (low importance)
        let config = ConsolidationConfig {
            prune_threshold: 0.5, // High threshold so our 0.01 episode gets pruned
            decay_rate: 1.0,      // Fast decay
            ..Default::default()
        };
        let consolidator = Consolidator::new(config);
        let report = consolidator.consolidate(&store).unwrap();

        assert!(
            report.episodes_pruned > 0,
            "Should have pruned the low-importance episode"
        );
    }

    #[test]
    fn test_consolidation_keep_important() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let store = EpisodicStore::new(db);

        let session_id = store.create_session("test").unwrap();

        // Store a high-importance episode
        store
            .store_episode(
                &session_id,
                "user",
                "critical: remember this forever",
                1.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(store.count().unwrap(), 1);

        // Run with default config — high importance + recent = high retention
        let consolidator = Consolidator::with_defaults();
        let report = consolidator.consolidate(&store).unwrap();

        assert_eq!(
            report.episodes_pruned, 0,
            "Should not prune high-importance recent episode"
        );
        assert_eq!(report.episodes_remaining, 1);
    }

    #[test]
    fn test_promotion_detection() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let store = EpisodicStore::new(db);

        let session_id = store.create_session("test").unwrap();
        store
            .store_episode(
                &session_id,
                "user",
                "I love Rust programming",
                0.8,
                Some("work"),
                None,
            )
            .unwrap();

        // Reinforce multiple times to cross promotion threshold
        let episodes = store.get_session_history(&session_id, 1).unwrap();
        let ep_id = &episodes[0].id;
        store.reinforce(ep_id).unwrap();
        store.reinforce(ep_id).unwrap();
        store.reinforce(ep_id).unwrap();

        let config = ConsolidationConfig {
            promotion_threshold: 3,
            prune_threshold: 0.0, // Don't prune anything
            ..Default::default()
        };
        let consolidator = Consolidator::new(config);
        let report = consolidator.consolidate(&store).unwrap();

        assert!(
            report.episodes_promoted > 0,
            "Reinforced episode should be a promotion candidate"
        );
        assert_eq!(report.promotion_candidates.len(), 1);
        assert_eq!(report.promotion_candidates[0].namespace, "work");
        assert!(report.promotion_candidates[0].content.contains("Rust"));
    }
}
