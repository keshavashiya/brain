//! HabitEngine — pattern detection and proactive behavior.

use chrono::{Datelike, NaiveTime, Timelike, Utc};
use chrono_tz::Tz;
use rusqlite::OptionalExtension;
use std::collections::{HashMap, HashSet};
use storage::SqlitePool;

use crate::{GangliaError, ProactiveMessage, TopicPattern};

/// Habit engine configuration (mirrors `core::ProactivityConfig`).
#[derive(Debug, Clone)]
pub struct HabitConfig {
    /// Maximum proactive messages per UTC calendar day.
    pub max_per_day: u32,
    /// Minimum minutes between consecutive proactive messages.
    pub min_interval_minutes: u32,
    /// Quiet-hours start (HH:MM, UTC).  No messages during this window.
    pub quiet_start: String,
    /// Quiet-hours end (HH:MM, UTC).
    pub quiet_end: String,
    /// User's local timezone (e.g. "America/New_York", "Europe/London").
    pub timezone: String,
    /// Minimum observations before a pattern is considered stable.
    pub min_occurrences: usize,
    /// How many days back to scan for patterns.
    pub lookback_days: u32,
}

impl Default for HabitConfig {
    fn default() -> Self {
        Self {
            max_per_day: 5,
            min_interval_minutes: 60,
            quiet_start: "22:00".to_string(),
            quiet_end: "08:00".to_string(),
            timezone: "UTC".to_string(),
            min_occurrences: 3,
            lookback_days: 30,
        }
    }
}

/// Detects behavioral patterns in episodic memory and emits proactive messages.
pub struct HabitEngine {
    db: SqlitePool,
    config: HabitConfig,
}

impl HabitEngine {
    /// Create a new habit engine backed by the given SQLite pool.
    pub fn new(db: SqlitePool, config: HabitConfig) -> Self {
        Self { db, config }
    }

    /// Create the `habit_state` key-value table (idempotent).
    pub fn ensure_tables(&self) -> Result<(), GangliaError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS habit_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );",
            )?;
            Ok(())
        })?;
        Ok(())
    }

    // ── Pattern detection ────────────────────────────────────────────────────

    /// Scan episodic memory for recurring (keyword, day-of-week, hour) patterns.
    ///
    /// Returns patterns ordered by occurrence count descending.
    pub fn detect_patterns(&self) -> Result<Vec<TopicPattern>, GangliaError> {
        let cutoff = Utc::now() - chrono::TimeDelta::days(self.config.lookback_days as i64);
        let cutoff_str = cutoff.to_rfc3339();

        let rows: Vec<(String, String, Option<String>)> = self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT content, timestamp, agent FROM episodes
                 WHERE timestamp >= ?1
                 ORDER BY timestamp ASC",
            )?;
            let rows = stmt
                .query_map([&cutoff_str], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(rows)
        })?;

        const STOPWORDS: &[&str] = &[
            "i", "me", "my", "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may",
            "might", "must", "can", "could", "to", "of", "in", "on", "at", "for", "with", "by",
            "from", "that", "this", "it", "its", "and", "or", "but", "not", "no", "about", "what",
            "how", "when", "where", "who", "which", "brain", "remember", "recall", "store", "fact",
            "help", "also", "just", "then", "than",
        ];

        // (keyword, day_of_week, hour) → (count, agents seen)
        type PatternKey = (String, u8, u8);
        type PatternValue = (usize, HashSet<Option<String>>);
        let mut counts: HashMap<PatternKey, PatternValue> = HashMap::new();

        for (content, timestamp, agent) in &rows {
            let dt = chrono::DateTime::parse_from_rfc3339(timestamp)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let dow = dt.weekday().num_days_from_monday() as u8;
            let hour = dt.hour() as u8;

            // Use unique keywords per episode to avoid duplicate inflation
            let mut seen: HashSet<String> = HashSet::new();
            for word in content.split_whitespace() {
                let kw = brain_core::normalize_keyword(word);
                if kw.len() >= 4 && !STOPWORDS.contains(&kw.as_str()) && seen.insert(kw.clone()) {
                    let entry = counts
                        .entry((kw, dow, hour))
                        .or_insert_with(|| (0, HashSet::new()));
                    entry.0 += 1;
                    entry.1.insert(agent.clone());
                }
            }
        }

        let mut patterns: Vec<TopicPattern> = counts
            .into_iter()
            .filter(|(_, (count, _))| *count >= self.config.min_occurrences)
            .map(|((topic, day_of_week, hour), (occurrences, agents))| {
                // If all occurrences come from a single known agent, attribute it
                let agent = if agents.len() == 1 {
                    agents.into_iter().next().flatten()
                } else {
                    None
                };
                TopicPattern {
                    topic,
                    day_of_week,
                    hour,
                    occurrences,
                    agent,
                }
            })
            .collect();

        patterns.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));
        Ok(patterns)
    }

    // ── Rate limiting ─────────────────────────────────────────────────────────

    /// Returns `true` if the current time in the configured timezone is within quiet hours.
    pub fn is_quiet_time(&self) -> bool {
        let tz: Tz = self.config.timezone.parse().unwrap_or_else(|_| {
            tracing::warn!(tz = %self.config.timezone, "Invalid timezone, falling back to UTC");
            chrono_tz::UTC
        });
        let now = Utc::now().with_timezone(&tz);
        let current = NaiveTime::from_hms_opt(now.hour(), now.minute(), 0).unwrap_or_default();

        let parse = |s: &str| NaiveTime::parse_from_str(s, "%H:%M").unwrap_or_default();
        let start = parse(&self.config.quiet_start);
        let end = parse(&self.config.quiet_end);

        if start <= end {
            current >= start && current < end
        } else {
            // Overnight window e.g. 22:00–08:00
            current >= start || current < end
        }
    }

    /// Returns the number of proactive messages sent today (UTC calendar day).
    fn daily_sent_count(&self) -> Result<u32, GangliaError> {
        let today = Utc::now().format("%Y-%m-%d").to_string();

        let stored_date: Option<String> = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT value FROM habit_state WHERE key = 'daily_sent_date'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| e.into())
        })?;

        if stored_date.as_deref() != Some(today.as_str()) {
            return Ok(0);
        }

        let count: u32 = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT value FROM habit_state WHERE key = 'daily_sent_count'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map(|v| v.and_then(|s| s.parse::<u32>().ok()).unwrap_or(0))
            .map_err(|e| e.into())
        })?;

        Ok(count)
    }

    /// Returns minutes elapsed since the last proactive message was sent.
    /// Returns `u32::MAX` if no message has ever been sent.
    fn minutes_since_last_sent(&self) -> Result<u32, GangliaError> {
        let ts: Option<String> = self.db.with_conn(|conn| {
            conn.query_row(
                "SELECT value FROM habit_state WHERE key = 'last_sent_at'",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| e.into())
        })?;

        match ts {
            None => Ok(u32::MAX),
            Some(s) => {
                let last = chrono::DateTime::parse_from_rfc3339(&s)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now() - chrono::TimeDelta::hours(25));
                let mins = (Utc::now() - last).num_minutes().max(0) as u32;
                Ok(mins)
            }
        }
    }

    /// Persist a proactive send event — increments daily counter and records timestamp.
    pub fn record_sent(&self) -> Result<(), GangliaError> {
        let today = Utc::now().format("%Y-%m-%d").to_string();
        let now = Utc::now().to_rfc3339();
        let new_count = self.daily_sent_count()? + 1;

        self.db.with_conn(|conn| {
            conn.execute(
                "INSERT INTO habit_state (key, value) VALUES ('daily_sent_date', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [&today],
            )?;
            conn.execute(
                "INSERT INTO habit_state (key, value) VALUES ('daily_sent_count', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [&new_count.to_string()],
            )?;
            conn.execute(
                "INSERT INTO habit_state (key, value) VALUES ('last_sent_at', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [&now],
            )?;
            Ok(())
        })?;
        Ok(())
    }

    /// Returns `true` if all rate-limit and quiet-hour gates are open.
    pub fn can_send_proactive(&self) -> Result<bool, GangliaError> {
        if self.is_quiet_time() {
            return Ok(false);
        }
        if self.daily_sent_count()? >= self.config.max_per_day {
            return Ok(false);
        }
        if self.minutes_since_last_sent()? < self.config.min_interval_minutes {
            return Ok(false);
        }
        Ok(true)
    }

    // ── Main entry point ──────────────────────────────────────────────────────

    /// Generate a proactive message if conditions are met.
    pub fn generate_proactive(&self) -> Result<Option<ProactiveMessage>, GangliaError> {
        if !self.can_send_proactive()? {
            return Ok(None);
        }

        let now = Utc::now();
        let current_dow = now.weekday().num_days_from_monday() as u8;
        let current_hour = now.hour() as u8;

        let patterns = self.detect_patterns()?;

        let matching = patterns
            .into_iter()
            .find(|p| p.day_of_week == current_dow && p.hour == current_hour);

        if let Some(pattern) = matching {
            self.record_sent()?;
            let content = if let Some(ref agent) = pattern.agent {
                format!(
                    "{} often works on \"{}\" around this time. \
                     Want to review or update anything?",
                    agent, pattern.topic
                )
            } else {
                format!(
                    "You usually work on \"{}\" around this time. \
                     Want to review or update anything?",
                    pattern.topic
                )
            };
            Ok(Some(ProactiveMessage {
                content,
                triggered_by: pattern.topic.clone(),
                created_at: now,
                agent: pattern.agent.clone(),
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> HabitEngine {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let engine = HabitEngine::new(pool, HabitConfig::default());
        engine.ensure_tables().unwrap();
        engine
    }

    #[test]
    fn test_ensure_tables_idempotent() {
        let engine = test_engine();
        engine.ensure_tables().unwrap();
    }

    #[test]
    fn test_no_patterns_when_empty() {
        let engine = test_engine();
        let patterns = engine.detect_patterns().unwrap();
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_can_send_when_no_history() {
        let engine = test_engine();
        let count = engine.daily_sent_count().unwrap();
        assert_eq!(count, 0);
        let mins = engine.minutes_since_last_sent().unwrap();
        assert_eq!(mins, u32::MAX);
    }

    #[test]
    fn test_record_sent_increments_count() {
        let engine = test_engine();
        assert_eq!(engine.daily_sent_count().unwrap(), 0);
        engine.record_sent().unwrap();
        assert_eq!(engine.daily_sent_count().unwrap(), 1);
        engine.record_sent().unwrap();
        assert_eq!(engine.daily_sent_count().unwrap(), 2);
    }

    #[test]
    fn test_record_sent_updates_last_sent() {
        let engine = test_engine();
        assert_eq!(engine.minutes_since_last_sent().unwrap(), u32::MAX);
        engine.record_sent().unwrap();
        assert!(engine.minutes_since_last_sent().unwrap() < 2);
    }

    #[test]
    fn test_quiet_time_overnight_range() {
        let engine = HabitEngine::new(
            storage::SqlitePool::open_memory().unwrap(),
            HabitConfig {
                quiet_start: "00:00".to_string(),
                quiet_end: "23:59".to_string(),
                ..Default::default()
            },
        );
        assert!(engine.is_quiet_time());
    }

    #[test]
    fn test_max_per_day_blocks_send() {
        let engine = HabitEngine::new(
            storage::SqlitePool::open_memory().unwrap(),
            HabitConfig {
                max_per_day: 1,
                quiet_start: "00:00".to_string(),
                quiet_end: "00:00".to_string(),
                min_interval_minutes: 0,
                ..Default::default()
            },
        );
        engine.ensure_tables().unwrap();
        engine.record_sent().unwrap();

        assert!(!engine.can_send_proactive().unwrap());
    }

    #[test]
    fn test_generate_proactive_returns_none_when_no_patterns() {
        let engine = HabitEngine::new(
            storage::SqlitePool::open_memory().unwrap(),
            HabitConfig {
                quiet_start: "00:00".to_string(),
                quiet_end: "00:00".to_string(),
                min_interval_minutes: 0,
                min_occurrences: 3,
                ..Default::default()
            },
        );
        engine.ensure_tables().unwrap();
        let result = engine.generate_proactive().unwrap();
        assert!(result.is_none());
    }
}
