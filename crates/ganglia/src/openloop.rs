//! Open-loop detection — finds unresolved commitments in episodic memory.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use storage::SqlitePool;

use crate::{GangliaError, ProactiveMessage};

/// Configuration for open-loop (unresolved commitment) detection.
#[derive(Debug, Clone)]
pub struct OpenLoopConfig {
    /// How many hours back to scan for commitments.
    pub scan_window_hours: u32,
    /// Hours after a commitment before it's flagged as unresolved.
    pub resolution_window_hours: u32,
    /// Maximum reminders to generate per check cycle.
    pub max_reminders: usize,
}

impl Default for OpenLoopConfig {
    fn default() -> Self {
        Self {
            scan_window_hours: 72,
            resolution_window_hours: 24,
            max_reminders: 3,
        }
    }
}

/// An unresolved commitment found in episodic memory.
#[derive(Debug, Clone)]
pub struct OpenLoop {
    pub commitment: String,
    pub topic: String,
    pub committed_at: String,
    pub agent: Option<String>,
}

/// Detects unresolved commitments ("open loops") in episodic memory.
///
/// Scans for commitment phrases like "I need to", "remind me to", etc.
/// and checks whether a subsequent episode references the same topic.
/// If no resolution is found within `resolution_window_hours`, a
/// reminder is surfaced.
///
/// When constructed with `with_llm()`, `detect_open_loops_async()` uses
/// LLM-driven analysis with automatic fallback to keyword heuristics.
pub struct OpenLoopDetector {
    db: SqlitePool,
    config: OpenLoopConfig,
    llm: Option<Arc<dyn cortex::LlmProvider>>,
}

/// Phrases that signal a user commitment or intention.
const COMMITMENT_PHRASES: &[&str] = &[
    "i'll",
    "i will",
    "i need to",
    "i should",
    "i must",
    "remind me to",
    "don't forget",
    "need to remember",
    "going to",
    "plan to",
    "want to",
    "have to",
    "todo",
    "to-do",
];

/// Words that signal a commitment has been resolved.
const RESOLUTION_MARKERS: &[&str] = &[
    "done",
    "finished",
    "completed",
    "did it",
    "checked off",
    "resolved",
    "took care",
    "handled",
    "sorted",
    "already",
];

/// A fetched episode row: (id, content, timestamp, agent).
type EpisodeRow = (String, String, String, Option<String>);

impl OpenLoopDetector {
    pub fn new(db: SqlitePool, config: OpenLoopConfig) -> Self {
        Self {
            db,
            config,
            llm: None,
        }
    }

    /// Create an OpenLoopDetector backed by the given LLM provider.
    pub fn with_llm(
        db: SqlitePool,
        config: OpenLoopConfig,
        llm: Arc<dyn cortex::LlmProvider>,
    ) -> Self {
        Self {
            db,
            config,
            llm: Some(llm),
        }
    }

    /// Fetch user episodes within the scan window.
    fn fetch_episodes(&self) -> Result<Vec<EpisodeRow>, GangliaError> {
        let scan_cutoff =
            Utc::now() - chrono::TimeDelta::hours(self.config.scan_window_hours as i64);
        let scan_str = scan_cutoff.to_rfc3339();

        self.db
            .with_conn(|conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, content, timestamp, agent FROM episodes
                 WHERE timestamp >= ?1 AND role = 'user'
                 ORDER BY timestamp ASC",
                )?;
                let rows = stmt
                    .query_map([&scan_str], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, Option<String>>(3)?,
                        ))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .map_err(Into::into)
    }

    /// Scan episodic memory for unresolved commitments (keyword heuristic).
    pub fn detect_open_loops(&self) -> Result<Vec<OpenLoop>, GangliaError> {
        let now = Utc::now();
        let resolution_cutoff =
            now - chrono::TimeDelta::hours(self.config.resolution_window_hours as i64);

        let episodes = self.fetch_episodes()?;

        let mut open_loops = Vec::new();

        for (id, content, timestamp, agent) in &episodes {
            let lower = content.to_lowercase();

            // Extract topic from commitment phrase.
            // Search and slice from the same string (`lower`) to avoid byte offset
            // mismatch when lowercasing changes string length (e.g. ß → ss).
            let topic = COMMITMENT_PHRASES.iter().find_map(|phrase| {
                lower.find(phrase).map(|pos| {
                    let after = &lower[pos + phrase.len()..];
                    let trimmed = after
                        .trim()
                        .trim_start_matches(|c: char| !c.is_alphanumeric());
                    trimmed
                        .split_whitespace()
                        .take(8)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
            });

            let topic = match topic {
                Some(t) if t.len() >= 3 => t,
                _ => continue,
            };

            // Only flag commitments that are old enough
            let committed_dt = DateTime::parse_from_rfc3339(timestamp)
                .map(|d| d.with_timezone(&Utc))
                .unwrap_or(now);
            if committed_dt > resolution_cutoff {
                continue;
            }

            // Extract meaningful keywords from the topic for matching
            let topic_words: Vec<String> = topic
                .split_whitespace()
                .map(brain::normalize_keyword)
                .filter(|w| w.len() >= 4)
                .collect();
            if topic_words.is_empty() {
                continue;
            }

            // Check if any later episode resolves this commitment
            let resolved = episodes.iter().any(|(eid, econtent, ets, _)| {
                if eid == id {
                    return false;
                }
                let edt = DateTime::parse_from_rfc3339(ets)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or(now);
                if edt <= committed_dt {
                    return false;
                }
                let elower = econtent.to_lowercase();

                let has_topic_ref = topic_words
                    .iter()
                    .filter(|w| elower.contains(w.as_str()))
                    .count()
                    >= topic_words.len().clamp(1, 2);

                let has_resolution_marker = RESOLUTION_MARKERS.iter().any(|m| elower.contains(m));

                (has_topic_ref && has_resolution_marker)
                    || topic_words
                        .iter()
                        .filter(|w| elower.contains(w.as_str()))
                        .count()
                        >= topic_words.len().max(2)
            });

            if !resolved {
                open_loops.push(OpenLoop {
                    commitment: content.clone(),
                    topic: topic.clone(),
                    committed_at: timestamp.clone(),
                    agent: agent.clone(),
                });
            }
        }

        open_loops.truncate(self.config.max_reminders);
        Ok(open_loops)
    }

    /// Detect open loops using LLM when available, keyword fallback otherwise.
    pub async fn detect_open_loops_async(&self) -> Result<Vec<OpenLoop>, GangliaError> {
        if self.llm.is_none() {
            return self.detect_open_loops();
        }

        let episodes = self.fetch_episodes()?;
        if episodes.is_empty() {
            return Ok(Vec::new());
        }

        let llm = match self.llm.as_ref() {
            Some(l) => l,
            None => return self.detect_open_loops(),
        };
        let timeout = tokio::time::Duration::from_millis(2000);
        match tokio::time::timeout(timeout, self.detect_with_llm(llm, &episodes)).await {
            Ok(Ok(loops)) => Ok(loops),
            Ok(Err(e)) => {
                tracing::debug!("LLM open-loop detection failed: {e}, falling back to keywords");
                self.detect_open_loops()
            }
            Err(_) => {
                tracing::debug!("LLM open-loop detection timed out, falling back to keywords");
                self.detect_open_loops()
            }
        }
    }

    /// Ask the LLM to identify unresolved commitments.
    async fn detect_with_llm(
        &self,
        llm: &Arc<dyn cortex::LlmProvider>,
        episodes: &[EpisodeRow],
    ) -> Result<Vec<OpenLoop>, cortex::LlmError> {
        let resolution_cutoff =
            Utc::now() - chrono::TimeDelta::hours(self.config.resolution_window_hours as i64);
        let cutoff_str = resolution_cutoff.to_rfc3339();

        let mut message_lines = String::new();
        for (i, (_id, content, ts, _agent)) in episodes.iter().enumerate() {
            message_lines.push_str(&format!("[{i}] ({ts}) {content}\n"));
        }

        let prompt = format!(
            "Analyze these user messages for unresolved commitments.\n\
             A commitment is when the user says they will/need/should/plan to do something.\n\
             A commitment is resolved if a later message indicates it was done/finished/completed.\n\
             Only flag commitments older than: {cutoff_str}\n\
             Return ONLY JSON array: [{{\"index\":N,\"topic\":\"brief description\",\"resolved\":false}}]\n\
             Return [] if none found.\n\
             Messages:\n{message_lines}"
        );

        let messages = vec![cortex::Message::user(prompt)];

        let response = llm.generate(&messages).await?;
        let parsed =
            parse_commitment_response(&response.content, episodes, self.config.max_reminders)?;
        Ok(parsed)
    }

    /// Generate proactive reminder messages for unresolved commitments (keyword heuristic).
    pub fn generate_reminders(&self) -> Result<Vec<ProactiveMessage>, GangliaError> {
        let loops = self.detect_open_loops()?;
        Ok(format_reminders(loops))
    }

    /// Generate proactive reminder messages using LLM when available, keyword fallback otherwise.
    pub async fn generate_reminders_async(&self) -> Result<Vec<ProactiveMessage>, GangliaError> {
        let loops = self.detect_open_loops_async().await?;
        Ok(format_reminders(loops))
    }
}

/// Format open loops into proactive reminder messages.
fn format_reminders(loops: Vec<OpenLoop>) -> Vec<ProactiveMessage> {
    loops
        .into_iter()
        .map(|ol| {
            let content = if let Some(ref agent) = ol.agent {
                format!(
                    "Open loop from {}: you mentioned \"{}\" — still pending. Want to follow up?",
                    agent, ol.topic
                )
            } else {
                format!(
                    "Open loop: you mentioned \"{}\" — still pending. Want to follow up?",
                    ol.topic
                )
            };
            ProactiveMessage {
                content,
                triggered_by: format!("open_loop:{}", ol.topic),
                created_at: Utc::now(),
                agent: ol.agent,
            }
        })
        .collect()
}

/// Parse LLM response into OpenLoop entries. Tries direct JSON, then finds `[...]`.
fn parse_commitment_response(
    raw: &str,
    episodes: &[EpisodeRow],
    max_reminders: usize,
) -> Result<Vec<OpenLoop>, cortex::LlmError> {
    #[derive(Deserialize)]
    struct CommitmentEntry {
        index: usize,
        topic: String,
        #[serde(default)]
        resolved: bool,
    }

    let trimmed = raw.trim();

    let entries: Vec<CommitmentEntry> = if let Ok(parsed) =
        serde_json::from_str::<Vec<CommitmentEntry>>(trimmed)
    {
        parsed
    } else if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            serde_json::from_str::<Vec<CommitmentEntry>>(&trimmed[start..=end]).map_err(|e| {
                cortex::LlmError::InvalidFormat(format!("Could not parse commitment JSON: {e}"))
            })?
        } else {
            return Err(cortex::LlmError::InvalidFormat(
                "No closing bracket in commitment response".to_string(),
            ));
        }
    } else {
        return Err(cortex::LlmError::InvalidFormat(
            "No JSON array in commitment response".to_string(),
        ));
    };

    let mut loops: Vec<OpenLoop> = entries
        .into_iter()
        .filter(|e| !e.resolved && e.index < episodes.len())
        .map(|e| {
            let (_, content, ts, agent) = &episodes[e.index];
            OpenLoop {
                commitment: content.clone(),
                topic: e.topic,
                committed_at: ts.clone(),
                agent: agent.clone(),
            }
        })
        .collect();

    loops.truncate(max_reminders);
    Ok(loops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_detector() -> (OpenLoopDetector, SqlitePool) {
        let pool = storage::SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES ('test-session', 'test')",
                [],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
        let config = OpenLoopConfig {
            scan_window_hours: 72,
            resolution_window_hours: 1,
            max_reminders: 5,
        };
        let detector = OpenLoopDetector::new(pool.clone(), config);
        (detector, pool)
    }

    fn insert_episode(pool: &SqlitePool, id: &str, content: &str, hours_ago: i64) {
        let ts = (Utc::now() - chrono::TimeDelta::hours(hours_ago)).to_rfc3339();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, timestamp)
                 VALUES (?1, 'test-session', 'user', ?2, ?3)",
                rusqlite::params![id, content, ts],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_open_loop_no_episodes() {
        let (detector, _) = test_detector();
        let loops = detector.detect_open_loops().unwrap();
        assert!(loops.is_empty());
    }

    #[test]
    fn test_open_loop_unresolved_commitment() {
        let (detector, pool) = test_detector();
        insert_episode(
            &pool,
            "e1",
            "I need to update the documentation for the API",
            2,
        );
        let loops = detector.detect_open_loops().unwrap();
        assert_eq!(loops.len(), 1);
        assert!(loops[0].topic.contains("update"));
        assert!(loops[0].topic.contains("documentation"));
    }

    #[test]
    fn test_open_loop_resolved_commitment() {
        let (detector, pool) = test_detector();
        insert_episode(
            &pool,
            "e1",
            "I need to update the documentation for the API",
            3,
        );
        insert_episode(
            &pool,
            "e2",
            "I finished the documentation update for the API",
            0,
        );
        let loops = detector.detect_open_loops().unwrap();
        assert!(loops.is_empty(), "should be resolved, but got: {:?}", loops);
    }

    #[test]
    fn test_open_loop_too_recent_not_flagged() {
        let (detector, pool) = test_detector();
        let ts = (Utc::now() - chrono::TimeDelta::minutes(30)).to_rfc3339();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, timestamp)
                 VALUES ('e1', 'test-session', 'user', 'I need to review the pull request', ?1)",
                [&ts],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
        let loops = detector.detect_open_loops().unwrap();
        assert!(loops.is_empty(), "recent commitment should not be flagged");
    }

    #[test]
    fn test_open_loop_generate_reminders() {
        let (detector, pool) = test_detector();
        insert_episode(
            &pool,
            "e1",
            "I should refactor the authentication module",
            5,
        );
        let reminders = detector.generate_reminders().unwrap();
        assert_eq!(reminders.len(), 1);
        assert!(reminders[0].content.contains("Open loop"));
        assert!(reminders[0].content.contains("refactor"));
        assert!(reminders[0].triggered_by.starts_with("open_loop:"));
    }

    #[test]
    fn test_open_loop_max_reminders_cap() {
        let (detector, pool) = test_detector();
        for i in 0..8 {
            insert_episode(
                &pool,
                &format!("e{i}"),
                &format!("I need to handle task_{i:04} in the project"),
                10 + i as i64,
            );
        }
        let loops = detector.detect_open_loops().unwrap();
        assert!(
            loops.len() <= 5,
            "should cap at max_reminders, got {}",
            loops.len()
        );
    }

    #[test]
    fn test_open_loop_assistant_messages_ignored() {
        let (detector, pool) = test_detector();
        let ts = (Utc::now() - chrono::TimeDelta::hours(5)).to_rfc3339();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, timestamp)
                 VALUES ('e1', 'test-session', 'assistant', 'I will help you with that task', ?1)",
                [&ts],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
        let loops = detector.detect_open_loops().unwrap();
        assert!(loops.is_empty(), "assistant messages should be ignored");
    }

    #[test]
    fn test_open_loop_with_agent_attribution() {
        let (detector, pool) = test_detector();
        let ts = (Utc::now() - chrono::TimeDelta::hours(5)).to_rfc3339();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, timestamp, agent)
                 VALUES ('e1', 'test-session', 'user', 'I need to deploy the staging server', ?1, 'devops-agent')",
                [&ts],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();

        let loops = detector.detect_open_loops().unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].agent.as_deref(), Some("devops-agent"));

        let reminders = detector.generate_reminders().unwrap();
        assert!(reminders[0].content.contains("devops-agent"));
    }

    #[test]
    fn test_open_loop_non_ascii_no_panic() {
        let (detector, pool) = test_detector();
        let ts = (Utc::now() - chrono::TimeDelta::hours(5)).to_rfc3339();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, timestamp)
                 VALUES ('e1', 'test-session', 'user', 'I need to update the Straße config', ?1)",
                [&ts],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();
        let loops = detector.detect_open_loops().unwrap();
        assert_eq!(loops.len(), 1);
    }

    // ── JSON parser tests ───────────────────────────────────────────────────

    fn sample_episodes() -> Vec<EpisodeRow> {
        vec![
            (
                "e0".into(),
                "I need to update the API docs".into(),
                "2024-01-01T10:00:00+00:00".into(),
                None,
            ),
            (
                "e1".into(),
                "The docs are done now".into(),
                "2024-01-02T15:00:00+00:00".into(),
                None,
            ),
            (
                "e2".into(),
                "I should fix the login bug".into(),
                "2024-01-03T09:00:00+00:00".into(),
                Some("dev-agent".into()),
            ),
        ]
    }

    #[test]
    fn test_parse_commitment_clean_json() {
        let episodes = sample_episodes();
        let json = r#"[{"index":2,"topic":"fix login bug","resolved":false}]"#;
        let loops = parse_commitment_response(json, &episodes, 5).unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].topic, "fix login bug");
        assert_eq!(loops[0].agent.as_deref(), Some("dev-agent"));
    }

    #[test]
    fn test_parse_commitment_embedded_json() {
        let episodes = sample_episodes();
        let raw = r#"Here are the results: [{"index":0,"topic":"update API docs","resolved":false}] done"#;
        let loops = parse_commitment_response(raw, &episodes, 5).unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].topic, "update API docs");
    }

    #[test]
    fn test_parse_commitment_resolved_filtered() {
        let episodes = sample_episodes();
        let json = r#"[{"index":0,"topic":"update docs","resolved":true},{"index":2,"topic":"fix bug","resolved":false}]"#;
        let loops = parse_commitment_response(json, &episodes, 5).unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].topic, "fix bug");
    }

    #[test]
    fn test_parse_commitment_empty_array() {
        let episodes = sample_episodes();
        let loops = parse_commitment_response("[]", &episodes, 5).unwrap();
        assert!(loops.is_empty());
    }

    #[test]
    fn test_parse_commitment_invalid() {
        let episodes = sample_episodes();
        assert!(parse_commitment_response("no json here", &episodes, 5).is_err());
    }

    #[test]
    fn test_parse_commitment_out_of_bounds_index() {
        let episodes = sample_episodes();
        let json = r#"[{"index":99,"topic":"nonexistent","resolved":false}]"#;
        let loops = parse_commitment_response(json, &episodes, 5).unwrap();
        assert!(loops.is_empty());
    }

    // ── Async tests ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_detect_async_no_llm_equals_sync() {
        let (detector, _) = test_detector();
        let sync_result = detector.detect_open_loops().unwrap();
        let async_result = detector.detect_open_loops_async().await.unwrap();
        assert_eq!(sync_result.len(), async_result.len());
    }

    // ── Mock LLM tests ─────────────────────────────────────────────────────

    struct MockLlm {
        response: String,
    }

    #[async_trait::async_trait]
    impl cortex::LlmProvider for MockLlm {
        async fn generate(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<cortex::Response, cortex::LlmError> {
            Ok(cortex::Response::text(self.response.clone(), None))
        }

        async fn generate_stream(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = Result<cortex::ResponseChunk, cortex::LlmError>>
                        + Send,
                >,
            >,
            cortex::LlmError,
        > {
            Err(cortex::LlmError::ProviderUnavailable("mock".to_string()))
        }

        async fn health_check(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "mock"
        }

        fn model(&self) -> &str {
            "mock"
        }

        async fn list_models(&self) -> Result<Vec<String>, cortex::LlmError> {
            Ok(vec!["mock".to_string()])
        }
    }

    #[tokio::test]
    async fn test_detect_async_with_mock_llm() {
        let pool = storage::SqlitePool::open_memory().unwrap();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES ('test-session', 'test')",
                [],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();

        let ts = (Utc::now() - chrono::TimeDelta::hours(5)).to_rfc3339();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO episodes (id, session_id, role, content, timestamp)
                 VALUES ('e1', 'test-session', 'user', 'I need to deploy the staging fix', ?1)",
                [&ts],
            )
            .unwrap();
            Ok(())
        })
        .unwrap();

        let mock = Arc::new(MockLlm {
            response: r#"[{"index":0,"topic":"deploy staging fix","resolved":false}]"#.to_string(),
        });

        let detector = OpenLoopDetector::with_llm(
            pool,
            OpenLoopConfig {
                scan_window_hours: 72,
                resolution_window_hours: 1,
                max_reminders: 5,
            },
            mock,
        );

        let loops = detector.detect_open_loops_async().await.unwrap();
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].topic, "deploy staging fix");
    }

    #[tokio::test]
    async fn test_detect_async_llm_bad_json_falls_back() {
        let (_, pool) = test_detector();
        insert_episode(
            &pool,
            "e1",
            "I need to update the documentation for the API",
            2,
        );

        let mock = Arc::new(MockLlm {
            response: "Sorry, I can't understand the request".to_string(),
        });

        let detector = OpenLoopDetector::with_llm(
            pool,
            OpenLoopConfig {
                scan_window_hours: 72,
                resolution_window_hours: 1,
                max_reminders: 5,
            },
            mock,
        );

        let loops = detector.detect_open_loops_async().await.unwrap();
        assert_eq!(loops.len(), 1);
        assert!(loops[0].topic.contains("update"));
    }

    struct SlowMockLlm;

    #[async_trait::async_trait]
    impl cortex::LlmProvider for SlowMockLlm {
        async fn generate(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<cortex::Response, cortex::LlmError> {
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
            Ok(cortex::Response::text("[]", None))
        }

        async fn generate_stream(
            &self,
            _messages: &[cortex::Message],
        ) -> Result<
            std::pin::Pin<
                Box<
                    dyn futures::Stream<Item = Result<cortex::ResponseChunk, cortex::LlmError>>
                        + Send,
                >,
            >,
            cortex::LlmError,
        > {
            Err(cortex::LlmError::ProviderUnavailable("mock".to_string()))
        }

        async fn health_check(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "slow-mock"
        }
        fn model(&self) -> &str {
            "slow-mock"
        }
        async fn list_models(&self) -> Result<Vec<String>, cortex::LlmError> {
            Ok(vec!["slow-mock".to_string()])
        }
    }

    #[tokio::test]
    async fn test_detect_async_timeout_falls_back() {
        let (_, pool) = test_detector();
        insert_episode(
            &pool,
            "e1",
            "I need to update the documentation for the API",
            2,
        );

        let detector = OpenLoopDetector::with_llm(
            pool,
            OpenLoopConfig {
                scan_window_hours: 72,
                resolution_window_hours: 1,
                max_reminders: 5,
            },
            Arc::new(SlowMockLlm),
        );

        let loops = detector.detect_open_loops_async().await.unwrap();
        assert_eq!(loops.len(), 1);
        assert!(loops[0].topic.contains("update"));
    }

    #[tokio::test]
    async fn test_generate_reminders_async_formats_correctly() {
        let (_, pool) = test_detector();
        insert_episode(
            &pool,
            "e1",
            "I should refactor the authentication module",
            5,
        );

        let mock = Arc::new(MockLlm {
            response: r#"[{"index":0,"topic":"refactor auth module","resolved":false}]"#
                .to_string(),
        });

        let detector = OpenLoopDetector::with_llm(
            pool,
            OpenLoopConfig {
                scan_window_hours: 72,
                resolution_window_hours: 1,
                max_reminders: 5,
            },
            mock,
        );

        let reminders = detector.generate_reminders_async().await.unwrap();
        assert_eq!(reminders.len(), 1);
        assert!(reminders[0].content.contains("Open loop"));
        assert!(reminders[0].content.contains("refactor auth module"));
        assert!(reminders[0].triggered_by.starts_with("open_loop:"));
    }
}
