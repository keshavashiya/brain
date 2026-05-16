//! Rolling-window loop detector for tool dispatch. Hashes each
//! `(tool_id, canonical-args)` shape into a per-principal window;
//! when a hash repeats more than `threshold` times inside the window
//! we return a `LoopDetected` error and emit a
//! `BrainEvent::Error { source: "loop_detector", … }` so the Live tab
//! and `brain tail` surface the runaway.
//!
//! Args are canonicalized (sorted object keys, recursive) before
//! hashing so `{"a":1,"b":2}` and `{"b":2,"a":1}` collapse to the
//! same shape — required because LLM tool-call args round-trip
//! through serde and key order isn't stable.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use chrono::Utc;
use observe::{BrainEvent, Observer};
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::warn;
use uuid::Uuid;

/// Tuning knobs for [`LoopDetector`].
#[derive(Debug, Clone)]
pub struct LoopDetectorConfig {
    /// Rolling window size — how many recent calls are kept per
    /// principal. Older entries are dropped FIFO.
    pub window: usize,
    /// A hash repeating more than this many times inside the window
    /// trips the detector. Threshold = 4 means the 5th identical call
    /// fires.
    pub threshold: u32,
}

impl Default for LoopDetectorConfig {
    fn default() -> Self {
        Self {
            window: 16,
            threshold: 4,
        }
    }
}

/// Returned by [`LoopDetector::check`] when the rolling window shows
/// the same `(tool_id, args)` shape repeating beyond threshold.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LoopDetectorError {
    #[error("tool {tool_id} repeated {count} times in window of {window}")]
    LoopDetected {
        tool_id: String,
        count: u32,
        window: usize,
    },
}

/// Per-principal rolling-hash loop detector.
pub struct LoopDetector {
    config: LoopDetectorConfig,
    observer: Option<Arc<dyn Observer>>,
    // principal → window of recent (hash, tool_id) pairs. Storing
    // tool_id alongside the hash keeps observer messages readable
    // without a separate reverse map.
    state: Mutex<HashMap<String, VecDeque<(u64, String)>>>,
}

impl LoopDetector {
    pub fn new(config: LoopDetectorConfig) -> Self {
        Self {
            config,
            observer: None,
            state: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    pub fn config(&self) -> &LoopDetectorConfig {
        &self.config
    }

    /// Record one tool call, scoped by `principal`. Returns
    /// `Ok(())` while inside threshold, `Err(LoopDetected)` once the
    /// same hash appears more than `threshold` times in the window.
    ///
    /// `principal` is the audit-log principal identifier (or any
    /// stable string per agent); use `""` for unauthenticated callers
    /// — the per-principal scope still isolates anonymous traffic
    /// from named agents.
    pub async fn check(
        &self,
        principal: &str,
        tool_id: &str,
        args: &serde_json::Value,
    ) -> Result<(), LoopDetectorError> {
        let hash = hash_call(tool_id, args);
        let count = {
            let mut state = self.state.lock().await;
            let window = state.entry(principal.to_string()).or_default();
            window.push_back((hash, tool_id.to_string()));
            while window.len() > self.config.window {
                window.pop_front();
            }
            window.iter().filter(|(h, _)| *h == hash).count() as u32
        };
        if count > self.config.threshold {
            warn!(
                principal = principal,
                tool_id = tool_id,
                count = count,
                window = self.config.window,
                "loop detector tripped",
            );
            if let Some(observer) = &self.observer {
                let _ = observer
                    .publish(BrainEvent::Error {
                        id: Uuid::new_v4(),
                        source: "loop_detector".to_string(),
                        message: format!(
                            "tool {tool_id} repeated {count} times in window {len}",
                            len = self.config.window,
                        ),
                        ts: Utc::now(),
                    })
                    .await;
            }
            return Err(LoopDetectorError::LoopDetected {
                tool_id: tool_id.to_string(),
                count,
                window: self.config.window,
            });
        }
        Ok(())
    }

    /// Clear the rolling state for one principal. Useful after a
    /// human confirms the loop was intentional.
    pub async fn reset(&self, principal: &str) {
        self.state.lock().await.remove(principal);
    }

    /// Test/inspection helper: how many entries are currently held
    /// for `principal`.
    pub async fn window_len(&self, principal: &str) -> usize {
        self.state
            .lock()
            .await
            .get(principal)
            .map(|q| q.len())
            .unwrap_or(0)
    }
}

fn hash_call(tool_id: &str, args: &serde_json::Value) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool_id.hash(&mut hasher);
    canonical_json(args).hash(&mut hasher);
    hasher.finish()
}

/// Recursive canonicalizer — objects with sorted keys; arrays
/// preserve order. Result is a deterministic string suitable for
/// hashing.
fn canonical_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let body: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_else(|_| String::from("\"\"")),
                        canonical_json(v)
                    )
                })
                .collect();
            format!("{{{}}}", body.join(","))
        }
        serde_json::Value::Array(arr) => {
            let body: Vec<String> = arr.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
pub(crate) fn canonical_json_for_test(v: &serde_json::Value) -> String {
    canonical_json(v)
}
