//! Filesystem reflex — watches a configured set of paths via the
//! `notify` crate and emits one [`ReflexEvent`] per debounced
//! change.
//!
//! Debouncing is delegated to `notify-debouncer-full`, which
//! coalesces bursts (typical "save in editor" fires 3+ raw events)
//! and emits one logical event per `(path, kind)` after the window.
//!
//! Trigger format: `"fs:<absolute-path>"` — stable across runs so
//! audit can correlate firings to the watched location.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt;
use notify::{EventKind, RecursiveMode};
use notify_debouncer_full::new_debouncer;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::{ReflexError, ReflexEvent, ReflexSource, ReflexStream};

/// Tuning for [`FsReflex`].
#[derive(Debug, Clone)]
pub struct FsReflexConfig {
    /// Paths to watch. Each entry can be a file or directory; if it's
    /// a directory and [`Self::recursive`] is true, every descendant
    /// is watched too.
    pub paths: Vec<PathBuf>,
    /// Watch directories recursively (no effect on file paths).
    pub recursive: bool,
    /// Coalesce raw events for the same `(path, kind)` within this
    /// window into one emitted event. Default 200ms.
    pub debounce: Duration,
}

impl FsReflexConfig {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            recursive: false,
            debounce: Duration::from_millis(200),
        }
    }

    pub fn recursive(mut self, r: bool) -> Self {
        self.recursive = r;
        self
    }

    pub fn debounce(mut self, d: Duration) -> Self {
        self.debounce = d;
        self
    }
}

/// One filesystem change after debouncing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FsChange {
    /// Absolute path that changed, as reported by `notify`.
    pub path: String,
    /// One of `created` / `modified` / `removed` / `other`. Lossy on
    /// purpose — platform-specific subkinds aren't useful to the
    /// downstream signal pipeline and don't round-trip cleanly
    /// across Linux/macOS/Windows backends.
    pub kind: String,
}

fn classify_kind(k: &EventKind) -> &'static str {
    match k {
        EventKind::Create(_) => "created",
        EventKind::Modify(_) => "modified",
        EventKind::Remove(_) => "removed",
        EventKind::Access(_) | EventKind::Any | EventKind::Other => "other",
    }
}

/// Notify-backed reflex source.
pub struct FsReflex {
    name: String,
    config: FsReflexConfig,
}

impl FsReflex {
    pub fn new(name: impl Into<String>, config: FsReflexConfig) -> Self {
        Self {
            name: name.into(),
            config,
        }
    }

    pub fn config(&self) -> &FsReflexConfig {
        &self.config
    }
}

#[async_trait]
impl ReflexSource for FsReflex {
    fn name(&self) -> &str {
        &self.name
    }

    async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError> {
        // Channel from the OS-watcher callback into the async world.
        // Bound is loose — debouncer batches handle bursts.
        let (out_tx, out_rx) = mpsc::channel::<ReflexEvent>(64);
        // Capture in std::sync::mpsc-style channel that notify-debouncer-full
        // uses as its callback sink. We pump from there into our tokio mpsc.
        let debounce = self.config.debounce;
        let paths = self.config.paths.clone();
        let recursive = self.config.recursive;

        let (raw_tx, raw_rx) = std::sync::mpsc::channel();
        // Construct the debouncer on a blocking thread because
        // `notify` opens platform handles synchronously and we don't
        // want to occupy the reactor while it boots.
        let mut debouncer = new_debouncer(debounce, None, move |res| {
            // The watcher thread calls back into this closure. We
            // forward the raw `Result<Vec<DebouncedEvent>, _>` to
            // the pump task; that task converts to `ReflexEvent`s
            // and pushes through the async mpsc.
            let _ = raw_tx.send(res);
        })
        .map_err(|e| ReflexError::Backend(format!("debouncer init: {e}")))?;

        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        for p in &paths {
            debouncer
                .watch(p, mode)
                .map_err(|e| ReflexError::Backend(format!("watch {p:?}: {e}")))?;
        }

        // Pump task — owns the debouncer (drop = stop watching) and
        // forwards each debounced batch into the async stream. We
        // poll the std::sync::mpsc receiver with a short timeout so
        // we can exit promptly when the subscriber drops `out_rx`
        // (otherwise the blocking thread sits in `recv()` forever
        // and prevents cargo's test process from exiting).
        tokio::task::spawn_blocking(move || {
            let _debouncer = debouncer; // keep alive for the loop
            loop {
                if out_tx.is_closed() {
                    return;
                }
                match raw_rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(Ok(events)) => {
                        for ev in events {
                            for path in &ev.event.paths {
                                let change = FsChange {
                                    path: path.display().to_string(),
                                    kind: classify_kind(&ev.event.kind).to_string(),
                                };
                                let trigger = format!("fs:{}", change.path);
                                let payload = serde_json::to_value(&change)
                                    .unwrap_or(serde_json::Value::Null);
                                let evt = ReflexEvent::new(trigger, payload);
                                if out_tx.blocking_send(evt).is_err() {
                                    return; // subscriber dropped
                                }
                            }
                        }
                    }
                    Ok(Err(errs)) => {
                        for e in errs {
                            warn!(error = ?e, "fs reflex backend error");
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
        });

        Ok(ReceiverStream::new(out_rx).boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::Duration;
    use tempfile::TempDir;

    /// Subscribe, drop a file edit on the watched path, expect one
    /// debounced event with the right trigger.
    #[tokio::test]
    async fn fs_reflex_emits_one_event_after_debounce() {
        // Watch the parent directory rather than a single file —
        // FSEvents (macOS) is unreliable for per-file watches but
        // robust for directories.
        let dir = TempDir::new().expect("tempdir");
        let file = dir.path().join("watched.txt");
        std::fs::write(&file, "initial").unwrap();

        let reflex = Arc::new(FsReflex::new(
            "fs-test",
            FsReflexConfig::new(vec![dir.path().to_path_buf()])
                .recursive(false)
                .debounce(Duration::from_millis(80)),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");

        // Brief settle so the OS watcher is wired before we mutate.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Write a few times to force a burst — debouncer must collapse.
        for _ in 0..3 {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&file)
                .unwrap();
            writeln!(f, "tick").unwrap();
        }

        // Wait at most ~3 seconds for an event (CI safety margin).
        let event = tokio::time::timeout(Duration::from_secs(3), stream.next())
            .await
            .expect("debouncer should emit within timeout")
            .expect("stream still open");
        assert!(
            event.trigger.starts_with("fs:"),
            "trigger {:?} should start with fs:",
            event.trigger
        );
        let kind = event
            .payload
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            matches!(kind, "modified" | "created" | "other"),
            "kind {kind} should be a known classification"
        );
    }

    #[tokio::test]
    async fn fs_reflex_reports_watch_error_for_missing_path() {
        let reflex = Arc::new(FsReflex::new(
            "fs-missing",
            FsReflexConfig::new(vec![PathBuf::from(
                "/this/path/does/not/exist/almost/certainly",
            )]),
        ));
        let result = reflex.subscribe().await;
        assert!(result.is_err(), "watching a non-existent path must error");
    }

    #[test]
    fn classify_kind_maps_each_variant() {
        use notify::event::{CreateKind, ModifyKind, RemoveKind};
        assert_eq!(
            classify_kind(&EventKind::Create(CreateKind::Any)),
            "created"
        );
        assert_eq!(
            classify_kind(&EventKind::Modify(ModifyKind::Any)),
            "modified"
        );
        assert_eq!(
            classify_kind(&EventKind::Remove(RemoveKind::Any)),
            "removed"
        );
        assert_eq!(classify_kind(&EventKind::Other), "other");
    }
}
