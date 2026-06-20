//! Scheduled database backups.
//!
//! A thin orchestration layer over [`storage::SqlitePool::backup_into`]: it
//! writes a timestamped `VACUUM INTO` snapshot into a backup directory and
//! prunes the directory back to the configured retention count, newest kept.
//!
//! The cadence is owned by the caller — `serve` drives [`run_backup`] from a
//! background maintenance loop (alongside the consolidator/compactor), with the
//! same battery etiquette. Backups are pure maintenance, so they run as a
//! direct task rather than through the reflex event surface (a reflex emits
//! signals, it never executes).

use std::path::{Path, PathBuf};

use storage::SqlitePool;
use thiserror::Error;

/// Filename prefix and extension for snapshot files. The middle segment is an
/// RFC3339-ish UTC timestamp with filesystem-safe separators.
const PREFIX: &str = "brain-";
const EXT: &str = ".db";

#[derive(Debug, Error)]
pub enum BackupError {
    #[error("snapshot failed: {0}")]
    Snapshot(#[from] storage::sqlite::SqliteError),
    #[error("backup directory I/O at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Outcome of one backup cycle.
#[derive(Debug, Clone)]
pub struct BackupReport {
    /// Path of the snapshot just written.
    pub snapshot: PathBuf,
    /// Snapshot files removed by retention pruning.
    pub pruned: Vec<PathBuf>,
}

/// Write a timestamped snapshot of `pool` into `dir`, then prune so that at
/// most `retain` snapshots remain (newest kept). `retain` is clamped to a
/// minimum of 1 — a backup run never deletes the snapshot it just wrote.
pub fn run_backup(
    pool: &SqlitePool,
    dir: &Path,
    retain: usize,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<BackupReport, BackupError> {
    let snapshot = dir.join(snapshot_name(now));
    pool.backup_into(&snapshot)?;

    let pruned = prune(dir, retain.max(1))?;
    Ok(BackupReport { snapshot, pruned })
}

/// Snapshot filename for an instant, e.g. `brain-2026-06-20T141503Z.db`.
/// Colons are dropped so the name is valid on every filesystem.
fn snapshot_name(now: chrono::DateTime<chrono::Utc>) -> String {
    format!("{PREFIX}{}{EXT}", now.format("%Y-%m-%dT%H%M%SZ"))
}

/// Remove all but the `retain` newest snapshot files in `dir`. Ordering is by
/// filename, which sorts chronologically because the timestamp is fixed-width.
/// Returns the paths removed.
fn prune(dir: &Path, retain: usize) -> Result<Vec<PathBuf>, BackupError> {
    let mut snapshots = list_snapshots(dir)?;
    if snapshots.len() <= retain {
        return Ok(Vec::new());
    }
    // Newest last → the ones to drop are at the front.
    snapshots.sort();
    let drop_count = snapshots.len() - retain;
    let mut pruned = Vec::with_capacity(drop_count);
    for path in snapshots.into_iter().take(drop_count) {
        match std::fs::remove_file(&path) {
            Ok(()) => pruned.push(path),
            // A concurrent run may have removed it; not fatal.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(BackupError::Io {
                    path: path.display().to_string(),
                    source,
                })
            }
        }
    }
    Ok(pruned)
}

/// Snapshot files in `dir` (those matching the `brain-*.db` naming). A missing
/// directory yields an empty list rather than an error.
fn list_snapshots(dir: &Path) -> Result<Vec<PathBuf>, BackupError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(BackupError::Io {
                path: dir.display().to_string(),
                source,
            })
        }
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with(PREFIX) && name.ends_with(EXT) {
                out.push(path);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};

    fn pool() -> SqlitePool {
        SqlitePool::open_memory().unwrap()
    }

    #[test]
    fn snapshot_name_is_filesystem_safe_and_chronological() {
        let a = snapshot_name(Utc.with_ymd_and_hms(2026, 6, 20, 9, 5, 3).unwrap());
        let b = snapshot_name(Utc.with_ymd_and_hms(2026, 6, 20, 14, 15, 3).unwrap());
        assert_eq!(a, "brain-2026-06-20T090503Z.db");
        assert!(!a.contains(':'), "colons are not portable in filenames");
        // Lexical order matches chronological order (fixed-width timestamp).
        assert!(a < b);
    }

    #[test]
    fn run_backup_writes_snapshot_and_keeps_within_retention() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool();

        // Three runs at distinct timestamps, retain = 2.
        let base = Utc.with_ymd_and_hms(2026, 6, 20, 0, 0, 0).unwrap();
        let mut last = None;
        for i in 0..3 {
            let report = run_backup(&pool, dir.path(), 2, base + Duration::minutes(i)).unwrap();
            assert!(report.snapshot.exists());
            last = Some(report);
        }

        let remaining = list_snapshots(dir.path()).unwrap();
        assert_eq!(remaining.len(), 2, "retention should cap at 2 snapshots");
        // The most recent run pruned exactly one (the oldest).
        assert_eq!(last.unwrap().pruned.len(), 1);

        // The newest snapshot is still present.
        let mut names: Vec<_> = remaining
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names[1], snapshot_name(base + Duration::minutes(2)));
    }

    #[test]
    fn retain_is_clamped_so_a_run_never_deletes_its_own_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool();
        let now = Utc.with_ymd_and_hms(2026, 6, 20, 0, 0, 0).unwrap();
        // retain = 0 must still leave the snapshot just written.
        let report = run_backup(&pool, dir.path(), 0, now).unwrap();
        assert!(report.snapshot.exists());
        assert_eq!(list_snapshots(dir.path()).unwrap().len(), 1);
    }

    #[test]
    fn prune_ignores_unrelated_files() {
        let dir = tempfile::tempdir().unwrap();
        let pool = pool();
        std::fs::write(dir.path().join("notes.txt"), b"keep me").unwrap();
        std::fs::write(dir.path().join("brain.db.bak-v3"), b"old").unwrap();

        let now = Utc.with_ymd_and_hms(2026, 6, 20, 0, 0, 0).unwrap();
        run_backup(&pool, dir.path(), 1, now).unwrap();

        // Unrelated files are untouched; only brain-*.db are managed.
        assert!(dir.path().join("notes.txt").exists());
        assert!(dir.path().join("brain.db.bak-v3").exists());
        assert_eq!(list_snapshots(dir.path()).unwrap().len(), 1);
    }
}
