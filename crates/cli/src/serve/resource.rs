//! Runtime resource probe backing the resource sampler
//! (`background::spawn_resource_sampler`).
//!
//! This lives at the binary edge — deliberately *not* in the dependency-free
//! `metrics` leaf crate — because reading RSS/CPU needs the `sysinfo`
//! dependency. The gauge *store* (`metrics::ResourceMetrics`) stays dep-free;
//! this module is what fills it each tick.
//!
//! macOS + Linux read every gauge. On a platform `sysinfo` can't probe (Windows
//! best-effort) the corresponding field degrades to `None` rather than failing
//! the whole sample — the snapshot is partial, never an error.

use std::fs;
use std::path::{Path, PathBuf};

use metrics::ResourceSnapshot;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// Stateful probe over the current process. Held across sampler ticks so
/// `sysinfo` can compute CPU utilisation as a delta since the previous refresh
/// — consequently the *first* sample reports `0%` CPU (no prior baseline), and
/// every subsequent one reflects usage since the last tick.
pub(crate) struct ResourceProbe {
    system: System,
    /// `None` if the platform won't surface the current pid — RSS/CPU then
    /// degrade to `None` while disk/connection gauges still work.
    pid: Option<Pid>,
    /// Resolved (tilde-expanded) `~/.brain` path whose on-disk footprint the
    /// disk gauge measures.
    data_dir: PathBuf,
}

impl ResourceProbe {
    pub(crate) fn new(data_dir: PathBuf) -> Self {
        Self {
            system: System::new(),
            pid: sysinfo::get_current_pid().ok(),
            data_dir,
        }
    }

    /// Sample the current gauges. `open_connections` is supplied by the caller
    /// from the SQLite pool's state — it is not an OS-level reading, so the
    /// probe stays decoupled from `storage`.
    pub(crate) fn sample(&mut self, open_connections: Option<u64>) -> ResourceSnapshot {
        let (rss_bytes, cpu_pct) = self.sample_process();
        ResourceSnapshot {
            rss_bytes,
            cpu_pct,
            open_connections,
            disk_bytes: dir_size(&self.data_dir),
        }
    }

    /// RSS (bytes) and CPU (percent, single-core basis) for this process, or
    /// `None` apiece when the pid/platform can't be read.
    fn sample_process(&mut self) -> (Option<u64>, Option<f64>) {
        let Some(pid) = self.pid else {
            return (None, None);
        };
        self.system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory().with_cpu(),
        );
        match self.system.process(pid) {
            Some(p) => (Some(p.memory()), Some(f64::from(p.cpu_usage()))),
            None => (None, None),
        }
    }
}

/// Total size in bytes of every regular file under `dir`, walked iteratively.
///
/// Symlinks are not followed (avoids cycles and double-counting); entries that
/// can't be read are skipped so a transient permission error never sinks the
/// whole sample. Returns `None` only when the root directory itself is
/// unreadable (e.g. not yet created) — that reads back as an unavailable gauge.
fn dir_size(dir: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut read_any = false;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = fs::read_dir(&path) else {
            continue;
        };
        read_any = true;
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if let Ok(meta) = entry.metadata() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    read_any.then_some(total)
}

/// A single gauge crossing its configured ceiling — the edge that warrants
/// exactly one `ResourcePressure` event. Units match the threshold: MiB for
/// `rss`/`disk`, percent for `cpu`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Crossing {
    pub(crate) gauge: &'static str,
    pub(crate) value: f64,
    pub(crate) threshold: f64,
    pub(crate) severity: &'static str,
}

/// Per-gauge over/under state for edge-triggered pressure detection. A gauge
/// that rises above its ceiling yields one [`Crossing`]; while it stays over it
/// yields nothing more; only after it drops back under and re-crosses does it
/// fire again. This is the same edge discipline `BudgetCrossed` uses — the bus
/// gets the signal once per event, not once per 30-second sample.
#[derive(Default)]
pub(crate) struct PressureTracker {
    rss_over: bool,
    cpu_over: bool,
    disk_over: bool,
}

/// Bytes per mebibyte — the unit the RSS/disk thresholds are expressed in.
const MIB: f64 = (1024 * 1024) as f64;

impl PressureTracker {
    /// Evaluate one snapshot against the configured ceilings, returning a
    /// crossing for each gauge that has *just* risen above its ceiling since the
    /// previous sample. A threshold of `0` disables that gauge; an unavailable
    /// gauge (`None`) is skipped without disturbing its state.
    ///
    /// Only `warn` is produced today: the config carries a single ceiling per
    /// gauge, so there is no `critical` line to cross yet. The event's
    /// `severity` field is already `critical`-capable for when one lands.
    pub(crate) fn evaluate(
        &mut self,
        snap: &metrics::ResourceSnapshot,
        thresholds: &brain::config::ResourceThresholds,
    ) -> Vec<Crossing> {
        let mut out = Vec::new();
        let checks: [(&mut bool, Option<f64>, f64, &'static str); 3] = [
            (
                &mut self.rss_over,
                snap.rss_bytes.map(|b| b as f64 / MIB),
                thresholds.rss_mb as f64,
                "rss",
            ),
            (&mut self.cpu_over, snap.cpu_pct, thresholds.cpu_pct, "cpu"),
            (
                &mut self.disk_over,
                snap.disk_bytes.map(|b| b as f64 / MIB),
                thresholds.disk_mb as f64,
                "disk",
            ),
        ];
        for (was_over, value, threshold, gauge) in checks {
            if let Some(crossing) = edge(was_over, value, threshold, gauge) {
                out.push(crossing);
            }
        }
        out
    }
}

/// Edge detector for one gauge: mutates `was_over` and returns a [`Crossing`]
/// only on a fresh upward crossing.
fn edge(
    was_over: &mut bool,
    value: Option<f64>,
    threshold: f64,
    gauge: &'static str,
) -> Option<Crossing> {
    // A zero (or negative) threshold disables the gauge — reset and stay silent.
    if threshold <= 0.0 {
        *was_over = false;
        return None;
    }
    // Unavailable reading: no edge, and leave the prior state untouched.
    let value = value?;
    if value >= threshold {
        if *was_over {
            None // already reported on the way up
        } else {
            *was_over = true;
            Some(Crossing {
                gauge,
                value,
                threshold,
                severity: "warn",
            })
        }
    } else {
        *was_over = false;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn probe_reads_real_process_gauges() {
        let dir = std::env::temp_dir();
        let mut probe = ResourceProbe::new(dir);
        let snap = probe.sample(Some(3));

        // RSS is always non-zero for a live process on macOS/Linux. (On a
        // platform sysinfo can't probe, this is `None` — accept either rather
        // than fail the build there.)
        if let Some(rss) = snap.rss_bytes {
            assert!(rss > 0, "live process must report non-zero RSS");
        }
        // CPU on the first sample has no baseline → 0%, but must be present
        // wherever RSS is.
        assert_eq!(snap.cpu_pct.is_some(), snap.rss_bytes.is_some());
        // Connection count is passed through verbatim.
        assert_eq!(snap.open_connections, Some(3));
    }

    #[test]
    fn dir_size_sums_files_and_skips_missing() {
        let tmp = std::env::temp_dir().join(format!("brain-dirsize-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);

        // Missing directory → unavailable gauge.
        assert_eq!(dir_size(&tmp), None);

        // One 100-byte file in the root and one in a subdir → 200 bytes total.
        fs::create_dir_all(tmp.join("sub")).unwrap();
        for rel in ["a.bin", "sub/b.bin"] {
            let mut f = fs::File::create(tmp.join(rel)).unwrap();
            f.write_all(&[0u8; 100]).unwrap();
        }
        assert_eq!(dir_size(&tmp), Some(200));

        fs::remove_dir_all(&tmp).unwrap();
    }

    /// Thresholds tuned low so a tiny snapshot trips them: 100 MiB RSS, 50% CPU,
    /// 100 MiB disk.
    fn test_thresholds() -> brain::config::ResourceThresholds {
        brain::config::ResourceThresholds {
            rss_mb: 100,
            cpu_pct: 50.0,
            disk_mb: 100,
        }
    }

    fn snap_rss_mb(mb: u64) -> metrics::ResourceSnapshot {
        metrics::ResourceSnapshot {
            rss_bytes: Some(mb * 1024 * 1024),
            ..Default::default()
        }
    }

    #[test]
    fn fires_once_on_crossing_then_stays_silent() {
        let th = test_thresholds();
        let mut tracker = PressureTracker::default();

        // Under the ceiling → silent.
        assert!(tracker.evaluate(&snap_rss_mb(50), &th).is_empty());

        // First sample over the ceiling → exactly one warn crossing.
        let crossings = tracker.evaluate(&snap_rss_mb(150), &th);
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].gauge, "rss");
        assert_eq!(crossings[0].severity, "warn");
        assert_eq!(crossings[0].value, 150.0);
        assert_eq!(crossings[0].threshold, 100.0);

        // Still over on the next two samples → no spam.
        assert!(tracker.evaluate(&snap_rss_mb(160), &th).is_empty());
        assert!(tracker.evaluate(&snap_rss_mb(300), &th).is_empty());
    }

    #[test]
    fn re_fires_after_dropping_back_under() {
        let th = test_thresholds();
        let mut tracker = PressureTracker::default();

        assert_eq!(tracker.evaluate(&snap_rss_mb(150), &th).len(), 1);
        // Drops back under — re-arms, no event on the way down.
        assert!(tracker.evaluate(&snap_rss_mb(40), &th).is_empty());
        // Crosses again → fires again.
        assert_eq!(tracker.evaluate(&snap_rss_mb(150), &th).len(), 1);
    }

    #[test]
    fn zero_threshold_disables_the_gauge() {
        let mut th = test_thresholds();
        th.rss_mb = 0;
        let mut tracker = PressureTracker::default();
        // Way over what would be a ceiling, but the gauge is disabled.
        assert!(tracker.evaluate(&snap_rss_mb(100_000), &th).is_empty());
    }

    #[test]
    fn unavailable_gauge_emits_nothing_and_keeps_state() {
        let th = test_thresholds();
        let mut tracker = PressureTracker::default();

        // Over the ceiling → fires.
        assert_eq!(tracker.evaluate(&snap_rss_mb(150), &th).len(), 1);
        // Gauge goes unavailable: no event, and the "over" state is preserved
        // (so a later still-over reading doesn't re-fire).
        assert!(tracker
            .evaluate(&metrics::ResourceSnapshot::default(), &th)
            .is_empty());
        assert!(tracker.evaluate(&snap_rss_mb(150), &th).is_empty());
    }

    #[test]
    fn independent_gauges_cross_independently() {
        let th = test_thresholds();
        let mut tracker = PressureTracker::default();
        let snap = metrics::ResourceSnapshot {
            rss_bytes: Some(150 * 1024 * 1024),
            cpu_pct: Some(75.0),
            disk_bytes: Some(10 * 1024 * 1024), // under the 100 MiB disk ceiling
            open_connections: Some(2),
        };
        let mut crossings = tracker.evaluate(&snap, &th);
        crossings.sort_by_key(|c| c.gauge);
        assert_eq!(crossings.len(), 2);
        assert_eq!(crossings[0].gauge, "cpu");
        assert_eq!(crossings[0].value, 75.0);
        assert_eq!(crossings[1].gauge, "rss");
    }
}
