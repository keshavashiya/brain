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
            open_fds: open_fd_count(),
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

/// Count of open file descriptors held by this process — the early-warning
/// gauge for an fd leak (sockets/files/pipes that never get closed).
///
/// Read from the kernel's per-process fd directory: `/proc/self/fd` on Linux,
/// `/dev/fd` on macOS/BSD. Both list one entry per open descriptor. The reading
/// itself transiently opens one fd (the directory handle), so the count is
/// `actual + 1` while sampling — immaterial for a leak gauge watching a trend.
/// Returns `None` on platforms without such a directory (e.g. Windows), the
/// same degrade-to-unavailable contract as the other gauges.
fn open_fd_count() -> Option<u64> {
    #[cfg(target_os = "linux")]
    let fd_dir = "/proc/self/fd";
    #[cfg(not(target_os = "linux"))]
    let fd_dir = "/dev/fd";

    fs::read_dir(fd_dir)
        .ok()
        .map(|entries| entries.flatten().count() as u64)
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

impl Crossing {
    /// A human-readable, actionable advisory — the body of the proactive
    /// notification surfaced to the user when a gauge crosses its ceiling.
    ///
    /// Each line describes what Brain *actually* measures (this process and its
    /// `~/.brain` data directory), never an OS-wide claim, and suggests a
    /// concrete next step. Edge-triggering upstream guarantees one advisory per
    /// crossing rather than one per sample.
    pub(crate) fn advisory(&self) -> String {
        match self.gauge {
            "rss" => format!(
                "Brain's memory use is {:.0} MiB, over the {:.0} MiB ceiling. \
                 If it keeps climbing, restart the daemon to reclaim it.",
                self.value, self.threshold
            ),
            "cpu" => format!(
                "Brain's CPU use is {:.0}%, over the {:.0}% ceiling. \
                 A background task may be stuck — check `brain status`.",
                self.value, self.threshold
            ),
            "disk" => format!(
                "Brain's data directory (~/.brain) is using {:.0} MiB, over the \
                 {:.0} MiB ceiling. Run memory consolidation or prune old \
                 episodes to reclaim space.",
                self.value, self.threshold
            ),
            "fds" => format!(
                "Brain is holding {:.0} open file descriptors, over the {:.0} \
                 ceiling. This can signal a descriptor leak — restart the daemon \
                 if it keeps climbing.",
                self.value, self.threshold
            ),
            other => format!(
                "Brain resource '{other}' is at {:.0}, over its {:.0} ceiling.",
                self.value, self.threshold
            ),
        }
    }
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
    fds_over: bool,
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
        let checks: [(&mut bool, Option<f64>, f64, &'static str); 4] = [
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
            (
                &mut self.fds_over,
                snap.open_fds.map(|n| n as f64),
                thresholds.open_fds as f64,
                "fds",
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

/// One gauge that has *just* moved far outside its learned baseline — the edge
/// that warrants exactly one `MetricAnomaly` event. Units match [`Crossing`]:
/// MiB for `rss`/`disk`, percent for `cpu`, a count for `fds`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Deviation {
    pub(crate) gauge: &'static str,
    pub(crate) value: f64,
    /// The learned baseline (EWMA mean) the value was judged against.
    pub(crate) expected: f64,
    /// Signed deviation in learned standard deviations.
    pub(crate) z_score: f64,
}

impl Deviation {
    /// A human-readable advisory for the proactive notification, naming the
    /// learned norm rather than a fixed ceiling — the value, where it normally
    /// sits, and how far out it has jumped.
    pub(crate) fn advisory(&self) -> String {
        let dir = if self.z_score >= 0.0 {
            "above"
        } else {
            "below"
        };
        let unit = match self.gauge {
            "rss" | "disk" => " MiB",
            "cpu" => "%",
            _ => "",
        };
        let what = match self.gauge {
            "rss" => "memory use",
            "cpu" => "CPU use",
            "disk" => "data-directory size (~/.brain)",
            "fds" => "open file descriptors",
            other => other,
        };
        format!(
            "Brain's {what} is {:.0}{unit}, well {dir} its usual ~{:.0}{unit} \
             ({:.1}σ out). This is unusual for this machine — worth a look at \
             `brain status` if it persists.",
            self.value,
            self.expected,
            self.z_score.abs(),
        )
    }
}

/// Per-gauge learned baselines with edge-triggered anomaly detection. Each gauge
/// keeps a [`StreamMonitor`](brain::StreamMonitor) of its own normal range; a
/// reading more than `sensitivity` learned standard deviations out yields one
/// [`Deviation`], and — like [`PressureTracker`] — nothing more while it stays
/// out, re-arming only once it returns inside the band. The baseline keeps
/// learning on every sample, so a sustained shift is absorbed as the new normal.
pub(crate) struct LearnedNormalTracker {
    rss: brain::StreamMonitor,
    cpu: brain::StreamMonitor,
    disk: brain::StreamMonitor,
    fds: brain::StreamMonitor,
}

impl LearnedNormalTracker {
    pub(crate) fn new(cfg: &brain::config::LearnedNormalConfig) -> Self {
        let monitor = || brain::StreamMonitor::new(cfg.alpha, cfg.warmup_samples, cfg.sensitivity);
        Self {
            rss: monitor(),
            cpu: monitor(),
            disk: monitor(),
            fds: monitor(),
        }
    }

    /// Feed one snapshot into every gauge's baseline, returning a deviation for
    /// each gauge that has *just* moved outside its learned band. An unavailable
    /// gauge (`None`) is skipped without disturbing its baseline or edge state,
    /// the same contract as [`PressureTracker::evaluate`].
    pub(crate) fn evaluate(&mut self, snap: &metrics::ResourceSnapshot) -> Vec<Deviation> {
        let mut out = Vec::new();
        let checks: [(&mut brain::StreamMonitor, Option<f64>, &'static str); 4] = [
            (&mut self.rss, snap.rss_bytes.map(|b| b as f64 / MIB), "rss"),
            (&mut self.cpu, snap.cpu_pct, "cpu"),
            (
                &mut self.disk,
                snap.disk_bytes.map(|b| b as f64 / MIB),
                "disk",
            ),
            (&mut self.fds, snap.open_fds.map(|n| n as f64), "fds"),
        ];
        for (monitor, value, gauge) in checks {
            // Unavailable reading skips the gauge without disturbing its state.
            let Some(value) = value else { continue };
            if let Some(a) = monitor.observe(value) {
                out.push(Deviation {
                    gauge,
                    value: a.value,
                    expected: a.expected,
                    z_score: a.z_score,
                });
            }
        }
        out
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
        // A live process always holds at least stdio (fds 0/1/2) on the
        // platforms with an fd directory (Linux /proc/self/fd, macOS /dev/fd).
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(
            snap.open_fds.is_some_and(|n| n >= 3),
            "live process must report its open fds, got {:?}",
            snap.open_fds
        );
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
            open_fds: 200,
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
            open_fds: Some(10), // under the 200 fd ceiling
        };
        let mut crossings = tracker.evaluate(&snap, &th);
        crossings.sort_by_key(|c| c.gauge);
        assert_eq!(crossings.len(), 2);
        assert_eq!(crossings[0].gauge, "cpu");
        assert_eq!(crossings[0].value, 75.0);
        assert_eq!(crossings[1].gauge, "rss");
    }

    #[test]
    fn advisory_is_gauge_specific_and_actionable() {
        // Each gauge gets a distinct, actionable line that names what was
        // actually measured (process / ~/.brain), with the value + ceiling.
        let cases = [
            ("rss", "memory", "restart"),
            ("cpu", "CPU", "brain status"),
            ("disk", "~/.brain", "consolidation"),
            ("fds", "file descriptors", "leak"),
        ];
        for (gauge, names, advises) in cases {
            let c = Crossing {
                gauge,
                value: 150.0,
                threshold: 100.0,
                severity: "warn",
            };
            let msg = c.advisory();
            assert!(msg.contains(names), "{gauge}: expected '{names}' in: {msg}");
            assert!(
                msg.contains(advises),
                "{gauge}: expected advice '{advises}' in: {msg}"
            );
            assert!(msg.contains("150"), "{gauge}: value missing in: {msg}");
            assert!(msg.contains("100"), "{gauge}: ceiling missing in: {msg}");
        }
    }

    #[test]
    fn advisory_handles_unknown_gauge() {
        let c = Crossing {
            gauge: "future_gauge",
            value: 9.0,
            threshold: 5.0,
            severity: "warn",
        };
        let msg = c.advisory();
        assert!(msg.contains("future_gauge"));
        assert!(msg.contains('9') && msg.contains('5'));
    }

    /// A sensitive, fast-warming config so tests learn a baseline in a few
    /// samples and trip on a clear spike.
    fn test_learned_cfg() -> brain::config::LearnedNormalConfig {
        brain::config::LearnedNormalConfig {
            enabled: true,
            sensitivity: 4.0,
            warmup_samples: 5,
            alpha: 0.3,
        }
    }

    #[test]
    fn learned_normal_silent_while_stable_then_fires_once_on_spike() {
        let mut tracker = LearnedNormalTracker::new(&test_learned_cfg());
        // A noisy-but-stable RSS baseline around ~100 MiB — never an anomaly.
        for mb in [98u64, 102, 99, 101, 100, 103, 97, 100] {
            assert!(
                tracker.evaluate(&snap_rss_mb(mb)).is_empty(),
                "stable readings must not flag"
            );
        }
        // A 10x spike, well under any reasonable ceiling but far outside the
        // learned band → exactly one deviation.
        let devs = tracker.evaluate(&snap_rss_mb(1000));
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].gauge, "rss");
        assert_eq!(devs[0].value, 1000.0);
        assert!(
            devs[0].z_score > 4.0,
            "positive anomaly, got {}",
            devs[0].z_score
        );
        assert!(
            devs[0].expected > 90.0 && devs[0].expected < 130.0,
            "expected tracks the learned ~100, got {}",
            devs[0].expected
        );

        // Still spiking on the next sample → no spam (edge discipline).
        assert!(tracker.evaluate(&snap_rss_mb(1000)).is_empty());
    }

    #[test]
    fn learned_normal_warmup_suppresses_early_swings() {
        let mut tracker = LearnedNormalTracker::new(&test_learned_cfg());
        // Wild swings during warmup (< 5 samples) are never flagged.
        for mb in [10u64, 5000, 10, 9000] {
            assert!(tracker.evaluate(&snap_rss_mb(mb)).is_empty());
        }
    }

    #[test]
    fn learned_normal_re_arms_after_returning_to_band() {
        let mut tracker = LearnedNormalTracker::new(&test_learned_cfg());
        for mb in [98u64, 102, 99, 101, 100, 103, 97, 100] {
            tracker.evaluate(&snap_rss_mb(mb));
        }
        assert_eq!(
            tracker.evaluate(&snap_rss_mb(1000)).len(),
            1,
            "first spike fires"
        );
        // A long, stable recovery re-arms the edge (no event on the way back in)
        // and lets the EWMA variance inflated by the spike settle back down, so
        // the band returns to ~its learned width.
        for _ in 0..40 {
            assert!(
                tracker.evaluate(&snap_rss_mb(100)).is_empty(),
                "stable recovery must not flag"
            );
        }
        // …so a later spike fires again.
        assert_eq!(
            tracker.evaluate(&snap_rss_mb(1000)).len(),
            1,
            "re-armed spike fires"
        );
    }

    #[test]
    fn learned_normal_advisory_names_the_learned_norm() {
        let dev = Deviation {
            gauge: "rss",
            value: 1000.0,
            expected: 100.0,
            z_score: 9.2,
        };
        let msg = dev.advisory();
        assert!(msg.contains("memory use"), "names the gauge: {msg}");
        assert!(msg.contains("1000"), "names the value: {msg}");
        assert!(msg.contains("100"), "names the learned norm: {msg}");
        assert!(msg.contains("above"), "names the direction: {msg}");
        assert!(msg.contains('σ'), "names the deviation magnitude: {msg}");
    }

    #[test]
    fn fd_gauge_crosses_its_own_ceiling() {
        let th = test_thresholds(); // open_fds ceiling = 200
        let mut tracker = PressureTracker::default();

        let under = metrics::ResourceSnapshot {
            open_fds: Some(50),
            ..Default::default()
        };
        assert!(tracker.evaluate(&under, &th).is_empty());

        let over = metrics::ResourceSnapshot {
            open_fds: Some(250),
            ..Default::default()
        };
        let crossings = tracker.evaluate(&over, &th);
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].gauge, "fds");
        assert_eq!(crossings[0].value, 250.0);
        assert_eq!(crossings[0].threshold, 200.0);

        // Edge discipline holds for the new gauge too: silent while it stays over.
        assert!(tracker.evaluate(&over, &th).is_empty());
    }
}
