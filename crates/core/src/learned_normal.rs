//! Learned-normal baselines for runtime metric streams.
//!
//! The static [`PressureTracker`](../../brainos_cli) answers "is this gauge over
//! a *configured* ceiling?". This module answers a complementary question: "is
//! this reading anomalous *relative to what this gauge normally does on this
//! machine*?" — a learned baseline rather than a hand-set threshold. A gauge can
//! sit well under its ceiling and still be flagged when it jumps far outside its
//! own learned band (an early-warning signal a fixed ceiling can't give), and a
//! machine whose "normal" load is high never trips simply for being busy.
//!
//! The baseline is an exponentially-weighted moving average of the stream's mean
//! and variance ([`EwmaBaseline`]): recent readings weigh more, so a genuine
//! regime shift is *absorbed* as the new normal after a while rather than
//! alarming forever. Detection is a z-score — how many learned standard
//! deviations a reading sits from the learned mean.
//!
//! Two guards keep it quiet until it actually knows the machine:
//! - a **warmup**: no judgement is made until `warmup` readings have been seen,
//!   so the first minutes after boot never alarm;
//! - a **zero-variance gate**: a perfectly flat stream has no learned spread to
//!   judge against, so it is never flagged on its first wobble.
//!
//! The type is pure online arithmetic with no dependencies, so it lives in the
//! dependency-free `core` leaf and is exhaustively unit-testable. Edge-trigger
//! discipline (one alert per excursion) is layered on top by the caller, exactly
//! as the static pressure tracker does.

/// One evaluable reading judged against the learned baseline as it stood
/// *before* this reading was folded in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reading {
    /// The learned mean (EWMA) the reading was judged against.
    pub mean: f64,
    /// Signed deviation in learned standard deviations: positive = above
    /// normal, negative = below.
    pub z_score: f64,
}

/// An exponentially-weighted moving baseline of one metric stream's mean and
/// variance, with a warmup gate.
///
/// Feed every sample through [`observe`](EwmaBaseline::observe); it always
/// updates the baseline (so learning continues through anomalies and the band
/// adapts to a sustained shift) and returns a [`Reading`] only once the stream
/// is evaluable (past warmup, with non-zero learned variance).
#[derive(Debug, Clone)]
pub struct EwmaBaseline {
    /// Smoothing factor in `(0, 1]`: larger forgets faster (tracks recent
    /// readings), smaller is steadier (a longer memory of normal).
    alpha: f64,
    /// Readings required before any judgement is made.
    warmup: u64,
    /// Readings seen so far (saturating).
    count: u64,
    /// Current EWMA mean.
    mean: f64,
    /// Current EWMA variance (West's incremental update).
    variance: f64,
}

impl EwmaBaseline {
    /// A baseline with smoothing `alpha` ∈ `(0, 1]` and a `warmup` count. `alpha`
    /// is clamped into a sane open interval so a misconfigured `0`/`1` can't make
    /// the baseline never learn or never remember.
    pub fn new(alpha: f64, warmup: u64) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 1.0),
            warmup,
            count: 0,
            mean: 0.0,
            variance: 0.0,
        }
    }

    /// Readings seen so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The current learned mean (meaningful only once `count > 0`).
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Fold one sample into the baseline and judge it.
    ///
    /// Returns `Some(Reading)` — the mean and z-score the sample was measured
    /// against — when the stream is evaluable: at least `warmup` prior readings
    /// and a non-zero learned variance. Returns `None` during warmup or for a
    /// perfectly flat stream (no spread to judge against). The baseline is
    /// updated either way, so it keeps learning regardless of the verdict.
    pub fn observe(&mut self, x: f64) -> Option<Reading> {
        // Judge against the baseline as it stands *before* incorporating `x`:
        // a reading is anomalous relative to the prior normal, then becomes part
        // of it.
        let reading = if self.count >= self.warmup && self.variance > 0.0 {
            let stddev = self.variance.sqrt();
            Some(Reading {
                mean: self.mean,
                z_score: (x - self.mean) / stddev,
            })
        } else {
            None
        };

        if self.count == 0 {
            self.mean = x;
            self.variance = 0.0;
        } else {
            // West (1979) incremental EWMA mean + variance.
            let diff = x - self.mean;
            let incr = self.alpha * diff;
            self.mean += incr;
            self.variance = (1.0 - self.alpha) * (self.variance + diff * incr);
        }
        self.count = self.count.saturating_add(1);

        reading
    }
}

/// A reading that has *just* moved outside its learned band — the edge that
/// warrants exactly one alert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anomaly {
    /// The value that tripped the band, in the stream's natural unit.
    pub value: f64,
    /// The learned baseline (EWMA mean) it was judged against.
    pub expected: f64,
    /// Signed deviation in learned standard deviations: positive = above
    /// normal, negative = below.
    pub z_score: f64,
}

/// One metric stream's learned baseline plus edge-triggered anomaly detection.
///
/// Feed every sample through [`observe`](StreamMonitor::observe): it returns an
/// [`Anomaly`] only on a *fresh* move outside `sensitivity` learned standard
/// deviations, then stays silent while the stream remains out, re-arming once it
/// returns inside the band. The baseline keeps learning on every sample, so a
/// sustained shift is absorbed as the new normal. This is the reusable detector
/// behind both the resource-gauge tracker and the per-turn telemetry monitor.
#[derive(Debug, Clone)]
pub struct StreamMonitor {
    baseline: EwmaBaseline,
    sensitivity: f64,
    in_anomaly: bool,
}

impl StreamMonitor {
    /// A monitor with the given EWMA `alpha`, `warmup` count, and anomaly
    /// `sensitivity` (learned standard deviations out before a reading counts).
    pub fn new(alpha: f64, warmup: u64, sensitivity: f64) -> Self {
        Self {
            baseline: EwmaBaseline::new(alpha, warmup),
            sensitivity,
            in_anomaly: false,
        }
    }

    /// Fold one sample in and report a fresh band excursion, if any. `None`
    /// during warmup, while the reading is within the band, or while the stream
    /// is already known to be out (edge discipline — one alert per excursion).
    pub fn observe(&mut self, value: f64) -> Option<Anomaly> {
        let reading = self.baseline.observe(value)?;
        if reading.z_score.abs() >= self.sensitivity {
            if self.in_anomaly {
                None // already reported on the way out
            } else {
                self.in_anomaly = true;
                Some(Anomaly {
                    value,
                    expected: reading.mean,
                    z_score: reading.z_score,
                })
            }
        } else {
            self.in_anomaly = false;
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_silent_while_stable_then_fires_once() {
        let mut m = StreamMonitor::new(0.3, 5, 4.0);
        for x in [98.0, 102.0, 99.0, 101.0, 100.0, 103.0, 97.0, 100.0] {
            assert!(m.observe(x).is_none(), "stable readings never flag");
        }
        let a = m.observe(1000.0).expect("fresh excursion");
        assert!(a.z_score > 4.0);
        assert!(a.expected > 90.0 && a.expected < 130.0);
        // Edge discipline: still out → silent.
        assert!(m.observe(1000.0).is_none());
    }

    #[test]
    fn monitor_warmup_suppresses_early_swings() {
        let mut m = StreamMonitor::new(0.3, 5, 4.0);
        for x in [10.0, 5000.0, 10.0, 9000.0] {
            assert!(m.observe(x).is_none(), "no alert before warmup");
        }
    }

    #[test]
    fn silent_during_warmup() {
        let mut b = EwmaBaseline::new(0.3, 5);
        // Even a wild swing is ignored until warmup is satisfied.
        for x in [100.0, 100.0, 5000.0, 100.0] {
            assert!(b.observe(x).is_none(), "no judgement before warmup");
        }
    }

    #[test]
    fn flat_stream_never_flags() {
        let mut b = EwmaBaseline::new(0.3, 3);
        // A perfectly constant stream has zero learned variance, so even a later
        // identical reading yields no z-score (no spread to judge against).
        for _ in 0..20 {
            assert!(b.observe(100.0).is_none());
        }
    }

    #[test]
    fn spike_after_learning_reports_large_z() {
        let mut b = EwmaBaseline::new(0.3, 5);
        // Learn a noisy-but-stable baseline around ~100.
        for x in [98.0, 102.0, 99.0, 101.0, 100.0, 103.0, 97.0, 100.0] {
            b.observe(x);
        }
        // A large excursion lands many standard deviations out.
        let r = b
            .observe(1000.0)
            .expect("evaluable after warmup with variance");
        assert!(
            r.z_score > 5.0,
            "a 10x spike should be a strong positive anomaly, got z={}",
            r.z_score
        );
        assert!(
            r.mean > 90.0 && r.mean < 110.0,
            "mean tracks ~100, got {}",
            r.mean
        );
    }

    #[test]
    fn low_excursion_reports_negative_z() {
        let mut b = EwmaBaseline::new(0.3, 5);
        for x in [50.0, 52.0, 48.0, 51.0, 49.0, 50.0, 53.0, 47.0] {
            b.observe(x);
        }
        let r = b.observe(0.0).expect("evaluable");
        assert!(
            r.z_score < -3.0,
            "a drop to zero is a negative anomaly, got z={}",
            r.z_score
        );
    }

    #[test]
    fn sustained_shift_is_absorbed_as_new_normal() {
        let mut b = EwmaBaseline::new(0.3, 5);
        for x in [10.0, 12.0, 9.0, 11.0, 10.0, 13.0, 8.0, 10.0] {
            b.observe(x);
        }
        // First reading at the new level is a clear anomaly…
        let first = b.observe(100.0).expect("evaluable");
        assert!(
            first.z_score > 5.0,
            "first jump flagged, got z={}",
            first.z_score
        );
        // …but feeding the new level repeatedly pulls the mean up so the z-score
        // shrinks toward normal — the band has learned the new regime.
        let mut last_z = first.z_score;
        for _ in 0..30 {
            if let Some(r) = b.observe(100.0) {
                last_z = r.z_score;
            }
        }
        assert!(
            last_z.abs() < 1.0,
            "sustained new level should be absorbed, ended at z={last_z}"
        );
    }

    #[test]
    fn alpha_is_clamped() {
        // alpha 0 would freeze learning; it's clamped above zero so the mean
        // still moves.
        let mut b = EwmaBaseline::new(0.0, 1);
        b.observe(10.0);
        b.observe(20.0);
        assert!(b.mean() > 10.0, "clamped alpha must still update the mean");
    }
}
