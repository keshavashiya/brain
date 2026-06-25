//! Per-turn telemetry → learned-normal monitor.
//!
//! Subscribes to the observability bus and folds each `TurnCompleted` event
//! (the per-turn telemetry the chat pipeline publishes) into learned baselines
//! for turn *latency* and *token cost*. When a turn lands far outside its own
//! learned band — "your turns are suddenly 3× slower than normal", or one turn
//! cost an order of magnitude more tokens than usual — it emits a
//! `MetricAnomaly` and a proactive notification, the same surfaces the resource
//! sampler feeds.
//!
//! This is the conversational counterpart to the resource-gauge learned-normal
//! tracker: same detector ([`brain::StreamMonitor`]), same edge discipline (one
//! alert per excursion), same warmup-before-it-knows-you guard. It reads only
//! `TurnCompleted` and writes only `MetricAnomaly`, so it cannot feed back on
//! its own input.

use std::sync::Arc;

/// Stream label for the turn-latency baseline (milliseconds).
const LATENCY_STREAM: &str = "turn.latency_ms";
/// Stream label for the turn token-cost baseline (prompt + completion tokens).
const TOKENS_STREAM: &str = "turn.tokens";

/// The learned baselines for one process's turn stream — latency and token
/// cost. Kept separate from the spawn loop so the event→anomaly mapping is pure
/// and exhaustively testable, the same split as the resource tracker.
struct TurnBaselines {
    latency: brain::StreamMonitor,
    tokens: brain::StreamMonitor,
}

impl TurnBaselines {
    fn new(cfg: &brain::config::LearnedNormalConfig) -> Self {
        let monitor = || brain::StreamMonitor::new(cfg.alpha, cfg.warmup_samples, cfg.sensitivity);
        Self {
            latency: monitor(),
            tokens: monitor(),
        }
    }

    /// Fold one completed turn into both baselines, returning a labelled anomaly
    /// for each stream that has *just* moved outside its learned band.
    fn observe(
        &mut self,
        duration_ms: u64,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Vec<(&'static str, brain::Anomaly)> {
        let total_tokens = input_tokens.saturating_add(output_tokens);
        let mut out = Vec::new();
        if let Some(a) = self.latency.observe(duration_ms as f64) {
            out.push((LATENCY_STREAM, a));
        }
        if let Some(a) = self.tokens.observe(total_tokens as f64) {
            out.push((TOKENS_STREAM, a));
        }
        out
    }
}

/// Subscribe to the bus and watch per-turn telemetry for anomalies.
///
/// Must be spawned *before* turns start running so the broadcast subscription
/// sees them. A no-op (logs and returns) when learned-normal is disabled or no
/// observability bus is wired — the per-turn telemetry needs the bus anyway.
pub(super) fn spawn_turn_baseline(
    processor: Arc<signal::SignalProcessor>,
    cfg: brain::config::LearnedNormalConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    if !cfg.enabled {
        return;
    }
    let Some(observer) = processor.observer() else {
        tracing::warn!(
            "Turn-telemetry baseline: no observability bus wired — per-turn anomalies disabled"
        );
        return;
    };
    let mut rx = observer.subscribe();
    let observer = observer.clone();
    let p = processor.clone();

    let mut baselines = TurnBaselines::new(&cfg);

    set.spawn(async move {
        loop {
            match rx.recv().await {
                Ok(observe::BrainEvent::TurnCompleted {
                    duration_ms,
                    input_tokens,
                    output_tokens,
                    ..
                }) => {
                    for (stream, a) in baselines.observe(duration_ms, input_tokens, output_tokens) {
                        tracing::warn!(
                            stream,
                            value = a.value,
                            expected = a.expected,
                            z_score = a.z_score,
                            "Turn anomaly: {stream} far outside its learned baseline"
                        );
                        let ev = observe::BrainEvent::MetricAnomaly {
                            id: uuid::Uuid::new_v4(),
                            stream: stream.to_string(),
                            value: a.value,
                            expected: a.expected,
                            z_score: a.z_score,
                            ts: chrono::Utc::now(),
                        };
                        let _ = observer.publish(ev).await;
                        if let Some(router) = p.notification_router() {
                            router
                                .deliver(signal::notification::ProactiveNotification {
                                    content: advisory(stream, &a),
                                    triggered_by: format!("metric_anomaly:{stream}"),
                                    priority: 2,
                                    agent: None,
                                })
                                .await;
                        }
                    }
                }
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::RecvError::Lagged(missed)) => {
                    tracing::warn!(
                        missed,
                        "Turn-telemetry baseline lagged behind the bus — skipped turns not learned"
                    );
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
        Ok(())
    });
    tracing::info!("Turn-telemetry baseline scheduled (bus → learned-normal)");
}

/// A human-readable advisory for a per-turn anomaly, naming the learned norm
/// rather than a fixed threshold.
fn advisory(stream: &str, a: &brain::Anomaly) -> String {
    let dir = if a.z_score >= 0.0 { "above" } else { "below" };
    match stream {
        LATENCY_STREAM => format!(
            "A chat turn took {:.0} ms, well {dir} the usual ~{:.0} ms ({:.1}σ out). \
             The model or machine may be under load — worth a look if it persists.",
            a.value,
            a.expected,
            a.z_score.abs(),
        ),
        TOKENS_STREAM => format!(
            "A chat turn used {:.0} tokens, well {dir} the usual ~{:.0} ({:.1}σ out). \
             An unusually large prompt or response — check the context if it persists.",
            a.value,
            a.expected,
            a.z_score.abs(),
        ),
        other => format!(
            "Turn metric '{other}' is {:.0}, well {dir} its usual ~{:.0} ({:.1}σ out).",
            a.value,
            a.expected,
            a.z_score.abs(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sensitive, fast-warming config so a few synthetic turns establish a
    /// baseline and a clear spike trips it.
    fn cfg() -> brain::config::LearnedNormalConfig {
        brain::config::LearnedNormalConfig {
            enabled: true,
            sensitivity: 4.0,
            warmup_samples: 5,
            alpha: 0.3,
        }
    }

    #[test]
    fn slow_turn_flags_latency_once_after_learning() {
        let mut b = TurnBaselines::new(&cfg());
        // A run of normal turns: ~900 ms latency, ~1000 tokens (small jitter so
        // the baselines learn a non-zero spread).
        for (ms, tok) in [
            (880, 980),
            (920, 1010),
            (900, 990),
            (910, 1005),
            (890, 995),
            (905, 1000),
        ] {
            assert!(
                b.observe(ms, tok / 2, tok / 2).is_empty(),
                "normal turns never flag"
            );
        }
        // One turn that takes 10x as long but costs normal tokens: latency
        // anomaly only.
        let out = b.observe(9000, 500, 500);
        assert_eq!(out.len(), 1, "only latency should flag");
        assert_eq!(out[0].0, LATENCY_STREAM);
        assert!(out[0].1.z_score > 4.0, "positive latency anomaly");
        assert!(
            out[0].1.expected > 800.0 && out[0].1.expected < 1000.0,
            "expected tracks the learned ~900 ms, got {}",
            out[0].1.expected
        );

        // Edge discipline: a second slow turn at the same level does not re-spam.
        assert!(b
            .observe(9000, 500, 500)
            .iter()
            .all(|(s, _)| *s != LATENCY_STREAM));
    }

    #[test]
    fn expensive_turn_flags_tokens() {
        let mut b = TurnBaselines::new(&cfg());
        for _ in 0..6 {
            // Stable latency, small token jitter around ~1000.
            b.observe(900, 510, 500);
            b.observe(900, 490, 500);
        }
        // A turn that costs an order of magnitude more tokens at normal latency.
        let out = b.observe(900, 9000, 6000);
        assert!(
            out.iter()
                .any(|(s, a)| *s == TOKENS_STREAM && a.z_score > 4.0),
            "a token blowout should flag the tokens stream: {out:?}"
        );
    }

    #[test]
    fn warmup_suppresses_early_turns() {
        let mut b = TurnBaselines::new(&cfg());
        // Wild swings during warmup (< 5 turns) never flag.
        for (ms, tok) in [(100, 100), (9000, 9000), (100, 100), (8000, 8000)] {
            assert!(b.observe(ms, tok, 0).is_empty());
        }
    }

    #[test]
    fn advisory_names_the_stream_and_learned_norm() {
        let a = brain::Anomaly {
            value: 9000.0,
            expected: 900.0,
            z_score: 8.1,
        };
        let lat = advisory(LATENCY_STREAM, &a);
        assert!(lat.contains("9000") && lat.contains("900"), "{lat}");
        assert!(
            lat.contains("ms") && lat.contains("above") && lat.contains('σ'),
            "{lat}"
        );

        let tok = advisory(TOKENS_STREAM, &a);
        assert!(tok.contains("tokens"), "{tok}");
    }
}
