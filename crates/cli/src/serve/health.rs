//! Service health monitoring (Issue 135).
//!
//! Each configured [`ServiceCheck`](brain::config::ServiceCheck) drives one
//! bounded background probe loop (see `background::spawn_service_monitor`) that
//! reaches an external endpoint — an HTTP GET or a raw TCP connect — on a
//! cadence. This module owns the two pieces that loop composes:
//!
//! * [`probe`] — a single reachability check, returning `Ok(())` when the
//!   service is up or `Err(reason)` with a concise, human-readable cause.
//! * [`HealthEdge`] — an up/down edge tracker so a notification fires only on a
//!   *transition* between reachable and unreachable, never once per probe while
//!   the service holds one state. This is the same edge discipline the resource
//!   sampler's `PressureTracker` uses.
//!
//! [`advisory`] renders the proactive-notification body for a transition.

use std::time::Duration;

use brain::config::{ServiceCheck, ServiceCheckKind};

/// Result of one probe: `Ok(())` when reachable, `Err(reason)` otherwise. The
/// reason is a short phrase suitable for an alert body (`connection refused`,
/// `timed out`, `HTTP 503`, …), never a full backtrace.
pub(crate) type ProbeResult = Result<(), String>;

/// Probe a single service once. The HTTP client is supplied by the caller so it
/// (and its timeout) is built once per loop rather than once per probe.
pub(crate) async fn probe(client: &reqwest::Client, svc: &ServiceCheck) -> ProbeResult {
    match svc.kind {
        ServiceCheckKind::Http => probe_http(client, svc).await,
        ServiceCheckKind::Tcp => probe_tcp(svc).await,
    }
}

/// HTTP GET probe. Healthy when the status equals `expect_status` (if set) or is
/// any 2xx otherwise. The client carries the per-probe timeout, so a hang is
/// reported as a timeout error rather than blocking the loop.
async fn probe_http(client: &reqwest::Client, svc: &ServiceCheck) -> ProbeResult {
    match client.get(&svc.target).send().await {
        Ok(resp) => {
            let status = resp.status();
            match svc.expect_status {
                Some(want) if status.as_u16() != want => {
                    Err(format!("HTTP {} (expected {want})", status.as_u16()))
                }
                Some(_) => Ok(()),
                None if status.is_success() => Ok(()),
                None => Err(format!("HTTP {}", status.as_u16())),
            }
        }
        Err(e) => Err(http_error_reason(&e)),
    }
}

/// Condense a `reqwest::Error` into a short, user-facing phrase. The full
/// `Display` includes the URL and a chain; we want just the failure mode.
fn http_error_reason(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "timed out".to_string()
    } else if e.is_connect() {
        "connection failed".to_string()
    } else {
        "request failed".to_string()
    }
}

/// Raw TCP connect probe. Healthy when the connection is accepted before the
/// timeout; `target` is `host:port` (a hostname is resolved). The stream is
/// dropped immediately — this only checks that the port is accepting.
async fn probe_tcp(svc: &ServiceCheck) -> ProbeResult {
    let timeout = Duration::from_secs(svc.timeout_secs.max(1));
    match tokio::time::timeout(timeout, tokio::net::TcpStream::connect(&svc.target)).await {
        Ok(Ok(_stream)) => Ok(()),
        Ok(Err(e)) => Err(tcp_error_reason(&e)),
        Err(_elapsed) => Err("timed out".to_string()),
    }
}

/// Condense a TCP connect `io::Error` into a short phrase.
fn tcp_error_reason(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::ConnectionRefused => "connection refused".to_string(),
        std::io::ErrorKind::TimedOut => "timed out".to_string(),
        _ => "connection failed".to_string(),
    }
}

/// Up/down edge tracker for one service. [`evaluate`](Self::evaluate) records
/// the latest reachability and returns `Some(now_healthy)` only when the state
/// *changed* in a way worth alerting on — so the user hears about a service
/// going down or recovering, but not about it staying up sample after sample.
///
/// The very first *healthy* result is a silent baseline (a service that is up
/// at startup is the expected case, not news); a first *unhealthy* result does
/// alert, since a configured service that is already unreachable is worth
/// surfacing immediately.
#[derive(Default)]
pub(crate) struct HealthEdge {
    last: Option<bool>,
}

impl HealthEdge {
    /// Fold one probe outcome in. Returns `Some(true)` on a recovery edge,
    /// `Some(false)` on a failure edge (including the initial down), and `None`
    /// when nothing alert-worthy changed.
    pub(crate) fn evaluate(&mut self, healthy: bool) -> Option<bool> {
        let prev = self.last.replace(healthy);
        match prev {
            Some(p) if p == healthy => None, // unchanged — no edge
            None if healthy => None,         // initial healthy baseline — silent
            _ => Some(healthy),              // down, recovery, or initial down
        }
    }
}

/// Body of the proactive notification for one health transition. Names the
/// service and the endpoint actually probed, and either reports the failure
/// cause (on the way down) or a plain recovery (on the way up).
pub(crate) fn advisory(svc: &ServiceCheck, healthy: bool, detail: &str) -> String {
    if healthy {
        format!(
            "Service '{}' ({}) recovered — it's reachable again.",
            svc.name, svc.target
        )
    } else if detail.is_empty() {
        format!("Service '{}' ({}) is unreachable.", svc.name, svc.target)
    } else {
        format!(
            "Service '{}' ({}) is unreachable: {}.",
            svc.name, svc.target, detail
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tcp_check(name: &str, target: &str) -> ServiceCheck {
        ServiceCheck {
            name: name.to_string(),
            kind: ServiceCheckKind::Tcp,
            target: target.to_string(),
            interval_secs: 60,
            timeout_secs: 2,
            expect_status: None,
        }
    }

    #[tokio::test]
    async fn tcp_probe_reaches_a_live_listener() {
        // Bind an ephemeral port, then prove the probe connects to it.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let svc = tcp_check("live", &addr.to_string());

        let client = reqwest::Client::new();
        assert!(probe(&client, &svc).await.is_ok());
    }

    #[tokio::test]
    async fn tcp_probe_reports_a_closed_port_as_down() {
        // Bind then immediately drop the listener so the port is closed; the
        // OS rarely reuses it instantly, so connect should be refused.
        let addr = {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            listener.local_addr().unwrap()
        };
        let svc = tcp_check("dead", &addr.to_string());

        let client = reqwest::Client::new();
        let result = probe(&client, &svc).await;
        assert!(result.is_err(), "closed port must probe as down");
    }

    #[test]
    fn edge_is_silent_on_a_healthy_baseline_then_steady_state() {
        let mut edge = HealthEdge::default();
        assert_eq!(edge.evaluate(true), None, "initial up is a silent baseline");
        assert_eq!(edge.evaluate(true), None, "staying up emits nothing");
    }

    #[test]
    fn edge_alerts_on_initial_down_and_every_transition() {
        let mut edge = HealthEdge::default();
        // A service already down at first probe is worth surfacing.
        assert_eq!(edge.evaluate(false), Some(false), "initial down alerts");
        assert_eq!(edge.evaluate(false), None, "staying down emits nothing");
        assert_eq!(edge.evaluate(true), Some(true), "recovery alerts");
        assert_eq!(edge.evaluate(false), Some(false), "going down again alerts");
    }

    #[test]
    fn advisory_names_service_target_and_state() {
        let svc = tcp_check("postgres", "127.0.0.1:5432");

        let down = advisory(&svc, false, "connection refused");
        assert!(down.contains("postgres"));
        assert!(down.contains("127.0.0.1:5432"));
        assert!(down.contains("unreachable"));
        assert!(down.contains("connection refused"));

        let up = advisory(&svc, true, "");
        assert!(up.contains("postgres"));
        assert!(up.contains("recovered"));

        // Down with no detail still reads cleanly (no dangling colon).
        let bare = advisory(&svc, false, "");
        assert!(bare.contains("unreachable."));
        assert!(!bare.contains("unreachable:"));
    }
}
