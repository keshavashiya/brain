//! Network diagnostics — the executor behind `brain net check/trace/cert` and
//! the `net.check` / `net.trace` / `net.cert` native capabilities (Issue 139).
//!
//! Three probes, each with a structured report and a `render()` for human/LLM
//! output:
//!
//! * **check** — DNS resolution + a timed TCP connect to a host[:port]. Pure
//!   Rust (std + tokio), no subprocess, no sandbox concern.
//! * **cert** — a TLS handshake that *captures* the presented certificate chain
//!   with a permissive verifier, so it can inspect expired / self-signed /
//!   untrusted certs and **report** the problem rather than fail. X.509 fields
//!   (subject, issuer, validity, SANs) are parsed with `x509-cert`. Also pure
//!   Rust.
//! * **trace** — `traceroute` (Unix). traceroute needs privileged raw sockets,
//!   so — exactly like `brain logs analyze --source system` — it runs as a
//!   timeout-bounded child process rather than inside the exec sandbox (Apple's
//!   Seatbelt and Linux namespaces both deny the raw-socket path). This is the
//!   macOS-sandbox gotcha called out for slice 139.
//!
//! The same core functions back both surfaces: the CLI calls them directly and
//! prints `render()`; [`NetDiagnostics`] wraps them to satisfy the
//! [`cortex::actions::NetDiagnosticsBackend`] trait the chat tool-loop dispatches
//! through.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::net::TcpStream;

/// Default port used when a `check`/`cert` target names only a host. 443 is the
/// most useful default for a local-first kernel probing HTTPS API endpoints.
pub const DEFAULT_PORT: u16 = 443;

/// Per-probe wall-clock budget. Diagnostics must fail fast, not hang.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Errors from a network probe.
#[derive(Debug, thiserror::Error)]
pub enum NetError {
    #[error("{0}")]
    Msg(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl NetError {
    fn msg(s: impl Into<String>) -> Self {
        NetError::Msg(s.into())
    }
}

// ─── target parsing ─────────────────────────────────────────────────────────

/// Split a target into `(host, port)`. Accepts a bare host, `host:port`, or a
/// full URL (`https://host[:port]/…`); falls back to `default_port` when no
/// port is present.
fn split_host_port(target: &str, default_port: u16) -> Result<(String, u16), NetError> {
    let target = target.trim();
    if target.is_empty() {
        return Err(NetError::msg("empty target"));
    }

    // A full URL: let `url` extract host + port.
    if target.contains("://") {
        let parsed =
            url::Url::parse(target).map_err(|e| NetError::msg(format!("invalid URL: {e}")))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| NetError::msg("URL has no host"))?
            .to_string();
        let port = parsed.port_or_known_default().unwrap_or(default_port);
        return Ok((host, port));
    }

    // `host:port` — but don't mistake a bare IPv6 literal for one. We only
    // accept the simple `host:port` shape here; bracketed IPv6 falls through to
    // the bare-host branch.
    if let Some((host, port)) = target.rsplit_once(':') {
        if !host.is_empty() && !host.contains(':') {
            if let Ok(p) = port.parse::<u16>() {
                return Ok((host.to_string(), p));
            }
        }
    }

    Ok((target.to_string(), default_port))
}

// ─── check ──────────────────────────────────────────────────────────────────

/// Result of a reachability check.
pub struct CheckReport {
    pub host: String,
    pub port: u16,
    /// Resolved addresses (may be empty if DNS failed — then `dns_error` is set).
    pub addresses: Vec<String>,
    pub dns_error: Option<String>,
    /// `Ok(rtt)` on a successful TCP connect, `Err(reason)` otherwise. `None`
    /// when DNS failed and there was nothing to connect to.
    pub connect: Option<Result<Duration, String>>,
}

impl CheckReport {
    pub fn render(&self) -> String {
        let mut s = format!("net check {}:{}\n", self.host, self.port);
        match &self.dns_error {
            Some(e) => s.push_str(&format!("  DNS:     failed — {e}\n")),
            None => s.push_str(&format!(
                "  DNS:     {} → {}\n",
                self.addresses.len(),
                self.addresses.join(", ")
            )),
        }
        match &self.connect {
            None => s.push_str("  TCP:     skipped (no address)\n"),
            Some(Ok(rtt)) => {
                s.push_str(&format!("  TCP:     connected in {} ms\n", rtt.as_millis()))
            }
            Some(Err(e)) => s.push_str(&format!("  TCP:     failed — {e}\n")),
        }
        s.push_str(&format!(
            "  Status:  {}",
            if self.reachable() {
                "reachable"
            } else {
                "unreachable"
            }
        ));
        s
    }

    fn reachable(&self) -> bool {
        matches!(self.connect, Some(Ok(_)))
    }
}

/// Resolve `target` and time a TCP connect to it.
pub async fn check(target: &str) -> Result<CheckReport, NetError> {
    let (host, port) = split_host_port(target, DEFAULT_PORT)?;

    // DNS — `lookup_host` wants `host:port` and yields `SocketAddr`s.
    let lookup = tokio::time::timeout(
        PROBE_TIMEOUT,
        tokio::net::lookup_host(format!("{host}:{port}")),
    )
    .await;

    let mut report = CheckReport {
        host: host.clone(),
        port,
        addresses: Vec::new(),
        dns_error: None,
        connect: None,
    };

    let addrs: Vec<std::net::SocketAddr> = match lookup {
        Err(_) => {
            report.dns_error = Some("resolution timed out".to_string());
            return Ok(report);
        }
        Ok(Err(e)) => {
            report.dns_error = Some(e.to_string());
            return Ok(report);
        }
        Ok(Ok(iter)) => iter.collect(),
    };
    report.addresses = addrs.iter().map(|a| a.ip().to_string()).collect();
    if addrs.is_empty() {
        report.dns_error = Some("no addresses returned".to_string());
        return Ok(report);
    }

    // TCP connect to the first address, timing the handshake.
    let start = Instant::now();
    let connect = tokio::time::timeout(PROBE_TIMEOUT, TcpStream::connect(addrs[0])).await;
    report.connect = Some(match connect {
        Err(_) => Err("connection timed out".to_string()),
        Ok(Err(e)) => Err(e.to_string()),
        Ok(Ok(_stream)) => Ok(start.elapsed()),
    });
    Ok(report)
}

// ─── cert ─────────────────────────────────────────────────────────────────

/// Result of a TLS certificate inspection.
pub struct CertReport {
    pub host: String,
    pub port: u16,
    pub tls_version: Option<String>,
    pub cipher_suite: Option<String>,
    pub subject: String,
    pub issuer: String,
    pub not_before: Option<chrono::DateTime<chrono::Utc>>,
    pub not_after: Option<chrono::DateTime<chrono::Utc>>,
    pub sans: Vec<String>,
    pub chain_len: usize,
}

impl CertReport {
    /// Days until expiry (negative if already expired), or `None` when the
    /// `notAfter` date couldn't be parsed.
    pub fn days_until_expiry(&self) -> Option<i64> {
        self.not_after
            .map(|exp| (exp - chrono::Utc::now()).num_days())
    }

    pub fn render(&self) -> String {
        let mut s = format!("net cert {}:{}\n", self.host, self.port);
        if let Some(v) = &self.tls_version {
            let cipher = self.cipher_suite.as_deref().unwrap_or("?");
            s.push_str(&format!("  TLS:      {v} ({cipher})\n"));
        }
        s.push_str(&format!("  Subject:  {}\n", self.subject));
        s.push_str(&format!("  Issuer:   {}\n", self.issuer));
        if let Some(nb) = self.not_before {
            s.push_str(&format!(
                "  Valid:    {}\n",
                nb.format("%Y-%m-%d %H:%M UTC")
            ));
        }
        match (self.not_after, self.days_until_expiry()) {
            (Some(na), Some(days)) => {
                let note = if days < 0 {
                    format!("EXPIRED {} days ago", -days)
                } else {
                    format!("{days} days left")
                };
                s.push_str(&format!(
                    "  Expires:  {} ({note})\n",
                    na.format("%Y-%m-%d %H:%M UTC")
                ));
            }
            _ => s.push_str("  Expires:  (unparsed)\n"),
        }
        if !self.sans.is_empty() {
            s.push_str(&format!("  SANs:     {}\n", self.sans.join(", ")));
        }
        s.push_str(&format!("  Chain:    {} certificate(s)", self.chain_len));
        s
    }
}

/// A rustls verifier that accepts every certificate. We are *inspecting*, not
/// trusting: capturing an expired or self-signed chain to report on it is the
/// whole point, so a real verifier would defeat the diagnostic. Trust decisions
/// live elsewhere (reqwest's verified client handles actual egress).
#[derive(Debug)]
struct AcceptAny;

impl tokio_rustls::rustls::client::danger::ServerCertVerifier for AcceptAny {
    fn verify_server_cert(
        &self,
        _end_entity: &tokio_rustls::rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[tokio_rustls::rustls::pki_types::CertificateDer<'_>],
        _server_name: &tokio_rustls::rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: tokio_rustls::rustls::pki_types::UnixTime,
    ) -> Result<tokio_rustls::rustls::client::danger::ServerCertVerified, tokio_rustls::rustls::Error>
    {
        Ok(tokio_rustls::rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &tokio_rustls::rustls::pki_types::CertificateDer<'_>,
        _dss: &tokio_rustls::rustls::DigitallySignedStruct,
    ) -> Result<
        tokio_rustls::rustls::client::danger::HandshakeSignatureValid,
        tokio_rustls::rustls::Error,
    > {
        Ok(tokio_rustls::rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &tokio_rustls::rustls::pki_types::CertificateDer<'_>,
        _dss: &tokio_rustls::rustls::DigitallySignedStruct,
    ) -> Result<
        tokio_rustls::rustls::client::danger::HandshakeSignatureValid,
        tokio_rustls::rustls::Error,
    > {
        Ok(tokio_rustls::rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<tokio_rustls::rustls::SignatureScheme> {
        use tokio_rustls::rustls::SignatureScheme::*;
        vec![
            RSA_PKCS1_SHA256,
            RSA_PKCS1_SHA384,
            RSA_PKCS1_SHA512,
            ECDSA_NISTP256_SHA256,
            ECDSA_NISTP384_SHA384,
            RSA_PSS_SHA256,
            RSA_PSS_SHA384,
            RSA_PSS_SHA512,
            ED25519,
        ]
    }
}

/// Open a TLS connection to `target` and report on the presented certificate.
pub async fn cert(target: &str) -> Result<CertReport, NetError> {
    use tokio_rustls::rustls::{self, pki_types::ServerName};
    use tokio_rustls::TlsConnector;

    let (host, port) = split_host_port(target, DEFAULT_PORT)?;

    let config = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAny))
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));

    let server_name = ServerName::try_from(host.clone())
        .map_err(|_| NetError::msg(format!("invalid server name '{host}'")))?;

    let tcp = tokio::time::timeout(PROBE_TIMEOUT, TcpStream::connect((host.as_str(), port)))
        .await
        .map_err(|_| NetError::msg("connection timed out"))??;

    let tls = tokio::time::timeout(PROBE_TIMEOUT, connector.connect(server_name, tcp))
        .await
        .map_err(|_| NetError::msg("TLS handshake timed out"))?
        .map_err(|e| NetError::msg(format!("TLS handshake failed: {e}")))?;

    let (_io, conn) = tls.get_ref();
    let tls_version = conn.protocol_version().map(|v| format!("{v:?}"));
    let cipher_suite = conn
        .negotiated_cipher_suite()
        .map(|c| format!("{:?}", c.suite()));
    let chain = conn
        .peer_certificates()
        .ok_or_else(|| NetError::msg("server presented no certificate"))?;
    let leaf = chain
        .first()
        .ok_or_else(|| NetError::msg("empty certificate chain"))?;

    let parsed = parse_leaf(leaf.as_ref())?;
    Ok(CertReport {
        host,
        port,
        tls_version,
        cipher_suite,
        subject: parsed.subject,
        issuer: parsed.issuer,
        not_before: parsed.not_before,
        not_after: parsed.not_after,
        sans: parsed.sans,
        chain_len: chain.len(),
    })
}

struct ParsedCert {
    subject: String,
    issuer: String,
    not_before: Option<chrono::DateTime<chrono::Utc>>,
    not_after: Option<chrono::DateTime<chrono::Utc>>,
    sans: Vec<String>,
}

/// Parse the leaf certificate's display fields with `x509-cert`. SAN parsing is
/// best-effort: a malformed extension yields an empty SAN list, never an error,
/// since the validity/subject/issuer are the load-bearing facts.
fn parse_leaf(der: &[u8]) -> Result<ParsedCert, NetError> {
    use x509_cert::der::{oid::AssociatedOid, Decode};
    use x509_cert::ext::pkix::{name::GeneralName, SubjectAltName};
    use x509_cert::Certificate;

    let cert = Certificate::from_der(der)
        .map_err(|e| NetError::msg(format!("certificate parse failed: {e}")))?;
    let tbs = &cert.tbs_certificate;

    let to_utc = |t: &x509_cert::time::Time| -> Option<chrono::DateTime<chrono::Utc>> {
        let secs = t.to_unix_duration().as_secs() as i64;
        chrono::DateTime::from_timestamp(secs, 0)
    };

    let mut sans = Vec::new();
    if let Some(exts) = &tbs.extensions {
        for ext in exts {
            if ext.extn_id == SubjectAltName::OID {
                if let Ok(san) = SubjectAltName::from_der(ext.extn_value.as_bytes()) {
                    for name in &san.0 {
                        if let GeneralName::DnsName(dns) = name {
                            sans.push(dns.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(ParsedCert {
        subject: tbs.subject.to_string(),
        issuer: tbs.issuer.to_string(),
        not_before: to_utc(&tbs.validity.not_before),
        not_after: to_utc(&tbs.validity.not_after),
        sans,
    })
}

// ─── trace ────────────────────────────────────────────────────────────────

/// Result of a traceroute.
pub struct TraceReport {
    pub host: String,
    pub raw: String,
}

impl TraceReport {
    pub fn render(&self) -> String {
        format!("net trace {}\n{}", self.host, self.raw.trim_end())
    }
}

/// Trace the route to `target`. traceroute needs privileged raw sockets, so it
/// runs as a timeout-bounded child process (not inside the exec sandbox, which
/// denies the raw-socket path on both macOS Seatbelt and Linux namespaces — the
/// same reason `brain logs analyze --source system` bypasses it).
#[cfg(unix)]
pub async fn trace(target: &str) -> Result<TraceReport, NetError> {
    // Strip any scheme/port: traceroute wants a bare host.
    let (host, _) = split_host_port(target, DEFAULT_PORT)?;

    // `-m 20` caps hops; `-w 2` caps per-hop wait. A generous outer timeout
    // guards against the whole run hanging.
    let output = tokio::time::timeout(
        Duration::from_secs(40),
        tokio::process::Command::new("traceroute")
            .args(["-m", "20", "-w", "2", &host])
            .output(),
    )
    .await
    .map_err(|_| NetError::msg("traceroute timed out"))?
    .map_err(|e| {
        NetError::msg(format!(
            "could not run `traceroute` (is it installed?): {e}"
        ))
    })?;

    let mut raw = String::from_utf8_lossy(&output.stdout).into_owned();
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let err = err.trim();
        if !err.is_empty() {
            raw.push_str(&format!("\n[traceroute stderr] {err}"));
        }
    }
    if raw.trim().is_empty() {
        return Err(NetError::msg("traceroute produced no output"));
    }
    Ok(TraceReport { host, raw })
}

#[cfg(not(unix))]
pub async fn trace(_target: &str) -> Result<TraceReport, NetError> {
    Err(NetError::msg(
        "`net trace` is only supported on Unix platforms",
    ))
}

// ─── cortex backend wiring ──────────────────────────────────────────────────

/// The [`cortex::actions::NetDiagnosticsBackend`] implementation. Each method
/// runs the matching probe and returns the rendered report (or an error string
/// the dispatcher relays). The chat tool-loop dispatches `net.check/trace/cert`
/// through this; the CLI calls the free functions above directly.
pub struct NetDiagnostics;

#[async_trait::async_trait]
impl cortex::actions::NetDiagnosticsBackend for NetDiagnostics {
    async fn check(&self, target: &str) -> Result<String, cortex::actions::ActionError> {
        check(target)
            .await
            .map(|r| r.render())
            .map_err(to_action_err)
    }

    async fn trace(&self, target: &str) -> Result<String, cortex::actions::ActionError> {
        trace(target)
            .await
            .map(|r| r.render())
            .map_err(to_action_err)
    }

    async fn cert(&self, target: &str) -> Result<String, cortex::actions::ActionError> {
        cert(target)
            .await
            .map(|r| r.render())
            .map_err(to_action_err)
    }
}

fn to_action_err(e: NetError) -> cortex::actions::ActionError {
    cortex::actions::ActionError::ExecutionFailed(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_host_port_handles_bare_host_pair_and_url() {
        assert_eq!(
            split_host_port("example.com", 443).unwrap(),
            ("example.com".to_string(), 443)
        );
        assert_eq!(
            split_host_port("example.com:8443", 443).unwrap(),
            ("example.com".to_string(), 8443)
        );
        assert_eq!(
            split_host_port("https://example.com/path", 443).unwrap(),
            ("example.com".to_string(), 443)
        );
        assert_eq!(
            split_host_port("http://example.com", 443).unwrap(),
            ("example.com".to_string(), 80)
        );
        assert_eq!(
            split_host_port("https://example.com:9000/x", 443).unwrap(),
            ("example.com".to_string(), 9000)
        );
    }

    #[test]
    fn split_host_port_rejects_empty() {
        assert!(split_host_port("   ", 443).is_err());
    }

    #[test]
    fn check_report_renders_reachable_and_unreachable() {
        let reachable = CheckReport {
            host: "h".into(),
            port: 443,
            addresses: vec!["1.2.3.4".into()],
            dns_error: None,
            connect: Some(Ok(Duration::from_millis(12))),
        };
        let r = reachable.render();
        assert!(r.contains("connected in 12 ms"));
        assert!(r.contains("Status:  reachable"));
        assert!(reachable.reachable());

        let dns_fail = CheckReport {
            host: "h".into(),
            port: 443,
            addresses: vec![],
            dns_error: Some("nxdomain".into()),
            connect: None,
        };
        let r = dns_fail.render();
        assert!(r.contains("DNS:     failed — nxdomain"));
        assert!(r.contains("unreachable"));
        assert!(!dns_fail.reachable());
    }

    #[test]
    fn cert_report_flags_expiry() {
        let mut report = CertReport {
            host: "h".into(),
            port: 443,
            tls_version: Some("TLSv1_3".into()),
            cipher_suite: Some("TLS13_AES_128_GCM_SHA256".into()),
            subject: "CN=h".into(),
            issuer: "CN=CA".into(),
            not_before: Some(chrono::Utc::now() - chrono::Duration::days(30)),
            // +1h buffer so the truncating `num_days()` lands on 40 and not 39
            // when `days_until_expiry()` re-reads the clock microseconds later.
            not_after: Some(
                chrono::Utc::now() + chrono::Duration::days(40) + chrono::Duration::hours(1),
            ),
            sans: vec!["h".into(), "www.h".into()],
            chain_len: 2,
        };
        assert_eq!(report.days_until_expiry(), Some(40));
        assert!(report.render().contains("40 days left"));
        assert!(report.render().contains("SANs:     h, www.h"));

        report.not_after = Some(chrono::Utc::now() - chrono::Duration::days(5));
        assert_eq!(report.days_until_expiry(), Some(-5));
        assert!(report.render().contains("EXPIRED 5 days ago"));
    }

    #[test]
    fn trace_report_renders_with_header() {
        let t = TraceReport {
            host: "h".into(),
            raw: "1  gw  1ms\n2  isp  10ms\n".into(),
        };
        let r = t.render();
        assert!(r.starts_with("net trace h\n"));
        assert!(r.contains("gw"));
    }
}
