//! `brain net check/trace/cert` — network diagnostics (Issue 139).
//!
//! Thin CLI surface over the probes in [`backends::net`]: the operator runs
//! them directly (no consent gate — they're invoking the binary themselves),
//! and we print the same `render()` the chat tool-loop relays. The executor,
//! the sandbox rationale for `trace`, and the probe details all live in
//! `backends::net`; this module only parses args and prints reports.

use anyhow::Result;

#[derive(clap::Subcommand)]
pub enum NetAction {
    /// Check reachability: resolve DNS and time a TCP connect to a host[:port].
    ///
    /// Accepts a bare host, `host:port`, or a full URL; defaults to port 443.
    Check {
        /// Target host, `host:port`, or URL.
        target: String,
    },
    /// Trace the network route (hops) to a host.
    ///
    /// Uses the system `traceroute` (Unix only); it needs privileged raw
    /// sockets, so it runs as a bounded child process.
    Trace {
        /// Target host (scheme/port are ignored).
        target: String,
    },
    /// Inspect the TLS certificate chain a host presents.
    ///
    /// Reports subject, issuer, validity window (flagging expiry), and SANs.
    /// Uses a permissive verifier so it can *report* on expired/self-signed
    /// certs rather than fail.
    Cert {
        /// Target host, `host:port`, or URL (defaults to port 443).
        target: String,
    },
}

/// Entry point for `brain net <action>`.
pub async fn cmd_net(action: NetAction) -> Result<()> {
    let report = match action {
        NetAction::Check { target } => backends::net::check(&target).await?.render(),
        NetAction::Trace { target } => backends::net::trace(&target).await?.render(),
        NetAction::Cert { target } => backends::net::cert(&target).await?.render(),
    };
    println!("{report}");
    Ok(())
}
