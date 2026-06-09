//! Canonical verb vocabulary for Brain's capability kernel.
//!
//! Every verb that flows through identity authorization, confirmation
//! gating, and intent routing must appear here. A verb is the
//! `(namespace, action)` pair Brain uses to address a capability — e.g.
//! `memory.store`, `shell.exec`, `mcp.mount`. Verbs are part of the
//! kernel's stable contract: adding one is a code change (new
//! [`crate::IntentToken`] consumer, new authz mapping in
//! `signal::authz::intent_to_auth`, new pipeline handler), not a config
//! edit. External plugins extend Brain by mounting MCP servers whose
//! tools land at `mcp.{tool_name}` — they do not invent new verbs.
//!
//! ## Why constants instead of `verbs.toml`
//!
//! The v1.0.0 RFC §158 nominally called for a `verbs.toml` registry +
//! startup validation that every authz-mapped verb appears in it. We
//! ship the same guarantee at *compile time* instead — `VERBS` is a
//! `&[VerbSpec]` and `signal::authz::tests::every_static_verb_is_in_registry`
//! asserts that every typed-Intent → AuthorizationRequest mapping
//! resolves through [`lookup`]. A typo in `intent_to_auth` therefore
//! fails the test suite rather than the runtime. Operators do not need
//! to edit a verb file because verbs are kernel surface, not
//! configuration.

/// Documentation hint for a verb's confirmation tier. The authoritative
/// tier mapping lives in `signal::authz::tier_for_verb`; this enum
/// mirrors the five `identity::Tier` variants without depending on the
/// `identity` crate so this module stays dependency-light. The
/// `signal::authz::tests::tier_hints_match_authz` cross-check asserts
/// the two stay in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierHint {
    Read,
    Write,
    Execute,
    Destructive,
    External,
}

impl TierHint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Destructive => "destructive",
            Self::External => "external",
        }
    }
}

/// A single capability verb. The `tier_hint` is the conservative
/// confirmation tier the kernel applies if the typed-Intent path
/// doesn't override it; it matches `signal::authz::tier_for_verb`'s
/// fallback for the same `(ns, action)` pair.
#[derive(Debug, Clone, Copy)]
pub struct VerbSpec {
    pub ns: &'static str,
    pub action: &'static str,
    pub tier_hint: TierHint,
    pub summary: &'static str,
}

impl VerbSpec {
    /// Dotted-string rendering for log/audit lines (`memory.store`).
    pub fn dotted(&self) -> String {
        format!("{}.{}", self.ns, self.action)
    }
}

/// The kernel's full verb vocabulary as of v0.4.0.
///
/// Adding a new verb requires:
/// 1. Appending an entry here with a non-empty `summary`.
/// 2. Mapping the producing [`crate::Intent`] variant in
///    `signal::authz::intent_to_auth` (or extending `tier_for_verb`
///    if the verb arrives via `Intent::ToolCall`).
/// 3. Routing the variant to a handler in `signal::pipeline`.
///
/// The unique-pair invariant is enforced by [`Self::check_invariants`]
/// at test time, not by a HashMap — the list is small enough that
/// linear scan during boot is cheaper than a static-init map.
pub const VERBS: &[VerbSpec] = &[
    // ── Memory ────────────────────────────────────────────────────────
    VerbSpec {
        ns: "memory",
        action: "store",
        tier_hint: TierHint::Write,
        summary: "Store a subject-predicate-object fact in semantic memory.",
    },
    VerbSpec {
        ns: "memory",
        action: "delete",
        tier_hint: TierHint::Destructive,
        summary: "Delete matching semantic facts (irreversible).",
    },
    VerbSpec {
        ns: "memory",
        action: "import",
        tier_hint: TierHint::Write,
        summary: "Bulk-import memory entries from a filesystem path.",
    },
    VerbSpec {
        ns: "memory",
        action: "export",
        tier_hint: TierHint::Write,
        summary: "Bulk-export memory entries to a filesystem path.",
    },
    // ── Shell / Terminal ─────────────────────────────────────────────
    VerbSpec {
        ns: "shell",
        action: "exec",
        tier_hint: TierHint::Execute,
        summary: "Run a one-shot sandboxed command via the action dispatcher.",
    },
    VerbSpec {
        ns: "terminal",
        action: "open",
        tier_hint: TierHint::Execute,
        summary: "Open a persistent PTY session through the terminal bridge.",
    },
    VerbSpec {
        ns: "terminal",
        action: "close",
        tier_hint: TierHint::Write,
        summary: "Close a previously-opened PTY session.",
    },
    // ── Network egress ────────────────────────────────────────────────
    VerbSpec {
        ns: "net",
        action: "http",
        tier_hint: TierHint::External,
        summary: "Perform an outbound HTTP request (web search, fetch).",
    },
    VerbSpec {
        ns: "net",
        action: "check",
        tier_hint: TierHint::External,
        summary: "Check reachability of a host: DNS resolution + timed TCP connect.",
    },
    VerbSpec {
        ns: "net",
        action: "trace",
        tier_hint: TierHint::External,
        summary: "Trace the network route (hops) to a host.",
    },
    VerbSpec {
        ns: "net",
        action: "cert",
        tier_hint: TierHint::External,
        summary: "Inspect the TLS certificate chain a host presents (validity, SANs).",
    },
    VerbSpec {
        ns: "notify",
        action: "send",
        tier_hint: TierHint::External,
        summary: "Send a message through the channel dispatcher.",
    },
    // ── Scheduling / tasks ───────────────────────────────────────────
    VerbSpec {
        // Gates up-front (fires later, possibly unattended) but reversible via
        // schedule.cancel — External, not Destructive/Write. Must stay in sync
        // with `signal::authz::tier_for_verb` and `pipeline/lifecycle.rs`.
        ns: "schedule",
        action: "create",
        tier_hint: TierHint::External,
        summary: "Persist a scheduled-intent row.",
    },
    VerbSpec {
        ns: "schedule",
        action: "cancel",
        tier_hint: TierHint::Write,
        summary: "Cancel a scheduled-intent row.",
    },
    VerbSpec {
        ns: "task",
        action: "decompose",
        tier_hint: TierHint::Execute,
        summary: "Decompose a request into an orchestrator task plan.",
    },
    VerbSpec {
        ns: "task",
        action: "cancel",
        tier_hint: TierHint::Write,
        summary: "Abort a running orchestrator task.",
    },
    VerbSpec {
        ns: "signal",
        action: "cancel",
        tier_hint: TierHint::Write,
        summary: "Cancel an in-flight signal pipeline run.",
    },
    // ── Agents ────────────────────────────────────────────────────────
    VerbSpec {
        ns: "agent",
        action: "delegate",
        tier_hint: TierHint::Execute,
        summary: "Hand a subtask to a discovered subprocess agent.",
    },
    // ── Approvals / audit ─────────────────────────────────────────────
    VerbSpec {
        ns: "approval",
        action: "respond",
        tier_hint: TierHint::Write,
        summary: "Approve or reject a pending confirmation by nonce.",
    },
    VerbSpec {
        ns: "approval",
        action: "revoke",
        tier_hint: TierHint::Write,
        summary: "Revoke a standing approval grant.",
    },
    VerbSpec {
        ns: "audit",
        action: "prune",
        tier_hint: TierHint::Destructive,
        summary: "Prune audit-log rows older than the given duration.",
    },
    VerbSpec {
        ns: "security",
        action: "audit",
        tier_hint: TierHint::Read,
        summary: "Audit the security posture of the current configuration.",
    },
    // ── System inspection (logs + baselines) ──────────────────────────
    VerbSpec {
        ns: "logs",
        action: "analyze",
        tier_hint: TierHint::Read,
        summary: "Analyse recent logs for recurring error/warning patterns.",
    },
    VerbSpec {
        ns: "baseline",
        action: "capture",
        tier_hint: TierHint::Write,
        summary: "Capture a system-baseline snapshot for later drift detection.",
    },
    VerbSpec {
        ns: "baseline",
        action: "diff",
        tier_hint: TierHint::Read,
        summary: "Diff a baseline against another snapshot or the live state.",
    },
    VerbSpec {
        ns: "baseline",
        action: "list",
        tier_hint: TierHint::Read,
        summary: "List stored baseline snapshots.",
    },
    // ── Channel / preferences ─────────────────────────────────────────
    VerbSpec {
        ns: "channel",
        action: "configure",
        tier_hint: TierHint::Write,
        summary: "Adjust a channel-preference weight or pin.",
    },
    VerbSpec {
        ns: "proactivity",
        action: "configure",
        tier_hint: TierHint::Write,
        summary: "Toggle or rate-limit the proactivity engine.",
    },
    // ── Filesystem (path-scoped) ──────────────────────────────────────
    VerbSpec {
        ns: "fs",
        action: "read",
        tier_hint: TierHint::Read,
        summary: "Read a filesystem path inside security.allowed_paths.",
    },
    // ── MCP host ──────────────────────────────────────────────────────
    VerbSpec {
        ns: "mcp",
        action: "mount",
        tier_hint: TierHint::External,
        summary: "Mount an external MCP server (stdio or HTTP transport).",
    },
    VerbSpec {
        ns: "mcp",
        action: "unmount",
        tier_hint: TierHint::Write,
        summary: "Unmount a previously-mounted MCP server.",
    },
];

/// Look up a verb by namespace + action. Returns `None` for verbs not
/// in the kernel vocabulary — callers should treat this as a hard
/// invariant violation for typed Intent variants, and as expected for
/// dynamically-discovered MCP tool verbs (`mcp.{tool_name}` is *not*
/// in [`VERBS`]; only the bare `mcp.mount` / `mcp.unmount` host-control
/// verbs are).
pub fn lookup(ns: &str, action: &str) -> Option<&'static VerbSpec> {
    VERBS.iter().find(|v| v.ns == ns && v.action == action)
}

/// Every distinct namespace currently in the vocabulary, deduped.
/// Useful for telemetry and startup banners.
pub fn namespaces() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = VERBS.iter().map(|v| v.ns).collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Every `(ns, action)` pair in [`VERBS`] must be unique. Linear
    /// scan during lookup is fine; silent duplicates would mask routing
    /// bugs.
    #[test]
    fn verb_pairs_are_unique() {
        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for v in VERBS {
            assert!(
                seen.insert((v.ns, v.action)),
                "duplicate verb pair: {}.{}",
                v.ns,
                v.action
            );
        }
    }

    /// Every entry must carry a non-empty human summary; the summary
    /// is what `brain doctor` / observability surfaces show.
    #[test]
    fn every_verb_has_a_summary() {
        for v in VERBS {
            assert!(
                !v.summary.trim().is_empty(),
                "{}.{} has empty summary",
                v.ns,
                v.action
            );
        }
    }

    #[test]
    fn lookup_finds_known_pair() {
        let v = lookup("memory", "store").expect("memory.store registered");
        assert_eq!(v.tier_hint, TierHint::Write);
        assert_eq!(v.dotted(), "memory.store");
    }

    #[test]
    fn tier_hint_string_form_is_stable() {
        assert_eq!(TierHint::Read.as_str(), "read");
        assert_eq!(TierHint::Destructive.as_str(), "destructive");
    }

    #[test]
    fn lookup_returns_none_for_unknown_pair() {
        assert!(lookup("not", "real").is_none());
        // mcp.{tool_name} is dynamic — only mount/unmount are static.
        assert!(lookup("mcp", "any_tool_name").is_none());
    }

    #[test]
    fn namespaces_are_deduped_and_sorted() {
        let ns = namespaces();
        let mut sorted = ns.clone();
        sorted.sort_unstable();
        assert_eq!(ns, sorted);
        let mut deduped = ns.clone();
        deduped.dedup();
        assert_eq!(ns, deduped);
        assert!(ns.contains(&"memory"));
        assert!(ns.contains(&"mcp"));
    }

    #[test]
    fn vocabulary_contains_expected_size() {
        // Spot-check: 23 verbs through v0.4.0, +3 net diagnostics
        // (net.check/trace/cert), +1 security.audit, +1 logs.analyze,
        // +3 baseline.* (capture/diff/list) = 31. Bumping this intentionally
        // is fine — the test exists so silent additions (or removals)
        // surface in review.
        assert_eq!(VERBS.len(), 31);
    }
}
