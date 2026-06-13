use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Everything Brain might want to surface to the user, audit log, or remote consumers.
///
/// Summary types are deliberately string-shaped payload bags so this crate
/// doesn't take a hard dependency on higher-level crates (`brainos-identity`
/// for `Principal`, `brainos-intent` for `IntentToken`, etc.). Tighter
/// payload types can be substituted without changing the variant set.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrainEvent {
    SignalReceived {
        id: Uuid,
        signal: SignalSummary,
        ts: DateTime<Utc>,
    },
    IntentClassified {
        id: Uuid,
        intent: IntentSummary,
        confidence: f32,
        ts: DateTime<Utc>,
    },
    /// Reserved for a future dispatch-time emission point. Currently
    /// not produced by any emitter — kept in the enum so consumers can
    /// already handle it and so the serde wire shape is fixed before
    /// the first emitter lands. Wire it from the dispatch layer
    /// (`pipeline/dispatch.rs`) once route selection is on the
    /// observability pillar's roadmap.
    ToolRouteResolved {
        id: Uuid,
        route: ToolRouteSummary,
        ts: DateTime<Utc>,
    },
    ConfirmationRequested {
        id: Uuid,
        nonce: String,
        reason: String,
        ts: DateTime<Utc>,
    },
    ConfirmationResolved {
        id: Uuid,
        nonce: String,
        decision: String,
        ts: DateTime<Utc>,
    },
    /// Reserved for future emission from the MCP / capability backends.
    /// Pairs with [`BrainEvent::ToolCallFinished`] to bracket a single
    /// tool invocation. Not produced yet — keep matching for forward
    /// compat.
    ToolCallStarted {
        id: Uuid,
        tool_id: String,
        args_redacted: serde_json::Value,
        ts: DateTime<Utc>,
    },
    /// See [`BrainEvent::ToolCallStarted`] — reserved, not yet emitted.
    ToolCallFinished {
        id: Uuid,
        tool_id: String,
        outcome: OutcomeSummary,
        duration_ms: u64,
        ts: DateTime<Utc>,
    },
    ReflexFired {
        id: Uuid,
        trigger_id: String,
        payload: serde_json::Value,
        ts: DateTime<Utc>,
    },
    AuditAppended {
        id: Uuid,
        /// The `AuditEntry.id` (UUID-as-string) — matches the existing audit
        /// schema in `crates/audit`. (Spec §8.1 said `audit_row_id: i64`, but
        /// the audit table is keyed by string UUID, not SQLite ROWID.)
        audit_entry_id: String,
        principal: Option<PrincipalSummary>,
        ts: DateTime<Utc>,
    },
    /// Reserved for emission from the budget enforcement layer when a
    /// watermark (warn / hard) is crossed. Not yet wired — the budget
    /// crate currently signals decisions via return values only.
    BudgetCrossed {
        id: Uuid,
        watermark: f32,
        window: String,
        ts: DateTime<Utc>,
    },
    /// Emitted by the resource sampler when a runtime gauge crosses a configured
    /// ceiling. Edge-triggered — once per crossing, not once per sample — the
    /// same discipline as `BudgetCrossed`. `gauge` and `severity` are strings so
    /// `observe` stays free of an `observability`/config dependency, exactly as
    /// `BreakerStateChange` keeps it free of `resilience`.
    ResourcePressure {
        id: Uuid,
        /// Which gauge crossed: `"rss" | "cpu" | "disk" | "fds"`. (The
        /// SQLite-connection gauge has no configured ceiling, so it never
        /// appears here.)
        gauge: String,
        /// Sampled value at the crossing, in the gauge's threshold unit — MiB
        /// for `rss`/`disk`, percent for `cpu` — so it is directly comparable to
        /// `threshold`.
        value: f64,
        /// The configured ceiling that was crossed, same unit as `value`.
        threshold: f64,
        /// `"warn" | "critical"`.
        severity: String,
        ts: DateTime<Utc>,
    },
    BreakerStateChange {
        id: Uuid,
        tool_id: String,
        from: String,
        to: String,
        ts: DateTime<Utc>,
    },
    Error {
        id: Uuid,
        source: String,
        message: String,
        ts: DateTime<Utc>,
    },
    TerminalSessionOpened {
        id: Uuid,
        session_id: String,
        program: String,
        args: Vec<String>,
        cwd: Option<String>,
        principal: Option<PrincipalSummary>,
        ts: DateTime<Utc>,
    },
    TerminalSessionClosed {
        id: Uuid,
        session_id: String,
        exit_code: i32,
        was_killed: bool,
        principal: Option<PrincipalSummary>,
        ts: DateTime<Utc>,
    },
    /// Orchestrator phase transition. `from` is `"none"` for the initial
    /// entry into `Planning`; subsequent transitions name the prior phase.
    /// The string shape (rather than a typed enum) keeps `observe` free of
    /// an `orchestrate` dep, just like `BreakerStateChange` keeps it free
    /// of `resilience`.
    TaskStateChange {
        id: Uuid,
        task_id: String,
        from: String,
        to: String,
        ts: DateTime<Utc>,
    },
    /// Emitted by the connectivity probe when the kernel's network view
    /// crosses between `online`, `degraded`, and `offline`. Edge-triggered —
    /// once per transition, not once per probe round. `state`/`previous` are
    /// strings so `observe` stays free of a `core` dependency, exactly as
    /// `BreakerStateChange` keeps it free of `resilience`.
    ConnectivityChanged {
        id: Uuid,
        /// New state: `"online" | "degraded" | "offline"`.
        state: String,
        /// State before the transition, same vocabulary.
        previous: String,
        /// Human-readable cause, e.g. `"2 of 3 endpoints unreachable"`.
        detail: String,
        ts: DateTime<Utc>,
    },
    /// Emitted by the power probe when the machine crosses between external
    /// power and battery. Edge-triggered — once per transition, not once per
    /// probe round. `state`/`previous` are strings so `observe` stays free of
    /// a `core` dependency, exactly as `ConnectivityChanged` does.
    PowerStateChanged {
        id: Uuid,
        /// New state: `"external" | "battery"`.
        state: String,
        /// State before the transition, same vocabulary.
        previous: String,
        /// Human-readable cause, e.g. `"battery at 47%"`.
        detail: String,
        ts: DateTime<Utc>,
    },
    /// Emitted by a service-health probe when a monitored endpoint crosses
    /// between reachable and unreachable. Edge-triggered — once per transition,
    /// not once per probe — the same discipline as `ResourcePressure`. `target`
    /// and `detail` are strings so `observe` stays free of a config dependency.
    ServiceHealthChanged {
        id: Uuid,
        /// Configured service label (`monitoring.services[].name`).
        service: String,
        /// The probed endpoint: a URL (`http`) or `host:port` (`tcp`).
        target: String,
        /// `true` when the service just became reachable, `false` when it just
        /// became unreachable.
        healthy: bool,
        /// Human-readable reason for the new state — the failure cause on the
        /// way down, empty on recovery.
        detail: String,
        ts: DateTime<Utc>,
    },
    /// Emitted by the manifest-health sweep when a capability's runtime health
    /// crosses (e.g. verified→degraded when the embedding model goes down, or
    /// →breaker-open). Edge-triggered — once per transition, not once per
    /// sweep. `state`/`previous` are the health-kind strings
    /// (`"verified" | "degraded" | "breaker-open" | "precondition-failed"`) so
    /// `observe` stays free of a `core` dependency, exactly as
    /// `ConnectivityChanged` does.
    CapabilityHealthChanged {
        id: Uuid,
        /// The affected capability's `tool_id` (e.g. `"memory.store"`).
        tool: String,
        /// New health kind.
        state: String,
        /// Health kind before the transition; empty when there was no prior
        /// reading (first sweep / newly registered).
        previous: String,
        /// Human-readable reason for the new state, empty on recovery.
        detail: String,
        ts: DateTime<Utc>,
    },
    /// Emitted by the baseline backend when a `baseline.diff` run detects
    /// drift (a no-drift diff emits nothing). Labels are the same
    /// human-readable strings the rendered diff uses, so `observe` stays free
    /// of a `backends` dependency, exactly as `BreakerStateChange` keeps it
    /// free of `resilience`.
    BaselineDrift {
        id: Uuid,
        /// Comparison base, e.g. `"baseline v3 (2026-06-12 09:14)"`.
        from: String,
        /// Comparison target, e.g. `"current live state"` or another
        /// stored baseline label.
        to: String,
        /// Count of facts present in `to` but not `from`.
        added: u64,
        /// Count of facts present in `from` but not `to`.
        removed: u64,
        /// Count of facts whose value differs between the two.
        changed: u64,
        /// The affected fact keys, capped by the emitter to keep the
        /// event compact.
        keys: Vec<String>,
        ts: DateTime<Utc>,
    },
}

impl BrainEvent {
    /// The variant discriminant as a lowercase string. Used by SSE/WS filters
    /// (`?kind=tool_call_started`) without re-serialising the event.
    pub fn kind(&self) -> &'static str {
        match self {
            BrainEvent::SignalReceived { .. } => "signal_received",
            BrainEvent::IntentClassified { .. } => "intent_classified",
            BrainEvent::ToolRouteResolved { .. } => "tool_route_resolved",
            BrainEvent::ConfirmationRequested { .. } => "confirmation_requested",
            BrainEvent::ConfirmationResolved { .. } => "confirmation_resolved",
            BrainEvent::ToolCallStarted { .. } => "tool_call_started",
            BrainEvent::ToolCallFinished { .. } => "tool_call_finished",
            BrainEvent::ReflexFired { .. } => "reflex_fired",
            BrainEvent::AuditAppended { .. } => "audit_appended",
            BrainEvent::BudgetCrossed { .. } => "budget_crossed",
            BrainEvent::ResourcePressure { .. } => "resource_pressure",
            BrainEvent::BreakerStateChange { .. } => "breaker_state_change",
            BrainEvent::Error { .. } => "error",
            BrainEvent::TerminalSessionOpened { .. } => "terminal_session_opened",
            BrainEvent::TerminalSessionClosed { .. } => "terminal_session_closed",
            BrainEvent::TaskStateChange { .. } => "task_state_change",
            BrainEvent::ConnectivityChanged { .. } => "connectivity_changed",
            BrainEvent::PowerStateChanged { .. } => "power_state_changed",
            BrainEvent::ServiceHealthChanged { .. } => "service_health_changed",
            BrainEvent::CapabilityHealthChanged { .. } => "capability_health_changed",
            BrainEvent::BaselineDrift { .. } => "baseline_drift",
        }
    }

    /// The event correlation id. Multiple events sharing this id belong to one signal flow.
    pub fn id(&self) -> Uuid {
        match self {
            BrainEvent::SignalReceived { id, .. }
            | BrainEvent::IntentClassified { id, .. }
            | BrainEvent::ToolRouteResolved { id, .. }
            | BrainEvent::ConfirmationRequested { id, .. }
            | BrainEvent::ConfirmationResolved { id, .. }
            | BrainEvent::ToolCallStarted { id, .. }
            | BrainEvent::ToolCallFinished { id, .. }
            | BrainEvent::ReflexFired { id, .. }
            | BrainEvent::AuditAppended { id, .. }
            | BrainEvent::BudgetCrossed { id, .. }
            | BrainEvent::ResourcePressure { id, .. }
            | BrainEvent::BreakerStateChange { id, .. }
            | BrainEvent::Error { id, .. }
            | BrainEvent::TerminalSessionOpened { id, .. }
            | BrainEvent::TerminalSessionClosed { id, .. }
            | BrainEvent::TaskStateChange { id, .. }
            | BrainEvent::ConnectivityChanged { id, .. }
            | BrainEvent::PowerStateChanged { id, .. }
            | BrainEvent::ServiceHealthChanged { id, .. }
            | BrainEvent::CapabilityHealthChanged { id, .. }
            | BrainEvent::BaselineDrift { id, .. } => *id,
        }
    }

    /// Optional `tool_id` filter target; `None` for events not associated with a tool.
    pub fn tool_id(&self) -> Option<&str> {
        match self {
            BrainEvent::ToolCallStarted { tool_id, .. }
            | BrainEvent::ToolCallFinished { tool_id, .. }
            | BrainEvent::BreakerStateChange { tool_id, .. } => Some(tool_id.as_str()),
            BrainEvent::CapabilityHealthChanged { tool, .. } => Some(tool.as_str()),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignalSummary {
    pub source: String,
    pub channel: String,
    pub sender: String,
    pub namespace: String,
    pub content_preview: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IntentSummary {
    pub kind: String,
    pub args_redacted: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolRouteSummary {
    pub tool_id: String,
    pub source: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutcomeSummary {
    pub status: String,
    pub error: Option<String>,
}

/// Loose summary view of `brainos_identity::Principal`. Kept summary-shaped
/// here so a tighter typed payload is a swap, not a variant rename, and so
/// `observe` doesn't depend on `identity`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PrincipalSummary {
    pub user_id: String,
    pub agent_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_strings_are_snake_case() {
        let ev = BrainEvent::Error {
            id: Uuid::nil(),
            source: "test".into(),
            message: "m".into(),
            ts: Utc::now(),
        };
        assert_eq!(ev.kind(), "error");
    }

    #[test]
    fn roundtrip_tool_call_started_through_json() {
        let id = Uuid::new_v4();
        let ts = Utc::now();
        let original = BrainEvent::ToolCallStarted {
            id,
            tool_id: "mcp:fs:read".into(),
            args_redacted: serde_json::json!({"path": "/tmp/x"}),
            ts,
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "tool_call_started");
        assert_eq!(decoded.id(), id);
        assert_eq!(decoded.tool_id(), Some("mcp:fs:read"));
    }

    #[test]
    fn id_accessor_returns_per_variant_id() {
        let id = Uuid::new_v4();
        let ts = Utc::now();
        let ev = BrainEvent::BudgetCrossed {
            id,
            watermark: 0.75,
            window: "daily".into(),
            ts,
        };
        assert_eq!(ev.id(), id);
        assert_eq!(ev.tool_id(), None);
    }

    #[test]
    fn roundtrip_resource_pressure_through_json() {
        let id = Uuid::new_v4();
        let ts = Utc::now();
        let original = BrainEvent::ResourcePressure {
            id,
            gauge: "rss".into(),
            value: 2304.0,
            threshold: 2048.0,
            severity: "warn".into(),
            ts,
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "resource_pressure");
        assert_eq!(decoded.id(), id);
        // Not tool-scoped, so it never matches a `?tool_id=` filter.
        assert_eq!(decoded.tool_id(), None);
        match decoded {
            BrainEvent::ResourcePressure {
                gauge,
                value,
                threshold,
                severity,
                ..
            } => {
                assert_eq!(gauge, "rss");
                assert_eq!(value, 2304.0);
                assert_eq!(threshold, 2048.0);
                assert_eq!(severity, "warn");
            }
            other => panic!("decoded to the wrong variant: {other:?}"),
        }
    }

    #[test]
    fn roundtrip_connectivity_changed_through_json() {
        let id = Uuid::new_v4();
        let original = BrainEvent::ConnectivityChanged {
            id,
            state: "offline".into(),
            previous: "online".into(),
            detail: "2 of 2 endpoints unreachable".into(),
            ts: Utc::now(),
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "connectivity_changed");
        assert_eq!(decoded.id(), id);
        assert_eq!(decoded.tool_id(), None);
        match decoded {
            BrainEvent::ConnectivityChanged {
                state,
                previous,
                detail,
                ..
            } => {
                assert_eq!(state, "offline");
                assert_eq!(previous, "online");
                assert_eq!(detail, "2 of 2 endpoints unreachable");
            }
            other => panic!("decoded to the wrong variant: {other:?}"),
        }
    }

    #[test]
    fn roundtrip_power_state_changed_through_json() {
        let id = Uuid::new_v4();
        let original = BrainEvent::PowerStateChanged {
            id,
            state: "battery".into(),
            previous: "external".into(),
            detail: "battery at 47%".into(),
            ts: Utc::now(),
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "power_state_changed");
        assert_eq!(decoded.id(), id);
        assert_eq!(decoded.tool_id(), None);
        match decoded {
            BrainEvent::PowerStateChanged {
                state,
                previous,
                detail,
                ..
            } => {
                assert_eq!(state, "battery");
                assert_eq!(previous, "external");
                assert_eq!(detail, "battery at 47%");
            }
            other => panic!("decoded to the wrong variant: {other:?}"),
        }
    }

    #[test]
    fn roundtrip_baseline_drift_through_json() {
        let id = Uuid::new_v4();
        let original = BrainEvent::BaselineDrift {
            id,
            from: "baseline v3 (2026-06-12 09:14)".into(),
            to: "current live state".into(),
            added: 2,
            removed: 0,
            changed: 1,
            keys: vec!["llm.model".into(), "adapter.http".into()],
            ts: Utc::now(),
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "baseline_drift");
        assert_eq!(decoded.id(), id);
        assert_eq!(decoded.tool_id(), None);
        match decoded {
            BrainEvent::BaselineDrift {
                added,
                removed,
                changed,
                keys,
                ..
            } => {
                assert_eq!((added, removed, changed), (2, 0, 1));
                assert_eq!(keys.len(), 2);
            }
            other => panic!("decoded to the wrong variant: {other:?}"),
        }
    }

    #[test]
    fn roundtrip_service_health_changed_through_json() {
        let id = Uuid::new_v4();
        let ts = Utc::now();
        let original = BrainEvent::ServiceHealthChanged {
            id,
            service: "ollama".into(),
            target: "http://localhost:11434/api/tags".into(),
            healthy: false,
            detail: "connection refused".into(),
            ts,
        };

        let json = serde_json::to_string(&original).unwrap();
        let decoded: BrainEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.kind(), "service_health_changed");
        assert_eq!(decoded.id(), id);
        // Not tool-scoped, so it never matches a `?tool_id=` filter.
        assert_eq!(decoded.tool_id(), None);
        match decoded {
            BrainEvent::ServiceHealthChanged {
                service,
                target,
                healthy,
                detail,
                ..
            } => {
                assert_eq!(service, "ollama");
                assert_eq!(target, "http://localhost:11434/api/tags");
                assert!(!healthy);
                assert_eq!(detail, "connection refused");
            }
            other => panic!("decoded to the wrong variant: {other:?}"),
        }
    }
}
