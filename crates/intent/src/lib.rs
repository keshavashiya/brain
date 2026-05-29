//! # Brain Intent — Standardized Intent Token (SIT) schema
//!
//! Provides the universal envelope (`IntentToken`) that classifiers emit and
//! routers resolve into concrete tool invocations. The schema is wire-stable:
//! the [`SCHEMA`] constant identifies the version of the envelope shape.
//!
//! This crate is intentionally dependency-light — it defines data types and
//! trait signatures only. Concrete classifier, router, registry and index
//! implementations live in higher-level crates that compose this schema.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Wire identifier for the current SIT envelope. Bumped on any
/// breaking change to [`IntentToken`].
pub const SCHEMA: &str = "intent-token/1";

// ─── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum IntentError {
    #[error("unknown verb: {0}.{1}")]
    UnknownVerb(String, String),

    #[error("missing capability: {0}")]
    MissingCapability(String),

    #[error("constraint failed: {0}")]
    ConstraintFailed(String),

    #[error("schema mismatch: {0}")]
    Schema(String),

    #[error("unknown tool: {0}")]
    UnknownTool(String),

    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

// ─── Intent token ───────────────────────────────────────────────────────────

/// The universal envelope that flows from a classifier through a router into
/// an executor. `verb` + `object` describe *what* the caller wants;
/// `required_capabilities` and `constraints` describe *under what conditions*
/// the action may run; `provenance` records *who asked*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentToken {
    pub schema: String,
    pub id: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<Uuid>,
    pub verb: Verb,
    pub object: Object,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub modifiers: BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_capabilities: Vec<String>,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<Constraint>,
    pub namespace: String,
}

impl IntentToken {
    /// Build a new token with a fresh id and the current schema string.
    pub fn new(verb: Verb, object: Object, provenance: Provenance, namespace: String) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            id: Uuid::new_v4(),
            parent_id: None,
            verb,
            object,
            modifiers: BTreeMap::new(),
            required_capabilities: Vec::new(),
            provenance,
            confidence: None,
            constraints: Vec::new(),
            namespace,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Verb {
    pub namespace: String,
    pub action: String,
}

impl Verb {
    pub fn new(namespace: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            action: action.into(),
        }
    }

    /// Dotted-string rendering (`memory.store`) for manifests and logs.
    pub fn dotted(&self) -> String {
        format!("{}.{}", self.namespace, self.action)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Object {
    pub kind: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum Provenance {
    User {
        raw_input: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ui_origin: Option<String>,
        ts: DateTime<Utc>,
    },
    Llm {
        model: String,
        call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_input: Option<String>,
        ts: DateTime<Utc>,
    },
    Reflex {
        trigger: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_input: Option<String>,
        ts: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Constraint {
    PathExists {
        path: String,
    },
    NetReachable {
        host: String,
        port: u16,
    },
    EnvSet {
        name: String,
    },
    UserPresent,
    Custom {
        name: String,
        args: serde_json::Value,
    },
}

// ─── Tool / route types ─────────────────────────────────────────────────────

/// Opaque identifier for a native action backend (`memory`, `web_search`, …).
/// Kept as a newtype to keep `ToolRoute` self-describing without forcing the
/// schema crate to know about every backend in the workspace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub String);

impl BackendId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolSource {
    McpServer { server: String },
    NativeBackend { backend: BackendId },
    Terminal,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolAnnotations {
    #[serde(default)]
    pub read_only_hint: bool,
    #[serde(default)]
    pub destructive_hint: bool,
    #[serde(default)]
    pub idempotent_hint: bool,
}

/// Reasoner-facing usage guidance for a capability. This is
/// what lets a *planner* or the chat reasoner choose the right tool instead
/// of guessing from a one-line description: when it applies, when it does
/// not, what must be true first, roughly what it costs, and the safety
/// `tier` the action carries (awareness ≠ permission — execution is still
/// gated by the consent/audit path; this string is purely descriptive).
///
/// Every field is optional so the struct is backward-compatible on the wire
/// (`#[serde(default)]` on the `ToolDescriptor::usage` field) and cheap to
/// fill incrementally — MCP-sourced tools may carry none of it, a
/// hand-authored native backend descriptor may carry all of it.
///
/// Free-text fields sourced from an untrusted MCP server are subject to the
/// same injection caveat as [`ToolDescriptor::description`]; render them
/// through [`sanitization`] before they reach a system prompt.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_to_use: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when_not_to: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preconditions: Vec<String>,
    /// Coarse cost hint — e.g. "free / local", "network call", "LLM tokens".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<String>,
    /// A short example invocation or phrasing that triggers this capability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example: Option<String>,
    /// Safety tier as a string (`"read"`, `"write"`, `"execute"`,
    /// `"external"`, `"destructive"`). Kept as a plain string so the schema
    /// crate stays free of an `identity` dependency; registration sites
    /// stamp it from the known [`Tier`](identity) equivalent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
}

impl ToolUsage {
    /// True when no guidance has been supplied — lets producers skip the
    /// field on the wire so MCP-sourced tools that carry none stay tidy.
    pub fn is_empty(&self) -> bool {
        self.when_to_use.is_none()
            && self.when_not_to.is_none()
            && self.preconditions.is_empty()
            && self.cost.is_none()
            && self.example.is_none()
            && self.tier.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub tool_id: String,
    pub source: ToolSource,
    pub verb: Verb,
    /// **UNTRUSTED.** When sourced from an MCP server this string is
    /// attacker-controllable (CVE-2025-54136 / "MCPoison" class — a
    /// hostile server can ship a description crafted to inject
    /// instructions into the user's LLM context). Never inline this
    /// directly into a system prompt; route it through
    /// [`sanitization::render_tool_description_for_prompt`] which
    /// strips control bytes, caps length, and fences the body inside
    /// a labeled untrusted block. The hash-pin layer in
    /// `brainos-mcphost` detects rug-pull *changes* to this field;
    /// the sanitizer is what stops a single hostile description from
    /// landing as live system instructions.
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub annotations: ToolAnnotations,
    /// Reasoner-facing usage guidance. Optional + defaulted
    /// so existing producers/consumers stay wire-compatible.
    #[serde(default, skip_serializing_if = "ToolUsage::is_empty")]
    pub usage: ToolUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolRoute {
    Mcp {
        server: String,
        tool: String,
    },
    NativeBackend {
        backend: BackendId,
    },
    Terminal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_hint: Option<String>,
    },
    HumanConfirm {
        ask: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredTool {
    pub tool: ToolDescriptor,
    pub score: f32,
}

// ─── Traits ─────────────────────────────────────────────────────────────────

/// Resolves a token into a concrete invocation target. Implementations score
/// candidates from a [`CapabilityIndex`] and return the best route.
#[async_trait]
pub trait IntentRouter: Send + Sync {
    async fn resolve(&self, tok: &IntentToken) -> Result<ToolRoute, IntentError>;
}

/// Authoritative catalog of every tool the host can dispatch to. The MCP host
/// and native backends register here on mount; the router queries here when
/// resolving a token.
#[async_trait]
pub trait ToolRegistry: Send + Sync {
    async fn register(&self, descriptor: ToolDescriptor) -> Result<(), IntentError>;
    async fn deregister(&self, tool_id: &str) -> Result<(), IntentError>;
    async fn list(&self) -> Vec<ToolDescriptor>;
    async fn get(&self, tool_id: &str) -> Option<ToolDescriptor>;
}

/// Semantic top-k search over the registered tools. The default scoring
/// strategy combines verb match, capability overlap, and embedding similarity
/// against the token's surface form.
#[async_trait]
pub trait CapabilityIndex: Send + Sync {
    async fn search(&self, q: &str, caps: &[String], k: usize) -> Vec<ScoredTool>;
    async fn upsert(&self, t: &ToolDescriptor) -> Result<(), IntentError>;
}

/// Per-tool circuit-breaker snapshot. The router calls this for every
/// candidate during [`IntentRouter::resolve`]; tools whose breaker is
/// `Open` are excluded from scoring so a recently-failing tool can't be
/// chosen until the cooldown elapses. The concrete breaker lives in
/// `brainos-resilience`; the trait keeps the schema crate independent
/// of the resilience layer.
#[async_trait]
pub trait BreakerCheck: Send + Sync {
    async fn is_open(&self, tool_id: &str) -> bool;
}

// ─── Default implementations ────────────────────────────────────────────────

/// Default [`IntentRouter`] that scores registered tools against a token and
/// returns the highest-ranked candidate's route. Pure-data routing — no
/// embeddings (the semantic [`CapabilityIndex`] lands in a separate slice);
/// this router relies on verb match plus capability overlap.
///
/// Scoring (higher is better):
/// - exact verb match (namespace + action) → `+2.0`
/// - namespace-only match → `+1.0`
/// - MCP coarse-verb fallback (`mcp.<tool_name>` matches the SIT's action) → `+0.5`
/// - Jaccard overlap of `required_capabilities` × `tool.capabilities` → `× 1.5`
///
/// When no tool clears `0.0`, the router emits
/// `ToolRoute::HumanConfirm { ask }` describing the unresolved verb — the
/// pipeline surfaces that back to the user instead of guessing.
pub struct DefaultIntentRouter {
    registry: Arc<dyn ToolRegistry>,
    breakers: Option<Arc<dyn BreakerCheck>>,
}

impl DefaultIntentRouter {
    pub fn new(registry: Arc<dyn ToolRegistry>) -> Self {
        Self {
            registry,
            breakers: None,
        }
    }

    /// Wire a [`BreakerCheck`] so `Open` tools are excluded from scoring.
    /// Without one, every registered tool is considered.
    pub fn with_breakers(mut self, breakers: Arc<dyn BreakerCheck>) -> Self {
        self.breakers = Some(breakers);
        self
    }

    /// Score a single candidate against the token. Public so callers /
    /// tests can probe the ranking without invoking `resolve`.
    pub fn score(tok: &IntentToken, tool: &ToolDescriptor) -> f32 {
        let mut score = 0.0_f32;
        if tok.verb == tool.verb {
            score += 2.0;
        } else if tok.verb.namespace == tool.verb.namespace {
            score += 1.0;
        }
        if tool.verb.namespace == "mcp" && tool.verb.action == tok.verb.action {
            score += 0.5;
        }
        score += jaccard(&tok.required_capabilities, &tool.capabilities) * 1.5;
        score
    }
}

#[async_trait]
impl IntentRouter for DefaultIntentRouter {
    async fn resolve(&self, tok: &IntentToken) -> Result<ToolRoute, IntentError> {
        let tools = self.registry.list().await;
        let mut best: Option<(ToolDescriptor, f32)> = None;
        for t in tools {
            if let Some(breakers) = &self.breakers {
                if breakers.is_open(&t.tool_id).await {
                    continue;
                }
            }
            let s = Self::score(tok, &t);
            match best {
                None if s > 0.0 => best = Some((t, s)),
                Some((_, b)) if s > b => best = Some((t, s)),
                _ => {}
            }
        }
        Ok(match best {
            Some((tool, _)) => route_for(&tool),
            None => ToolRoute::HumanConfirm {
                ask: format!(
                    "No tool registered for verb '{}.{}' — review the capability registry or add a matching backend.",
                    tok.verb.namespace, tok.verb.action,
                ),
            },
        })
    }
}

fn route_for(tool: &ToolDescriptor) -> ToolRoute {
    match &tool.source {
        ToolSource::McpServer { server } => ToolRoute::Mcp {
            server: server.clone(),
            // MCP tools registered by `mcphost` stamp the wire tool name into
            // the verb's action slot, so the canonical handle stays
            // round-trippable without re-parsing `tool_id`.
            tool: tool.verb.action.clone(),
        },
        ToolSource::NativeBackend { backend } => ToolRoute::NativeBackend {
            backend: backend.clone(),
        },
        ToolSource::Terminal => ToolRoute::Terminal { session_hint: None },
    }
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let set_a: std::collections::HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(String::as_str).collect();
    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

// ─── In-memory tool registry ────────────────────────────────────────────────

/// In-memory [`ToolRegistry`] backed by a `RwLock<HashMap>`. The default
/// registry the MCP host and native backends register into on mount; the
/// router queries this when resolving an [`IntentToken`].
#[derive(Default)]
pub struct InMemoryToolRegistry {
    tools: RwLock<HashMap<String, ToolDescriptor>>,
}

impl InMemoryToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of currently-registered tools. Useful for tests and metrics.
    pub fn len(&self) -> usize {
        self.tools.read().expect("registry lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools
            .read()
            .expect("registry lock poisoned")
            .is_empty()
    }
}

#[async_trait]
impl ToolRegistry for InMemoryToolRegistry {
    async fn register(&self, descriptor: ToolDescriptor) -> Result<(), IntentError> {
        let mut tools = self.tools.write().expect("registry lock poisoned");
        tools.insert(descriptor.tool_id.clone(), descriptor);
        Ok(())
    }

    async fn deregister(&self, tool_id: &str) -> Result<(), IntentError> {
        let mut tools = self.tools.write().expect("registry lock poisoned");
        if tools.remove(tool_id).is_none() {
            return Err(IntentError::UnknownTool(tool_id.to_string()));
        }
        Ok(())
    }

    async fn list(&self) -> Vec<ToolDescriptor> {
        self.tools
            .read()
            .expect("registry lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    async fn get(&self, tool_id: &str) -> Option<ToolDescriptor> {
        self.tools
            .read()
            .expect("registry lock poisoned")
            .get(tool_id)
            .cloned()
    }
}

pub mod sanitization;
pub mod verbs;

#[cfg(test)]
mod tests;
