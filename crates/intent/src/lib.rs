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

    /// The original free-text the token was derived from, when known. Carried
    /// in [`Provenance`] (always for `User`, best-effort for `Llm`/`Reflex`).
    /// The router embeds this for the semantic term — a structured token has no
    /// surface form of its own, so without it the cosine term drops out and
    /// scoring is lexical-only.
    pub fn surface_text(&self) -> Option<&str> {
        match &self.provenance {
            Provenance::User { raw_input, .. } => Some(raw_input.as_str()),
            Provenance::Llm { raw_input, .. } | Provenance::Reflex { raw_input, .. } => {
                raw_input.as_deref()
            }
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

impl ToolDescriptor {
    /// The text projection embedded for semantic capability retrieval. Folds
    /// the dotted verb, description, capability tags, and the human-facing
    /// usage hints (`when_to_use`, `example`) — the same surface a user
    /// paraphrases against — into one string. Pure, so the embedding step can
    /// live at the wiring layer (which owns the concrete embedder) while the
    /// schema crate stays dependency-free. Mirrors `graph_embed::node_text`.
    pub fn embedding_text(&self) -> String {
        let mut parts = vec![self.verb.dotted(), self.description.clone()];
        if !self.capabilities.is_empty() {
            parts.push(self.capabilities.join(" "));
        }
        if let Some(w) = &self.usage.when_to_use {
            parts.push(w.clone());
        }
        if let Some(e) = &self.usage.example {
            parts.push(e.clone());
        }
        parts.retain(|p| !p.trim().is_empty());
        parts.join(" — ")
    }
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

/// Embeds a free-text query for semantic capability matching. The router holds
/// one optionally (via [`DefaultIntentRouter::with_embedder`]) so the schema
/// crate stays free of any concrete embedder — the same independence
/// [`BreakerCheck`] buys against the resilience layer. Returns `None` on any
/// failure so the router falls back to lexical-only scoring; implementations
/// must never panic.
///
/// `namespace` is the token's namespace: implementations must honour data
/// residency — a `local_only` namespace must never reach a remote embedder
/// (use the deterministic fallback, as the graph embed path does).
#[async_trait]
pub trait QueryEmbedder: Send + Sync {
    async fn embed_query(&self, text: &str, namespace: &str) -> Option<Vec<f32>>;
}

/// Embeds a tool descriptor's [`embedding_text`](ToolDescriptor::embedding_text)
/// at registration so the router / advertiser can rank it by cosine. The
/// MCP host holds one optionally so server tools embed on mount, the same way
/// native descriptors embed at boot. Unlike [`QueryEmbedder`] there is no
/// namespace / residency gate — a tool description is catalog metadata, not
/// namespaced user data. Returns `None` on any failure so the descriptor
/// registers unembedded and scoring falls back to lexical-only for it.
#[async_trait]
pub trait DescriptorEmbedder: Send + Sync {
    async fn embed_descriptor(&self, text: &str) -> Option<Vec<f32>>;
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
    embedder: Option<Arc<dyn QueryEmbedder>>,
}

impl DefaultIntentRouter {
    pub fn new(registry: Arc<dyn ToolRegistry>) -> Self {
        Self {
            registry,
            breakers: None,
            embedder: None,
        }
    }

    /// Wire a [`BreakerCheck`] so `Open` tools are excluded from scoring.
    /// Without one, every registered tool is considered.
    pub fn with_breakers(mut self, breakers: Arc<dyn BreakerCheck>) -> Self {
        self.breakers = Some(breakers);
        self
    }

    /// Wire a [`QueryEmbedder`] so `resolve` adds a semantic cosine term over
    /// the token's [`surface_text`](IntentToken::surface_text) against each
    /// tool's `embedding`. Without one — or when the token has no surface form,
    /// the embedder fails, or a tool has no embedding — scoring is lexical-only
    /// and byte-identical to before this slice.
    pub fn with_embedder(mut self, embedder: Arc<dyn QueryEmbedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Lexical score of a candidate against the token (verb + capability
    /// overlap). Public so callers / tests can probe the ranking without
    /// invoking `resolve`. Equivalent to [`score_hybrid`](Self::score_hybrid)
    /// with no query embedding.
    pub fn score(tok: &IntentToken, tool: &ToolDescriptor) -> f32 {
        Self::score_hybrid(tok, tool, None)
    }

    /// Score a candidate, folding in a semantic cosine term when both a
    /// `query_embedding` and the tool's `embedding` are present. The cosine
    /// (`[0, 1]`) is weighted by [`ROUTER_SEMANTIC_WEIGHT`] and added on top of
    /// the lexical signals — it can resolve a verb the lexical signals leave
    /// ambiguous but never overrides an exact verb match.
    pub fn score_hybrid(
        tok: &IntentToken,
        tool: &ToolDescriptor,
        query_embedding: Option<&[f32]>,
    ) -> f32 {
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
        if let (Some(q), Some(t)) = (query_embedding, tool.embedding.as_deref()) {
            score += cosine_similarity(q, t) * ROUTER_SEMANTIC_WEIGHT;
        }
        score
    }
}

#[async_trait]
impl IntentRouter for DefaultIntentRouter {
    async fn resolve(&self, tok: &IntentToken) -> Result<ToolRoute, IntentError> {
        let tools = self.registry.list().await;
        // Embed the token's surface form once (not per-candidate). Drops out to
        // lexical-only scoring when no embedder is wired, the token has no
        // surface text, or the embed fails.
        let query_embedding: Option<Vec<f32>> = match (&self.embedder, tok.surface_text()) {
            (Some(embedder), Some(text)) if !text.trim().is_empty() => {
                embedder.embed_query(text, &tok.namespace).await
            }
            _ => None,
        };
        let mut best: Option<(ToolDescriptor, f32)> = None;
        for t in tools {
            if let Some(breakers) = &self.breakers {
                if breakers.is_open(&t.tool_id).await {
                    continue;
                }
            }
            let s = Self::score_hybrid(tok, &t, query_embedding.as_deref());
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

/// Cosine similarity between two embedding vectors, clamped to `[0, 1]`.
///
/// Re-exported from [`synapse`] so the router and the docs-retrieval assistant
/// rank by the exact same primitive. See [`synapse::cosine_similarity`] for the
/// drop-out and negative-flooring semantics.
pub use synapse::cosine_similarity;

/// Weight applied to the cosine term in [`DefaultIntentRouter::score`]. Sized
/// to a namespace-only match (`+1.0`) at unit cosine and to a full Jaccard
/// overlap (`× 1.5`) just above it, so a strong semantic match can resolve a
/// verb the lexical signals leave ambiguous without overriding an exact verb
/// hit (`+2.0`).
const ROUTER_SEMANTIC_WEIGHT: f32 = 1.5;

// Set-overlap of capability lists is shared with the docs-retrieval assistant.
use synapse::jaccard;

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
