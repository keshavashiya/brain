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
use std::sync::RwLock;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub tool_id: String,
    pub source: ToolSource,
    pub verb: Verb,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub annotations: ToolAnnotations,
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

// ─── Default implementations ────────────────────────────────────────────────

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

#[cfg(test)]
mod tests;
