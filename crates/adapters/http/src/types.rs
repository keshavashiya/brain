//! Request/response types and error enums for the HTTP adapter.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum HttpAdapterError {
    #[error("Server error: {0}")]
    Server(String),
}

// ─── Request / Response DTOs ─────────────────────────────────────────────────

/// Incoming signal body (POST /v1/signals).
#[derive(Debug, Deserialize)]
pub struct SignalRequest {
    pub source: Option<String>,
    pub channel: Option<String>,
    pub sender: Option<String>,
    pub content: String,
    pub metadata: Option<HashMap<String, String>>,
    /// Memory namespace (default: "personal").
    pub namespace: Option<String>,
    /// Originating agent identity (e.g. "claude-code", "open-code").
    pub agent: Option<String>,
    /// Session ID for conversation continuity. Send back to reuse a session.
    pub session_id: Option<String>,
}

/// Search request body (POST /v1/memory/search).
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: Option<usize>,
    /// Filter results to this namespace only (optional).
    pub namespace: Option<String>,
}

/// Namespace statistics (GET /v1/memory/namespaces).
#[derive(Debug, Serialize)]
pub struct NamespaceJson {
    pub namespace: String,
    pub fact_count: i64,
    pub episode_count: i64,
}

/// Export envelope (GET /v1/memory/export).
#[derive(Debug, Serialize)]
pub struct ExportJson {
    pub version: String,
    pub exported_at: String,
    pub facts: Vec<signal::ExportedFact>,
    pub episodes: Vec<signal::ExportedEpisode>,
}

/// Import request body (POST /v1/memory/import).
#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub facts: Vec<signal::ExportedFact>,
    pub episodes: Vec<signal::ExportedEpisode>,
    /// If true, preview what would be imported without writing.
    #[serde(default)]
    pub dry_run: bool,
}

/// Import response (POST /v1/memory/import).
#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub facts_imported: usize,
    pub episodes_imported: usize,
    pub facts_already_existed: usize,
    pub episodes_already_existed: usize,
    pub embedded: usize,
    pub embed_failed: usize,
}

/// A single fact in JSON form (GET /v1/memory/facts, search results).
#[derive(Debug, Serialize)]
pub struct FactJson {
    pub id: String,
    pub namespace: String,
    pub category: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub distance: Option<f32>,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}
