//! Per-host index of tools published by mounted MCP servers.
//!
//! The host auto-registers a server's catalog on mount and removes it on
//! unmount; the intent router queries the index to resolve a `(verb_ns,
//! verb_action)` pair to a concrete [`ToolDescriptor`]. The trait surface is
//! deliberately minimal — the full hybrid-scoring router lives outside this
//! crate and will consume this same index.
//!
//! Tool names are parsed as `verb_ns.verb_action` on the first `.`; tools
//! without a dot are stored under the empty namespace and can be matched
//! with `verb_ns = ""`.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::types::ToolDescriptor;

/// Index of tools published by mounted MCP servers.
pub trait ToolCapabilityIndex: Send + Sync {
    /// Replace the tool set published by `server`. Idempotent — calling
    /// `upsert` for an already-known server overwrites the previous entry.
    fn upsert(&self, server: &str, tools: Vec<ToolDescriptor>);

    /// Drop every tool published by `server`. Returns the number of tools
    /// removed (0 if the server was unknown).
    fn remove(&self, server: &str) -> usize;

    /// Resolve a `(verb_ns, verb_action)` pair to every matching tool.
    /// Action `"*"` matches every tool in the namespace.
    fn find(&self, verb_ns: &str, verb_action: &str) -> Vec<ToolDescriptor>;

    /// Snapshot of every indexed tool. Stable order is not guaranteed.
    fn snapshot(&self) -> Vec<ToolDescriptor>;

    /// Return up to `k` tools most relevant to `query`, ranked by keyword
    /// overlap over the tool name and description. This is the seam the
    /// chat tools channel uses to keep the advertised set small: we never
    /// hand the model the whole catalogue, only the best-matching slice.
    ///
    /// The default scores [`snapshot`](ToolCapabilityIndex::snapshot) with
    /// [`score_top_k`]; a distributed router can override with a
    /// hybrid/embedding scorer. Lower-relevance tools still fill remaining
    /// slots up to `k`, so a query that matches nothing degrades to "the
    /// first `k` tools" rather than an empty set.
    fn top_k(&self, query: &str, k: usize) -> Vec<ToolDescriptor> {
        score_top_k(self.snapshot(), query, k)
    }
}

/// Rank `tools` by keyword overlap with `query` and return the best `k`.
///
/// Scoring is deliberately simple — case-insensitive alphanumeric term
/// overlap over each tool's name and description — so it carries no model
/// or embedding dependency. Results are sorted by score (desc) then name
/// (asc) for a stable order; zero-score tools sort last and are included
/// only to fill remaining slots up to `k`. An empty `query` or `k == 0`
/// short-circuits to a name-sorted prefix / empty vec respectively.
pub fn score_top_k(mut tools: Vec<ToolDescriptor>, query: &str, k: usize) -> Vec<ToolDescriptor> {
    if k == 0 {
        return Vec::new();
    }
    let terms = tokenize(query);
    tools.sort_by(|a, b| {
        let sa = score_tool(a, &terms);
        let sb = score_tool(b, &terms);
        sb.cmp(&sa).then_with(|| a.name.cmp(&b.name))
    });
    tools.truncate(k);
    tools
}

use synapse::tokenize;

/// Number of query `terms` that appear anywhere in the tool's name or
/// description (each term counted once).
fn score_tool(tool: &ToolDescriptor, terms: &[String]) -> usize {
    if terms.is_empty() {
        return 0;
    }
    let mut haystack = tool.name.to_lowercase();
    if let Some(desc) = &tool.description {
        haystack.push(' ');
        haystack.push_str(&desc.to_lowercase());
    }
    terms
        .iter()
        .filter(|t| haystack.contains(t.as_str()))
        .count()
}

/// Default in-process [`ToolCapabilityIndex`], backed by a `RwLock<HashMap>`
/// keyed by server name. Suitable for single-process deployments — a
/// distributed router will plug in its own implementation.
#[derive(Default)]
pub struct InMemoryToolCapabilityIndex {
    by_server: RwLock<HashMap<String, Vec<ToolDescriptor>>>,
}

impl InMemoryToolCapabilityIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shared() -> Arc<dyn ToolCapabilityIndex> {
        Arc::new(Self::new())
    }
}

impl ToolCapabilityIndex for InMemoryToolCapabilityIndex {
    fn upsert(&self, server: &str, tools: Vec<ToolDescriptor>) {
        let mut guard = self.by_server.write().expect("capability index poisoned");
        guard.insert(server.to_string(), tools);
    }

    fn remove(&self, server: &str) -> usize {
        let mut guard = self.by_server.write().expect("capability index poisoned");
        guard.remove(server).map(|v| v.len()).unwrap_or(0)
    }

    fn find(&self, verb_ns: &str, verb_action: &str) -> Vec<ToolDescriptor> {
        let guard = self.by_server.read().expect("capability index poisoned");
        guard
            .values()
            .flat_map(|tools| tools.iter())
            .filter(|t| {
                let (ns, action) = parse_verb(&t.name);
                ns == verb_ns && (verb_action == "*" || action == verb_action)
            })
            .cloned()
            .collect()
    }

    fn snapshot(&self) -> Vec<ToolDescriptor> {
        let guard = self.by_server.read().expect("capability index poisoned");
        guard.values().flat_map(|t| t.iter().cloned()).collect()
    }
}

/// Split a tool name on the first `.`. Names without a `.` have an empty
/// namespace and the whole string as the action.
fn parse_verb(name: &str) -> (&str, &str) {
    match name.split_once('.') {
        Some((ns, action)) => (ns, action),
        None => ("", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn td(server: &str, name: &str) -> ToolDescriptor {
        ToolDescriptor {
            server: server.into(),
            name: name.into(),
            description: None,
            input_schema: json!({"type": "object"}),
        }
    }

    fn td_desc(server: &str, name: &str, description: &str) -> ToolDescriptor {
        ToolDescriptor {
            server: server.into(),
            name: name.into(),
            description: Some(description.into()),
            input_schema: json!({"type": "object"}),
        }
    }

    #[test]
    fn upsert_then_find_by_namespace_and_action() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert(
            "fs",
            vec![td("fs", "fs.read_text_file"), td("fs", "fs.write_file")],
        );
        let hits = idx.find("fs", "read_text_file");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "fs.read_text_file");
    }

    #[test]
    fn wildcard_action_returns_whole_namespace() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert(
            "fs",
            vec![td("fs", "fs.read_text_file"), td("fs", "fs.write_file")],
        );
        idx.upsert("git", vec![td("git", "git.commit")]);
        let mut hits = idx.find("fs", "*");
        hits.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].name, "fs.read_text_file");
        assert_eq!(hits[1].name, "fs.write_file");
    }

    #[test]
    fn upsert_overwrites_previous_tools_for_server() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert("fs", vec![td("fs", "fs.read_text_file")]);
        idx.upsert("fs", vec![td("fs", "fs.write_file")]);
        assert!(idx.find("fs", "read_text_file").is_empty());
        assert_eq!(idx.find("fs", "write_file").len(), 1);
    }

    #[test]
    fn remove_drops_servers_tools() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert(
            "fs",
            vec![td("fs", "fs.read_text_file"), td("fs", "fs.write_file")],
        );
        assert_eq!(idx.remove("fs"), 2);
        assert!(idx.find("fs", "*").is_empty());
    }

    #[test]
    fn remove_unknown_server_is_noop() {
        let idx = InMemoryToolCapabilityIndex::new();
        assert_eq!(idx.remove("ghost"), 0);
    }

    #[test]
    fn dotless_tool_names_match_empty_namespace() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert("misc", vec![td("misc", "ping")]);
        let hits = idx.find("", "ping");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "ping");
    }

    #[test]
    fn snapshot_returns_all_tools() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert("fs", vec![td("fs", "fs.read_text_file")]);
        idx.upsert("git", vec![td("git", "git.commit")]);
        assert_eq!(idx.snapshot().len(), 2);
    }

    #[test]
    fn top_k_ranks_keyword_matches_first() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert(
            "fs",
            vec![
                td_desc("fs", "fs.read_text_file", "Read the contents of a file"),
                td_desc("fs", "fs.write_file", "Write data to a file"),
            ],
        );
        idx.upsert(
            "web",
            vec![td_desc("web", "web.search", "Search the web for a query")],
        );

        let hits = idx.top_k("search the web", 2);
        assert_eq!(hits.len(), 2);
        // The web search tool matches the most query terms, so it ranks first.
        assert_eq!(hits[0].name, "web.search");
    }

    #[test]
    fn top_k_caps_result_count() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert(
            "fs",
            vec![
                td("fs", "fs.read_text_file"),
                td("fs", "fs.write_file"),
                td("fs", "fs.list_dir"),
            ],
        );
        assert_eq!(idx.top_k("file", 2).len(), 2);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let idx = InMemoryToolCapabilityIndex::new();
        idx.upsert("fs", vec![td("fs", "fs.read_text_file")]);
        assert!(idx.top_k("anything", 0).is_empty());
    }

    #[test]
    fn top_k_no_match_falls_back_to_filling_slots() {
        // A query that matches nothing still yields tools (best-effort fill)
        // up to k, name-sorted, rather than an empty advertised set.
        let tools = vec![
            td("git", "git.commit"),
            td("fs", "fs.read_text_file"),
            td("web", "web.search"),
        ];
        let hits = score_top_k(tools, "zzz_no_such_term", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].name, "fs.read_text_file");
        assert_eq!(hits[1].name, "git.commit");
    }

    #[test]
    fn score_top_k_empty_query_returns_name_sorted_prefix() {
        let tools = vec![
            td("web", "web.search"),
            td("fs", "fs.read_text_file"),
            td("git", "git.commit"),
        ];
        let hits = score_top_k(tools, "", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].name, "fs.read_text_file");
        assert_eq!(hits[1].name, "git.commit");
    }
}
