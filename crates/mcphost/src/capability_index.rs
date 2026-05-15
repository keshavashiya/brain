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
pub trait CapabilityIndex: Send + Sync {
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
}

/// Default in-process [`CapabilityIndex`], backed by a `RwLock<HashMap>`
/// keyed by server name. Suitable for single-process deployments — a
/// distributed router will plug in its own implementation.
#[derive(Default)]
pub struct InMemoryCapabilityIndex {
    by_server: RwLock<HashMap<String, Vec<ToolDescriptor>>>,
}

impl InMemoryCapabilityIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shared() -> Arc<dyn CapabilityIndex> {
        Arc::new(Self::new())
    }
}

impl CapabilityIndex for InMemoryCapabilityIndex {
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

    #[test]
    fn upsert_then_find_by_namespace_and_action() {
        let idx = InMemoryCapabilityIndex::new();
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
        let idx = InMemoryCapabilityIndex::new();
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
        let idx = InMemoryCapabilityIndex::new();
        idx.upsert("fs", vec![td("fs", "fs.read_text_file")]);
        idx.upsert("fs", vec![td("fs", "fs.write_file")]);
        assert!(idx.find("fs", "read_text_file").is_empty());
        assert_eq!(idx.find("fs", "write_file").len(), 1);
    }

    #[test]
    fn remove_drops_servers_tools() {
        let idx = InMemoryCapabilityIndex::new();
        idx.upsert(
            "fs",
            vec![td("fs", "fs.read_text_file"), td("fs", "fs.write_file")],
        );
        assert_eq!(idx.remove("fs"), 2);
        assert!(idx.find("fs", "*").is_empty());
    }

    #[test]
    fn remove_unknown_server_is_noop() {
        let idx = InMemoryCapabilityIndex::new();
        assert_eq!(idx.remove("ghost"), 0);
    }

    #[test]
    fn dotless_tool_names_match_empty_namespace() {
        let idx = InMemoryCapabilityIndex::new();
        idx.upsert("misc", vec![td("misc", "ping")]);
        let hits = idx.find("", "ping");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "ping");
    }

    #[test]
    fn snapshot_returns_all_tools() {
        let idx = InMemoryCapabilityIndex::new();
        idx.upsert("fs", vec![td("fs", "fs.read_text_file")]);
        idx.upsert("git", vec![td("git", "git.commit")]);
        assert_eq!(idx.snapshot().len(), 2);
    }
}
