//! Parsing MCP server definitions out of *other tools'* config files.
//!
//! Most MCP clients (Claude Desktop, Cursor, Windsurf, …) store their servers
//! in a JSON file under a common `"mcpServers"` shape:
//!
//! ```json
//! { "mcpServers": {
//!     "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] },
//!     "github":     { "url": "https://mcp.example.com/sse" }
//! } }
//! ```
//!
//! This module is the *pure* parser over that text — it turns a config blob into
//! [`DiscoveredServer`] records and never touches the filesystem or the network,
//! so it is fully unit-testable. The serve layer owns the platform-specific
//! paths, the read, and the proposal; mounting anything stays a consented user
//! action (`/mcp-mount`), never automatic.

use serde::Deserialize;

/// One MCP server found in another tool's config, with enough detail to propose
/// mounting it into Brain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredServer {
    /// The server's key in the foreign config (e.g. `"filesystem"`).
    pub name: String,
    /// How it is reached.
    pub transport: DiscoveredTransport,
}

/// The transport a discovered server uses, normalised to Brain's vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveredTransport {
    /// A local subprocess speaking MCP over stdio.
    Stdio { command: String, args: Vec<String> },
    /// A remote endpoint (HTTP/SSE).
    Remote { url: String },
}

impl DiscoveredServer {
    /// The `/mcp-mount` command a user would run to mount this server — what the
    /// discovery nudge shows so adopting a server is one copy-paste away (still
    /// consented, and the user can add egress scopes before running it).
    pub fn mount_hint(&self) -> String {
        match &self.transport {
            DiscoveredTransport::Stdio { command, args } => {
                let mut cmd = command.clone();
                if !args.is_empty() {
                    cmd.push(' ');
                    cmd.push_str(&args.join(" "));
                }
                format!("/mcp-mount {} stdio {}", self.name, cmd)
            }
            DiscoveredTransport::Remote { url } => {
                format!("/mcp-mount {} streamable_http {}", self.name, url)
            }
        }
    }
}

/// The foreign config shape we read — only the `mcpServers` map, everything else
/// ignored. Tolerant: a file with no `mcpServers` key parses to an empty map.
#[derive(Debug, Deserialize)]
struct McpConfigFile {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: std::collections::BTreeMap<String, ServerEntry>,
}

/// One entry under `mcpServers`. Either a stdio command or a remote URL; an
/// entry that is neither (or malformed) is dropped rather than failing the file.
#[derive(Debug, Deserialize)]
struct ServerEntry {
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    url: Option<String>,
}

/// Parse the MCP servers declared in one foreign config blob.
///
/// Returns every well-formed server, sorted by name for a stable proposal
/// order. A blob that doesn't parse as JSON, or carries no `mcpServers`, yields
/// an empty list — discovery degrades silently rather than erroring on a config
/// file shaped differently than expected.
pub fn parse_mcp_servers(json: &str) -> Vec<DiscoveredServer> {
    let Ok(parsed) = serde_json::from_str::<McpConfigFile>(json) else {
        return Vec::new();
    };
    parsed
        .mcp_servers
        .into_iter()
        .filter_map(|(name, entry)| {
            // A URL wins if present (remote); else a non-empty command (stdio);
            // else the entry is unusable and dropped.
            let transport = if let Some(url) = entry.url.filter(|u| !u.trim().is_empty()) {
                DiscoveredTransport::Remote { url }
            } else if let Some(command) = entry.command.filter(|c| !c.trim().is_empty()) {
                DiscoveredTransport::Stdio {
                    command,
                    args: entry.args,
                }
            } else {
                return None;
            };
            Some(DiscoveredServer { name, transport })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stdio_and_remote_entries() {
        let json = r#"{
            "mcpServers": {
                "filesystem": { "command": "npx", "args": ["-y", "server-fs", "/tmp"] },
                "github": { "url": "https://mcp.example.com/sse" }
            }
        }"#;
        let servers = parse_mcp_servers(json);
        assert_eq!(servers.len(), 2);
        // Sorted by name: filesystem, github.
        assert_eq!(servers[0].name, "filesystem");
        assert_eq!(
            servers[0].transport,
            DiscoveredTransport::Stdio {
                command: "npx".into(),
                args: vec!["-y".into(), "server-fs".into(), "/tmp".into()],
            }
        );
        assert_eq!(
            servers[1].transport,
            DiscoveredTransport::Remote {
                url: "https://mcp.example.com/sse".into()
            }
        );
    }

    #[test]
    fn drops_unusable_entries_but_keeps_good_ones() {
        let json = r#"{
            "mcpServers": {
                "ok": { "command": "run-me" },
                "empty": {},
                "blank": { "command": "  " }
            }
        }"#;
        let servers = parse_mcp_servers(json);
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "ok");
    }

    #[test]
    fn missing_key_or_bad_json_is_empty() {
        assert!(parse_mcp_servers(r#"{"other": 1}"#).is_empty());
        assert!(parse_mcp_servers("not json at all").is_empty());
        assert!(parse_mcp_servers("").is_empty());
    }

    #[test]
    fn mount_hint_is_a_valid_mcp_mount_command() {
        let stdio = DiscoveredServer {
            name: "fs".into(),
            transport: DiscoveredTransport::Stdio {
                command: "npx".into(),
                args: vec!["-y".into(), "server-fs".into()],
            },
        };
        assert_eq!(stdio.mount_hint(), "/mcp-mount fs stdio npx -y server-fs");

        let remote = DiscoveredServer {
            name: "gh".into(),
            transport: DiscoveredTransport::Remote {
                url: "https://x/sse".into(),
            },
        };
        assert_eq!(
            remote.mount_hint(),
            "/mcp-mount gh streamable_http https://x/sse"
        );
    }
}
