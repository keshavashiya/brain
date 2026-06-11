//! The unified grants ledger: one read-only view answering "what can
//! Brain currently see and do, and on whose authority?".
//!
//! Standing authority is scattered across stores and config sections —
//! runtime standing approvals, the shell exec allowlist, file-read
//! roots, API keys, mounted MCP servers, configured LLM providers.
//! Each is individually inspectable, but trust requires the union in
//! one screen, with every line carrying its provenance (config vs
//! runtime grant) and, where one exists, its revoke path.

use uuid::Uuid;

use crate::types::*;
use crate::SignalProcessor;

impl SignalProcessor {
    /// Handle `Intent::List { resource: Grants }` (`/grants`).
    pub(super) async fn handle_list_grants(
        &self,
        signal_id: Uuid,
        prepend_nudges: &(impl Fn(SignalResponse) -> SignalResponse + ?Sized),
    ) -> Result<PipelineResult, SignalError> {
        let mut md = crate::render::Markdown::new();
        md.push_heading(3, "Grants ledger");
        md.push_line("Everything Brain can currently see or do, and on whose authority.");

        // ── Runtime grants: standing approvals ─────────────────────────
        md.push_heading(4, "Standing approvals — granted by you at runtime");
        match &self.safety.standing_approvals {
            Some(store) => {
                let grants = store.list_active().await.map_err(|e| {
                    SignalError::Processing(format!("Failed to list standing approvals: {e}"))
                })?;
                if grants.is_empty() {
                    md.push_bullet(0, "none");
                } else {
                    for g in &grants {
                        let note = g
                            .note
                            .as_deref()
                            .map(|n| format!(" — {n}"))
                            .unwrap_or_default();
                        md.push_bullet(
                            0,
                            format!(
                                "`{}` — **{}** may `{}.{}` (granted {}){note}",
                                g.id,
                                g.agent_id,
                                g.verb_ns,
                                g.verb_action,
                                g.granted_at.format("%Y-%m-%d"),
                            ),
                        );
                    }
                    md.push_line("Revoke with `/approval-revoke <id>`.");
                }
            }
            None => md.push_bullet(0, "standing-approval store not wired"),
        }

        // ── Runtime mounts: MCP servers ────────────────────────────────
        md.push_heading(4, "MCP servers — mounted with your consent");
        match self.mcp_host() {
            Some(host) => {
                let servers = host.list_servers().await;
                if servers.is_empty() {
                    md.push_bullet(0, "none mounted");
                } else {
                    for s in &servers {
                        let quarantine = if s.quarantined {
                            " — ⚠ quarantined (tool catalog changed; tools disabled)"
                        } else {
                            ""
                        };
                        md.push_bullet(
                            0,
                            format!(
                                "**{}** — {} tool(s), mounted {}{quarantine}",
                                s.name,
                                s.tool_count,
                                s.mounted_at.format("%Y-%m-%d"),
                            ),
                        );
                    }
                    md.push_line("Revoke with `/mcp-unmount <name>`.");
                }
            }
            None => md.push_bullet(0, "MCP host not configured"),
        }

        // ── Config grants: shell commands ──────────────────────────────
        let sec = &self.config.security;
        md.push_heading(4, "Shell commands — config `security.exec_allowlist`");
        if sec.exec_allowlist.is_empty() {
            md.push_bullet(0, "empty — no commands allowlisted");
        } else {
            let cmds: Vec<String> = sec
                .exec_allowlist
                .iter()
                .map(|c| format!("`{c}`"))
                .collect();
            md.push_bullet(0, format!("{} command(s): {}", cmds.len(), cmds.join(", ")));
        }

        // ── Config grants: file-read roots ─────────────────────────────
        md.push_heading(4, "File access — config `security.allowed_paths`");
        if sec.allowed_paths.is_empty() {
            md.push_bullet(
                0,
                "not set — read-only file access defaults to your home directory",
            );
        } else {
            for p in &sec.allowed_paths {
                md.push_bullet(0, format!("`{p}`"));
            }
        }

        // ── Config grants: API keys ────────────────────────────────────
        md.push_heading(4, "API keys — config `access.api_keys`");
        if self.config.access.api_keys.is_empty() {
            md.push_bullet(0, "none configured");
        } else {
            for k in &self.config.access.api_keys {
                let agent = k
                    .agent_id
                    .as_deref()
                    .map(|a| format!(" — agent `{a}`"))
                    .unwrap_or_default();
                md.push_bullet(
                    0,
                    format!(
                        "**{}** — scopes: {}{agent}",
                        k.name,
                        k.permissions.join(", ")
                    ),
                );
            }
        }

        // ── Runtime review queue: quarantined memory writers ───────────
        // The flip side of the write grants above: writes from agents
        // nobody vouched for are held here until reviewed.
        let quarantined = self.quarantined_memory_counts();
        if !quarantined.is_empty() {
            md.push_heading(4, "Unreviewed memory writers — quarantined until approved");
            for q in &quarantined {
                md.push_bullet(
                    0,
                    format!(
                        "**{}** — {} fact(s) and {} episode(s) held (excluded from recall)",
                        q.agent, q.facts, q.episodes,
                    ),
                );
            }
            md.push_line("Approve with `/memory-approve <agent>`.");
        }

        // ── Config grants: LLM providers ───────────────────────────────
        md.push_heading(4, "LLM providers — config `llm.providers`");
        let providers = &self.config.llm.providers;
        if providers.is_empty() {
            // Legacy single-provider mode: describe the live chain instead
            // of reading the deprecated config fields.
            md.push_bullet(
                0,
                format!("**{}** — model `{}`", self.llm.name(), self.llm.model()),
            );
        } else {
            for p in providers {
                let locality = if p.base_url.is_empty() {
                    String::new()
                } else if brain::url_is_loopback(&p.base_url) {
                    " — local (loopback)".to_string()
                } else {
                    format!(" — remote ({})", p.base_url)
                };
                md.push_bullet(0, format!("**{}** ({}){locality}", p.name, p.kind));
            }
        }
        let chain = if self.llm.is_local() {
            "stays on this machine (every provider in the chain is loopback)"
        } else {
            "can leave this machine (at least one provider in the chain is remote)"
        };
        md.push_line(format!("Active chat chain: {chain}."));

        // ── Egress limits: local-only namespaces ───────────────────────
        // The inverse of a grant — what Brain is *not* allowed to share —
        // belongs on the same screen.
        let local_only = self.config.memory.local_only_namespaces();
        if !local_only.is_empty() {
            md.push_heading(4, "Local-only namespaces — config `memory.namespaces`");
            for ns in local_only {
                md.push_bullet(
                    0,
                    format!("`{ns}` — never sent to a remote provider or embedder"),
                );
            }
        }

        let resp = prepend_nudges(SignalResponse::ok(signal_id, md.build()));
        Ok(PipelineResult::Complete(resp))
    }
}
