//! Command catalog — the CLI half of the product self-model.
//!
//! Walks the clap [`Cli`](crate::Cli) definition so the [`brain::ProductSelfModel`]
//! the SOUL is grounded on lists the *real* subcommands. clap stays the single
//! source of truth: add a subcommand to the `Commands` enum and it appears here
//! automatically, with no separate catalog to maintain.

use clap::CommandFactory;

/// Derive the command catalog from the clap definition.
///
/// One [`brain::CommandDoc`] per top-level subcommand: its name, the first line
/// of its `about`/doc-comment, and its argument identifiers. The `--verbose`
/// global flag is intentionally omitted — it's not a command.
pub fn command_catalog() -> Vec<brain::CommandDoc> {
    crate::Cli::command()
        .get_subcommands()
        .map(|sub| {
            let summary = sub
                .get_about()
                .map(|s| s.to_string())
                .unwrap_or_default()
                // Doc-comments can be multi-line; keep the first line terse.
                .lines()
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();
            let args = sub
                .get_arguments()
                .filter(|a| !a.is_global_set())
                .map(|a| a.get_id().to_string())
                .collect();
            brain::CommandDoc {
                name: sub.get_name().to_string(),
                summary,
                args,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_lists_real_commands_only() {
        let catalog = command_catalog();
        let names: Vec<&str> = catalog.iter().map(|c| c.name.as_str()).collect();
        // A representative sample of the real clap surface.
        for expected in ["init", "chat", "status", "doctor", "serve", "capabilities"] {
            assert!(
                names.contains(&expected),
                "missing `{expected}` in {names:?}"
            );
        }
        // Phantom commands the SOUL once fabricated must not appear.
        for phantom in ["send", "run", "reload"] {
            assert!(!names.contains(&phantom), "phantom `{phantom}` present");
        }
        // Every command carries a one-line summary from its doc-comment.
        assert!(catalog.iter().all(|c| !c.summary.is_empty()));
    }

    /// F4 — self-model fact integrity. The lifecycle POLICY_FACT hand-negates
    /// some commands and points to others as the real path. Cross-check those
    /// claims against the live clap catalog so a future rename/addition can't
    /// leave a stale negation (the SOUL listing *and* denying the same verb).
    #[test]
    fn policy_command_claims_agree_with_clap_catalog() {
        let names: Vec<String> = command_catalog().into_iter().map(|c| c.name).collect();

        for denied in brain::DENIED_COMMANDS {
            assert!(
                !names.iter().any(|n| n == denied),
                "POLICY_FACTS deny `brain {denied}`, but it now exists in the clap \
                 catalog — update the lifecycle policy fact and DENIED_COMMANDS",
            );
        }
        for affirmed in brain::AFFIRMED_COMMANDS {
            assert!(
                names.iter().any(|n| n == affirmed),
                "POLICY_FACTS point users to `brain {affirmed}`, but it's missing from \
                 the clap catalog — the remediation advice is now a fabrication",
            );
        }
    }
}
