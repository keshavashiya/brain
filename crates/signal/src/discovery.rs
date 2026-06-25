//! Capability discovery: finding faculties the user has never used.
//!
//! The learned-fitness store knows which capabilities have been *proven*; this
//! is its mirror image — the capabilities with authored, user-facing guidance
//! that have **no recorded use at all**. The serve layer surfaces one on a slow
//! cadence as a gentle "did you know Brain can…" nudge, so a faculty the user
//! never knew about doesn't stay invisible forever.
//!
//! The manifest×fitness join lives here (where both the registry and the
//! `pub(crate)` fitness store are reachable); the cadence, the
//! suggest-each-once bookkeeping, and the notification live at the serve edge.

use crate::SignalProcessor;

/// A capability the user has never used, ready to surface as a discovery nudge.
/// Carries just the human-facing bits the nudge needs — never raw schema.
#[derive(Debug, Clone, PartialEq)]
pub struct UntriedCapability {
    /// Stable id, used to suppress re-suggesting the same capability.
    pub tool_id: String,
    /// Dotted verb (e.g. `net.check`) named in the suggestion.
    pub verb: String,
    /// Authored "use this when…" guidance — the reason to surface it.
    pub when_to_use: String,
    /// An authored example invocation, when the descriptor carries one.
    pub example: Option<String>,
}

impl SignalProcessor {
    /// User-facing capabilities with authored guidance that have no recorded
    /// use yet, best candidates for a discovery nudge.
    ///
    /// Empty when no capability registry is wired, or when learned fitness is
    /// disabled — without the fitness signal "untried" is indistinguishable from
    /// "untracked", so we decline to guess rather than nudge about everything.
    pub async fn untried_capabilities(&self) -> Vec<UntriedCapability> {
        if !self.fitness().enabled() {
            return Vec::new();
        }
        let Some(registry) = self.tool_registry() else {
            return Vec::new();
        };
        let manifest = registry.list().await;
        select_untried(manifest, |tool_id| {
            // A capability counts as tried once it has any recorded invocation.
            // A read error is treated as "tried" (fail closed: never nudge about
            // something we can't confirm is unused).
            match self.fitness().fitness(tool_id) {
                Ok(Some(f)) => f.uses > 0,
                Ok(None) => false,
                Err(_) => true,
            }
        })
    }
}

/// Filter a manifest to the user-facing, untried capabilities. Pure over the
/// `is_tried` predicate so the selection rule is unit-testable without a
/// database. A capability qualifies when it carries non-empty `when_to_use`
/// guidance (the mark of a faculty meant to be surfaced to a person, not
/// internal plumbing) and `is_tried` reports it unused.
pub(crate) fn select_untried(
    manifest: Vec<intent::ToolDescriptor>,
    is_tried: impl Fn(&str) -> bool,
) -> Vec<UntriedCapability> {
    manifest
        .into_iter()
        .filter_map(|t| {
            let when_to_use = t
                .usage
                .when_to_use
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())?;
            if is_tried(&t.tool_id) {
                return None;
            }
            Some(UntriedCapability {
                verb: t.verb.dotted(),
                tool_id: t.tool_id,
                when_to_use,
                example: t.usage.example.filter(|s| !s.trim().is_empty()),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use intent::{ToolDescriptor, ToolSource, ToolUsage, Verb};

    fn descriptor(ns: &str, action: &str, when: Option<&str>) -> ToolDescriptor {
        ToolDescriptor {
            tool_id: format!("{ns}.{action}"),
            source: ToolSource::NativeBackend {
                backend: intent::BackendId(ns.to_string()),
            },
            verb: Verb::new(ns, action),
            description: format!("{ns} {action}"),
            input_schema: serde_json::json!({}),
            output_schema: None,
            capabilities: Vec::new(),
            annotations: Default::default(),
            usage: ToolUsage {
                when_to_use: when.map(str::to_string),
                ..Default::default()
            },
            embedding: None,
        }
    }

    #[test]
    fn surfaces_only_untried_user_facing_capabilities() {
        let manifest = vec![
            descriptor("net", "check", Some("test if a host is reachable")),
            descriptor("web", "fetch", Some("grab a web page")),
            // No authored guidance → internal plumbing, never surfaced.
            descriptor("fs", "read", None),
        ];
        // `web.fetch` has been used; the others have not.
        let untried = select_untried(manifest, |id| id == "web.fetch");

        assert_eq!(untried.len(), 1, "only the untried, user-facing one");
        assert_eq!(untried[0].tool_id, "net.check");
        assert_eq!(untried[0].verb, "net.check");
        assert_eq!(untried[0].when_to_use, "test if a host is reachable");
    }

    #[test]
    fn empty_when_everything_is_tried() {
        let manifest = vec![descriptor("net", "check", Some("test reachability"))];
        assert!(select_untried(manifest, |_| true).is_empty());
    }

    #[test]
    fn blank_guidance_is_not_surfaced() {
        let manifest = vec![descriptor("x", "y", Some("   "))];
        assert!(select_untried(manifest, |_| false).is_empty());
    }
}
