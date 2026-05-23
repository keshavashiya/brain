//! Chat-time path attachments.
//!
//! When a user references a local path in a chat message ("summarise
//! /Users/me/notes", "what's in ~/Downloads/x?"), Brain reads that path
//! on their behalf and feeds the snapshot to the LLM as grounding —
//! *without* leaving the normal chat flow. The SOUL prompt, memory,
//! profile, and the user's literal question all stay intact.
//!
//! Phase B (this module) handles detection, security, and snapshotting.
//! Phase D wires the produced [`Attachment`]s into the context assembler
//! so the LLM actually sees them.
//!
//! Security: every candidate path is canonicalized and rejected unless
//! it lives under `security.allowed_paths` (default `$HOME`). Failures
//! become [`AttachmentOutcome::Skipped`] rather than fatal errors — the
//! turn proceeds through the normal SOUL pipeline and Brain can mention
//! the skip to the user if it's relevant.

use cortex::context::{Attachment, SkippedAttachment};

use crate::pipeline::{
    build_directory_snapshot, build_file_snapshot, expand_user_path, extract_path_tokens,
    friendly_io_error, path_under_any_root, resolve_allowed_roots,
};

/// Cap on attachments per chat turn. Bounds the prompt size when a user
/// pastes many paths; in practice 1-2 is the common case.
pub(crate) const MAX_CHAT_ATTACHMENTS: usize = 4;

/// Why a candidate path didn't produce an attachment. Surfaced through
/// the assembler as a `<SKIPPED_PATH reason="…"/>` tag so Brain can
/// mention it ("I couldn't read X because …") instead of silently
/// dropping the user's reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AttachmentSkipReason {
    /// `canonicalize` failed — path doesn't exist or isn't reachable.
    NotFound(String),
    /// Path canonicalized fine but lives outside `security.allowed_paths`.
    OutsideAllowedPaths,
    /// Resolved to something other than a regular file or directory
    /// (socket, block device, …).
    UnsupportedKind,
}

impl std::fmt::Display for AttachmentSkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(why) => write!(f, "{why}"),
            Self::OutsideAllowedPaths => {
                write!(f, "path is outside the configured allowed_paths sandbox")
            }
            Self::UnsupportedKind => write!(f, "not a regular file or directory"),
        }
    }
}

/// Result of building attachments from a chat message: the successfully
/// snapshotted [`Attachment`]s plus the [`SkippedAttachment`]s for
/// references we couldn't honour.
#[derive(Debug, Default)]
pub(crate) struct ChatAttachments {
    pub attached: Vec<Attachment>,
    pub skipped: Vec<SkippedAttachment>,
}

impl ChatAttachments {
    pub fn is_empty(&self) -> bool {
        self.attached.is_empty() && self.skipped.is_empty()
    }
}

/// Find every path-like token in `message`, validate it against
/// `allowed_paths`, and produce attachments (or skip records). Capped
/// at [`MAX_CHAT_ATTACHMENTS`]. Order matches the order tokens appear
/// in the message; dedup is left to `extract_path_tokens`.
pub(crate) fn build_chat_attachments(message: &str, allowed_paths: &[String]) -> ChatAttachments {
    let mut out = ChatAttachments::default();
    for token in extract_path_tokens(message)
        .into_iter()
        .take(MAX_CHAT_ATTACHMENTS)
    {
        match validate_and_snapshot(&token, allowed_paths) {
            Ok(att) => out.attached.push(att),
            Err(reason) => out.skipped.push(SkippedAttachment {
                display_path: token,
                reason: reason.to_string(),
            }),
        }
    }
    out
}

/// Resolve, sandbox-check, and snapshot a single path token.
pub(crate) fn validate_and_snapshot(
    token: &str,
    allowed_paths: &[String],
) -> Result<Attachment, AttachmentSkipReason> {
    let expanded = expand_user_path(token);
    let requested = std::path::PathBuf::from(&expanded);

    let canonical = std::fs::canonicalize(&requested)
        .map_err(|e| AttachmentSkipReason::NotFound(friendly_io_error(&e)))?;

    let roots = resolve_allowed_roots(allowed_paths);
    if !path_under_any_root(&canonical, &roots) {
        return Err(AttachmentSkipReason::OutsideAllowedPaths);
    }

    let metadata = std::fs::metadata(&canonical)
        .map_err(|e| AttachmentSkipReason::NotFound(friendly_io_error(&e)))?;

    let snapshot = if metadata.is_dir() {
        build_directory_snapshot(&canonical)
    } else if metadata.is_file() {
        build_file_snapshot(&canonical)
    } else {
        return Err(AttachmentSkipReason::UnsupportedKind);
    };

    Ok(Attachment {
        display_path: token.to_string(),
        snapshot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tempfile::tempdir` lives under `/var/folders/…` on macOS, which
    /// is a symlink to `/private/var/folders/…`. We canonicalize here
    /// so allowed-root checks (which also canonicalize) compare apples
    /// to apples in tests.
    fn canon(p: &std::path::Path) -> std::path::PathBuf {
        std::fs::canonicalize(p).expect("canonicalize tempdir")
    }

    #[test]
    fn no_path_tokens_yields_empty() {
        let out = build_chat_attachments("hey how are you doing today", &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn file_inside_sandbox_attaches() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("note.txt");
        std::fs::write(&p, "hello").unwrap();
        let root = canon(dir.path()).display().to_string();

        let out = build_chat_attachments(&format!("read {}", p.display()), &[root]);
        assert_eq!(out.attached.len(), 1);
        assert!(out.skipped.is_empty());
        let att = &out.attached[0];
        assert!(att.snapshot.contains("hello"));
        assert!(att.display_path.contains("note.txt"));
    }

    #[test]
    fn directory_inside_sandbox_attaches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();
        let root = canon(dir.path()).display().to_string();

        let out = build_chat_attachments(&format!("look at {}", dir.path().display()), &[root]);
        assert_eq!(out.attached.len(), 1);
        let att = &out.attached[0];
        assert!(att.snapshot.contains("a.txt"));
        assert!(att.snapshot.contains("b.txt"));
    }

    #[test]
    fn path_outside_sandbox_is_skipped() {
        // Sandbox is a fresh tempdir; the candidate lives in a *different*
        // tempdir, so it cannot possibly be under the sandbox root.
        let sandbox = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let p = outside.path().join("secret.txt");
        std::fs::write(&p, "nope").unwrap();
        let root = canon(sandbox.path()).display().to_string();

        let out = build_chat_attachments(&format!("read {}", p.display()), &[root]);
        assert!(out.attached.is_empty());
        assert_eq!(out.skipped.len(), 1);
        assert!(
            out.skipped[0].reason.contains("outside"),
            "unexpected reason: {}",
            out.skipped[0].reason
        );
    }

    #[test]
    fn nonexistent_path_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let root = canon(dir.path()).display().to_string();
        let missing = dir.path().join("does-not-exist.txt");

        let out = build_chat_attachments(&format!("summarise {}", missing.display()), &[root]);
        assert!(out.attached.is_empty());
        assert_eq!(out.skipped.len(), 1);
        assert_eq!(out.skipped[0].display_path, missing.display().to_string());
    }

    #[test]
    fn caps_at_max_attachments() {
        let dir = tempfile::tempdir().unwrap();
        let root = canon(dir.path()).display().to_string();
        // Create more paths than the cap allows.
        let mut tokens = Vec::new();
        for i in 0..(MAX_CHAT_ATTACHMENTS + 3) {
            let p = dir.path().join(format!("f{i}.txt"));
            std::fs::write(&p, "x").unwrap();
            tokens.push(p.display().to_string());
        }
        let message = format!("read {}", tokens.join(" and "));

        let out = build_chat_attachments(&message, &[root]);
        assert_eq!(out.attached.len(), MAX_CHAT_ATTACHMENTS);
    }

    #[test]
    fn pathless_message_with_inspect_phrasing_is_a_noop() {
        // Regression: "describe the project" / "tell me about Rust" have
        // no concrete path; extract_path_tokens returns nothing and we
        // produce zero attachments. Phase A removed the regex that used
        // to misroute these to ProjectInspect.
        for msg in [
            "describe the project",
            "tell me about Rust",
            "look at this issue",
        ] {
            assert!(build_chat_attachments(msg, &[]).is_empty(), "msg: {msg}");
        }
    }
}
