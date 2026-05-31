//! Path / filesystem helpers and the `security.allowed_paths` sandbox gate.
//!
//! Shared by chat-time attachment snapshotting (`crate::attachment`) and the
//! decompose path-excerpt collector. Public-to-the-crate items are
//! re-exported by `crate::pipeline` so external callers (`crate::attachment`)
//! keep their existing paths.

/// Resolve `security.allowed_paths` into a list of canonicalized roots.
/// Empty input defaults to `$HOME`. Entries that fail to canonicalize
/// (e.g. typo, missing directory) are dropped with a warning — a
/// misconfigured entry must not silently widen the sandbox.
pub(crate) fn resolve_allowed_roots(configured: &[String]) -> Vec<std::path::PathBuf> {
    let raw: Vec<String> = if configured.is_empty() {
        std::env::var("HOME").into_iter().collect()
    } else {
        configured.to_vec()
    };
    raw.into_iter()
        .filter_map(|entry| {
            let expanded = expand_user_path(&entry);
            match std::fs::canonicalize(&expanded) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(
                        entry = %entry,
                        error = %e,
                        "security.allowed_paths entry could not be canonicalized — ignored"
                    );
                    None
                }
            }
        })
        .collect()
}

/// True when `candidate` is equal to or a descendant of any entry in
/// `roots`. Both sides should already be canonicalized.
pub(crate) fn path_under_any_root(
    candidate: &std::path::Path,
    roots: &[std::path::PathBuf],
) -> bool {
    roots
        .iter()
        .any(|root| candidate == root.as_path() || candidate.starts_with(root))
}

/// Expand a leading `~` to the user's home directory and lexically
/// normalise the result so `..` and `.` segments cannot smuggle the path
/// out of an `allowed_paths` root (Issue 129).
///
/// Lexical (not filesystem) normalisation is intentional: the caller's
/// allow-list check runs on the textual prefix and is happy to accept a
/// path that does not yet exist, so we cannot rely on `canonicalize`
/// (which would also follow symlinks at probe time — a different
/// hazard). Comparison-shape only:
///   * `~/foo/../bar`     → `<HOME>/bar`
///   * `~/./foo`          → `<HOME>/foo`
///   * `/a//b/./c/../d`   → `/a/b/d`
///
/// Anything without a `~` prefix is normalised in place but otherwise
/// returned as-is so the caller can still resolve relative segments
/// against cwd.
pub(crate) fn expand_user_path(p: &str) -> String {
    let raw = if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let mut out = std::path::PathBuf::from(home);
            out.push(rest);
            out
        } else {
            std::path::PathBuf::from(p)
        }
    } else if p == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            std::path::PathBuf::from(home)
        } else {
            std::path::PathBuf::from(p)
        }
    } else {
        std::path::PathBuf::from(p)
    };
    lexical_normalize(&raw).to_string_lossy().into_owned()
}

/// Lexically collapse `.` and `..` segments without touching the
/// filesystem. Absolute paths can never escape the root via this rule
/// (a `..` at the root pops to itself); relative paths whose `..`
/// segments outnumber the prior components turn into a bare relative
/// rump that the caller's cwd join will still detect.
fn lexical_normalize(p: &std::path::Path) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};
    let mut out = PathBuf::new();
    for component in p.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let popped = out.pop();
                if !popped {
                    // Preserve relative-path-with-leading-`..` so the
                    // caller can still notice and reject it; absolute
                    // paths reach `Component::RootDir` first and never
                    // hit this branch with an empty `out`.
                    out.push(Component::ParentDir);
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// Map an io::Error to a one-liner the user can act on. Avoids exposing
/// the bare Rust error format ("No such file or directory (os error 2)").
pub(crate) fn friendly_io_error(e: &std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::NotFound => "no such path".to_string(),
        std::io::ErrorKind::PermissionDenied => "permission denied".to_string(),
        std::io::ErrorKind::InvalidInput => "invalid path".to_string(),
        _ => e.to_string(),
    }
}

/// Directories not worth surfacing in the snapshot — they bloat the
/// listing and rarely help a summary.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    ".svelte-kit",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
];

/// File extensions to skip during the inline-probe pass. This is a
/// *performance* filter, not an editorial one: we skip these to avoid
/// reading multi-MB media files only to throw them away when UTF-8
/// validation fails. Text-shaped extensions still flow through the
/// extractor and may or may not return readable content.
const BINARY_PROBE_SKIP: &[&str] = &[
    "webp", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "ico", "heic", "heif", "svg", "mp4",
    "mov", "avi", "mkv", "webm", "m4v", "mp3", "wav", "flac", "ogg", "m4a", "aac", "zip", "tar",
    "gz", "tgz", "bz2", "xz", "7z", "rar", "exe", "dll", "so", "dylib", "bin", "ttf", "woff",
    "woff2", "otf", "eot", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "class", "jar", "pyc",
    "wasm",
];

/// Extensions that almost always hold human-readable narrative content.
/// Used only to *order* the inline-probe pass — files with these
/// extensions are probed first within the entry-budget so a README or
/// `_chat.txt` shows up ahead of arbitrary configs. Not a hard filter:
/// everything else still gets a chance (binary-skip applied first).
const NARRATIVE_EXTS: &[&str] = &[
    "md", "markdown", "txt", "text", "rst", "asciidoc", "adoc", "org", "log",
];

/// Build a content-neutral directory snapshot:
/// 1. Top-level entries listing (capped, with overflow count).
/// 2. Extension histogram so the LLM sees content shape past the cap.
/// 3. Inline excerpts of up to a few readable top-level files.
///
/// No hunt for anchor filenames, no source-landmark walks, no editorial
/// framing ("no anchor files found…"). The SOUL prompt decides what
/// kind of directory this is from what the snapshot shows.
///
/// **Blocking.** Uses `std::fs::read_dir` / `metadata`. Async callers
/// must wrap in `tokio::task::spawn_blocking` — current callers
/// (`attachment::build_chat_attachments`, `collect_path_excerpts`) are
/// themselves only invoked from `spawn_blocking` in `pipeline/`.
pub(crate) fn build_directory_snapshot(root: &std::path::Path, char_budget: usize) -> String {
    let mut out = String::new();

    let mut entries: Vec<(String, bool)> = match std::fs::read_dir(root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if SKIP_DIRS.contains(&name.as_str()) {
                    return None;
                }
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                Some((name, is_dir))
            })
            .collect(),
        Err(e) => {
            return format!("(failed to read directory: {})", friendly_io_error(&e));
        }
    };
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // (1) Entry listing — dirs first, then files, alphabetical within each.
    // Scale listing density with available char budget so large-window
    // models see more entries. Default budget ~7500 chars → 1× scale.
    out.push_str("Top-level entries:\n");
    let listing_budget = char_budget.max(7500);
    let max_listed = (listing_budget * 40 / 7500).clamp(8, 500);
    for (i, (name, is_dir)) in entries.iter().enumerate() {
        if i == max_listed {
            out.push_str(&format!(
                "  … (+{} more entries omitted)\n",
                entries.len() - max_listed
            ));
            break;
        }
        out.push_str(&format!("  {}{}\n", name, if *is_dir { "/" } else { "" }));
    }

    // (2) Extension histogram over *all* files (including those past
    // the listing cap), so a 100-image folder shows "100 .webp" instead
    // of just whatever fit in the listing window.
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut no_ext = 0usize;
    for (name, is_dir) in &entries {
        if *is_dir {
            continue;
        }
        match std::path::Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
        {
            Some(ext) if !ext.is_empty() => {
                *counts.entry(ext.to_ascii_lowercase()).or_insert(0) += 1;
            }
            _ => no_ext += 1,
        }
    }
    if !counts.is_empty() || no_ext > 0 {
        let mut bucket: Vec<(String, usize)> = counts.into_iter().collect();
        bucket.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.push_str("\nFile types:\n");
        for (ext, count) in &bucket {
            out.push_str(&format!("  {count} .{ext}\n"));
        }
        if no_ext > 0 {
            out.push_str(&format!("  {no_ext} (no extension)\n"));
        }
    }

    // (3) Inline readable top-level files, scaled with budget. Narrative
    // extensions go first so README / _chat.txt / notes.md surface
    // ahead of arbitrary configs; everything else gets a chance after.
    // Default budget (~7500 chars) → scale=1x → 3 files at 4KB each.
    // A 128k-window model (~138K chars) → scale≈18x → 50 files at 72KB each.
    let inline_scale = (char_budget / 7500).clamp(1, 64);
    let max_inlined = (3usize.saturating_mul(inline_scale)).min(50);
    let probe_bytes = (4usize * 1024).saturating_mul(inline_scale).min(256 * 1024);
    let ext_of = |name: &str| {
        std::path::Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
    };
    let is_narrative = |name: &str| {
        ext_of(name)
            .as_deref()
            .map(|e| NARRATIVE_EXTS.contains(&e))
            .unwrap_or(false)
    };
    let mut inlined = 0usize;
    for narrative_first in [true, false] {
        if inlined == max_inlined {
            break;
        }
        for (name, is_dir) in &entries {
            if inlined == max_inlined {
                break;
            }
            if *is_dir {
                continue;
            }
            if is_narrative(name) != narrative_first {
                continue;
            }
            if let Some(ext) = ext_of(name) {
                if BINARY_PROBE_SKIP.contains(&ext.as_str()) {
                    continue;
                }
            }
            let p = root.join(name);
            match crate::extract::read_path_as_text(&p, probe_bytes) {
                Ok(body) => {
                    out.push_str(&format!(
                        "\n--- {name} (first {} bytes) ---\n{body}\n",
                        probe_bytes
                    ));
                    inlined += 1;
                }
                Err(_) => continue,
            }
        }
    }

    out
}

/// Snapshot of a single file: path + content, capped at `char_budget`
/// characters (effectively the attachment's share of the LLM context
/// window). Floor of 12 KB so small-window models still get usable
/// grounding. Binary files are reported as such instead of being fed
/// through.
pub(crate) fn build_file_snapshot(p: &std::path::Path, char_budget: usize) -> String {
    let cap = char_budget.max(12 * 1024);
    let mut out = format!("File: {}\n\n", p.display());
    out.push_str(&read_truncated(p, cap));
    out
}

/// Read the first `cap` bytes of a path, returning a string. Routes
/// through the format-aware extractor first so PDFs (and other
/// supported binary formats) come back as real text. Falls back to a
/// raw UTF-8 read for plain text files; non-text binaries return a
/// short "(binary)" stub so the LLM doesn't see garbled bytes.
fn read_truncated(path: &std::path::Path, cap: usize) -> String {
    match crate::extract::read_path_as_text(path, cap) {
        Ok(s) => format!("{s}\n"),
        Err(crate::extract::ExtractError::Io(e)) => {
            format!("(read failed: {})\n", friendly_io_error(&e))
        }
        Err(crate::extract::ExtractError::NotText) => "(binary file — not displayed)\n".to_string(),
        Err(crate::extract::ExtractError::Pdf(why)) => {
            format!("(PDF parse failed: {why})\n")
        }
    }
}

// ── Auto-context expansion for decompose ───────────────────────────────────

/// Maximum number of distinct paths we'll attach to a single decompose
/// request. Caps the prompt size so a request that pastes a dozen files
/// can't blow the LLM context window.
const MAX_DECOMPOSE_PATHS: usize = 4;
/// Per-file content cap when building the decomposer's relevant_facts.
/// Tighter than `read_truncated`'s 12 KB because the decomposer needs a
/// nudge, not a full code-review-quality excerpt.
const DECOMPOSE_FILE_BYTES: usize = 3 * 1024;
/// Bare filenames that are recognised as path tokens even without a
/// directory separator. Common manifests + CI files only — the goal is
/// to surface real grounding, not to scoop arbitrary identifiers.
const BARE_MANIFEST_NAMES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "Gemfile",
    "Makefile",
    "justfile",
    "Justfile",
    "README.md",
    "CHANGELOG.md",
    "ARCHITECTURE.md",
    "Dockerfile",
];

/// Scan a free-form request for path-like tokens. Conservative on
/// purpose — only tokens that are unambiguously paths (absolute,
/// home-relative, explicitly relative, or contain a slash plus a
/// recognisable file extension) qualify. A bare word like `brain` is
/// NOT treated as a path even if it happens to be a directory in cwd.
pub(crate) fn extract_path_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        // Trim wrapping punctuation from each end independently. We only
        // strip a trailing `.` because `.github/workflows/...` is a real
        // path token, while `.../ci.yml.` (sentence terminator) isn't.
        let token = raw.trim_start_matches(['(', '[', '{', '\'', '"', '`']);
        let token = token.trim_end_matches(['.', ')', ']', '}', '\'', '"', '!', '?', '`', ':']);
        if token.is_empty() {
            continue;
        }
        if !is_pathlike(token) {
            continue;
        }
        if !out.iter().any(|p: &String| p == token) {
            out.push(token.to_string());
        }
    }
    out
}

fn is_pathlike(s: &str) -> bool {
    if s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("../")
        || s.starts_with("~/")
        || s == "~"
    {
        return true;
    }
    if BARE_MANIFEST_NAMES.contains(&s) {
        return true;
    }
    if !s.contains('/') {
        return false;
    }
    // Relative path with at least one slash AND the basename has an
    // extension — covers `crates/foo/Cargo.toml`, `.github/workflows/ci.yml`,
    // etc., without falling for prose like `and/or`.
    let basename = s.rsplit('/').next().unwrap_or("");
    if basename.contains('.') {
        return true;
    }
    // Common workflow/config dot-dirs.
    s.starts_with(".github/") || s.starts_with(".vscode/") || s.starts_with(".cargo/")
}

/// Issue 130: decompose no longer extracts path tokens from free user
/// text. The previous behavior scanned `request` for anything that looked
/// like a path (`/etc/passwd`, `~/.ssh/id_rsa`, `crates/foo/Cargo.toml`,
/// …), read the file, and inlined the contents into the decomposer's
/// `relevant_facts`. That gave any caller (or any text the LLM later
/// fed back) a primitive for arbitrary local-file exfiltration over the
/// reply channel.
///
/// The function is kept (call sites in lifecycle.rs remain) but returns
/// empty. Users who want file contents in their decompose context must
/// use the explicit `attach <path>` flow, which goes through the
/// `allowed_paths` gate.
#[allow(dead_code, unused_imports)]
pub(crate) fn collect_path_excerpts(_request: &str) -> Vec<String> {
    Vec::new()
}

/// Old free-text scanner kept private for tests / `attach` callers. NOT
/// called from decompose anymore (Issue 130).
#[allow(dead_code)]
fn collect_path_excerpts_legacy(request: &str) -> Vec<String> {
    let cwd = std::env::current_dir().ok();
    extract_path_tokens(request)
        .into_iter()
        .take(MAX_DECOMPOSE_PATHS)
        .filter_map(|tok| {
            let expanded = expand_user_path(&tok);
            let mut pb = std::path::PathBuf::from(&expanded);
            if pb.is_relative() {
                if let Some(base) = &cwd {
                    pb = base.join(&pb);
                }
            }
            build_decompose_excerpt(&tok, &pb)
        })
        .collect()
}

fn build_decompose_excerpt(token: &str, pb: &std::path::Path) -> Option<String> {
    let meta = std::fs::metadata(pb).ok()?;
    if meta.is_file() {
        // Route through the extractor so PDFs (and any other binary
        // formats we add later) come back as real text, not a refusal
        // that pushes the planner into `grep -a` workarounds.
        match crate::extract::read_path_as_text(pb, DECOMPOSE_FILE_BYTES) {
            Ok(body) => Some(format!("File `{token}`:\n```\n{body}\n```")),
            Err(e) => {
                tracing::debug!(path = %pb.display(), error = %e, "decompose excerpt skipped");
                None
            }
        }
    } else if meta.is_dir() {
        let mut entries: Vec<String> = std::fs::read_dir(pb)
            .ok()?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if SKIP_DIRS.contains(&name.as_str()) {
                    return None;
                }
                let suffix = if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    "/"
                } else {
                    ""
                };
                Some(format!("{name}{suffix}"))
            })
            .collect();
        entries.sort();
        let shown: Vec<String> = entries.iter().take(20).cloned().collect();
        let extra = entries.len().saturating_sub(shown.len());
        let extra_line = if extra > 0 {
            format!(", +{extra} more")
        } else {
            String::new()
        };
        Some(format!(
            "Directory `{token}` ({} entries{extra_line}):\n  {}",
            entries.len(),
            shown.join("\n  ")
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod directory_snapshot_tests {
    use super::build_directory_snapshot;
    use std::fs;

    /// Default char budget (~7500 chars = 2500 tokens × 3).
    const TEST_BUDGET: usize = 7500;

    #[test]
    fn empty_directory_lists_no_entries_and_no_histogram() {
        let dir = tempfile::tempdir().unwrap();
        let snap = build_directory_snapshot(dir.path(), TEST_BUDGET);
        assert!(snap.contains("Top-level entries:"));
        assert!(!snap.contains("File types:"));
        // No leftover editorial framing from the old anchor-hunt code.
        assert!(!snap.contains("no anchor files found"));
        assert!(!snap.contains("README"));
        assert!(!snap.contains("Cargo.toml"));
    }

    #[test]
    fn whatsapp_shaped_folder_inlines_chat_text_not_media() {
        // Regression scenario: a folder of .webp stickers + one
        // narrative .txt. The old anchor-hunt missed _chat.txt entirely
        // and emitted "(no anchor files found)". The new snapshot must
        // surface _chat.txt's content and show the .webp count in the
        // histogram.
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("_chat.txt"),
            "[01/07/2025] Alice: hey\n[01/07/2025] Bob: yo",
        )
        .unwrap();
        for i in 0..5 {
            // Non-UTF-8 bytes so the extractor returns NotText.
            fs::write(
                dir.path().join(format!("IMG_{i:03}.webp")),
                [0u8, 0xff, 0xfe],
            )
            .unwrap();
        }
        let snap = build_directory_snapshot(dir.path(), TEST_BUDGET);

        assert!(
            snap.contains("_chat.txt"),
            "snapshot missed _chat.txt entry"
        );
        assert!(
            snap.contains("Alice: hey"),
            "snapshot didn't inline narrative text:\n{snap}"
        );
        assert!(snap.contains("File types:"), "histogram missing:\n{snap}");
        assert!(snap.contains("5 .webp"), "webp count missing:\n{snap}");
        assert!(snap.contains("1 .txt"), "txt count missing:\n{snap}");
    }

    #[test]
    fn pure_binary_folder_emits_listing_and_histogram_only() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..3 {
            fs::write(dir.path().join(format!("a{i}.jpg")), [0u8, 0xff]).unwrap();
        }
        let snap = build_directory_snapshot(dir.path(), TEST_BUDGET);
        assert!(snap.contains("3 .jpg"));
        // Binary files must not be inlined (and the BINARY_PROBE_SKIP
        // list short-circuits the probe before extract.rs even reads).
        assert!(!snap.contains("--- a0.jpg"));
    }

    #[test]
    fn narrative_files_are_probed_before_other_text_files() {
        // A folder with two readable files: notes.md (narrative) and
        // settings.toml (config). Both extensions are non-binary. The
        // narrative pass must inline notes.md first.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("notes.md"), "# Personal notes\n\nimportant").unwrap();
        fs::write(dir.path().join("settings.toml"), "key = \"value\"").unwrap();
        let snap = build_directory_snapshot(dir.path(), TEST_BUDGET);

        let md_pos = snap.find("--- notes.md").expect("notes.md not inlined");
        let toml_pos = snap
            .find("--- settings.toml")
            .expect("settings.toml not inlined");
        assert!(
            md_pos < toml_pos,
            "narrative file should be inlined before non-narrative:\n{snap}"
        );
    }

    #[test]
    fn caps_inlined_files_at_three() {
        // Five readable .md files; only three should be inlined.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..5 {
            fs::write(
                dir.path().join(format!("note{i}.md")),
                format!("content {i}"),
            )
            .unwrap();
        }
        let snap = build_directory_snapshot(dir.path(), TEST_BUDGET);
        let inlined_count = snap.matches("--- note").count();
        assert_eq!(inlined_count, 3, "should inline exactly 3 files:\n{snap}");
    }

    #[test]
    fn entry_overflow_shows_total_count() {
        // 45 entries (> MAX_LISTED = 40). The overflow line must
        // surface the remaining count so the LLM doesn't undercount.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..45 {
            fs::write(dir.path().join(format!("f{i:02}.dat")), "x").unwrap();
        }
        let snap = build_directory_snapshot(dir.path(), TEST_BUDGET);
        assert!(
            snap.contains("+5 more entries omitted"),
            "overflow line missing:\n{snap}"
        );
        // Histogram still covers all 45 files.
        assert!(
            snap.contains("45 .dat"),
            "histogram should count all files:\n{snap}"
        );
    }
}

#[cfg(test)]
mod allowed_paths_gate_tests {
    // Sandbox enforcement for path reads. Used by chat-time path
    // attachments (`crate::attachment`) and decompose path excerpts.
    use super::{path_under_any_root, resolve_allowed_roots};
    use std::fs;

    #[test]
    fn empty_config_defaults_to_home() {
        let roots = resolve_allowed_roots(&[]);
        if let Some(home) = std::env::var_os("HOME") {
            let home_canonical = fs::canonicalize(home).expect("HOME must canonicalize");
            assert_eq!(roots, vec![home_canonical]);
        }
    }

    #[test]
    fn rejects_path_outside_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let inside = tmp.path().join("ok");
        fs::create_dir_all(&inside).unwrap();
        let outside_dir = tempfile::tempdir().unwrap();

        let root_canonical = fs::canonicalize(tmp.path()).unwrap();
        let inside_canonical = fs::canonicalize(&inside).unwrap();
        let outside_canonical = fs::canonicalize(outside_dir.path()).unwrap();

        assert!(path_under_any_root(
            &inside_canonical,
            std::slice::from_ref(&root_canonical)
        ));
        assert!(!path_under_any_root(
            &outside_canonical,
            std::slice::from_ref(&root_canonical)
        ));
    }

    #[test]
    fn symlink_escape_is_rejected_via_canonicalization() {
        // sandbox/inner -> /tmp/escape (symlink). resolve_allowed_roots
        // canonicalizes sandbox into the real path; path_under_any_root
        // sees the resolved escape target and refuses.
        let sandbox = tempfile::tempdir().unwrap();
        let escape = tempfile::tempdir().unwrap();
        let link = sandbox.path().join("inner");
        #[cfg(unix)]
        std::os::unix::fs::symlink(escape.path(), &link).unwrap();
        #[cfg(not(unix))]
        {
            // Symlinks on other platforms aren't reliable here — skip.
            return;
        }

        let sandbox_root = fs::canonicalize(sandbox.path()).unwrap();
        let resolved_link = fs::canonicalize(&link).unwrap();
        assert!(
            !path_under_any_root(&resolved_link, &[sandbox_root]),
            "symlink to outside path must not be considered inside the root"
        );
    }

    #[test]
    fn malformed_root_is_dropped_not_widened() {
        // A nonexistent path in `allowed_paths` should be silently
        // dropped, never reinterpreted as "allow everything".
        let roots = resolve_allowed_roots(&["/this/path/definitely/does/not/exist".to_string()]);
        assert!(
            roots.is_empty(),
            "broken entries must drop, not widen the sandbox"
        );
    }
}

#[cfg(test)]
mod path_extraction_tests {
    use super::*;

    #[test]
    fn extracts_absolute_relative_and_workflow_paths() {
        let text = "perform CI from .github/workflows/ci.yml \
                    in /Users/me/proj — also check ./crates/foo/Cargo.toml.";
        let paths = extract_path_tokens(text);
        assert!(paths.contains(&".github/workflows/ci.yml".to_string()));
        assert!(paths.contains(&"/Users/me/proj".to_string()));
        assert!(paths.contains(&"./crates/foo/Cargo.toml".to_string()));
    }

    #[test]
    fn ignores_prose_words_with_slashes() {
        let paths = extract_path_tokens("evaluate true/false logic and/or branches");
        assert!(paths.is_empty(), "got {paths:?}");
    }

    #[test]
    fn picks_up_bare_manifests() {
        let paths = extract_path_tokens("look at Cargo.toml and package.json please");
        assert!(paths.contains(&"Cargo.toml".to_string()));
        assert!(paths.contains(&"package.json".to_string()));
    }

    #[test]
    fn dedupes_repeated_paths() {
        let paths = extract_path_tokens("/a/b /a/b /a/b/c");
        assert_eq!(paths, vec!["/a/b".to_string(), "/a/b/c".to_string()]);
    }

    #[test]
    fn collect_excerpts_returns_empty_after_issue_130() {
        // Issue 130: decompose no longer pulls file content from free-text
        // path tokens, even when the file exists, to remove the prompt-
        // injection / data-exfil primitive. Explicit attach is the
        // sanctioned path.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("snippet.txt");
        std::fs::write(&path, "hello world").expect("write");
        let request = format!("look at {} please", path.display());
        let excerpts = collect_path_excerpts(&request);
        assert!(
            excerpts.is_empty(),
            "decompose must not inline file content from free text — got {excerpts:?}"
        );
    }

    #[test]
    fn collect_excerpts_silently_skips_missing_paths() {
        let excerpts = collect_path_excerpts("touch /tmp/does-not-exist-9384234");
        assert!(excerpts.is_empty());
    }
}
