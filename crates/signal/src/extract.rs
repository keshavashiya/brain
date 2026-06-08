//! Path-to-text extraction.
//!
//! When a user references a file path, we'd like to feed its content to
//! the LLM as grounding (for chat-time path attachments) or to the
//! decomposer as context (for `decompose_task`). Plain text files just
//! need a UTF-8 read; binary formats need decoding first.
//!
//! This module dispatches by extension to a per-format extractor and
//! returns clean UTF-8 text. Failures are surfaced as `Err` so callers
//! can decide whether to skip the file or report the failure to the user.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("file is binary or otherwise unreadable as text")]
    NotText,
    #[error("PDF parse failed: {0}")]
    Pdf(String),
}

/// Read the file at `path` as UTF-8 text, dispatching by extension.
/// Returns up to `cap` bytes of text after extraction (so a 200-page
/// PDF doesn't blow the LLM context window).
///
/// **Blocking.** Uses synchronous `std::fs::read` and (for PDFs) the
/// CPU-bound `pdf_extract` parser. Callers from async contexts must
/// wrap this in `tokio::task::spawn_blocking` — all current call sites
/// (`attachment::build_chat_attachments`, `pipeline::paths::*`,
/// `pipeline::paths::collect_path_excerpts`) do so at the pipeline
/// boundary in `pipeline/{conversation,lifecycle}.rs`.
pub(crate) fn read_path_as_text(path: &Path, cap: usize) -> Result<String, ExtractError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());

    let raw = match extension.as_deref() {
        Some("pdf") => extract_pdf(path)?,
        _ => extract_plain(path)?,
    };

    if raw.is_empty() {
        return Err(ExtractError::NotText);
    }
    // This text becomes LLM grounding, so mask credential-shaped values before
    // anything downstream sees them. Mask *before* truncation so a secret
    // straddling the `cap` boundary can't survive as a partial token. This is
    // the single chokepoint every file-content grounding path funnels through
    // (chat attachments, directory inline, decompose excerpts).
    let masked = crate::secrets::mask_secrets(&raw);
    Ok(truncate_to_cap(&masked, cap))
}

/// Cap grounding text at `cap`, appending a marker when it was actually cut.
/// The slice is taken on `char` boundaries (`chars().take`), so the result is
/// always valid UTF-8 and the function can never panic on a multibyte boundary.
/// Returns the text unchanged when it already fits.
fn truncate_to_cap(text: &str, cap: usize) -> String {
    if text.len() > cap {
        let mut s: String = text.chars().take(cap).collect();
        s.push_str("\n…[truncated]");
        s
    } else {
        text.to_string()
    }
}

fn extract_plain(path: &Path) -> Result<String, ExtractError> {
    let bytes = std::fs::read(path)?;
    match std::str::from_utf8(&bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Err(ExtractError::NotText),
    }
}

/// Run the PDF text extractor in a blocking thread. `pdf-extract`'s API
/// is synchronous and CPU-bound, so we hop off the async executor to
/// avoid stalling other tasks while a multi-page PDF parses.
fn extract_pdf(path: &Path) -> Result<String, ExtractError> {
    pdf_extract::extract_text(path).map_err(|e| ExtractError::Pdf(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("note.txt");
        std::fs::write(&p, "hello plain text").unwrap();
        let s = read_path_as_text(&p, 1024).unwrap();
        assert_eq!(s, "hello plain text");
    }

    #[test]
    fn binary_returns_not_text() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("blob.bin");
        std::fs::write(&p, [0u8, 159, 146, 150]).unwrap();
        match read_path_as_text(&p, 1024) {
            Err(ExtractError::NotText) => {}
            other => panic!("expected NotText, got {other:?}"),
        }
    }

    #[test]
    fn truncates_to_cap() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("big.txt");
        let body = "x".repeat(5000);
        std::fs::write(&p, &body).unwrap();
        let s = read_path_as_text(&p, 100).unwrap();
        assert!(s.starts_with(&"x".repeat(100)));
        assert!(s.contains("[truncated]"));
    }

    // ── Fuzz target (F5) ─────────────────────────────────────────────────
    //
    // `truncate_to_cap` caps untrusted file content (extracted + secret-masked)
    // before it becomes LLM grounding. It slices a `&str` at a `char` count,
    // which is the classic spot to panic on a UTF-8 boundary — this proves it
    // can't, across arbitrary text and any cap (including 0), and pins the
    // truncation contract.
    const TRUNC_MARKER: &str = "\n…[truncated]";

    #[test]
    fn fuzz_truncate_to_cap_invariants() {
        bolero::check!()
            .with_type::<(String, u16)>()
            .for_each(|(text, cap): &(String, u16)| {
                let cap = *cap as usize;
                let out = truncate_to_cap(text, cap);

                if text.len() <= cap {
                    // Fits: returned verbatim, no marker added.
                    assert_eq!(&out, text, "short text was altered: {text:?} @ {cap}");
                } else {
                    // Cut: ends with the marker, and the kept prefix is at most
                    // `cap` chars (the slice is char-counted, never byte-sliced).
                    let kept = out
                        .strip_suffix(TRUNC_MARKER)
                        .expect("truncated output must carry the marker");
                    assert!(
                        kept.chars().count() <= cap,
                        "kept more than cap chars: {} > {cap}",
                        kept.chars().count()
                    );
                }
            });
    }

    #[test]
    fn pdf_extension_routes_to_pdf_extractor() {
        // We can't easily build a real PDF in a unit test, so instead
        // verify the dispatch path: a junk .pdf file should produce a
        // Pdf parse error, not a NotText error. That confirms the
        // extension routing works.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bogus.pdf");
        std::fs::write(&p, b"not a pdf").unwrap();
        match read_path_as_text(&p, 1024) {
            Err(ExtractError::Pdf(_)) => {}
            other => panic!("expected Pdf error, got {other:?}"),
        }
    }
}
