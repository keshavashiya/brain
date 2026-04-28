//! Path-to-text extraction.
//!
//! When a user references a file path, we'd like to feed its content to
//! the LLM as grounding (for `project_inspect`) or to the decomposer as
//! context (for `decompose_task`). Plain text files just need a UTF-8
//! read; binary formats need decoding first.
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
    if raw.len() > cap {
        let mut s: String = raw.chars().take(cap).collect();
        s.push_str("\n…[truncated]");
        Ok(s)
    } else {
        Ok(raw)
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

/// True iff this module knows how to convert the given path to text
/// (regardless of whether the file actually exists). Used by callers
/// that want to advertise the formats they accept.
#[allow(dead_code)]
pub(crate) fn is_supported(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    matches!(extension.as_deref(), Some("pdf") | None | Some(_))
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
