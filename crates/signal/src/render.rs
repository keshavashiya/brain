//! Tiny CommonMark builder used by per-intent handlers to emit chat
//! responses that termimad can render predictably.
//!
//! The builder enforces three things hand-rolled `format!` strings keep
//! getting wrong:
//! 1. **Blank line before lists** — without it, CommonMark folds the
//!    first bullet into the preceding paragraph and termimad collapses
//!    the layout.
//! 2. **Real `-` bullets** with two-space-per-level indent — never
//!    Unicode `•`, which termimad renders inconsistently, and never
//!    four-space indent (which CommonMark interprets as a code block).
//! 3. **Trim trailing whitespace** — single trailing `\n` only, so the
//!    CLI renderer doesn't double-space between Brain answer and
//!    prompt.
//!
//! Use this from any `handle_*` that returns text to the user. Avoid
//! `.push_str("  • ...\n")` patterns.

/// Builder for a markdown chat response. Methods consume and return
/// `Self` so call sites can chain `.heading(…).bullet(…).build()`
/// without intermediate `let mut` bindings.
#[derive(Default)]
pub(crate) struct Markdown {
    buf: String,
    /// True iff the last appended block was a list item. Used so a
    /// `line` or `heading` immediately after a list automatically
    /// inserts the blank line CommonMark needs to terminate the list.
    last_was_list: bool,
}

impl Markdown {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Append a paragraph line. Inserts a separating blank line if the
    /// previous block was a list — otherwise the line gets folded.
    pub(crate) fn line(mut self, text: impl AsRef<str>) -> Self {
        if self.last_was_list {
            self.buf.push('\n');
        }
        self.buf.push_str(text.as_ref());
        self.buf.push('\n');
        self.last_was_list = false;
        self
    }

    /// Append an empty line as a paragraph separator. Idempotent — back-
    /// to-back calls collapse to a single blank line.
    pub(crate) fn blank(mut self) -> Self {
        if !self.buf.ends_with("\n\n") {
            if !self.buf.ends_with('\n') {
                self.buf.push('\n');
            }
            self.buf.push('\n');
        }
        self.last_was_list = false;
        self
    }

    /// Append an ATX-style heading (`#`, `##`, `###`). Level is clamped
    /// to `1..=6`. Always preceded and followed by a blank line so
    /// surrounding text doesn't fuse into the heading.
    pub(crate) fn heading(mut self, level: usize, text: impl AsRef<str>) -> Self {
        let level = level.clamp(1, 6);
        self = self.blank();
        self.buf.push_str(&"#".repeat(level));
        self.buf.push(' ');
        self.buf.push_str(text.as_ref());
        self.buf.push('\n');
        self.last_was_list = false;
        self
    }

    /// Append a bullet at the given nesting level (0-based). The first
    /// bullet after a non-list block gets an extra blank line so
    /// CommonMark recognises the list start.
    pub(crate) fn bullet(mut self, level: usize, text: impl AsRef<str>) -> Self {
        if !self.last_was_list {
            // Need a blank line so the first bullet starts a list block.
            if !self.buf.is_empty() && !self.buf.ends_with("\n\n") {
                if !self.buf.ends_with('\n') {
                    self.buf.push('\n');
                }
                self.buf.push('\n');
            }
        }
        let indent = "  ".repeat(level);
        self.buf.push_str(&indent);
        self.buf.push_str("- ");
        self.buf.push_str(text.as_ref());
        self.buf.push('\n');
        self.last_was_list = true;
        self
    }

    /// Convenience for "- **key**: value" bullets at the given level.
    pub(crate) fn kv(self, level: usize, key: &str, value: impl AsRef<str>) -> Self {
        self.bullet(level, format!("**{}**: {}", key, value.as_ref()))
    }

    /// Finalise the buffer with a single trailing newline. Trims any
    /// excess so the CLI doesn't render an empty paragraph after the
    /// last block.
    pub(crate) fn build(self) -> String {
        let trimmed = self.buf.trim_end_matches('\n');
        format!("{trimmed}\n")
    }

    // ── Mutating variants for incremental builders (loops) ────────────
    //
    // The consume-self chain is nice for short calls but awkward inside
    // a `for` loop where each iteration appends one item. These
    // `push_*` aliases mutate in place so handlers can write
    // `md.push_bullet(0, x)` from inside a loop without juggling
    // ownership.
    pub(crate) fn push_line(&mut self, text: impl AsRef<str>) {
        let mut tmp = std::mem::take(self);
        tmp = tmp.line(text);
        *self = tmp;
    }
    #[allow(dead_code)]
    pub(crate) fn push_blank(&mut self) {
        let mut tmp = std::mem::take(self);
        tmp = tmp.blank();
        *self = tmp;
    }
    pub(crate) fn push_heading(&mut self, level: usize, text: impl AsRef<str>) {
        let mut tmp = std::mem::take(self);
        tmp = tmp.heading(level, text);
        *self = tmp;
    }
    pub(crate) fn push_bullet(&mut self, level: usize, text: impl AsRef<str>) {
        let mut tmp = std::mem::take(self);
        tmp = tmp.bullet(level, text);
        *self = tmp;
    }
    pub(crate) fn push_kv(&mut self, level: usize, key: &str, value: impl AsRef<str>) {
        let mut tmp = std::mem::take(self);
        tmp = tmp.kv(level, key, value);
        *self = tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_then_bullets_emits_blank_line_separator() {
        let s = Markdown::new()
            .heading(3, "Stored facts")
            .bullet(0, "**user**")
            .bullet(1, "name: Keshav")
            .bullet(1, "works_on: Brain OS")
            .build();
        // The heading must be on its own paragraph and the bullets must
        // have a leading blank line so CommonMark recognises the list.
        assert!(
            s.contains("### Stored facts\n\n- **user**\n  - name: Keshav"),
            "got: {s:?}"
        );
    }

    #[test]
    fn line_after_list_gets_separating_blank() {
        let s = Markdown::new()
            .bullet(0, "first")
            .bullet(0, "second")
            .line("Tell me more.")
            .build();
        assert!(s.contains("- second\n\nTell me more.\n"), "got: {s:?}");
    }

    #[test]
    fn build_normalises_trailing_whitespace() {
        let s = Markdown::new().line("hello").blank().blank().build();
        assert_eq!(s, "hello\n");
    }

    #[test]
    fn nested_bullets_use_two_space_indent_not_four() {
        // Four-space indent would be parsed as a CommonMark code block
        // and termimad would render it as monospaced text.
        let s = Markdown::new()
            .bullet(0, "outer")
            .bullet(1, "inner")
            .build();
        assert!(s.contains("- outer\n  - inner\n"), "got: {s:?}");
        assert!(!s.contains("    -"), "must not produce 4-space indent");
    }
}
