//! Terminal rendering: status lines, the Brain markdown skin, GFM
//! preprocessing (HTML `<br>` normalization, code-fence cleanup, wide-table
//! reflow), and the labelled response renderers shared by both chat paths.

use std::io::{stdout, Write};

use crossterm::cursor::MoveToColumn;
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{Clear, ClearType};
use crossterm::ExecutableCommand;

/// Render a transient progress line ("routing…", "thinking…") that will be
/// overwritten when the real response is rendered. Only used in the
/// non-interactive (one-shot) path; the interactive loop drops status
/// frames because rustyline's external printer can't overwrite lines.
pub(super) fn render_status_line(message: &str) -> std::io::Result<()> {
    let mut out = stdout();
    out.execute(MoveToColumn(0))?;
    out.execute(Clear(ClearType::CurrentLine))?;
    out.execute(SetAttribute(Attribute::Dim))?;
    out.execute(SetForegroundColor(Color::DarkGrey))?;
    out.execute(Print(format!("  {message}")))?;
    out.execute(ResetColor)?;
    out.execute(SetAttribute(Attribute::Reset))?;
    out.flush()?;
    Ok(())
}

/// Clear the transient status line before rendering real output.
pub(super) fn clear_status_line() -> std::io::Result<()> {
    let mut out = stdout();
    out.execute(MoveToColumn(0))?;
    out.execute(Clear(ClearType::CurrentLine))?;
    out.flush()?;
    Ok(())
}

/// A reasonable terminal width for markdown rendering. Falls back to 80
/// when stdout isn't a TTY (e.g. piped output) so wrapped lines stay
/// sane. Capped at 100 because lines longer than that hurt readability
/// even on wide monitors — the eye loses track of the wrap target.
fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(c, _)| c as usize)
        .unwrap_or(80)
        .clamp(40, 100)
}

/// Markdown skin tuned for Brain's dark-terminal aesthetic.
fn brain_skin() -> termimad::MadSkin {
    let mut skin = termimad::MadSkin::default_dark();
    use termimad::crossterm::style::Color::*;
    skin.bold.set_fg(White);
    skin.italic.set_fg(AnsiValue(244));
    skin.inline_code.set_fg(Yellow);
    skin.code_block.set_fg(AnsiValue(252));
    skin.headers.iter_mut().for_each(|h| h.set_fg(Green));
    skin.bullet.set_fg(Green);
    skin.quote_mark.set_fg(AnsiValue(244));
    skin
}

/// Replace HTML break tags (commonly emitted inside GFM table cells) with
/// real newlines so termimad can wrap them properly. UTF-8 safe.
pub(super) fn preprocess_markdown(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find('<') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        if let Some(consumed) = match_br_tag(tail) {
            out.push('\n');
            rest = &tail[consumed..];
        } else {
            out.push('<');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Strip the language info-string from opening code fences
/// (```` ```bash ```` → ```` ``` ````). termimad renders the info-string as a
/// literal first line *inside* the code block (so `bash` shows on its own line
/// above the command), which reads as a rendering bug. Removing the info-string
/// at the source leaves a clean fenced block.
///
/// Only the fence line itself is rewritten: a bare fence (```` ``` ````, used
/// for both closing fences and language-less openings) and longer fences
/// (```` ```` ````, which carry no info-string) are left untouched, as is all
/// fenced content.
pub(super) fn strip_code_fence_langs(input: &str) -> String {
    input
        .split('\n')
        .map(|line| {
            let trimmed = line.trim_start();
            let Some(info) = trimmed.strip_prefix("```") else {
                return line.to_string();
            };
            // A bare fence has nothing after the backticks; a longer fence
            // starts with another backtick. Neither carries a language token.
            if info.trim().is_empty() || info.starts_with('`') {
                return line.to_string();
            }
            // Opening fence with a language/info token → keep just the fence,
            // preserving the (≤3 space) indentation CommonMark allows.
            let indent = &line[..line.len() - trimmed.len()];
            format!("{indent}```")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// If `s` starts with a `<br>`, `<br/>`, or `<br />` tag (case-insensitive),
/// returns the byte length of the tag. Tries longest variants first so a
/// `<br />` input doesn't match the shorter `<br>` prefix.
fn match_br_tag(s: &str) -> Option<usize> {
    for variant in ["<br />", "<br/>", "<br>"] {
        let len = variant.len();
        if s.len() >= len && s.as_bytes()[..len].eq_ignore_ascii_case(variant.as_bytes()) {
            return Some(len);
        }
    }
    None
}

/// Reflow GFM tables that are too wide for the terminal into definition-style
/// bullet lists, one bullet per cell (`- **<header>:** <cell>`), with a blank
/// line between rows. termimad otherwise char-wraps each cell into a vertical
/// stack of single characters once the table's natural width exceeds the
/// terminal, which is unreadable. Tables that fit within `width` are left
/// untouched so termimad renders them as real tables.
pub(super) fn reflow_wide_tables(input: &str, width: usize) -> String {
    let lines: Vec<&str> = input.split('\n').collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        // A table block is a header row immediately followed by a delimiter row.
        if i + 1 < lines.len() && is_table_row(lines[i]) && is_table_delimiter(lines[i + 1]) {
            let header = split_table_row(lines[i]);
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut j = i + 2;
            while j < lines.len() && is_table_row(lines[j]) {
                rows.push(split_table_row(lines[j]));
                j += 1;
            }
            if table_natural_width(&header, &rows) > width {
                out.push(reflow_table(&header, &rows));
            } else {
                // Fits — keep the original lines verbatim for termimad.
                out.extend(lines[i..j].iter().map(|s| s.to_string()));
            }
            i = j;
        } else {
            out.push(lines[i].to_string());
            i += 1;
        }
    }
    out.join("\n")
}

/// A line that could be a table row: contains a `|` and at least one
/// non-pipe character.
fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.contains('|') && t.chars().any(|c| c != '|' && !c.is_whitespace())
}

/// The GFM delimiter row: every cell is dashes with optional leading/trailing
/// colons (alignment markers), and there is at least one dash overall.
fn is_table_delimiter(line: &str) -> bool {
    let cells = split_table_row(line);
    if cells.is_empty() {
        return false;
    }
    let mut saw_dash = false;
    for cell in &cells {
        let bytes = cell.trim().trim_start_matches(':').trim_end_matches(':');
        if bytes.is_empty() || !bytes.chars().all(|c| c == '-') {
            return false;
        }
        saw_dash = true;
    }
    saw_dash
}

/// Split a table row into trimmed cells, dropping the empty cells produced by
/// the conventional leading/trailing `|`.
fn split_table_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| c.trim().to_string()).collect()
}

/// Estimate the rendered width of a table: summed max column widths plus the
/// `| ` / ` | ` / ` |` border padding termimad draws.
fn table_natural_width(header: &[String], rows: &[Vec<String>]) -> usize {
    let cols = header
        .len()
        .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if cols == 0 {
        return 0;
    }
    let mut widths = vec![0usize; cols];
    for (idx, cell) in header.iter().enumerate() {
        widths[idx] = widths[idx].max(cell.chars().count());
    }
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            if idx < cols {
                widths[idx] = widths[idx].max(cell.chars().count());
            }
        }
    }
    // Each column contributes its content plus `| ` + trailing space, and the
    // table closes with a final `|`: 3 chars of border per column + 1.
    widths.iter().sum::<usize>() + cols * 3 + 1
}

/// Render a table as a bullet list: each row becomes a group of
/// `- **<header>:** <cell>` bullets, groups separated by a blank line.
fn reflow_table(header: &[String], rows: &[Vec<String>]) -> String {
    let mut blocks: Vec<String> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut lines: Vec<String> = Vec::with_capacity(row.len());
        for (idx, cell) in row.iter().enumerate() {
            if cell.is_empty() {
                continue;
            }
            match header.get(idx).filter(|h| !h.is_empty()) {
                Some(h) => lines.push(format!("- **{h}:** {cell}")),
                None => lines.push(format!("- {cell}")),
            }
        }
        if !lines.is_empty() {
            blocks.push(lines.join("\n"));
        }
    }
    blocks.join("\n\n")
}

/// Label printed before a rendered response body.
#[derive(Clone, Copy)]
pub(super) enum ResponseLabel {
    Brain,
    Proactive,
    Error,
}

impl ResponseLabel {
    /// Hand-rolled ANSI prefix so the rendered output can be assembled as
    /// a `String` and shipped through rustyline's external printer. We
    /// don't go via crossterm here because crossterm wants a writer.
    fn ansi_prefix(self) -> &'static str {
        match self {
            // bold + green / yellow / red, reset at end
            ResponseLabel::Brain => "\x1b[1;32mBrain:\x1b[0m\n",
            ResponseLabel::Proactive => "\x1b[1;33m[proactive]\x1b[0m\n",
            ResponseLabel::Error => "\x1b[1;31mError:\x1b[0m\n",
        }
    }

    fn write_prefix_direct(self) -> std::io::Result<()> {
        let mut out = stdout();
        match self {
            ResponseLabel::Brain => {
                out.execute(SetForegroundColor(Color::Green))?;
                out.execute(SetAttribute(Attribute::Bold))?;
                out.execute(Print("Brain:"))?;
            }
            ResponseLabel::Proactive => {
                out.execute(SetForegroundColor(Color::Yellow))?;
                out.execute(SetAttribute(Attribute::Bold))?;
                out.execute(Print("[proactive]"))?;
            }
            ResponseLabel::Error => {
                out.execute(SetForegroundColor(Color::Red))?;
                out.execute(SetAttribute(Attribute::Bold))?;
                out.execute(Print("Error:"))?;
            }
        }
        out.execute(SetAttribute(Attribute::Reset))?;
        out.execute(ResetColor)?;
        println!();
        out.flush()?;
        Ok(())
    }
}

/// Run the markdown body through the preprocessing passes (HTML `<br>`
/// normalization + wide-table reflow) and termimad, returning the rendered
/// string with trailing newlines trimmed. Shared by both render paths.
fn render_markdown_body(body: &str, width: usize) -> String {
    let processed = preprocess_markdown(body);
    let processed = strip_code_fence_langs(&processed);
    let processed = reflow_wide_tables(&processed, width);
    let skin = brain_skin();
    let formatted = skin.text(&processed, Some(width));
    formatted.to_string().trim_end_matches('\n').to_string()
}

/// Build the full rendered string (label + markdown body + trailing blank
/// line) for a response. Empty bodies render as empty so the caller can
/// skip them.
pub(super) fn render_to_string(label: ResponseLabel, body: &str) -> String {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return String::new();
    }
    let rendered = render_markdown_body(trimmed, terminal_width());
    let mut out = String::with_capacity(rendered.len() + 32);
    out.push_str(label.ansi_prefix());
    out.push_str(&rendered);
    out.push('\n');
    out
}

/// Direct-render path used by the one-shot (non-interactive) chat —
/// writes straight to stdout via crossterm.
pub(super) fn render_response_direct(label: ResponseLabel, body: &str) {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return;
    }
    let _ = label.write_prefix_direct();
    let rendered = render_markdown_body(trimmed, terminal_width());
    print!("{rendered}\n\n");
    // best-effort: a failed stdout flush means the pipe is closed and the
    // process is exiting anyway — nothing actionable to recover.
    let _ = stdout().flush();
}

/// Unlabeled one-shot render: the markdown body, no `Brain:`/`[proactive]`
/// label. Used for the approval-gate prompt + guidance, which are direct
/// responses to the user's own request and shouldn't carry a chat or nudge
/// label. (Deterministic subcommands like `brain capabilities` print their
/// pre-formatted body verbatim instead — see the `Plain` finalize branch.)
pub(super) fn render_plain_direct(body: &str) {
    let trimmed = body.trim_end();
    if trimmed.is_empty() {
        return;
    }
    let rendered = render_markdown_body(trimmed, terminal_width());
    print!("{rendered}\n\n");
    // best-effort: a failed stdout flush means the pipe is closed and the
    // process is exiting anyway — nothing actionable to recover.
    let _ = stdout().flush();
}
