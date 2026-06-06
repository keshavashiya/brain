use super::frames::*;
use super::reader::*;
use super::render::*;
use super::signals::*;

#[test]
fn signals_line_lists_every_canonical_name() {
    let line = signals_line();
    for sig in SIGNALS {
        assert!(line.contains(sig.name), "banner missing {}", sig.name);
    }
}

#[test]
fn signal_catalog_mirrors_the_signals_table() {
    let catalog = signal_catalog();
    assert_eq!(catalog.len(), SIGNALS.len());
    for (doc, sig) in catalog.iter().zip(SIGNALS) {
        assert_eq!(doc.name, sig.name);
        assert_eq!(doc.summary, sig.summary);
    }
    // The real in-chat signals are present so the self-model can ground
    // them; the phantom the SOUL once fabricated is not.
    let names: Vec<&str> = catalog.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"/status"));
    assert!(!names.contains(&"/msg"));
}

#[test]
fn strip_code_fence_langs_drops_info_string_only() {
    let input = "before\n```bash\nls -la\n```\nafter";
    let out = strip_code_fence_langs(input);
    // The language token is gone from the opening fence…
    assert!(!out.contains("```bash"), "info string survived: {out}");
    // …but the fence, content, and closing fence remain intact.
    assert_eq!(out, "before\n```\nls -la\n```\nafter");
}

#[test]
fn strip_code_fence_langs_preserves_indent_and_other_fences() {
    // Indented opening fence keeps its indentation.
    assert_eq!(strip_code_fence_langs("  ```rust"), "  ```");
    // Bare fences (closing / language-less) are untouched.
    assert_eq!(strip_code_fence_langs("```"), "```");
    // A longer fence carries no info string and is left alone.
    assert_eq!(strip_code_fence_langs("````"), "````");
    // Non-fence lines pass through verbatim, even with inline backticks.
    assert_eq!(strip_code_fence_langs("use `cargo` now"), "use `cargo` now");
}

#[test]
fn signals_help_covers_names_aliases_and_summaries() {
    let help = signals_help();
    for sig in SIGNALS {
        assert!(help.contains(sig.name), "help missing {}", sig.name);
        assert!(
            help.contains(sig.summary),
            "help missing summary for {}",
            sig.name
        );
        for alias in sig.aliases {
            assert!(help.contains(alias), "help missing alias {alias}");
        }
    }
}

#[test]
fn preprocess_replaces_br_variants() {
    let input = "a<br>b<br/>c<br />d<BR>e";
    assert_eq!(preprocess_markdown(input), "a\nb\nc\nd\ne");
}

#[test]
fn preprocess_leaves_other_html_alone() {
    let input = "see <code>foo</code> and <em>x</em>";
    assert_eq!(preprocess_markdown(input), input);
}

#[test]
fn preprocess_passes_plain_text_through() {
    let input = "Hello, world!\n\nNo HTML here.";
    assert_eq!(preprocess_markdown(input), input);
}

const FOUR_COL_TABLE: &str = "\
| Tool | Tier | Network | Description |
| --- | --- | --- | --- |
| web_search | External | yes | Search the public web for fresh information |
| shell_exec | Execute | no | Run an allowlisted command in the sandbox |";

#[test]
fn wide_table_reflows_to_bullets() {
    // At 80 cols the 4-column table is far too wide and must reflow.
    let out = reflow_wide_tables(FOUR_COL_TABLE, 80);
    assert!(!out.contains("---"), "delimiter row should be gone: {out}");
    assert!(out.contains("- **Tool:** web_search"));
    assert!(out.contains("- **Network:** yes"));
    assert!(out.contains("- **Tool:** shell_exec"));
    // Rows are separated by a blank line.
    assert!(out.contains("\n\n- **Tool:** shell_exec"));
}

#[test]
fn narrow_table_is_left_untouched() {
    // A table that comfortably fits is handed to termimad verbatim.
    let table = "\
| A | B |
| --- | --- |
| 1 | 2 |";
    assert_eq!(reflow_wide_tables(table, 100), table);
}

#[test]
fn reflow_preserves_surrounding_prose() {
    let input = format!("Here are my tools:\n\n{FOUR_COL_TABLE}\n\nThat's all.");
    let out = reflow_wide_tables(&input, 80);
    assert!(out.starts_with("Here are my tools:"));
    assert!(out.trim_end().ends_with("That's all."));
    assert!(out.contains("- **Description:** Search the public web for fresh information"));
}

#[test]
fn text_without_tables_is_unchanged() {
    let input = "Just a line with a | pipe but no table.\nAnd another.";
    assert_eq!(reflow_wide_tables(input, 80), input);
}

#[test]
fn extract_complete_body_signal_response_shape() {
    let frame = serde_json::json!({
        "type": "complete",
        "response": {
            "response": {"type": "Text", "value": "hello"}
        }
    });
    assert_eq!(extract_complete_body(&frame), "hello");
}

#[test]
fn extract_complete_body_legacy_shape() {
    let frame = serde_json::json!({
        "type": "complete",
        "response": {"value": "legacy body"}
    });
    assert_eq!(extract_complete_body(&frame), "legacy body");
}

#[test]
fn extract_complete_body_missing_returns_empty() {
    let frame = serde_json::json!({"type": "complete"});
    assert_eq!(extract_complete_body(&frame), "");
}

#[test]
fn accumulator_collects_chunks() {
    let mut acc = ResponseAccumulator::new();
    acc.push_chunk("hello ");
    acc.push_chunk("world");
    assert_eq!(acc.body, "hello world");
}

#[test]
fn accumulator_complete_does_not_overwrite_streamed_chunks() {
    let mut acc = ResponseAccumulator::new();
    acc.push_chunk("streamed");
    acc.set_complete_body("ignored");
    assert_eq!(acc.body, "streamed");
}

#[test]
fn default_style_is_chat() {
    assert_eq!(RenderStyle::default(), RenderStyle::Chat);
    assert_eq!(ResponseAccumulator::new().style, RenderStyle::Chat);
}

#[test]
fn status_frame_is_recorded_for_spinner() {
    // Chat mode records the stage so the elapsed spinner can display it.
    let mut acc = ResponseAccumulator::new();
    let status =
        serde_json::json!({"type": "status", "stage": "thinking", "message": "thinking…"});
    assert!(matches!(
        apply_frame(&mut acc, &status),
        FrameOutcome::Continue
    ));
    assert_eq!(acc.status.as_deref(), Some("thinking…"));
}

#[test]
fn plain_style_does_not_record_status() {
    let mut acc = ResponseAccumulator::with_style(RenderStyle::Plain);
    let status = serde_json::json!({"type": "status", "message": "routing…"});
    apply_frame(&mut acc, &status);
    assert!(acc.status.is_none());
}

#[test]
fn plain_style_suppresses_status_line() {
    // In Plain mode a `status` frame must not render the `routing…` line.
    // We can't capture stdout here, but the gating reads `acc.style`, so
    // assert the style is carried and the frame is still consumed cleanly.
    let mut acc = ResponseAccumulator::with_style(RenderStyle::Plain);
    assert_eq!(acc.style, RenderStyle::Plain);
    let status = serde_json::json!({"type": "status", "message": "routing…"});
    assert!(matches!(
        apply_frame(&mut acc, &status),
        FrameOutcome::Continue
    ));
    // Status frames never contribute to the body or set a chat label.
    assert!(acc.body.is_empty());
    assert!(acc.label.is_none());
}

#[test]
fn accumulator_complete_fills_when_empty() {
    let mut acc = ResponseAccumulator::new();
    acc.set_complete_body("batch body");
    assert_eq!(acc.body, "batch body");
}

#[test]
fn apply_frame_routes_chunk_into_accumulator() {
    let mut acc = ResponseAccumulator::new();
    let frame = serde_json::json!({"type": "chunk", "content": "tok"});
    assert!(matches!(
        apply_frame(&mut acc, &frame),
        FrameOutcome::Continue
    ));
    assert_eq!(acc.body, "tok");
}

#[test]
fn apply_frame_returns_complete_on_terminal_frame() {
    let mut acc = ResponseAccumulator::new();
    let frame = serde_json::json!({
        "type": "complete",
        "response": {"response": {"value": "done"}}
    });
    let outcome = apply_frame(&mut acc, &frame);
    assert!(matches!(outcome, FrameOutcome::Complete));
}

#[test]
fn apply_frame_returns_error_with_message() {
    let mut acc = ResponseAccumulator::new();
    let frame = serde_json::json!({"type": "error", "message": "boom"});
    match apply_frame(&mut acc, &frame) {
        FrameOutcome::Error(msg) => assert_eq!(msg, "boom"),
        _ => panic!("expected Error"),
    }
}

#[test]
fn apply_frame_returns_approval_with_prompt() {
    // One-shot: an approval gate must be terminal (carry the prompt so the
    // caller can render guidance and return) rather than Continue —
    // otherwise the loop blocks to the server-side nonce timeout (W1).
    let mut acc = ResponseAccumulator::new();
    let frame = serde_json::json!({"type": "approval_request", "content": "approve abc123?"});
    match apply_frame(&mut acc, &frame) {
        FrameOutcome::Approval(body) => assert_eq!(body, "approve abc123?"),
        _ => panic!("expected Approval"),
    }
}

#[test]
fn apply_frame_approval_falls_back_when_content_missing() {
    let mut acc = ResponseAccumulator::new();
    let frame = serde_json::json!({"type": "approval_request"});
    match apply_frame(&mut acc, &frame) {
        FrameOutcome::Approval(body) => assert_eq!(body, "Approval required."),
        _ => panic!("expected Approval"),
    }
}

#[test]
fn render_to_string_includes_label_prefix() {
    let s = render_to_string(ResponseLabel::Brain, "hello");
    assert!(s.contains("Brain:"));
    assert!(s.contains("hello"));
}

#[test]
fn status_for_printer_skips_overwrite_on_first_print() {
    let s = render_status_for_printer("routing", "routing…", false);
    assert!(!s.starts_with("\x1b[1A"));
    assert!(s.contains("routing…"));
}

#[test]
fn status_for_printer_overwrites_subsequent_lines() {
    let s = render_status_for_printer("thinking", "thinking…", true);
    assert!(s.starts_with("\x1b[1A\x1b[2K"));
    assert!(s.contains("thinking…"));
}

#[test]
fn overwrite_prefix_noop_on_empty_or_disabled() {
    assert_eq!(with_overwrite_prefix(String::new(), true), "");
    assert_eq!(
        with_overwrite_prefix("Brain:\nhi\n".to_string(), false),
        "Brain:\nhi\n"
    );
}

#[test]
fn overwrite_prefix_prepends_when_enabled() {
    let out = with_overwrite_prefix("Brain:\nhi\n".to_string(), true);
    assert!(out.starts_with("\x1b[1A\x1b[2K"));
    assert!(out.contains("Brain:"));
}

#[test]
fn slash_command_detected_for_known_form() {
    assert!(looks_like_slash_command("/quit"));
    assert!(looks_like_slash_command("/status"));
    assert!(looks_like_slash_command("/staus")); // typo still flagged
    assert!(looks_like_slash_command("/foo bar baz"));
}

#[test]
fn slash_command_rejected_for_paths_and_messages() {
    assert!(!looks_like_slash_command("/Users/me/file.docx"));
    assert!(!looks_like_slash_command(
        "/Users/me/file.docx what is this?"
    ));
    assert!(!looks_like_slash_command("hello"));
    assert!(!looks_like_slash_command(""));
}

#[test]
fn render_to_string_empty_body_yields_empty() {
    assert!(render_to_string(ResponseLabel::Brain, "").is_empty());
    assert!(render_to_string(ResponseLabel::Brain, "   \n  ").is_empty());
}
