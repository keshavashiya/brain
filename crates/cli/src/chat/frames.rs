//! One-shot WS frame accumulation: buffers chunk/complete frames into a
//! single response and routes each inbound frame to a [`FrameOutcome`].
//! The interactive path has its own routing in [`super::reader`].

use std::io::{stdout, Write};

use super::render::{
    clear_status_line, render_plain_direct, render_response_direct, ResponseLabel,
};

/// How a one-shot response is rendered. `Chat` is the conversational
/// `brain chat "…"` look (transient `routing…` status line + `Brain:`
/// label). `Plain` is for deterministic subcommands like `brain capabilities`
/// that ride the same WS path but should print only their body — no status
/// line, no chat label.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(super) enum RenderStyle {
    #[default]
    Chat,
    Plain,
}

/// Aggregates incoming WS frames into a single buffered response for the
/// one-shot path. The interactive path prints through an external printer
/// instead.
pub(super) struct ResponseAccumulator {
    pub(super) body: String,
    pub(super) label: Option<ResponseLabel>,
    pub(super) style: RenderStyle,
    /// Latest pipeline stage message ("routing…", "thinking…") for the
    /// one-shot elapsed spinner to display while frames are awaited.
    pub(super) status: Option<String>,
}

impl ResponseAccumulator {
    pub(super) fn new() -> Self {
        Self::with_style(RenderStyle::Chat)
    }

    pub(super) fn with_style(style: RenderStyle) -> Self {
        Self {
            body: String::new(),
            label: None,
            style,
            status: None,
        }
    }

    pub(super) fn push_chunk(&mut self, content: &str) {
        if content.is_empty() {
            return;
        }
        self.label.get_or_insert(ResponseLabel::Brain);
        self.body.push_str(content);
    }

    pub(super) fn set_complete_body(&mut self, content: &str) {
        if self.body.is_empty() {
            self.body.push_str(content);
        }
        self.label.get_or_insert(ResponseLabel::Brain);
    }

    pub(super) fn render_proactive(content: &str) {
        let _ = clear_status_line();
        render_response_direct(ResponseLabel::Proactive, content);
    }

    pub(super) fn render_approval_prompt(content: &str) {
        let _ = clear_status_line();
        // The gate body is self-describing ("Approval needed (external): …").
        // A `[proactive]` label on top misframes a direct response to the
        // user's own request as an unsolicited nudge — render it unlabeled.
        render_plain_direct(content);
    }

    pub(super) fn render_error(message: &str) {
        let _ = clear_status_line();
        render_response_direct(ResponseLabel::Error, message);
    }

    pub(super) fn finalize(self) -> Option<String> {
        match self.style {
            RenderStyle::Plain => {
                let trimmed = self.body.trim_end();
                if !trimmed.is_empty() {
                    // Deterministic subcommands (`brain capabilities`) send
                    // pre-formatted plain text. Print it verbatim rather than
                    // through the markdown renderer, which de-indents the
                    // manifest's `when:` lines and right-pads wrapped lines
                    // with trailing spaces. Plain never draws a status line
                    // (status frames + the spinner are suppressed upstream),
                    // so there's nothing to clear first.
                    print!("{trimmed}\n\n");
                    // best-effort: a closed stdout pipe means we're exiting
                    // anyway — nothing actionable to recover.
                    let _ = stdout().flush();
                }
            }
            RenderStyle::Chat => {
                if let Some(label) = self.label {
                    let _ = clear_status_line();
                    render_response_direct(label, &self.body);
                }
            }
        }
        Some(self.body).filter(|t| !t.is_empty())
    }
}

/// Extract the body from a `complete` frame. The daemon may shape this as
/// either `{response: {response: {value: "..."}}}` (SignalResponse with a
/// Text variant) or `{response: {value: "...", Error: "..."}}` (legacy).
/// We try both and fall back to empty string.
pub(super) fn extract_complete_body(frame: &serde_json::Value) -> &str {
    let response = match frame.get("response") {
        Some(r) => r,
        None => return "",
    };
    if let Some(s) = response
        .get("response")
        .and_then(|c| c.get("value"))
        .and_then(|v| v.as_str())
    {
        return s;
    }
    if let Some(s) = response.get("value").and_then(|v| v.as_str()) {
        return s;
    }
    if let Some(s) = response.get("Error").and_then(|v| v.as_str()) {
        return s;
    }
    ""
}

/// Result of routing a single inbound text frame into the one-shot
/// accumulator. The interactive loop has its own routing.
pub(super) enum FrameOutcome {
    Continue,
    Complete,
    /// The daemon parked the signal on a confirmation gate. A one-shot
    /// client has no stdin loop to answer it, so this is terminal here:
    /// render the prompt + guidance and return rather than blocking to the
    /// server-side nonce timeout. Carries the prompt body (which includes
    /// the nonce the daemon minted).
    Approval(String),
    Error(String),
}

/// Guidance appended after a one-shot approval prompt. Without this the CLI
/// would block until the daemon timed the nonce out (60s External / 300s
/// Destructive); instead we explain how to actually grant the action.
pub(super) const ONE_SHOT_APPROVAL_HINT: &str = "\
This action needs your approval, which can't be answered in one-shot mode.
- Interactive: run `brain chat`, then reply `approve <nonce>` (or `reject <nonce>`).
- Standing grant: pre-authorize it once via the `[confirm] standing_approvals` \
config so future runs skip the gate.";

pub(super) fn apply_frame(acc: &mut ResponseAccumulator, frame: &serde_json::Value) -> FrameOutcome {
    match frame.get("type").and_then(|v| v.as_str()) {
        Some("status") => {
            // Plain (deterministic-subcommand) output suppresses the
            // transient `routing…` line; it's chat-render chrome. Otherwise
            // record the stage for the elapsed spinner (driven by the frame
            // loop) rather than rendering here, so the line keeps ticking.
            if acc.style != RenderStyle::Plain {
                if let Some(msg) = frame.get("message").and_then(|m| m.as_str()) {
                    acc.status = Some(msg.to_string());
                }
            }
            FrameOutcome::Continue
        }
        Some("proactive") => {
            if let Some(content) = frame.get("content").and_then(|c| c.as_str()) {
                ResponseAccumulator::render_proactive(content);
            }
            FrameOutcome::Continue
        }
        Some("approval_request") => {
            let body = frame
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("Approval required.")
                .to_string();
            FrameOutcome::Approval(body)
        }
        Some("chunk") => {
            if let Some(content) = frame.get("content").and_then(|c| c.as_str()) {
                acc.push_chunk(content);
            }
            FrameOutcome::Continue
        }
        Some("complete") => {
            let body = extract_complete_body(frame);
            acc.set_complete_body(body);
            FrameOutcome::Complete
        }
        Some("error") => {
            let msg = frame
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error")
                .to_string();
            FrameOutcome::Error(msg)
        }
        _ => FrameOutcome::Continue,
    }
}
