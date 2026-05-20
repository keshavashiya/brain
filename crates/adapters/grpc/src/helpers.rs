//! Small adapter-private conversion helpers used by handlers.

/// Convert protobuf empty string to Option (proto3 defaults strings to "").
pub(crate) fn non_empty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

pub(crate) fn response_to_string(content: signal::ResponseContent) -> String {
    match content {
        signal::ResponseContent::Text(t) => t,
        signal::ResponseContent::Json(v) => v.to_string(),
        signal::ResponseContent::Error(e) => e,
    }
}
