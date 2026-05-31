//! Credential masking for file content read as LLM grounding.
//!
//! When a user asks Brain to summarise a path ("what's in `~/.brain`?"), the
//! file is read on their behalf and its content is rendered into the prompt as
//! grounding (see [`crate::extract::read_path_as_text`]). That content is
//! *untrusted, user-owned data* — a `config.yaml` routinely holds an
//! `api_key: sk-or-v1-…`. Without masking, the raw secret flows straight into
//! the LLM transcript (and, with a remote model, off the machine) — exactly the
//! leak the local-first promise forbids.
//!
//! This is distinct from [`observe::Redactor`], which scrubs *vault-marked*
//! sentinels from bus payloads we constructed ourselves. Here the secret was
//! never marked — it's arbitrary text on disk — so we detect it by shape:
//!
//! - **Credential-named keys.** A `key: value` / `key = value` line whose key
//!   normalizes to a sensitive name (`api_key`, `secret`, `token`, `password`,
//!   …) has its value replaced, structure preserved (`api_key: [redacted]`).
//! - **Known key shapes.** High-signal token prefixes (`sk-…`, `ghp_…`,
//!   `AKIA…`, `xoxb-…`, `AIza…`) are masked wherever they appear — even
//!   embedded in prose, a URL, or JSON with no recognizable key.
//!
//! Detection is deliberately recall-biased: over-masking a non-secret is a
//! cosmetic blemish in a grounding snapshot; leaking a real key is a breach.

/// What a masked secret is replaced with. ASCII so it renders cleanly in the
/// prompt and is obvious to both the model and a human reading the transcript.
const REDACTED: &str = "[redacted]";

/// Substrings that, when present in a line's *normalized* key (lowercased,
/// non-alphanumeric stripped), mark the value as a credential. Chosen to catch
/// `api_key`, `apiKey`, `API-KEY`, `llm.api_key`, `client_secret`, … while
/// avoiding innocents — note bare `auth` is excluded so `author:` is untouched.
const SENSITIVE_KEY_MARKERS: &[&str] = &[
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "passphrase",
    "accesskey",
    "privatekey",
    "credential",
    "authtoken",
    "bearertoken",
];

/// Known credential token shapes: a literal prefix plus the minimum number of
/// trailing body chars (`[A-Za-z0-9_-]`, plus `.`) that distinguishes a real
/// key from an ordinary word sharing the prefix. Bodies are greedy.
struct TokenShape {
    prefix: &'static str,
    /// Minimum body length *after* the prefix for a match to fire.
    min_body: usize,
}

const TOKEN_SHAPES: &[TokenShape] = &[
    // OpenAI / OpenRouter (`sk-`, `sk-or-v1-…`, `sk-proj-…`).
    TokenShape {
        prefix: "sk-",
        min_body: 16,
    },
    // GitHub fine-grained + classic PATs / OAuth / app tokens.
    TokenShape {
        prefix: "github_pat_",
        min_body: 20,
    },
    TokenShape {
        prefix: "ghp_",
        min_body: 20,
    },
    TokenShape {
        prefix: "gho_",
        min_body: 20,
    },
    TokenShape {
        prefix: "ghu_",
        min_body: 20,
    },
    TokenShape {
        prefix: "ghs_",
        min_body: 20,
    },
    TokenShape {
        prefix: "ghr_",
        min_body: 20,
    },
    // AWS access key id.
    TokenShape {
        prefix: "AKIA",
        min_body: 16,
    },
    // Slack bot / user / app / refresh tokens.
    TokenShape {
        prefix: "xoxb-",
        min_body: 10,
    },
    TokenShape {
        prefix: "xoxp-",
        min_body: 10,
    },
    TokenShape {
        prefix: "xoxa-",
        min_body: 10,
    },
    TokenShape {
        prefix: "xoxr-",
        min_body: 10,
    },
    // Google API key.
    TokenShape {
        prefix: "AIza",
        min_body: 30,
    },
];

/// Mask credential-shaped values in untrusted file content before it becomes
/// LLM grounding. Returns the text with secrets replaced by [`REDACTED`];
/// non-secret content is returned unchanged.
pub(crate) fn mask_secrets(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    // `split_inclusive` keeps the trailing `\n` on each piece, so appending the
    // masked lines in order reconstructs the document verbatim apart from the
    // redactions.
    for line in input.split_inclusive('\n') {
        out.push_str(&mask_line(line));
    }
    out
}

/// Mask one line: credential-named `key: value` first (whole value), then a
/// token-shape sweep over whatever remains.
fn mask_line(line: &str) -> String {
    if let Some(masked) = mask_keyed_value(line) {
        return masked;
    }
    mask_token_shapes(line)
}

/// If `line` is `<indent><key><sep><value>` (`sep` is `:` or `=`) and `key`
/// normalizes to a sensitive name, return the line with `value` replaced by
/// [`REDACTED`], preserving indentation, key, and separator. Returns `None`
/// when the line isn't a sensitive key assignment.
fn mask_keyed_value(line: &str) -> Option<String> {
    // Work on the content without the trailing newline, re-attached at the end.
    let (content, newline) = match line.strip_suffix('\n') {
        Some(rest) => (rest, "\n"),
        None => (line, ""),
    };

    // Split on the first `:` or `=`. Whichever comes first wins so
    // `url = http://x:8080` keys on `url`, not the port colon.
    let sep_idx = content
        .char_indices()
        .find(|&(_, c)| c == ':' || c == '=')
        .map(|(i, _)| i)?;
    let (key_part, rest) = content.split_at(sep_idx);
    let sep = &rest[..1];
    let value = &rest[1..];

    // An empty value (e.g. a bare `secret:` parent key in YAML) has nothing to
    // mask — leave it so we don't redact structural keys.
    if value.trim().is_empty() {
        return None;
    }
    if !is_sensitive_key(key_part) {
        return None;
    }

    // Preserve the value's leading whitespace so `api_key: x` stays readable.
    let leading_ws: String = value.chars().take_while(|c| c.is_whitespace()).collect();
    Some(format!("{key_part}{sep}{leading_ws}{REDACTED}{newline}"))
}

/// True when `key`, normalized to lowercase alphanumerics, contains any
/// [`SENSITIVE_KEY_MARKERS`] entry. Strips YAML/env/quote noise implicitly via
/// the normalization (`export FOO_TOKEN`, `"api-key"`, `- secret` all reduce).
fn is_sensitive_key(key: &str) -> bool {
    let normalized: String = key
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if normalized.is_empty() {
        return false;
    }
    SENSITIVE_KEY_MARKERS.iter().any(|m| normalized.contains(m))
}

/// Replace every [`TOKEN_SHAPES`] match in `line`, leaving other text intact.
/// Matches only at a token boundary (start, or after a non-`[A-Za-z0-9]`
/// char) so we don't fire inside a longer identifier.
fn mask_token_shapes(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if at_boundary {
            if let Some(end) = match_token_at(line, i) {
                out.push_str(REDACTED);
                i = end;
                continue;
            }
        }
        // Advance one UTF-8 char (token shapes are ASCII, but the surrounding
        // text may not be).
        let ch_len = utf8_char_len(bytes[i]);
        out.push_str(&line[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// If a [`TOKEN_SHAPES`] entry matches starting at byte `start`, return the
/// exclusive end index of the full token (prefix + body). The body greedily
/// consumes `[A-Za-z0-9_.-]`; the match fires only if the body meets the
/// shape's `min_body`.
fn match_token_at(line: &str, start: usize) -> Option<usize> {
    let rest = &line[start..];
    for shape in TOKEN_SHAPES {
        if let Some(after_prefix) = rest.strip_prefix(shape.prefix) {
            let body_len = after_prefix
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-' || *b == b'.')
                .count();
            if body_len >= shape.min_body {
                return Some(start + shape.prefix.len() + body_len);
            }
        }
    }
    None
}

/// Byte length of the UTF-8 char beginning with `b`.
fn utf8_char_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_openrouter_key_in_yaml() {
        let cfg = "llm:\n  provider: openrouter\n  api_key: sk-or-v1-abcdef0123456789abcdef\n";
        let out = mask_secrets(cfg);
        assert!(!out.contains("sk-or-v1-abcdef"), "raw key survived: {out}");
        assert!(out.contains("api_key: [redacted]"), "structure lost: {out}");
        // Non-secret lines are untouched.
        assert!(out.contains("provider: openrouter"));
    }

    #[test]
    fn masks_by_key_name_regardless_of_value_shape() {
        // Value has no recognizable token shape, but the key is sensitive.
        let out = mask_secrets("password = hunter2plaintext\n");
        assert!(!out.contains("hunter2plaintext"));
        assert!(out.contains("password = [redacted]"));
    }

    #[test]
    fn masks_token_shape_without_a_key() {
        // A bare token in prose / JSON, no `key: value` framing.
        let out = mask_secrets("the token is ghp_0123456789ABCDEFabcdef0123456789 in there");
        assert!(!out.contains("ghp_0123456789"));
        assert!(out.contains("[redacted]"));
        assert!(out.contains("the token is"));
        assert!(out.contains("in there"));
    }

    #[test]
    fn masks_secret_embedded_in_url() {
        let out = mask_secrets("curl https://api.example.com/v1?key=sk-abcdef0123456789ghijkl\n");
        assert!(!out.contains("sk-abcdef0123456789"));
        assert!(out.contains("https://api.example.com/v1?key=[redacted]"));
    }

    #[test]
    fn leaves_short_lookalikes_alone() {
        // `sk-` with too short a body is an ordinary word, not a key.
        let out = mask_secrets("sk-12 is not a key; neither is sketch or ghp_x\n");
        assert_eq!(out, "sk-12 is not a key; neither is sketch or ghp_x\n");
    }

    #[test]
    fn does_not_mask_structural_or_innocent_keys() {
        let yaml = "author: Jane\nenabled: true\nport: 5433\nsecret:\n  nested: real_value\n";
        let out = mask_secrets(yaml);
        // `author` contains no sensitive marker; `secret:` with an empty value
        // is a structural parent, not an assignment.
        assert!(out.contains("author: Jane"));
        assert!(out.contains("enabled: true"));
        assert!(out.contains("port: 5433"));
        assert!(out.contains("secret:\n"));
        // The nested value under `secret:` keys on `nested`, which is innocent.
        assert!(out.contains("nested: real_value"));
    }

    #[test]
    fn first_separator_wins_for_key_detection() {
        // `url`'s value contains a colon (port); the key is `url`, not sensitive.
        let out = mask_secrets("url: http://host:8080/path\n");
        assert_eq!(out, "url: http://host:8080/path\n");
    }

    #[test]
    fn preserves_non_secret_content_exactly() {
        let text = "# Notes\nThe quick brown fox.\n- item one\n- item two\n";
        assert_eq!(mask_secrets(text), text);
    }

    #[test]
    fn handles_no_trailing_newline() {
        let out = mask_secrets("api_key: sk-or-v1-abcdef0123456789abcdef");
        assert_eq!(out, "api_key: [redacted]");
    }

    #[test]
    fn masks_multiple_secrets_in_one_document() {
        let doc = "a: sk-aaaaaaaaaaaaaaaaaaaa\nb: ghp_bbbbbbbbbbbbbbbbbbbbbbbb\n";
        let out = mask_secrets(doc);
        assert!(!out.contains("sk-aaaa"));
        assert!(!out.contains("ghp_bbbb"));
        assert_eq!(out.matches("[redacted]").count(), 2);
    }
}
