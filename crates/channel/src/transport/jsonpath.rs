//! Narrow JSONPath subset used by transport presets.
//!
//! Only what preset files actually need — we deliberately refuse to grow
//! this into a full implementation. If a platform's API needs filter
//! expressions or recursive descent, that's the signal to use a real
//! bridge process, not to accrete path syntax here.
//!
//! Grammar supported:
//! - `$` — root
//! - `$.foo` — field access (dot syntax)
//! - `$.foo.bar` — nested fields
//! - `$.foo[N]` — array index (N ≥ 0)
//! - `$.foo[-N]` — negative index from end (N ≥ 1; `-1` = last)
//! - `$.foo[*]` — wildcard, yields every element
//! - Combinations: `$.result[*].message.text`, `$.result[-1].update_id`
//!
//! Not supported: filters `[?(...)]`, recursive descent `..`, unions
//! `[a,b]`, slices `[a:b]`, functions.

use serde_json::Value;

/// Parse error surfaced when a preset ships a malformed path.
#[derive(Debug, thiserror::Error)]
pub enum JsonPathError {
    #[error("jsonpath must start with '$': {0}")]
    MissingRoot(String),
    #[error("unbalanced brackets in jsonpath: {0}")]
    UnbalancedBrackets(String),
    #[error("invalid index in jsonpath: {0}")]
    InvalidIndex(String),
    #[error("empty segment in jsonpath: {0}")]
    EmptySegment(String),
    #[error("trailing '.' in jsonpath: {0}")]
    TrailingDot(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Segment {
    Field(String),
    Index(i64),
    Wildcard,
}

#[derive(Debug, Clone)]
pub struct JsonPath {
    raw: String,
    segments: Vec<Segment>,
}

impl JsonPath {
    pub fn parse(input: &str) -> Result<Self, JsonPathError> {
        let trimmed = input.trim();
        if !trimmed.starts_with('$') {
            return Err(JsonPathError::MissingRoot(input.to_string()));
        }
        let mut segments = Vec::new();
        let mut rest = &trimmed[1..];

        while !rest.is_empty() {
            if let Some(after) = rest.strip_prefix('.') {
                // field name runs until next '.' or '[' or end
                let end = after.find(['.', '[']).unwrap_or(after.len());
                if end == 0 {
                    if after.is_empty() {
                        return Err(JsonPathError::TrailingDot(input.to_string()));
                    }
                    return Err(JsonPathError::EmptySegment(input.to_string()));
                }
                segments.push(Segment::Field(after[..end].to_string()));
                rest = &after[end..];
            } else if let Some(after) = rest.strip_prefix('[') {
                let close = after
                    .find(']')
                    .ok_or_else(|| JsonPathError::UnbalancedBrackets(input.to_string()))?;
                let body = &after[..close];
                if body == "*" {
                    segments.push(Segment::Wildcard);
                } else {
                    let idx = body
                        .parse::<i64>()
                        .map_err(|_| JsonPathError::InvalidIndex(body.to_string()))?;
                    segments.push(Segment::Index(idx));
                }
                rest = &after[close + 1..];
            } else {
                return Err(JsonPathError::EmptySegment(input.to_string()));
            }
        }

        Ok(Self {
            raw: input.to_string(),
            segments,
        })
    }

    pub fn raw(&self) -> &str {
        &self.raw
    }

    /// Evaluate against a JSON value. Returns *all* matches — a path with
    /// a wildcard yields one result per array element.
    pub fn eval<'a>(&self, root: &'a Value) -> Vec<&'a Value> {
        let mut current: Vec<&Value> = vec![root];
        for seg in &self.segments {
            let mut next: Vec<&Value> = Vec::new();
            for v in current {
                match (seg, v) {
                    (Segment::Field(name), Value::Object(map)) => {
                        if let Some(child) = map.get(name) {
                            next.push(child);
                        }
                    }
                    (Segment::Index(i), Value::Array(arr)) => {
                        let len = arr.len() as i64;
                        let idx = if *i < 0 { len + *i } else { *i };
                        if idx >= 0 && (idx as usize) < arr.len() {
                            next.push(&arr[idx as usize]);
                        }
                    }
                    (Segment::Wildcard, Value::Array(arr)) => {
                        next.extend(arr.iter());
                    }
                    (Segment::Wildcard, Value::Object(map)) => {
                        next.extend(map.values());
                    }
                    _ => {}
                }
            }
            current = next;
        }
        current
    }

    /// Evaluate and coerce the first match to a string — the common case
    /// for extractors.
    pub fn eval_string(&self, root: &Value) -> Option<String> {
        self.eval(root).into_iter().next().and_then(value_to_string)
    }
}

/// Coerce a JSON value into a string the way preset extractors expect:
/// strings pass through, numbers/bools render via Display, everything else
/// is serialised as JSON.
pub fn value_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn root_only() {
        let p = JsonPath::parse("$").unwrap();
        let v = json!({"a": 1});
        assert_eq!(p.eval(&v).len(), 1);
    }

    #[test]
    fn field_access() {
        let p = JsonPath::parse("$.a").unwrap();
        assert_eq!(
            p.eval_string(&json!({"a": "hello"})).as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn nested_field() {
        let p = JsonPath::parse("$.a.b").unwrap();
        assert_eq!(
            p.eval_string(&json!({"a": {"b": 42}})).as_deref(),
            Some("42")
        );
    }

    #[test]
    fn index_positive() {
        let p = JsonPath::parse("$.xs[1]").unwrap();
        assert_eq!(
            p.eval_string(&json!({"xs": [10, 20, 30]})).as_deref(),
            Some("20")
        );
    }

    #[test]
    fn index_negative() {
        let p = JsonPath::parse("$.xs[-1]").unwrap();
        assert_eq!(
            p.eval_string(&json!({"xs": [10, 20, 30]})).as_deref(),
            Some("30")
        );
    }

    #[test]
    fn wildcard_array() {
        let p = JsonPath::parse("$.xs[*]").unwrap();
        let v = json!({"xs": [1, 2, 3]});
        let results = p.eval(&v);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn wildcard_then_field() {
        let p = JsonPath::parse("$.msgs[*].text").unwrap();
        let v = json!({"msgs": [{"text": "a"}, {"text": "b"}]});
        let texts: Vec<String> = p
            .eval(&v)
            .iter()
            .filter_map(|x| value_to_string(x))
            .collect();
        assert_eq!(texts, vec!["a", "b"]);
    }

    #[test]
    fn telegram_shape() {
        // Real-ish Telegram getUpdates response.
        let resp = json!({
            "ok": true,
            "result": [
                {"update_id": 100, "message": {"text": "hi", "chat": {"id": 5}, "from": {"id": 9}}},
                {"update_id": 101, "message": {"text": "bye", "chat": {"id": 5}, "from": {"id": 9}}}
            ]
        });
        let messages = JsonPath::parse("$.result[*]").unwrap().eval(&resp);
        assert_eq!(messages.len(), 2);
        let cursor = JsonPath::parse("$.result[-1].update_id")
            .unwrap()
            .eval_string(&resp);
        assert_eq!(cursor.as_deref(), Some("101"));
        let first = &messages[0];
        let text = JsonPath::parse("$.message.text")
            .unwrap()
            .eval_string(first);
        assert_eq!(text.as_deref(), Some("hi"));
    }

    #[test]
    fn missing_root_errors() {
        assert!(matches!(
            JsonPath::parse("foo.bar"),
            Err(JsonPathError::MissingRoot(_))
        ));
    }

    #[test]
    fn unbalanced_brackets_errors() {
        assert!(matches!(
            JsonPath::parse("$.a[0"),
            Err(JsonPathError::UnbalancedBrackets(_))
        ));
    }

    #[test]
    fn trailing_dot_errors() {
        assert!(matches!(
            JsonPath::parse("$.a."),
            Err(JsonPathError::TrailingDot(_))
        ));
    }

    #[test]
    fn out_of_bounds_is_no_match() {
        let p = JsonPath::parse("$.xs[10]").unwrap();
        assert!(p.eval(&json!({"xs": [1]})).is_empty());
    }
}
