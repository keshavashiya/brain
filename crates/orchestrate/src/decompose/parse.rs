//! JSON → step parsing: the lenient `RawStep` DTO and its tolerant
//! deserializers, plus the `RawStep` → `TaskStep` conversion shared by the
//! decompose and replan paths.

use serde::Deserialize;

use crate::step::{StepAction, TaskStep};

use super::DecompositionError;

/// Raw step as parsed from LLM JSON output.
///
/// Every nullable field uses `deserialize_with = "null_to_default"` so a
/// JSON `null` (which the LLM frequently emits) deserializes the same as
/// a missing field. Without this the entire plan parse fails with
/// `invalid type: null, expected sequence` when the LLM helpfully
/// includes `"depends_on": null`.
#[derive(Debug, Deserialize)]
pub(super) struct RawStep {
    #[serde(default, deserialize_with = "lenient_required_string")]
    pub(super) description: String,
    #[serde(default, deserialize_with = "lenient_required_string")]
    pub(super) action_type: String,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) command: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) query: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) spec: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) agent: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) artifact: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) channel: Option<String>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) message: Option<String>,
    #[serde(default, deserialize_with = "lenient_usize_vec")]
    pub(super) depends_on: Vec<usize>,
    #[serde(default, deserialize_with = "lenient_optional_string")]
    pub(super) tier: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub(super) estimated_tokens: Option<u64>,
}

/// Deserialize `null` as `T::default()`. The LLM emits `null` for empty
/// lists/strings/numbers regularly; without this every such field
/// crashes the whole plan parse.
fn null_to_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    let opt = Option::<T>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// Lenient string deserializer for `Option<String>` fields. Accepts:
///   - `null` / missing → `None`
///   - empty string → `None` (so a stray `""` doesn't silently override
///     a default like `"default"`)
///   - any string → `Some(s)`
///   - integer / float / bool → coerced to its `to_string()` form
///
/// Returning `None` instead of failing is the right behavior: the LLM
/// occasionally emits `"command": 0` or `"query": null`. A parse failure
/// here used to discard the entire plan; instead we let the field be
/// empty and let the per-action validation in `decompose_impl` /
/// `replan_after_failure` produce a precise "step N has no command"
/// error message that points at the actual problem step.
fn lenient_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Option<String>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string, integer, float, bool, or null")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(if v.is_empty() {
                None
            } else {
                Some(v.to_string())
            })
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(if v.is_empty() { None } else { Some(v) })
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(self)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
        fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
            Ok(Some(v.to_string()))
        }
    }
    deserializer.deserialize_any(V)
}

/// Lenient `String` deserializer for required string fields
/// (`description`, `action_type`). Same coercion rules as
/// `lenient_optional_string` but produces an empty string instead of
/// `None`, deferring the "missing required field" complaint to the
/// per-step validator which has more context to give a useful error.
fn lenient_required_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(lenient_optional_string(deserializer)?.unwrap_or_default())
}

/// Lenient `Vec<usize>` deserializer for `depends_on`. The LLM
/// sometimes emits a bare integer (`"depends_on": 1`) or `null` instead
/// of an array. Coerce single ints to a one-element vec, null/missing
/// to an empty vec, and accept normal arrays as-is. Anything we can't
/// interpret yields an empty vec — the worst case is the step has no
/// dependencies, which the orchestrator's sequential-fallback logic
/// repairs at planning time.
fn lenient_usize_vec<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, SeqAccess, Visitor};
    use std::fmt;

    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Vec<usize>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("array of indices, single index, or null")
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
        fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
            d.deserialize_any(self)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(vec![v as usize])
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            if v < 0 {
                Ok(Vec::new())
            } else {
                Ok(vec![v as usize])
            }
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(elem) = seq.next_element::<serde_json::Value>()? {
                if let Some(n) = elem.as_u64() {
                    out.push(n as usize);
                } else if let Some(n) = elem.as_i64() {
                    if n >= 0 {
                        out.push(n as usize);
                    }
                }
                // anything else (string, null, object) is silently dropped
            }
            Ok(out)
        }
    }
    deserializer.deserialize_any(V)
}

/// Convert one raw LLM step into a `TaskStep` using the populated UUID
/// table so `depends_on` indices resolve to ids. Lifted out of the
/// original `decompose` impl so the replan path can share it.
pub(super) fn build_task_step(i: usize, raw: RawStep, ids: &[String]) -> TaskStep {
    let depends_on: Vec<String> = raw
        .depends_on
        .iter()
        .filter_map(|&idx| ids.get(idx).cloned())
        .collect();

    let action = match raw.action_type.as_str() {
        "research" => StepAction::Research {
            query: raw.query.unwrap_or_else(|| raw.description.clone()),
        },
        "plan" => StepAction::Plan {
            output: raw.spec.unwrap_or_default(),
        },
        "implement" => StepAction::Implement {
            spec: raw.spec.unwrap_or_else(|| raw.description.clone()),
            agent: raw.agent.unwrap_or_else(|| "default".to_string()),
        },
        "execute" => StepAction::Execute {
            command: raw.command.unwrap_or_default(),
            workdir: std::env::current_dir().unwrap_or_default(),
        },
        "test" => StepAction::Test {
            command: raw.command.unwrap_or_else(|| "cargo test".to_string()),
            workdir: std::env::current_dir().unwrap_or_default(),
        },
        "shell" => StepAction::Shell {
            command: raw.command.unwrap_or_default(),
            workdir: std::env::current_dir().unwrap_or_default(),
        },
        "review" => StepAction::Review {
            artifact: raw.artifact.unwrap_or_else(|| raw.description.clone()),
        },
        "notify" => StepAction::Notify {
            channel: raw.channel.unwrap_or_else(|| "default".to_string()),
            message: raw.message.unwrap_or_else(|| raw.description.clone()),
        },
        _ => StepAction::Plan {
            output: raw.description.clone(),
        },
    };

    let tier = match raw.tier.as_deref() {
        Some("read") => audit::ActionTier::Read,
        Some("write") => audit::ActionTier::Write,
        Some("destructive") => audit::ActionTier::Destructive,
        Some("external") => audit::ActionTier::External,
        _ => audit::ActionTier::Execute,
    };

    let tier = match (&action, tier) {
        (StepAction::Notify { .. }, audit::ActionTier::External) => audit::ActionTier::Read,
        (_, t) => t,
    };

    TaskStep {
        id: ids[i].clone(),
        description: raw.description,
        action,
        depends_on,
        tier,
        estimated_tokens: raw.estimated_tokens.unwrap_or(0),
    }
}

/// Parse LLM JSON output into raw step structs.
pub(super) fn parse_steps(raw: &str) -> Result<Vec<RawStep>, DecompositionError> {
    // Try to extract JSON array from potentially markdown-wrapped output.
    let trimmed = raw.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    serde_json::from_str(json_str).map_err(|e| DecompositionError::Parse(e.to_string()))
}
