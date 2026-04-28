//! Built-in presets for OpenAI-compatible LLM providers.
//!
//! Users set `provider: groq` (or `openrouter`, `deepseek`, etc.) in config
//! without hand-assembling the base URL. If `base_url` is also provided, it
//! wins — the preset is only consulted when base_url is empty.

/// Preset entry: a friendly name → OpenAI-compatible base URL.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    pub name: &'static str,
    pub base_url: &'static str,
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "openai",
        base_url: "https://api.openai.com/v1",
    },
    Preset {
        name: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
    },
    Preset {
        name: "groq",
        base_url: "https://api.groq.com/openai/v1",
    },
    Preset {
        name: "deepseek",
        base_url: "https://api.deepseek.com/v1",
    },
    Preset {
        name: "together",
        base_url: "https://api.together.xyz/v1",
    },
    Preset {
        name: "gemini-compat",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
    },
    // NVIDIA's "build" platform — generous free tier (40 RPM at the time
    // of writing) and OpenAI-compatible. Get a key at
    // https://build.nvidia.com/settings/api-keys
    Preset {
        name: "nvidia",
        base_url: "https://integrate.api.nvidia.com/v1",
    },
];

/// Look up a preset by name. Case-insensitive.
pub fn resolve(name: &str) -> Option<&'static Preset> {
    let needle = name.trim().to_ascii_lowercase();
    PRESETS.iter().find(|p| p.name == needle)
}

/// All preset names — useful for CLI help and config validation.
pub fn names() -> impl Iterator<Item = &'static str> {
    PRESETS.iter().map(|p| p.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_known_presets() {
        assert_eq!(
            resolve("groq").map(|p| p.base_url),
            Some("https://api.groq.com/openai/v1"),
        );
        assert_eq!(
            resolve("OpenRouter").map(|p| p.base_url),
            Some("https://openrouter.ai/api/v1"),
        );
        assert_eq!(
            resolve("  deepseek  ").map(|p| p.base_url),
            Some("https://api.deepseek.com/v1"),
        );
    }

    #[test]
    fn unknown_preset_is_none() {
        assert!(resolve("nonesuch").is_none());
    }

    #[test]
    fn names_includes_all() {
        let all: Vec<_> = names().collect();
        assert!(all.contains(&"openai"));
        assert!(all.contains(&"groq"));
        assert!(all.contains(&"deepseek"));
        assert!(all.contains(&"gemini-compat"));
        assert!(all.contains(&"nvidia"));
    }

    #[test]
    fn nvidia_preset_targets_integrate_api() {
        // Locked-in: the build.nvidia.com onboarding flow hands keys
        // valid against integrate.api.nvidia.com — not build.nvidia.com.
        assert_eq!(
            resolve("nvidia").map(|p| p.base_url),
            Some("https://integrate.api.nvidia.com/v1")
        );
    }
}
