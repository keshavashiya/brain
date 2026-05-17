use std::collections::HashMap;

use figment::{providers::Serialized, Figment};

use super::loader::expand_tilde;
use super::*;

#[test]
fn test_default_config() {
    let config = BrainConfig::default();
    assert_eq!(config.brain.data_dir, "~/.brain");
    assert_eq!(config.llm.provider, "ollama");
    assert_eq!(config.embedding.dimensions, 768);
    assert!(!config.encryption.enabled);
    assert_eq!(
        config.actions.web_search.provider,
        WebSearchProvider::DuckDuckGo
    );
    assert_eq!(config.actions.scheduling.mode, SchedulingMode::PersistOnly);
    // Proactivity defaults are synced to `default.yaml` (Issue 36).
    assert!(config.proactivity.enabled);
    assert_eq!(config.proactivity.max_per_day, 2);
    assert_eq!(config.proactivity.quiet_hours.start, "20:00");
    assert_eq!(config.proactivity.quiet_hours.end, "10:00");
    assert!(config.adapters.http.enabled);
}

/// Issue 36 regression: the four `proactivity` fields the audit
/// flagged (enabled / max_per_day / quiet_hours.{start,end}) must
/// stay in sync between `BrainConfig::default()` and the embedded
/// `default.yaml`. The full-config round-trip currently surfaces
/// unrelated drift in `agents.*`, `llm.providers`, and `reflex.*`
/// — out of scope for this slice; pick those up in a follow-up
/// before any test widens to the whole shape.
#[test]
fn default_matches_yaml_for_proactivity() {
    let yaml_source = include_str!("../../default.yaml");
    let yaml_loaded: BrainConfig =
        serde_yaml::from_str(yaml_source).expect("embedded default.yaml must deserialize");
    let struct_default = BrainConfig::default();

    assert_eq!(
        yaml_loaded.proactivity.enabled, struct_default.proactivity.enabled,
        "proactivity.enabled drifted"
    );
    assert_eq!(
        yaml_loaded.proactivity.max_per_day, struct_default.proactivity.max_per_day,
        "proactivity.max_per_day drifted"
    );
    assert_eq!(
        yaml_loaded.proactivity.quiet_hours.start, struct_default.proactivity.quiet_hours.start,
        "proactivity.quiet_hours.start drifted"
    );
    assert_eq!(
        yaml_loaded.proactivity.quiet_hours.end, struct_default.proactivity.quiet_hours.end,
        "proactivity.quiet_hours.end drifted"
    );
}

#[test]
fn test_expand_tilde() {
    let expanded = expand_tilde("~/.brain");
    assert!(!expanded.to_str().unwrap().starts_with('~'));
    assert!(expanded.to_str().unwrap().ends_with(".brain"));
}

#[test]
fn test_data_dir_paths() {
    let config = BrainConfig::default();
    let data = config.data_dir();
    assert!(data.to_str().unwrap().ends_with(".brain"));
    assert!(config.sqlite_path().to_str().unwrap().ends_with("brain.db"));
    assert!(config
        .ruvector_path()
        .to_str()
        .unwrap()
        .ends_with("ruvector"));
}

#[test]
fn test_load_from_defaults() {
    let figment = Figment::new().merge(Serialized::defaults(BrainConfig::default()));
    let config: BrainConfig = figment.extract().unwrap();
    assert_eq!(config.llm.model, "qwen2.5-coder:7b");
    assert_eq!(config.memory.search.rrf_k, 60);
    assert_eq!(config.memory.search.pre_fusion_limit, 50);
    assert!((config.memory.search.importance_weight - 0.3).abs() < f64::EPSILON);
    assert!((config.memory.search.recency_weight - 0.2).abs() < f64::EPSILON);
    assert!((config.memory.search.decay_rate - 0.01).abs() < f64::EPSILON);
}

fn writable_test_data_dir() -> String {
    std::env::temp_dir()
        .join("brain-core-tests")
        .to_string_lossy()
        .to_string()
}

fn validated_config() -> BrainConfig {
    let mut c = BrainConfig::default();
    c.brain.data_dir = writable_test_data_dir();
    c.access.api_keys.clear();
    c
}

#[test]
fn test_validate_generated_key_no_warning() {
    let mut config = BrainConfig::default();
    config.brain.data_dir = writable_test_data_dir();
    let warnings = config.validate().expect("default config should be valid");
    assert!(
        !warnings.iter().any(|w| w.contains("No API keys")),
        "should not have empty-keys warning with a generated key, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_no_api_keys_warning() {
    let config = validated_config();
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings.iter().any(|w| w.contains("No API keys")),
        "expected no-api-keys warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_port_conflict_is_hard_error() {
    let mut config = validated_config();
    config.adapters.ws.port = config.adapters.http.port;
    let err = config
        .validate()
        .expect_err("should fail with port conflict");
    assert!(
        err.contains("Port conflict"),
        "unexpected error message: {err}"
    );
}

#[test]
fn test_validate_bad_llm_url_is_hard_error() {
    let mut config = validated_config();
    config.llm.base_url = "ftp://invalid.example.com".to_string();
    let err = config.validate().expect_err("should fail with bad URL");
    assert!(
        err.contains("Invalid LLM base_url"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_validate_high_temperature_warning() {
    let mut config = validated_config();
    config.llm.temperature = 2.0;
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings.iter().any(|w| w.contains("temperature")),
        "expected temperature warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_consolidation_interval_zero_warning() {
    let mut config = validated_config();
    config.memory.consolidation.enabled = true;
    config.memory.consolidation.interval_hours = 0;
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings.iter().any(|w| w.contains("interval_hours")),
        "expected interval warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_actions_defaults_deserialize() {
    let config = BrainConfig::default();
    // On by default; DuckDuckGo HTML is the zero-config built-in so
    // web search works without Docker or an API key.
    assert!(config.actions.web_search.enabled);
    assert_eq!(
        config.actions.web_search.provider,
        WebSearchProvider::DuckDuckGo
    );
    assert_eq!(config.actions.web_search.default_top_k, 5);
    assert_eq!(config.actions.scheduling.mode, SchedulingMode::PersistOnly);
    assert!(!config.actions.messaging.enabled);
}

#[test]
fn test_validate_actions_warning_custom_without_endpoint() {
    let mut config = validated_config();
    config.actions.web_search.enabled = true;
    config.actions.web_search.provider = WebSearchProvider::Custom;
    config.actions.web_search.endpoint.clear();
    config.actions.messaging.enabled = true;
    config.actions.messaging.channels.clear();
    let warnings = config.validate().expect("config should still be valid");
    assert!(warnings.iter().any(|w| w.contains("'custom'")));
    assert!(warnings.iter().any(|w| w.contains("messaging")));
}

#[test]
fn test_validate_tavily_without_api_key_warning() {
    let mut config = validated_config();
    config.actions.web_search.enabled = true;
    config.actions.web_search.provider = WebSearchProvider::Tavily;
    config.actions.web_search.api_key.clear();
    let warnings = config.validate().expect("config should still be valid");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("'tavily'") && w.contains("api_key")),
        "expected tavily api_key warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_searxng_no_web_search_warning() {
    let mut config = validated_config();
    config.actions.web_search.enabled = true;
    config.actions.web_search.provider = WebSearchProvider::Searxng;
    let warnings = config.validate().expect("config should still be valid");
    assert!(
        !warnings.iter().any(|w| w.contains("web_search")),
        "SearXNG with default endpoint should not trigger web_search warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_http_and_https_urls_accepted() {
    let mut config = validated_config();
    config.llm.base_url = "https://api.example.com/v1".to_string();
    assert!(config.validate().is_ok());

    config.llm.base_url = "http://localhost:11434".to_string();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_all_unique_ports_ok() {
    let config = validated_config();
    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_timeout_zero_warning() {
    let mut config = validated_config();
    config.actions.web_search.timeout_ms = 0;
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("timeout_ms") && w.contains("0")),
        "expected timeout_ms=0 warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_timeout_too_high_warning() {
    let mut config = validated_config();
    config.actions.messaging.timeout_ms = 60_000;
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("timeout_ms") && w.contains("60000")),
        "expected high timeout warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_resilience_max_retries_warning() {
    let mut config = validated_config();
    config.actions.resilience.max_retries = 15;
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("max_retries") && w.contains("15")),
        "expected max_retries warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_resilience_threshold_zero_warning() {
    let mut config = validated_config();
    config.actions.resilience.circuit_breaker_threshold = 0;
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("circuit_breaker_threshold")),
        "expected circuit_breaker_threshold=0 warning, got: {:?}",
        warnings
    );
}

#[test]
fn test_resilience_defaults() {
    let res = ResilienceConfig::default();
    assert_eq!(res.max_retries, 2);
    assert_eq!(res.retry_base_ms, 500);
    assert_eq!(res.circuit_breaker_threshold, 5);
    assert_eq!(res.circuit_breaker_cooldown_secs, 60);
}

#[test]
fn test_channel_config_old_format_compat() {
    let yaml = r#"
            enabled: false
            timeout_ms: 3000
            channels:
              alerts: "https://example.com/hook"
              ops: "https://slack.example.com/webhook"
        "#;
    let cfg: MessagingActionConfig =
        serde_yaml::from_str(yaml).expect("old format should deserialize");
    assert_eq!(cfg.channels.len(), 2);
    assert_eq!(cfg.channels["alerts"].url, "https://example.com/hook");
    assert!(cfg.channels["alerts"].body.is_empty());
    assert!(cfg.channels["alerts"].headers.is_empty());
}

#[test]
fn test_channel_config_new_format() {
    let yaml = r#"
            enabled: true
            timeout_ms: 3000
            channels:
              alerts:
                url: "https://hooks.slack.com/services/T/B/x"
                body: '{"text": "{{content}}"}'
                headers:
                  Authorization: "Bearer tok123"
        "#;
    let cfg: MessagingActionConfig =
        serde_yaml::from_str(yaml).expect("new format should deserialize");
    assert_eq!(cfg.channels.len(), 1);
    let ch = &cfg.channels["alerts"];
    assert_eq!(ch.url, "https://hooks.slack.com/services/T/B/x");
    assert_eq!(ch.body, r#"{"text": "{{content}}"}"#);
    assert_eq!(ch.headers["Authorization"], "Bearer tok123");
}

#[test]
fn test_channel_config_mixed_format() {
    let yaml = r#"
            enabled: true
            timeout_ms: 3000
            channels:
              simple: "https://example.com/hook"
              custom:
                url: "https://discord.com/api/webhooks/123/abc"
                body: '{"content": "{{content}}"}'
        "#;
    let cfg: MessagingActionConfig =
        serde_yaml::from_str(yaml).expect("mixed format should deserialize");
    assert_eq!(cfg.channels.len(), 2);
    assert_eq!(cfg.channels["simple"].url, "https://example.com/hook");
    assert!(cfg.channels["simple"].body.is_empty());
    let custom = &cfg.channels["custom"];
    assert_eq!(custom.url, "https://discord.com/api/webhooks/123/abc");
    assert!(!custom.body.is_empty());
    assert!(custom.headers.is_empty());
}

#[test]
fn test_validate_channel_empty_url_warning() {
    let mut config = validated_config();
    config.actions.messaging.enabled = true;
    config.actions.messaging.channels.insert(
        "bad".into(),
        ChannelConfig {
            url: "".into(),
            body: String::new(),
            headers: HashMap::new(),
        },
    );
    let warnings = config.validate().expect("should be valid");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("channels.bad") && w.contains("url is empty")),
        "expected empty-url warning, got: {:?}",
        warnings
    );
}
