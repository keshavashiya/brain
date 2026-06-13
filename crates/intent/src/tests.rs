use super::*;
use chrono::TimeZone;
use serde_json::json;

fn fixed_ts() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 5, 15, 12, 0, 0).unwrap()
}

fn sample_token(prov: Provenance) -> IntentToken {
    let mut tok = IntentToken::new(
        Verb::new("shell", "exec"),
        Object {
            kind: "command".into(),
            value: json!({ "command": "ls", "args": ["-la"] }),
        },
        prov,
        "personal".into(),
    );
    tok.required_capabilities = vec!["shell.exec".into(), "fs.read".into()];
    tok.confidence = Some(0.92);
    tok.constraints = vec![
        Constraint::PathExists {
            path: "/tmp".into(),
        },
        Constraint::UserPresent,
    ];
    tok.modifiers
        .insert("dry_run".into(), serde_json::Value::Bool(true));
    tok
}

#[test]
fn schema_constant_pinned() {
    assert_eq!(SCHEMA, "intent-token/1");
}

#[test]
fn intent_token_serde_roundtrip_user() {
    let tok = sample_token(Provenance::User {
        raw_input: "list the temp dir".into(),
        ui_origin: Some("cli".into()),
        ts: fixed_ts(),
    });
    let s = serde_json::to_string(&tok).unwrap();
    let back: IntentToken = serde_json::from_str(&s).unwrap();
    assert_eq!(back.schema, SCHEMA);
    assert_eq!(back.verb, tok.verb);
    assert_eq!(back.namespace, tok.namespace);
    assert_eq!(back.required_capabilities, tok.required_capabilities);
    assert_eq!(back.confidence, tok.confidence);
    assert_eq!(back.constraints.len(), 2);
    assert!(matches!(back.provenance, Provenance::User { .. }));
}

#[test]
fn provenance_llm_serde() {
    let p = Provenance::Llm {
        model: "claude-opus-4-7".into(),
        call_id: "call_123".into(),
        raw_input: None,
        ts: fixed_ts(),
    };
    let s = serde_json::to_string(&p).unwrap();
    assert!(s.contains("\"source\":\"llm\""));
    let back: Provenance = serde_json::from_str(&s).unwrap();
    match back {
        Provenance::Llm { model, call_id, .. } => {
            assert_eq!(model, "claude-opus-4-7");
            assert_eq!(call_id, "call_123");
        }
        _ => panic!("expected llm variant"),
    }
}

#[test]
fn provenance_reflex_serde() {
    let p = Provenance::Reflex {
        trigger: "fs:/tmp/foo".into(),
        raw_input: Some("changed".into()),
        ts: fixed_ts(),
    };
    let s = serde_json::to_string(&p).unwrap();
    assert!(s.contains("\"source\":\"reflex\""));
    let back: Provenance = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, Provenance::Reflex { .. }));
}

#[test]
fn constraint_variants_serde() {
    let cs = vec![
        Constraint::PathExists {
            path: "/etc".into(),
        },
        Constraint::NetReachable {
            host: "example.com".into(),
            port: 443,
        },
        Constraint::EnvSet {
            name: "HOME".into(),
        },
        Constraint::UserPresent,
        Constraint::Custom {
            name: "battery_above".into(),
            args: json!({ "min_pct": 20 }),
        },
    ];
    let s = serde_json::to_string(&cs).unwrap();
    assert!(s.contains("\"op\":\"path_exists\""));
    assert!(s.contains("\"op\":\"net_reachable\""));
    assert!(s.contains("\"op\":\"env_set\""));
    assert!(s.contains("\"op\":\"user_present\""));
    assert!(s.contains("\"op\":\"custom\""));
    let back: Vec<Constraint> = serde_json::from_str(&s).unwrap();
    assert_eq!(back.len(), 5);
}

#[test]
fn tool_descriptor_serde_roundtrip() {
    let td = ToolDescriptor {
        tool_id: "mcp:filesystem:read_text_file".into(),
        source: ToolSource::McpServer {
            server: "filesystem".into(),
        },
        verb: Verb::new("fs", "read"),
        description: "Read a UTF-8 text file".into(),
        input_schema: json!({ "type": "object", "properties": { "path": { "type": "string" } } }),
        output_schema: None,
        capabilities: vec!["fs.read".into()],
        annotations: ToolAnnotations {
            read_only_hint: true,
            destructive_hint: false,
            idempotent_hint: true,
        },
        usage: ToolUsage::default(),
        embedding: Some(vec![0.1, 0.2, 0.3]),
    };
    let s = serde_json::to_string(&td).unwrap();
    let back: ToolDescriptor = serde_json::from_str(&s).unwrap();
    assert_eq!(back.tool_id, td.tool_id);
    assert!(back.annotations.read_only_hint);
    assert_eq!(back.embedding.as_ref().map(|v| v.len()), Some(3));
}

#[test]
fn tool_route_variants_serde() {
    let routes = vec![
        ToolRoute::Mcp {
            server: "filesystem".into(),
            tool: "read_text_file".into(),
        },
        ToolRoute::NativeBackend {
            backend: BackendId::new("memory"),
        },
        ToolRoute::Terminal {
            session_hint: Some("default".into()),
        },
        ToolRoute::HumanConfirm {
            ask: "Run this?".into(),
        },
    ];
    let s = serde_json::to_string(&routes).unwrap();
    let back: Vec<ToolRoute> = serde_json::from_str(&s).unwrap();
    assert_eq!(back.len(), 4);
    assert!(matches!(back[0], ToolRoute::Mcp { .. }));
    assert!(matches!(back[3], ToolRoute::HumanConfirm { .. }));
}

#[test]
fn intent_error_display() {
    let e = IntentError::UnknownVerb("shell".into(), "exec".into());
    assert_eq!(e.to_string(), "unknown verb: shell.exec");
    let e = IntentError::MissingCapability("fs.read".into());
    assert_eq!(e.to_string(), "missing capability: fs.read");
}

fn sample_descriptor(tool_id: &str, verb_ns: &str, verb_action: &str) -> ToolDescriptor {
    ToolDescriptor {
        tool_id: tool_id.into(),
        source: ToolSource::McpServer {
            server: "stub".into(),
        },
        verb: Verb::new(verb_ns, verb_action),
        description: format!("{tool_id} description"),
        input_schema: json!({ "type": "object" }),
        output_schema: None,
        capabilities: vec![format!("{verb_ns}.{verb_action}")],
        annotations: ToolAnnotations::default(),
        usage: ToolUsage::default(),
        embedding: None,
    }
}

#[tokio::test]
async fn in_memory_registry_register_and_get() {
    let reg = InMemoryToolRegistry::new();
    assert!(reg.is_empty());
    reg.register(sample_descriptor("mcp:fs:read", "fs", "read"))
        .await
        .unwrap();
    assert_eq!(reg.len(), 1);
    let got = reg.get("mcp:fs:read").await.unwrap();
    assert_eq!(got.verb.namespace, "fs");
    assert!(reg.get("missing").await.is_none());
}

#[tokio::test]
async fn in_memory_registry_register_overwrites_same_id() {
    let reg = InMemoryToolRegistry::new();
    reg.register(sample_descriptor("dup", "memory", "store"))
        .await
        .unwrap();
    let mut updated = sample_descriptor("dup", "memory", "store");
    updated.description = "updated".into();
    reg.register(updated).await.unwrap();
    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get("dup").await.unwrap().description, "updated");
}

#[tokio::test]
async fn in_memory_registry_deregister_known_and_unknown() {
    let reg = InMemoryToolRegistry::new();
    reg.register(sample_descriptor("a", "fs", "read"))
        .await
        .unwrap();
    reg.deregister("a").await.unwrap();
    assert!(reg.is_empty());
    let err = reg.deregister("a").await.unwrap_err();
    match err {
        IntentError::UnknownTool(id) => assert_eq!(id, "a"),
        other => panic!("expected UnknownTool, got {other:?}"),
    }
}

fn sample_user_token(verb_ns: &str, verb_action: &str, caps: &[&str]) -> IntentToken {
    let mut tok = IntentToken::new(
        Verb::new(verb_ns, verb_action),
        Object {
            kind: "intent_args".into(),
            value: serde_json::Value::Null,
        },
        Provenance::User {
            raw_input: format!("{verb_ns}.{verb_action}"),
            ui_origin: None,
            ts: fixed_ts(),
        },
        "personal".into(),
    );
    tok.required_capabilities = caps.iter().map(|s| s.to_string()).collect();
    tok
}

fn descriptor_with(
    tool_id: &str,
    source: ToolSource,
    verb_ns: &str,
    verb_action: &str,
    caps: &[&str],
) -> ToolDescriptor {
    ToolDescriptor {
        tool_id: tool_id.into(),
        source,
        verb: Verb::new(verb_ns, verb_action),
        description: tool_id.into(),
        input_schema: json!({ "type": "object" }),
        output_schema: None,
        capabilities: caps.iter().map(|s| s.to_string()).collect(),
        annotations: ToolAnnotations::default(),
        usage: ToolUsage::default(),
        embedding: None,
    }
}

#[tokio::test]
async fn router_exact_verb_match_routes_to_mcp() {
    let registry = Arc::new(InMemoryToolRegistry::new());
    registry
        .register(descriptor_with(
            "mcp:fs:read_text_file",
            ToolSource::McpServer {
                server: "fs".into(),
            },
            "fs",
            "read",
            &["fs.read"],
        ))
        .await
        .unwrap();
    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>);
    let tok = sample_user_token("fs", "read", &["fs.read"]);
    let route = router.resolve(&tok).await.unwrap();
    match route {
        ToolRoute::Mcp { server, tool } => {
            assert_eq!(server, "fs");
            assert_eq!(tool, "read");
        }
        other => panic!("expected ToolRoute::Mcp, got {other:?}"),
    }
}

#[tokio::test]
async fn router_no_candidate_returns_human_confirm() {
    let registry = Arc::new(InMemoryToolRegistry::new()) as Arc<dyn ToolRegistry>;
    let router = DefaultIntentRouter::new(registry);
    let tok = sample_user_token("memory", "store", &[]);
    let route = router.resolve(&tok).await.unwrap();
    match route {
        ToolRoute::HumanConfirm { ask } => {
            assert!(ask.contains("memory.store"));
        }
        other => panic!("expected HumanConfirm, got {other:?}"),
    }
}

#[tokio::test]
async fn router_capability_overlap_breaks_tie() {
    let registry = Arc::new(InMemoryToolRegistry::new());
    // Two tools with same exact verb match — winner decided by capability
    // Jaccard against the token's required_capabilities.
    registry
        .register(descriptor_with(
            "tool:narrow",
            ToolSource::NativeBackend {
                backend: BackendId::new("narrow"),
            },
            "shell",
            "exec",
            &["shell.exec"],
        ))
        .await
        .unwrap();
    registry
        .register(descriptor_with(
            "tool:broad",
            ToolSource::NativeBackend {
                backend: BackendId::new("broad"),
            },
            "shell",
            "exec",
            &["shell.exec", "fs.read", "net.http"],
        ))
        .await
        .unwrap();
    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>);
    let tok = sample_user_token("shell", "exec", &["shell.exec"]);
    let route = router.resolve(&tok).await.unwrap();
    // Narrow tool has a tighter Jaccard (1/1=1.0) vs broad (1/3≈0.33).
    match route {
        ToolRoute::NativeBackend { backend } => assert_eq!(backend.as_str(), "narrow"),
        other => panic!("expected NativeBackend(narrow), got {other:?}"),
    }
}

#[tokio::test]
async fn router_namespace_match_only_still_resolves() {
    let registry = Arc::new(InMemoryToolRegistry::new());
    registry
        .register(descriptor_with(
            "tool:fs:list",
            ToolSource::Terminal,
            "fs",
            "list",
            &[],
        ))
        .await
        .unwrap();
    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>);
    // SIT wants fs.read; only fs.list registered → namespace match (+1.0).
    let tok = sample_user_token("fs", "read", &[]);
    let route = router.resolve(&tok).await.unwrap();
    assert!(matches!(route, ToolRoute::Terminal { session_hint: None }));
}

#[tokio::test]
async fn router_mcp_coarse_fallback_picks_action_match() {
    let registry = Arc::new(InMemoryToolRegistry::new());
    registry
        .register(descriptor_with(
            "mcp:gh:create_issue",
            ToolSource::McpServer {
                server: "gh".into(),
            },
            "mcp",
            "create_issue",
            &[],
        ))
        .await
        .unwrap();
    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>);
    // SIT verb is "issue.create_issue" — same action segment, different ns.
    // The MCP coarse fallback gives +0.5, enough to route over the empty
    // baseline.
    let tok = sample_user_token("issue", "create_issue", &[]);
    let route = router.resolve(&tok).await.unwrap();
    match route {
        ToolRoute::Mcp { server, tool } => {
            assert_eq!(server, "gh");
            assert_eq!(tool, "create_issue");
        }
        other => panic!("expected Mcp(gh,create_issue), got {other:?}"),
    }
}

struct AlwaysOpenBreakers;

#[async_trait::async_trait]
impl BreakerCheck for AlwaysOpenBreakers {
    async fn is_open(&self, _tool_id: &str) -> bool {
        true
    }
}

struct OpenOnly {
    tool_id: String,
}

#[async_trait::async_trait]
impl BreakerCheck for OpenOnly {
    async fn is_open(&self, tool_id: &str) -> bool {
        tool_id == self.tool_id
    }
}

#[tokio::test]
async fn router_skips_open_breakers_and_falls_through() {
    let registry = Arc::new(InMemoryToolRegistry::new());
    registry
        .register(descriptor_with(
            "mcp:fs:read_text_file",
            ToolSource::McpServer {
                server: "fs".into(),
            },
            "fs",
            "read",
            &["fs.read"],
        ))
        .await
        .unwrap();
    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>)
        .with_breakers(Arc::new(AlwaysOpenBreakers) as Arc<dyn BreakerCheck>);
    let tok = sample_user_token("fs", "read", &["fs.read"]);
    let route = router.resolve(&tok).await.unwrap();
    match route {
        ToolRoute::HumanConfirm { ask } => {
            assert!(ask.contains("fs.read"));
        }
        other => panic!("expected HumanConfirm (open breaker), got {other:?}"),
    }
}

#[tokio::test]
async fn router_open_breaker_excludes_only_that_tool() {
    let registry = Arc::new(InMemoryToolRegistry::new());
    registry
        .register(descriptor_with(
            "tool:broken",
            ToolSource::NativeBackend {
                backend: BackendId::new("broken"),
            },
            "shell",
            "exec",
            &["shell.exec"],
        ))
        .await
        .unwrap();
    registry
        .register(descriptor_with(
            "tool:healthy",
            ToolSource::NativeBackend {
                backend: BackendId::new("healthy"),
            },
            "shell",
            "exec",
            &["shell.exec"],
        ))
        .await
        .unwrap();
    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>).with_breakers(
        Arc::new(OpenOnly {
            tool_id: "tool:broken".into(),
        }) as Arc<dyn BreakerCheck>,
    );
    let tok = sample_user_token("shell", "exec", &["shell.exec"]);
    let route = router.resolve(&tok).await.unwrap();
    match route {
        ToolRoute::NativeBackend { backend } => assert_eq!(backend.as_str(), "healthy"),
        other => panic!("expected NativeBackend(healthy), got {other:?}"),
    }
}

#[test]
fn router_score_components() {
    let exact = descriptor_with("t", ToolSource::Terminal, "fs", "read", &["fs.read"]);
    let tok = sample_user_token("fs", "read", &["fs.read"]);
    let s = DefaultIntentRouter::score(&tok, &exact);
    // 2.0 (exact verb) + 1.5 * 1.0 (Jaccard 1/1) = 3.5
    assert!((s - 3.5).abs() < 1e-6, "exact-match score was {s}");
}

#[test]
fn cosine_similarity_basics() {
    // Identical direction → 1.0.
    assert!((cosine_similarity(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
    // Orthogonal → 0.0.
    assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    // Opposed → floored to 0.0 (never a penalty).
    assert!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]).abs() < 1e-6);
    // Degenerate inputs drop out to 0.0.
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
}

#[test]
fn score_hybrid_without_embedding_is_lexical_only() {
    let tok = sample_user_token("fs", "read", &["fs.read"]);
    let mut tool = descriptor_with("t", ToolSource::Terminal, "fs", "read", &["fs.read"]);
    // No query embedding → identical to lexical `score`.
    assert_eq!(
        DefaultIntentRouter::score_hybrid(&tok, &tool, None),
        DefaultIntentRouter::score(&tok, &tool),
    );
    // Tool has an embedding but the query doesn't → still lexical-only.
    tool.embedding = Some(vec![1.0, 0.0]);
    assert_eq!(
        DefaultIntentRouter::score_hybrid(&tok, &tool, None),
        DefaultIntentRouter::score(&tok, &tool),
    );
    // Query embedding present but tool has none → lexical-only.
    let mut bare = tool.clone();
    bare.embedding = None;
    assert_eq!(
        DefaultIntentRouter::score_hybrid(&tok, &bare, Some(&[1.0, 0.0])),
        DefaultIntentRouter::score(&tok, &bare),
    );
}

#[test]
fn score_hybrid_adds_weighted_cosine_when_both_present() {
    let tok = sample_user_token("fs", "read", &[]);
    let mut tool = descriptor_with("t", ToolSource::Terminal, "fs", "read", &[]);
    tool.embedding = Some(vec![1.0, 0.0]);
    let lexical = DefaultIntentRouter::score(&tok, &tool);
    // Unit cosine → lexical + ROUTER_SEMANTIC_WEIGHT (1.5).
    let s = DefaultIntentRouter::score_hybrid(&tok, &tool, Some(&[1.0, 0.0]));
    assert!((s - (lexical + 1.5)).abs() < 1e-6, "hybrid score was {s}");
}

/// Test embedder: returns a fixed vector for one phrase, orthogonal otherwise.
struct StubEmbedder;

#[async_trait::async_trait]
impl QueryEmbedder for StubEmbedder {
    async fn embed_query(&self, text: &str, _namespace: &str) -> Option<Vec<f32>> {
        if text.contains("grab that webpage") {
            Some(vec![1.0, 0.0])
        } else {
            Some(vec![0.0, 1.0])
        }
    }
}

#[tokio::test]
async fn router_semantic_term_resolves_lexically_ambiguous_verb() {
    // Two tools in the same namespace, neither an exact verb match for the
    // token — lexical scoring ties them. The semantic embedding breaks the tie.
    let registry = Arc::new(InMemoryToolRegistry::new());
    let mut http = descriptor_with(
        "net.http",
        ToolSource::NativeBackend {
            backend: BackendId::new("net"),
        },
        "net",
        "http",
        &[],
    );
    http.embedding = Some(vec![1.0, 0.0]); // aligned with "grab that webpage"
    let mut ping = descriptor_with(
        "net.ping",
        ToolSource::NativeBackend {
            backend: BackendId::new("net"),
        },
        "net",
        "ping",
        &[],
    );
    ping.embedding = Some(vec![0.0, 1.0]); // orthogonal
    registry.register(http).await.unwrap();
    registry.register(ping).await.unwrap();

    let router = DefaultIntentRouter::new(registry as Arc<dyn ToolRegistry>)
        .with_embedder(Arc::new(StubEmbedder));
    // Token verb is `net.fetch` (namespace-only match against both), surface
    // text "grab that webpage" — zero lexical overlap with either tool's verb.
    let mut tok = sample_user_token("net", "fetch", &[]);
    tok.provenance = Provenance::User {
        raw_input: "grab that webpage".into(),
        ui_origin: None,
        ts: fixed_ts(),
    };
    let route = router.resolve(&tok).await.unwrap();
    match route {
        ToolRoute::NativeBackend { backend } => assert_eq!(backend.as_str(), "net"),
        other => panic!("expected NativeBackend(net), got {other:?}"),
    }
}

#[tokio::test]
async fn in_memory_registry_list_returns_all() {
    let reg = InMemoryToolRegistry::new();
    reg.register(sample_descriptor("a", "fs", "read"))
        .await
        .unwrap();
    reg.register(sample_descriptor("b", "shell", "exec"))
        .await
        .unwrap();
    let mut ids: Vec<String> = reg.list().await.into_iter().map(|t| t.tool_id).collect();
    ids.sort();
    assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn intent_token_new_defaults() {
    let tok = IntentToken::new(
        Verb::new("memory", "store"),
        Object {
            kind: "fact".into(),
            value: json!({"subject": "x"}),
        },
        Provenance::User {
            raw_input: "remember x".into(),
            ui_origin: None,
            ts: fixed_ts(),
        },
        "personal".into(),
    );
    assert_eq!(tok.schema, SCHEMA);
    assert!(tok.parent_id.is_none());
    assert!(tok.modifiers.is_empty());
    assert!(tok.required_capabilities.is_empty());
    assert!(tok.constraints.is_empty());
}
