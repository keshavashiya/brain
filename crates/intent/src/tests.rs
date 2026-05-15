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
