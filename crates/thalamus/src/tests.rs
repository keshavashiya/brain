use std::sync::Arc;

use cortex::actions::Action;

use super::*;
use crate::classifier::PATTERNS;

#[test]
fn all_patterns_compile() {
    for p in PATTERNS {
        let _ = &**p.regex;
    }
}

struct MockFallback {
    response: Option<Classification>,
}

/// Records every (input, history) pair the classifier passes in so tests
/// can assert that history actually reaches the LLM call.
struct RecordingFallback {
    response: Classification,
    seen: std::sync::Mutex<Vec<(String, Vec<cortex::llm::Message>)>>,
}

#[async_trait::async_trait]
impl IntentFallback for RecordingFallback {
    async fn classify_with_llm(&self, input: &str) -> Option<Classification> {
        self.classify_with_history(input, &[]).await
    }
    async fn classify_with_history(
        &self,
        input: &str,
        history: &[cortex::llm::Message],
    ) -> Option<Classification> {
        self.seen
            .lock()
            .unwrap()
            .push((input.to_string(), history.to_vec()));
        Some(self.response.clone())
    }
}

impl MockFallback {
    fn chat() -> Self {
        Self {
            response: Some(Classification {
                intent: Intent::Chat {
                    content: "mock".to_string(),
                },
                confidence: 0.7,
                method: ClassificationMethod::Llm,
                extracted_facts: Vec::new(),
            }),
        }
    }

    fn unavailable() -> Self {
        Self { response: None }
    }
}

#[async_trait::async_trait]
impl IntentFallback for MockFallback {
    async fn classify_with_llm(&self, _input: &str) -> Option<Classification> {
        self.response.clone()
    }
}

#[tokio::test]
async fn test_classify_store_fact_regex_fallback() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Remember that I like coffee").await;

    assert!(
        matches!(result.intent, Intent::StoreFact { .. }),
        "Expected StoreFact, got {:?}",
        result.intent
    );
    assert_eq!(result.method, ClassificationMethod::Regex);
}

#[tokio::test]
async fn test_classify_recall_regex_fallback() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Tell me about Project Brain").await;

    assert!(
        matches!(result.intent, Intent::Recall { .. }),
        "Expected Recall, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_conversational_meta_question_is_not_recall() {
    // "What did we discuss?" is a conversational follow-up — the LLM should
    // answer from session history, not from a memory lookup of "we discuss".
    let classifier = IntentClassifier::new();
    let result = classifier.classify("What did we discuss yesterday?").await;

    assert!(
        !matches!(result.intent, Intent::Recall { .. }),
        "Conversational follow-up should not be Recall, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_classify_execute_command_regex_fallback() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Run ls -la").await;

    assert!(
        matches!(result.intent, Intent::ExecuteCommand { .. }),
        "Expected ExecuteCommand, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn explicit_store_strips_imperative_wrapper() {
    let classifier = IntentClassifier::new();
    for (input, want_object) in [
        (
            "store this fact: my deploy script is ops/deploy.sh",
            "my deploy script is ops/deploy.sh",
        ),
        ("remember that I prefer dark mode", "I prefer dark mode"),
        (
            "note the staging DB is read-only",
            "the staging DB is read-only",
        ),
        (
            "store that the API key rotates monthly",
            "the API key rotates monthly",
        ),
    ] {
        let result = classifier.classify(input).await;
        match result.intent {
            Intent::StoreFact {
                subject,
                predicate,
                object,
            } => {
                assert_eq!(subject, "user");
                assert_eq!(predicate, "said");
                assert_eq!(object, want_object, "input {input:?} kept the wrapper");
            }
            other => panic!("{input:?}: expected StoreFact, got {other:?}"),
        }
    }
}

#[test]
fn normalize_command_strips_filler_and_wrappers() {
    use crate::classifier::normalize_command;

    // Plain command is untouched.
    assert_eq!(normalize_command("ls -la"), "ls -la");
    // Conversational filler before the binary is dropped.
    assert_eq!(normalize_command("the command: ls ~/.brain"), "ls ~/.brain");
    assert_eq!(normalize_command("the following: cargo test"), "cargo test");
    assert_eq!(normalize_command("command: git status"), "git status");
    // Shell wrappers around the whole command are peeled.
    assert_eq!(normalize_command("`git status`"), "git status");
    assert_eq!(normalize_command("$(cargo build)"), "cargo build");
    assert_eq!(normalize_command("\"echo hi\""), "echo hi");
    // A bare `cmd` token is a real binary, not filler.
    assert_eq!(normalize_command("cmd /c dir"), "cmd /c dir");
    // …but the explicit `cmd:` preamble is filler.
    assert_eq!(normalize_command("cmd: dir"), "dir");
}

#[tokio::test]
async fn test_classify_execute_command_with_filler_phrasing() {
    let classifier = IntentClassifier::new();
    for (input, want) in [
        ("run the command: ls -la", "ls"),
        ("please run `git status`", "git"),
        ("execute the following: cargo test", "cargo"),
    ] {
        let result = classifier.classify(input).await;
        match result.intent {
            Intent::ExecuteCommand { command, .. } => {
                assert_eq!(command, want, "{input:?} parsed binary as {command:?}");
            }
            other => panic!("{input:?}: expected ExecuteCommand, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn classify_with_history_passes_recent_turns_to_fallback() {
    use cortex::llm::Message;

    let recording = Arc::new(RecordingFallback {
        response: Classification {
            intent: Intent::Chat {
                content: "ack".to_string(),
            },
            confidence: 0.7,
            method: ClassificationMethod::Llm,
            extracted_facts: Vec::new(),
        },
        seen: std::sync::Mutex::new(Vec::new()),
    });
    let classifier = IntentClassifier::new().with_llm_fallback(recording.clone());
    let history = vec![
        Message::assistant("What's your username?"),
        Message::user("Hold on…"),
    ];
    let _ = classifier
        .classify_with_history("username : keshavashiya", &history)
        .await;
    let seen = recording.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "fallback should be invoked exactly once");
    let (input, hist) = &seen[0];
    assert_eq!(input, "username : keshavashiya");
    assert_eq!(hist.len(), 2, "history should be forwarded verbatim");
    assert_eq!(hist[0].content, "What's your username?");
}

#[tokio::test]
async fn legacy_classify_passes_no_history() {
    let recording = Arc::new(RecordingFallback {
        response: Classification {
            intent: Intent::Chat {
                content: "ack".to_string(),
            },
            confidence: 0.7,
            method: ClassificationMethod::Llm,
            extracted_facts: Vec::new(),
        },
        seen: std::sync::Mutex::new(Vec::new()),
    });
    let classifier = IntentClassifier::new().with_llm_fallback(recording.clone());
    let _ = classifier.classify("anything ambiguous").await;
    let seen = recording.seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(
        seen[0].1.is_empty(),
        "legacy classify() must not synthesise history"
    );
}

#[tokio::test]
async fn test_classify_decompose_task_uses_llm_when_available() {
    let classifier = IntentClassifier::new().with_llm_fallback(Arc::new(MockFallback {
        response: Some(Classification {
            intent: Intent::DecomposeTask {
                request: "build a CSV export feature".to_string(),
            },
            confidence: 0.7,
            method: ClassificationMethod::Llm,
            extracted_facts: Vec::new(),
        }),
    }));

    let result = classifier.classify("build a CSV export feature").await;
    assert_eq!(result.method, ClassificationMethod::Llm);
    assert!(
        matches!(result.intent, Intent::DecomposeTask { .. }),
        "Expected DecomposeTask, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_classify_query_agents_unfiltered() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("what agents do you have").await;
    match result.intent {
        Intent::QueryAgents { filter } => assert!(filter.is_empty()),
        other => panic!("Expected QueryAgents, got {other:?}"),
    }
}

#[tokio::test]
async fn test_classify_query_agents_filtered() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("which agents can code rust").await;
    match result.intent {
        Intent::QueryAgents { filter } => assert_eq!(filter.to_lowercase(), "rust"),
        other => panic!("Expected QueryAgents, got {other:?}"),
    }
}

#[tokio::test]
async fn test_classify_query_agents_why_not() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("why aren't you using aider").await;
    match result.intent {
        Intent::QueryAgents { filter } => assert_eq!(filter.to_lowercase(), "aider"),
        other => panic!("Expected QueryAgents, got {other:?}"),
    }
}

#[tokio::test]
async fn test_classify_web_search_regex_fallback() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Search for Rust programming").await;

    assert!(
        matches!(result.intent, Intent::WebSearch { .. }),
        "Expected WebSearch, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_classify_web_search_natural_phrasing_regex_fallback() {
    let classifier = IntentClassifier::new();

    let result = classifier
        .classify("can you search about Keshav Ashiya")
        .await;
    assert!(
        matches!(result.intent, Intent::WebSearch { .. }),
        "Expected WebSearch for 'can you search about ...', got {:?}",
        result.intent
    );

    let result = classifier.classify("please look up Rust language").await;
    assert!(
        matches!(result.intent, Intent::WebSearch { .. }),
        "Expected WebSearch for 'please look up ...', got {:?}",
        result.intent
    );

    let result = classifier
        .classify("could you find information about AI")
        .await;
    assert!(
        matches!(result.intent, Intent::WebSearch { .. }),
        "Expected WebSearch for 'could you find ...', got {:?}",
        result.intent
    );

    let result = classifier.classify("google Keshav Ashiya").await;
    assert!(
        matches!(result.intent, Intent::WebSearch { .. }),
        "Expected WebSearch for 'google ...', got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_classify_schedule_regex_fallback() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Remind me to call mom").await;

    assert!(
        matches!(result.intent, Intent::Schedule { .. }),
        "Expected Schedule, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_classify_status_slash_command() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/status").await;

    assert_eq!(result.intent, Intent::SystemStatus);
    assert_eq!(result.method, ClassificationMethod::Regex);
}

#[tokio::test]
async fn test_classify_chat_fallback() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Hello, how are you?").await;

    assert!(
        matches!(result.intent, Intent::Chat { .. }),
        "Expected Chat, got {:?}",
        result.intent
    );
    assert_eq!(result.method, ClassificationMethod::Fallback);
}

#[tokio::test]
async fn test_llm_classifies_ambiguous_input() {
    let classifier = IntentClassifier::new().with_llm_fallback(Arc::new(MockFallback::chat()));
    let result = classifier.classify("What's the weather?").await;
    assert_eq!(result.method, ClassificationMethod::Llm);
    assert!(
        matches!(result.intent, Intent::Chat { .. }),
        "LLM should classify ambiguous input; got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_regex_fallback_when_llm_unavailable() {
    let classifier =
        IntentClassifier::new().with_llm_fallback(Arc::new(MockFallback::unavailable()));

    let result = classifier.classify("Remember that I like coffee").await;
    assert_eq!(result.method, ClassificationMethod::Regex);
    assert!(
        matches!(result.intent, Intent::StoreFact { .. }),
        "Regex fallback should work; got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_do_you_remember_is_not_store_fact() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Do you remember my birthday?").await;

    assert!(
        !matches!(result.intent, Intent::StoreFact { .. }),
        "'Do you remember...' should NOT be StoreFact, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn test_find_the_bug_is_not_web_search() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("Find the bug in my code").await;

    assert!(
        !matches!(result.intent, Intent::WebSearch { .. }),
        "'Find the bug...' should NOT be WebSearch, got {:?}",
        result.intent
    );
}

#[test]
fn test_intent_to_action_store_fact() {
    let router = SignalRouter::new();
    let intent = Intent::StoreFact {
        subject: "user".to_string(),
        predicate: "likes".to_string(),
        object: "coffee".to_string(),
    };

    let action = router.intent_to_action(&intent);
    assert!(
        matches!(action, Some(Action::StoreFact { .. })),
        "Expected StoreFact action"
    );
}

#[test]
fn test_intent_to_action_system_status() {
    let router = SignalRouter::new();
    let intent = Intent::SystemStatus;

    let action = router.intent_to_action(&intent);
    assert!(action.is_none(), "SystemStatus should not map to action");
}

#[test]
fn test_regex_classification_has_empty_extracted_facts() {
    let classifier = IntentClassifier::new();
    // After audit Issues 13 + 14 the `remember …` / `forget …` prefixes
    // are handled by `classify_explicit`, so they no longer reach the
    // regex pipeline. Use a `recall` input that still routes through
    // `classify_regex` to exercise the same invariant.
    let result = classifier.classify_regex("recall my notes about Rust");
    assert!(
        result.unwrap().extracted_facts.is_empty(),
        "Regex classification should have empty extracted_facts"
    );
}

#[test]
fn test_fallback_classification_has_empty_extracted_facts() {
    let classifier = IntentClassifier::new();
    let result = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(classifier.classify("Hello, how are you?"));
    assert!(
        result.extracted_facts.is_empty(),
        "Fallback classification should have empty extracted_facts"
    );
}

#[test]
fn test_parse_json_payload_with_facts() {
    let json = r#"{
            "intent": "chat",
            "content": "I'm Keshav, a software engineer",
            "facts": [
                {"subject": "user", "predicate": "name_is", "object": "Keshav"},
                {"subject": "user", "predicate": "role_is", "object": "software engineer"}
            ]
        }"#;

    let payload = LlmIntentFallback::parse_json_payload(json).unwrap();
    assert_eq!(payload.intent, "chat");
    let facts = payload.facts.unwrap();
    assert_eq!(facts.len(), 2);
    assert_eq!(facts[0].predicate.as_deref(), Some("name_is"));
    assert_eq!(facts[0].object.as_deref(), Some("Keshav"));
    assert_eq!(facts[1].predicate.as_deref(), Some("role_is"));
    assert_eq!(facts[1].object.as_deref(), Some("software engineer"));
}

#[test]
fn test_parse_json_payload_with_empty_facts() {
    let json = r#"{"intent": "chat", "content": "hello", "facts": []}"#;
    let payload = LlmIntentFallback::parse_json_payload(json).unwrap();
    assert!(payload.facts.unwrap().is_empty());
}

#[test]
fn test_parse_json_payload_without_facts_field() {
    let json = r#"{"intent": "chat", "content": "hello"}"#;
    let payload = LlmIntentFallback::parse_json_payload(json).unwrap();
    assert!(payload.facts.is_none());
}

#[test]
fn test_extracted_fact_filters_empty_fields() {
    let raw_facts = vec![
        LlmFactPayload {
            subject: Some("user".to_string()),
            predicate: Some("name_is".to_string()),
            object: Some("Keshav".to_string()),
        },
        LlmFactPayload {
            subject: Some("user".to_string()),
            predicate: Some("".to_string()),
            object: Some("something".to_string()),
        },
        LlmFactPayload {
            subject: Some("user".to_string()),
            predicate: Some("likes".to_string()),
            object: None,
        },
    ];

    let extracted: Vec<ExtractedFact> = raw_facts
        .into_iter()
        .filter_map(|f| {
            let predicate = f.predicate.unwrap_or_default();
            let object = f.object.unwrap_or_default();
            if predicate.is_empty() || object.is_empty() {
                None
            } else {
                Some(ExtractedFact {
                    subject: f.subject.unwrap_or_else(|| "user".to_string()),
                    predicate,
                    object,
                })
            }
        })
        .collect();

    assert_eq!(extracted.len(), 1);
    assert_eq!(extracted[0].predicate, "name_is");
    assert_eq!(extracted[0].object, "Keshav");
}

#[test]
fn test_classifier_prompt_mentions_query_agents() {
    assert!(super::CLASSIFIER_SYSTEM_PROMPT.contains("query_agents"));
    assert!(super::CLASSIFIER_SYSTEM_PROMPT.contains("what agents do you have"));
}

#[tokio::test]
async fn test_approve_with_uuid_matches_fast_path() {
    let classifier = IntentClassifier::new();
    let result = classifier
        .classify("approve 9aa1b54e-23fd-4601-9355-fac5a0e386aa")
        .await;
    match result.intent {
        Intent::RespondToApproval { nonce, decision } => {
            assert_eq!(nonce, "9aa1b54e-23fd-4601-9355-fac5a0e386aa");
            assert_eq!(decision.to_lowercase(), "approve");
        }
        other => panic!("Expected RespondToApproval, got {other:?}"),
    }
    assert_eq!(result.method, ClassificationMethod::Regex);
}

#[tokio::test]
async fn test_reject_with_uuid_matches_fast_path() {
    let classifier = IntentClassifier::new();
    let result = classifier
        .classify("reject 9aa1b54e-23fd-4601-9355-fac5a0e386aa")
        .await;
    match result.intent {
        Intent::RespondToApproval { nonce, decision } => {
            assert_eq!(nonce, "9aa1b54e-23fd-4601-9355-fac5a0e386aa");
            assert_eq!(decision.to_lowercase(), "reject");
        }
        other => panic!("Expected RespondToApproval, got {other:?}"),
    }
}

#[tokio::test]
async fn test_bare_approve_classifies_to_respond() {
    let classifier = IntentClassifier::new();
    for input in ["approve", "y", "yes", "Approve"] {
        let result = classifier.classify(input).await;
        match result.intent {
            Intent::RespondToApproval { nonce, decision } => {
                assert!(nonce.is_empty(), "bare {input} carried a nonce: {nonce}");
                assert_eq!(decision, "approve", "{input} did not normalise to approve");
            }
            other => panic!("Expected RespondToApproval for '{input}', got {other:?}"),
        }
    }
    for input in ["reject", "n", "no", "Reject"] {
        let result = classifier.classify(input).await;
        match result.intent {
            Intent::RespondToApproval { nonce, decision } => {
                assert!(nonce.is_empty(), "bare {input} carried a nonce: {nonce}");
                assert_eq!(decision, "reject", "{input} did not normalise to reject");
            }
            other => panic!("Expected RespondToApproval for '{input}', got {other:?}"),
        }
    }
}

#[tokio::test]
async fn cancel_signal_regex_captures_uuid() {
    let classifier = IntentClassifier::new();
    let result = classifier
        .classify("cancel signal 550e8400-e29b-41d4-a716-446655440000")
        .await;
    match result.intent {
        Intent::CancelSignal { signal_id } => {
            assert_eq!(signal_id, "550e8400-e29b-41d4-a716-446655440000");
        }
        other => panic!("Expected CancelSignal, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_signal_does_not_collide_with_cancel_task() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("cancel task 42").await;
    match result.intent {
        Intent::CancelTask { .. } => {} // expected
        other => panic!("'cancel task 42' should classify as CancelTask, got {other:?}"),
    }
}

#[test]
fn to_intent_token_store_fact_maps_to_memory_store() {
    let intent = Intent::StoreFact {
        subject: "user".into(),
        predicate: "likes".into(),
        object: "rust".into(),
    };
    let prov = intent::Provenance::User {
        raw_input: "remember that I like rust".into(),
        ui_origin: None,
        ts: chrono::Utc::now(),
    };
    let tok = intent.to_intent_token(prov, "personal").expect("mappable");
    assert_eq!(tok.verb.namespace, "memory");
    assert_eq!(tok.verb.action, "store");
    assert_eq!(tok.namespace, "personal");
    assert_eq!(tok.required_capabilities, vec!["memory.store".to_string()]);
    assert_eq!(tok.object.value["subject"], "user");
}

#[test]
fn to_intent_token_chat_is_unmappable() {
    let intent = Intent::Chat {
        content: "hi".into(),
    };
    let prov = intent::Provenance::User {
        raw_input: "hi".into(),
        ui_origin: None,
        ts: chrono::Utc::now(),
    };
    assert!(intent.to_intent_token(prov, "personal").is_none());
}

#[test]
fn to_intent_token_execute_command_maps_to_shell_exec() {
    let intent = Intent::ExecuteCommand {
        command: "ls".into(),
        args: vec!["-la".into()],
    };
    let prov = intent::Provenance::User {
        raw_input: "run ls -la".into(),
        ui_origin: None,
        ts: chrono::Utc::now(),
    };
    let tok = intent.to_intent_token(prov, "personal").unwrap();
    assert_eq!(tok.verb.namespace, "shell");
    assert_eq!(tok.verb.action, "exec");
    assert_eq!(tok.object.value["command"], "ls");
    assert_eq!(tok.object.value["args"][0], "-la");
}

#[test]
fn to_intent_token_mount_mcp_server_maps() {
    let intent = Intent::MountMcpServer {
        name: "fs".into(),
        transport: "stdio".into(),
        command_or_url: "mcp-fs".into(),
    };
    let prov = intent::Provenance::User {
        raw_input: "/mcp-mount fs stdio mcp-fs".into(),
        ui_origin: None,
        ts: chrono::Utc::now(),
    };
    let tok = intent.to_intent_token(prov, "personal").unwrap();
    assert_eq!(tok.verb.namespace, "mcp");
    assert_eq!(tok.verb.action, "mount");
    assert_eq!(tok.required_capabilities, vec!["mcp.mount".to_string()]);
}

#[test]
fn to_intent_token_round_trip_via_toolcall_variant() {
    let mut tok = intent::IntentToken::new(
        intent::Verb::new("fs", "read"),
        intent::Object {
            kind: "intent_args".into(),
            value: serde_json::json!({ "path": "/etc" }),
        },
        intent::Provenance::Llm {
            model: "claude-opus-4-7".into(),
            call_id: "abc".into(),
            raw_input: None,
            ts: chrono::Utc::now(),
        },
        "personal".into(),
    );
    tok.required_capabilities = vec!["fs.read".into()];
    let original = tok.clone();
    let wrapped = Intent::ToolCall(Box::new(tok));
    let back = wrapped
        .to_intent_token(
            intent::Provenance::User {
                raw_input: "ignored — ToolCall preserves the inner token".into(),
                ui_origin: None,
                ts: chrono::Utc::now(),
            },
            "ignored",
        )
        .unwrap();
    assert_eq!(back.verb, original.verb);
    assert_eq!(back.required_capabilities, original.required_capabilities);
    assert_eq!(back.namespace, original.namespace);
}

#[tokio::test]
async fn tool_slash_classifies_to_toolcall_with_payload() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify(r#"/tool mcp.echo {"text":"hi"}"#).await;
    match result.intent {
        Intent::ToolCall(token) => {
            assert_eq!(token.verb.namespace, "mcp");
            assert_eq!(token.verb.action, "echo");
            assert_eq!(token.object.value["text"], "hi");
            assert!(matches!(token.provenance, intent::Provenance::User { .. }));
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_slash_without_payload_uses_null_value() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/tool fs.read").await;
    match result.intent {
        Intent::ToolCall(token) => {
            assert_eq!(token.verb.namespace, "fs");
            assert_eq!(token.verb.action, "read");
            assert!(token.object.value.is_null());
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_slash_with_invalid_json_falls_back_to_string_value() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/tool fs.write not-json").await;
    match result.intent {
        Intent::ToolCall(token) => {
            assert_eq!(token.object.value, serde_json::json!("not-json"));
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_slash_with_missing_dot_does_not_match() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/tool justverb").await;
    assert!(
        !matches!(result.intent, Intent::ToolCall(_)),
        "expected fallback, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn approval_list_slash_classifies_to_list_standing_approvals() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/approval-list").await;
    assert!(
        matches!(result.intent, Intent::ListStandingApprovals),
        "expected ListStandingApprovals, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn approval_revoke_slash_carries_id() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/approval-revoke abc-123").await;
    match result.intent {
        Intent::RevokeStandingApproval { id } => assert_eq!(id, "abc-123"),
        other => panic!("expected RevokeStandingApproval, got {other:?}"),
    }
}

#[tokio::test]
async fn approval_revoke_slash_without_id_falls_through() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/approval-revoke").await;
    assert!(
        !matches!(result.intent, Intent::RevokeStandingApproval { .. }),
        "bare /approval-revoke must not classify (id is required)"
    );
}

#[tokio::test]
async fn task_list_slash_classifies_to_list_tasks() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/task-list").await;
    assert!(
        matches!(result.intent, Intent::ListTasks),
        "expected ListTasks, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn task_status_slash_carries_id() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/task-status t-42").await;
    match result.intent {
        Intent::TaskStatus { task_id } => assert_eq!(task_id, "t-42"),
        other => panic!("expected TaskStatus, got {other:?}"),
    }
}

#[tokio::test]
async fn task_cancel_slash_carries_id() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/task-cancel abc-123").await;
    match result.intent {
        Intent::CancelTask { task_id } => assert_eq!(task_id, "abc-123"),
        other => panic!("expected CancelTask, got {other:?}"),
    }
}

#[tokio::test]
async fn task_cancel_slash_without_id_falls_through() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/task-cancel").await;
    assert!(
        !matches!(result.intent, Intent::CancelTask { .. }),
        "bare /task-cancel must not classify (id is required)"
    );
}

// ─── Intent::category (Issue 149) ───────────────────────────────────────────
//
// Build one representative of every variant so the category mapping is
// driven through the exhaustive match in `Intent::category`. Catches:
// (a) a future variant landing without a category declaration (compile
// error via exhaustiveness in `category()`), and (b) a category landing
// in the enum without any variant claiming it (the coverage assertion).

fn every_intent_variant() -> Vec<Intent> {
    use chrono::Utc;
    use intent::{IntentToken, Object, Provenance, Verb};

    vec![
        // Inspection
        Intent::Recall { query: "x".into() },
        Intent::MemorySummary,
        Intent::SystemStatus,
        Intent::ProactivityStatus,
        Intent::BudgetStatus { window: None },
        Intent::ListApprovals { status: None },
        Intent::ListStandingApprovals,
        Intent::ListSchedules,
        Intent::ListTasks,
        Intent::TaskStatus {
            task_id: "t".into(),
        },
        Intent::QueryAgents {
            filter: String::new(),
        },
        Intent::QueryAudit {
            filter: None,
            since: None,
            limit: None,
        },
        Intent::ListChannels,
        Intent::ChannelPreferences {
            namespace: None,
            category: None,
        },
        Intent::ListTerminalSessions,
        Intent::ListMcpServers,
        // Memory
        Intent::StoreFact {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        },
        Intent::Forget { target: "x".into() },
        // Action
        Intent::ExecuteCommand {
            command: "ls".into(),
            args: vec![],
        },
        Intent::WebSearch { query: "x".into() },
        Intent::SendMessage {
            channel: "c".into(),
            recipient: "r".into(),
            content: "hi".into(),
        },
        Intent::DelegateTask {
            agent: "claude-code".into(),
            prompt: "do".into(),
        },
        // Lifecycle
        Intent::Schedule {
            description: "d".into(),
            cron: None,
        },
        Intent::CancelSchedule { id: "1".into() },
        Intent::DecomposeTask {
            request: "build".into(),
        },
        Intent::CancelTask {
            task_id: "1".into(),
        },
        Intent::CancelSignal {
            signal_id: "1".into(),
        },
        Intent::OpenTerminalSession {
            program: "bash".into(),
            args: vec![],
            cwd: None,
        },
        Intent::CloseTerminalSession {
            session_id: "s".into(),
        },
        Intent::MountMcpServer {
            name: "m".into(),
            transport: "stdio".into(),
            command_or_url: "x".into(),
        },
        Intent::UnmountMcpServer { name: "m".into() },
        // Governance
        Intent::RespondToApproval {
            nonce: "n".into(),
            decision: "approve".into(),
        },
        Intent::RevokeStandingApproval { id: "g".into() },
        Intent::PruneAudit {
            older_than: "30d".into(),
        },
        Intent::SetChannelPreference {
            channel: "c".into(),
            category: "k".into(),
            weight: 1.0,
            pinned: false,
        },
        Intent::SetProactivity {
            enabled: true,
            until: None,
        },
        // Capability
        Intent::ToolCall(Box::new(IntentToken::new(
            Verb::new("memory", "store"),
            Object {
                kind: "intent_args".into(),
                value: serde_json::Value::Null,
            },
            Provenance::User {
                raw_input: "test".into(),
                ui_origin: None,
                ts: Utc::now(),
            },
            "personal".into(),
        ))),
        // Conversation
        Intent::Chat {
            content: "hi".into(),
        },
    ]
}

#[test]
fn category_mapping_matches_intended_taxonomy() {
    // Spot-check one representative per category. The exhaustive match in
    // `Intent::category` covers the rest at compile time.
    assert_eq!(
        Intent::Recall { query: "x".into() }.category(),
        IntentCategory::Inspection
    );
    assert_eq!(
        Intent::StoreFact {
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
        }
        .category(),
        IntentCategory::Memory
    );
    assert_eq!(
        Intent::ExecuteCommand {
            command: "ls".into(),
            args: vec![],
        }
        .category(),
        IntentCategory::Action
    );
    assert_eq!(
        Intent::Schedule {
            description: "d".into(),
            cron: None,
        }
        .category(),
        IntentCategory::Lifecycle
    );
    assert_eq!(
        Intent::RespondToApproval {
            nonce: "n".into(),
            decision: "approve".into(),
        }
        .category(),
        IntentCategory::Governance
    );
    assert_eq!(
        Intent::Chat {
            content: "hi".into(),
        }
        .category(),
        IntentCategory::Conversation
    );
    // ToolCall → Capability
    let token = intent::IntentToken::new(
        intent::Verb::new("custom", "thing"),
        intent::Object {
            kind: "intent_args".into(),
            value: serde_json::Value::Null,
        },
        intent::Provenance::User {
            raw_input: "x".into(),
            ui_origin: None,
            ts: chrono::Utc::now(),
        },
        "personal".into(),
    );
    assert_eq!(
        Intent::ToolCall(Box::new(token)).category(),
        IntentCategory::Capability
    );
}

#[test]
fn every_category_has_at_least_one_variant() {
    use std::collections::HashSet;

    let categories: HashSet<IntentCategory> = every_intent_variant()
        .iter()
        .map(Intent::category)
        .collect();

    for expected in [
        IntentCategory::Inspection,
        IntentCategory::Memory,
        IntentCategory::Action,
        IntentCategory::Lifecycle,
        IntentCategory::Governance,
        IntentCategory::Capability,
        IntentCategory::Conversation,
    ] {
        assert!(
            categories.contains(&expected),
            "no Intent variant maps to {expected:?} \
             — either drop the category or add a variant"
        );
    }
}

/// Tripwire: `every_intent_variant()` must enumerate one example per
/// `Intent` variant. If the enum grows without the helper being
/// updated, the per-variant category coverage assertions silently
/// stop checking the new variant. The exact count is intentional —
/// bump it deliberately when adding a variant, and add a sample to
/// `every_intent_variant()` in the same change.
#[test]
fn every_intent_variant_helper_is_exhaustive() {
    assert_eq!(
        every_intent_variant().len(),
        38,
        "every_intent_variant() must list one example per Intent variant; \
         update it (and bump this count) when the enum changes"
    );
}

/// Stronger than the spot-check above: assert *every* sample variant
/// lands in the category the taxonomy comment promised. Catches the
/// case where a future variant is added to the `category()` match
/// arm under the wrong heading.
#[test]
fn every_intent_variant_has_the_expected_category() {
    fn expected(intent: &Intent) -> IntentCategory {
        match intent {
            Intent::Recall { .. }
            | Intent::MemorySummary
            | Intent::SystemStatus
            | Intent::ProactivityStatus
            | Intent::BudgetStatus { .. }
            | Intent::ListApprovals { .. }
            | Intent::ListStandingApprovals
            | Intent::ListSchedules
            | Intent::ListTasks
            | Intent::TaskStatus { .. }
            | Intent::QueryAgents { .. }
            | Intent::QueryAudit { .. }
            | Intent::ListChannels
            | Intent::ChannelPreferences { .. }
            | Intent::ListTerminalSessions
            | Intent::ListMcpServers
            | Intent::ListCapabilities => IntentCategory::Inspection,
            Intent::StoreFact { .. } | Intent::Forget { .. } => IntentCategory::Memory,
            Intent::ExecuteCommand { .. }
            | Intent::WebSearch { .. }
            | Intent::SendMessage { .. }
            | Intent::DelegateTask { .. } => IntentCategory::Action,
            Intent::Schedule { .. }
            | Intent::CancelSchedule { .. }
            | Intent::DecomposeTask { .. }
            | Intent::CancelTask { .. }
            | Intent::CancelSignal { .. }
            | Intent::OpenTerminalSession { .. }
            | Intent::CloseTerminalSession { .. }
            | Intent::MountMcpServer { .. }
            | Intent::UnmountMcpServer { .. } => IntentCategory::Lifecycle,
            Intent::RespondToApproval { .. }
            | Intent::RevokeStandingApproval { .. }
            | Intent::PruneAudit { .. }
            | Intent::SetChannelPreference { .. }
            | Intent::SetProactivity { .. } => IntentCategory::Governance,
            Intent::ToolCall(_) => IntentCategory::Capability,
            Intent::Chat { .. } => IntentCategory::Conversation,
        }
    }

    for intent in every_intent_variant() {
        let want = expected(&intent);
        let got = intent.category();
        assert_eq!(
            got, want,
            "intent {intent:?} categorised as {got:?}, expected {want:?}"
        );
    }
}

/// `IntentCategory` is `Serialize` + `Deserialize` because it surfaces
/// in observability payloads (`BrainEvent::IntentClassified`, audit
/// metadata). Pin the wire form so downstream consumers don't break
/// when the enum changes.
#[test]
fn intent_category_serde_wire_form_is_stable() {
    let pairs = [
        (IntentCategory::Inspection, "\"Inspection\""),
        (IntentCategory::Memory, "\"Memory\""),
        (IntentCategory::Action, "\"Action\""),
        (IntentCategory::Lifecycle, "\"Lifecycle\""),
        (IntentCategory::Governance, "\"Governance\""),
        (IntentCategory::Capability, "\"Capability\""),
        (IntentCategory::Conversation, "\"Conversation\""),
    ];
    for (cat, wire) in pairs {
        let s = serde_json::to_string(&cat).unwrap();
        assert_eq!(s, wire, "wire form for {cat:?}");
        let back: IntentCategory = serde_json::from_str(&s).unwrap();
        assert_eq!(back, cat);
    }
}
