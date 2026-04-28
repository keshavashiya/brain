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
async fn classify_with_history_passes_recent_turns_to_fallback() {
    use cortex::llm::{Message, Role};

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
        Message {
            role: Role::Assistant,
            content: "What's your username?".to_string(),
        },
        Message {
            role: Role::User,
            content: "Hold on…".to_string(),
        },
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
    let result = classifier.classify_regex("Remember that I like coffee");
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
async fn test_project_inspect_regex_picks_up_absolute_paths() {
    let classifier = IntentClassifier::new();
    let inputs = [
        "look at /Users/keshav/code/brain",
        "tell me about /tmp/proj",
        "summarise the project at /Users/keshav/Developer/workspace/brain",
        "describe the codebase at ~/code/brain",
        "give me a detailed report on /Users/keshav/Developer/workspace/brain",
    ];
    for input in inputs {
        let result = classifier.classify(input).await;
        match result.intent {
            Intent::ProjectInspect { path, .. } => {
                assert!(!path.is_empty(), "no path captured from '{input}'");
                assert!(
                    path.starts_with('/') || path.starts_with('~'),
                    "captured path is not absolute: {path:?} (input: {input})"
                );
            }
            other => panic!("Expected ProjectInspect for '{input}', got {other:?}"),
        }
    }
}

#[tokio::test]
async fn test_project_inspect_does_not_swallow_chat() {
    // "Look at this issue" / "tell me about Rust" must NOT match
    // ProjectInspect — only inputs with an actual path should.
    let classifier = IntentClassifier::new();
    for input in [
        "look at this issue",
        "tell me about Rust",
        "describe the project",
    ] {
        let result = classifier.classify(input).await;
        if let Intent::ProjectInspect { path, .. } = &result.intent {
            panic!(
                "input '{input}' matched ProjectInspect with path={path:?}, \
                 should have fallen through to chat/other"
            );
        }
    }
}
