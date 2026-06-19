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

#[tokio::test]
async fn explicit_store_strips_politeness_wrapper() {
    let classifier = IntentClassifier::new();
    // The reported non-determinism: "Please remember X" missed the
    // deterministic explicit path (it matched on "remember", not "please")
    // and fell through to the LLM lottery. These must now route to StoreFact
    // deterministically (method = Regex), with the wrapper peeled off.
    for (input, want_object) in [
        (
            "Please remember: VORTEX key rotates on the 15th",
            "VORTEX key rotates on the 15th",
        ),
        (
            "please remember that I prefer dark mode",
            "I prefer dark mode",
        ),
        (
            "pls note the staging DB is read-only",
            "the staging DB is read-only",
        ),
        (
            "please kindly remember the API key rotates monthly",
            "the API key rotates monthly",
        ),
    ] {
        let result = classifier.classify(input).await;
        assert_eq!(
            result.method,
            ClassificationMethod::Regex,
            "{input:?} should hit the deterministic explicit path"
        );
        match result.intent {
            Intent::StoreFact { object, .. } => {
                assert_eq!(object, want_object, "{input:?} kept the wrapper")
            }
            other => panic!("{input:?}: expected StoreFact, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn polite_recall_question_is_not_store() {
    let classifier = IntentClassifier::new();
    // Mood-changing wrappers ("can you …") are NOT stripped, so an
    // interrogative stays a question and never becomes a store.
    let result = classifier.classify("can you remember my birthday?").await;
    assert!(
        !matches!(result.intent, Intent::StoreFact { .. }),
        "'can you remember…?' should not be StoreFact, got {:?}",
        result.intent
    );
}

#[test]
fn strip_request_prefix_unwraps_stacked_politeness() {
    use crate::classifier::strip_request_prefix;
    assert_eq!(strip_request_prefix("please remember X"), "remember X");
    assert_eq!(
        strip_request_prefix("Please kindly note Y"),
        "note Y" // case-insensitive + stacked
    );
    // No wrapper — untouched.
    assert_eq!(strip_request_prefix("remember Z"), "remember Z");
    // "can you" is intentionally not a wrapper.
    assert_eq!(
        strip_request_prefix("can you remember W"),
        "can you remember W"
    );
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

// ── Property tests ────────────────────────────────────────────────
//
// `normalize_command` only ever *strips* — shell wrappers and leading filler —
// to leave the real binary at the head. These pin that contract over arbitrary
// input: it never panics, never invents tokens or characters, returns a trimmed
// string, and the head it leaves is never itself a filler token.
mod normalize_command_props {
    use crate::classifier::{is_command_filler, normalize_command};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// The result is always trimmed — no leading or trailing whitespace
        /// survives the peel-and-join.
        #[test]
        fn output_is_trimmed(raw in ".*") {
            let out = normalize_command(&raw);
            prop_assert_eq!(&out, out.trim());
        }

        /// Normalization only removes: it never invents a token or a byte.
        /// The output has no more whitespace tokens than the input and is no
        /// longer than the trimmed input.
        #[test]
        fn never_invents_tokens_or_length(raw in ".*") {
            let out = normalize_command(&raw);
            prop_assert!(out.split_whitespace().count() <= raw.split_whitespace().count());
            prop_assert!(out.len() <= raw.trim().len());
        }

        /// The core guarantee: whatever is left at the head is the real binary,
        /// never a leading-filler token — an empty result is the only exception.
        #[test]
        fn head_is_never_filler(raw in ".*") {
            let out = normalize_command(&raw);
            if let Some(head) = out.split_whitespace().next() {
                prop_assert!(!is_command_filler(head));
            }
        }
    }
}

// ── Fuzz target (F3) ──────────────────────────────────────────────
//
// `classify_regex` runs ~30 regexes plus their capture extractors over raw,
// untrusted natural-language input — the regex path of intent classification.
// (`normalize_command` is already covered by the property tests above; the
// `ExecuteCommand` extractor calls it internally, so this fuzz subsumes that
// path while also exercising every other pattern + extractor.) The regex
// crate guarantees linear-time matching (no ReDoS), so the open question is
// whether any extractor panics on an adversarial capture — this proves it
// doesn't, and pins two output invariants. The classifier is built once and
// shared across iterations (the regex set compiles only at construction).
mod classify_regex_fuzz {
    use super::*;

    #[test]
    fn fuzz_classify_regex_invariants() {
        bolero::check!()
            .with_type::<String>()
            .for_each(|input: &String| {
                // Built per-iteration: bolero runs the body under
                // `catch_unwind`, and `IntentClassifier` holds a `dyn
                // IntentFallback` that isn't `RefUnwindSafe`, so it can't be
                // captured across the boundary. Construction is cheap — it only
                // clones `Arc`s off the static `PATTERNS` regex set.
                let classifier = IntentClassifier::new();
                let Some(c) = classifier.classify_regex(input) else {
                    return;
                };
                // (1) The regex path only ever produces Regex-method results.
                assert!(
                    matches!(c.method, ClassificationMethod::Regex),
                    "non-regex method from classify_regex: {input:?} -> {c:?}"
                );
                // (2) An ExecuteCommand binary is the head token after
                // normalization — always a single whitespace-free token (or
                // empty when the capture normalized away).
                if let Intent::ExecuteCommand { command, .. } = &c.intent {
                    assert!(
                        !command.contains(char::is_whitespace),
                        "ExecuteCommand binary contains whitespace: {input:?} -> {command:?}"
                    );
                }
            });
    }
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
        .classify_with_history("username : alice", &history)
        .await;
    let seen = recording.seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "fallback should be invoked exactly once");
    let (input, hist) = &seen[0];
    assert_eq!(input, "username : alice");
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
async fn test_remind_me_what_is_recall_not_schedule() {
    let classifier = IntentClassifier::new();
    // The reported misroute: a recall question opening a scheduling gate
    // because "remind" greedily matched SCHEDULE_RE.
    for input in [
        "Remind me what I said was the risky part",
        "remind me where I put the staging key",
        "remind me who owns that service",
        "Remind me when's that review",
    ] {
        let result = classifier.classify(input).await;
        assert!(
            matches!(result.intent, Intent::Chat { .. }),
            "Expected Chat (recall) for {input:?}, got {:?}",
            result.intent
        );
        // It must be the deterministic regex guard that caught it, not the
        // bare fallback — proves SCHEDULE_RE didn't claim it first.
        assert_eq!(
            result.method,
            ClassificationMethod::Regex,
            "Expected the regex guard to route {input:?}, got {:?}",
            result.method
        );
    }
}

#[tokio::test]
async fn test_remind_me_to_still_schedules() {
    let classifier = IntentClassifier::new();
    // The guard must not swallow genuine reminders.
    for input in [
        "Remind me to call mom",
        "remind me in 5 minutes to check the build",
        "remind me to deploy at 5pm",
    ] {
        let result = classifier.classify(input).await;
        assert!(
            matches!(result.intent, Intent::Schedule { .. }),
            "Expected Schedule for {input:?}, got {:?}",
            result.intent
        );
    }
}

#[tokio::test]
async fn test_setup_phrasing_schedules() {
    // Regression: "set up a daily reminder …" matched no scheduling regex,
    // fell through to the LLM/Chat fallback, and the conversational SOUL
    // denied it could schedule at all. These must route to Schedule
    // deterministically (regex), so the capability is reached regardless
    // of what the chat model believes about itself.
    let classifier = IntentClassifier::new();
    for input in [
        "Set up a daily reminder at 9am to review my open pull requests",
        "set a daily reminder to water the plants",
        "create a reminder to call the dentist tomorrow",
        "add a recurring task to back up the database every night",
        "set up a weekly schedule to clean the logs",
    ] {
        let result = classifier.classify(input).await;
        assert!(
            matches!(result.intent, Intent::Schedule { .. }),
            "Expected Schedule for {input:?}, got {:?}",
            result.intent
        );
        assert_eq!(
            result.method,
            ClassificationMethod::Regex,
            "Expected the deterministic regex to route {input:?}, got {:?}",
            result.method
        );
    }
}

#[tokio::test]
async fn test_setup_phrasing_without_schedule_noun_is_not_schedule() {
    // The setup-verb branch must not swallow ordinary "set up / create /
    // add a <thing>" requests that have nothing to do with scheduling.
    let classifier = IntentClassifier::new();
    for input in [
        "set up the dev environment",
        "create a new feature branch",
        "add a unit test for the parser",
    ] {
        let result = classifier.classify_regex(input);
        assert!(
            !matches!(
                result,
                Some(Classification {
                    intent: Intent::Schedule { .. },
                    ..
                })
            ),
            "{input:?} must not route to Schedule, got {result:?}",
        );
    }
}

#[tokio::test]
async fn test_reachability_routes_to_chat_not_web_search() {
    // Regression: "is github.com reachable" matched no web route, fell to
    // the fast-tier classifier, and was tagged web_search → net.http (an
    // external fetch behind a consent gate) instead of the read-only
    // net.check probe. These must route to Chat (the tool-loop then runs
    // net.check) deterministically via the regex, not the LLM.
    let classifier = IntentClassifier::new();
    for input in [
        "Can you check whether github.com is reachable right now?",
        "is api.example.com:443 reachable",
        "can you reach api.github.com",
        "ping github.com",
        "is github.com up",
        "is the server down",
    ] {
        let result = classifier.classify(input).await;
        assert!(
            matches!(result.intent, Intent::Chat { .. }),
            "Expected Chat for {input:?}, got {:?}",
            result.intent
        );
        assert_eq!(
            result.method,
            ClassificationMethod::Regex,
            "Expected the deterministic regex to route {input:?}, got {:?}",
            result.method
        );
    }
}

#[tokio::test]
async fn test_explicit_web_search_still_wins_over_reachability_guard() {
    // The reachability guard runs only after the pattern loop, so an
    // explicit "search …" / "fetch <url>" request still routes to
    // WebSearch even if it happens to contain a reachability word.
    let classifier = IntentClassifier::new();
    let result = classifier.classify_regex("search for reachable goals frameworks");
    assert!(
        matches!(
            result,
            Some(Classification {
                intent: Intent::WebSearch { .. },
                ..
            })
        ),
        "explicit search must win over the reachability guard, got {result:?}",
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
fn store_fact_or_chat_stores_a_distilled_triple() {
    let intent = store_fact_or_chat(
        Some("user".into()),
        Some("uses".into()),
        Some("Postgres".into()),
        "remember I use Postgres",
    );
    match intent {
        Intent::StoreFact {
            subject,
            predicate,
            object,
        } => {
            assert_eq!(subject, "user");
            assert_eq!(predicate, "uses");
            assert_eq!(object, "Postgres");
        }
        other => panic!("expected StoreFact, got {other:?}"),
    }
}

#[test]
fn store_fact_or_chat_rejects_raw_request_echo() {
    // The WS4 failure: a compound imperative routed to store_fact with no
    // distilled object. The old code filed the whole sentence as
    // "user said <request>"; now it must fall back to chat.
    let input = "Can you access this terminal and type a message and share it to our memory";
    // (a) no object at all → chat
    let no_object = store_fact_or_chat(Some("user".into()), Some("said".into()), None, input);
    assert!(
        matches!(no_object, Intent::Chat { .. }),
        "missing object should be chat, got {no_object:?}"
    );
    // (b) object that just echoes the raw request → chat
    let echo = store_fact_or_chat(
        Some("user".into()),
        Some("said".into()),
        Some(input.to_string()),
        input,
    );
    assert!(
        matches!(echo, Intent::Chat { .. }),
        "raw-echo object should be chat, got {echo:?}"
    );
}

#[test]
fn store_fact_or_chat_requires_predicate_and_object() {
    let input = "do the thing";
    assert!(matches!(
        store_fact_or_chat(
            Some("user".into()),
            Some("".into()),
            Some("x".into()),
            input
        ),
        Intent::Chat { .. }
    ));
    assert!(matches!(
        store_fact_or_chat(
            Some("user".into()),
            Some("likes".into()),
            Some("  ".into()),
            input
        ),
        Intent::Chat { .. }
    ));
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

/// Reachability/connectivity-diagnostic phrasing must be steered away from
/// web_search (→ net.http) and toward chat, where the net.check/trace/cert
/// capabilities live. Guards the disambiguation rule that fixes the
/// "is github.com reachable" → net.http misroute.
#[test]
fn test_classifier_prompt_separates_reachability_from_web_search() {
    let prompt: &str = &super::CLASSIFIER_SYSTEM_PROMPT;
    assert!(prompt.contains("reachable"));
    assert!(prompt.contains("net.check"));
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
        Intent::Cancel {
            target: CancelTarget::Signal,
            id,
        } => {
            assert_eq!(id, "550e8400-e29b-41d4-a716-446655440000");
        }
        other => panic!("Expected Cancel(Signal), got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_signal_does_not_collide_with_cancel_task() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("cancel task 42").await;
    match result.intent {
        Intent::Cancel {
            target: CancelTarget::Task,
            ..
        } => {} // expected
        other => panic!("'cancel task 42' should classify as Cancel(Task), got {other:?}"),
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
        matches!(
            result.intent,
            Intent::List {
                resource: Resource::StandingApprovals,
                ..
            }
        ),
        "expected List(StandingApprovals), got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn approve_with_ttl_qualifier_rides_in_decision() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("approve abc-123 for 1h").await;
    match result.intent {
        Intent::RespondToApproval { nonce, decision } => {
            assert_eq!(nonce, "abc-123");
            assert_eq!(decision, "approve for 1h");
        }
        other => panic!("expected RespondToApproval, got {other:?}"),
    }
}

#[tokio::test]
async fn approve_here_without_nonce_is_a_qualifier_not_a_nonce() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("approve here for 2d").await;
    match result.intent {
        Intent::RespondToApproval { nonce, decision } => {
            assert_eq!(nonce, "", "`here` must not be mistaken for a nonce");
            assert_eq!(decision, "approve here for 2d");
        }
        other => panic!("expected RespondToApproval, got {other:?}"),
    }
}

#[tokio::test]
async fn approve_with_nonce_and_here_qualifier() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("approve abc-123 here").await;
    match result.intent {
        Intent::RespondToApproval { nonce, decision } => {
            assert_eq!(nonce, "abc-123");
            assert_eq!(decision, "approve here");
        }
        other => panic!("expected RespondToApproval, got {other:?}"),
    }
}

#[tokio::test]
async fn approve_of_free_form_chat_does_not_match_approval() {
    let classifier = IntentClassifier::new();
    let result = classifier
        .classify("approve of this plan because it works")
        .await;
    assert!(
        !matches!(result.intent, Intent::RespondToApproval { .. }),
        "free-form text must not classify as an approval response, got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn memory_approve_slash_carries_agent() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/memory-approve agent-x").await;
    match result.intent {
        Intent::ApproveMemoryWriter { agent } => assert_eq!(agent, "agent-x"),
        other => panic!("expected ApproveMemoryWriter, got {other:?}"),
    }
}

#[tokio::test]
async fn grants_slash_classifies_to_list_grants() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/grants").await;
    assert!(
        matches!(
            result.intent,
            Intent::List {
                resource: Resource::Grants,
                ..
            }
        ),
        "expected List(Grants), got {:?}",
        result.intent
    );
}

#[tokio::test]
async fn approval_revoke_slash_carries_id() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/approval-revoke abc-123").await;
    match result.intent {
        Intent::Cancel {
            target: CancelTarget::StandingApproval,
            id,
        } => assert_eq!(id, "abc-123"),
        other => panic!("expected Cancel(StandingApproval), got {other:?}"),
    }
}

#[tokio::test]
async fn mcp_reconsent_slash_carries_server_name() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/mcp-reconsent github").await;
    match result.intent {
        Intent::ReconsentMcpServer { name } => assert_eq!(name, "github"),
        other => panic!("expected ReconsentMcpServer, got {other:?}"),
    }
}

#[tokio::test]
async fn mcp_reconsent_slash_without_name_falls_through() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/mcp-reconsent").await;
    assert!(
        !matches!(result.intent, Intent::ReconsentMcpServer { .. }),
        "bare /mcp-reconsent must not classify (server name is required)"
    );
}

#[tokio::test]
async fn approval_revoke_slash_without_id_falls_through() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/approval-revoke").await;
    assert!(
        !matches!(
            result.intent,
            Intent::Cancel {
                target: CancelTarget::StandingApproval,
                ..
            }
        ),
        "bare /approval-revoke must not classify (id is required)"
    );
}

#[tokio::test]
async fn task_list_slash_classifies_to_list_tasks() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/task-list").await;
    assert!(
        matches!(
            result.intent,
            Intent::List {
                resource: Resource::Tasks,
                ..
            }
        ),
        "expected List(Tasks), got {:?}",
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
        Intent::Cancel {
            target: CancelTarget::Task,
            id,
        } => assert_eq!(id, "abc-123"),
        other => panic!("expected Cancel(Task), got {other:?}"),
    }
}

#[tokio::test]
async fn task_cancel_slash_without_id_falls_through() {
    let classifier = IntentClassifier::new();
    let result = classifier.classify("/task-cancel").await;
    assert!(
        !matches!(
            result.intent,
            Intent::Cancel {
                target: CancelTarget::Task,
                ..
            }
        ),
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
        // One List sample per Resource so the per-key category derivation is
        // exercised for every listable collection.
        Intent::List {
            resource: Resource::Approvals,
            filter: None,
        },
        Intent::List {
            resource: Resource::StandingApprovals,
            filter: None,
        },
        Intent::List {
            resource: Resource::Schedules,
            filter: None,
        },
        Intent::List {
            resource: Resource::Tasks,
            filter: None,
        },
        Intent::List {
            resource: Resource::Channels,
            filter: None,
        },
        Intent::List {
            resource: Resource::TerminalSessions,
            filter: None,
        },
        Intent::List {
            resource: Resource::McpServers,
            filter: None,
        },
        Intent::List {
            resource: Resource::Capabilities,
            filter: None,
        },
        Intent::List {
            resource: Resource::Grants,
            filter: None,
        },
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
        Intent::ChannelPreferences {
            namespace: None,
            category: None,
        },
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
        // The Lifecycle-category Cancel targets.
        Intent::Cancel {
            target: CancelTarget::Schedule,
            id: "1".into(),
        },
        Intent::DecomposeTask {
            request: "build".into(),
        },
        Intent::Cancel {
            target: CancelTarget::Task,
            id: "1".into(),
        },
        Intent::Cancel {
            target: CancelTarget::Signal,
            id: "1".into(),
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
        Intent::ReconsentMcpServer { name: "m".into() },
        // Governance
        Intent::RespondToApproval {
            nonce: "n".into(),
            decision: "approve".into(),
        },
        Intent::ApproveMemoryWriter { agent: "a".into() },
        // The Governance-category Cancel target.
        Intent::Cancel {
            target: CancelTarget::StandingApproval,
            id: "g".into(),
        },
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
/// `Intent` variant — and, for the generic `List` / `Cancel` verbs, one per
/// [`Resource`] / [`CancelTarget`] value so the per-key category derivation is
/// fully exercised. If the enum (or a Resource/CancelTarget) grows without the
/// helper being updated, the per-variant category coverage assertions silently
/// stop checking the new case. The exact count is intentional — bump it
/// deliberately and add a sample in the same change.
#[test]
fn every_intent_variant_helper_is_exhaustive() {
    assert_eq!(
        every_intent_variant().len(),
        42,
        "every_intent_variant() must list one example per Intent variant \
         (and per Resource / CancelTarget value); update it (and bump this \
         count) when the enum changes"
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
            | Intent::List { .. }
            | Intent::TaskStatus { .. }
            | Intent::QueryAgents { .. }
            | Intent::QueryAudit { .. }
            | Intent::ChannelPreferences { .. } => IntentCategory::Inspection,
            Intent::StoreFact { .. } | Intent::Forget { .. } => IntentCategory::Memory,
            Intent::ExecuteCommand { .. }
            | Intent::WebSearch { .. }
            | Intent::SendMessage { .. }
            | Intent::DelegateTask { .. } => IntentCategory::Action,
            // A standing-approval revoke is Governance; the other Cancel targets
            // (schedule/task/signal) are Lifecycle — the split the per-key
            // derivation is meant to capture.
            Intent::Cancel {
                target: CancelTarget::StandingApproval,
                ..
            } => IntentCategory::Governance,
            Intent::Schedule { .. }
            | Intent::Cancel { .. }
            | Intent::DecomposeTask { .. }
            | Intent::OpenTerminalSession { .. }
            | Intent::CloseTerminalSession { .. }
            | Intent::MountMcpServer { .. }
            | Intent::UnmountMcpServer { .. }
            | Intent::ReconsentMcpServer { .. } => IntentCategory::Lifecycle,
            Intent::RespondToApproval { .. }
            | Intent::ApproveMemoryWriter { .. }
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

// ── Taxonomy drift guards (F1/F2) ──────────────────────────────────────────
//
// `taxonomy::INTENT_SPECS` is the single source of truth for the control-plane
// vocabulary. These four tests make any drift between the `Intent` enum, the
// `PATTERNS` regex table, and the `CLASSIFIER_SYSTEM_PROMPT` a test failure
// instead of a silent natural-language regression.

mod taxonomy_drift_guards {
    use super::*;
    use crate::taxonomy::{spec_for_key, NlRouting, INTENT_SPECS};

    /// One representative value of every `Intent` variant. Hand-maintained,
    /// paired with the compiler-exhaustive [`Intent::key`]: adding a variant
    /// forces a new `key()` arm (compile error), and the completeness test
    /// below fails until this list and `INTENT_SPECS` gain the matching row.
    fn sample_intents() -> Vec<Intent> {
        use intent::{IntentToken, Object, Provenance, Verb};
        let s = String::new;
        vec![
            // Inspection
            Intent::Recall { query: s() },
            Intent::MemorySummary,
            Intent::SystemStatus,
            Intent::ProactivityStatus,
            Intent::BudgetStatus { window: None },
            // One List per Resource — each is a distinct wire key.
            Intent::List {
                resource: Resource::Approvals,
                filter: None,
            },
            Intent::List {
                resource: Resource::StandingApprovals,
                filter: None,
            },
            Intent::List {
                resource: Resource::Schedules,
                filter: None,
            },
            Intent::List {
                resource: Resource::Tasks,
                filter: None,
            },
            Intent::List {
                resource: Resource::Channels,
                filter: None,
            },
            Intent::List {
                resource: Resource::TerminalSessions,
                filter: None,
            },
            Intent::List {
                resource: Resource::McpServers,
                filter: None,
            },
            Intent::List {
                resource: Resource::Capabilities,
                filter: None,
            },
            Intent::List {
                resource: Resource::Grants,
                filter: None,
            },
            Intent::TaskStatus { task_id: s() },
            Intent::QueryAgents { filter: s() },
            Intent::QueryAudit {
                filter: None,
                since: None,
                limit: None,
            },
            Intent::ChannelPreferences {
                namespace: None,
                category: None,
            },
            // Memory
            Intent::StoreFact {
                subject: s(),
                predicate: s(),
                object: s(),
            },
            Intent::Forget { target: s() },
            // Action
            Intent::ExecuteCommand {
                command: s(),
                args: Vec::new(),
            },
            Intent::WebSearch { query: s() },
            Intent::SendMessage {
                channel: s(),
                recipient: s(),
                content: s(),
            },
            Intent::DelegateTask {
                agent: s(),
                prompt: s(),
            },
            // Lifecycle
            Intent::Schedule {
                description: s(),
                cron: None,
            },
            Intent::Cancel {
                target: CancelTarget::Schedule,
                id: s(),
            },
            Intent::DecomposeTask { request: s() },
            Intent::Cancel {
                target: CancelTarget::Task,
                id: s(),
            },
            Intent::Cancel {
                target: CancelTarget::Signal,
                id: s(),
            },
            Intent::OpenTerminalSession {
                program: s(),
                args: Vec::new(),
                cwd: None,
            },
            Intent::CloseTerminalSession { session_id: s() },
            Intent::MountMcpServer {
                name: s(),
                transport: s(),
                command_or_url: s(),
            },
            Intent::UnmountMcpServer { name: s() },
            Intent::ReconsentMcpServer { name: s() },
            // Governance
            Intent::RespondToApproval {
                nonce: s(),
                decision: s(),
            },
            Intent::ApproveMemoryWriter { agent: s() },
            Intent::Cancel {
                target: CancelTarget::StandingApproval,
                id: s(),
            },
            Intent::PruneAudit { older_than: s() },
            Intent::SetChannelPreference {
                channel: s(),
                category: s(),
                weight: 0.0,
                pinned: false,
            },
            Intent::SetProactivity {
                enabled: true,
                until: None,
            },
            // Capability
            Intent::ToolCall(Box::new(IntentToken::new(
                Verb::new("ns", "action"),
                Object {
                    kind: "intent_args".into(),
                    value: serde_json::Value::Null,
                },
                Provenance::User {
                    raw_input: s(),
                    ui_origin: None,
                    ts: chrono::Utc::now(),
                },
                "personal".into(),
            ))),
            // Conversation
            Intent::Chat { content: s() },
        ]
    }

    /// Guard 1 — completeness: every `Intent` variant has exactly one
    /// `IntentSpec`, and the table has no orphan rows. A new variant without a
    /// spec (or a spec without a variant) trips this.
    #[test]
    fn every_variant_has_exactly_one_spec() {
        let samples = sample_intents();

        // No duplicate keys in the table, and the table is a bijection with
        // the sample set (which covers every variant — see `sample_intents`).
        let spec_keys: std::collections::BTreeSet<&str> =
            INTENT_SPECS.iter().map(|s| s.key).collect();
        assert_eq!(
            spec_keys.len(),
            INTENT_SPECS.len(),
            "duplicate key in INTENT_SPECS"
        );

        let sample_keys: std::collections::BTreeSet<&str> =
            samples.iter().map(|i| i.key()).collect();
        assert_eq!(
            sample_keys.len(),
            samples.len(),
            "two sample intents share a key() — sample list is malformed"
        );

        assert_eq!(
            sample_keys, spec_keys,
            "Intent variants and INTENT_SPECS disagree. Add the missing \
             IntentSpec row (and a sample_intents() entry) for any new variant, \
             or remove the orphan spec row."
        );
    }

    /// Guard 2 — category agreement: `Intent::category()` must match the
    /// category declared in the spec table for every variant.
    #[test]
    fn category_matches_spec_for_every_variant() {
        for intent in sample_intents() {
            let spec = spec_for_key(intent.key())
                .unwrap_or_else(|| panic!("no spec for key {:?}", intent.key()));
            assert_eq!(
                intent.category(),
                spec.category,
                "category mismatch for {:?}: enum says {:?}, table says {:?}",
                intent.key(),
                intent.category(),
                spec.category,
            );
        }
    }

    /// Guard 3 — regex coverage: every `PATTERNS` entry maps to a known,
    /// non-slash-only spec, and every `RegexOnly` verb actually has at least
    /// one regex (so it can't silently lose its only NL surface).
    #[test]
    fn regex_coverage_agrees_with_table() {
        use std::collections::BTreeSet;

        let regex_keys: BTreeSet<&str> = PATTERNS.iter().map(|p| p.base_intent.key()).collect();

        for key in &regex_keys {
            let spec = spec_for_key(key)
                .unwrap_or_else(|| panic!("PATTERNS entry {key:?} has no IntentSpec row"));
            assert_ne!(
                spec.nl_routable,
                NlRouting::SlashOnly,
                "{key:?} is marked SlashOnly but has a regex in PATTERNS — \
                 update its NlRouting to RegexOnly or LlmFallback",
            );
        }

        for spec in INTENT_SPECS {
            if spec.nl_routable == NlRouting::RegexOnly {
                assert!(
                    regex_keys.contains(spec.key),
                    "{:?} is RegexOnly but has no PATTERNS entry — its only \
                     natural-language surface is gone",
                    spec.key,
                );
            }
        }
    }

    /// Guard 4 — prompt coverage: the classifier prompt's "Valid intents:"
    /// line lists exactly the `LlmFallback` keys, no more and no less. This
    /// pins the documented "26 of 39 NL-routable" gap as a conscious per-verb
    /// choice — adding an `LlmFallback` verb without listing it in the prompt
    /// (or vice versa) fails here.
    #[test]
    fn prompt_lists_exactly_the_llm_fallback_keys() {
        use std::collections::BTreeSet;

        let prompt: &str = &super::CLASSIFIER_SYSTEM_PROMPT;
        let line = prompt
            .lines()
            .find(|l| l.starts_with("Valid intents:"))
            .expect("prompt must have a 'Valid intents:' line");
        let listed: BTreeSet<&str> = line
            .trim_start_matches("Valid intents:")
            .trim()
            .trim_end_matches('.')
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let expected: BTreeSet<&str> = INTENT_SPECS
            .iter()
            .filter(|s| s.nl_routable == NlRouting::LlmFallback)
            .map(|s| s.key)
            .collect();

        assert_eq!(
            listed, expected,
            "classifier prompt's 'Valid intents:' line has drifted from the \
             LlmFallback set in INTENT_SPECS. Update the prompt or the table's \
             nl_routable so they agree.",
        );
    }

    /// The generated prompt must stay well-formed: the fixed header and the
    /// hand-written rules/JSON contract survive assembly, and there is exactly
    /// one generated `Valid intents:` line between them. Guards against
    /// `build_classifier_system_prompt` dropping or duplicating a section.
    #[test]
    fn generated_prompt_is_well_formed() {
        let prompt: &str = &super::CLASSIFIER_SYSTEM_PROMPT;

        assert!(
            prompt.starts_with(super::CLASSIFIER_PROMPT_HEADER),
            "prompt must begin with the fixed header",
        );
        assert!(
            prompt.contains(super::CLASSIFIER_PROMPT_RULES),
            "prompt must embed the hand-written rules/JSON-contract block",
        );
        assert_eq!(
            prompt
                .lines()
                .filter(|l| l.starts_with("Valid intents:"))
                .count(),
            1,
            "prompt must have exactly one generated 'Valid intents:' line",
        );
    }
}
