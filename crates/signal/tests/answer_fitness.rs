//! DoD for answer-quality fitness (L1): a deliberately-degraded model loses
//! routing share for the task kind it does badly at, while a single-model
//! install never diverges from the static default.
//!
//! This drives the real `cerebellum::AnswerFitnessStore` (SQLite, decay) and
//! the real `signal::answer_fitness::select_tier` — the deterministic
//! stand-in for the full chat loop, exactly as the plan scopes it.

use brainos_signal::answer_fitness::{select_tier, TaskKind};
use cerebellum::{AnswerFitnessStore, AnswerOutcome};
use storage::SqlitePool;

const HALF_LIFE_HOURS: f64 = 30.0 * 24.0;
const MIN_JUDGED: i64 = 8;
const MARGIN: f32 = 0.15;

fn store() -> AnswerFitnessStore {
    let pool = SqlitePool::open_memory().unwrap();
    let s = AnswerFitnessStore::new(pool, true, HALF_LIFE_HOURS);
    s.ensure_tables().unwrap();
    s
}

/// Record `n` outcomes of one kind for `(kind, model)`.
fn record_n(s: &AnswerFitnessStore, kind: TaskKind, model: &str, outcome: AnswerOutcome, n: usize) {
    for _ in 0..n {
        s.record(kind.as_str(), model, outcome).unwrap();
    }
}

fn route(
    s: &AnswerFitnessStore,
    kind: TaskKind,
    deep: &str,
    balanced: &str,
    fast: &str,
) -> cortex::llm::TaskTier {
    let k = kind.as_str();
    let dq = s.quality(k, deep).unwrap();
    let bq = s.quality(k, balanced).unwrap();
    let fq = s.quality(k, fast).unwrap();
    select_tier(
        (deep, dq.as_ref()),
        (balanced, bq.as_ref()),
        (fast, fq.as_ref()),
        MIN_JUDGED,
        MARGIN,
    )
}

#[test]
fn degraded_deep_model_loses_routing_share_for_its_bad_kind() {
    use cortex::llm::TaskTier;
    let s = store();
    let (deep, balanced, fast) = ("openai/deep-m", "ollama/bal-m", "ollama/fast-m");

    // Deep answers `coding` badly; balanced answers it well. Both clear the
    // evidence bar.
    record_n(&s, TaskKind::Coding, deep, AnswerOutcome::Fail, 10);
    record_n(&s, TaskKind::Coding, balanced, AnswerOutcome::Gold, 10);

    // Routing share for `coding` shifts off the degraded deep model.
    assert_eq!(
        route(&s, TaskKind::Coding, deep, balanced, fast),
        TaskTier::Balanced,
        "coding should route to the better-answering balanced tier"
    );

    // A kind with no evidence is unaffected — the bias is per task kind.
    assert_eq!(
        route(&s, TaskKind::FactualQa, deep, balanced, fast),
        TaskTier::Deep,
        "an unjudged kind keeps the static default"
    );
}

#[test]
fn no_shift_below_the_evidence_bar() {
    use cortex::llm::TaskTier;
    let s = store();
    let (deep, balanced, fast) = ("openai/deep-m", "ollama/bal-m", "ollama/fast-m");

    // Deep clearly worse, but only a handful of judged turns each (< MIN_JUDGED).
    record_n(&s, TaskKind::Coding, deep, AnswerOutcome::Fail, 3);
    record_n(&s, TaskKind::Coding, balanced, AnswerOutcome::Gold, 3);

    assert_eq!(
        route(&s, TaskKind::Coding, deep, balanced, fast),
        TaskTier::Deep,
        "too little evidence must not move routing"
    );
}

#[test]
fn single_model_install_never_diverges() {
    use cortex::llm::TaskTier;
    let s = store();
    // With `llm.tiers` unset every tier wraps the same chain → same model key.
    let m = "ollama/only-m";

    // Even a long run of failures cannot downgrade — there is no other model
    // to move to, so routing stays byte-identical to the static default.
    record_n(&s, TaskKind::Coding, m, AnswerOutcome::Fail, 50);

    assert_eq!(
        route(&s, TaskKind::Coding, m, m, m),
        TaskTier::Deep,
        "identical models across tiers can never trigger a downgrade"
    );
}

#[test]
fn disabled_store_is_inert() {
    use cortex::llm::TaskTier;
    let pool = SqlitePool::open_memory().unwrap();
    let s = AnswerFitnessStore::new(pool, false, HALF_LIFE_HOURS);
    s.ensure_tables().unwrap();
    // Records are dropped; quality always None → selector returns the default.
    record_n(
        &s,
        TaskKind::Coding,
        "openai/deep-m",
        AnswerOutcome::Fail,
        20,
    );
    record_n(
        &s,
        TaskKind::Coding,
        "ollama/bal-m",
        AnswerOutcome::Gold,
        20,
    );
    assert_eq!(
        route(
            &s,
            TaskKind::Coding,
            "openai/deep-m",
            "ollama/bal-m",
            "ollama/fast-m"
        ),
        TaskTier::Deep,
    );
}
