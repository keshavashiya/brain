use super::*;

async fn test_store() -> (SemanticStore, tempfile::TempDir) {
    let db = SqlitePool::open_memory().unwrap();
    let ruv_dir = tempfile::tempdir().unwrap();
    let ruv = RuVectorStore::open(ruv_dir.path(), 384).await.unwrap();
    ruv.ensure_tables().await.unwrap();
    (SemanticStore::new(db, ruv), ruv_dir)
}

fn dummy_vector(val: f32) -> Vec<f32> {
    let mut v = vec![0.0; 384];
    let idx = (val * 100.0) as usize % 384;
    v[idx] = 1.0;
    v
}

#[tokio::test]
async fn test_store_and_get_fact() {
    let (store, _dir) = test_store().await;

    let id = store
        .store_fact(
            "personal",
            "personal",
            "user",
            "name_is",
            "Keshav",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();

    let fact = store.get_fact(&id).unwrap().unwrap();
    assert_eq!(fact.namespace, "personal");
    assert_eq!(fact.subject, "user");
    assert_eq!(fact.predicate, "name_is");
    assert_eq!(fact.object, "Keshav");
    assert_eq!(fact.category, "personal");
}

#[tokio::test]
async fn test_get_facts_by_category() {
    let (store, _dir) = test_store().await;

    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "name_is",
            "Keshav",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "likes",
            "Rust",
            0.9,
            None,
            dummy_vector(0.2),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "personal",
            "work",
            "user",
            "role_is",
            "developer",
            0.8,
            None,
            dummy_vector(0.3),
            None,
        )
        .await
        .unwrap();

    let personal = store.get_facts_by_category("personal", None).unwrap();
    assert_eq!(personal.len(), 2);

    let work = store.get_facts_by_category("work", None).unwrap();
    assert_eq!(work.len(), 1);

    let scoped = store
        .get_facts_by_category("personal", Some("personal"))
        .unwrap();
    assert_eq!(scoped.len(), 2);

    let cross = store.get_facts_by_category("work", Some("other")).unwrap();
    assert_eq!(cross.len(), 0);
}

#[tokio::test]
async fn test_get_facts_about() {
    let (store, _dir) = test_store().await;

    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "name_is",
            "Keshav",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "likes",
            "Rust",
            0.9,
            None,
            dummy_vector(0.2),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "personal",
            "personal",
            "Alice",
            "knows",
            "user",
            0.5,
            None,
            dummy_vector(0.3),
            None,
        )
        .await
        .unwrap();

    let about_user = store.get_facts_about("user").unwrap();
    assert_eq!(about_user.len(), 2);
}

#[tokio::test]
async fn test_fact_count() {
    let (store, _dir) = test_store().await;

    assert_eq!(store.count().unwrap(), 0);
    store
        .store_fact(
            "personal",
            "test",
            "a",
            "b",
            "c",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();
    assert_eq!(store.count().unwrap(), 1);
}

#[tokio::test]
async fn test_vector_search() {
    let (store, _dir) = test_store().await;

    let mut v1 = vec![0.0f32; 384];
    v1[0] = 1.0;
    let mut v2 = vec![0.0f32; 384];
    v2[1] = 1.0;

    store
        .store_fact(
            "personal",
            "test",
            "rust",
            "is",
            "fast",
            1.0,
            None,
            v1.clone(),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "personal", "test", "python", "is", "popular", 1.0, None, v2, None,
        )
        .await
        .unwrap();

    let results = store.search_similar(v1, 2, None, None).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].fact.subject, "rust");
}

#[tokio::test]
async fn test_update_fact() {
    let (store, _dir) = test_store().await;

    let old_id = store
        .store_fact(
            "personal",
            "personal",
            "user",
            "location",
            "NYC",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();

    let new_id = store
        .update_fact(&old_id, "SF", dummy_vector(0.2))
        .await
        .unwrap();

    let new_fact = store.get_fact(&new_id).unwrap().unwrap();
    assert_eq!(new_fact.object, "SF");
    assert_eq!(new_fact.namespace, "personal");
    assert_eq!(store.count().unwrap(), 1);
}

#[tokio::test]
async fn test_namespace_isolation() {
    let (store, _dir) = test_store().await;

    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "hobby",
            "coding",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "work",
            "work",
            "user",
            "role",
            "developer",
            1.0,
            None,
            dummy_vector(0.2),
            None,
        )
        .await
        .unwrap();

    let personal = store.list_by_namespace(Some("personal")).unwrap();
    assert_eq!(personal.len(), 1);
    assert_eq!(personal[0].namespace, "personal");

    let work = store.list_by_namespace(Some("work")).unwrap();
    assert_eq!(work.len(), 1);
    assert_eq!(work[0].namespace, "work");

    let all = store.list_all().unwrap();
    assert_eq!(all.len(), 2);
}

#[tokio::test]
async fn test_list_namespaces() {
    let (store, _dir) = test_store().await;

    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "hobby",
            "coding",
            1.0,
            None,
            dummy_vector(0.1),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "name",
            "Keshav",
            1.0,
            None,
            dummy_vector(0.2),
            None,
        )
        .await
        .unwrap();
    store
        .store_fact(
            "work",
            "work",
            "user",
            "role",
            "developer",
            1.0,
            None,
            dummy_vector(0.3),
            None,
        )
        .await
        .unwrap();

    let namespaces = store.list_namespaces().unwrap();
    assert_eq!(namespaces.len(), 2);

    let personal_ns = namespaces
        .iter()
        .find(|n| n.namespace == "personal")
        .unwrap();
    assert_eq!(personal_ns.fact_count, 2);

    let work_ns = namespaces.iter().find(|n| n.namespace == "work").unwrap();
    assert_eq!(work_ns.fact_count, 1);
}

// Forget regression: the classifier hands find_facts_matching the whole user
// sentence, not a distilled topic. Token-keyed matching must still locate the
// stored fact so the Forget intent actually deletes it (privacy). The old
// whole-string `LIKE %<sentence>%` matched nothing and the forget no-op'd.
#[tokio::test]
async fn find_facts_matching_keys_on_topic_not_whole_sentence() {
    let (store, _dir) = test_store().await;
    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "access_token_is",
            "TEMPTOKEN-9981",
            1.0,
            None,
            dummy_vector(0.2),
            None,
        )
        .await
        .unwrap();
    // An unrelated fact that must NOT match the token target.
    store
        .store_fact(
            "personal",
            "personal",
            "user",
            "likes",
            "black coffee",
            1.0,
            None,
            dummy_vector(0.4),
            None,
        )
        .await
        .unwrap();

    let hits = store
        .find_facts_matching(
            "Actually forget what I said about my temporary access token.",
            Some("personal"),
        )
        .unwrap();

    assert_eq!(hits.len(), 1, "should match exactly the access-token fact");
    assert_eq!(hits[0].predicate, "access_token_is");

    // Back-compat: a bare single-word target still matches via substring.
    let coffee = store
        .find_facts_matching("coffee", Some("personal"))
        .unwrap();
    assert_eq!(coffee.len(), 1);
    assert_eq!(coffee[0].object, "black coffee");
}
