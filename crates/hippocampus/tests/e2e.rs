#[tokio::test]
async fn test_brain_e2e_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- BRAIN OS E2E VERIFICATION ---");

    let temp_dir = tempfile::TempDir::new()?;
    let db_path = temp_dir.path().join("test.db");
    let ruv_path = temp_dir.path().join("ruv");

    let pool = storage::sqlite::SqlitePool::open(&db_path)?;
    // pool.ensure_tables()?; // SqlitePool::open already runs migrations
    let ruv = storage::ruvector::RuVectorStore::open(&ruv_path, 384).await?;
    ruv.ensure_tables().await?;

    let store = hippocampus::semantic::SemanticStore::new(pool.clone(), ruv);

    // 1. Hierarchical Namespace Verification
    println!("\n[1] Testing Hierarchical Namespaces...");
    
    fn vec384(val: f32) -> Vec<f32> {
        let mut v = vec![0.0; 384];
        let idx = (val * 100.0) as usize % 384;
        v[idx] = 1.0;
        v
    }

    store.store_fact("work", "personal", "user", "has", "job", 1.0, None, vec384(0.1), None).await?;
    store.store_fact("work/project-a", "personal", "user", "working_on", "brain", 1.0, None, vec384(0.2), None).await?;
    store.store_fact("personal", "personal", "user", "lives_in", "earth", 1.0, None, vec384(0.3), None).await?;

    // Query 'work' should include 'work/project-a'
    let work_facts = store.list_by_namespace(Some("work"))?;
    println!("Facts in 'work': {}", work_facts.len());
    for f in &work_facts {
        println!(" - {}: {} {} {}", f.namespace, f.subject, f.predicate, f.object);
    }
    assert!(work_facts.len() >= 2, "Hierarchical query for 'work' failed");

    // Query 'work/project-a' should ONLY include itself
    let project_facts = store.list_by_namespace(Some("work/project-a"))?;
    println!("Facts in 'work/project-a': {}", project_facts.len());
    assert_eq!(project_facts.len(), 1, "Strict query for 'work/project-a' failed");

    // 2. Fact Deduplication Verification
    println!("\n[2] Testing Fact Deduplication...");
    let v_dup = vec384(0.5);
    let id1 = store.store_fact("dedup", "test", "sky", "is", "blue", 1.0, None, v_dup.clone(), None).await?;
    
    // Identical fact (subject, predicate, object, vector) -> returns same ID
    let id2 = store.store_fact("dedup", "test", "sky", "is", "blue", 1.0, None, v_dup.clone(), None).await?;
    assert_eq!(id1, id2, "Identical fact should return same ID");
    println!(" - Identical fact deduplicated successfully");

    // Similar vector, different object -> updates/supersedes
    let mut v_sim = v_dup.clone();
    v_sim[383] = 0.01; // very slight change
    let id3 = store.store_fact("dedup", "test", "sky", "is", "azure", 1.0, None, v_sim, None).await?;
    assert_ne!(id1, id3, "Superseded fact should have new ID");
    
    let fact1 = store.get_fact(&id1)?;
    assert!(fact1.is_some(), "Old fact should still exist in DB");
    
    let active_facts = store.list_by_namespace(Some("dedup"))?;
    assert_eq!(active_facts.len(), 1, "Should only have 1 active fact after superseding");
    assert_eq!(active_facts[0].object, "azure");
    println!(" - Fact superseded successfully (azure > blue)");

    // 3. SOUL System Prompt Verification (Mock check)
    // Note: We can't easily access cortex from hippocampus tests without adding it as a dev-dependency
    // but we already verified it in cortex unit tests.
    /*
    println!("\n[3] Testing SOUL System Prompt...");
    let cortex_ctx = cortex::context::ContextAssembler::with_defaults();
    let messages = cortex_ctx.assemble("test", &[], &[]);
    let system_prompt = &messages[0].content;
    assert!(system_prompt.contains("SOUL"), "System prompt missing SOUL");
    assert!(system_prompt.contains("biologically-inspired"), "System prompt missing biologically-inspired");
    println!(" - SOUL system prompt verified");
    */

    println!("\n--- ALL VERIFICATIONS PASSED ---");
    Ok(())
}
