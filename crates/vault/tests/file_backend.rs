//! Integration tests for the encrypted-file vault backend.
//!
//! These tests must work on every platform without touching the OS
//! keychain, so they always instantiate `FileBackend` directly and drive
//! the high-level `DefaultVault` wrapper around it.

use std::sync::Arc;

use brainos_vault::file::{FileBackend, PassphraseSource};
use brainos_vault::vault::DefaultVault;
use brainos_vault::{
    BackendKind, CredentialValue, CredentialVault, InjectionShape, VaultBackend, VaultError,
};
use tempfile::TempDir;

fn make_vault(dir: &TempDir, passphrase: &str) -> DefaultVault {
    let backend = FileBackend::new(
        dir.path().to_path_buf(),
        PassphraseSource::Direct(passphrase.to_string()),
    );
    DefaultVault::new(VaultBackend::File(backend))
}

async fn init(dir: &TempDir, passphrase: &str) {
    let backend = FileBackend::new(
        dir.path().to_path_buf(),
        PassphraseSource::Direct(passphrase.to_string()),
    );
    backend.init().await.expect("vault init");
}

#[tokio::test]
async fn round_trip_store_get_delete() {
    let dir = TempDir::new().unwrap();
    init(&dir, "hunter2").await;
    let vault = make_vault(&dir, "hunter2");

    vault
        .store(
            "github",
            "token",
            CredentialValue::new("ghp_SECRET_value_123".to_string()),
            InjectionShape::EnvVar {
                name: "GITHUB_TOKEN".into(),
            },
        )
        .await
        .unwrap();

    let injected = vault.get("github", "token").await.unwrap();
    assert_eq!(injected.value.as_str(), "ghp_SECRET_value_123");
    assert!(matches!(
        injected.shape,
        InjectionShape::EnvVar { ref name } if name == "GITHUB_TOKEN"
    ));

    let listed = vault.list(None).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].tool, "github");
    assert_eq!(listed[0].key, "token");
    assert_eq!(listed[0].backend, "file");

    vault.delete("github", "token").await.unwrap();
    assert!(matches!(
        vault.get("github", "token").await,
        Err(VaultError::NotFound { .. })
    ));
}

#[tokio::test]
async fn wrong_passphrase_rejected() {
    let dir = TempDir::new().unwrap();
    init(&dir, "correct-horse").await;

    // First vault sets a value successfully.
    let good = make_vault(&dir, "correct-horse");
    good.store(
        "t",
        "k",
        CredentialValue::new("v".into()),
        InjectionShape::EnvVar {
            name: "X".to_string(),
        },
    )
    .await
    .unwrap();

    // Second vault with a different passphrase must fail fast on verifier.
    let bad = make_vault(&dir, "wrong-passphrase");
    match bad.get("t", "k").await {
        Err(VaultError::BadPassphrase) => {}
        other => panic!("expected BadPassphrase, got {other:?}"),
    }
}

#[tokio::test]
async fn debug_never_prints_value() {
    let v = CredentialValue::new("SUPER_SECRET".into());
    let formatted = format!("{v:?}");
    assert!(
        !formatted.contains("SUPER_SECRET"),
        "Debug impl leaked value: {formatted}"
    );
    assert!(formatted.contains("redacted"));
}

#[tokio::test]
async fn missing_verifier_reports_uninitialised() {
    let dir = TempDir::new().unwrap();
    let vault = make_vault(&dir, "any");
    match vault.get("t", "k").await {
        Err(VaultError::BackendUnavailable(msg)) => {
            assert!(msg.contains("brain vault init"), "{msg}");
        }
        other => panic!("expected BackendUnavailable, got {other:?}"),
    }
}

#[tokio::test]
async fn list_filters_by_tool() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");

    for (tool, key) in [("github", "token"), ("jira", "api_key"), ("jira", "oauth")] {
        vault
            .store(
                tool,
                key,
                CredentialValue::new(format!("value-{tool}-{key}")),
                InjectionShape::EnvVar {
                    name: format!("{}_{}", tool.to_uppercase(), key.to_uppercase()),
                },
            )
            .await
            .unwrap();
    }

    let all = vault.list(None).await.unwrap();
    assert_eq!(all.len(), 3);

    let jira = vault.list(Some("jira")).await.unwrap();
    assert_eq!(jira.len(), 2);
    assert!(jira.iter().all(|m| m.tool == "jira"));
}

#[tokio::test]
async fn backend_kind_is_file() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");
    assert_eq!(vault.backend_kind(), BackendKind::File);
}

/// Audit coupling: every `get` records an entry and the metadata never
/// contains the raw credential value.
#[tokio::test]
async fn audit_never_logs_credential_value() {
    use audit::{AuditQuerySpec, AuditTrail, SqliteAuditTrail};
    use storage::SqlitePool;

    let storage_dir = TempDir::new().unwrap();
    let db_path = storage_dir.path().join("audit.db");
    let pool = SqlitePool::open(&db_path).unwrap();
    let audit = Arc::new(SqliteAuditTrail::new(pool));
    audit.ensure_tables().unwrap();

    let vault_dir = TempDir::new().unwrap();
    init(&vault_dir, "pw").await;
    let backend = FileBackend::new(
        vault_dir.path().to_path_buf(),
        PassphraseSource::Direct("pw".to_string()),
    );
    let vault = DefaultVault::new(VaultBackend::File(backend)).with_audit(audit.clone() as Arc<_>);

    let secret = "NEVER_LOG_ME_PLEASE";
    vault
        .store(
            "github",
            "token",
            CredentialValue::new(secret.to_string()),
            InjectionShape::EnvVar { name: "GH".into() },
        )
        .await
        .unwrap();
    let _ = vault.get("github", "token").await.unwrap();
    vault.delete("github", "token").await.unwrap();

    let entries = audit
        .query(AuditQuerySpec::default())
        .await
        .expect("audit query");
    assert!(
        entries.len() >= 3,
        "expected >=3 audit entries, got {}",
        entries.len()
    );

    for entry in &entries {
        let encoded = serde_json::to_string(entry).unwrap();
        assert!(
            !encoded.contains(secret),
            "audit entry leaked credential: {encoded}"
        );
    }

    let actions: Vec<_> = entries.iter().map(|e| e.action.as_str()).collect();
    assert!(actions.iter().any(|a| a.contains("vault.store")));
    assert!(actions.iter().any(|a| a.contains("vault.get")));
    assert!(actions.iter().any(|a| a.contains("vault.delete")));
}
