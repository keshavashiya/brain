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

/// Storing under an existing (tool, key) overwrites: subsequent `get`
/// returns the new value, not the original.
#[tokio::test]
async fn store_overwrites_existing_entry() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");

    vault
        .store(
            "github",
            "token",
            CredentialValue::new("old".into()),
            InjectionShape::EnvVar {
                name: "GH".to_string(),
            },
        )
        .await
        .unwrap();
    vault
        .store(
            "github",
            "token",
            CredentialValue::new("new".into()),
            InjectionShape::Header {
                name: "X-Token".to_string(),
            },
        )
        .await
        .unwrap();

    let got = vault.get("github", "token").await.unwrap();
    assert_eq!(got.value.as_str(), "new");
    assert!(matches!(got.shape, InjectionShape::Header { ref name } if name == "X-Token"));
}

#[tokio::test]
async fn get_missing_key_returns_not_found() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");
    match vault.get("ghost", "nope").await {
        Err(VaultError::NotFound { tool, key }) => {
            assert_eq!(tool, "ghost");
            assert_eq!(key, "nope");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_missing_key_returns_not_found() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");
    match vault.delete("ghost", "nope").await {
        Err(VaultError::NotFound { .. }) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
}

/// `FileBackend::init` is idempotent: calling twice with the same
/// passphrase is a no-op. Calling twice with a different passphrase
/// returns `BadPassphrase` and leaves the original verifier in place.
#[tokio::test]
async fn init_is_idempotent_for_matching_passphrase() {
    use brainos_vault::file::{FileBackend, PassphraseSource};

    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let b1 = FileBackend::new(path.clone(), PassphraseSource::Direct("pw".into()));
    b1.init().await.unwrap();
    // Second init with same passphrase must succeed.
    let b2 = FileBackend::new(path.clone(), PassphraseSource::Direct("pw".into()));
    b2.init().await.unwrap();

    // Different passphrase fails fast on verifier.
    let b3 = FileBackend::new(path, PassphraseSource::Direct("other".into()));
    match b3.init().await {
        Err(VaultError::BadPassphrase) => {}
        other => panic!("expected BadPassphrase on re-init, got {other:?}"),
    }
}

/// AES-GCM is authenticated: flipping a byte in the ciphertext is
/// detectable on read. The vault must surface a `Crypto` (or
/// `InvalidData` for length corruption) error rather than silently
/// returning garbage.
#[tokio::test]
async fn tampered_blob_fails_to_decrypt() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");

    vault
        .store(
            "github",
            "token",
            CredentialValue::new("valid".into()),
            InjectionShape::EnvVar { name: "G".into() },
        )
        .await
        .unwrap();

    // Locate and corrupt the .enc blob by flipping the last byte
    // (inside the GCM tag).
    let enc = dir.path().join("github").join("token.enc");
    let mut bytes = std::fs::read(&enc).unwrap();
    let last = bytes.last_mut().expect("non-empty blob");
    *last ^= 0x01;
    std::fs::write(&enc, &bytes).unwrap();

    match vault.get("github", "token").await {
        Err(VaultError::Crypto(_)) => {}
        other => panic!("expected Crypto error after tamper, got {other:?}"),
    }
}

/// Truncating the blob below the 12-byte nonce length surfaces as
/// `InvalidData`, not a panic.
#[tokio::test]
async fn truncated_blob_returns_invalid_data() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");
    vault
        .store(
            "t",
            "k",
            CredentialValue::new("v".into()),
            InjectionShape::EnvVar { name: "X".into() },
        )
        .await
        .unwrap();
    let enc = dir.path().join("t").join("k.enc");
    std::fs::write(&enc, [0u8; 4]).unwrap();
    match vault.get("t", "k").await {
        Err(VaultError::InvalidData(msg)) => assert!(msg.contains("too short")),
        other => panic!("expected InvalidData for short blob, got {other:?}"),
    }
}

/// Tool/key names with FS-unsafe characters route through `sanitize`
/// (anything outside `[A-Za-z0-9._-]` becomes `_`). The entry must
/// still be retrievable using the original unsanitized name, and the
/// `list` filter must match both the raw and sanitized form.
#[tokio::test]
async fn names_with_unsafe_chars_are_sanitized_round_trip() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");

    let tool = "git/hub";
    let key = "tok en";
    vault
        .store(
            tool,
            key,
            CredentialValue::new("v".into()),
            InjectionShape::EnvVar { name: "X".into() },
        )
        .await
        .unwrap();

    let got = vault.get(tool, key).await.unwrap();
    assert_eq!(got.value.as_str(), "v");

    // On-disk dirname is sanitized.
    assert!(dir.path().join("git_hub").exists());

    // list(Some(unsanitized)) must still match (the impl tries both).
    let listed = vault.list(Some(tool)).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].tool, "git_hub");
}

/// After `get`, the entry's `last_used_at` is populated.
#[tokio::test]
async fn last_used_at_populates_after_first_get() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");
    vault
        .store(
            "t",
            "k",
            CredentialValue::new("v".into()),
            InjectionShape::EnvVar { name: "X".into() },
        )
        .await
        .unwrap();

    let before = vault.list(Some("t")).await.unwrap();
    assert_eq!(before.len(), 1);
    assert!(before[0].last_used_at.is_none());

    vault.get("t", "k").await.unwrap();

    let after = vault.list(Some("t")).await.unwrap();
    assert_eq!(after.len(), 1);
    assert!(after[0].last_used_at.is_some(), "{:?}", after[0]);
}

#[tokio::test]
async fn list_on_empty_uninitialised_dir_returns_empty() {
    // No init(), no store — list against an empty dir is `Ok([])`,
    // not an error. (The init-check only gates store/get/delete.)
    let dir = TempDir::new().unwrap();
    let vault = make_vault(&dir, "pw");
    let listed = vault.list(None).await.unwrap();
    assert!(listed.is_empty());
}

/// Non-ASCII / multi-byte values survive the encrypt/decrypt round
/// trip and the `From<u8>`-style String→bytes path doesn't corrupt them.
#[tokio::test]
async fn non_ascii_value_round_trips() {
    let dir = TempDir::new().unwrap();
    init(&dir, "pw").await;
    let vault = make_vault(&dir, "pw");
    let secret = "café-🔑-Ω";
    vault
        .store(
            "t",
            "k",
            CredentialValue::new(secret.to_string()),
            InjectionShape::EnvVar { name: "X".into() },
        )
        .await
        .unwrap();
    let got = vault.get("t", "k").await.unwrap();
    assert_eq!(got.value.as_str(), secret);
}
