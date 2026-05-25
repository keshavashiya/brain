//! Unit-test coverage for the small types and the `PassphraseSource`/
//! `resolve_backend` / `BackendKind` helpers. Sits alongside
//! `file_backend.rs` (which covers end-to-end store/get/list scenarios)
//! so the small-type tests don't bloat that file.

use std::path::PathBuf;

use brainos_vault::file::PassphraseSource;
use brainos_vault::vault::{resolve_backend, BackendSelection, VaultConfig};
use brainos_vault::{
    BackendKind, CredentialMetadata, CredentialValue, InjectedCredential, InjectionShape,
    VaultBackend, VaultError,
};
use tempfile::TempDir;

#[test]
fn credential_value_basic_api() {
    let v = CredentialValue::new("hunter2".to_string());
    assert_eq!(v.as_str(), "hunter2");
    assert_eq!(v.len(), 7);
    assert!(!v.is_empty());

    let empty = CredentialValue::new(String::new());
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    let from_string: CredentialValue = String::from("xyz").into();
    assert_eq!(from_string.as_str(), "xyz");
}

#[test]
fn credential_value_debug_redacts_with_length() {
    let v = CredentialValue::new("12345".to_string());
    let dbg = format!("{v:?}");
    assert!(!dbg.contains("12345"));
    assert!(dbg.contains("redacted"));
    assert!(dbg.contains('5'), "length should appear in debug: {dbg}");
}

#[test]
fn injected_credential_constructors() {
    let env = InjectedCredential::env("HOME", CredentialValue::new("/tmp".into()));
    assert!(matches!(env.shape, InjectionShape::EnvVar { ref name } if name == "HOME"));
    assert_eq!(env.value.as_str(), "/tmp");

    let hdr = InjectedCredential::header("X-Auth", "tok".to_string());
    assert!(matches!(hdr.shape, InjectionShape::Header { ref name } if name == "X-Auth"));
    assert_eq!(hdr.value.as_str(), "tok");

    let arg = InjectedCredential::arg(3, CredentialValue::new("v".into()));
    assert!(matches!(arg.shape, InjectionShape::Arg { position } if position == 3));
}

#[test]
fn injection_shape_serde_tagged_roundtrip() {
    let shapes = [
        InjectionShape::EnvVar { name: "X".into() },
        InjectionShape::Header { name: "Y".into() },
        InjectionShape::Arg { position: 2 },
    ];
    for s in shapes {
        let j = serde_json::to_value(&s).unwrap();
        // tagged with `shape`; rename_all = snake_case
        assert!(j.get("shape").is_some(), "missing tag: {j}");
        let back: InjectionShape = serde_json::from_value(j).unwrap();
        assert_eq!(back, s);
    }

    let env_json = serde_json::json!({"shape": "env_var", "name": "X"});
    assert_eq!(
        serde_json::from_value::<InjectionShape>(env_json).unwrap(),
        InjectionShape::EnvVar { name: "X".into() }
    );
}

#[test]
fn credential_metadata_serde_roundtrip() {
    let m = CredentialMetadata {
        tool: "github".into(),
        key: "token".into(),
        backend: "file".into(),
        created_at: "2026-05-25T00:00:00Z".into(),
        last_used_at: Some("2026-05-25T01:00:00Z".into()),
        shape: InjectionShape::EnvVar {
            name: "GITHUB_TOKEN".into(),
        },
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: CredentialMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(back, m);
}

#[test]
fn backend_kind_display_strings() {
    assert_eq!(BackendKind::File.to_string(), "file");
    assert_eq!(BackendKind::Keychain.to_string(), "keychain");
    assert_eq!(BackendKind::SecretService.to_string(), "secret-service");
}

#[test]
fn backend_selection_default_is_auto() {
    assert_eq!(BackendSelection::default(), BackendSelection::Auto);
    let cfg = VaultConfig::default();
    assert_eq!(cfg.backend, BackendSelection::Auto);
    assert!(cfg.dir.is_none());
    assert!(cfg.passphrase_file.is_none());
}

#[test]
fn passphrase_source_direct_resolves() {
    let p = PassphraseSource::Direct("abc".into());
    assert_eq!(p.resolve().unwrap(), "abc");
}

#[test]
fn passphrase_source_file_trims_trailing_newlines() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("pp.txt");
    std::fs::write(&path, "secretpw\r\n\n").unwrap();
    let p = PassphraseSource::File(path);
    assert_eq!(p.resolve().unwrap(), "secretpw");
}

#[test]
fn passphrase_source_envvar_missing_returns_passphrase_missing() {
    let name = "BRAIN_VAULT_TEST_UNSET_XYZ_42";
    // Defensive: ensure not set in the runner environment.
    std::env::remove_var(name);
    let p = PassphraseSource::EnvVar(name.into());
    match p.resolve() {
        Err(VaultError::PassphraseMissing) => {}
        other => panic!("expected PassphraseMissing, got {other:?}"),
    }
}

#[test]
fn resolve_backend_file_selection_yields_file_backend() {
    let dir = TempDir::new().unwrap();
    let cfg = VaultConfig {
        backend: BackendSelection::File,
        dir: Some(dir.path().to_path_buf()),
        passphrase_file: None,
    };
    let backend = resolve_backend(&cfg).expect("file resolve");
    assert!(matches!(backend, VaultBackend::File(_)));
    assert_eq!(backend.kind(), BackendKind::File);
}

#[test]
fn resolve_backend_auto_prefers_existing_file_vault() {
    // If `.verifier` exists in the configured dir, Auto sticks with file
    // even on macOS/Linux where a keychain would otherwise be picked.
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join(".verifier"), "$argon2id$irrelevant").unwrap();
    let cfg = VaultConfig {
        backend: BackendSelection::Auto,
        dir: Some(dir.path().to_path_buf()),
        passphrase_file: None,
    };
    let backend = resolve_backend(&cfg).expect("auto resolve with sticky file");
    assert_eq!(backend.kind(), BackendKind::File);
}

#[test]
fn resolve_backend_missing_dir_and_no_home_errors() {
    // Force the HOME-not-set branch by clearing HOME and leaving dir unset.
    // Other tests in this process don't read HOME directly, but be polite
    // and restore.
    let saved = std::env::var_os("HOME");
    std::env::remove_var("HOME");

    let cfg = VaultConfig {
        backend: BackendSelection::File,
        dir: None,
        passphrase_file: Some(PathBuf::from("/nonexistent/pp")),
    };
    let result = resolve_backend(&cfg);

    if let Some(h) = saved {
        std::env::set_var("HOME", h);
    }
    match result {
        Err(VaultError::BackendUnavailable(msg)) => assert!(msg.contains("HOME")),
        Err(other) => panic!("expected BackendUnavailable for HOME-not-set, got {other:?}"),
        Ok(_) => panic!("expected error, got Ok(VaultBackend)"),
    }
}

#[test]
fn vault_error_display_strings() {
    let nf = VaultError::NotFound {
        tool: "t".into(),
        key: "k".into(),
    };
    assert!(nf.to_string().contains("tool=t"));
    assert!(nf.to_string().contains("key=k"));

    let bp = VaultError::BadPassphrase;
    assert!(bp.to_string().contains("verifier"));

    let pm = VaultError::PassphraseMissing;
    assert!(pm.to_string().contains("passphrase"));

    let cr = VaultError::Crypto("boom".into());
    assert!(cr.to_string().contains("boom"));
}
