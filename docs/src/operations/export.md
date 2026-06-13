# Export & Import

## Export

```bash
brain export                    # Plaintext JSON export
brain export --encrypt          # Encrypted envelope (default when at-rest encryption is on)
brain export --output file.json
```

The export envelope includes:
- All episodic and semantic memories
- Procedural memory (trigger patterns)
- Memory namespaces and residency markers
- Config metadata (no secrets)

Encrypted exports are self-contained envelopes using AES-256-GCM with a fresh Argon2id-derived key and embedded salt — portable across machines.

## Import

```bash
brain import file.json          # Import plaintext
brain import file.enc           # Import encrypted (auto-detected)
```

Imports preserve namespace boundaries. Encrypted imports prompt for passphrase.
