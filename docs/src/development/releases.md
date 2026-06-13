# Release Process

## Versioning

Brain follows semantic versioning. The workspace version is locked across all 33 crates.

## Release pipeline

Releases are driven by `scripts/release.sh` (local) and `.github/workflows/release.yml` (CI):

### Local (human-driven)

```bash
scripts/release.sh X.Y.Z    # Validate → CI check → publish → tag
scripts/release.sh X.Y.Z --dry-run  # Dry run (no publish)
scripts/release.sh X.Y.Z --skip-ci  # Skip CI check (re-runs)
```

Steps:
1. Validates clean tree, version match, populated CHANGELOG
2. Runs CI parity (fmt + clippy + tests)
3. Publishes all crates in dependency order via `scripts/publish-order.sh`
4. Creates annotated `vX.Y.Z` tag and pushes

### CI (automated off the pushed tag)

Triggered by pushing a `v*` tag:
- Builds `brain-<target>.tar.gz` + `.sha256` for macOS/Linux (x86_64 + aarch64)
- Generates SPDX SBOM
- Creates GitHub Release with binaries + checksums

## Changelog

Every release requires an updated `CHANGELOG.md` with the `[X.Y.Z]` section populated. Extract release notes with:

```bash
scripts/changelog-extract.sh X.Y.Z
```
