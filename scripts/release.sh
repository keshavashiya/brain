#!/usr/bin/env bash
# Brain OS release driver.
#
# Publishing 31 crates to crates.io is irreversible and a partial failure
# mid-publish leaves the world inconsistent (see docs/JOURNAL.md). This
# script makes the local-driven half of a release safe and repeatable:
# it validates everything that can be validated *before* the first
# irreversible step, then publishes in dependency order, then tags.
#
# The tag push is what triggers .github/workflows/release.yml, which builds
# the platform binaries and creates the GitHub Release. So the division is:
#   - this script:      validate -> cargo publish -> git tag + push
#   - release.yml (CI): binaries + GitHub Release + SBOM, on the pushed tag
#
# Manual prep before running (kept manual so a human owns the version edit):
#   1. Bump `version` under [workspace.package] in Cargo.toml.
#   2. Rename CHANGELOG.md's `## [Unreleased]` to `## [X.Y.Z] — YYYY-MM-DD`
#      (and start a fresh empty [Unreleased] above it).
#   3. Commit both on a release branch / main.
#   4. Run:  ./scripts/release.sh 0.5.0
#
# Flags:
#   --dry-run      cargo publish --dry-run for every crate; no tag, no push.
#   --skip-ci      Skip the local CI-parity gate (fmt/clippy/check/test).
#   --skip-publish Skip cargo publish (tag-only; e.g. re-tagging a release
#                  whose crates already published).
#   --no-push      Create the tag locally but do not push it.
#   --allow-dirty  Pass through to cargo publish AND skip the clean-tree
#                  check (use only when you know what you're doing).
#
# Usage:
#   ./scripts/release.sh 0.5.0
#   ./scripts/release.sh 0.5.0 --dry-run

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- args --------------------------------------------------------------
version=""
dry_run=0
skip_ci=0
skip_publish=0
no_push=0
allow_dirty=0

for arg in "$@"; do
    case "$arg" in
        --dry-run)      dry_run=1 ;;
        --skip-ci)      skip_ci=1 ;;
        --skip-publish) skip_publish=1 ;;
        --no-push)      no_push=1 ;;
        --allow-dirty)  allow_dirty=1 ;;
        -*)             die "unknown flag: $arg" ;;
        *)
            if [ -n "$version" ]; then
                die "unexpected extra argument: $arg"
            fi
            version="${arg#v}"
            ;;
    esac
done

[ -n "$version" ] || die "usage: $0 <version> [--dry-run] [--skip-ci] [--skip-publish] [--no-push] [--allow-dirty]"

# Sanity: X.Y.Z semver shape.
if ! printf '%s' "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    die "version '$version' is not X.Y.Z"
fi

tag="v$version"

# ---- 1. clean working tree --------------------------------------------
if [ "$allow_dirty" -eq 0 ]; then
    if [ -n "$(git status --porcelain)" ]; then
        die "working tree is dirty; commit or stash first (or pass --allow-dirty)"
    fi
fi

# ---- 2. workspace version matches requested release --------------------
workspace_version="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
if [ "$workspace_version" != "$version" ]; then
    die "Cargo.toml [workspace.package] version is '$workspace_version' but you asked to release '$version'.
       Bump the version in Cargo.toml first (see header of this script)."
fi
info "Workspace version: $workspace_version"

# ---- 3. CHANGELOG section exists and is non-empty ----------------------
if ! ./scripts/changelog-extract.sh "$version" >/dev/null; then
    die "CHANGELOG.md has no populated '## [$version]' section. Stamp it before releasing."
fi
info "CHANGELOG.md '[$version]' section is populated"

# ---- 4. tag must not already exist -------------------------------------
if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    die "tag $tag already exists locally"
fi

# ---- 5. CI parity gate -------------------------------------------------
# Mirror the CI stage list (see feedback: run the full set locally before
# a release, not just cargo check). Skippable for re-runs.
if [ "$skip_ci" -eq 0 ]; then
    info "CI parity: cargo fmt --check"
    cargo fmt --all -- --check
    info "CI parity: cargo clippy -D warnings"
    cargo clippy --workspace --all-targets -- -D warnings
    info "CI parity: cargo check (no default features)"
    cargo check -p brainos --no-default-features
    cargo check -p brainos --no-default-features --features encryption
    info "CI parity: cargo check --workspace --locked"
    cargo check --workspace --locked
    info "CI parity: cargo test --workspace --locked"
    cargo test --workspace --locked
    info "CI parity: cargo build --workspace --locked"
    cargo build --workspace --locked
else
    warn "--skip-ci: skipping local CI-parity gate"
fi

# ---- 6. publish crates in dependency order -----------------------------
if [ "$skip_publish" -eq 0 ]; then
    order="$(./scripts/publish-order.sh)"
    publish_flags=(--locked)
    [ "$dry_run" -eq 1 ] && publish_flags+=(--dry-run)
    [ "$allow_dirty" -eq 1 ] && publish_flags+=(--allow-dirty)

    total="$(printf '%s\n' "$order" | grep -c .)"
    n=0
    while IFS= read -r crate; do
        [ -n "$crate" ] || continue
        n=$((n + 1))
        info "[$n/$total] cargo publish -p $crate ${publish_flags[*]}"
        # cargo (>=1.66) waits for the just-published crate to be available
        # in the index before returning, so the next crate in the order can
        # resolve it. No manual sleep needed.
        cargo publish -p "$crate" "${publish_flags[@]}"
    done <<< "$order"
else
    warn "--skip-publish: not publishing to crates.io"
fi

if [ "$dry_run" -eq 1 ]; then
    bold "Dry run complete. No tag created, nothing pushed."
    exit 0
fi

# ---- 7. tag + push (push triggers release.yml binary builds) -----------
notes="$(./scripts/changelog-extract.sh "$version")"
info "Creating annotated tag $tag"
git tag -a "$tag" -m "Brain OS $version" -m "$notes"

if [ "$no_push" -eq 1 ]; then
    bold "Tag $tag created locally. --no-push set; run 'git push origin $tag' to trigger the release workflow."
    exit 0
fi

info "Pushing $tag to origin"
git push origin "$tag"

bold "Released $version."
echo "  - crates.io: published in dependency order"
echo "  - tag $tag pushed; .github/workflows/release.yml will build binaries + the GitHub Release"
