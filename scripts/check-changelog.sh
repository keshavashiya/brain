#!/usr/bin/env bash
# Changelog validation.
#
# Two checks:
#   1. Structural: the workspace version declared in Cargo.toml has a
#      matching `## [VERSION]` section in CHANGELOG.md.
#   2. PR diff:  if any Rust source or Cargo manifest changed against the
#      base branch, CHANGELOG.md must also have changed (under [Unreleased]
#      or the current version section).
#
# The diff check is skipped when no base branch can be determined (e.g.
# running on `main` push). Override the base with $BASE_REF.
#
# Pass --no-diff to skip the diff check explicitly (useful for local runs
# on a clean tree).

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

check_diff=1
for arg in "$@"; do
    case "$arg" in
        --no-diff) check_diff=0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

violations=0

# -- 1. Structural: workspace version has a matching CHANGELOG section ----
workspace_version="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
if [ -z "$workspace_version" ]; then
    echo "error: could not parse workspace version from Cargo.toml" >&2
    violations=$((violations + 1))
elif ! grep -qE "^## \[$workspace_version\]" CHANGELOG.md; then
    echo "error: CHANGELOG.md is missing a section for workspace version $workspace_version" >&2
    echo "       expected a line like:  ## [$workspace_version] — <date|unreleased>" >&2
    violations=$((violations + 1))
fi

# -- 2. PR diff: source changes require a changelog touch ----------------
if [ "$check_diff" -eq 1 ]; then
    base_ref="${BASE_REF:-}"
    if [ -z "$base_ref" ]; then
        # Best-effort: prefer origin/main, fall back to main, else skip.
        if git rev-parse --verify origin/main >/dev/null 2>&1; then
            base_ref="origin/main"
        elif git rev-parse --verify main >/dev/null 2>&1; then
            base_ref="main"
        fi
    fi

    if [ -z "$base_ref" ]; then
        echo "note: no base ref detected; skipping changelog diff check"
    else
        # Files changed vs base. `git diff --name-only base...HEAD` gives the
        # merge-base diff (i.e. only what this branch added).
        changed="$(git diff --name-only "$base_ref"...HEAD 2>/dev/null || true)"
        if [ -z "$changed" ]; then
            echo "note: no changes against $base_ref; skipping changelog diff check"
        else
            needs_changelog=0
            while IFS= read -r file; do
                case "$file" in
                    crates/*/src/*|crates/*/*/src/*|Cargo.toml|crates/*/Cargo.toml|crates/*/*/Cargo.toml)
                        needs_changelog=1
                        break
                        ;;
                esac
            done <<< "$changed"

            if [ "$needs_changelog" -eq 1 ]; then
                if ! grep -qx "CHANGELOG.md" <<< "$changed"; then
                    echo "error: source or manifest files changed vs $base_ref but CHANGELOG.md was not updated" >&2
                    echo "       add an entry under [Unreleased] (or the active version section) — or document why this is a no-op for users." >&2
                    violations=$((violations + 1))
                fi
            fi
        fi
    fi
fi

if [ "$violations" -ne 0 ]; then
    echo "" >&2
    echo "Found $violations changelog violation(s)." >&2
    exit 1
fi

echo "check-changelog: OK"
