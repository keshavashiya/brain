#!/usr/bin/env bash
# Dependency staleness audit (Issue 178).
#
# Advisory by default: reports outdated *direct* dependencies and exits 0 so it
# never blocks routine work — Dependabot is the blocking-ish path (it opens PRs
# weekly), and security is enforced by `cargo audit` + `cargo deny` in CI. Pass
# --strict to make a stale dependency a non-zero exit (release-gate use).
#
# Policy: docs/DEPENDENCY_POLICY.md
#
# Run locally:
#   ./scripts/check-freshness.sh            # advisory
#   ./scripts/check-freshness.sh --strict   # fail if anything is outdated

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

strict=0
[ "${1:-}" = "--strict" ] && strict=1

note() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*" >&2; }

if ! command -v cargo-outdated >/dev/null 2>&1; then
    warn "cargo-outdated not installed."
    echo "Install it with:  cargo install cargo-outdated --locked" >&2
    # Not having the tool is not a failure of the dependency graph itself.
    exit 0
fi

note "Checking for outdated direct dependencies (cargo outdated --root-deps-only)"

# --root-deps-only: only the deps we declare, not the whole transitive closure
# (transitives are Dependabot/cargo-deny territory). --workspace: all members.
# cargo-outdated exits non-zero when it finds outdated deps; capture that
# instead of letting `set -e` abort so we can choose the policy.
set +e
out="$(cargo outdated --workspace --root-deps-only 2>&1)"
code=$?
set -e

printf '%s\n' "$out"

# "All dependencies are up to date" is cargo-outdated's clean-state message.
if printf '%s' "$out" | grep -qiE 'all dependencies are up to date'; then
    note "No outdated direct dependencies."
    exit 0
fi

if [ "$strict" -eq 1 ]; then
    warn "Outdated direct dependencies found (strict mode)."
    exit 1
fi

# Advisory mode: surface but don't fail. cargo-outdated's own exit code is
# informational here.
if [ "$code" -ne 0 ]; then
    note "Outdated dependencies above are advisory — schedule bumps via Dependabot."
fi
exit 0
