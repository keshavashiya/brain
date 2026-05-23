#!/usr/bin/env bash
# Pre-commit gate. Mirrors the CI jobs that are cheap enough to run locally:
#   - cargo fmt --all --check
#   - cargo clippy --workspace -- -D warnings
#   - ./scripts/check-crate-names.sh
#
# Heavier jobs (cargo check across feature combos, cargo test) are deliberately
# left for CI / pre-push, since this hook runs on every commit.
#
# Skip with `git commit --no-verify` when you really mean it.

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

failures=0

step() {
    local label="$1"; shift
    printf '\033[1;34m==>\033[0m %s\n' "$label"
    if ! "$@"; then
        printf '\033[1;31mFAIL\033[0m %s\n' "$label" >&2
        failures=$((failures + 1))
    fi
}

step "cargo fmt --all -- --check"   cargo fmt --all -- --check
step "scripts/check-crate-names.sh" ./scripts/check-crate-names.sh
step "cargo clippy --workspace"     cargo clippy --workspace --all-targets -- -D warnings

if [ "$failures" -ne 0 ]; then
    printf '\n\033[1;31m%s pre-commit check(s) failed.\033[0m\n' "$failures" >&2
    echo "Fix the issues above, or bypass with 'git commit --no-verify' if you know what you're doing." >&2
    exit 1
fi

printf '\n\033[1;32mpre-commit: OK\033[0m\n'
