#!/usr/bin/env bash
# Module boundary enforcement.
#
# Walks every workspace crate's [dependencies] section (skipping
# build/dev-deps) and flags edges that violate the project's layering
# rules:
#
#   1. No workspace crate may depend on `brainos` (the CLI binary crate).
#      Cargo will refuse this cycle if it ever shipped, but catching it
#      earlier gives a clearer error than a `cycle detected` from cargo.
#   2. Transport adapters under `crates/adapters/` may not depend on each
#      other. Each adapter is a leaf — the binary composes them, they
#      don't compose siblings.
#
# Add new rules below as the architecture stabilises.
#
# Run locally:  ./scripts/check-boundaries.sh

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

violations=0

note() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() {
    printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2
    violations=$((violations + 1))
}

# Set of every internal workspace crate folder.
crate_dirs=()
while IFS= read -r dir; do
    crate_dirs+=("$dir")
done < <(
    find crates -mindepth 1 -maxdepth 1 -type d ! -name adapters
    find crates/adapters -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true
)

# Adapter dirs only.
adapter_dirs=()
while IFS= read -r dir; do
    adapter_dirs+=("$dir")
done < <(find crates/adapters -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true)

# Return the [dependencies] block only — i.e. everything between
# `[dependencies]` and the next `[section]`. Includes blank lines.
deps_block() {
    awk '
        /^\[dependencies\][[:space:]]*$/ { in_block=1; next }
        /^\[/ { in_block=0 }
        in_block { print }
    ' "$1"
}

# Rule 1: `brainos` must not appear as a [dependencies] entry anywhere.
note "Rule 1: no workspace crate depends on the CLI binary crate (brainos)"
for dir in "${crate_dirs[@]}"; do
    cargo_toml="$dir/Cargo.toml"
    [ -f "$cargo_toml" ] || continue
    if deps_block "$cargo_toml" | grep -qE '^[[:space:]]*brainos[[:space:]]*='; then
        fail "$dir depends on brainos (the CLI binary crate)"
    fi
done

# Rule 2: adapters may not depend on each other.
note "Rule 2: transport adapters do not depend on each other"
# Collect aliases declared inside [workspace.dependencies] that point at
# crates/adapters/...
declare -a adapter_aliases=()
in_block=0
while IFS= read -r line; do
    case "$line" in
        '[workspace.dependencies]'*) in_block=1; continue ;;
        '['*) in_block=0; continue ;;
    esac
    [ "$in_block" -eq 1 ] || continue
    case "$line" in
        *'path = "crates/adapters/'*)
            alias="${line%%=*}"
            alias="${alias// /}"
            alias="${alias//	/}"
            [ -n "$alias" ] && adapter_aliases+=("$alias")
            ;;
    esac
done < Cargo.toml

for adir in "${adapter_dirs[@]}"; do
    cargo_toml="$adir/Cargo.toml"
    [ -f "$cargo_toml" ] || continue
    block="$(deps_block "$cargo_toml")"
    for alias in "${adapter_aliases[@]}"; do
        # Skip self.
        if grep -q "path = \"$adir\"" Cargo.toml | grep -q "^$alias\s*=" ; then
            continue
        fi
        if printf '%s' "$block" | grep -qE "^[[:space:]]*$alias[[:space:]]*="; then
            # Confirm this alias isn't the adapter's own alias.
            self_alias=""
            for a in "${adapter_aliases[@]}"; do
                if grep -qE "^$a\s*=\s*\{.*\"$adir\"" Cargo.toml; then
                    self_alias="$a"; break
                fi
            done
            if [ "$alias" != "$self_alias" ]; then
                fail "$adir depends on sibling adapter alias '$alias'"
            fi
        fi
    done
done

if [ "$violations" -ne 0 ]; then
    echo "" >&2
    echo "Found $violations module-boundary violation(s)." >&2
    exit 1
fi

echo "check-boundaries: OK"
