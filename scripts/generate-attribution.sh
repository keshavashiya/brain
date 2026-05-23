#!/usr/bin/env bash
# Generate THIRD_PARTY_LICENSES.md from the current dependency graph.
#
# Wraps `cargo about` so contributors and CI invoke it the same way.
# Requires `cargo-about` (install: `cargo install cargo-about --locked`).
#
# Flags:
#   --check   Generate to a temp file and diff against the committed copy.
#             Exits non-zero if they differ. Used by CI to catch drift.

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

out="THIRD_PARTY_LICENSES.md"
mode="write"
for arg in "$@"; do
    case "$arg" in
        --check) mode="check" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if ! command -v cargo-about >/dev/null 2>&1; then
    echo "error: cargo-about not installed. Run: cargo install cargo-about --locked" >&2
    exit 1
fi

if [ "$mode" = "check" ]; then
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT
    cargo about generate about.hbs > "$tmp"
    if ! diff -u "$out" "$tmp"; then
        echo "" >&2
        echo "error: $out is stale. Run ./scripts/generate-attribution.sh and commit the result." >&2
        exit 1
    fi
    echo "attribution: up to date"
else
    cargo about generate about.hbs > "$out"
    echo "wrote $out"
fi
