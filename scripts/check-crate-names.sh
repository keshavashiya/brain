#!/usr/bin/env bash
# Enforce the single-word crate-name convention.
#
# Rules (per docs/CONVENTIONS.md):
#   - Folder names under crates/ are a single lowercase word with no
#     underscores or hyphens. The only exception is the grouping folder
#     `crates/adapters/` which contains transport-adapter crates.
#   - Workspace-dependency aliases (the keys in [workspace.dependencies])
#     are a single word with no underscores.
#
# Exits 0 on success, 1 on any violation.

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

violations=0

# -- Folder-name check ---------------------------------------------------
# Treat crates/adapters/ as a grouping folder; its children are the crate
# folders. Everything else under crates/ is a leaf crate folder.
while IFS= read -r dir; do
    name="$(basename "$dir")"
    case "$name" in
        *_*|*-*)
            echo "error: crate folder name contains underscore or hyphen: $dir" >&2
            violations=$((violations + 1))
            ;;
    esac
done < <(
    find crates -mindepth 1 -maxdepth 1 -type d ! -name adapters
    find crates/adapters -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true
)

# -- Workspace-alias check -----------------------------------------------
# Parse [workspace.dependencies] block. An alias key with an underscore is
# a violation. Skip lines that are clearly not internal aliases (we only
# care about keys that point at `path = "crates/..."`).
in_block=0
while IFS= read -r line; do
    # Detect block boundaries.
    case "$line" in
        '[workspace.dependencies]'*) in_block=1; continue ;;
        '['*) in_block=0; continue ;;
    esac
    [ "$in_block" -eq 1 ] || continue

    # Only consider lines that declare an internal path-dep (i.e. contain
    # `path = "crates/`).
    case "$line" in
        *'path = "crates/'*) ;;
        *) continue ;;
    esac

    # Extract the alias key (text before `=` on the line).
    alias="${line%%=*}"
    alias="${alias// /}"
    alias="${alias//	/}"

    case "$alias" in
        *_*)
            echo "error: workspace alias contains underscore: '$alias'" >&2
            violations=$((violations + 1))
            ;;
    esac
done < Cargo.toml

if [ "$violations" -ne 0 ]; then
    echo "" >&2
    echo "Found $violations crate-naming violation(s)." >&2
    echo "See docs/CONVENTIONS.md (Crate naming section)." >&2
    exit 1
fi

echo "crate-names: OK"
