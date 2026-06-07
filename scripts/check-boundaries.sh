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
#   3. No dependency cycles among internal workspace crates. Cargo refuses
#      to build a cyclic crate graph, but it reports the failure as a terse
#      `cyclic package dependency` deep in a compile; detecting it here from
#      the manifests alone is fast (no build) and prints the offending path.
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

# Rule 3: no dependency cycles among internal crates.
note "Rule 3: no dependency cycles among internal workspace crates"

# alias<TAB>crate-path for every internal (path-based) workspace dependency.
# These aliases are exactly the set of internal crate handles; each crate
# pulls a sibling via `<alias> = { workspace = true }` in [dependencies].
alias_paths="$(
    awk '
        /^\[workspace\.dependencies\]/ { in_block=1; next }
        /^\[/ { in_block=0 }
        in_block && /path[[:space:]]*=[[:space:]]*"crates\// {
            s=$0
            sub(/^.*path[[:space:]]*=[[:space:]]*"/, "", s)
            sub(/".*$/, "", s)
            print $1 "\t" s
        }
    ' Cargo.toml
)"

# Space-padded set of internal aliases for cheap membership tests.
internal=" $(printf '%s\n' "$alias_paths" | cut -f1 | tr '\n' ' ') "

# Emit "<from> <to>" runtime-dependency edges (build/dev-deps excluded — cargo
# permits cycles through those, so they must not count here).
tab="$(printf '\t')"
edges="$(
    printf '%s\n' "$alias_paths" | while IFS="$tab" read -r alias path; do
        [ -n "$alias" ] || continue
        [ -f "$path/Cargo.toml" ] || continue
        deps_block "$path/Cargo.toml" | while IFS= read -r line; do
            dep="$(printf '%s' "$line" | sed -n 's/^[[:space:]]*\([A-Za-z0-9_-]*\)[[:space:]]*=.*/\1/p')"
            [ -n "$dep" ] || continue
            case "$internal" in
                *" $dep "*) printf '%s %s\n' "$alias" "$dep" ;;
            esac
        done
    done
)"

# DFS three-colour cycle detection. Prints the offending path top-down back to
# the repeated node, e.g. "signal <- cortex <- signal".
cycle="$(
    printf '%s\n' "$edges" | awk '
        { if ($1 != "") { adj[$1]=adj[$1] " " $2; nodes[$1]=1; nodes[$2]=1 } }
        END {
            for (n in nodes) color[n]=0
            for (n in nodes) if (color[n]==0) { if (visit(n)) exit 0 }
        }
        function visit(u,   i,cnt,arr,v,j,line) {
            color[u]=1; sp++; stk[sp]=u
            cnt=split(adj[u], arr, " ")
            for (i=1;i<=cnt;i++) {
                v=arr[i]; if (v=="") continue
                if (color[v]==1) {
                    line=v
                    for (j=sp;j>=1;j--) { line=line " <- " stk[j]; if (stk[j]==v) break }
                    print line
                    return 1
                } else if (color[v]==0) {
                    if (visit(v)) return 1
                }
            }
            color[u]=2; sp--; return 0
        }
    '
)"

if [ -n "$cycle" ]; then
    fail "internal dependency cycle: $cycle"
fi

if [ "$violations" -ne 0 ]; then
    echo "" >&2
    echo "Found $violations module-boundary violation(s)." >&2
    exit 1
fi

echo "check-boundaries: OK"
