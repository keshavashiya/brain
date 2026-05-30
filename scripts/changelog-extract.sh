#!/usr/bin/env bash
# Extract the release-notes body for a single version section from
# CHANGELOG.md.
#
# Given a version like `0.5.0`, prints everything between the
# `## [0.5.0] ...` heading and the next `## [` heading (exclusive),
# with leading/trailing blank lines trimmed. Used by:
#   - scripts/release.sh        (validates the section is non-empty)
#   - .github/workflows/release.yml  (release body)
#
# Usage:
#   ./scripts/changelog-extract.sh 0.5.0
#   ./scripts/changelog-extract.sh v0.5.0      # leading 'v' is tolerated
#
# Exit codes:
#   0  section found and printed
#   1  no matching `## [VERSION]` heading
#   2  usage error

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
changelog="$root_dir/CHANGELOG.md"

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <version>" >&2
    exit 2
fi

# Tolerate a leading 'v' so callers can pass either a tag or a bare version.
version="${1#v}"

if [ ! -f "$changelog" ]; then
    echo "error: $changelog not found" >&2
    exit 2
fi

# awk pulls the lines strictly between the target heading and the next
# `## [` heading. We match the version literally (escaping regex metachars
# in the dots) by comparing the bracketed token rather than a regex.
body="$(awk -v ver="$version" '
    # Heading lines look like:  ## [0.5.0] — 2026-05-30
    /^## \[/ {
        # Extract the token inside the first [...] on the line.
        line = $0
        sub(/^## \[/, "", line)
        sub(/\].*$/, "", line)
        if (in_section) {
            # We hit the next section heading; stop.
            exit
        }
        if (line == ver) {
            in_section = 1
            next
        }
    }
    in_section { print }
' "$changelog")"

if [ -z "${body//[$'\n\t ']/}" ]; then
    # Either the heading was absent, or the section had no content.
    if ! grep -qE "^## \[${version//./\\.}\]" "$changelog"; then
        echo "error: no '## [$version]' section in CHANGELOG.md" >&2
        exit 1
    fi
    echo "error: '## [$version]' section in CHANGELOG.md is empty" >&2
    exit 1
fi

# Trim leading and trailing blank lines (portable awk; BSD/GNU alike).
printf '%s\n' "$body" | awk '
    { lines[NR] = $0 }
    END {
        start = 1
        while (start <= NR && lines[start] ~ /^[[:space:]]*$/) start++
        end = NR
        while (end >= start && lines[end] ~ /^[[:space:]]*$/) end--
        for (i = start; i <= end; i++) print lines[i]
    }
'
