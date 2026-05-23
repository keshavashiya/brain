#!/usr/bin/env bash
# Install repo-managed git hooks by pointing `core.hooksPath` at scripts/git-hooks.
# This keeps the hooks version-controlled and makes opt-in explicit.

set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root_dir"

hooks_dir="scripts/git-hooks"
mkdir -p "$hooks_dir"

cat > "$hooks_dir/pre-commit" <<'HOOK'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/pre-commit.sh" "$@"
HOOK
chmod +x "$hooks_dir/pre-commit"

git config core.hooksPath "$hooks_dir"

echo "Installed git hooks (core.hooksPath = $hooks_dir)."
echo "Disable with: git config --unset core.hooksPath"
