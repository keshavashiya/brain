#!/usr/bin/env bash
# Brain OS one-line installer for Linux and macOS.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/keshavashiya/brain/main/scripts/install.sh | sh
#
# Order of preference:
#   1. Pre-built binary from the latest GitHub Release (matching host arch).
#   2. Fallback: `cargo install brainos` from crates.io if a Rust toolchain
#      is present.
#   3. Final fallback: `cargo install --git https://github.com/keshavashiya/brain`.
#
# Flags via env:
#   BRAIN_VERSION   Pin a release tag (default: latest).
#   BRAIN_PREFIX    Install dir (default: $HOME/.local/bin).
#   BRAIN_NO_BIN    If set, skip binary attempt and go straight to cargo.

set -eu

REPO="keshavashiya/brain"
PREFIX="${BRAIN_PREFIX:-$HOME/.local/bin}"
VERSION="${BRAIN_VERSION:-latest}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- detect platform ---------------------------------------------------
uname_s="$(uname -s 2>/dev/null || echo unknown)"
uname_m="$(uname -m 2>/dev/null || echo unknown)"
case "$uname_s" in
    Linux)  os="unknown-linux-gnu" ;;
    Darwin) os="apple-darwin" ;;
    *)      die "unsupported OS: $uname_s (try 'cargo install brainos' instead)" ;;
esac
case "$uname_m" in
    x86_64|amd64) arch="x86_64" ;;
    arm64|aarch64) arch="aarch64" ;;
    *) die "unsupported architecture: $uname_m" ;;
esac
target="${arch}-${os}"
info "Detected platform: $target"

# ---- check for `brain` already on PATH ---------------------------------
if command -v brain >/dev/null 2>&1; then
    current="$(brain --version 2>/dev/null || true)"
    warn "brain is already installed: $current"
    warn "Reinstalling will overwrite. Continue? (Ctrl-C to abort)"
    sleep 2
fi

# ---- try pre-built binary ----------------------------------------------
install_from_binary() {
    [ -n "${BRAIN_NO_BIN:-}" ] && return 1

    if [ "$VERSION" = "latest" ]; then
        url_base="https://github.com/$REPO/releases/latest/download"
    else
        url_base="https://github.com/$REPO/releases/download/$VERSION"
    fi
    tarball="brain-${target}.tar.gz"
    url="$url_base/$tarball"

    info "Probing $url"
    if ! curl -fsSL --head "$url" >/dev/null 2>&1; then
        warn "No pre-built binary at $url (tag may not have published assets yet)"
        return 1
    fi

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    info "Downloading $tarball"
    curl -fsSL "$url" -o "$tmp/$tarball" || return 1
    tar -xzf "$tmp/$tarball" -C "$tmp" || return 1

    mkdir -p "$PREFIX"
    install -m 0755 "$tmp/brain" "$PREFIX/brain" || return 1
    return 0
}

# ---- cargo fallback ----------------------------------------------------
install_from_cargo() {
    if ! command -v cargo >/dev/null 2>&1; then
        die "cargo not found, and no pre-built binary available for $target. \
Install Rust first (https://rustup.rs) or try again after a Brain release ships a binary for this platform."
    fi
    info "Installing via cargo (this builds from source — may take several minutes)"
    if cargo install brainos --locked; then
        return 0
    fi
    warn "crates.io install failed; trying git source"
    cargo install --git "https://github.com/$REPO" --locked brainos
}

if install_from_binary; then
    bold "Installed brain to $PREFIX/brain"
else
    install_from_cargo
fi

# ---- next steps --------------------------------------------------------
case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *) warn "$PREFIX is not on your PATH. Add it to your shell rc:"
       printf '  export PATH=\"%s:$PATH\"\n' "$PREFIX" ;;
esac

cat <<EOF

$(bold "Next steps:")
  brain init      # write ~/.brain/config.yaml + print your API key
  brain doctor    # verify dependencies (Ollama, models, ports)
  brain start     # wake the daemon

Documentation: https://github.com/$REPO#readme
EOF
