# Brain OS — multi-stage Dockerfile for the `brain` binary.
#
# Stage 1 builds a static-ish release binary against musl when targeting
# linux/amd64 or linux/arm64. Stage 2 ships only the binary on a minimal
# distroless base.
#
# Build for the host platform:
#   docker build -t brainos:dev .
#
# Build multi-arch (requires buildx):
#   docker buildx build \
#     --platform linux/amd64,linux/arm64 \
#     --tag ghcr.io/keshavashiya/brain:0.4.0 \
#     --push .
#
# The default feature set (`grpc`, `encryption`, `ganglia`) is enabled.
# Override with `--build-arg FEATURES='--no-default-features'` etc.

ARG RUST_VERSION=1.85
ARG DEBIAN_FRONTEND=noninteractive

# ---------- Stage 1: build ----------
FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /src

# System deps. protoc is vendored via `protobuf-src` in the grpc adapter's
# build.rs, so we don't need a system protoc. We do need a C toolchain for
# rusqlite's bundled SQLite and for HNSW's C extensions, plus pkg-config for
# anything that probes via it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        ca-certificates \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Cache layer for dependencies. Copy manifests first; the workspace touches
# every crate's Cargo.toml so a single source-file edit doesn't bust the
# dep-cache layer.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

ARG FEATURES=""
RUN cargo build --release --locked --bin brain ${FEATURES} && \
    strip /src/target/release/brain

# ---------- Stage 2: runtime ----------
FROM gcr.io/distroless/cc-debian12:nonroot

LABEL org.opencontainers.image.title="Brain OS"
LABEL org.opencontainers.image.description="Local-first AI memory engine"
LABEL org.opencontainers.image.source="https://github.com/keshavashiya/brain"
LABEL org.opencontainers.image.licenses="MIT"

# Brain stores state under ~/.brain by default; the distroless `nonroot`
# user is uid 65532. Mount a volume here to persist across container
# restarts:  -v brain-data:/home/nonroot/.brain
USER nonroot:nonroot
WORKDIR /home/nonroot

COPY --from=builder /src/target/release/brain /usr/local/bin/brain

# Default HTTP / WS / gRPC ports — adjust via config if you remap.
EXPOSE 7777 7778 7779

ENTRYPOINT ["/usr/local/bin/brain"]
CMD ["serve", "--http", "--ws", "--mcp"]
