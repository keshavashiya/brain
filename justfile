# Brain — Task Runner

default:
    @echo "Brain Development Tasks"
    @echo ""
    @echo "Build:"
    @echo "  build          - Build workspace (debug)"
    @echo "  buildrelease   - Build workspace (release)"
    @echo ""
    @echo "Test:"
    @echo "  test           - Run all tests"
    @echo "  testcrate      - Run tests for a specific crate"
    @echo ""
    @echo "Dev:"
    @echo "  run            - Run brain CLI"
    @echo "  chat           - Start interactive chat"
    @echo "  status         - Show brain status"
    @echo "  fmt            - Format code"
    @echo "  lint           - Run clippy"
    @echo "  check          - Check without building"
    @echo ""
    @echo "Setup:"
    @echo "  downloadmodels - Download ONNX embedding model"
    @echo ""
    @echo "Clean:"
    @echo "  clean          - Clean build artifacts"

# Build
build:
    cargo build --workspace

buildrelease:
    cargo build --release --workspace

check:
    cargo check --workspace

# Test
test:
    cargo test --workspace

testcrate crate:
    cargo test -p {{crate}}

# Run
run *args:
    cargo run --bin brain -- {{args}}

chat *msg:
    cargo run --bin brain -- chat {{msg}}

status:
    cargo run --bin brain -- status

# Dev tools
fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace -- -D warnings

fmtcheck:
    cargo fmt --all -- --check

# Setup
downloadmodels:
    #!/usr/bin/env bash
    set -e
    mkdir -p models
    if [ ! -f models/bge-small-en-v1.5.onnx ]; then
        echo "Downloading BGE-small-en-v1.5 ONNX model..."
        curl -L -o models/bge-small-en-v1.5.onnx \
            "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx"
        echo "Model downloaded"
    else
        echo "Model already exists"
    fi

# Clean
clean:
    cargo clean
