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
    @echo "  test-integration - Run integration tests"
    @echo "  testcrate      - Run tests for a specific crate"
    @echo ""
    @echo "Dev:"
    @echo "  run            - Run brain CLI"
    @echo "  chat           - Start interactive chat"
    @echo "  status         - Show brain status"
    @echo "  serve          - Start all adapters (foreground)"
    @echo "  serve-dev      - Start adapters with debug logging"
    @echo "  ci             - Run fmt + clippy + tests"
    @echo "  health         - Quick local health check"
    @echo "  fmt            - Format code"
    @echo "  lint           - Run clippy"
    @echo "  check          - Check without building"
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

test-integration:
    cargo test --workspace --tests

testcrate crate:
    cargo test -p {{crate}}

# Run
run *args:
    cargo run --bin brain -- {{args}}

chat *msg:
    cargo run --bin brain -- chat {{msg}}

status:
    cargo run --bin brain -- status

serve *args:
    cargo run --bin brain -- serve {{args}}

serve-dev *args:
    RUST_LOG=debug cargo run --bin brain -- serve {{args}}

# Dev tools
fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace -- -D warnings

fmtcheck:
    cargo fmt --all -- --check

ci:
    cargo fmt --all -- --check
    cargo clippy --workspace -- -D warnings
    cargo test --workspace

health:
    cargo run --bin brain -- status
    @curl -sf http://127.0.0.1:19789/health >/dev/null && echo "HTTP OK" || echo "HTTP not running"

# Clean
clean:
    cargo clean
