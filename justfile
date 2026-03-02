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
    @echo "  serve          - Start all adapters (foreground)"
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

# Dev tools
fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace -- -D warnings

fmtcheck:
    cargo fmt --all -- --check

# Clean
clean:
    cargo clean
