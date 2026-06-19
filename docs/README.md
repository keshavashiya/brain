# Brain OS documentation

The docs are an [mdBook](https://rust-lang.github.io/mdBook/). Source lives in
`src/`, config in `book.toml`. Build with `mdbook build` (or `mdbook serve` for
live preview). Deployed to GitHub Pages by `.github/workflows/pages.yml`.

## "Ask Brain" docs assistant

A floating widget that answers questions about Brain OS by retrieving the most
relevant doc sections — **verbatim**. It is intentionally *extractive*, not
generative: it never writes prose, so it can only surface what the docs already
say. There is no chatbot to host and no server in the loop.

**What runs where**

| Stage | Where | What |
|-------|-------|------|
| Embed doc sections | build time (CI) | `docs/tools/index.mjs` embeds each section with MiniLM → `src/assistant/brain-index.json` |
| Embed the question | the visitor's browser | transformers.js, same model, downloaded once and cached |
| Rank sections | the visitor's browser (WASM) | `docs/assistant/` — Brain's **own** `synapse` similarity code (`crates/synapse`) compiled to `wasm32`. The same `cosine_similarity` the capability router uses. |

So the assistant genuinely *is* Brain: the ranking that runs in the tab is the
engine's real code, not a re-implementation. No question leaves the browser.

### Building the assistant locally

Two generated artifacts must exist before `mdbook build` (both are gitignored —
CI rebuilds them, see `pages.yml`):

```sh
# 1. Retrieval engine → src/assistant/pkg/   (needs: rustup target add wasm32-unknown-unknown, wasm-pack)
wasm-pack build docs/assistant --target web --out-dir ../src/assistant/pkg --release

# 2. Search index → src/assistant/brain-index.json   (needs: Node 20+)
cd docs/tools && npm ci && node index.mjs

# 3. The book itself
cd ../.. && mdbook build docs
```

If you skip steps 1–2 and just `mdbook build`, the widget still renders but
reports it couldn't load (the index/engine 404) — the rest of the docs and the
built-in search are unaffected.

### Changing the embedding model

Set it in **one** place: `MODEL` in `docs/tools/index.mjs`. The id is written
into `brain-index.json`; the widget reads it back and loads the matching model,
so the two sides can never drift. Re-run the indexer after changing it.

### Layout

```
docs/
  book.toml              mdBook config (wires assets/assistant.{css,js})
  src/**.md              the documentation
  src/assistant/         generated: brain-index.json + pkg/ (wasm)  [gitignored]
  assets/assistant.css   widget styling (theme-aware)
  assets/assistant.js    widget: loads wasm + model, renders results
  assistant/             Rust crate: the WASM retrieval engine (own workspace)
  tools/index.mjs        build-time indexer
```
