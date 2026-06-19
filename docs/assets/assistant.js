/* Brain OS docs assistant — the in-browser "Ask Brain" widget.
 *
 * What runs where:
 *   - Ranking is Brain's own Rust `synapse` similarity code, compiled to WASM
 *     (docs/assistant → assistant/pkg). It runs in this tab; no server sees the
 *     question.
 *   - The question is embedded with the same MiniLM sentence model the index
 *     was built with, via transformers.js, downloaded once and cached.
 *   - Results are doc sections returned VERBATIM. The widget never generates
 *     prose, so it can only repeat what the docs already say.
 *
 * Graceful degradation: if the embedding model can't load, search falls back to
 * the WASM engine's lexical (keyword) term — still real, just less fuzzy.
 */
(function () {
  "use strict";

  // transformers.js pinned to the version the build-time indexer used, so
  // tokenizer/runtime behaviour matches. (Embeddings are determined by the
  // model weights from the HF hub, identical on both sides.)
  var TRANSFORMERS_CDN =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
  var TOP_K = 6;

  // mdBook declares `const path_to_root` (e.g. "../") in a classic <script> in
  // each page's head. As a top-level `const` it is a global *lexical* binding —
  // reachable by bare name from other classic scripts, but NOT a property of
  // `window`. `typeof` guards the case where it is somehow absent.
  var BASE = typeof path_to_root !== "undefined" ? path_to_root : ""; // eslint-disable-line no-undef

  // Resolve a site-relative asset path to an absolute URL. Needed because a
  // dynamic import() of a *bare* specifier (e.g. "assistant/pkg/x.js" on the
  // root page, where BASE is "") is rejected by the ES module loader; absolute
  // URLs always import/fetch correctly regardless of page depth.
  function asset(p) {
    return new URL(BASE + p, document.baseURI).href;
  }

  var state = {
    booted: false, // DOM built
    loading: false, // engine/model load in flight
    engine: null, // WASM DocsEngine
    embed: null, // transformers.js pipeline, or null = lexical-only
  };

  // ─── DOM ──────────────────────────────────────────────────────────────────

  var els = {};

  function build() {
    if (state.booted) return;
    state.booted = true;

    var launcher = document.createElement("button");
    launcher.id = "brain-ask-launcher";
    launcher.type = "button";
    launcher.innerHTML =
      '<span class="brain-ask-dot">🧠</span><span>Ask Brain</span>';
    launcher.addEventListener("click", openPanel);

    var panel = document.createElement("div");
    panel.id = "brain-ask-panel";
    panel.innerHTML = [
      '<div class="brain-ask-head">',
      "  <strong>🧠 Ask Brain</strong>",
      '  <button class="brain-ask-close" type="button" aria-label="Close">×</button>',
      "</div>",
      '<form class="brain-ask-form">',
      '  <input type="search" autocomplete="off" placeholder="Ask about Brain OS…" aria-label="Ask about Brain OS" />',
      '  <button type="submit">Ask</button>',
      "</form>",
      '<div class="brain-ask-status"></div>',
      '<div class="brain-ask-results"></div>',
      '<div class="brain-ask-foot"></div>',
    ].join("\n");

    document.body.appendChild(launcher);
    document.body.appendChild(panel);

    els.launcher = launcher;
    els.panel = panel;
    els.form = panel.querySelector(".brain-ask-form");
    els.input = panel.querySelector("input");
    els.status = panel.querySelector(".brain-ask-status");
    els.results = panel.querySelector(".brain-ask-results");
    els.foot = panel.querySelector(".brain-ask-foot");

    panel.querySelector(".brain-ask-close").addEventListener("click", closePanel);
    els.form.addEventListener("submit", onSubmit);
  }

  function openPanel() {
    els.panel.classList.add("open");
    els.launcher.style.display = "none";
    els.input.focus();
    ensureEngine();
  }
  function closePanel() {
    els.panel.classList.remove("open");
    els.launcher.style.display = "";
  }

  function setStatus(msg) {
    els.status.textContent = msg || "";
  }
  function setFoot(html) {
    els.foot.innerHTML = html;
  }

  // ─── Engine / model loading (lazy, on first open) ──────────────────────────

  async function ensureEngine() {
    if (state.engine || state.loading) return;
    state.loading = true;
    setStatus("Loading Brain’s retrieval engine…");
    try {
      // 1) Real Brain ranking code, compiled to WASM.
      var wasm = await import(asset("assistant/pkg/brain_docs_assistant.js"));
      await wasm.default(asset("assistant/pkg/brain_docs_assistant_bg.wasm"));
      var indexText = await fetch(asset("assistant/brain-index.json")).then(
        function (r) {
          if (!r.ok) throw new Error("index " + r.status);
          return r.text();
        }
      );
      state.engine = new wasm.DocsEngine(indexText);

      // 2) The embedding model (the heavy, one-time download). Optional — if it
      //    fails we keep going in keyword mode.
      setStatus("Loading the embedding model (first time only)…");
      try {
        var t = await import(TRANSFORMERS_CDN);
        state.embed = await t.pipeline("feature-extraction", state.engine.model);
      } catch (e) {
        state.embed = null;
        console.warn("[ask-brain] embedding model unavailable, keyword mode", e);
      }

      setStatus("");
      setFoot(footBadge());
      els.input.focus();
    } catch (e) {
      console.error("[ask-brain] failed to load engine", e);
      setStatus("Couldn’t load the assistant. The docs search box still works.");
    } finally {
      state.loading = false;
    }
  }

  function footBadge() {
    var mode = state.embed ? "semantic + keyword" : "keyword mode";
    return (
      "Ranked by Brain, compiled to WASM — runs entirely in your browser, " +
      "no server saw this question. Answers are pulled verbatim from these docs. " +
      "<br><em>" +
      mode +
      "</em>"
    );
  }

  // ─── Query ─────────────────────────────────────────────────────────────────

  async function onSubmit(ev) {
    ev.preventDefault();
    var q = els.input.value.trim();
    if (!q) return;
    if (!state.engine) {
      await ensureEngine();
      if (!state.engine) return;
    }

    setStatus("Searching…");
    els.results.innerHTML = "";

    var vec = new Float32Array(0);
    if (state.embed) {
      try {
        var out = await state.embed(q, { pooling: "mean", normalize: true });
        vec = out.data instanceof Float32Array ? out.data : Float32Array.from(out.data);
      } catch (e) {
        console.warn("[ask-brain] embed failed, keyword fallback", e);
      }
    }

    var hits;
    try {
      hits = state.engine.search(q, vec, TOP_K);
    } catch (e) {
      console.error("[ask-brain] search failed", e);
      setStatus("Search failed — try rephrasing.");
      return;
    }

    setStatus("");
    render(hits);
  }

  function render(hits) {
    if (!hits || hits.length === 0) {
      els.results.innerHTML =
        '<p class="brain-ask-hit-snippet">No matching section found. Try different words, or browse the sidebar.</p>';
      return;
    }
    var frag = document.createDocumentFragment();
    hits.forEach(function (h) {
      var a = document.createElement("a");
      a.className = "brain-ask-hit";
      a.href = BASE + h.url;
      var title = document.createElement("div");
      title.className = "brain-ask-hit-title";
      title.textContent = h.title;
      var snip = document.createElement("p");
      snip.className = "brain-ask-hit-snippet";
      snip.textContent = h.snippet;
      a.appendChild(title);
      a.appendChild(snip);
      frag.appendChild(a);
    });
    els.results.innerHTML = "";
    els.results.appendChild(frag);
  }

  // ─── Boot ──────────────────────────────────────────────────────────────────

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
