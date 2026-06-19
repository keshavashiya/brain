// Build-time indexer for the Brain OS docs assistant.
//
// Walks docs/src/**.md, splits each page into heading-anchored sections, embeds
// every section with the SAME sentence model the browser uses at query time
// (so cosine similarity is meaningful), and writes a single static index file.
//
//   node index.mjs
//
// Output: docs/src/assistant/brain-index.json
//   { model, dim, generated, chunks: [{ title, url, text, vector }] }
//
// The model id is written into the file; the widget reads it back and loads the
// matching model. Change the model in ONE place — MODEL below — and both sides
// stay in lockstep.

import { pipeline, env } from "@huggingface/transformers";
import { readdir, readFile, writeFile, mkdir } from "node:fs/promises";
import { join, relative, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

// Pin the model cache to a fixed dir so CI can cache it across runs (avoids
// re-downloading the model every build). Gitignored.
env.cacheDir = join(HERE, ".cache");
const SRC = join(HERE, "..", "src");
const OUT_DIR = join(SRC, "assistant");
const OUT = join(OUT_DIR, "brain-index.json");

// Must match the model the widget loads in the browser.
const MODEL = "Xenova/all-MiniLM-L6-v2";

// Sections shorter than this (after cleaning) are folded into the previous one
// so we don't index near-empty stubs.
const MIN_CHARS = 80;

// ─── Markdown helpers ────────────────────────────────────────────────────────

// Replicate mdBook's `normalize_id` exactly so anchors match the generated
// pages. Per character: keep alphanumerics / `_` / `-` (ASCII-lower-cased),
// map EACH whitespace char to a single `-`, drop everything else. Note it does
// NOT collapse consecutive separators, so "Relays & Transports" → the space
// before and after the dropped "&" both become hyphens → "relays--transports".
function slugify(text) {
  let out = "";
  for (const ch of text) {
    if (/[\p{L}\p{N}]/u.test(ch) || ch === "_" || ch === "-") {
      out += ch >= "A" && ch <= "Z" ? ch.toLowerCase() : ch;
    } else if (/\s/u.test(ch)) {
      out += "-";
    }
  }
  return out;
}

// Strip markdown down to readable prose for embedding + display.
function clean(md) {
  return md
    .replace(/```[\s\S]*?```/g, " ")        // fenced code blocks
    .replace(/`([^`]+)`/g, "$1")            // inline code
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")  // images
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1") // links → link text
    .replace(/^[>\s]*\|.*$/gm, " ")         // table rows
    .replace(/[*_#>]/g, " ")                // residual md punctuation
    .replace(/<!--[\s\S]*?-->/g, " ")       // html comments
    .replace(/\s+/g, " ")
    .trim();
}

// Split one markdown file into [{ heading, level, anchor, body }].
function sections(md) {
  const lines = md.split("\n");
  const out = [];
  let cur = null;
  let inFence = false;
  for (const line of lines) {
    if (/^```/.test(line.trim())) inFence = !inFence;
    const m = !inFence && line.match(/^(#{1,4})\s+(.*)$/);
    if (m) {
      if (cur) out.push(cur);
      const heading = m[2].trim();
      cur = { heading, level: m[1].length, anchor: slugify(heading), body: "" };
    } else if (cur) {
      cur.body += line + "\n";
    } else {
      // Preamble before the first heading — attach to a synthetic intro.
      cur = { heading: "", level: 1, anchor: "", body: line + "\n" };
    }
  }
  if (cur) out.push(cur);
  return out;
}

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(p);
    else if (entry.name.endsWith(".md")) yield p;
  }
}

// ─── Build ───────────────────────────────────────────────────────────────────

const chunks = [];

for await (const file of walk(SRC)) {
  const rel = relative(SRC, file).replace(/\\/g, "/");
  if (rel.startsWith("assistant/")) continue; // never index our own assets
  if (rel === "SUMMARY.md") continue; // mdBook's nav file, not a rendered page
  if (rel === "intro.md") continue; // landing page — broad marketing prose that
  // acts as an attractor for almost any "how do I…" query; exclude so concrete
  // how-to sections rank instead.
  const md = await readFile(file, "utf8");
  const url = rel.replace(/\.md$/, ".html").replace(/(^|\/)README\.html$/, "$1index.html");

  // Page title = first H1, else file path.
  const h1 = md.match(/^#\s+(.*)$/m);
  const pageTitle = h1 ? h1[1].trim() : rel.replace(/\.md$/, "");

  let pending = null; // for folding short sections forward
  for (const s of sections(md)) {
    const bodyText = clean(s.body);
    const headText = clean(s.heading);
    const isPageHeading = s.level === 1 && headText === clean(pageTitle);
    const title = !headText || isPageHeading ? pageTitle : `${pageTitle} · ${headText}`;
    const anchor = s.anchor && !isPageHeading ? `#${s.anchor}` : "";
    const text = [headText, bodyText].filter(Boolean).join(". ");

    const chunk = { title, url: url + anchor, text };
    if (pending) {
      // Merge a too-short previous section into this one.
      chunk.title = pending.title;
      chunk.url = pending.url;
      chunk.text = `${pending.text} ${text}`.trim();
      pending = null;
    }
    if (chunk.text.length < MIN_CHARS) {
      pending = chunk; // hold and fold into the next section
      continue;
    }
    chunks.push(chunk);
  }
  if (pending) chunks.push(pending); // trailing short section, keep it anyway
}

console.log(`Indexing ${chunks.length} sections from docs/src …`);

const embed = await pipeline("feature-extraction", MODEL);
let dim = 0;
for (const c of chunks) {
  const t = await embed(c.text, { pooling: "mean", normalize: true });
  c.vector = Array.from(t.data);
  dim = c.vector.length;
}

await mkdir(OUT_DIR, { recursive: true });
await writeFile(
  OUT,
  JSON.stringify({ model: MODEL, dim, generated: new Date().toISOString(), chunks }),
);

const kb = Math.round((await readFile(OUT)).length / 1024);
console.log(`✓ ${chunks.length} chunks · ${dim}-dim · ${kb} KB → ${relative(join(HERE, ".."), OUT)}`);
