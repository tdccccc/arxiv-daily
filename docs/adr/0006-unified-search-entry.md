# ADR 0006: Unified search entry — one box, two corpora, plus a dual-tab similar-papers modal

Status: Accepted (2026-08-07 Helm P4 follow-up design)

Related: ADR 0001 (one TypeScript core, two hosts); ADR 0004 (personal-library-guided discovery); ADR 0005 (scoped desktop library access).

## Context

The plugin now has two retrieval mechanisms with different corpora and engines:

1. **Lexical relevance over the paper index** (BM25F-style) — instant, offline, no model. Used by the Dashboard filter box (filters daily-report rows) and by the per-row "Find similar papers" button (similar papers among daily-report papers).
2. **Embedding full-text search over the personal-library knowledge base** (multilingual-e5-small, cosine over chunk vectors) — async, requires the local model and an indexed library; currently reachable only through the command palette (`search-personal-library-fulltext`), with results shown as a Notice.

The open design question was how full-text retrieval should get a first-class entry. The Dashboard already has one search input (the filter box), so adding a second resident search bar for the library was visually redundant; a mode switch on the existing box (filter vs library search) creates two semantics on one input and mis-trigger risk; command-palette-only keeps the feature hidden.

The two corpora are not the same set of papers: daily-report papers live in the paper index, library papers live in the catalog + full-text KB. A paper may appear in both (same arXiv ID), but mostly they differ. Physical directory topology (the plugin output dir and the user-selected library may be siblings or nested) does not change this.

## Decision

### 1. One search box, dual results

The Dashboard search box remains the single text input. On a query it:

- (a) filters daily-report rows lexically — existing behavior, unchanged and instant;
- (b) when a knowledge base exists, asynchronously embeds the query and renders library matches as a separate results block (similarity score + best-passage evidence + open actions), following the existing SimilarPapersModal conventions.

No mode switch. If the KB is absent or not authorized, only (a) is shown.

### 2. Per-row button opens a dual-tab modal

The existing row-level "Find similar papers" button opens one modal with two tabs:

- **库内相似 (library)** — full-text embedding retrieval over the KB, primary;
- **日报相似 (daily)** — the existing lexical results over daily-report papers.

When no KB exists, only the daily tab shows. One button, one modal, two clearly labeled result sections.

### 3. From-paper query source: title + abstract

The from-paper query is built from the row paper's title + abstract (title-only fallback when the abstract is empty). Both daily-report index records and library catalog records carry an abstract (catalog `abstract` is a required field; it may be an empty string when arXiv identification failed). The PDF is never the query source — PDFs only exist on the retrieval-target side (the KB corpus).

## Consequences

- One input now spans two engines; the UI must label the two result sections clearly so users understand the difference.
- The KB result block is async relative to row filtering (embedding inference; the model is cached in the renderer, so latency is on the order of hundreds of ms).
- The library-similarity path reuses the KB search mechanics already verified in the Obsidian runtime (P4).
- The lexical daily-papers similarity capability is preserved behind the second tab rather than removed.
