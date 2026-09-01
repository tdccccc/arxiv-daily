# ADR 0013: The library index covers titles and abstracts, not full text

Status: Accepted (2026-09-01 design session)

Related: ADR 0005 (scoped desktop library access); ADR 0008 (opt-in remote embedding), whose full-text disclosure this narrows; ADR 0006 (unified search entry).

## Context

The library index chunks each PDF's full text, embeds every chunk, and indexes the same chunks lexically for BM25. Measured on the test library: **207 papers produced 23,423 chunks — about 113 per paper — and a 140 MB index at 768 dimensions.** A thousand-paper library extrapolates to roughly 113,000 chunks, 680 MB, and 113,000 embedding calls. Local embedding at that volume effectively requires a GPU, and any change that invalidates the index means paying it again.

What the full text buys is narrower than its cost suggests:

- **Search shows no passages.** A result is a paper title and a score; no excerpt reaches the researcher. Chunk granularity affects ranking only.
- **Clustering uses max-chunk cosine.** Two papers are judged related if any chunk of one resembles any chunk of the other, so full text does sharpen direction discovery — a method mentioned only in section 4 can pull two papers together.
- **Lexical search covers the same chunks**, so a term appearing only in the methods section is findable today.

The trade is therefore real but bounded: full text buys ranking granularity and some clustering reach, at roughly a hundred times the index cost, for a capability whose evidence the product never displays.

## Decision

### 1. Index titles and abstracts only

One or two chunks per paper instead of a hundred. A thousand-paper library becomes a few megabytes and a few thousand embedding calls: rebuildable in minutes, on a CPU.

### 2. No conclusion section for now

Extending the indexed text to a paper's conclusion was considered and deferred. It is the obvious next increment if abstract-level retrieval proves too coarse, and deferring costs one more rebuild — cheap at this size, which is the point of the decision above.

### 3. Existing indexes are rebuilt, not migrated

The personal library subsystem has never shipped in a release tag, so no one outside this repository holds an index. Rebuilding is free exactly once, and this is that once.

## Consequences

- Retrieval, both dense and lexical, becomes abstract-level. **A paper whose only mention of a technique is in its methods or experiments will no longer be found by searching for that technique.** This is the real cost, and it is larger than a ranking regression.
- Direction clustering loses its below-the-abstract reach. Whether the directions it proposes get noticeably coarser is a measurement to make, not an assumption to carry.
- Index size stops being a reason to avoid rebuilding, which makes changes to chunking, the embedding model, or the derivation contract ordinary rather than expensive.
- **The full-text processing depth may no longer have a subject.** ADR 0008 splits consent by depth precisely because whole papers left the machine; if only titles and abstracts are ever embedded, remote embedding sends what metadata-and-abstract consent already covers. Whether the depth distinction collapses is left to a decision that revisits ADR 0008 directly rather than being settled as a side effect here.
- Passage evidence, already unshown pending a quality bar, becomes unavailable rather than merely unshown. Showing it again would require reinstating full-text chunks.
