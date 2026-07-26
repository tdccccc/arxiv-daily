# P2 — source-adapter

<!-- Filename 02-source-adapter.md ↔ P2 -->
goal_ref: ../goal.md
status: done

## Outcome

A host-neutral `SourceAdapter` port and normalized content DTO exist; the daily pipeline discovers papers and fetches full text through an `ArxivSourceAdapter` that wraps today's arXiv fetch/parse/extract path, without user-visible Dashboard/path regressions.

## Assumptions

- P1 paperKey/schema 4 is already on this branch.
- Existing `pipeline/paper-content.ts` `PaperContent` (abstractConclusion/fullSections) remains the arXiv extractor DTO; the adapter contract uses a separate normalized shape to avoid a repo-wide rename in P2.
- Summarizer/filter still consume id/title/authors/abstract-shaped objects; adapter output is mapped at the pipeline boundary.
- Fake second source is **P3**, not this phase.

## Approach

1. Add `packages/core/src/sources/` with types + `SourceAdapter` + `ArxivSourceAdapter`.
2. Normalized content: abstract, optional sections, fullTextFallback, quality, canonicalUrl (+ optional provenance).
3. Move date listing (multi-category /recent + parse + optional abstract enrichment) behind `listForDate`.
4. Move body fetch behind `fetchContent`, delegating to existing `PaperContentFetcher`.
5. Wire `ArxivPipeline` to depend on `SourceAdapter` (keep constructing arXiv adapter from existing fetcher/paperFetcher for host compatibility).
6. Focused unit tests for adapter mapping; full core suite stays green.

## Tasks

- [x] Define source types and `SourceAdapter` interface.
- [x] Implement `ArxivSourceAdapter` (listForDate + fetchContent).
- [x] Re-home pipeline discovery and content fetch onto the adapter.
- [x] Export from `@arxiv-daily/core` root.
- [x] Adapter + regression tests; typecheck + core tests green.
- [x] Update goal/journal when done.

## Verification

- Core tests pass (including pipeline).
- Core typecheck + boundaries OK.
- arXiv-only behavior: same daily keys short paths, filter/summarize inputs still have id/title/abstract.

## Abort / reshape triggers

- If rewiring forces rewriting summarizer contracts end-to-end → L2: adapter only for discovery first, content map stays internal.
- If multi-category failure semantics cannot be preserved → stop and reshape list result type.
