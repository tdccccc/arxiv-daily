# P1 — deterministic assembler

goal_ref: ../goal.md
status: done

## Outcome

Validated per-paper structured fields can be rendered into the existing Chinese or English daily Markdown contract without model-controlled counts, sections, ordering, or links.

## Assumptions

- The existing `PaperSummary` fields are sufficient for daily search/index consumers.
- Configured topic order and input paper order are the desired deterministic output order.
- Titles, authors, source sections, arXiv IDs, categories, and detail links should come from trusted pipeline input rather than model output.

## Approach

Add a small pure assembler module with a structured summary contract aligned to `PaperSummary`. It will own localized labels, topic grouping, counts, empty sections, and paper Markdown blocks, while preserving compatibility with `extractPaperSummaries`.

## Tasks

- [x] Define the structured per-paper summary contract.
- [x] Implement Chinese and English deterministic assembly.
- [x] Validate duplicate/missing/unknown summary IDs before rendering.
- [x] Add topic-order, empty-topic, detail-link, count, and parser round-trip tests.

## Verification

- Run assembler and daily-summary-parser tests.
- Assembled Markdown parses back into all five indexed summary fields for every paper.

## Abort / reshape triggers

- If existing Markdown consumers require model-authored structure outside the five fields, stop and extend the explicit contract rather than preserving free-form output.
- If input category tags do not map uniquely to configured topics, reject assembly rather than silently relocating papers.
