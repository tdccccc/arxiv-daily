# P1 — paper-workflow

goal_ref: ../goal.md
created: 2026-10-02T11:59:48+08:00
updated: 2026-10-02T11:59:48+08:00
revision: 1

## Outcome

All indexed discoveries can be selected, read and marked in the workbench, including daily entries without a detailed note.

## Assumptions

- PaperIndex is authoritative for discovered-paper identity and marks. Missing index entries remain unknown rather than fabricated by parsing Markdown on GET.
- Existing inbox/to_read/read/reading/saved/ignored values and priority are preserved; this UI offers explicit inbox/to_read/read and independent star controls.
- Original Markdown browsing remains a secondary path for full daily reports and standalone summary files.

## Approach

Add protected read endpoints for paper lists/detail and a narrow optimistic mark endpoint using PaperIndexStore.mutate. Core Dashboard query supplies search/order/filter and occurrence provenance; validate actual note paths through the document catalog. Reuse the configured CLI for on-demand detail generation. Add a unified paper surface and date lists while retaining calendar and Markdown routes.

Visual thesis: retain the wide calendar sidebar, use compact paper rows and readable summary sections with restrained state/favorite controls.
Content plan: date/library list, paper summary/detail/source actions, independent reading/favorite marks, full Markdown and summary-file escape hatches.
Interaction thesis: explicit selections, persisted-only mark updates with failure recovery, stable filters/history and mobile return navigation.

## Chunks

### Chunk 1 — Paper query and mark API

- change kind: behavior change
- strategy: strict Red-Green at real HTTP/persistence boundaries
- Red: list/detail/mark routes fail before implementation
- Green: index-only papers visible, date/search/filters correct, real note availability, marks survive a fresh store/server, favorites independent, idempotent writes preserve unrelated data, stale conflicts and malformed/foreign requests rejected
- regressions: CLI server/calendar/inspection tests and typecheck/boundaries
- [ ] implementation and tests accepted

### Chunk 2 — Unified paper reader

- change kind: behavior change
- strategy: strict Red-Green for DOM/API behavior; direct browser review for layout
- Red: unified paper rows/detail/mark controls absent
- Green: daily list→paper detail→mark/filter/reload, failures without false UI success, stale responses, original Markdown/summary-file routes, explicit detail generation and refresh all pass
- regressions: existing calendar/reader tests, CLI typecheck, portable fixture process tests
- [ ] implementation and tests accepted

## Phase verification

- Actual bundled CLI API and browser: discover at least one paper with only a short summary and one with a detailed note; mark to-read/read/starred independently and verify through a fresh product query.
- Calendar date opens its indexed papers; empty/missing-index dates explain the difference and retain full-report access.
- Reading does not call a model; explicit fixture generation writes the normal note and refreshes availability. Source Markdown bytes remain unchanged by mark operations.
- Desktop/mobile screenshots, normal console checks and existing workspace boundaries pass.

## Abort / reshape triggers

- If a mark cannot be persisted through existing index operations, do not introduce a second store or edit raw JSON outside the core transaction.
- If a date lacks an index, do not mislabel that as zero selected papers or perform hidden index repair.
