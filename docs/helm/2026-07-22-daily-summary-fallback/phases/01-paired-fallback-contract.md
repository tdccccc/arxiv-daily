# P1 — paired fallback contract

goal_ref: ../goal.md
status: done

## Outcome

Daily assembly accepts validated paper/result pairs, renders structured and safe fallback blocks deterministically, and lets parsers distinguish fallback IDs from generated summaries.

## Assumptions

- Existing trusted paper metadata plus the original abstract is sufficient for a useful fallback block.
- A stable HTML-comment marker can identify fallback blocks without changing normal summary parsing.

## Approach

Replace parallel paper/summary arrays with discriminated paired slots, validate all trusted assembly inputs through an independently callable preflight, and keep fallback rendering and parser behavior explicit and deterministic.

## Tasks

- [x] Define paired `DailyPaperSlot` structured/fallback result types and adapt P1 call sites.
- [x] Add independently callable preflight validation for paper/topic identity, category membership, and required trusted metadata.
- [x] Render localized typed fallback blocks with direct-arXiv guidance, trusted abstract/unavailable text, markers, and accurate counts/order.
- [x] Sanitize fallback abstract text against heading, list, HTML comment, and marker injection.
- [x] Make summary extraction skip fallback blocks and expose fallback ID extraction.
- [x] Adapt focused assembler/parser tests for valid behavior, validation, fallback safety, and parser behavior.
- [x] Run focused tests, packages/core typecheck, and `git diff --check`.

## Verification

- `npm test --workspace @arxiv-daily/core -- --run tests/daily-summary-assembler.test.ts tests/daily-summary-parser.test.ts tests/summarizer.test.ts`
- `npm run typecheck --workspace @arxiv-daily/core`
- `git diff --check`
- Structured-only output remains compatible; mixed output has stable total/fallback counts and topic/input order.

## Abort / reshape triggers

- If trusted original abstracts are unavailable at the assembly boundary, stop and reshape rather than sourcing fallback content from LLM output.
- If fallback identification requires parsing localized prose, stop and introduce a stable typed marker before proceeding.
