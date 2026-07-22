# P2 — validation integration

goal_ref: ../goal.md
status: done

## Outcome

Accepted per-paper summaries obey the canonical inline-math contract, malformed responses use the existing validation retries/fallback, and canonical values remain identical through report assembly and index persistence.

## Assumptions

- `parseDailyPaperSummary()` is the single retry-capable acceptance boundary for LLM semantic fields.
- Deterministic and Rescue assembly already preserve accepted field bytes through the shared one-line renderer.
- The existing report parser and `PaperIndex.setSummaries()` path can demonstrate persistence without a new normalization layer.

## Approach

Invoke the P1 scanner after existing strict JSON/string checks, translate issues into a field-specific `DailyPaperSummaryValidationError`, and store only successful canonical values. Strengthen bilingual prompts and correction guidance, then trace representative values through retry, assembly, parser projection, and pipeline index tests.

## Tasks

- [x] Integrate canonicalization into every semantic field in `parseDailyPaperSummary()` with safe, stable diagnostics.
- [x] Update bilingual system prompts and correction guidance with the Obsidian `$...$` contract.
- [x] Add per-paper tests for conversion, typed retries, third-attempt recovery, and three-attempt fallback.
- [x] Add focused assembly/Rescue/parser/pipeline coverage proving canonical report and PaperIndex values agree.
- [x] Confirm transport, permanent provider/configuration, cancellation, and fallback-abstract behavior are unchanged.
- [x] Run focused core tests, core typecheck, and `git diff --check`, then checkpoint P2.

## Verification

- `npm test -w @arxiv-daily/core -- --run packages/core/tests/scientific-markdown-math.test.ts packages/core/tests/daily-paper-summary.test.ts packages/core/tests/daily-summary-assembler.test.ts packages/core/tests/daily-summary-rescue.test.ts packages/core/tests/summarizer.test.ts packages/core/tests/pipeline.test.ts`
- `npm run typecheck -w @arxiv-daily/core`
- `git diff --check`
- Invalid math retries only through `DailyPaperSummaryValidationError`; exhausted validation affects only that paper.
- Canonical `$...$` is present in generated Markdown and parsed/indexed summaries without later rewriting.

## Abort / reshape triggers

- If accepted canonical values change in the renderer/parser path, stop and resolve the existing contract rather than adding a second canonicalizer.
- If math failures are classified as transport or assembly failures, stop and restore the typed per-paper validation boundary.
