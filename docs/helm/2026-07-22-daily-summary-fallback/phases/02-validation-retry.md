# P2 — validation retry

goal_ref: ../goal.md
status: done

## Outcome

Each paper receives at most three total logical summary calls; validation exhaustion and exhausted transient transport become typed fallback slots, while cancellation and permanent provider errors still abort the day.

## Assumptions

- `DailyPaperSummaryValidationError` can identify every strict parse/schema/id failure without message matching.
- LlmClient already owns transport retries; application code must not add another transport retry layer.

## Approach

Keep one-attempt `summarizeDailyPaper`, wrap it with a validation-only retry path (max total 3 logical calls), and convert exhausted validation or exhausted transient transport into structured fallback results that the deterministic assembler can render.

## Tasks

- [x] Export `DailyPaperSummaryValidationError` and throw it for every strict parse/schema/id failure.
- [x] Add a validation-only retry path with maximum total 3 logical LLM calls and corrective prompts on attempts 2/3.
- [x] Expand fallback results with stable `reasonCode`, `attempts`, and trusted `originalAbstract`; resolve abstract without ambiguous duplication.
- [x] Continue sequentially on validation/transport exhaustion fallbacks; propagate cancellation and permanent LLM errors.
- [x] Advance progress once per paper outcome and forward metrics on every real logical LLM call; log ID/reason/attempts only.
- [x] Adjust focused unit/pipeline tests for first/third success, three-failure fallback, transport/auth/cancel behavior, ordering, progress, and metrics.
- [x] Run focused tests, packages/core typecheck, and `git diff --check`.

## Verification

- Focused daily-paper-summary, summarizer, assembler, and relevant pipeline error-handling tests pass.
- A paper with three invalid responses ends as fallback with `attempts: 3`; a transport exhaustion ends as fallback without extra app retries.
- 401/403 and cancellation still abort the day before writing.
- `npm run typecheck --workspace @arxiv-daily/core`
- `git diff --check`

## Abort / reshape triggers

- If permanent vs transport classification cannot reuse existing helpers, stop and reshape rather than inventing message-based heuristics.
- If fallback abstract resolution requires untrusted LLM text, stop and keep trusted pipeline sources only.
