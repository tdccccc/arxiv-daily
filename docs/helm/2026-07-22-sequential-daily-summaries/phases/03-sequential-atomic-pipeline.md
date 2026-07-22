# P3 — sequential atomic pipeline

goal_ref: ../goal.md
status: done

## Outcome

All visible papers are summarized in input order with exactly one structured LLM call per paper and maximum concurrency one, then assembled and written only after every paper succeeds.

## Assumptions

- The existing pipeline daily-summary failure boundary already maps uncaught LLM and validation failures correctly and writes only after summarization returns.
- A per-paper completion callback on `SummarizerDeps` is sufficient to report existing `summarize-daily` stage progress without coupling the summarizer to pipeline adapters.
- The P1 assembler and P2 per-paper summarizer contracts contain all metadata and content needed by daily generation.

## Approach

Replace free-form multi-paper generation and repair logic with a plain awaited loop over papers, collect structured summaries and trusted assembler metadata in the same input order, report each successful completion, and assemble only after all calls and cancellation checks pass. Adapt pipeline mocks and add focused orchestration and atomicity coverage.

## Tasks

- [x] Replace daily batching/free-form orchestration with sequential structured per-paper calls and deterministic assembly.
- [x] Preserve trusted metadata, cancellation boundaries, metrics observation, and `dailyCharLimit` compatibility.
- [x] Wire successful per-paper completion to `summarize-daily` progress counts.
- [x] Remove obsolete daily free-form helpers/imports while preserving detail summarization.
- [x] Update summarizer and pipeline mocks/tests for ordering, concurrency, parsing, progress, failures, cancellation, and metrics.
- [x] Run focused core tests, core typecheck, and `git diff --check`.

## Verification

- Focused tests cover summarizer, assembler, daily paper summary, parser, pipeline, and pipeline error handling.
- N papers produce N valid per-paper calls in exact input order with maximum in-flight one and complete parseable output.
- A failed or cancelled later paper never reaches assembly/write, while successful calls aggregate metrics and emit progress through N/N.
- `npm --prefix packages/core run typecheck` and `git diff --check` pass from this worktree.

## Abort / reshape triggers

- If per-paper failures are swallowed or the pipeline writes any partial daily report, stop and restore the atomic boundary before proceeding.
- If progress wiring requires a new stage or changes metrics collection semantics, reshape to a smaller observer callback rather than broadening adapter interfaces.
