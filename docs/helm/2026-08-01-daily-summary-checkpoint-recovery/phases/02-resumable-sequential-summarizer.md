# P2 — resumable sequential summarizer

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

Sequential daily summarization reuses each compatible structured-summary checkpoint and durably checkpoints every newly completed result before starting another paper, while preserving existing assembly and failure semantics.

## Assumptions

- Pipeline already owns the report date, effective `LlmSettings`, output settings, and storage-backed checkpoint store inputs needed by the P1 contract.
- Recovered results should use the current `DailySummaryAssemblyPaper`, so links, detail status, category, and final input order never come from checkpoint state.
- A compatible checkpoint hit counts as completed progress but contributes no current-run LLM metrics.
- A checkpoint upsert failure is a transient summarization failure and must prevent the next paper's LLM call.
- `transport-exhausted` entries remain misses through `lookupReusable`, so recovery retries only that paper while retaining earlier reusable work.

## Approach

Extend `SummarizerDeps` with explicit report-scoped checkpoint dependencies and effective `LlmSettings`. For each paper, derive the P1 compatibility input, try `lookupReusable`, and otherwise execute the existing validation-retry path. Await `upsert` for every newly produced result before adding its slot, publishing progress, or advancing the loop. Build slots from current assembly metadata in the original order and leave deterministic assembly, rescue, fallback logging, cancellation, and final daily-report writing unchanged.

Wire the store from the core pipeline composition using its existing `StorageAdapter`/output boundary rather than teaching plugin and CLI different recovery rules. Tests must distinguish reused from generated work without adding historical checkpoint metrics to the current run.

## Tasks

- [x] Define the smallest summarizer checkpoint dependency contract and pass report date, effective LLM settings, and the shared store from pipeline without host-specific behavior.
- [x] Add compatible lookup to the sequential loop, rebuild recovered slots from current assembly metadata, and preserve input order across mixed hits and misses.
- [x] Await atomic upsert after each newly completed structured or typed-fallback result and before slot completion, progress publication, or the next LLM call.
- [x] Preserve cancellation, fallback logging/counts, deterministic assembly/rescue, and current-run metrics semantics for recovered and generated entries.
- [x] Add focused tests for all-hit, partial-hit, stale/corrupt miss, validation-fallback hit, transport-fallback retry, cancellation boundaries, checkpoint write failure, ordering, progress, metrics, and byte-identical assembly.
- [x] Run summarizer/pipeline focused tests, core typecheck, and `git diff --check`.

## Verification

- K compatible entries among N selected papers cause exactly N-K per-paper LLM workflows, in the same order as the missing papers, with maximum in-flight one.
- A new result is visible after store reconstruction before the next paper starts; injected upsert failure prevents later LLM calls and preserves the prior durable document.
- Compatible validation-exhausted fallback is reused; compatible transport-exhausted fallback is retried and replaced by the new result.
- Recovered and generated slots follow current selected-paper order and use current assembly links/detail metadata; uninterrupted and resumed runs assemble byte-identical Markdown for identical results.
- Progress reaches N/N including recovered entries, while current-run metrics contain only calls made during the resumed run.
- Cancellation before lookup, after lookup, after generation, and after durable upsert never becomes fallback or starts an unintended next call.
- Focused tests, `npm --prefix packages/core run typecheck`, and `git diff --check` pass.

## Abort / reshape triggers

- If pipeline cannot provide effective generation identity without exposing secrets to checkpoint persistence, stop and reshape the dependency boundary.
- If recovered progress cannot be represented without changing the public progress stage contract, keep the existing completed/total callback and defer richer reused/generated counters.
- If a recovered result requires persisted slot metadata to assemble correctly, stop and repair the current-metadata reconstruction rule rather than broadening the checkpoint schema.
- If checkpoint failure classification requires host-specific exception handling, define a typed core error boundary instead of branching by host.
