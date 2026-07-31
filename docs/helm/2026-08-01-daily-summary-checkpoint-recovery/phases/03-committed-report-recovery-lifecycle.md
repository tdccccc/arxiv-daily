# P3 — committed report recovery lifecycle

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

A complete daily report immediately becomes the authoritative recovery boundary, stale checkpoints are removed best-effort without masking report/index outcomes, and both production hosts instantiate the same core checkpoint store.

## Assumptions

- `MarkdownWriter.writeDaily()` returning successfully means the complete daily report is durably committed through the existing atomic writer.
- Once that commit exists, the existing daily-report repair path can reconstruct Paper Index projections without any checkpoint entry.
- Checkpoint cleanup is bookkeeping after commit: failure is observable but cannot make a committed report uncommitted.
- Plugin and CLI already expose the same `StorageAdapter`, current output settings, and logger required to construct the P1 store.
- Data export/import compatibility and custom-output portability can be verified and corrected in P4 without weakening this lifecycle.

## Approach

Give the pipeline a lifecycle checkpoint port that extends the summarizer's lookup/upsert needs with date removal. Immediately after a successful `writeDaily`, best-effort remove the date's primary, backup, and temporary checkpoint artifacts before observing cancellation or mutating the Paper Index. When a daily report already exists at entry, perform the same best-effort cleanup before the existing idempotent index repair. Cleanup errors log warnings only; report authority and the original completed, cancelled, or index-failure result remain unchanged.

Construct `DailySummaryCheckpointStore` in both plugin and CLI composition roots from each host's existing storage and current output settings. Hosts inject dependencies only; all reuse and cleanup policy remains in core.

## Tasks

- [x] Define a pipeline lifecycle port with `removeAll` while keeping the summarizer dependent only on lookup/upsert.
- [x] Best-effort clean checkpoint artifacts immediately after a fresh daily-report commit and before subsequent cancellation/index work.
- [x] Best-effort clean stale checkpoints whenever an existing daily report is detected, including no-index and failed-repair paths.
- [x] Preserve cancellation and index-repair semantics across write failure, post-commit cancellation, cleanup failure, and each index mutation failure; add cancellation checks between derived index mutations where safe.
- [x] Instantiate and inject the core checkpoint store in plugin and CLI composition roots using current output settings and warning redaction/logging.
- [x] Add focused lifecycle, retry, cleanup-artifact, and host-composition tests.
- [x] Run core/plugin/CLI focused tests and typechecks plus `git diff --check`.

## Verification

- A failed `writeDaily` never invokes cleanup; a successful `writeDaily` invokes cleanup before any Paper Index mutation or post-write cancellation result.
- Cleanup failure logs a warning and never replaces an otherwise completed, cancelled, or failed-transient result.
- A fresh post-commit index failure leaves the daily report authoritative; rerun skips summarization and repairs from the report.
- Existing-daily entry cleans stale checkpoint state before repair even when no Paper Index exists or repair later fails.
- Cancellation between index mutations stops later mutations; rerun repairs all derived index state from the committed report.
- Plugin and CLI create `DailySummaryCheckpointStore` with their own shared storage and current output settings, with no host-specific recovery policy.
- Focused tests, relevant package typechecks, and `git diff --check` pass.

## Abort / reshape triggers

- If `writeDaily` success cannot be treated as durable commit on either host, stop and repair writer atomicity rather than delaying checkpoint authority ambiguously.
- If cleanup must inspect report content or result quality, stop and keep it lifecycle-only; the existence of the committed report is the authority marker.
- If host composition requires duplicating compatibility or reuse logic, move that logic back into core before proceeding.
- If adding cancellation checks would make a committed report less recoverable, preserve report-first repair semantics and reshape the mutation sequence.
