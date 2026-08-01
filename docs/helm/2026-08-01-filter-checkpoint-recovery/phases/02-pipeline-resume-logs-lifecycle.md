# P2 — pipeline resume, logs, and lifecycle

goal_ref: ../goal.md
updated: 2026-08-01

## Outcome

An interrupted pipeline reuses a compatible validated filter batch before downstream work, operators can see filter and per-paper summary checkpoint decisions, and both checkpoint kinds obey one committed-report cleanup lifecycle.

## Assumptions

- Filter checkpoint lookup belongs inside the filter orchestration where exact request identity, current paper metadata, cancellation, metrics, and the LLM call meet.
- Recovered filter decisions contribute no historical LLM metrics but preserve current progress and downstream behavior.
- A strictly valid empty filter batch may remain reusable because no daily report is committed on the zero-result path.
- Cleanup must attempt every checkpoint store even when one removal fails.

## Approach

Extend the paper-filter dependency seam with report date, LLM settings, and a minimal filter checkpoint port. Perform lookup before the LLM call, rebuild hits from current paper metadata, and await persistence of validated misses before returning to the pipeline. Add stable info logs for filter and summary hit/miss/persisted events at cancellation-safe durability boundaries. Replace the summary-specific lifecycle dependency with a Core date-scoped aggregate that independently cleans both stores after a fresh daily commit or when an existing daily report is authoritative.

## Tasks

- [x] Add filter checkpoint lookup/save orchestration with current-metadata reconstruction and no historical metrics.
- [x] Preserve malformed response, transport error, cancellation, zero-result, order, progress, and downstream failure semantics.
- [x] Add filter `checkpoint hit/miss/persisted` info logs only when a store exists.
- [x] Add per-paper summary `checkpoint hit/miss/persisted` info logs only when a store exists.
- [x] Generalize report-date checkpoint cleanup so every store is attempted and committed-report authority remains unchanged.
- [x] Add focused filter, summarizer, and pipeline lifecycle/cancellation/failure tests.
- [x] Run focused Core typecheck/tests and `git diff --check`, reconcile review findings, and commit P2.

## Verification

- On a compatible rerun, filter lookup returns before `llm.call`; recovered records rebuild output from current `PaperMeta` objects in persisted record order.
- On a miss, a validated result is durably saved before Paper Index or content work; save failure stops the pipeline.
- Hit results produce no current-run generation metrics, while progress and final output remain unchanged.
- Logs never report a store-less miss or announce a hit/persisted result before its cancellation boundary.
- Fresh and existing daily reports trigger both cleanup stores; one cleanup error does not block the other or override authority.

## Recorded evidence

- Core typecheck passed.
- Seven focused suites passed with 245 tests covering filter, summarizer, pipeline lifecycle/error handling, both checkpoint stores, and cancellation.
- `git diff --check` passed.
- Independent lifecycle/correctness review reported no findings.

## Abort / reshape triggers

- If filter checkpoint persistence errors cannot be distinguished from LLM errors without changing public run semantics, introduce a typed internal error rather than misleading operator output.
- If a cleanup aggregate obscures which store failed, preserve per-store labels in warnings while retaining best-effort all-store execution.
- If recovered records cannot be rebuilt exclusively from current paper metadata, stop rather than persist presentation metadata.
