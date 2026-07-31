# P1 — transport deadlines

goal_ref: ../goal.md
status: done

## Outcome

Every arXiv attempt has a hard logical deadline that releases the shared queue, and immediate retries are limited to typed network/timeout failures plus 408, 429, and 5xx statuses.

## Assumptions

- Logical timeout is preferable to permanent queue blockage when Obsidian cannot physically cancel requestUrl.
- Separate text and binary deadlines avoid both slow metadata recovery and premature large-download failure.
- Shared transport error typing can be added compatibly to the host adapter contract.

## Approach

Define host-neutral transport errors, normalize Node and Obsidian adapter behavior, add fetcher deadlines and a defensive watchdog, then tighten retry classification and verify hung/cancelled/late-settling interleavings.

## Tasks

- [x] Add typed network/timeout transport errors and contract tests.
- [x] Normalize Node deadlines, cancellation, network errors, and late settlement.
- [x] Add Obsidian logical timeout/cancellation with injectable request implementation.
- [x] Pass bounded text/binary deadlines from ArxivFetcher and enforce a defensive watchdog.
- [x] Retry only typed transport errors and selected HTTP statuses.
- [x] Add focused queue-recovery, adapter, cancellation, and classification tests.
- [x] Run focused tests, affected typechecks, and boundary checks.

## Verification

- A never-settling client times out and the next queued arXiv request starts.
- Caller cancellation starts no retry and later queue work recovers.
- Plain/local errors and missing binary bodies make one attempt; typed network/timeout failures retry.
- Node and Obsidian adapters settle logically by deadline and consume late rejections.
- Focused core/node/plugin suites and boundary checks pass.

## Abort / reshape triggers

- If logical timeout cannot prevent unhandled late request failures, isolate adapter races before proceeding.
- If releasing the Obsidian slot causes unacceptable overlap semantics, document and reshape the coordinator policy rather than restoring indefinite blockage.
