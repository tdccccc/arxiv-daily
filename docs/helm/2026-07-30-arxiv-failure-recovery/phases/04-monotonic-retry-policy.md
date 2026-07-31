# P4 — monotonic retry policy

goal_ref: ../goal.md
status: done

## Outcome

Scheduler results match persisted exhaustion state, and request spacing plus server cooldowns use monotonic time with safe fast deferral for long Retry-After windows.

## Assumptions

- A run should fail transiently and return control rather than remain open for an hour-scale Retry-After.
- A full server minimum can be retained in process-local monotonic state without a matching long timer.
- Persisted state returned by StateStore is authoritative for scheduler-facing results.

## Approach

Normalize scheduler results from persisted status, improve cancellation/permanent messaging, extract an injectable monotonic coordinator, and represent long server cooldowns as typed transient deferrals rather than truncated waits.

## Tasks

- [x] Normalize retry-exhausted scheduler return/history/log/progress state to persisted permanent status.
- [x] Add clear cancellation and permanent failure formatting in plugin-facing paths.
- [x] Extract or inject a monotonic coordinator clock for spacing and cooldown accounting.
- [x] Parse full Retry-After minimum without unsafe overflow or wall-clock drift.
- [x] Wait short cooldowns cancellably in chunks and fail fast with typed transient deferral for long cooldowns.
- [x] Classify and format deferred failures consistently at source/pipeline/UI boundaries.
- [x] Add adversarial scheduler, clock, header, cooldown-extension, deferral, and cancellation tests.
- [x] Run focused suites, affected typechecks, and boundary checks.

## Verification

- Attempt 9 remains transient; attempt 10 returns, persists, logs, and records permanent exhaustion consistently.
- Wall-clock rollback or leap cannot lengthen or bypass three-second spacing.
- HTTP-date conversion is fixed at receipt and later wall-clock changes do not alter cooldown.
- A two-hour Retry-After causes no second HTTP call or two-hour timer; calls fail fast until the full monotonic deadline passes.
- Later longer cooldowns extend already sleeping/queued work and never shorten an active deadline.

## Abort / reshape triggers

- If the scheduler cannot preserve a useful exhaustion reason within existing result types, add a typed reason field only after checking persistence compatibility.
- If coordinator extraction destabilizes P1 queue semantics, retain the current queue shape and inject clocks/state incrementally rather than rewriting it wholesale.
