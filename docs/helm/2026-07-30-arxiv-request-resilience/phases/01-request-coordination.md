# P1 — request coordination

goal_ref: ../goal.md
status: done

## Outcome

Every initial and retried arXiv HTTP attempt is globally serialized, starts at least three seconds after the preceding attempt, and honors a shared, non-shortened Retry-After cooldown.

## Assumptions

- Process-local coordination is sufficient for the plugin and CLI runtime model.
- Serializing the actual HTTP operation is acceptable despite reducing arXiv network concurrency.
- Existing HTTP adapters already preserve response status and headers.

## Approach

Move the arXiv gate inside each retry attempt, model Retry-After as an absolute shared deadline, introduce a typed HTTP error and status-aware predicate, and clamp unsafe request-delay configuration at both runtime and configuration boundaries.

## Tasks

- [x] Replace the initial-request delay with a shared serialized attempt coordinator and a three-second floor.
- [x] Record 429/503 Retry-After as a monotonic absolute cooldown and keep waits cancellable.
- [x] Add typed arXiv HTTP errors, actionable formatting, and status-aware retry classification.
- [x] Normalize or reject unsafe request-delay configuration while preserving runtime compatibility.
- [x] Add focused tests for serialization, spacing, cooldown, status policy, cancellation, and configuration.
- [x] Run focused tests, typecheck, and boundary checks; record any reshape decision.

## Verification

- `npm run test -w @arxiv-daily/core -- --run packages/core/tests/arxiv-fetcher.test.ts`
- Focused CLI configuration tests.
- `npm run typecheck -w @arxiv-daily/core`
- `npm run check:boundaries`
- Recorded HTTP attempt timestamps are globally ordered, non-overlapping, and at least 3000 ms apart.

## Abort / reshape triggers

- If fake-timer behavior makes module-global coordination untestable without production-only hooks, introduce an explicit coordinator dependency rather than brittle module resets.
- If enforcing the floor at validation time blocks valid legacy settings, retain runtime clamping and migrate settings instead of failing whole runs.
