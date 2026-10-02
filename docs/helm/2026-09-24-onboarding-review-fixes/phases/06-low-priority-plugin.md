# P6 — low-priority-plugin

goal_ref: ../goal.md
created: 2026-09-24T23:05:00+08:00
updated: 2026-09-24T23:40:00+08:00
revision: 2

## Outcome

The remaining low-priority plugin findings are fixed: turning on daily runs with "Run today" returns at once instead of holding the settings page for the whole run; a malformed legacy run record no longer stops the plugin from loading; structural topic/category edits roll back and report when saving fails, and topic field saves report failures instead of leaving unhandled rejections; the legacy Thinking mode toggle follows a Reasoning effort choice.

## Assumptions

- Starting the "Run today" run in the background after the enable intent is saved keeps the existing intent-queue guarantees (a later disable still wins; the stale run result is ignored).
- Dropping invalid legacy run-state entries (with a warning) is acceptable: they are unreadable by the strict store anyway.

## Approach

Local fixes in `plugin/main.ts` and `plugin/src/settings/tab.ts`, each test-first, using the existing lifecycle and legacy settings test harnesses.

## Chunks

### Chunk 1 — enable + Run today does not block (F15)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: `tests/settings-lifecycle.test.ts`: with `tickToday` pending, `setScheduleEnabled(true)` + "run" resolves true, and a following `setScheduleEnabled(false)` resolves without waiting for the run — fails today (awaits the run)
- Green check: focused vitest
- regression checks: existing lifecycle intent-queue tests; `cd plugin && npm test`
- [x] implementation and tests accepted

### Chunk 2 — malformed legacy run state does not stop loading (F17)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: test: migrating a legacy run state with one invalid entry keeps the valid entries, drops the invalid one with a warning, and does not throw — fails today (strict store rejects the write)
- Green check: focused vitest
- regression checks: `cd plugin && npm test`
- [x] implementation and tests accepted

### Chunk 3 — topic and category saves report and roll back (C1)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: with persistence failing, adding/deleting a topic or changing categories restores the previous list and reports; a failing topic field save reports instead of an unhandled rejection — fail today
- Green check: focused vitest in `tests/settings-declarative-tab.test.ts`
- regression checks: `cd plugin && npm test`
- [x] implementation and tests accepted

### Chunk 4 — legacy Thinking mode follows Reasoning effort (C4)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: legacy test: with thinking off, choosing an effort turns thinking on and the Thinking mode toggle shows on — fails today (toggle stays off)
- Green check: focused vitest in `tests/settings-tab.test.ts`
- regression checks: `cd plugin && npm test`
- [x] implementation and tests accepted

## Phase verification

- Full check set from goal.md constraints

## Abort / reshape triggers

- If running "Run today" in the background breaks the intent-queue guarantees in the existing lifecycle tests, stop and reshape chunk 1.
