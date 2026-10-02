# P3 — first-report-date

goal_ref: ../goal.md
created: 2026-09-24T21:45:00+08:00
updated: 2026-09-24T22:05:00+08:00
revision: 2

## Outcome

"Generate first report" runs the latest day arXiv has already announced (not later than today), so it works on weekends and before today's announcement; runs started from the guide or the "Run today" command refresh any open dashboard.

## Assumptions

- `plugin.recentDates.refresh()` returns the announced dates from `/recent`; `runForDateNow` accepts any of those dates (the dashboard calendar already runs them).
- "Run today" (command and dashboard) keeps meaning today; only its dashboard refresh changes.

## Approach

Pick the first-report date from the recent-dates snapshot (latest date ≤ today in the configured timezone), falling back to today when the refresh fails or returns nothing, and say which day is being generated. After the run, refresh open dashboards.

## Chunks

### Chunk 1 — first report uses the latest announced date (F3)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests with the clock on a Saturday: recent dates {Thu, Fri} → `runForDateNow` gets Friday; recent refresh rejects → today is used; dates after today are ignored — first fails today (today is used)
- Green check: focused vitest in `plugin/tests/settings-declarative-tab.test.ts`
- regression checks: `cd plugin && npm run typecheck && npm test`
- [x] implementation and tests accepted

### Chunk 2 — runs refresh open dashboards (F18)

- change kind: bug fix
- strategy: strict Red-Green-Refactor
- Red / baseline signal: tests: after `generateFirstReport` and after the "Run today" command, open dashboard leaves get `refreshFromVault` — fail today
- Green check: focused vitest (`tests/settings-declarative-tab.test.ts`, `tests/commands.test.ts`)
- regression checks: `cd plugin && npm test`
- [x] implementation and tests accepted

## Phase verification

- Full check set from goal.md constraints

## Abort / reshape triggers

- If `runForDateNow` rejects past announced dates in some state (e.g. already completed), stop and reconsider which date the guide should pick.
