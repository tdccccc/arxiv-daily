# dashboard-perf-hub Codex Report

## Status

- A: Completed. Replaced the dashboard `LogModal` with `HubModal` in `plugin/src/dashboard/view.ts`. The hub has Logs, Run History, and Diagnostics tabs with common Refresh, Clear, Copy, and Close controls.
- B: Completed. `reloadIndex()` now filters vault markdown files to `dailyDir` and `papersDir`, then logs the scanned/total file count.
- C: Completed. Dashboard table rendering now uses 20-row pagination with prev/next controls and `Showing X-Y of N papers` count text.
- D: Completed. Added an in-memory daily-path cache that skips `syncDashboardHistory()` when the daily file path set is unchanged and existing entries are already loaded.
- E: Completed. Changed `MAX_BUFFER_SIZE` in `plugin/src/services/logger.ts` from 1000 to 5000.

## Files Changed

- `plugin/src/dashboard/view.ts`
- `plugin/src/services/logger.ts`
- `plugin/styles.css`
- `plugin/tests/dashboard-view.test.ts`
- `plugin/tests/logger.test.ts`
- `docs/tasks/dashboard-perf-hub/codex-report.md`

## Verification

- `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npm test -- tests/dashboard-view.test.ts tests/logger.test.ts`
  - Passed: 18 tests.
- `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npx tsc --noEmit`
  - Passed with zero errors.
- `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npm run build`
  - Passed.
- Additional check: `cd /home/tiandc/Documents/code/arxiv-daily/plugin && npm test`
  - Failed in pre-existing summarizer prompt/snapshot expectations in `tests/summarizer.test.ts`.
  - The failures reference prompt wording/category display changes outside this task's modified dashboard/logger files.

## Notes

- Preserved the existing uncommitted changes in the worktree and did not run `git add` or `git commit`.
- Existing command-palette run-history and diagnostics modals in `plugin/src/commands.ts` were left untouched. The dashboard More menu now opens the new hub modal.

## Suggested Commit Message

```text
perf(dashboard): add hub modal and paginated reload cache

Replace the dashboard log modal with a three-tab hub for logs, run history,
and diagnostics. Limit dashboard reload scans to configured daily and paper
directories, add 20-row table pagination, and skip history sync when the
daily file path set is unchanged.

Increase the in-memory logger buffer to 5000 entries and add focused tests
for dashboard filtering, sync-cache decisions, pagination, and log retention.

Verification:
- npm test -- tests/dashboard-view.test.ts tests/logger.test.ts
- npx tsc --noEmit
- npm run build
```
