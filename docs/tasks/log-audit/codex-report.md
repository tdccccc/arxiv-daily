# Codex Report: log-audit

## Status

success

## Completed Sections

- A: Added dashboard `logger.info` coverage for batch actions, no-op batch results, completed batch updates, and detail-summary deletion.
- B: Added `plugin.logger.info` / `warn` calls alongside command-palette `Notice` paths in `commands.ts`.
- C: Verified dashboard manual run buttons now log before user-triggered runs (`Run Today`, calendar run, run pending, retry failed).
- D: Added missing manual-fetch logs for request start, invalid IDs, existing-note read failures, metadata success/not-found, no full text, index writes, detail-note writes, and paperPath storage.
- E: Added paper-note logs for reusing existing notes, finding default-path notes, creating notes, and storing `paperPath`.
- F: Added onboarding setup-status logging and wired it into the dashboard setup-blocked run path.
- G: Searched all `plugin/src` catch blocks. No truly empty catch blocks remain. Comment-only catches are only in already-done run-history/pipeline files and were intentionally left untouched.

## Logger Lines Added

- `plugin/src/commands.ts`
  - `plugin.logger.info(message)` in the shared command `notice()` helper.
  - `plugin.logger.warn("run history load failed", e)`.
  - `plugin.logger.info("arXiv Daily: run history copied")`.
  - `plugin.logger.warn("Could not copy run history; text is selectable", e)`.
  - `plugin.logger.warn("diagnostics load failed", e)`.
  - `plugin.logger.info("arXiv Daily: diagnostics copied")`.
  - `plugin.logger.warn("Could not copy diagnostics; text is selectable", e)`.
  - `plugin.logger.warn("diagnostics: failed to read arXiv ID from ...", e)`.
- `plugin/src/dashboard/view.ts`
  - Workspace/open-settings/setup-state/manual-run/batch/PDF/open-URL paths now log via `plugin.logger.info`.
  - Dashboard command/control failures and PDF/delete failures now log via `plugin.logger.warn`.
  - A shared dashboard `notice()` helper mirrors dashboard Notices to the logger.
- `plugin/src/services/manual-fetch.ts`
  - Added `logger.info` for request start, metadata fetched, existing notes, index updates, saved status, removing empty notes, writing detail notes, storing paperPath, and write completion.
  - Added `logger.warn` for invalid IDs, existing-note read failure, missing arXiv entries, missing full text, and existing-index refresh failure paths.
  - Existing `logger.error` failure paths remain in place.
- `plugin/src/services/paper-note.ts`
  - Added `plugin.logger.info` for existing note reuse, default-path discovery, paperPath storage, and note creation.
- `plugin/src/onboarding.ts`
  - Added `logSetupStatus()` helper using `logger.info`.

## Changed Files

- `plugin/src/commands.ts`
- `plugin/src/dashboard/view.ts`
- `plugin/src/services/manual-fetch.ts`
- `plugin/src/services/paper-note.ts`
- `plugin/src/onboarding.ts`
- `docs/tasks/log-audit/codex-report.md`

## Verification

Command:

```bash
cd /home/tiandc/Documents/code/arxiv-daily/plugin && npx tsc --noEmit
```

Result: passed with zero TypeScript errors.

Additional checks:

```bash
rg -n -U "catch\\s*(?:\\([^)]*\\))?\\s*\\{\\s*\\}" plugin/src
```

Result: no truly empty catch blocks.

```bash
cd /home/tiandc/Documents/code/arxiv-daily/plugin && npm test
```

Result: failed in `tests/summarizer.test.ts` with 3 failures and 1 snapshot mismatch. The failures compare prompt/category wording and golden snapshots for daily summarization. The affected implementation files (`plugin/src/prompts/*`, `plugin/src/pipeline/summarizer.ts`) had pre-existing uncommitted changes and were outside this logging task, so they were left untouched.

## Risks / Follow-ups

- `plugin/src/services/run-history.ts` and pipeline cache cleanup files still have intentional comment-only catch blocks. They are in the already-done/out-of-scope set, so they were not changed.
- `plugin/src/settings/tab.ts` model-fetch Notices were already outside sections A-G and were left unchanged.

## Suggested Commit Message

```text
feat(logging): add audit logs for user-triggered plugin actions

Add logger coverage for command Notices, dashboard manual and batch actions,
manual detail fetches, paper-note creation, onboarding setup status, and
diagnostic catch paths so user-triggered operations appear in the in-memory
log buffer.

Verification:
- cd plugin && npx tsc --noEmit
```
