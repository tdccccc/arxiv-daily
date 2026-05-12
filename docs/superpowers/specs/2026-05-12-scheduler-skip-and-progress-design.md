# Scheduler boot tick, skip-existing, and status-bar progress — Design

**Date:** 2026-05-12
**Scope:** `plugin/` (Obsidian plugin), three coupled fixes/improvements.

## Background

Two regressions surfaced after a BRAT reinstall:

1. **Install kicks off a 5-day backlog summarization.** `main.ts:66` starts the scheduler unconditionally when `schedule.enabled`. `SchedulerService.start()` (`plugin/src/services/scheduler.ts:35-37`) registers the interval *and* immediately invokes `this.tick()`. `tick()` then walks `lookbackDays` (default 5) days; with an empty `runState` (BRAT wiped `data.json`), every day inside the window looks `pending`, so all of them get processed.
2. **Pre-existing files get silently overwritten.** `MarkdownWriter.backupIfExists()` (`plugin/src/pipeline/markdown-writer.ts:67-76`) renames the existing file to `*.bak.md` and writes a new one on top. There is no skip path. Evidence: `plugin_test/arxiv-daily/daily/2026-05-11.bak.md`. The "already exists" semantics already live in `manual-fetch.ts:60-63` for the manual-id flow; the scheduler path has no equivalent.

A third, related gap: while a long run is in progress, the user has no way to tell which date / which stage / how many papers in.

## Requirements

The fix must:

1. On plugin load, only do work for today (still respecting `runAtLocal`). Interval ticks keep the full lookback behavior.
2. Never overwrite existing daily or paper files. If a daily file for a date exists, skip the whole date (no fetch, no LLM call). If an individual paper file exists, skip just that paper.
3. Show live progress on the Obsidian status bar: which date, which pipeline stage, counter (no ETA).
4. No regressions in `manual-fetch` (which already has its own duplicate check).
5. No new persisted-data shape changes beyond what's necessary.

## Non-goals

- Removing the existing `.bak.md` files in the user's vault — that's their housekeeping.
- ETA estimation. LLM stage durations swing widely; an unreliable ETA is worse than none.
- Configurable progress format / hide-status-bar toggle. Can be added later if requested.
- Migrating `runState` shape, adding new statuses, or persisting progress.

## Block 1 — Boot tick only covers today

**File:** `plugin/src/services/scheduler.ts`

Refactor:

- Extract the per-date branch of `tick()` (lines 60-72) into `private async tickDate(date: string, today: string, now: Date): Promise<void>`. It preserves all current guards: `isDone`, `running`, scheduled-time gate (`date === today && minutesNow < scheduledMin`), and `failed_transient` cooldown.
- `tick()` becomes the lookback loop (current behavior), iterating `i` and calling `tickDate(...)`.
- Add `private async tickToday(): Promise<void>` that computes `today` and calls `tickDate(today, today, now)` once.
- `start()` replaces its `this.tick().catch(...)` line with `this.tickToday().catch(...)`. Interval-fired ticks still call `this.tick()` (full lookback).

Manual flows (`runForDateNow`, `runAllPending`) are unchanged.

### Why this shape

- Keeps the lookback semantics on the interval — users who leave Obsidian closed for a few days still get backfill, just not at install time.
- The `runAtLocal` gate still applies to `tickToday` (because today === today branch checks it), so the "don't run before 09:30" preference holds.
- One call site change in `start()`; no new fields, no behavior changes for manual triggers.

## Block 2 — Skip existing files; remove silent overwrite

### 2a. Pipeline-level pre-check (daily file)

**File:** `plugin/src/pipeline/pipeline.ts`

At the very top of `runForDate(dateStr)` (before fetch /recent), check whether the daily file exists:

```ts
if (await this.deps.writer.dailyExists(dateStr)) {
  this.deps.logger.info(`pipeline: daily ${dateStr} already exists, skipping`);
  return { kind: "completed", papersWritten: 0 };
}
```

The path-building stays inside `MarkdownWriter` (see 2c).

Returning `completed` means `state-store.setCompleted` runs in `scheduler.tryRun`, which flips runState to `completed` and `isDone(date)` returns true on every subsequent tick. Side effect: `papersWritten` is recorded as `0` for skipped dates. We accept that tradeoff (confirmed in brainstorming) because the alternative is to re-stat the file on every tick.

To avoid duplicating path logic, expose `dailyExists(dateStr): Promise<boolean>` on `MarkdownWriter` rather than reaching into `vault.adapter` from the pipeline.

### 2b. Per-paper skip in detail loop

**File:** `plugin/src/pipeline/pipeline.ts`

In the step-8 detail loop, before calling `summarizePaperDetail`, check existence. Skip the LLM call entirely if the file exists:

```ts
for (const p of detailPapers) {
  if (await this.deps.writer.paperDetailExists(p.id)) {
    logger.info(`pipeline: detail ${p.id} already exists, skipping`);
    continue;
  }
  // ... existing summarize + write
}
```

Add `paperDetailExists(id: string): Promise<boolean>` to `MarkdownWriter` for the same reason.

### 2c. Remove `backupIfExists`; fail loud on accidental overwrite

**File:** `plugin/src/pipeline/markdown-writer.ts`

- Delete the `backupIfExists` method.
- In `writeDaily`, `writePaperDetail`, `writeEmptyDaily`: replace the `await this.backupIfExists(path)` call with an `if (await this.opts.vault.adapter.exists(path)) throw new Error(...)`.
- Add the two new existence-check methods (`dailyExists`, `paperDetailExists`) and centralize the path-building helpers as private methods to avoid drift.

Why throw instead of silently no-op: callers are *supposed* to have checked. If they didn't, that's a bug we want to surface, not paper over with a backup file.

The thrown error bubbles up to `tryRun` → caught and translated to `failed_transient`. The user sees a Notice; the state remains recoverable.

### 2d. Manual-fetch interaction

`ManualFetchService.fetchAndSummarize` (`plugin/src/services/manual-fetch.ts:60-63`) already returns `{ kind: "already_exists" }` before any network call, so its happy path is unchanged. The strictness change in 2c does mean: if a race makes the file appear between the pre-check and the write, `writePaperDetail` will throw instead of silently overwriting. The caller's catch path translates that to `{ kind: "error", reason: ... }` — strictly safer than the current silent backup.

## Block 3 — Status-bar progress (counter, no ETA)

### Interface

**New file:** `plugin/src/services/progress.ts`

```ts
export type ProgressStage =
  | "fetch-recent"
  | "enrich-abstract"
  | "filter"
  | "fetch-content"
  | "summarize-daily"
  | "write-detail";

export interface ProgressReporter {
  /** Called by scheduler at the start of each date in a lookback run. */
  setBatch(currentDay: number, totalDays: number, date: string): void;
  /** Called by pipeline at the start of each stage (and per-paper inside loops). */
  setStage(stage: ProgressStage, current?: number, total?: number): void;
  /** Called when scheduler finishes (or no work to do). */
  setIdle(lastCompletedDate?: string): void;
}
```

A `NoopProgressReporter` (all methods no-op) is provided for tests and for environments without a status bar (defensive — `addStatusBarItem` is always available in Obsidian desktop).

### Status-bar controller

**New file:** `plugin/src/services/status-bar.ts`

`StatusBarController` implements `ProgressReporter`. Holds:

- `el: HTMLElement` (from `addStatusBarItem()`)
- `batch: { current: number; total: number; date: string } | null`
- `stage: { stage: ProgressStage; current?: number; total?: number } | null`
- `lastCompletedDate?: string`

Each setter updates internal state and calls `private render()`. Render formats:

- Idle without history: `arXiv: idle`
- Idle with history: `arXiv: idle · last 2026-05-11`
- Single-date run: `arXiv: 2026-05-10 · summarize` (when `batch.total === 1`)
- Batch run: `arXiv: 2026-05-10 [2/5] · fetch 3/8` (when `batch.total > 1`)

Stage label table (kept short for the status bar):

| Stage | Label |
| --- | --- |
| `fetch-recent` | `fetch /recent` |
| `enrich-abstract` | `abstracts` |
| `filter` | `filter` |
| `fetch-content` | `fetch i/n` (uses current/total) |
| `summarize-daily` | `summarize` |
| `write-detail` | `detail i/n` |

`lastCompletedDate` survives across runs in-memory (not persisted). On plugin start it's derived from `stateStore.snapshot()` — pick the most recent `status === "completed"` date.

### Wiring

**File:** `plugin/main.ts`

- onload: `const progress = new StatusBarController(this.addStatusBarItem(), this.stateStore);` (constructor seeds `lastCompletedDate` from snapshot).
- `SchedulerService` constructor: extend `SchedulerDeps` with `progress: ProgressReporter` and store it.
- `buildPipeline()`: pass `progress` into `PipelineDeps`.
- `ArxivPipeline.runForDate()` calls `progress.setStage(...)` at each step boundary; inside the per-paper loops (step 6 fetch-content, step 8 write-detail) calls `setStage(stage, i+1, total)`.
- `SchedulerService` calls `progress.setBatch(...)` for *every* code path that drives runs, and `progress.setIdle(lastCompletedDate)` at the end of each:
  - `tick()` (interval lookback): `setBatch(i+1, lookbackDays, date)` in the loop body, `setIdle` after the loop.
  - `tickToday()`: `setBatch(1, 1, today)` then run, `setIdle` after.
  - `runForDateNow(date)`: `setBatch(1, 1, date)` then run, `setIdle` after.
  - `runAllPending()`: `setBatch(i+1, lookbackDays, date)` in the loop, `setIdle` after.
- `tryRun` updates an internal `lastCompletedDate` on `completed`; on the next `setIdle` call the controller reads it.

`onunload` clears the status bar element (Obsidian auto-cleans, but explicit is safer).

### Why a single-callback design

Producer = pipeline + scheduler; consumer = status-bar controller. Two writers, one reader. Direct callback injection is simpler than an event bus. If we later want a "history modal that also subscribes," the upgrade path is to wrap the reporter in a fan-out — but YAGNI for now.

### Testing surface

- `MarkdownWriter`: unit tests for `dailyExists`/`paperDetailExists` and the new throw-on-exists behavior. Use a fake `Vault` (already common in existing tests, see `plugin/tests/`).
- `ArxivPipeline.runForDate`: test that an existing daily file short-circuits to `{ kind: "completed", papersWritten: 0 }` without invoking fetcher/llm — verify with mocks throwing if called.
- `ArxivPipeline.runForDate`: test that an existing paper file skips that paper's `summarizePaperDetail` call but proceeds with others.
- `SchedulerService.start`: test that `tickToday` is called once and that interval ticks call full `tick`. Use injected `now` and a stub `runForDate`.
- `StatusBarController`: test that label rendering matches each case (idle/idle+last/single/batch/per-paper-counter).

## Architecture summary

```
main.onload
  ├── StateStore.load
  ├── StatusBarController(statusBarEl, stateStore)        ◄── seeds lastCompletedDate
  ├── SchedulerService({ ..., progress })
  └── scheduler.start()
        └── tickToday()                                    ◄── new: only today
              └── tickDate(today, today, now)              ◄── extracted from old tick()
                  └── tryRun(date)
                        ├── store.setRunning
                        ├── pipeline.runForDate(date)
                        │     ├── writer.dailyExists  → return early if true
                        │     ├── stage emits → progress.setStage(...)
                        │     └── for each detail: writer.paperDetailExists → skip if true
                        └── store.setCompleted / setFailed
  (interval) → tick() → for each i in lookbackDays: tickDate(...) → progress.setBatch(...)
```

## Risk / rollback

- If the throw-on-exists guard fires unexpectedly, the run is marked `failed_transient` and the user sees a Notice. Manual rollback: revert `markdown-writer.ts` to bring back `backupIfExists`.
- The `papersWritten: 0` sentinel for skipped daily files is the only semantic change to persisted runState. It's only read by `StateModal` for display, so impact is cosmetic.
- Status-bar controller is additive; failure to construct it must not break the plugin. Wrap construction in try/catch; on failure inject `NoopProgressReporter`.
