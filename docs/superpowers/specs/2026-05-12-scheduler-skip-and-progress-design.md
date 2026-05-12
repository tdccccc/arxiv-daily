# Scheduler enable-gating, skip-existing, status-bar progress — Design

**Date:** 2026-05-12 (revised)
**Scope:** `plugin/` (Obsidian plugin), three coupled changes.

## Background

Two regressions surfaced after a BRAT reinstall:

1. **Install kicks off a 5-day backlog summarization.** `main.ts:66` starts the scheduler unconditionally when `schedule.enabled`. Combined with `enabled: true` as the default and BRAT wiping `data.json` (so `runState` is empty), the scheduler immediately processes the entire `lookbackDays` window (default 5) at install time.
2. **Pre-existing files get silently overwritten.** `MarkdownWriter.backupIfExists()` renames the existing file to `*.bak.md` and writes a new one on top. Evidence: `plugin_test/arxiv-daily/daily/2026-05-11.bak.md`.

A third gap: while a long run is in progress, the user has no way to tell which date / which stage / how many papers in.

## Requirements (user-visible behavior)

After this change the plugin will:

1. Be **disabled by default** on fresh install (and BRAT reinstall, where `data.json` is wiped) — nothing summarizes until the user explicitly enables it.
2. Surface the enabled/disabled state in **two places** (ribbon menu and settings panel), with one synchronized toggle.
3. On Enable: start the scheduler interval **and** trigger one summary for today. Skip the trigger silently if today is a weekend (Sat/Sun in `arxiv.timezone`); the interval will pick up the next workday.
4. On Disable: stop the scheduler interval. Files already in the vault are untouched. Manual commands (Run for today / Run all pending / Run for date / Summarize by ID) keep working regardless of state.
5. On Obsidian restart with `enabled=true` saved: start the interval and run a today-only tick (not a full lookback). Weekend-skip rule still applies.
6. Never overwrite an existing daily or paper file. If a daily file for a date exists, skip the whole date (no fetch, no LLM call). If an individual paper file exists, skip just that paper's detail report.
7. Show live progress on the Obsidian status bar: which date, which pipeline stage, paper counter. No ETA.
8. Existing users (their saved `enabled=true` persists) are not disrupted — their plugin continues to operate normally on upgrade. Only fresh installs see the disabled default.

## Non-goals

- Removing the existing `.bak.md` files in the user's vault — that's their housekeeping.
- ETA estimation. LLM stage durations swing widely; an unreliable ETA is worse than none.
- Holiday detection (only weekend-skip; arXiv US holidays fall through to current transient-retry behavior).
- Persisting any new state shape.
- Changing manual command behavior.

## Block 1 — Enable-gating and boot-tick scoping

### 1a. Default change

`DEFAULT_SETTINGS.schedule.enabled` flips from `true` to `false`. Existing users' persisted `enabled: true` is preserved by the load-and-merge step (settings are deep-merged in `mergeSettings`, so a saved `true` wins over the new default).

### 1b. Plugin-level toggle action

`main.ts` gains a single method that the ribbon and the settings tab both call:

```ts
async setScheduleEnabled(enabled: boolean): Promise<void>
```

Behavior:
- If `enabled === this.settings.schedule.enabled`: no-op.
- Persists `settings.schedule.enabled = enabled`.
- If turning ON: `scheduler.start()` (which now reads the flag) and `scheduler.tickToday()` (one-shot, see 1d).
- If turning OFF: `scheduler.stop()`.

This replaces the existing `restartScheduler()` use site in the settings tab. The settings-tab toggle now calls `setScheduleEnabled(v)` instead of `restartScheduler()`. This unifies behavior: changing the toggle from the settings panel also triggers today's summary, matching the ribbon-menu Enable action.

### 1c. Ribbon menu changes

In `registerCommands()` the ribbon-icon menu gains a status header and a toggle item at the top, above the existing items. Sketch:

```
arXiv Daily — Status: Enabled        (read-only header item, disabled)
─────────────────────────────────
Disable                              (toggle; reads "Enable" when off)
─────────────────────────────────
Run for today
Run all pending (lookback)
Run for specific date…
Summarize by arXiv ID…
```

Implementation note: the existing `MenuItem.setDisabled` from the obsidian mock is fine for the read-only header line.

The status header text is computed at menu-open time from `plugin.settings.schedule.enabled` (the menu is rebuilt on each click of the ribbon, so this is naturally live).

### 1d. Scheduler refactor (boot tick scope)

**File:** `plugin/src/services/scheduler.ts`

- Extract the per-date branch of `tick()` (current lines 60-72) into a private helper:

  ```ts
  private async tickDate(
    date: string,
    opts: {
      now: Date;
      // When set, "if (minutesNow < scheduledMin) continue" applies.
      // tick() supplies this when date === today. tickToday() omits it.
      timeGate?: { scheduledMin: number; minutesNow: number };
    },
  ): Promise<PipelineResult | undefined>
  ```

  Preserves `isDone`, `running`, and `failed_transient` cooldown guards exactly as today. `runAtLocal` only applies when `opts.timeGate` is provided.

- `tick()` becomes the lookback loop, calling `tickDate(...)` per iteration with the time gate set for today only (current behavior).

- New `async tickToday(): Promise<PipelineResult | { kind: "skipped"; reason: string } | undefined>`:
  - Early return `{ kind: "skipped", reason: "disabled" }` if `!schedule.enabled`.
  - Computes `today` in `arxiv.timezone`.
  - **Weekend check first:** if `isWeekendInTz(now, arxiv.timezone)`, returns `{ kind: "skipped", reason: "weekend" }` without calling `tickDate`. Caller updates status-bar reporter to show idle/weekend. Silent — no Notice.
  - Otherwise calls `tickDate(today, { now })` (no timeGate — Enable bypasses `runAtLocal`).
  - If `tickDate` returns undefined (guarded out by `isDone`/`running`/cooldown), surface as `{ kind: "skipped", reason: "..." }` for caller logging. Silent — no Notice.

- `start()` no longer calls `this.tick()` for the initial run. It only registers the interval. Initial tick (when appropriate) is invoked explicitly by `main.ts` onload and by `setScheduleEnabled(true)`.

- `restartScheduler()` in `main.ts` (used when `tickIntervalMin` changes in settings) keeps working: it calls `stop()` + `start()`. With the new `start()`, this just re-registers the interval — no surprise tick. That's an improvement over the previous behavior where changing the interval would silently re-run today.

### 1e. `main.ts` onload changes

Replace `if (this.settings.schedule.enabled) this.scheduler.start();` with:

```ts
if (this.settings.schedule.enabled) {
  this.scheduler.start();
  this.scheduler.tickToday().catch((e) =>
    this.logger.error("scheduler initial tickToday failed", e),
  );
}
```

This preserves the "user has been using the plugin, they restart Obsidian, today gets summarized" UX without ever firing the 5-day backfill at boot.

### 1f. Weekend detection helper

**File:** `plugin/src/utils/time.ts`

Add a tiny helper:

```ts
export function isWeekendInTz(d: Date, tz: string): boolean
```

Returns `true` if the calendar weekday of `d` in `tz` is Saturday or Sunday. Implementation uses `Intl.DateTimeFormat(en-US, { timeZone: tz, weekday: "short" })`. Other modules import this where needed.

## Block 2 — Skip existing files; remove silent overwrite

### 2a. Pipeline-level pre-check (daily file)

**File:** `plugin/src/pipeline/pipeline.ts`

At the top of `runForDate(dateStr)` (before fetch /recent), check whether the daily file exists:

```ts
if (await this.deps.writer.dailyExists(dateStr)) {
  this.deps.logger.info(`pipeline: daily ${dateStr} already exists, skipping`);
  return { kind: "completed", papersWritten: 0 };
}
```

Returning `completed` means `state-store.setCompleted` runs in `scheduler.tryRun`, which flips `runState` to `completed` and `isDone(date)` returns true on subsequent ticks. Side effect: `papersWritten` is recorded as `0` for skipped dates. We accept this tradeoff (confirmed in brainstorming) because the alternative is re-stat'ing the file on every tick.

### 2b. Per-paper skip in detail loop

In the step-8 detail loop, before calling `summarizePaperDetail`, check existence and skip the LLM call entirely:

```ts
for (const p of detailPapers) {
  if (await this.deps.writer.paperDetailExists(p.id)) {
    logger.info(`pipeline: detail ${p.id} already exists, skipping`);
    continue;
  }
  // ... existing summarize + write
}
```

### 2c. MarkdownWriter — add existence checks; remove `backupIfExists`

**File:** `plugin/src/pipeline/markdown-writer.ts`

- Add `dailyExists(dateStr): Promise<boolean>` and `paperDetailExists(id): Promise<boolean>` so callers don't reach into `vault.adapter` directly.
- Delete `backupIfExists`.
- In `writeDaily`, `writePaperDetail`, `writeEmptyDaily`: replace `await this.backupIfExists(path)` with `if (await this.opts.vault.adapter.exists(path)) throw new Error(...)`. Callers are expected to check first; the throw is a safety net.

Why throw instead of silent no-op: callers are *supposed* to have checked. If they didn't, that's a bug to surface, not paper over with a backup file. The thrown error bubbles up to `tryRun` → caught and translated to `failed_transient`. The user sees a Notice; the state remains recoverable.

### 2d. Manual-fetch interaction

`ManualFetchService.fetchAndSummarize` (`manual-fetch.ts:60-63`) already returns `{ kind: "already_exists" }` before any network call, so its happy path is unchanged. The strictness change in 2c does mean: if a race makes the file appear between the pre-check and the write, `writePaperDetail` will throw instead of silently overwriting. The caller's catch path translates that to `{ kind: "error", reason: ... }` — strictly safer than the current silent backup.

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
  setBatch(currentDay: number, totalDays: number, date: string): void;
  setStage(stage: ProgressStage, current?: number, total?: number): void;
  setIdle(lastCompletedDate?: string, reason?: "weekend" | "disabled"): void;
  setDisabled(): void;
}
```

A `NoopProgressReporter` (all methods no-op) is provided for tests and for safe fallback.

### Status-bar controller

**New file:** `plugin/src/services/status-bar.ts`

`StatusBarController` implements `ProgressReporter`. Holds `el`, internal `batch`/`stage`/`lastCompletedDate`/`disabled` state. Each setter updates state and calls `render()`. Render formats:

| State | Status bar text |
| --- | --- |
| Disabled | `arXiv: disabled` |
| Enabled, idle, no history | `arXiv: idle` |
| Enabled, idle, has history | `arXiv: idle · last 2026-05-11` |
| Enabled, idle, weekend skip | `arXiv: idle · weekend` |
| Single-date run | `arXiv: 2026-05-10 · summarize` |
| Batch run | `arXiv: 2026-05-10 [2/5] · fetch 3/8` |

Stage label table (short for status bar):

| Stage | Label |
| --- | --- |
| `fetch-recent` | `fetch /recent` |
| `enrich-abstract` | `abstracts` |
| `filter` | `filter` |
| `fetch-content` | `fetch i/n` |
| `summarize-daily` | `summarize` |
| `write-detail` | `detail i/n` |

`lastCompletedDate` is seeded at construction time from `stateStore.snapshot()` — pick the lexically max date with `status === "completed"`.

### Wiring

**File:** `plugin/main.ts`

- onload constructs `progress = new StatusBarController(this.addStatusBarItem(), this.stateStore)`.
- If `enabled === false`: progress shows disabled state.
- `setScheduleEnabled(true)` ends with `progress.setIdle(...)` so the bar transitions from "disabled" to "idle".
- `setScheduleEnabled(false)` calls `progress.setDisabled()`.
- `SchedulerDeps` gains `progress: ProgressReporter`.
- `PipelineDeps` gains `progress: ProgressReporter`.
- `ArxivPipeline.runForDate()` calls `progress.setStage(...)` at each step boundary; inside per-paper loops (step 6 fetch-content, step 8 write-detail) calls `setStage(stage, i+1, total)`.
- `SchedulerService` calls `setBatch` at the start of each run path and `setIdle` at the end:
  - `tick()` (interval lookback): `setBatch(i+1, lookbackDays, date)` in the loop body, `setIdle` after.
  - `tickToday()`: `setBatch(1, 1, today)` then run; on weekend skip, `setIdle(lastCompletedDate, "weekend")`.
  - `runForDateNow(date)`: `setBatch(1, 1, date)` then run; `setIdle` after.
  - `runAllPending()`: `setBatch(i+1, lookbackDays, date)` in the loop; `setIdle` after.

### Why a single-callback design

Producer = pipeline + scheduler; consumer = status-bar controller. Two writers, one reader. Direct callback injection is simpler than an event bus. If we later want a "history modal that also subscribes," wrap the reporter in a fan-out then.

### Testing surface

- `MarkdownWriter`: unit tests for `dailyExists`/`paperDetailExists`; verify writers throw on pre-existing path; verify `backupIfExists` is gone (no `.bak.md` produced).
- `ArxivPipeline.runForDate`: existing-daily short-circuits to `{ kind: "completed", papersWritten: 0 }` without calling fetcher/llm (mocks throw if called); existing-paper skips that paper but proceeds with others.
- `SchedulerService.start`: with `enabled=false`, `start()` does nothing (no interval registered). `tickToday` weekend-skip returns early. `setScheduleEnabled(true)` calls both `start` and `tickToday`.
- `StatusBarController`: render matches table per state (disabled / idle / idle+last / idle+weekend / single / batch / per-paper-counter).
- `isWeekendInTz` helper: verify Sat/Sun in Asia/Shanghai and in other tz.

## Architecture summary

```
main.onload
  ├── load settings (default enabled=false; saved enabled=true wins)
  ├── StatusBarController(statusBarEl, stateStore)
  │     (renders "arXiv: disabled" or "arXiv: idle · last YYYY-MM-DD")
  ├── SchedulerService({ ..., progress })
  └── if settings.schedule.enabled:
        scheduler.start()         (interval only)
        scheduler.tickToday()     (today-only, weekend-aware)

Ribbon menu (rebuilt on each open)
  ├── header: "Status: Enabled" | "Status: Disabled"
  ├── toggle: → plugin.setScheduleEnabled(!currentlyEnabled)
  └── (existing manual command items, unaffected)

Settings tab "启用自动调度" toggle
  └── → plugin.setScheduleEnabled(value)

plugin.setScheduleEnabled(true)
  └── scheduler.start() + scheduler.tickToday()

plugin.setScheduleEnabled(false)
  └── scheduler.stop() + progress.setDisabled()
```

## Risk / rollback

- Existing users could be confused by the new ribbon-menu header. The status header is non-interactive; no behavior change for them.
- If the `setScheduleEnabled` wiring breaks, the existing settings-tab toggle behavior is the fallback (it would just call `restartScheduler()` as before — we should keep this path testable).
- Status-bar controller is additive; failure to construct it must not break the plugin. Wrap construction in try/catch; on failure use `NoopProgressReporter`.
- The `papersWritten: 0` sentinel for skipped daily files is the only semantic change to persisted runState. It's only read by `StateModal` for display, so impact is cosmetic.
