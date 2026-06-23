# Dashboard Run Whitelist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dashboard calendar `Run` display follow the approved whitelist: today requires the local Run window; non-today dates require confirmation from the latest `/recent` refresh; all dates require no local report and no terminal/blocking run state.

**Architecture:** Add a small `/recent` date cache service that unions announce dates across configured categories, where any configured category containing a date makes that date eligible. Keep dashboard visual state as a pure projection: daily reports and terminal run state win first, then whitelist checks decide whether to show the green `Run` affordance. Refresh the cache on dashboard open/refresh and before dashboard run actions; scheduler can refresh the same cache when it wakes so UI and automatic polling share the same source of truth.

**Tech Stack:** TypeScript, Obsidian plugin APIs, Vitest, existing `ArxivFetcher`, `parseRecent`, `StateStore`, and dashboard view helpers.

---

## File Structure

- Create `plugin/src/services/recent-dates.ts`
  - Owns the latest `/recent` date snapshot for configured arXiv categories.
  - Uses one fetcher per refresh and unions dates from all successfully fetched configured categories.
  - Defaults to an empty date set when no category can be fetched or parsed, so non-today dashboard cells do not show `Run` without confirmation.

- Create `plugin/tests/services/recent-dates.test.ts`
  - Covers category union behavior, parse failures, and snapshot immutability.

- Modify `plugin/main.ts:16-112, 210-270`
  - Adds `recentDates` to the plugin instance.
  - Extracts `buildArxivFetcher()` so recent-date cache and pipeline construction share the same fetcher setup.

- Modify `plugin/src/dashboard/view.ts:1-180, 290-322, 904-980, 1053-1066, 1687-1705`
  - Adds pure whitelist helpers.
  - Uses recent-date cache snapshot when building calendar cells.
  - Refreshes recent-date cache on dashboard open/refresh and before dashboard run actions.

- Modify `plugin/src/services/scheduler.ts:1-180, 227-270`
  - Optionally accepts recent-date cache.
  - Refreshes cache when scheduled polling wakes.
  - Uses both `runAtLocal` and `runUntilLocal` so automatic polling only happens inside the Run window.

- Modify `plugin/tests/dashboard/calendar-state.test.ts`
  - Adds focused tests for today and non-today whitelist logic.

- Modify `plugin/tests/scheduler.test.ts`
  - Adds focused tests for the Run window end time and cache refresh on scheduled polling.

---

### Task 1: Add Recent Date Cache Service

**Files:**
- Create: `plugin/src/services/recent-dates.ts`
- Create: `plugin/tests/services/recent-dates.test.ts`

- [ ] **Step 1: Write the failing service tests**

Create `plugin/tests/services/recent-dates.test.ts`:

```typescript
import { describe, expect, it, vi } from "vitest";
import { RecentDatesCache } from "../../src/services/recent-dates";
import type { PluginSettings } from "../../src/settings/types";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";

function recentHtml(date: string, id: string): string {
  return `
    <html>
      <body>
        <dl id="articles">
          <h3>${dateHeader(date)}</h3>
          <dt><a title="Abstract">arXiv:${id}</a></dt>
          <dd>
            <div class="list-title">Title: Example ${id}</div>
            <div class="list-authors"><a>Author</a></div>
          </dd>
        </dl>
      </body>
    </html>
  `;
}

function dateHeader(date: string): string {
  const [year, month, day] = date.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-US", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

function settingsWithCategories(categories: string[]): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    arxiv: {
      ...DEFAULT_SETTINGS.arxiv,
      category: categories[0],
      categories,
    },
  };
}

describe("RecentDatesCache", () => {
  it("treats a date as present when any configured category has it", async () => {
    const fetcher = {
      fetchRecent: vi.fn(async (category: string) =>
        category === "cs.CL"
          ? recentHtml("2026-06-22", "2606.00001")
          : recentHtml("2026-06-19", "2606.00002"),
      ),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL", "astro-ph"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date("2026-06-23T01:00:00Z"),
    });

    await cache.refresh();

    expect(cache.hasDate("2026-06-22")).toBe(true);
    expect(cache.hasDate("2026-06-19")).toBe(true);
    expect(cache.hasDate("2026-06-18")).toBe(false);
    expect(fetcher.fetchRecent).toHaveBeenCalledWith("cs.CL");
    expect(fetcher.fetchRecent).toHaveBeenCalledWith("astro-ph");
  });

  it("keeps successful category dates when another category fails", async () => {
    const logger = { debug: vi.fn(), warn: vi.fn() };
    const fetcher = {
      fetchRecent: vi.fn(async (category: string) => {
        if (category === "astro-ph") throw new Error("network down");
        return recentHtml("2026-06-22", "2606.00001");
      }),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL", "astro-ph"]),
      buildFetcher: () => fetcher,
      logger,
      now: () => new Date("2026-06-23T01:00:00Z"),
    });

    const snapshot = await cache.refresh();

    expect(snapshot.status).toBe("ready");
    expect(cache.hasDate("2026-06-22")).toBe(true);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("recent dates refresh failed for astro-ph"),
    );
  });

  it("clears confirmed dates when every configured category fails", async () => {
    const fetcher = {
      fetchRecent: vi.fn(async () => {
        throw new Error("arXiv unavailable");
      }),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date("2026-06-23T01:00:00Z"),
    });

    const snapshot = await cache.refresh();

    expect(snapshot.status).toBe("failed");
    expect(snapshot.dates.size).toBe(0);
    expect(cache.hasDate("2026-06-22")).toBe(false);
  });

  it("returns immutable snapshots", async () => {
    const fetcher = {
      fetchRecent: vi.fn(async () => recentHtml("2026-06-22", "2606.00001")),
    };
    const cache = new RecentDatesCache({
      getSettings: () => settingsWithCategories(["cs.CL"]),
      buildFetcher: () => fetcher,
      logger: { debug: vi.fn(), warn: vi.fn() },
      now: () => new Date("2026-06-23T01:00:00Z"),
    });
    await cache.refresh();

    const snapshot = cache.snapshot();
    snapshot.dates.add("2026-01-01");

    expect(cache.hasDate("2026-01-01")).toBe(false);
  });
});
```

- [ ] **Step 2: Run the service tests and verify they fail**

Run from `plugin/`:

```bash
npm test -- tests/services/recent-dates.test.ts
```

Expected: FAIL because `../../src/services/recent-dates` does not exist.

- [ ] **Step 3: Implement the recent date cache**

Create `plugin/src/services/recent-dates.ts`:

```typescript
import { parseRecent } from "../pipeline/arxiv-parser";
import type { ArxivFetcher } from "../pipeline/arxiv-fetcher";
import { arxivCategories } from "../settings/categories";
import type { PluginSettings } from "../settings/types";
import type { Logger } from "./logger";

type RecentFetcher = Pick<ArxivFetcher, "fetchRecent">;
type RecentLogger = Pick<Logger, "debug" | "warn">;

export type RecentDatesStatus = "idle" | "ready" | "failed";

export interface RecentDatesSnapshot {
  status: RecentDatesStatus;
  dates: Set<string>;
  refreshedAt: number;
  error?: string;
}

export interface RecentDatesCacheDeps {
  getSettings: () => PluginSettings;
  buildFetcher: () => RecentFetcher;
  logger: RecentLogger;
  now?: () => Date;
}

export class RecentDatesCache {
  private state: RecentDatesSnapshot = {
    status: "idle",
    dates: new Set(),
    refreshedAt: 0,
  };

  constructor(private readonly deps: RecentDatesCacheDeps) {}

  snapshot(): RecentDatesSnapshot {
    return cloneSnapshot(this.state);
  }

  hasDate(date: string): boolean {
    return this.state.dates.has(date);
  }

  async refresh(): Promise<RecentDatesSnapshot> {
    const settings = this.deps.getSettings();
    const categories = arxivCategories(settings.arxiv);
    const fetcher = this.deps.buildFetcher();
    const dates = new Set<string>();
    const errors: string[] = [];

    for (const category of categories) {
      try {
        const html = await fetcher.fetchRecent(category);
        const buckets = parseRecent(html);
        for (const bucket of buckets) dates.add(bucket.announceDate);
      } catch (e) {
        const message = (e as Error).message;
        errors.push(`${category}: ${message}`);
        this.deps.logger.warn(
          `recent dates refresh failed for ${category}: ${message}`,
        );
      }
    }

    const refreshedAt = (this.deps.now ?? (() => new Date()))().getTime();
    this.state =
      dates.size > 0
        ? {
            status: "ready",
            dates,
            refreshedAt,
            error: errors.length > 0 ? errors.join("; ") : undefined,
          }
        : {
            status: "failed",
            dates: new Set(),
            refreshedAt,
            error: errors.join("; ") || "no /recent dates found",
          };

    this.deps.logger.debug(
      `recent dates refreshed: ${this.state.status}, ${this.state.dates.size} dates`,
    );
    return this.snapshot();
  }
}

function cloneSnapshot(snapshot: RecentDatesSnapshot): RecentDatesSnapshot {
  return {
    status: snapshot.status,
    dates: new Set(snapshot.dates),
    refreshedAt: snapshot.refreshedAt,
    error: snapshot.error,
  };
}
```

- [ ] **Step 4: Run the service tests and verify they pass**

Run from `plugin/`:

```bash
npm test -- tests/services/recent-dates.test.ts
```

Expected: PASS, all `RecentDatesCache` tests pass.

---

### Task 2: Wire Recent Date Cache Into the Plugin

**Files:**
- Modify: `plugin/main.ts:16-112, 210-270`

- [ ] **Step 1: Write a type/build failure checkpoint**

Run from `plugin/` before editing:

```bash
npm run build
```

Expected: PASS before this task starts. This establishes that any later build failure comes from the wiring changes.

- [ ] **Step 2: Add plugin-level cache wiring**

Modify `plugin/main.ts` with these concrete changes:

```typescript
import { RecentDatesCache } from "./src/services/recent-dates";
```

Add the property next to the other service fields:

```typescript
  recentDates!: RecentDatesCache;
```

In `onload()`, after `this.host = buildObsidianHostAdapters(...)` and before scheduler construction, initialize the cache:

```typescript
    this.recentDates = new RecentDatesCache({
      getSettings: () => this.settings,
      buildFetcher: () => this.buildArxivFetcher(),
      logger: this.logger,
    });
```

Pass it into the scheduler construction:

```typescript
      recentDates: this.recentDates,
```

Extract fetcher construction so pipeline/manual fetch/recent cache share one configuration source:

```typescript
  buildArxivFetcher(): ArxivFetcher {
    return new ArxivFetcher({
      category: this.settings.arxiv.category,
      categories: arxivCategories(this.settings.arxiv),
      http: this.host.http,
      logger: this.logger,
      requestDelayMs: this.settings.advanced.requestDelayMs,
    });
  }
```

Then update `buildSharedDeps()` so its first fetcher line becomes:

```typescript
    const fetcher = this.buildArxivFetcher();
```

Remove the old inline `new ArxivFetcher({ ... })` block from `buildSharedDeps()`.

- [ ] **Step 3: Run the build and verify wiring compiles**

Run from `plugin/`:

```bash
npm run build
```

Expected: FAIL at this point if `SchedulerDeps` has not yet accepted `recentDates`. Continue to Task 5 for scheduler typing, or temporarily omit the scheduler property until Task 5 if executing strictly one task at a time.

---

### Task 3: Add Pure Dashboard Run Whitelist Helpers

**Files:**
- Modify: `plugin/src/dashboard/view.ts:1-180`
- Modify: `plugin/tests/dashboard/calendar-state.test.ts`

- [ ] **Step 1: Write failing whitelist tests**

Append these tests to `plugin/tests/dashboard/calendar-state.test.ts` after the existing `Calendar Cell Builder` tests:

```typescript
import {
  isCalendarRunWhitelisted,
  type CalendarRunWhitelistInput,
} from "../../src/dashboard/view";

function whitelistInput(
  overrides: Partial<CalendarRunWhitelistInput> = {},
): CalendarRunWhitelistInput {
  return {
    date: "2026-06-23",
    today: "2026-06-23",
    now: new Date("2026-06-23T03:00:00Z"),
    timezone: "Asia/Shanghai",
    runAtLocal: "09:00",
    runUntilLocal: "18:00",
    inLookback: true,
    isWeekend: false,
    hasDailyReport: false,
    recentDates: new Set(["2026-06-22"]),
    ...overrides,
  };
}

describe("isCalendarRunWhitelisted", () => {
  it("shows today only inside the Run window", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({ now: new Date("2026-06-23T03:00:00Z") }),
      ),
    ).toBe(true);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({ now: new Date("2026-06-23T00:30:00Z") }),
      ),
    ).toBe(false);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({ now: new Date("2026-06-23T11:00:00Z") }),
      ),
    ).toBe(false);
  });

  it("shows non-today dates only when latest /recent cache contains the date", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-23",
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(true);
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-19",
          today: "2026-06-23",
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(false);
  });

  it("requires no local daily report for both today and non-today", () => {
    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-23",
          hasDailyReport: true,
          recentDates: new Set(["2026-06-22"]),
        }),
      ),
    ).toBe(false);
  });

  it("blocks terminal and running states but allows transient failures", () => {
    for (const runState of [
      runState("running"),
      runState("skipped"),
      runState("failed_permanent"),
      runState("completed", { papersWritten: 0 }),
    ]) {
      expect(
        isCalendarRunWhitelisted(
          whitelistInput({
            date: "2026-06-22",
            today: "2026-06-23",
            recentDates: new Set(["2026-06-22"]),
            runState,
          }),
        ),
      ).toBe(false);
    }

    expect(
      isCalendarRunWhitelisted(
        whitelistInput({
          date: "2026-06-22",
          today: "2026-06-23",
          recentDates: new Set(["2026-06-22"]),
          runState: runState("failed_transient"),
        }),
      ),
    ).toBe(true);
  });
});
```

If `calendar-state.test.ts` already imports from `../../src/dashboard/view`, merge the new symbols into that existing import instead of creating a second import from the same module.

- [ ] **Step 2: Run the focused dashboard tests and verify they fail**

Run from `plugin/`:

```bash
npm test -- tests/dashboard/calendar-state.test.ts
```

Expected: FAIL because `isCalendarRunWhitelisted` and `CalendarRunWhitelistInput` are not exported.

- [ ] **Step 3: Implement the pure whitelist helper**

In `plugin/src/dashboard/view.ts`, export these helpers near `resolveCalendarCellState()`:

```typescript
export interface CalendarRunWhitelistInput {
  date: string;
  today: string;
  now: Date;
  timezone: string;
  runAtLocal: string;
  runUntilLocal: string;
  inLookback: boolean;
  isWeekend: boolean;
  hasDailyReport: boolean;
  recentDates: ReadonlySet<string>;
  runState?: RunStateEntry;
}

export function isCalendarRunWhitelisted(
  input: CalendarRunWhitelistInput,
): boolean {
  if (!input.inLookback) return false;
  if (input.isWeekend) return false;
  if (input.hasDailyReport) return false;
  if (isRunStateBlockedForCalendarRun(input.runState)) return false;

  if (input.date === input.today) {
    return isWithinLocalRunWindow(
      input.now,
      input.timezone,
      input.runAtLocal,
      input.runUntilLocal,
    );
  }

  return input.recentDates.has(input.date);
}

function isRunStateBlockedForCalendarRun(runState?: RunStateEntry): boolean {
  return (
    runState?.status === "running" ||
    runState?.status === "skipped" ||
    runState?.status === "failed_permanent" ||
    (runState?.status === "completed" && runState.papersWritten === 0)
  );
}

function isWithinLocalRunWindow(
  now: Date,
  timezone: string,
  runAtLocal: string,
  runUntilLocal: string,
): boolean {
  const currentMinutes = minutesSinceMidnight(now, timezone);
  const start = parseHHMM(runAtLocal);
  const end = parseHHMM(runUntilLocal);
  const startMinutes = start.hour * 60 + start.minute;
  const endMinutes = end.hour * 60 + end.minute;

  if (startMinutes <= endMinutes) {
    return currentMinutes >= startMinutes && currentMinutes <= endMinutes;
  }
  return currentMinutes >= startMinutes || currentMinutes <= endMinutes;
}
```

Do not use `schedule.enabled` in this helper; the approved display rule gates today on the Run window, not on whether automatic scheduling is enabled.

- [ ] **Step 4: Run the focused dashboard tests and verify they pass**

Run from `plugin/`:

```bash
npm test -- tests/dashboard/calendar-state.test.ts
```

Expected: PASS for the existing calendar-state tests and the new whitelist tests.

---

### Task 4: Use the Whitelist and Recent Cache in Dashboard Rendering and Actions

**Files:**
- Modify: `plugin/src/dashboard/view.ts:290-322, 904-980, 1053-1066, 1687-1705`
- Modify: `plugin/tests/dashboard/calendar-state.test.ts`

- [ ] **Step 1: Add a build-cells regression test for `/recent` gating**

Add a pure helper test to `plugin/tests/dashboard/calendar-state.test.ts` that verifies `resolveCalendarCellState()` still respects terminal state before runnable:

```typescript
describe("calendar whitelist and resolution together", () => {
  it("keeps non-recent non-today dates empty even when no report exists", () => {
    const runnable = isCalendarRunWhitelisted(
      whitelistInput({
        date: "2026-06-19",
        today: "2026-06-23",
        recentDates: new Set(["2026-06-22"]),
      }),
    );

    expect(
      resolveCalendarCellState({
        runnable,
        emptyReason: "arxiv-not-updated",
      }),
    ).toEqual({ state: "empty", emptyReason: "arxiv-not-updated" });
  });
});
```

- [ ] **Step 2: Run the dashboard test and verify it fails before dashboard integration if helper behavior is missing**

Run from `plugin/`:

```bash
npm test -- tests/dashboard/calendar-state.test.ts
```

Expected: PASS if Task 3 is complete. This test locks the projection behavior before wiring it into `buildCalendarCells()`.

- [ ] **Step 3: Refresh recent dates on dashboard reload**

Modify `reloadIndex()` in `plugin/src/dashboard/view.ts` so the try block starts with:

```typescript
      await this.plugin.recentDates.refresh();
      await this.clearRunStateForMissingDailyReports();
```

This makes dashboard open and the existing refresh button update the recent-date snapshot because both call `reloadIndex()`.

- [ ] **Step 4: Replace the existing `isRunnable()` body with whitelist input**

Change the method signature in `plugin/src/dashboard/view.ts`:

```typescript
  private isRunnable(
    date: string,
    runState: RunStateEntry | undefined,
    hasDailyReport: boolean,
  ): boolean {
    const parsed = parseCalendarDate(date);
    const today = this.todayDate();
    const settings = this.plugin.settings;
    return isCalendarRunWhitelisted({
      date,
      today,
      now: new Date(),
      timezone: settings.arxiv.timezone,
      runAtLocal: settings.schedule.runAtLocal,
      runUntilLocal: settings.schedule.runUntilLocal,
      inLookback: this.getLookbackDates().has(date),
      isWeekend: parsed ? isWeekendDate(parsed) : false,
      hasDailyReport,
      recentDates: this.plugin.recentDates.snapshot().dates,
      runState,
    });
  }
```

Update `buildCalendarCells()` to pass state and file presence:

```typescript
      const report = byDate.get(cellDate.date);
      const dateRunState = runState[cellDate.date];
      const resolution = resolveCalendarCellState({
        report,
        runnable: this.isRunnable(cellDate.date, dateRunState, Boolean(report)),
        runState: dateRunState,
        emptyReason: this.getCalendarEmptyReason(cellDate.date),
      });
```

- [ ] **Step 5: Make empty reason match `/recent` absence**

Update `getCalendarEmptyReason(date: string)`:

```typescript
  private getCalendarEmptyReason(date: string): CalendarEmptyReason {
    const parsed = parseCalendarDate(date);
    if (parsed && isWeekendDate(parsed)) return "arxiv-not-updated";
    if (date === this.todayDate()) return "arxiv-not-updated";
    if (this.getLookbackDates().has(date)) return "arxiv-not-updated";
    return "outside-lookback";
  }
```

This means a lookback weekday that is not confirmed by `/recent` labels as `arXiv 未更新`, not as a missing local report.

- [ ] **Step 6: Refresh recent dates before dashboard run actions**

In `runDateFromCalendar(date: string)`, after setup validation and before the notice, add:

```typescript
    await this.plugin.recentDates.refresh();
    if (date !== this.todayDate() && !this.plugin.recentDates.hasDate(date)) {
      new Notice(`arXiv Daily ${date}: arXiv 未更新`);
      await this.reloadIndex();
      return;
    }
```

In `runToday()`, after `if (!this.gateFilter()) return;`, add:

```typescript
    await this.plugin.recentDates.refresh();
```

In `runAllPending()`, after `if (!this.gateFilter()) return;`, add:

```typescript
    await this.plugin.recentDates.refresh();
```

- [ ] **Step 7: Run focused dashboard tests and build**

Run from `plugin/`:

```bash
npm test -- tests/dashboard/calendar-state.test.ts
npm run build
```

Expected: PASS. Build should fail only if plugin typing for `recentDates` or scheduler deps is incomplete; finish Task 5 before final validation.

---

### Task 5: Apply Run Window End Time and Cache Refresh to Scheduler

**Files:**
- Modify: `plugin/src/services/scheduler.ts:1-180, 227-270`
- Modify: `plugin/tests/scheduler.test.ts`

- [ ] **Step 1: Add failing scheduler tests**

Append these tests to `plugin/tests/scheduler.test.ts` inside `describe("SchedulerService", () => { ... })`:

```typescript
  it("does not run scheduled polling after runUntilLocal", async () => {
    const store = makeStore();
    await store.load();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          enabled: true,
          runAtLocal: "09:00",
          runUntilLocal: "18:00",
        },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T11:01:00Z"), // 19:01 Shanghai
    });

    await svc.tick();

    expect(runForDate).not.toHaveBeenCalled();
  });

  it("refreshes recent dates when scheduled polling wakes inside the run window", async () => {
    const store = makeStore();
    await store.load();
    const recentDates = { refresh: vi.fn(async () => undefined) };
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          enabled: true,
          runAtLocal: "09:00",
          runUntilLocal: "18:00",
        },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"), // 13:00 Shanghai
      recentDates,
    });

    await svc.tick();

    expect(recentDates.refresh).toHaveBeenCalledTimes(1);
  });
```

- [ ] **Step 2: Run the focused scheduler tests and verify they fail**

Run from `plugin/`:

```bash
npm test -- tests/scheduler.test.ts
```

Expected: FAIL because `SchedulerDeps` does not accept `recentDates`, and because `tick()` does not yet check `runUntilLocal`.

- [ ] **Step 3: Extend scheduler deps and window helpers**

In `plugin/src/services/scheduler.ts`, add an interface near `SchedulerDeps`:

```typescript
export interface SchedulerRecentDates {
  refresh: () => Promise<unknown>;
  hasDate?: (date: string) => boolean;
}
```

Add to `SchedulerDeps`:

```typescript
  recentDates?: SchedulerRecentDates;
```

Add helper functions near the bottom of the file:

```typescript
function localMinutesFromHHMM(value: string): number {
  const parsed = parseHHMM(value);
  return parsed.hour * 60 + parsed.minute;
}

function isWithinRunWindow(
  minutesNow: number,
  startMinutes: number,
  endMinutes: number,
): boolean {
  if (startMinutes <= endMinutes) {
    return minutesNow >= startMinutes && minutesNow <= endMinutes;
  }
  return minutesNow >= startMinutes || minutesNow <= endMinutes;
}
```

- [ ] **Step 4: Gate scheduled polling by the full Run window and refresh cache**

In `tick()`, replace the start-time-only code:

```typescript
    const t = parseHHMM(s.schedule.runAtLocal);
    const scheduledMin = t.hour * 60 + t.minute;
```

with:

```typescript
    const scheduledMin = localMinutesFromHHMM(s.schedule.runAtLocal);
    const endMin = localMinutesFromHHMM(s.schedule.runUntilLocal);
    if (!isWithinRunWindow(minutesNow, scheduledMin, endMin)) {
      this.progress.setIdle(this.latestCompleted());
      return;
    }
    await this.deps.recentDates?.refresh();
```

Then keep the loop, but pass the same gate shape:

```typescript
        timeGate: isToday ? { scheduledMin, endMin, minutesNow } : undefined,
```

Update `tickDate()` option type:

```typescript
      timeGate?: { scheduledMin: number; endMin: number; minutesNow: number };
```

Replace the old time gate check:

```typescript
    if (opts.timeGate && opts.timeGate.minutesNow < opts.timeGate.scheduledMin) {
      return undefined;
    }
```

with:

```typescript
    if (
      opts.timeGate &&
      !isWithinRunWindow(
        opts.timeGate.minutesNow,
        opts.timeGate.scheduledMin,
        opts.timeGate.endMin,
      )
    ) {
      return undefined;
    }
```

In `tickTodayScheduled()`, replace the start-time-only parsing with:

```typescript
    const scheduledMin = localMinutesFromHHMM(s.schedule.runAtLocal);
    const endMin = localMinutesFromHHMM(s.schedule.runUntilLocal);
    if (!isWithinRunWindow(minutesNow, scheduledMin, endMin)) {
      this.progress.setIdle(this.latestCompleted());
      return { kind: "skipped", reason: "outside run window" };
    }
    await this.deps.recentDates?.refresh();
```

And pass:

```typescript
      timeGate: { scheduledMin, endMin, minutesNow },
```

- [ ] **Step 5: Keep run-all-pending aligned with confirmed `/recent` dates**

In `runAllPending()`, refresh once before the loop:

```typescript
    await this.deps.recentDates?.refresh();
```

Inside the loop, after `const date = formatDate(daysBefore(todayObj, i));`, add:

```typescript
      const isToday = date === formatDate(todayObj);
      if (!isToday && this.deps.recentDates?.hasDate && !this.deps.recentDates.hasDate(date)) {
        continue;
      }
```

This prevents the bulk dashboard action from attempting non-today dates that the latest cache does not confirm.

- [ ] **Step 6: Run scheduler tests**

Run from `plugin/`:

```bash
npm test -- tests/scheduler.test.ts
```

Expected: PASS for scheduler tests.

---

### Task 6: Full Validation and Trellis Notes

**Files:**
- Modify: `.trellis/tasks/06-23-plugin-settings-model-controls/design.md`
- Modify: `.trellis/tasks/06-23-plugin-settings-model-controls/prd.md`

- [ ] **Step 1: Update Trellis task notes with the approved Run whitelist**

Append a short section to `.trellis/tasks/06-23-plugin-settings-model-controls/design.md`:

```markdown
## Dashboard Run Whitelist Follow-up

Dashboard `Run` display is a whitelist:

- Today can show `Run` only on a workday inside `schedule.runAtLocal` through
  `schedule.runUntilLocal`, with no local daily report and no blocking run
  state.
- Non-today dates can show `Run` only when they are in the lookback window,
  not a weekend, present in the latest `/recent` date cache, have no local daily
  report, and have no blocking run state.
- Blocking run states are `running`, `skipped`, `failed_permanent`, and
  `completed` with `papersWritten === 0`.
- `failed_transient`, `pending`, and missing run state remain runnable when the
  date otherwise satisfies the whitelist.
- For multiple configured arXiv categories, a date is treated as present in
  `/recent` when any configured category contains that announce date.
```

Append matching acceptance criteria to `.trellis/tasks/06-23-plugin-settings-model-controls/prd.md`:

```markdown
- [ ] Dashboard calendar `Run` display uses the approved whitelist for today and
      non-today dates.
- [ ] Non-today dashboard `Run` cells require the latest `/recent` date cache,
      using a union across configured categories.
- [ ] Dashboard open, dashboard refresh, and dashboard run actions refresh the
      recent-date cache without querying arXiv on every cell render.
- [ ] Scheduler automatic polling respects both `runAtLocal` and
      `runUntilLocal`.
```

- [ ] **Step 2: Run focused validation**

Run from `plugin/`:

```bash
npm test -- tests/services/recent-dates.test.ts tests/dashboard/calendar-state.test.ts tests/scheduler.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run full plugin validation**

Run from `plugin/`:

```bash
npm test
npm run build
```

Expected: PASS for all tests and the production build.

- [ ] **Step 4: Check formatting-sensitive diff issues**

Run from repository root:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Review the touched files**

Run from repository root:

```bash
git diff -- plugin/src/services/recent-dates.ts plugin/src/dashboard/view.ts plugin/src/services/scheduler.ts plugin/main.ts plugin/tests/services/recent-dates.test.ts plugin/tests/dashboard/calendar-state.test.ts plugin/tests/scheduler.test.ts .trellis/tasks/06-23-plugin-settings-model-controls/design.md .trellis/tasks/06-23-plugin-settings-model-controls/prd.md
```

Expected: diff shows only recent-date cache, dashboard Run whitelist, scheduler Run window/cache integration, tests, and Trellis task notes. The manually adjusted calendar play icon CSS must not appear in this diff.

---

## Self-Review

- Spec coverage: The plan covers today Run display, non-today `/recent` gating, multi-category union behavior, cache refresh timing, terminal run states, transient retry, scheduler Run window usage, tests, and Trellis documentation.
- Placeholder scan: No placeholder tasks are left; each implementation step includes concrete file paths, code, commands, and expected results.
- Type consistency: `RecentDatesCache`, `RecentDatesSnapshot`, `CalendarRunWhitelistInput`, `isCalendarRunWhitelisted`, and `SchedulerRecentDates` names are used consistently across service, dashboard, scheduler, and tests.
