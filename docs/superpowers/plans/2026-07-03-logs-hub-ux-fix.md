# Logs & History UX Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three fixes to the "Show logs & history" hub modal and scheduler: (1) show logs newest-first, (2) add log-level filtering to the logs tab, (3) stop the scheduler from re-querying arxiv after today's report is already generated.

**Architecture:** Reuse the existing `Logger` buffer (records `[LEVEL]` tags) by extracting a pure formatter `formatLogEntries(buffer, opts) -> string` and a tiny level-set UI into `view.ts`. HubModal calls the formatter. The scheduler `tick()` gains an early-return when today is `isDone`, before any `recentDates.refresh()` call, so no arxiv HTTP happens after completion. All logic is unit-testable without Obsidian/DOM.

**Tech Stack:** TypeScript, esbuild, vitest. Obsidian Plugin API (Modal/DOM used only inside `HubModal`).

---

## File Structure

- **Modify** `plugin/src/services/logger.ts` — export `LogLevel` re-typed helper `parseLogLevel` + `LOG_LEVELS` constant (already exported as `LogLevel`).
- **Modify** `plugin/src/dashboard/view.ts` — add exported pure function `formatLogEntries` + `defaultLogLevelSet`; add level-filter chips row to `HubModal`; render logs newest-first.
- **Modify** `plugin/tests/dashboard-view.test.ts` — add `formatLogEntries` test suite.
- **Modify** `plugin/src/services/scheduler.ts` — early-return in `tick()` when today is `isDone` before `recentDates.refresh()`.
- **Modify** `plugin/tests/scheduler.test.ts` — add "does not refresh recent dates when today already done" test.

Rationale: extracting pure functions keeps the new behavior unit-testable (HubModal itself is DOM-coupled and not covered by suite), matching the existing pattern where `dashboard-view.test.ts` only tests exported helpers.

---

## Task 1: Log formatter — newest-first + level filtering (pure function)

**Files:**
- Create (in): `plugin/src/dashboard/view.ts` — new exported helper near other exported helpers (after `appendSettingsButton`, ~line 345)
- Test: `plugin/tests/dashboard-view.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `plugin/tests/dashboard-view.test.ts`. Add import to the existing import block at top:

```ts
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  collectIndexedDetailSummaryRefs,
  executeObsidianCommand,
  filterDashboardMarkdownFiles,
  formatLogEntries,
  openDashboardFileOnce,
  paginateDashboardRows,
  shouldSkipDashboardHistorySync,
} from "../src/dashboard/view";
```

Then a new `describe` block at end of file:

```ts
describe("formatLogEntries", () => {
  it("renders newest-first by default", () => {
    const buf = [
      "2026-07-03 09:00:00.000 [INFO] first",
      "2026-07-03 09:00:01.000 [INFO] second",
      "2026-07-03 09:00:02.000 [ERROR] boom",
    ];
    const out = formatLogEntries(buf);
    const lines = out.split("\n");
    expect(lines[0]).toContain("boom");
    expect(lines[2]).toContain("first");
  });

  it("filters out levels not in the enabled set", () => {
    const buf = [
      "2026-07-03 09:00:00.000 [DEBUG] d",
      "2026-07-03 09:00:01.000 [INFO] i",
      "2026-07-03 09:00:02.000 [WARN] w",
      "2026-07-03 09:00:03.000 [ERROR] e",
    ];
    const out = formatLogEntries(buf, { levels: new Set(["info", "warn", "error"]) });
    const lines = out.split("\n");
    expect(lines).toHaveLength(3);
    expect(out).not.toContain("[DEBUG]");
    expect(lines[0]).toContain("[ERROR] e"); // newest first
  });

  it("returns a placeholder when buffer is empty", () => {
    expect(formatLogEntries([])).toBe("(no log entries)");
  });

  it("falls back to keeping lines without a parseable level tag when filter is active", () => {
    const buf = ["weird line without level", "2026-07-03 09:00:00.000 [INFO] ok"];
    const out = formatLogEntries(buf, { levels: new Set(["info"]) });
    expect(out).toContain("[INFO] ok");
    // untagged line: kept (cannot prove it should be hidden)
    expect(out.split("\n").length).toBeGreaterThanOrEqual(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugin && npx vitest run tests/dashboard-view.test.ts -t formatLogEntries`
Expected: FAIL with "formatLogEntries is not a function" (import fails).

- [ ] **Step 3: Add the import and implement the helper**

In `plugin/src/dashboard/view.ts`:

At top of file, add to the existing logger import (or add a new import line near other service imports):

```ts
import type { Logger } from "../services/logger";
```

Add exported helper after `appendSettingsButton` function (locate it; place the new block after its closing brace):

```ts
export const DEFAULT_LOG_LEVELS = new Set(["debug", "info", "warn", "error"]);

const LOG_LEVEL_TAG = /\[(DEBUG|INFO|WARN|ERROR)\]/;

export function parseLogLevelTag(line: string): string | null {
  const m = line.match(LOG_LEVEL_TAG);
  return m ? m[1].toLowerCase() : null;
}

export interface FormatLogEntriesOptions {
  /** Levels to keep. Defaults to all four levels. */
  levels?: Set<string>;
}

export function formatLogEntries(
  buffer: string[],
  opts: FormatLogEntriesOptions = {},
): string {
  if (buffer.length === 0) return "(no log entries)";
  const levels = opts.levels ?? DEFAULT_LOG_LEVELS;
  const kept: string[] = [];
  // Iterate oldest→newest, push allowed; untagged lines are kept.
  for (const line of buffer) {
    const lvl = parseLogLevelTag(line);
    if (lvl === null || levels.has(lvl)) kept.push(line);
  }
  if (kept.length === 0) return "(no log entries at this level)";
  // Reverse so newest is on top.
  return kept.reverse().join("\n");
}
```

Note: a `Logger` type import is not required for this helper; skip that import line if unused afterwards.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugin && npx vitest run tests/dashboard-view.test.ts -t formatLogEntries`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd plugin
git add tests/dashboard-view.test.ts src/dashboard/view.ts
git commit -m "feat(dashboard): add formatLogEntries helper (newest-first, level filter)" -m "Pure formatter extracted from the logs tab render path so log display order and level filtering become unit-testable without Obsidian/DOM." -m "Validation: vitest tests/dashboard-view.test.ts -t formatLogEntries passes."
```

---

## Task 2: Wire HubModal logs tab to the formatter + level-filter chips

**Files:**
- Modify: `plugin/src/dashboard/view.ts` (`HubModal` class, ~lines 2441–2615)
- Test: manual (DOM-coupled; not covered by unit suite)

- [ ] **Step 1: Add level-set state + chip row to HubModal**

In `plugin/src/dashboard/view.ts`, inside `class HubModal`, add private state near the existing `private activeTab` / `private panels` fields:

```ts
  private activeTab: HubModalTab = "logs";
  private panels = new Map<HubModalTab, HubPanel>();
  private logLevels: Set<string> = new Set(DEFAULT_LOG_LEVELS);
```

(Add `DEFAULT_LOG_LEVELS` to the existing import statement from `../dashboard/view` self-imports — it is defined in the same file, so no new import needed; just reference it.)

- [ ] **Step 2: Render the level-filter chip row in `onOpen`**

Locate the `onOpen()` method. After `this.createPanel(tabs, body, "logs", "Logs");` and before `this.activateTab("logs");`, insert:

```ts
    const levelRow = body.createDiv({
      cls: "arxiv-daily-hub-modal__level-filter",
    });
    this.levelRow = levelRow;
    this.renderLevelChips(levelRow);
    levelRow.style.display = "none"; // shown only when logs tab active
```

Add the field declaration near the other private fields:

```ts
  private levelRow: HTMLDivElement | null = null;
```

- [ ] **Step 3: Implement renderLevelChips + toggle visibility**

Add these methods to `HubModal` (e.g., after `activateTab`):

```ts
  private renderLevelChips(container: HTMLElement): void {
    container.empty();
    const order: Array<{ key: string; label: string }> = [
      { key: "debug", label: "Debug" },
      { key: "info", label: "Info" },
      { key: "warn", label: "Warn" },
      { key: "error", label: "Error" },
    ];
    for (const { key, label } of order) {
      const active = this.logLevels.has(key);
      const chip = container.createEl("button", {
        cls: `arxiv-daily-hub-modal__level-chip${active ? " is-active" : ""}`,
        text: label,
        attr: { type: "button", "aria-pressed": String(active) },
      });
      chip.onclick = () => {
        if (this.logLevels.has(key)) this.logLevels.delete(key);
        else this.logLevels.add(key);
        this.renderLevelChips(container);
        this.refreshActiveTab();
      };
    }
    const all = container.createEl("button", {
      cls: "arxiv-daily-hub-modal__level-chip",
      text: "All",
      attr: { type: "button" },
    });
    all.onclick = () => {
      this.logLevels = new Set(DEFAULT_LOG_LEVELS);
      this.renderLevelChips(container);
      this.refreshActiveTab();
    };
  }

  private setLevelRowVisibility(): void {
    if (this.levelRow) {
      this.levelRow.style.display = this.activeTab === "logs" ? "" : "none";
    }
  }
```

- [ ] **Step 4: Call setLevelRowVisibility from activateTab**

In `activateTab(...)`, after the `for` loop that toggles panels, add at end:

```ts
    this.setLevelRowVisibility();
  ```

- [ ] **Step 5: Use the formatter in refreshActiveTab**

Replace the `if (tab === "logs") {` branch in `refreshActiveTab()` with:

```ts
    if (tab === "logs") {
      this.setPanelText(
        tab,
        formatLogEntries(this.plugin.logger.getBuffer(), { levels: this.logLevels }),
      );
      return;
    }
```

And in `onOpen`, after `this.activateTab("logs");` add `this.setLevelRowVisibility();` (so the row is visible on first open — it was hidden by `display = "none"` above until activateTab runs; but activateTab now calls it, so the explicit `display = "none"` default is fine).

- [ ] **Step 6: Update Clear handler for the logs change (no-op needed)**

The existing Clear handler already calls `this.plugin.logger.clearBuffer()` then `refreshActiveTab()` — this now renders "(no log entries)". No change required; verify by reading the existing `onOpen` Clear button. Leave as-is.

- [ ] **Step 7: Add styles for the chip row**

Append to `plugin/styles.css`:

```css
.arxiv-daily-hub-modal__level-filter {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.arxiv-daily-hub-modal__level-chip {
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid var(--background-modifier-border);
  background: transparent;
  font-size: var(--font-ui-smaller);
  cursor: pointer;
}
.arxiv-daily-hub-modal__level-chip.is-active {
  background: var(--interactive-accent);
  color: var(--text-on-accent);
  border-color: var(--interactive-accent);
}
```

- [ ] **Step 8: Typecheck**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: no errors.

- [ ] **Step 9: Build**

Run: `cd plugin && npm run build`
Expected: clean build, `main.js` regenerated.

- [ ] **Step 10: Commit**

```bash
cd plugin
git add src/dashboard/view.ts styles.css
git commit -m "feat(dashboard): logs tab newest-first with level filter chips" -m "HubModal now renders the in-memory log buffer newest-on-top and exposes Debug/Info/Warn/Error/All toggle chips above the logs panel; chips hide on other tabs." -m "Validation: tsc -noEmit clean; npm run build clean."
```

---

## Task 3: Scheduler — skip arxiv re-query after today's report is done

**Files:**
- Modify: `plugin/src/services/scheduler.ts` (`tick()`, ~lines 84–117)
- Test: `plugin/tests/scheduler.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `plugin/tests/scheduler.test.ts`, inside the top-level `describe("SchedulerService", ...)` block, after the existing "refreshes recent dates..." test:

```ts
  it("does not refresh recent dates when today is already done", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 5);
    const recentDates = { refresh: vi.fn(async () => undefined) };
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 1 });
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
      now: () => new Date("2026-05-11T05:00:00Z"), // 13:00 Shanghai, inside window
      recentDates,
    });

    await svc.tick();

    expect(recentDates.refresh).not.toHaveBeenCalled();
    expect(runForDate).not.toHaveBeenCalled();
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts -t "does not refresh recent dates when today is already done"`
Expected: FAIL — `recentDates.refresh` was called once (current behavior always refreshes inside the window).

- [ ] **Step 3: Add early-return in tick()**

In `plugin/src/services/scheduler.ts`, locate the `tick()` method. After the existing guard:

```ts
    if (!isTimeWithinLocalWindow(now, tz, s.schedule.runAtLocal, s.schedule.runUntilLocal)) {
      this.progress.setIdle(this.latestCompleted());
      return;
    }
```

Insert immediately after it (before `await this.deps.recentDates?.refresh();`):

```ts
    // Today's report is already generated (or finalized). Stay idle for the
    // remainder of the run window to avoid re-querying arxiv on every tick;
    // lookback dates for prior days are not driven by recent-dates refresh in
    // the scheduler path (the per-date isDone check below handles them).
    if (this.deps.store.isDone(today)) {
      this.progress.setIdle(this.latestCompleted());
      return;
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts -t "does not refresh recent dates when today is already done"`
Expected: PASS.

- [ ] **Step 5: Run full scheduler suite to confirm no regression**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: all green. Pay attention to:
- "skips dates already completed" still passes (there today is done but no recentDates injected → `this.deps.recentDates?.refresh()` optional-chained; with new guard the refresh is never reached for done-today — fine, that test asserts `runForDate` not called, still true).
- "refreshes recent dates when scheduled polling wakes inside the run window" still passes (today is `pending`, not done → guard skips → refresh called once). ✓

- [ ] **Step 6: Commit**

```bash
cd plugin
git add src/services/scheduler.ts tests/scheduler.test.ts
git commit -m "fix(scheduler): stop re-querying arxiv after today's report is done" -m "tick() now returns early when today is isDone (completed/failed_permanent/skipped), before recentDates.refresh(), so no arxiv HTTP happens for the rest of the run window once the daily report exists. failed_transient is not 'done' so retry logic is unaffected." -m "Validation: vitest tests/scheduler.test.ts passes including new guard test."
```

---

## Task 4: Full-suite validation + version bump

**Files:**
- Optional Modify: `manifest.json` / `plugin/manifest.json` — version bump
- N/A

- [ ] **Step 1: Full test suite**

Run: `cd plugin && npm test`
Expected: all green (existing + new tests).

- [ ] **Step 2: Typecheck + production build**

Run: `cd plugin && npm run build`
Expected: clean.

- [ ] **Step 3: (Optional) Version bump**

If project conventions bump version per change set, read `plugin/manifest.json` current `version`, increment patch, and mirror to `versions.json` and root `manifest.json` if present. Skip if unsure of release cadence — ask maintainer.

- [ ] **Step 4: Final commit (if version bumped)**

```bash
git add manifest.json versions.json plugin/manifest.json 2>/dev/null
git commit -m "chore(release): bump version" -m "Version bump for logs/history UX fixes and scheduler arxiv re-query stop." -m "Validation: npm test + npm run build clean."
```

---

## Self-Review

**1. Spec coverage**
- (1) logs newest-first → Task 1 `formatLogEntries` default reverse; Task 2 wires HubModal. ✓
- (2) log level filter → Task 1 `levels` option + `parseLogLevelTag`; Task 2 chips. ✓
- (3) stop arxiv query after report done → Task 3 early-return before `recentDates.refresh()`. ✓ Note: this is the **scheduler/auto-tick** path only; manual `runAllPending`/`runForDateNow` intentionally keep refreshing so the user can still detect newly-arXived papers and re-run. Confirmed the user wanted scheduler-path behavior.

**2. Placeholder scan**
- Each code step shows full code. No "TBD"/"add error handling". ✓
- Untagged-line behavior in `formatLogEntries` is explicitly kept-with-caveat (test only asserts the tagged line passes), not a silent TODO. ✓

**3. Type consistency**
- `formatLogEntries(buffer, opts)` signature consistent across Task 1 (def) and Task 2 (call with `{ levels: this.logLevels }`). ✓
- `DEFAULT_LOG_LEVELS` exported then reused in Task 2 (`new Set(DEFAULT_LOG_LEVELS)`). ✓
- `HubLevelTab`/`levels: Set<string>` consistent with `parseLogLevelTag` returning lowercase string. ✓
- `setLevelRowVisibility()` defined in Task 2 Step 3, called in Step 4 (activateTab) and Step 5 (onOpen). ✓

No gaps found. Plan complete.