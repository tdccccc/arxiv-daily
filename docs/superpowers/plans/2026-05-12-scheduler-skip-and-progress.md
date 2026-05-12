# Scheduler enable-gating, skip-existing, status-bar progress — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop fresh installs from auto-summarizing 5 days at once, stop overwriting existing daily/paper files, and surface live pipeline progress on the Obsidian status bar.

**Architecture:** Three coupled but independently-testable blocks. Block 1 reshapes the boot flow: default `schedule.enabled` to `false`, route a new `setScheduleEnabled()` method through ribbon + settings, scope the on-load tick to today (with weekend skip). Block 2 makes `MarkdownWriter` strict about pre-existing files and pushes existence-checks up into the pipeline so skipped dates cost zero LLM calls. Block 3 threads a small `ProgressReporter` callback through the scheduler and pipeline; a `StatusBarController` renders a one-line summary to Obsidian's status bar.

**Tech Stack:** TypeScript, esbuild, vitest with happy-dom, Obsidian plugin API. `obsidian` package is mocked in tests via `plugin/tests/__mocks__/obsidian.ts`.

**Spec:** `docs/superpowers/specs/2026-05-12-scheduler-skip-and-progress-design.md`

---

## Task 1: Weekend detection helper

**Files:**
- Modify: `plugin/src/utils/time.ts`
- Test: `plugin/tests/time.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `plugin/tests/time.test.ts` (inside the existing `describe("time utils", ...)` block):

```ts
  it("isWeekendInTz returns true for Saturday Shanghai", () => {
    // 2026-05-09 is a Saturday
    const d = new Date("2026-05-09T05:00:00Z"); // 13:00 Shanghai, Sat
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(true);
  });

  it("isWeekendInTz returns true for Sunday Shanghai", () => {
    // 2026-05-10 is a Sunday
    const d = new Date("2026-05-10T05:00:00Z");
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(true);
  });

  it("isWeekendInTz returns false for Monday Shanghai", () => {
    // 2026-05-11 is a Monday
    const d = new Date("2026-05-11T05:00:00Z");
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(false);
  });

  it("isWeekendInTz handles UTC-day-flip", () => {
    // 2026-05-09T18:00Z is 2026-05-10 (Sun) Shanghai
    const d = new Date("2026-05-09T18:00:00Z");
    expect(isWeekendInTz(d, "Asia/Shanghai")).toBe(true);
    // Same instant in UTC is still Sat
    expect(isWeekendInTz(d, "UTC")).toBe(true);
  });
```

And add `isWeekendInTz` to the import line at the top of the file:

```ts
import {
  todayInTz,
  formatDate,
  parseHHMM,
  minutesSinceMidnight,
  daysBefore,
  isWeekendInTz,
} from "../src/utils/time";
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd plugin && npx vitest run tests/time.test.ts`
Expected: FAIL with "isWeekendInTz is not exported" or similar.

- [ ] **Step 3: Implement the helper**

Append to `plugin/src/utils/time.ts`:

```ts
export function isWeekendInTz(now: Date, tz: string): boolean {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: tz,
    weekday: "short",
  });
  const weekday = fmt.format(now);
  return weekday === "Sat" || weekday === "Sun";
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd plugin && npx vitest run tests/time.test.ts`
Expected: PASS (all assertions green).

- [ ] **Step 5: Commit**

```bash
git add plugin/src/utils/time.ts plugin/tests/time.test.ts
git commit -m "feat(plugin): add isWeekendInTz helper"
```

---

## Task 2: MarkdownWriter existence-check methods

**Files:**
- Modify: `plugin/src/pipeline/markdown-writer.ts`
- Create: `plugin/tests/markdown-writer.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `plugin/tests/markdown-writer.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { MarkdownWriter } from "../src/pipeline/markdown-writer";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function makeVault(initialFiles: Record<string, string> = {}) {
  const files: Record<string, string> = { ...initialFiles };
  return {
    files,
    vault: {
      adapter: {
        async write(path: string, content: string) {
          files[path] = content;
        },
        async exists(path: string) {
          return Object.prototype.hasOwnProperty.call(files, path);
        },
        async mkdir(_path: string) {},
        async rename(from: string, to: string) {
          files[to] = files[from];
          delete files[from];
        },
        async remove(path: string) {
          delete files[path];
        },
      },
    } as any,
  };
}

function makeWriter(initialFiles: Record<string, string> = {}) {
  const { files, vault } = makeVault(initialFiles);
  const writer = new MarkdownWriter({
    vault,
    logger: new Logger("error"),
    arxiv: DEFAULT_SETTINGS.arxiv,
    output: DEFAULT_SETTINGS.output,
  });
  return { files, writer };
}

describe("MarkdownWriter existence checks", () => {
  it("dailyExists returns false when daily missing", async () => {
    const { writer } = makeWriter();
    expect(await writer.dailyExists("2026-05-11")).toBe(false);
  });

  it("dailyExists returns true when daily present", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    expect(await writer.dailyExists("2026-05-11")).toBe(true);
  });

  it("paperDetailExists returns false when paper missing", async () => {
    const { writer } = makeWriter();
    expect(await writer.paperDetailExists("2605.06587")).toBe(false);
  });

  it("paperDetailExists returns true when paper present", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/papers/2605.06587.md": "x",
    });
    expect(await writer.paperDetailExists("2605.06587")).toBe(true);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd plugin && npx vitest run tests/markdown-writer.test.ts`
Expected: FAIL with "dailyExists is not a function" or similar.

- [ ] **Step 3: Add the methods**

Add to `plugin/src/pipeline/markdown-writer.ts` inside the `MarkdownWriter` class (before `private tagsFor`):

```ts
  async dailyExists(dateStr: string): Promise<boolean> {
    const path = normalizePath(`${this.opts.output.dailyDir}/${dateStr}.md`);
    return await this.opts.vault.adapter.exists(path);
  }

  async paperDetailExists(id: string): Promise<boolean> {
    const path = normalizePath(`${this.opts.output.papersDir}/${id}.md`);
    return await this.opts.vault.adapter.exists(path);
  }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd plugin && npx vitest run tests/markdown-writer.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/pipeline/markdown-writer.ts plugin/tests/markdown-writer.test.ts
git commit -m "feat(plugin): MarkdownWriter.dailyExists / paperDetailExists"
```

---

## Task 3: MarkdownWriter — replace backupIfExists with throw-on-exists

**Files:**
- Modify: `plugin/src/pipeline/markdown-writer.ts`
- Modify: `plugin/tests/markdown-writer.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `plugin/tests/markdown-writer.test.ts`:

```ts
describe("MarkdownWriter strictness on existing files", () => {
  it("writeDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeDaily("2026-05-11", "new")).rejects.toThrow(
      /already exists/,
    );
  });

  it("writePaperDetail throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/papers/2605.06587.md": "x",
    });
    const paper = {
      id: "2605.06587",
      title: "T",
      authors: "A",
      abstract: "",
      category: "photo-z",
      isDetail: true,
      abstractConclusion: "",
      fullSections: null,
    };
    await expect(writer.writePaperDetail(paper as any, "2026-05-11", "x"))
      .rejects.toThrow(/already exists/);
  });

  it("writeEmptyDaily throws if file already exists", async () => {
    const { writer } = makeWriter({
      "arxiv-daily/daily/2026-05-11.md": "x",
    });
    await expect(writer.writeEmptyDaily("2026-05-11")).rejects.toThrow(
      /already exists/,
    );
  });

  it("writeDaily writes content (no bak file produced)", async () => {
    const { files, writer } = makeWriter();
    await writer.writeDaily("2026-05-11", "body");
    expect(files["arxiv-daily/daily/2026-05-11.md"]).toContain("body");
    expect(files["arxiv-daily/daily/2026-05-11.bak.md"]).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd plugin && npx vitest run tests/markdown-writer.test.ts`
Expected: FAIL — old code silently renames-and-overwrites instead of throwing.

- [ ] **Step 3: Replace backupIfExists with throw**

In `plugin/src/pipeline/markdown-writer.ts`:

Replace `backupIfExists` (private method at end of class) and the three call sites. The final file should have these three write methods:

```ts
  async writeDaily(dateStr: string, summary: string): Promise<string> {
    const path = normalizePath(`${this.opts.output.dailyDir}/${dateStr}.md`);
    await this.ensureDir(this.opts.output.dailyDir);
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`daily already exists: ${path}`);
    }
    const frontmatter = `---\ndate: ${dateStr}\ntags: [arxiv, daily]\n---\n\n`;
    await this.opts.vault.adapter.write(path, frontmatter + summary);
    this.opts.logger.info(`wrote daily: ${path}`);
    return path;
  }

  async writePaperDetail(
    paper: DailyPaperWithContent,
    dateStr: string,
    summary: string,
  ): Promise<string> {
    const path = normalizePath(`${this.opts.output.papersDir}/${paper.id}.md`);
    await this.ensureDir(this.opts.output.papersDir);
    if (await this.opts.vault.adapter.exists(path)) {
      throw new Error(`paper already exists: ${path}`);
    }
    const tags = this.tagsFor(paper);
    const fm =
      `---\n` +
      `title: "${escapeYaml(paper.title)}"\n` +
      `authors: "${escapeYaml(paper.authors)}"\n` +
      `arxiv: "${paper.id}"\n` +
      `date: ${dateStr}\n` +
      `tags: [${tags.join(", ")}]\n` +
      `---\n\n`;
    await this.opts.vault.adapter.write(path, fm + summary);
    this.opts.logger.info(`wrote paper: ${path}`);
    return path;
  }

  async writeEmptyDaily(dateStr: string): Promise<string> {
    const summary = `# arXiv ${this.opts.arxiv.category} 每日追踪 ${dateStr}\n\n今日未发现相关论文。\n`;
    return this.writeDaily(dateStr, summary);
  }
```

Delete the `backupIfExists` private method entirely.

- [ ] **Step 4: Run the full test suite**

Run: `cd plugin && npm test`
Expected: PASS — markdown-writer tests green, no other tests broken. (Pipeline tests should still pass because pipeline currently writes to a fresh stubbed writer.)

- [ ] **Step 5: Commit**

```bash
git add plugin/src/pipeline/markdown-writer.ts plugin/tests/markdown-writer.test.ts
git commit -m "feat(plugin): writers throw on pre-existing files; drop silent backup"
```

---

## Task 4: Pipeline — daily-exists pre-check

**Files:**
- Modify: `plugin/src/pipeline/pipeline.ts`
- Modify: `plugin/tests/pipeline.test.ts`

- [ ] **Step 1: Write the failing test**

Append to the `describe("ArxivPipeline", ...)` block in `plugin/tests/pipeline.test.ts`:

```ts
  it("short-circuits with completed when daily file already exists", async () => {
    const d = makeDeps();
    // Stub the writer to claim the daily exists.
    (d.writer as any).dailyExists = vi.fn().mockResolvedValue(true);
    (d.writer as any).paperDetailExists = vi.fn().mockResolvedValue(false);

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const result = await pipeline.runForDate("2026-05-11");
    expect(result.kind).toBe("completed");
    expect((result as any).papersWritten).toBe(0);
    // Should NOT have hit the network or LLM
    expect(d.fetcher.fetchRecent).not.toHaveBeenCalled();
    expect(d.llm.call).not.toHaveBeenCalled();
  });
```

Also extend `makeDeps()` in the same file: add `dailyExists` and `paperDetailExists` stubs returning `false` to the `writer` object (so existing tests keep working):

```ts
  const writer = {
    writeDaily: vi.fn(async (date: string, content: string) => {
      writes[`daily/${date}.md`] = content;
      return `daily/${date}.md`;
    }),
    writePaperDetail: vi.fn(async (p: any, date: string, content: string) => {
      writes[`papers/${p.id}.md`] = content;
      return `papers/${p.id}.md`;
    }),
    writeEmptyDaily: vi.fn(async (date: string) => {
      writes[`daily/${date}.md`] = "empty";
      return `daily/${date}.md`;
    }),
    dailyExists: vi.fn(async () => false),
    paperDetailExists: vi.fn(async () => false),
  };
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts -t "short-circuits"`
Expected: FAIL — `fetchRecent` is called because there's no pre-check.

- [ ] **Step 3: Add the pre-check to pipeline**

In `plugin/src/pipeline/pipeline.ts`, modify `runForDate` to start with:

```ts
  async runForDate(dateStr: string): Promise<PipelineResult> {
    const { fetcher, logger } = this.deps;
    logger.info(`pipeline: start for ${dateStr}`);

    // 0. Skip if daily already exists.
    if (await this.deps.writer.dailyExists(dateStr)) {
      logger.info(`pipeline: daily ${dateStr} already exists, skipping`);
      return { kind: "completed", papersWritten: 0 };
    }

    // 1. Fetch /recent
    // ... (existing code unchanged from here)
```

- [ ] **Step 4: Run the full pipeline test suite**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts`
Expected: PASS — new test green, existing tests still green.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/pipeline/pipeline.ts plugin/tests/pipeline.test.ts
git commit -m "feat(plugin): pipeline skips date when daily file already exists"
```

---

## Task 5: Pipeline — per-paper skip in detail loop

**Files:**
- Modify: `plugin/src/pipeline/pipeline.ts`
- Modify: `plugin/tests/pipeline.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `plugin/tests/pipeline.test.ts`:

```ts
  it("skips paper detail when paper file already exists", async () => {
    const d = makeDeps();

    // Use the same fixture-derived id as the other detail test
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];

    // First paper exists, others don't
    (d.writer as any).paperDetailExists = vi.fn(async (id: string) =>
      id === arxivId,
    );
    // Filter returns one paper as detail
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("筛选出相关论文")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: true }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub daily summary\n";
      }
      throw new Error("unexpected LLM call (paper detail should be skipped)");
    });
    d.fetcher.fetchAbstractsByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, "abstract"]]));
    // Provide non-null fullSections so the paper qualifies for the detail loop
    d.paperFetcher.fetch = vi.fn().mockResolvedValue({
      abstractConclusion: "## Abstract\nstub",
      fullSections: "## Section\nbody",
    });

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
    });
    const date = firstDateFromFixture();
    const result = await pipeline.runForDate(date);
    expect(result.kind).toBe("completed");
    // Daily was still written
    expect(d.writer.writeDaily).toHaveBeenCalled();
    // Paper detail file was NOT written (skipped)
    expect(d.writer.writePaperDetail).not.toHaveBeenCalled();
  });
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts -t "skips paper detail"`
Expected: FAIL — `writePaperDetail` is called (or the LLM stub throws because the detail summary call wasn't expected).

- [ ] **Step 3: Add the per-paper skip**

In `plugin/src/pipeline/pipeline.ts`, change the step-8 detail loop:

```ts
    // 8. Detail reports
    const detailPapers = enriched.filter((p) => p.isDetail && p.fullSections);
    for (const p of detailPapers) {
      if (await this.deps.writer.paperDetailExists(p.id)) {
        logger.info(`pipeline: detail ${p.id} already exists, skipping`);
        continue;
      }
      logger.info(`pipeline: detail report for ${p.id}`);
      try {
        const detail = await summarizePaperDetail(p, {
          llm: this.deps.llm,
          logger,
          arxivSettings: this.deps.arxiv,
          advanced: this.deps.advanced,
          llmTemperature: this.deps.llmSettings.temperature,
        });
        await this.deps.writer.writePaperDetail(p, dateStr, detail);
      } catch (e) {
        logger.error(`pipeline: detail failed for ${p.id}`, e);
      }
    }
```

- [ ] **Step 4: Run the pipeline tests**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/pipeline/pipeline.ts plugin/tests/pipeline.test.ts
git commit -m "feat(plugin): pipeline skips per-paper detail when file already exists"
```

---

## Task 6: Scheduler — extract tickDate (pure refactor)

**Files:**
- Modify: `plugin/src/services/scheduler.ts`

This is a pure refactor. The existing scheduler tests are the regression net. No new test needed.

- [ ] **Step 1: Refactor tick() to call tickDate()**

In `plugin/src/services/scheduler.ts`, replace `tick()` and add the new private helper. The class body should contain:

```ts
  async tick(): Promise<void> {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) return;
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();

    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const t = parseHHMM(s.schedule.runAtLocal);
    const scheduledMin = t.hour * 60 + t.minute;

    for (let i = 0; i < s.schedule.lookbackDays; i++) {
      const date = formatDate(daysBefore(todayObj, i));
      const isToday = date === today;
      await this.tickDate(date, {
        now,
        timeGate: isToday ? { scheduledMin, minutesNow } : undefined,
      });
    }
  }

  private async tickDate(
    date: string,
    opts: {
      now: Date;
      timeGate?: { scheduledMin: number; minutesNow: number };
    },
  ): Promise<PipelineResult | undefined> {
    const s = this.deps.getSettings();
    const entry = this.deps.store.get(date);
    if (this.deps.store.isDone(date)) return undefined;
    if (entry.status === "running") return undefined;

    if (opts.timeGate && opts.timeGate.minutesNow < opts.timeGate.scheduledMin) {
      return undefined;
    }

    if (entry.status === "failed_transient") {
      const tickMs = s.schedule.tickIntervalMin * 60_000;
      if (opts.now.getTime() - entry.lastAttempt < tickMs) return undefined;
    }

    return await this.tryRun(date);
  }
```

- [ ] **Step 2: Run the scheduler test suite**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: PASS — all existing tests still green.

- [ ] **Step 3: Commit**

```bash
git add plugin/src/services/scheduler.ts
git commit -m "refactor(plugin): extract per-date branch of tick() into tickDate()"
```

---

## Task 7: Scheduler — tickToday() with weekend skip

**Files:**
- Modify: `plugin/src/services/scheduler.ts`
- Modify: `plugin/tests/scheduler.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `plugin/tests/scheduler.test.ts` (inside the existing `describe`):

```ts
  it("tickToday returns skipped:disabled when schedule disabled", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn();
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: false },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    const result = await svc.tickToday();
    expect((result as any)?.kind).toBe("skipped");
    expect((result as any)?.reason).toBe("disabled");
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("tickToday returns skipped:weekend on Saturday", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn();
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 1 },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-09T05:00:00Z"), // 13:00 Shanghai, Sat
    });
    const result = await svc.tickToday();
    expect((result as any)?.kind).toBe("skipped");
    expect((result as any)?.reason).toBe("weekend");
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("tickToday runs today on a weekday and bypasses runAtLocal gate", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 2 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          runAtLocal: "23:59", // would gate tick() out
          lookbackDays: 1,
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      // 2026-05-11 is Monday in Asia/Shanghai
      now: () => new Date("2026-05-11T00:00:00Z"), // 08:00 Shanghai
    });
    const result = await svc.tickToday();
    expect((result as any)?.kind).toBe("completed");
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
  });

  it("tickToday respects isDone and returns skipped without running", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 3);
    const lock = new RunLock();
    const runForDate = vi.fn();
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 1 },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    const result = await svc.tickToday();
    expect((result as any)?.kind).toBe("skipped");
    expect(runForDate).not.toHaveBeenCalled();
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts -t "tickToday"`
Expected: FAIL — `tickToday is not a function`.

- [ ] **Step 3: Implement tickToday**

In `plugin/src/services/scheduler.ts`:

Update the imports at the top:

```ts
import {
  todayInTz,
  formatDate,
  parseHHMM,
  minutesSinceMidnight,
  daysBefore,
  isWeekendInTz,
} from "../utils/time";
```

Add the `tickToday` method to the class (after `tick`):

```ts
  /**
   * Manual / boot-time trigger limited to today.
   * Bypasses runAtLocal time gate; respects isDone / running / failed_transient cooldown.
   * Silently skips weekend without writing failure state.
   */
  async tickToday(): Promise<
    PipelineResult | { kind: "skipped"; reason: string } | undefined
  > {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) {
      return { kind: "skipped", reason: "disabled" };
    }
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    if (isWeekendInTz(now, tz)) {
      return { kind: "skipped", reason: "weekend" };
    }
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const result = await this.tickDate(today, { now });
    if (result === undefined) {
      // Guarded out (isDone / running / transient cooldown).
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: PASS — all old tests + new tickToday tests green.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/scheduler.ts plugin/tests/scheduler.test.ts
git commit -m "feat(plugin): SchedulerService.tickToday with weekend skip"
```

---

## Task 8: Scheduler — remove initial tick from start()

**Files:**
- Modify: `plugin/src/services/scheduler.ts`
- Modify: `plugin/tests/scheduler.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `plugin/tests/scheduler.test.ts`:

```ts
  it("start() no longer fires an immediate tick", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 0 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          runAtLocal: "00:01",
          lookbackDays: 5,
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    svc.start();
    // Yield a microtask to let any accidental immediate tick resolve.
    await Promise.resolve();
    await Promise.resolve();
    expect(runForDate).not.toHaveBeenCalled();
    svc.stop();
  });
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts -t "start"`
Expected: FAIL — current `start()` calls `this.tick()` immediately, which will fire `runForDate` for the 5-day window.

- [ ] **Step 3: Drop the immediate tick from start()**

In `plugin/src/services/scheduler.ts`, replace `start()`:

```ts
  start(): void {
    const min = this.deps.getSettings().schedule.tickIntervalMin;
    this.stop();
    const handle = setInterval(() => {
      this.tick().catch((e) =>
        this.deps.logger.error("scheduler tick failed", e),
      );
    }, Math.max(1, min) * 60_000);
    this.intervalHandle = handle as unknown as number;
  }
```

(No more `this.tick().catch(...)` line at the end. Callers — `main.ts` and `setScheduleEnabled` — are responsible for triggering the initial run via `tickToday()`.)

- [ ] **Step 4: Run the scheduler tests**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/scheduler.ts plugin/tests/scheduler.test.ts
git commit -m "refactor(plugin): scheduler start() only registers interval; no immediate tick"
```

---

## Task 9: ProgressReporter interface + NoopProgressReporter

**Files:**
- Create: `plugin/src/services/progress.ts`
- Create: `plugin/tests/progress.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `plugin/tests/progress.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { NoopProgressReporter, type ProgressStage } from "../src/services/progress";

describe("NoopProgressReporter", () => {
  it("implements all methods and returns void", () => {
    const r = new NoopProgressReporter();
    expect(() => r.setBatch(1, 1, "2026-05-11")).not.toThrow();
    expect(() => r.setStage("filter" as ProgressStage)).not.toThrow();
    expect(() => r.setStage("fetch-content" as ProgressStage, 1, 3)).not.toThrow();
    expect(() => r.setIdle()).not.toThrow();
    expect(() => r.setIdle("2026-05-11")).not.toThrow();
    expect(() => r.setIdle("2026-05-11", "weekend")).not.toThrow();
    expect(() => r.setDisabled()).not.toThrow();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd plugin && npx vitest run tests/progress.test.ts`
Expected: FAIL — file doesn't exist.

- [ ] **Step 3: Create the module**

Create `plugin/src/services/progress.ts`:

```ts
export type ProgressStage =
  | "fetch-recent"
  | "enrich-abstract"
  | "filter"
  | "fetch-content"
  | "summarize-daily"
  | "write-detail";

export type IdleReason = "weekend" | "disabled";

export interface ProgressReporter {
  setBatch(currentDay: number, totalDays: number, date: string): void;
  setStage(stage: ProgressStage, current?: number, total?: number): void;
  setIdle(lastCompletedDate?: string, reason?: IdleReason): void;
  setDisabled(): void;
}

export class NoopProgressReporter implements ProgressReporter {
  setBatch(_currentDay: number, _totalDays: number, _date: string): void {}
  setStage(_stage: ProgressStage, _current?: number, _total?: number): void {}
  setIdle(_lastCompletedDate?: string, _reason?: IdleReason): void {}
  setDisabled(): void {}
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd plugin && npx vitest run tests/progress.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/progress.ts plugin/tests/progress.test.ts
git commit -m "feat(plugin): ProgressReporter interface and Noop impl"
```

---

## Task 10: StatusBarController — render state table

**Files:**
- Create: `plugin/src/services/status-bar.ts`
- Create: `plugin/tests/status-bar.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `plugin/tests/status-bar.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { StatusBarController } from "../src/services/status-bar";
import { StateStore } from "../src/services/state-store";
import type { RunState } from "../src/settings/types";

function makeEl(): HTMLElement {
  return document.createElement("span");
}

function makeStore(initial: RunState = {}): StateStore {
  const data = { runState: { ...initial } };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

describe("StatusBarController", () => {
  it("renders 'arXiv: disabled' when constructed with disabled state", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: false });
    expect(el.textContent).toBe("arXiv: disabled");
  });

  it("renders 'arXiv: idle' with no history", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    new StatusBarController(el, store, { initiallyEnabled: true });
    expect(el.textContent).toBe("arXiv: idle");
  });

  it("renders 'arXiv: idle · last YYYY-MM-DD' with completed history", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-10");
    await store.setCompleted("2026-05-10", 5);
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 2);
    const el = makeEl();
    new StatusBarController(el, store, { initiallyEnabled: true });
    expect(el.textContent).toBe("arXiv: idle · last 2026-05-11");
  });

  it("setIdle with weekend reason shows '· weekend'", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setIdle(undefined, "weekend");
    expect(el.textContent).toBe("arXiv: idle · weekend");
  });

  it("renders single-date run as 'arXiv: DATE · stage'", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(1, 1, "2026-05-11");
    ctrl.setStage("summarize-daily");
    expect(el.textContent).toBe("arXiv: 2026-05-11 · summarize");
  });

  it("renders batch run as 'arXiv: DATE [n/N] · stage i/n'", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(2, 5, "2026-05-10");
    ctrl.setStage("fetch-content", 3, 8);
    expect(el.textContent).toBe("arXiv: 2026-05-10 [2/5] · fetch 3/8");
  });

  it("setDisabled overrides any prior state", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(1, 1, "2026-05-11");
    ctrl.setStage("filter");
    ctrl.setDisabled();
    expect(el.textContent).toBe("arXiv: disabled");
  });

  it("setIdle after disabled re-enables and uses idle text", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: false });
    ctrl.setIdle("2026-05-11");
    expect(el.textContent).toBe("arXiv: idle · last 2026-05-11");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd plugin && npx vitest run tests/status-bar.test.ts`
Expected: FAIL — file doesn't exist.

- [ ] **Step 3: Implement StatusBarController**

Create `plugin/src/services/status-bar.ts`:

```ts
import type { ProgressReporter, ProgressStage, IdleReason } from "./progress";
import type { StateStore } from "./state-store";

const STAGE_LABELS: Record<ProgressStage, string> = {
  "fetch-recent": "fetch /recent",
  "enrich-abstract": "abstracts",
  "filter": "filter",
  "fetch-content": "fetch",
  "summarize-daily": "summarize",
  "write-detail": "detail",
};

export interface StatusBarOpts {
  initiallyEnabled: boolean;
}

interface BatchState {
  currentDay: number;
  totalDays: number;
  date: string;
}

interface StageState {
  stage: ProgressStage;
  current?: number;
  total?: number;
}

export class StatusBarController implements ProgressReporter {
  private disabled: boolean;
  private batch: BatchState | null = null;
  private stage: StageState | null = null;
  private lastCompletedDate: string | undefined;
  private idleReason: IdleReason | undefined;

  constructor(
    private readonly el: HTMLElement,
    store: StateStore,
    opts: StatusBarOpts,
  ) {
    this.disabled = !opts.initiallyEnabled;
    this.lastCompletedDate = pickLastCompleted(store);
    this.render();
  }

  setBatch(currentDay: number, totalDays: number, date: string): void {
    this.disabled = false;
    this.idleReason = undefined;
    this.batch = { currentDay, totalDays, date };
    this.render();
  }

  setStage(stage: ProgressStage, current?: number, total?: number): void {
    this.disabled = false;
    this.idleReason = undefined;
    this.stage = { stage, current, total };
    this.render();
  }

  setIdle(lastCompletedDate?: string, reason?: IdleReason): void {
    this.disabled = false;
    this.batch = null;
    this.stage = null;
    this.idleReason = reason;
    if (lastCompletedDate) this.lastCompletedDate = lastCompletedDate;
    this.render();
  }

  setDisabled(): void {
    this.disabled = true;
    this.batch = null;
    this.stage = null;
    this.idleReason = undefined;
    this.render();
  }

  private render(): void {
    this.el.textContent = this.computeText();
  }

  private computeText(): string {
    if (this.disabled) return "arXiv: disabled";
    if (this.batch && this.stage) {
      const stagePart = formatStage(this.stage);
      if (this.batch.totalDays > 1) {
        return `arXiv: ${this.batch.date} [${this.batch.currentDay}/${this.batch.totalDays}] · ${stagePart}`;
      }
      return `arXiv: ${this.batch.date} · ${stagePart}`;
    }
    // Idle states
    if (this.idleReason === "weekend") return "arXiv: idle · weekend";
    if (this.lastCompletedDate) return `arXiv: idle · last ${this.lastCompletedDate}`;
    return "arXiv: idle";
  }
}

function formatStage(stage: StageState): string {
  const label = STAGE_LABELS[stage.stage];
  if (stage.current != null && stage.total != null) {
    return `${label} ${stage.current}/${stage.total}`;
  }
  return label;
}

function pickLastCompleted(store: StateStore): string | undefined {
  const snap = store.snapshot();
  const completed = Object.entries(snap)
    .filter(([, v]) => v.status === "completed")
    .map(([k]) => k)
    .sort();
  return completed[completed.length - 1];
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd plugin && npx vitest run tests/status-bar.test.ts`
Expected: PASS — all 8 tests green.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/status-bar.ts plugin/tests/status-bar.test.ts
git commit -m "feat(plugin): StatusBarController renders idle/run/disabled state"
```

---

## Task 11: Wire ProgressReporter into PipelineDeps and emit stages

**Files:**
- Modify: `plugin/src/pipeline/pipeline.ts`
- Modify: `plugin/tests/pipeline.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `plugin/tests/pipeline.test.ts`:

```ts
  it("emits progress stages in order", async () => {
    const d = makeDeps();
    const m = /arXiv:(\d{4}\.\d{4,5})/.exec(recentHtml)!;
    const arxivId = m[1];
    d.llm.call = vi.fn().mockImplementation(async (msgs: any[]) => {
      const sys = msgs[0]?.content ?? "";
      if (sys.includes("筛选出相关论文")) {
        return JSON.stringify({
          papers: [{ id: arxivId, category: "photo-z", detail: false }],
        });
      }
      if (sys.includes("每日论文追踪日报")) {
        return "## stub\n";
      }
      return "";
    });
    d.fetcher.fetchAbstractsByIds = vi
      .fn()
      .mockResolvedValue(new Map([[arxivId, "abstract"]]));

    const calls: Array<[string, number?, number?]> = [];
    const progress = {
      setBatch: vi.fn(),
      setStage: vi.fn((stage: string, current?: number, total?: number) =>
        calls.push([stage, current, total]),
      ),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    };

    const pipeline = new ArxivPipeline({
      fetcher: d.fetcher as any,
      paperFetcher: d.paperFetcher as any,
      writer: d.writer as any,
      llm: d.llm as any,
      logger: d.logger,
      arxiv: DEFAULT_SETTINGS.arxiv,
      advanced: DEFAULT_SETTINGS.advanced,
      output: DEFAULT_SETTINGS.output,
      llmSettings: DEFAULT_SETTINGS.llm,
      progress: progress as any,
    });
    await pipeline.runForDate(firstDateFromFixture());

    const stages = calls.map((c) => c[0]);
    expect(stages).toContain("fetch-recent");
    expect(stages).toContain("enrich-abstract");
    expect(stages).toContain("filter");
    expect(stages).toContain("fetch-content");
    expect(stages).toContain("summarize-daily");
  });
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts -t "emits progress"`
Expected: FAIL — `progress` is not a recognised dep.

- [ ] **Step 3: Wire progress into pipeline**

In `plugin/src/pipeline/pipeline.ts`:

Add the import at the top:

```ts
import type { ProgressReporter } from "../services/progress";
import { NoopProgressReporter } from "../services/progress";
```

Extend `PipelineDeps`:

```ts
export interface PipelineDeps {
  fetcher: ArxivFetcher;
  paperFetcher: PaperContentFetcher;
  writer: MarkdownWriter;
  llm: LlmClient;
  logger: Logger;
  arxiv: ArxivSettings;
  advanced: AdvancedSettings;
  output: OutputSettings;
  llmSettings: LlmSettings;
  progress?: ProgressReporter;
}
```

In the `ArxivPipeline` class, normalize progress in the constructor:

```ts
  private progress: ProgressReporter;

  constructor(private deps: PipelineDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }
```

Now add `setStage` calls at each stage boundary in `runForDate`. The full updated method (showing only the relevant emit lines, the rest unchanged):

```ts
  async runForDate(dateStr: string): Promise<PipelineResult> {
    const { fetcher, logger } = this.deps;
    logger.info(`pipeline: start for ${dateStr}`);

    // 0. Skip if daily already exists.
    if (await this.deps.writer.dailyExists(dateStr)) {
      logger.info(`pipeline: daily ${dateStr} already exists, skipping`);
      return { kind: "completed", papersWritten: 0 };
    }

    // 1. Fetch /recent
    this.progress.setStage("fetch-recent");
    let recentHtml: string;
    try {
      recentHtml = await fetcher.fetchRecent();
    } catch (e) {
      return {
        kind: "failed_transient",
        reason: `fetch /recent failed: ${(e as Error).message}`,
      };
    }

    // 2. Parse (no stage emit — fast)
    let buckets: DateBucket[];
    try {
      buckets = parseRecent(recentHtml);
    } catch (e) {
      return {
        kind: "failed_permanent",
        reason: `parse failed: ${(e as Error).message}`,
      };
    }
    const bucket = buckets.find((b) => b.announceDate === dateStr);
    if (!bucket) {
      return {
        kind: "failed_transient",
        reason: `date ${dateStr} not in /recent (have: ${buckets
          .map((b) => b.announceDate)
          .join(",")})`,
      };
    }
    logger.info(`pipeline: ${bucket.papers.length} papers for ${dateStr}`);

    // 3. Empty day
    if (bucket.papers.length === 0) {
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    // 4. Enrich abstracts
    this.progress.setStage("enrich-abstract");
    try {
      const ids = bucket.papers.map((p) => p.id);
      const absMap = await fetcher.fetchAbstractsByIds(ids);
      for (const p of bucket.papers) {
        const a = absMap.get(p.id);
        if (a) p.abstract = a;
      }
      logger.info(
        `pipeline: enriched ${absMap.size}/${ids.length} abstracts via Atom API`,
      );
    } catch (e) {
      logger.warn(
        `pipeline: abstract enrichment failed, continuing with titles only: ${(e as Error).message}`,
      );
    }

    // 5. LLM filter
    this.progress.setStage("filter");
    const filtered = await filterPapers(bucket.papers, {
      llm: this.deps.llm,
      logger,
      arxivSettings: this.deps.arxiv,
    });
    if (filtered.length === 0) {
      await this.deps.writer.writeEmptyDaily(dateStr);
      return { kind: "completed", papersWritten: 0 };
    }

    // 6. Fetch content for each filtered paper
    const enriched: DailyPaperWithContent[] = [];
    for (let i = 0; i < filtered.length; i++) {
      const p = filtered[i];
      this.progress.setStage("fetch-content", i + 1, filtered.length);
      try {
        const c = await this.deps.paperFetcher.fetch(p.id, {
          isDetail: p.isDetail,
          sectionCharLimit: this.deps.advanced.sectionCharLimit,
          paperCharLimit: this.deps.advanced.paperCharLimit,
          skipSections: this.deps.advanced.skipSections,
          prioritySections: this.deps.advanced.prioritySections,
        });
        enriched.push({
          ...p,
          abstractConclusion: c.abstractConclusion,
          fullSections: c.fullSections,
        });
      } catch (e) {
        logger.error(`pipeline: content fetch failed for ${p.id}`, e);
        enriched.push({
          ...p,
          abstractConclusion: `[获取失败] arXiv ID: ${p.id}`,
          fullSections: null,
        });
      }
    }

    // 7. Daily summary
    this.progress.setStage("summarize-daily");
    let dailySummary: string;
    try {
      dailySummary = await summarizeDaily(enriched, dateStr, {
        llm: this.deps.llm,
        logger,
        arxivSettings: this.deps.arxiv,
        advanced: this.deps.advanced,
        llmTemperature: this.deps.llmSettings.temperature,
      });
    } catch (e) {
      return {
        kind: "failed_transient",
        reason: `daily summary LLM failed: ${(e as Error).message}`,
      };
    }
    await this.deps.writer.writeDaily(dateStr, dailySummary);

    // 8. Detail reports
    const detailPapers = enriched.filter((p) => p.isDetail && p.fullSections);
    for (let i = 0; i < detailPapers.length; i++) {
      const p = detailPapers[i];
      if (await this.deps.writer.paperDetailExists(p.id)) {
        logger.info(`pipeline: detail ${p.id} already exists, skipping`);
        continue;
      }
      this.progress.setStage("write-detail", i + 1, detailPapers.length);
      logger.info(`pipeline: detail report for ${p.id}`);
      try {
        const detail = await summarizePaperDetail(p, {
          llm: this.deps.llm,
          logger,
          arxivSettings: this.deps.arxiv,
          advanced: this.deps.advanced,
          llmTemperature: this.deps.llmSettings.temperature,
        });
        await this.deps.writer.writePaperDetail(p, dateStr, detail);
      } catch (e) {
        logger.error(`pipeline: detail failed for ${p.id}`, e);
      }
    }

    return { kind: "completed", papersWritten: enriched.length };
  }
```

- [ ] **Step 4: Run the pipeline tests**

Run: `cd plugin && npx vitest run tests/pipeline.test.ts`
Expected: PASS — new progress test green, existing tests still green.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/pipeline/pipeline.ts plugin/tests/pipeline.test.ts
git commit -m "feat(plugin): pipeline emits ProgressReporter stages"
```

---

## Task 12: Wire ProgressReporter into SchedulerDeps and emit batch/idle

**Files:**
- Modify: `plugin/src/services/scheduler.ts`
- Modify: `plugin/tests/scheduler.test.ts`

- [ ] **Step 1: Write the failing tests**

Append to `plugin/tests/scheduler.test.ts`:

```ts
  it("tick calls progress.setBatch per date and setIdle at end", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const progress = {
      setBatch: vi.fn(),
      setStage: vi.fn(),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    };
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          runAtLocal: "00:01",
          lookbackDays: 3,
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
      progress: progress as any,
    });
    await svc.tick();
    // Three days were processed
    expect(progress.setBatch).toHaveBeenCalledTimes(3);
    expect(progress.setBatch).toHaveBeenCalledWith(1, 3, "2026-05-11");
    expect(progress.setBatch).toHaveBeenCalledWith(2, 3, "2026-05-10");
    expect(progress.setBatch).toHaveBeenCalledWith(3, 3, "2026-05-09");
    expect(progress.setIdle).toHaveBeenCalled();
  });

  it("tickToday weekend skip emits setIdle with weekend reason", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn();
    const progress = {
      setBatch: vi.fn(),
      setStage: vi.fn(),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    };
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 1 },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-09T05:00:00Z"), // Sat 13:00 Shanghai
      progress: progress as any,
    });
    await svc.tickToday();
    expect(progress.setIdle).toHaveBeenCalledWith(undefined, "weekend");
    expect(progress.setBatch).not.toHaveBeenCalled();
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts -t "progress"`
Expected: FAIL — scheduler doesn't accept a `progress` dep.

- [ ] **Step 3: Wire progress into scheduler**

In `plugin/src/services/scheduler.ts`:

Add the imports:

```ts
import type { ProgressReporter } from "./progress";
import { NoopProgressReporter } from "./progress";
```

Extend `SchedulerDeps`:

```ts
export interface SchedulerDeps {
  getSettings: () => PluginSettings;
  store: StateStore;
  lock: RunLock;
  runForDate: (date: string) => Promise<PipelineResult>;
  logger: Logger;
  now?: () => Date;
  progress?: ProgressReporter;
}
```

Add a private normalized progress field. The constructor stays auto-generated by TypeScript; add an accessor at the top of the class:

```ts
export class SchedulerService {
  private intervalHandle: number | null = null;
  private readonly progress: ProgressReporter;

  constructor(private deps: SchedulerDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }
```

Update each run-path method:

In `tick()`, wrap the loop and emit progress:

```ts
  async tick(): Promise<void> {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) return;
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();

    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const t = parseHHMM(s.schedule.runAtLocal);
    const scheduledMin = t.hour * 60 + t.minute;

    for (let i = 0; i < s.schedule.lookbackDays; i++) {
      const date = formatDate(daysBefore(todayObj, i));
      const isToday = date === today;
      this.progress.setBatch(i + 1, s.schedule.lookbackDays, date);
      await this.tickDate(date, {
        now,
        timeGate: isToday ? { scheduledMin, minutesNow } : undefined,
      });
    }
    this.progress.setIdle(this.latestCompleted());
  }
```

Update `tickToday`:

```ts
  async tickToday(): Promise<
    PipelineResult | { kind: "skipped"; reason: string } | undefined
  > {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) {
      return { kind: "skipped", reason: "disabled" };
    }
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      return { kind: "skipped", reason: "weekend" };
    }
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    this.progress.setBatch(1, 1, today);
    const result = await this.tickDate(today, { now });
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }
```

Update `runForDateNow`:

```ts
  async runForDateNow(
    date: string,
  ): Promise<PipelineResult | { kind: "skipped"; reason: string }> {
    const entry = this.deps.store.get(date);
    if (entry.status === "running") {
      return { kind: "skipped", reason: "already running" };
    }
    this.progress.setBatch(1, 1, date);
    const result = await this.tryRun(date);
    this.progress.setIdle(this.latestCompleted());
    return result ?? { kind: "skipped", reason: "lock held" };
  }
```

Update `runAllPending`:

```ts
  async runAllPending(): Promise<
    Array<{ date: string; result: PipelineResult | { kind: "skipped"; reason: string } }>
  > {
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    const todayObj = todayInTz(now, tz);

    const results: Array<{
      date: string;
      result: PipelineResult | { kind: "skipped"; reason: string };
    }> = [];

    for (let i = 0; i < s.schedule.lookbackDays; i++) {
      const date = formatDate(daysBefore(todayObj, i));
      const entry = this.deps.store.get(date);
      if (this.deps.store.isDone(date)) continue;
      if (entry.status === "running") {
        results.push({ date, result: { kind: "skipped", reason: "already running" } });
        continue;
      }
      this.progress.setBatch(i + 1, s.schedule.lookbackDays, date);
      const r = await this.tryRun(date);
      results.push({ date, result: r ?? { kind: "skipped", reason: "lock held" } });
    }
    this.progress.setIdle(this.latestCompleted());
    return results;
  }
```

Add the `latestCompleted` helper at the bottom of the class:

```ts
  private latestCompleted(): string | undefined {
    const snap = this.deps.store.snapshot();
    const done = Object.entries(snap)
      .filter(([, v]) => v.status === "completed")
      .map(([k]) => k)
      .sort();
    return done[done.length - 1];
  }
```

- [ ] **Step 4: Run the scheduler tests**

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/src/services/scheduler.ts plugin/tests/scheduler.test.ts
git commit -m "feat(plugin): scheduler emits ProgressReporter batch/idle events"
```

---

## Task 13: Flip default `schedule.enabled` to false

**Files:**
- Modify: `plugin/src/settings/defaults.ts`
- Modify: `plugin/tests/scheduler.test.ts` (test fixtures that pass settings inline)

- [ ] **Step 1: Update the default**

In `plugin/src/settings/defaults.ts`, change line 41:

```ts
  schedule: {
    enabled: false,
    runAtLocal: "09:30",
    tickIntervalMin: 20,
    lookbackDays: 5,
  },
```

- [ ] **Step 2: Update scheduler tests that rely on the old default**

Most scheduler tests already explicitly spread `schedule: { ...DEFAULT_SETTINGS.schedule, ... }`. The "schedule disabled" test explicitly sets `enabled: false`. We need to update any test whose intent assumed `enabled: true` and didn't override.

Run: `cd plugin && npx vitest run tests/scheduler.test.ts`
Expected: Some tests may now FAIL because they relied on the implicit default.

For each failure, prepend `enabled: true` to the test's `schedule` override. Example:

```ts
schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "00:01", lookbackDays: 1 },
```

Apply this fix in: `"runs today after runAtLocal"`, `"skips dates already completed"`, `"respects failed_transient backoff"`, `"runForDateNow ignores scheduled-time gate"`, `"runAllPending runs every pending date..."`, `"runAllPending ignores scheduled-time gate"`, `"tickToday returns skipped:weekend on Saturday"`, `"tickToday runs today on a weekday..."`, `"tickToday respects isDone..."`, `"tick calls progress.setBatch..."`, `"start() no longer fires..."`.

Also the `tickToday returns skipped:disabled` test already sets `enabled: false` — keep as is.

- [ ] **Step 3: Run full test suite**

Run: `cd plugin && npm test`
Expected: PASS — all tests green.

- [ ] **Step 4: Commit**

```bash
git add plugin/src/settings/defaults.ts plugin/tests/scheduler.test.ts
git commit -m "feat(plugin): default schedule.enabled to false (opt-in)"
```

---

## Task 14: Add `plugin.setScheduleEnabled()` and wire StatusBarController in main

**Files:**
- Modify: `plugin/main.ts`

- [ ] **Step 1: Update main.ts**

Edit `plugin/main.ts`:

Add imports:

```ts
import { StatusBarController } from "./src/services/status-bar";
import { NoopProgressReporter, type ProgressReporter } from "./src/services/progress";
```

Add fields and update the class:

```ts
export default class ArxivDailyPlugin extends Plugin {
  settings!: PluginSettings;
  logger!: Logger;
  stateStore!: StateStore;
  scheduler!: SchedulerService;
  manualFetch!: { fetchAndSummarize: ManualFetchService["fetchAndSummarize"] };
  progress!: ProgressReporter;
  private runLock = new RunLock();

  async onload() {
    await this.loadSettingsAndState();
    this.logger = new Logger(this.settings.advanced.logLevel);

    this.stateStore = new StateStore(
      async () => {
        const data = (await this.loadData()) as PersistedData | null;
        return { runState: data?.runState ?? {} };
      },
      async ({ runState }) => {
        await this.persistAll(runState);
      },
    );
    await this.stateStore.load();

    try {
      this.progress = new StatusBarController(
        this.addStatusBarItem(),
        this.stateStore,
        { initiallyEnabled: this.settings.schedule.enabled },
      );
    } catch (e) {
      this.logger.warn("status bar unavailable, using noop", e);
      this.progress = new NoopProgressReporter();
    }

    this.scheduler = new SchedulerService({
      getSettings: () => this.settings,
      store: this.stateStore,
      lock: this.runLock,
      logger: this.logger,
      runForDate: (date) => this.buildPipeline().runForDate(date),
      progress: this.progress,
    });

    this.manualFetch = {
      fetchAndSummarize: (raw: string, date: string) =>
        this.buildManualFetch().fetchAndSummarize(raw, date),
    };

    this.addSettingTab(new ArxivDailySettingTab(this.app, this));
    registerCommands(this);

    if (this.settings.schedule.enabled) {
      this.scheduler.start();
      this.scheduler
        .tickToday()
        .catch((e) =>
          this.logger.error("scheduler initial tickToday failed", e),
        );
    }
  }
```

Replace `restartScheduler()` with this:

```ts
  restartScheduler(): void {
    this.scheduler.stop();
    if (this.settings.schedule.enabled) this.scheduler.start();
  }

  async setScheduleEnabled(enabled: boolean): Promise<void> {
    if (this.settings.schedule.enabled === enabled) return;
    this.settings.schedule.enabled = enabled;
    await this.saveSettings();
    if (enabled) {
      this.scheduler.start();
      const result = await this.scheduler.tickToday();
      if (result && (result as any).kind === "skipped") {
        const reason = (result as any).reason;
        if (reason === "weekend") {
          this.logger.notice("arXiv Daily: weekend, no update — will check next workday");
        }
      }
    } else {
      this.scheduler.stop();
      this.progress.setDisabled();
    }
  }
```

Update `buildPipeline()` to pass progress:

```ts
  private buildPipeline(): ArxivPipeline {
    const { llm, fetcher, paperFetcher, writer } = this.buildSharedDeps();
    return new ArxivPipeline({
      fetcher,
      paperFetcher,
      writer,
      llm,
      logger: this.logger,
      arxiv: this.settings.arxiv,
      advanced: this.settings.advanced,
      output: this.settings.output,
      llmSettings: this.settings.llm,
      progress: this.progress,
    });
  }
```

- [ ] **Step 2: Build to verify it compiles**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: PASS — no type errors.

- [ ] **Step 3: Run tests (should still pass)**

Run: `cd plugin && npm test`
Expected: PASS — main.ts isn't directly tested but TS compile + unit tests must remain green.

- [ ] **Step 4: Commit**

```bash
git add plugin/main.ts
git commit -m "feat(plugin): main wires StatusBarController and setScheduleEnabled"
```

---

## Task 15: Settings tab routes "启用自动调度" through setScheduleEnabled

**Files:**
- Modify: `plugin/src/settings/tab.ts`

- [ ] **Step 1: Update the toggle handler**

In `plugin/src/settings/tab.ts`, replace lines 197-203 (the "启用自动调度" Setting). Old code:

```ts
    new Setting(containerEl).setName("启用自动调度").addToggle((t) =>
      t.setValue(s.schedule.enabled).onChange(async (v) => {
        s.schedule.enabled = v;
        await this.plugin.saveSettings();
        this.plugin.restartScheduler();
      }),
    );
```

Replace with:

```ts
    new Setting(containerEl)
      .setName("启用自动调度")
      .setDesc("启用后立即总结今天（周末跳过，等下个工作日）")
      .addToggle((t) =>
        t.setValue(s.schedule.enabled).onChange(async (v) => {
          await this.plugin.setScheduleEnabled(v);
        }),
      );
```

- [ ] **Step 2: TypeScript compile**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: PASS.

- [ ] **Step 3: Run tests**

Run: `cd plugin && npm test`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add plugin/src/settings/tab.ts
git commit -m "feat(plugin): settings toggle uses setScheduleEnabled"
```

---

## Task 16: Ribbon menu — status header + Enable/Disable toggle

**Files:**
- Modify: `plugin/src/commands.ts`

- [ ] **Step 1: Update ribbon menu construction**

In `plugin/src/commands.ts`, replace the entire `plugin.addRibbonIcon(...)` block at the bottom of `registerCommands()`:

```ts
  plugin.addRibbonIcon("calendar-clock", "arXiv Daily", (evt: MouseEvent) => {
    const menu = new Menu();

    const enabled = plugin.settings.schedule.enabled;

    // Status header (non-interactive)
    menu.addItem((item) =>
      item
        .setTitle(`Status: ${enabled ? "Enabled" : "Disabled"}`)
        .setIcon(enabled ? "circle-check" : "circle-slash")
        .setDisabled(true),
    );
    // Enable/Disable toggle
    menu.addItem((item) =>
      item
        .setTitle(enabled ? "Disable" : "Enable")
        .setIcon(enabled ? "pause" : "play")
        .onClick(async () => {
          await plugin.setScheduleEnabled(!enabled);
          new Notice(
            `arXiv Daily: ${!enabled ? "enabled" : "disabled"}`,
          );
        }),
    );

    menu.addSeparator();
    menu.addItem((item) =>
      item.setTitle("Run for today").setIcon("play").onClick(runToday),
    );
    menu.addSeparator();
    menu.addItem((item) =>
      item
        .setTitle("Run all pending (lookback)")
        .setIcon("layers")
        .onClick(runAllPending),
    );
    menu.addItem((item) =>
      item
        .setTitle("Run for specific date…")
        .setIcon("calendar")
        .onClick(openDatePicker),
    );
    menu.addItem((item) =>
      item
        .setTitle("Summarize by arXiv ID…")
        .setIcon("file-text")
        .onClick(openArxivIdPicker),
    );
    menu.showAtMouseEvent(evt);
  });
```

- [ ] **Step 2: TypeScript compile**

Run: `cd plugin && npx tsc -noEmit -skipLibCheck`
Expected: PASS.

- [ ] **Step 3: Run tests**

Run: `cd plugin && npm test`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add plugin/src/commands.ts
git commit -m "feat(plugin): ribbon menu shows enabled state and toggle"
```

---

## Task 17: Final build + manual smoke test

**Files:**
- Verify only.

- [ ] **Step 1: Production build**

Run: `cd plugin && npm run build`
Expected: succeeds, produces `plugin/main.js`.

- [ ] **Step 2: Manual smoke test in `plugin_test` vault**

Open Obsidian on the `plugin_test/arxiv-daily` vault with the rebuilt plugin. Verify, in order:

1. **Fresh state simulation:** Delete `plugin_test/arxiv-daily/.obsidian/plugins/obsidian-arxiv-daily/data.json` (if present) and reload Obsidian. Expected: status bar shows `arXiv: disabled`. No summarization runs.
2. **Ribbon menu:** Click the calendar icon. Top item shows `Status: Disabled`. Second item is `Enable`. Click Enable.
   - If today is a weekday: status bar transitions through the pipeline stages; daily and (optionally) paper files appear under `arxiv-daily/`.
   - If today is a weekend: status bar shows `arXiv: idle · weekend`; no files written; Notice says "weekend".
3. **Skip-existing:** Manually create a `daily/2026-05-11.md` file. Trigger "Run for date…" with `2026-05-11`. Expected: completes immediately, no LLM call, status bar shows `arXiv: idle · last 2026-05-11`. No `.bak.md` produced.
4. **Disable:** Open ribbon menu, click Disable. Status bar shows `arXiv: disabled`. Interval ticks stop (verify by waiting `tickIntervalMin` and observing no activity).
5. **Settings sync:** Open settings tab, "启用自动调度" toggle reflects current state. Toggle it; ribbon menu's next open reflects new state.

- [ ] **Step 3: Final commit (if any fixups were made)**

If smoke test surfaced fixes, commit them. Otherwise no commit needed.

---

## Self-Review Notes

**Spec coverage:**
- Block 1a (default flip) → Task 13
- Block 1b (setScheduleEnabled) → Task 14
- Block 1c (ribbon menu) → Task 16
- Block 1d (tickDate, tickToday, start()) → Tasks 6, 7, 8
- Block 1e (main onload) → Task 14
- Block 1f (isWeekendInTz) → Task 1
- Block 2a (daily pre-check) → Task 4
- Block 2b (paper skip in loop) → Task 5
- Block 2c (writer strictness) → Tasks 2, 3
- Block 2d (manual-fetch interaction) → no change required, covered by behavior in Task 3
- Block 3 (progress interface, status bar, wiring) → Tasks 9, 10, 11, 12
- Settings toggle integration → Task 15
- Smoke test → Task 17

All spec sections accounted for. Method names (`dailyExists`, `paperDetailExists`, `tickDate`, `tickToday`, `setScheduleEnabled`, `setBatch`, `setStage`, `setIdle`, `setDisabled`, `isWeekendInTz`) used consistently across tasks.
