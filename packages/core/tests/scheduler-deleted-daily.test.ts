import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { isReportWrittenRunState, STALE_RUNNING_RECOVERY_MS, StateStore } from "../src/services/state-store";
import { RunCancelledError } from "../src/services/cancellation";
import { RunLock } from "../src/services/run-lock";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { RunStateEntry } from "../src/settings/types";

const DATE = "2026-10-01";
const NOW = new Date(`${DATE}T10:00:00Z`);

const completed = (extra: Partial<RunStateEntry> = {}): RunStateEntry => ({
  status: "completed",
  attempts: 1,
  lastAttempt: NOW.getTime(),
  papersWritten: 5,
  outcome: "papers_written",
  ...extra,
});

async function fixture(
  entry: RunStateEntry,
  options: { dailyExists?: (date: string) => Promise<boolean>; withDep?: boolean } = {},
) {
  const durable = { runState: { [DATE]: entry } };
  const store = new StateStore(
    async () => structuredClone(durable),
    async (value) => { durable.runState = structuredClone(value.runState) as typeof durable.runState; },
  );
  await store.loadAuthoritative();
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = "UTC";
  settings.schedule = { enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00", tickIntervalMin: 20 };
  const run = vi.fn(async () => ({ kind: "completed" as const, papersWritten: 3, outcome: "papers_written" as const }));
  const exists = vi.fn(options.dailyExists ?? (async () => false));
  const info = vi.fn();
  const logger = new Logger("error");
  logger.info = info;
  const scheduler = new SchedulerService({
    getSettings: () => settings,
    store,
    lock: new RunLock(),
    runForDate: run,
    logger,
    now: () => NOW,
    ...(options.withDep === false ? {} : { dailyExists: exists }),
  });
  return { durable, store, run, exists, info, scheduler };
}

describe("manual run of a date whose daily report was deleted", () => {
  it("regenerates a completed date whose report is gone", async () => {
    const f = await fixture(completed());

    const result = await f.scheduler.runForDateNow(DATE);

    expect(result).toMatchObject({ kind: "completed", papersWritten: 3 });
    expect(f.run).toHaveBeenCalledTimes(1);
    expect(f.store.get(DATE)).toMatchObject({ status: "completed", papersWritten: 3 });
    expect(f.info).toHaveBeenCalledWith(expect.stringContaining(DATE));
  });

  it("regenerates a legacy completed entry that has no outcome", async () => {
    const f = await fixture(completed({ outcome: undefined }));

    expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed" });
    expect(f.run).toHaveBeenCalledTimes(1);
  });

  it("also regenerates through the CLI path (retryFailed)", async () => {
    const f = await fixture(completed());

    const result = await f.scheduler.runForDateNow(DATE, { retryFailed: true, trigger: "manual" });

    expect(result).toMatchObject({ kind: "completed", papersWritten: 3 });
    expect(f.run).toHaveBeenCalledTimes(1);
  });

  it("never overwrites a date whose daily report still exists", async () => {
    const f = await fixture(completed(), { dailyExists: async () => true });
    const before = structuredClone(f.durable);

    expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "already done" });
    expect(await f.scheduler.runForDateNow(DATE, { retryFailed: true })).toEqual({ kind: "skipped", reason: "already done" });
    expect(f.run).not.toHaveBeenCalled();
    expect(f.durable).toEqual(before);
  });

  it.each([
    ["no_matches", { outcome: "no_matches" as const, papersWritten: 0 }],
    ["no_updates", { outcome: "no_updates" as const, papersWritten: 0 }],
    ["legacy zero papers", { outcome: undefined, papersWritten: 0 }],
  ])("still answers already done for a completed %s date", async (_label, extra) => {
    const f = await fixture(completed(extra));

    expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "already done" });
    expect(f.run).not.toHaveBeenCalled();
    expect(f.exists).not.toHaveBeenCalled();
  });

  it("answers already done when the existence check throws", async () => {
    const f = await fixture(completed(), {
      dailyExists: async () => { throw new Error("disk offline"); },
    });

    expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "already done" });
    expect(f.run).not.toHaveBeenCalled();
    expect(f.store.get(DATE)).toMatchObject({ status: "completed", papersWritten: 5 });
  });

  it("answers already done when the host cannot check the report", async () => {
    const f = await fixture(completed(), { withDep: false });

    expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "already done" });
    expect(f.run).not.toHaveBeenCalled();
  });

  it("answers already done again once the regenerated report exists", async () => {
    const f = await fixture(completed());
    expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed" });
    f.exists.mockResolvedValue(true);

    expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "already done" });
    expect(f.run).toHaveBeenCalledTimes(1);
  });

  it("does not resurrect a deleted report from the scheduled tick or batch runs", async () => {
    const f = await fixture(completed());

    await runAutomaticPaths(f.scheduler);

    // Positive control: the batch paths do run other, never-completed lookback
    // dates; only the completed date with the deleted report is left alone.
    const ran = f.run.mock.calls.map((call) => (call as unknown[])[0]);
    expect(ran.length).toBeGreaterThan(0);
    expect(ran).not.toContain(DATE);
    expect(f.store.get(DATE)).toMatchObject({ status: "completed", papersWritten: 5 });
  });
});

async function runAutomaticPaths(scheduler: SchedulerService): Promise<void> {
  await scheduler.tickTodayScheduled();
  await scheduler.tickToday();
  await scheduler.tick();
  await scheduler.runAllPending();
  await scheduler.retryFailedInLookback();
}

describe("a deleted-report rerun that does not complete", () => {
  const scenarios: Array<[string, (run: ReturnType<typeof vi.fn>) => void, string]> = [
    ["failed_transient", (run) => run.mockResolvedValueOnce({ kind: "failed_transient", reason: "offline" }), "failed_transient"],
    ["failed_permanent", (run) => run.mockResolvedValueOnce({ kind: "failed_permanent", reason: "bad key" }), "failed_permanent"],
    ["cancelled", (run) => run.mockResolvedValueOnce({ kind: "cancelled", reason: "stop" }), "cancelled"],
    ["pending", (run) => run.mockResolvedValueOnce({ kind: "pending", reason: "not announced", outcome: "awaiting_announcement" }), "pending"],
    ["a thrown error", (run) => run.mockRejectedValueOnce(new Error("boom")), "failed_transient"],
    ["a thrown cancellation", (run) => run.mockRejectedValueOnce(new RunCancelledError("stop")), "cancelled"],
  ];

  it.each(scenarios)("restores the completed entry after %s", async (_label, arrange, expectedKind) => {
    const entry = completed();
    const f = await fixture(entry);
    arrange(f.run as unknown as ReturnType<typeof vi.fn>);

    const result = await f.scheduler.runForDateNow(DATE);

    expect(result.kind).toBe(expectedKind);
    expect(f.run).toHaveBeenCalledTimes(1);
    expect(f.store.get(DATE)).toEqual(entry);
    expect(f.durable.runState[DATE]).toEqual(entry);

    f.run.mockClear();
    await runAutomaticPaths(f.scheduler);

    const ran = f.run.mock.calls.map((call) => (call as unknown[])[0]);
    expect(ran.length).toBeGreaterThan(0);
    expect(ran).not.toContain(DATE);
    expect(f.store.get(DATE)).toEqual(entry);
  });

  it("can still be retried by another manual run", async () => {
    const f = await fixture(completed());
    f.run.mockResolvedValueOnce({ kind: "failed_transient" as const, reason: "offline" } as never);

    expect((await f.scheduler.runForDateNow(DATE)).kind).toBe("failed_transient");
    expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed", papersWritten: 3 });
    expect(f.store.get(DATE)).toMatchObject({ status: "completed", papersWritten: 3 });
  });

  it("leaves an explicit Force run discarding the old entry", async () => {
    const f = await fixture(completed());
    f.run.mockResolvedValueOnce({ kind: "failed_transient" as const, reason: "offline" } as never);

    expect((await f.scheduler.forceRunForDate(DATE)).kind).toBe("failed_transient");
    expect(f.store.get(DATE).status).toBe("failed_transient");
  });
});

describe("isReportWrittenRunState", () => {
  const entry = (extra: Partial<RunStateEntry>): RunStateEntry => ({
    status: "completed",
    attempts: 1,
    lastAttempt: 1,
    ...extra,
  });

  it("is true only for completed entries that wrote a report", () => {
    expect(isReportWrittenRunState(entry({ outcome: "papers_written", papersWritten: 4 }))).toBe(true);
    expect(isReportWrittenRunState(entry({ papersWritten: 4 }))).toBe(true);
  });

  it("is false for dates that never had a daily file", () => {
    expect(isReportWrittenRunState(undefined)).toBe(false);
    expect(isReportWrittenRunState(entry({ outcome: "no_matches", papersWritten: 0 }))).toBe(false);
    expect(isReportWrittenRunState(entry({ outcome: "no_updates", papersWritten: 0 }))).toBe(false);
    expect(isReportWrittenRunState(entry({ papersWritten: 0 }))).toBe(false);
    expect(isReportWrittenRunState(entry({}))).toBe(false);
  });

  it("is false for every status other than completed", () => {
    for (const status of ["pending", "running", "failed_transient", "failed_permanent", "skipped"] as const) {
      expect(isReportWrittenRunState(entry({ status, outcome: "papers_written", papersWritten: 4 }))).toBe(false);
    }
  });
});

describe("a deleted-report rerun whose result cannot be saved", () => {
  const ranDates = (run: { mock: { calls: unknown[][] } }) => run.mock.calls.map((call) => call[0]);

  it("keeps a cancelled rerun from reopening the date when saving its result fails once", async () => {
    const entry = completed();
    const f = await fixture(entry);
    const setPending = vi.spyOn(f.store, "setPending");
    setPending.mockRejectedValueOnce(new Error("disk busy"));
    const restore = vi.spyOn(f.store, "restoreRerunOrigin");
    restore.mockRejectedValueOnce(new Error("disk busy"));
    f.run.mockResolvedValueOnce({ kind: "cancelled", reason: "stop" } as never);

    expect((await f.scheduler.runForDateNow(DATE)).kind).toBe("cancelled");
    // Nothing was restored yet; the unfinished result is held for a retry.
    expect(f.store.get(DATE)).toMatchObject({ status: "running", rerunOf: entry });

    f.run.mockClear();
    await runAutomaticPaths(f.scheduler);

    expect(ranDates(f.run)).not.toContain(DATE);
    expect(f.store.get(DATE)).toEqual(entry);
    expect(f.durable.runState[DATE]).toEqual(entry);
    expect(setPending.mock.calls.map((call) => call[0])).not.toContain(DATE);
  });

  it("keeps a pending rerun completed when the retry happens through a batch run", async () => {
    const entry = completed();
    const f = await fixture(entry);
    vi.spyOn(f.store, "restoreRerunOrigin").mockRejectedValueOnce(new Error("disk busy"));
    f.run.mockResolvedValueOnce({ kind: "pending", reason: "not announced", outcome: "awaiting_announcement" } as never);

    expect((await f.scheduler.runForDateNow(DATE)).kind).toBe("pending");
    f.run.mockClear();
    await f.scheduler.runAllPending();
    await f.scheduler.retryFailedInLookback();

    expect(ranDates(f.run)).not.toContain(DATE);
    expect(f.store.get(DATE)).toEqual(entry);
  });

  it("restores the completed entry when saving the failure fails", async () => {
    const entry = completed();
    const f = await fixture(entry);
    const clear = vi.spyOn(f.store, "clearDate");
    vi.spyOn(f.store, "setFailed").mockRejectedValueOnce(new Error("disk busy"));
    f.run.mockResolvedValueOnce({ kind: "failed_transient", reason: "offline" } as never);

    expect((await f.scheduler.runForDateNow(DATE)).kind).toBe("failed_transient");
    expect(f.store.get(DATE)).toEqual(entry);
    expect(clear).not.toHaveBeenCalled();

    f.run.mockClear();
    await runAutomaticPaths(f.scheduler);
    expect(ranDates(f.run)).not.toContain(DATE);
    expect(f.store.get(DATE)).toEqual(entry);
  });

  it("still clears the date when an explicit Force run cannot save its failure", async () => {
    const f = await fixture(completed());
    vi.spyOn(f.store, "setFailed").mockRejectedValueOnce(new Error("disk busy"));
    f.run.mockResolvedValueOnce({ kind: "failed_transient", reason: "offline" } as never);

    expect((await f.scheduler.forceRunForDate(DATE)).kind).toBe("failed_transient");
    expect(f.store.get(DATE).status).toBe("pending");
  });

  it("does not regenerate a deleted report when an automatic caller finalizes pending work", async () => {
    const entry = completed();
    const f = await fixture(entry);
    const driver = (f.scheduler as unknown as { driver: { pendingNonCompletions: Map<string, unknown> } }).driver;
    // The pending result is seen when checked but is gone once the lock is held.
    let first = true;
    driver.pendingNonCompletions = new (class extends Map<string, unknown> {
      override has(key: string): boolean {
        if (first) { first = false; return true; }
        return super.has(key);
      }
    })();

    const attempt = await (f.scheduler as unknown as {
      driver: { attemptPendingCompletion(date: string, trigger: string, now: Date): Promise<{ handled: boolean; result?: unknown }> };
    }).driver.attemptPendingCompletion(DATE, "scheduler", NOW);

    expect(attempt.handled).toBe(true);
    expect(f.run).not.toHaveBeenCalled();
    expect(f.store.get(DATE)).toEqual(entry);
  });
});

describe("a deleted-report rerun whose process died", () => {
  type Durable = { runState: Record<string, RunStateEntry> };

  async function openProcess(durable: Durable, options: { exists?: boolean; inProcessLock?: boolean } = {}) {
    const store = new StateStore(
      async () => structuredClone(durable),
      async (value) => { durable.runState = structuredClone(value.runState) as Durable["runState"]; },
    );
    await store.loadAuthoritative();
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.arxiv.timezone = "UTC";
    settings.schedule = { enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00", tickIntervalMin: 20 };
    const run = vi.fn(async () => ({ kind: "completed" as const, papersWritten: 3, outcome: "papers_written" as const }));
    const warn = vi.fn();
    const logger = new Logger("error");
    logger.warn = warn;
    const scheduler = new SchedulerService({
      getSettings: () => settings,
      store,
      // A host lock shared across processes: recovery may assume the run is dead.
      lock: options.inProcessLock ? new RunLock() : new RunLock(async () => ({ release: async () => {} })),
      runForDate: run,
      logger,
      now: () => NOW,
      dailyExists: async () => options.exists ?? false,
    });
    return { store, run, scheduler, warn };
  }

  /** Starts a rerun in a first process whose pipeline never resolves, then abandons it. */
  async function crashDuringRerun(entry: RunStateEntry, force = false) {
    const durable: Durable = { runState: { [DATE]: entry } };
    const dying = await openProcess(durable);
    dying.run.mockImplementation(() => new Promise(() => {}));
    if (force) void dying.scheduler.forceRunForDate(DATE);
    else void dying.scheduler.runForDateNow(DATE);
    await vi.waitFor(() => expect(durable.runState[DATE]?.status).toBe("running"));
    return durable;
  }

  it("persists the completed entry on the running entry from the first write", async () => {
    const entry = completed();
    const durable = await crashDuringRerun(entry);

    expect(durable.runState[DATE]).toMatchObject({ status: "running", rerunOf: entry });
  });

  it("restores the completed entry instead of reopening the date", async () => {
    const entry = completed();
    const durable = await crashDuringRerun(entry);
    const next = await openProcess(durable);

    await runAutomaticPaths(next.scheduler);

    const ran = next.run.mock.calls.map((call) => (call as unknown[])[0]);
    expect(ran.length).toBeGreaterThan(0);
    expect(ran).not.toContain(DATE);
    expect(durable.runState[DATE]).toEqual(entry);
    expect(next.warn).toHaveBeenCalledWith(expect.stringContaining(DATE));
  });

  it.each([
    ["tickTodayScheduled", (s: SchedulerService) => s.tickTodayScheduled()],
    ["tickToday", (s: SchedulerService) => s.tickToday()],
    ["tick", (s: SchedulerService) => s.tick()],
    ["runAllPending", (s: SchedulerService) => s.runAllPending()],
    ["retryFailedInLookback", (s: SchedulerService) => s.retryFailedInLookback()],
  ])("%s alone leaves the date done", async (_name, trigger) => {
    const entry = completed();
    const durable = await crashDuringRerun(entry);
    const next = await openProcess(durable);

    await trigger(next.scheduler);

    expect(next.run.mock.calls.map((call) => (call as unknown[])[0])).not.toContain(DATE);
    expect(durable.runState[DATE]).toMatchObject({ status: "completed", papersWritten: 5 });
  });

  it("lets a manual run regenerate the report afterwards", async () => {
    const durable = await crashDuringRerun(completed());
    const next = await openProcess(durable);
    await runAutomaticPaths(next.scheduler);
    next.run.mockClear();

    expect(await next.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed", papersWritten: 3 });
    expect(next.run.mock.calls.map((call) => (call as unknown[])[0])).toEqual([DATE]);
    expect(durable.runState[DATE]).toMatchObject({ status: "completed", papersWritten: 3 });
    expect(durable.runState[DATE]?.rerunOf).toBeUndefined();
  });

  it("restores a marked entry left in a failed state by a crash before cleanup", async () => {
    const entry = completed();
    const durable: Durable = {
      runState: {
        [DATE]: { status: "failed_transient", attempts: 1, failureAttempts: 1, lastAttempt: NOW.getTime(), error: "offline", rerunOf: entry },
      },
    };
    const next = await openProcess(durable);

    await runAutomaticPaths(next.scheduler);

    expect(next.run.mock.calls.map((call) => (call as unknown[])[0])).not.toContain(DATE);
    expect(durable.runState[DATE]).toEqual(entry);
  });

  it("keeps a crashed explicit Force run retryable", async () => {
    const durable = await crashDuringRerun(completed(), true);

    expect(durable.runState[DATE]?.rerunOf).toBeUndefined();
    const next = await openProcess(durable);
    await next.scheduler.runAllPending();

    expect(next.run.mock.calls.map((call) => (call as unknown[])[0])).toContain(DATE);
  });

  describe("with a lock held only by this process", () => {
    const marked = (lastAttempt: number): RunStateEntry => ({
      status: "running",
      attempts: 1,
      failureAttempts: 0,
      lastAttempt,
      rerunOf: completed(),
    });

    it("leaves a young marked running entry alone and runs no automatic path on it", async () => {
      const durable: Durable = { runState: { [DATE]: marked(NOW.getTime() - STALE_RUNNING_RECOVERY_MS / 2) } };
      const before = structuredClone(durable.runState[DATE]);
      const next = await openProcess(durable, { inProcessLock: true });

      await runAutomaticPaths(next.scheduler);

      const ran = next.run.mock.calls.map((call) => (call as unknown[])[0]);
      expect(ran.length).toBeGreaterThan(0);
      expect(ran).not.toContain(DATE);
      expect(durable.runState[DATE]).toEqual(before);
    });

    it("restores an old marked running entry without running the date", async () => {
      const entry = completed();
      const durable: Durable = { runState: { [DATE]: marked(NOW.getTime() - STALE_RUNNING_RECOVERY_MS * 2) } };
      const next = await openProcess(durable, { inProcessLock: true });

      await runAutomaticPaths(next.scheduler);

      const ran = next.run.mock.calls.map((call) => (call as unknown[])[0]);
      expect(ran.length).toBeGreaterThan(0);
      expect(ran).not.toContain(DATE);
      expect(durable.runState[DATE]).toEqual(entry);
    });
  });
});
