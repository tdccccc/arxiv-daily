import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { isReportWrittenRunState, StateStore } from "../src/services/state-store";
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
