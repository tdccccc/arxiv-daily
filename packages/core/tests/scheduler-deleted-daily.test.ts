import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { StateStore } from "../src/services/state-store";
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

  it("does not resurrect a deleted report from the scheduled tick or batch runs", async () => {
    const f = await fixture(completed());

    await f.scheduler.tickTodayScheduled();
    await f.scheduler.tickToday();
    await f.scheduler.tick();
    await f.scheduler.runAllPending();
    await f.scheduler.retryFailedInLookback();

    // The batch paths do run other, never-completed lookback dates; only the
    // completed date with the deleted report must be left alone.
    expect(f.run.mock.calls.map((call) => (call as unknown[])[0])).not.toContain(DATE);
    expect(f.store.get(DATE)).toMatchObject({ status: "completed", papersWritten: 5 });
  });
});
