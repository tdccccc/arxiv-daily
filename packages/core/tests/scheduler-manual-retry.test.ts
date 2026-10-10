import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { StateStore } from "../src/services/state-store";
import { RunLock } from "../src/services/run-lock";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { RunState, RunStatus } from "../src/settings/types";

const DATE = "2026-10-01";
const NOW = new Date(`${DATE}T10:00:00Z`);

async function fixture(status: RunStatus) {
  const durable = { runState: { [DATE]: { status, attempts: 1, lastAttempt: NOW.getTime(), ...(status === "completed" ? { papersWritten: 2 } : {}) } } as RunState };
  const store = new StateStore(
    async () => structuredClone(durable),
    async value => { durable.runState = structuredClone(value.runState); },
  );
  await store.loadAuthoritative();
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = "UTC";
  settings.schedule = { enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00", tickIntervalMin: 20 };
  const run = vi.fn(async () => ({ kind: "completed" as const, papersWritten: 2 }));
  const clear = vi.spyOn(store, "clearDate");
  const makeScheduler = (lock = new RunLock()) => new SchedulerService({
    getSettings: () => settings, store, lock, runForDate: run,
    logger: new Logger("error"), now: () => NOW,
  });
  return { durable, store, run, clear, makeScheduler };
}

describe("explicit manual date retry", () => {
  it.each(["failed_permanent", "failed_transient"] as const)("executes a %s date again after an explicit retry", async status => {
    const f = await fixture(status);
    const result = await f.makeScheduler().runForDateNow(DATE, { retryFailed: true });
    expect(result).toMatchObject({ kind: "completed", papersWritten: 2 });
    expect(f.run).toHaveBeenCalledTimes(1);
    expect(f.store.get(DATE)).toMatchObject({ status: "completed", papersWritten: 2 });
  });

  it.each(["completed", "running"] as const)("preserves a %s date even when the caller requests failure retry", async status => {
    const f = await fixture(status);
    const before = structuredClone(f.durable);
    expect(await f.makeScheduler().runForDateNow(DATE, { retryFailed: true })).toMatchObject({ kind: "skipped" });
    expect(f.run).not.toHaveBeenCalled();
    expect(f.clear).not.toHaveBeenCalled();
    expect(f.durable).toEqual(before);
  });

  it("runs a new date normally without clearing its state", async () => {
    const f = await fixture("pending");
    expect(await f.makeScheduler().runForDateNow(DATE, { retryFailed: true })).toMatchObject({ kind: "completed" });
    expect(f.run).toHaveBeenCalledTimes(1);
    expect(f.clear).not.toHaveBeenCalled();
  });

  it("rechecks durable state under the acquired lock before retrying", async () => {
    const f = await fixture("failed_permanent");
    const lock = new RunLock(async () => {
      f.durable.runState[DATE] = { status: "completed", papersWritten: 7, attempts: 1, lastAttempt: NOW.getTime() };
      return { release: async () => {} };
    });
    expect(await f.makeScheduler(lock).runForDateNow(DATE, { retryFailed: true })).toMatchObject({ kind: "skipped" });
    expect(f.run).not.toHaveBeenCalled();
    expect(f.clear).not.toHaveBeenCalled();
    expect(f.durable.runState[DATE]).toMatchObject({ status: "completed", papersWritten: 7 });
  });

  it("does not alter a failed date while another process holds the vault lock", async () => {
    const f = await fixture("failed_permanent");
    const before = structuredClone(f.durable);
    expect(await f.makeScheduler(new RunLock(async () => null)).runForDateNow(DATE, { retryFailed: true })).toEqual({ kind: "skipped", reason: "lock held" });
    expect(f.clear).not.toHaveBeenCalled();
    expect(f.run).not.toHaveBeenCalled();
    expect(f.durable).toEqual(before);
  });

  it("keeps permanent failures terminal for automatic schedules and ordinary callers", async () => {
    const f = await fixture("failed_permanent");
    const scheduler = f.makeScheduler();
    await scheduler.tickTodayScheduled();
    expect(await scheduler.runForDateNow(DATE)).toMatchObject({ kind: "skipped" });
    expect(f.run).not.toHaveBeenCalled();
    expect(f.clear).not.toHaveBeenCalled();
    expect(f.store.get(DATE).status).toBe("failed_permanent");
  });
});
