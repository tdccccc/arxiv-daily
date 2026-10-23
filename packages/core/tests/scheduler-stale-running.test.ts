import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { StateStore, STALE_RUNNING_RECOVERY_MS } from "../src/services/state-store";
import { RunLock } from "../src/services/run-lock";
import { Logger } from "../src/services/logger";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { RunState, RunStateEntry } from "../src/settings/types";

const DATE = "2026-10-01";
const CRASHED_AT = new Date(`${DATE}T09:05:00Z`).getTime();
const RESTARTED_AT = new Date(`${DATE}T09:20:00Z`);
const MINUTE = 60_000;

const ownedLock = () => new RunLock(async () => ({ release: async () => {} }));

function fixture(lock: RunLock, entry: Partial<RunStateEntry> | null = {}, completionSaveFails = false) {
  const durable = {
    runState: (entry
      ? { [DATE]: { status: "running", attempts: 1, lastAttempt: CRASHED_AT, ...entry } }
      : {}) as RunState,
  };
  let failCompletionSave = completionSaveFails;
  const store = new StateStore(
    async () => structuredClone(durable),
    async value => {
      if (failCompletionSave && value.runState[DATE]?.status === "completed") throw new Error("disk full");
      durable.runState = structuredClone(value.runState);
    },
  );
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = "UTC";
  settings.schedule = { enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00", tickIntervalMin: 20 };
  let now = RESTARTED_AT;
  const run = vi.fn(async () => ({ kind: "completed" as const, papersWritten: 3 }));
  const logger = new Logger("error");
  const scheduler = new SchedulerService({
    getSettings: () => settings, store, lock, runForDate: run, logger, now: () => now,
  });
  return {
    durable, store, run, scheduler, logger,
    at: (next: Date) => { now = next; },
    commitCompletions: () => { failCompletionSave = false; },
  };
}

describe("a date left running by a run that died", () => {
  describe("while holding the Vault-wide run lock", () => {
    it("runs again on request fifteen minutes after the crash instead of reporting 'already running'", async () => {
      const f = fixture(ownedLock());
      await f.store.loadAuthoritative();

      expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed", papersWritten: 3 });

      expect(f.run).toHaveBeenCalledTimes(1);
      expect(f.durable.runState[DATE]).toMatchObject({ status: "completed", papersWritten: 3 });
    });

    it("runs again when forced", async () => {
      const f = fixture(ownedLock());
      await f.store.loadAuthoritative();

      expect(await f.scheduler.forceRunForDate(DATE)).toMatchObject({ kind: "completed" });

      expect(f.run).toHaveBeenCalledTimes(1);
      expect(f.durable.runState[DATE]?.status).toBe("completed");
    });

    it("includes the date in 'retry failed' and 'run pending'", async () => {
      const retry = fixture(ownedLock());
      await retry.store.loadAuthoritative();
      const retried = await retry.scheduler.retryFailedInLookback();
      expect(retried).toEqual([{ date: DATE, result: expect.objectContaining({ kind: "completed" }) }]);

      const pending = fixture(ownedLock());
      await pending.store.loadAuthoritative();
      const all = await pending.scheduler.runAllPending();
      expect(all.find(item => item.date === DATE)?.result).toMatchObject({ kind: "completed" });
      expect(pending.durable.runState[DATE]?.status).toBe("completed");
    });

    it("lets the scheduled tick recover the date, then retry it once the transient backoff has passed", async () => {
      const f = fixture(ownedLock());
      await f.store.loadAuthoritative();

      expect(await f.scheduler.tickTodayScheduled()).toEqual({ kind: "skipped", reason: "guarded" });
      expect(f.run).not.toHaveBeenCalled();
      expect(f.durable.runState[DATE]).toMatchObject({
        status: "failed_transient",
        attempts: 1,
        failureAttempts: 1,
        error: expect.stringContaining("recovered stale running state"),
      });

      // The backoff counts from the crashed attempt, not from the recovery.
      f.at(new Date(CRASHED_AT + 21 * MINUTE));
      expect(await f.scheduler.tickTodayScheduled()).toMatchObject({ kind: "completed" });
      expect(f.run).toHaveBeenCalledTimes(1);
      expect(f.durable.runState[DATE]?.status).toBe("completed");
    });

    it("counts the leftover as a failure toward the retry limit", async () => {
      const f = fixture(ownedLock(), { attempts: 10, failureAttempts: 9 });
      await f.store.loadAuthoritative();

      expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "skipped" });

      expect(f.run).not.toHaveBeenCalled();
      expect(f.durable.runState[DATE]).toMatchObject({
        status: "failed_permanent",
        error: expect.stringMatching(/^retries exhausted after 10 attempts: recovered stale running state/),
      });
    });

    it("leaves the record alone while another live process holds the lock", async () => {
      const f = fixture(new RunLock(async () => null));
      await f.store.loadAuthoritative();
      const before = structuredClone(f.durable);

      expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "lock held" });
      expect(await f.scheduler.forceRunForDate(DATE)).toEqual({ kind: "skipped", reason: "lock held" });
      expect(await f.scheduler.tickTodayScheduled()).toEqual({ kind: "skipped", reason: "guarded" });

      expect(f.run).not.toHaveBeenCalled();
      expect(f.durable).toEqual(before);
    });

    it("does not treat this process's own uncommitted completion as a leftover", async () => {
      const f = fixture(ownedLock(), null, true);
      await f.store.loadAuthoritative();

      expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "failed_transient" });
      expect(f.durable.runState[DATE]?.status).toBe("running");

      f.commitCompletions();
      expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed" });

      expect(f.run).toHaveBeenCalledTimes(1);
      expect(f.durable.runState[DATE]).toMatchObject({
        status: "completed",
        failureAttempts: 0,
        error: undefined,
      });
    });
  });

  describe("on a host without a cross-process lock", () => {
    it("keeps treating a recent running record as a live run", async () => {
      const f = fixture(new RunLock());
      await f.store.loadAuthoritative();
      const before = structuredClone(f.durable);

      expect(await f.scheduler.runForDateNow(DATE)).toEqual({ kind: "skipped", reason: "already running" });
      expect(await f.scheduler.forceRunForDate(DATE)).toEqual({ kind: "skipped", reason: "already running" });
      expect(await f.scheduler.tickTodayScheduled()).toEqual({ kind: "skipped", reason: "guarded" });

      expect(f.run).not.toHaveBeenCalled();
      expect(f.durable).toEqual(before);
    });

    it("recovers a record older than an hour on a later tick of the same long-lived scheduler", async () => {
      const f = fixture(new RunLock());
      await f.store.loadAuthoritative();

      // A scheduler that was already running when the record went stale is not
      // asked to restart: the first tick after the hour recovers and retries it.
      f.at(new Date(CRASHED_AT + STALE_RUNNING_RECOVERY_MS + MINUTE));
      expect(await f.scheduler.tickTodayScheduled()).toMatchObject({ kind: "completed" });

      expect(f.run).toHaveBeenCalledTimes(1);
      expect(f.durable.runState[DATE]).toMatchObject({ status: "completed", failureAttempts: 1 });
    });

    it("runs a stale record on request without waiting for a restart", async () => {
      const f = fixture(new RunLock());
      await f.store.loadAuthoritative();

      f.at(new Date(CRASHED_AT + STALE_RUNNING_RECOVERY_MS + MINUTE));
      expect(await f.scheduler.runForDateNow(DATE)).toMatchObject({ kind: "completed" });
      expect(f.run).toHaveBeenCalledTimes(1);
    });
  });
});
