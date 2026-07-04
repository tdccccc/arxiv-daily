import { describe, it, expect, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { Logger } from "../src/services/logger";
import { StateStore } from "../src/services/state-store";
import { RunLock } from "../src/services/run-lock";
import { RunCancellationService, RunCancelledError } from "../src/services/cancellation";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { RunHistoryRecord } from "../src/services/run-history";

function makeStore() {
  const data = { runState: {} as Record<string, any> };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

function makeHistory() {
  const records: RunHistoryRecord[] = [];
  return {
    records,
    store: {
      safeAppend: vi.fn(async (record: RunHistoryRecord) => {
        records.push(record);
      }),
    },
  };
}

describe("SchedulerService", () => {
  it("uses a replacement state store for later runs", async () => {
    const oldStore = makeStore();
    const newStore = makeStore();
    await oldStore.load();
    await newStore.load();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store: oldStore,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
    });

    svc.replaceStore(newStore);
    await svc.runForDateNow("2026-06-16");

    expect(oldStore.get("2026-06-16").status).toBe("pending");
    expect(newStore.get("2026-06-16").status).toBe("completed");
  });

  it("does not run before runAtLocal time", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 0 });
    const settings = {
      ...DEFAULT_SETTINGS,
      schedule: {
        ...DEFAULT_SETTINGS.schedule,
        runAtLocal: "23:59",

      },
    };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T00:00:00Z"), // 08:00 Shanghai
    });
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("runs today after runAtLocal", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 3 });
    const settings = {
      ...DEFAULT_SETTINGS,
      schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "00:01" },
    };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"), // 13:00 Shanghai
    });
    await svc.tick();
    // With lookbackDays=5, checks 05-11 (Mon), 05-10 (Sun, skip), 05-09 (Sat, skip), 05-08 (Fri), 05-07 (Thu)
    expect(runForDate).toHaveBeenCalledTimes(3);
    expect(store.get("2026-05-11").status).toBe("completed");
  });

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

  it("scheduled tick skips weekend dates in the lookback window", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          enabled: true,
          runAtLocal: "00:01",
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"), // Mon, includes Sun/Sat lookback
    });
    await svc.tick();
    // With lookbackDays=5, checks 05-11 (Mon), 05-10 (Sun, skip), 05-09 (Sat, skip), 05-08 (Fri), 05-07 (Thu)
    expect(runForDate).toHaveBeenCalledTimes(3);
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
    expect(runForDate).toHaveBeenCalledWith("2026-05-08");
    expect(runForDate).toHaveBeenCalledWith("2026-05-07");
    expect(store.get("2026-05-10").status).toBe("pending");
    expect(store.get("2026-05-09").status).toBe("pending");
  });

  it("skips dates already completed", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 5);
    const lock = new RunLock();
    const runForDate = vi.fn();
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("respects failed_transient backoff", async () => {
    const store = makeStore();
    await store.load();
    const fixedNow = new Date("2026-05-11T05:00:00Z"); // 13:00 Shanghai
    await store.setRunning("2026-05-11");
    await store.setFailed("2026-05-11", "transient", "x");
    const lock = new RunLock();
    const runForDate = vi.fn();
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => fixedNow,
    });
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("runForDateNow ignores scheduled-time gate", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 2 });
    const settings = {
      ...DEFAULT_SETTINGS,
      schedule: {
        ...DEFAULT_SETTINGS.schedule,
        runAtLocal: "23:59",

      },
    };
    const svc = new SchedulerService({
      getSettings: () => settings,
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T00:00:00Z"),
    });
    await svc.runForDateNow("2026-05-11");
    expect(runForDate).toHaveBeenCalledTimes(1);
  });

  it("preserves an existing non-zero completed count when an existing daily short-circuits", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-06-24");
    await store.setCompleted("2026-06-24", 10);
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 0 });
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-06-25T10:23:00Z"),
    });

    await svc.runForDateNow("2026-06-24");

    expect(store.get("2026-06-24").status).toBe("completed");
    expect(store.get("2026-06-24").papersWritten).toBe(10);
  });

  it("writes run history for started and completed outcomes", async () => {
    const store = makeStore();
    await store.load();
    const history = makeHistory();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 4 });
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-06-25T10:23:00Z"),
      runHistory: history.store,
      dailyPathForDate: (date) => `arxiv-daily/daily/${date}.md`,
    });

    await svc.runForDateNow("2026-06-24", { trigger: "calendar" });

    expect(history.records).toMatchObject([
      {
        event: "started",
        trigger: "calendar",
        date: "2026-06-24",
        status: "running",
        dailyPath: "arxiv-daily/daily/2026-06-24.md",
      },
      {
        event: "completed",
        trigger: "calendar",
        date: "2026-06-24",
        status: "completed",
        resultKind: "completed",
        papersWritten: 4,
        requestedPapersWritten: 4,
      },
    ]);
  });

  it("writes failed run history with errorMessage for thrown runs", async () => {
    const store = makeStore();
    await store.load();
    const history = makeHistory();
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate: vi.fn(async () => {
        throw new Error("network timeout");
      }),
      logger: new Logger("error"),
      now: () => new Date("2026-06-25T10:23:00Z"),
      runHistory: history.store,
      dailyPathForDate: (date) => `arxiv-daily/daily/${date}.md`,
    });

    await svc.runForDateNow("2026-06-24");

    expect(history.records.at(-1)).toMatchObject({
      event: "failed",
      trigger: "manual",
      date: "2026-06-24",
      status: "failed_transient",
      resultKind: "failed_transient",
      reason: "network timeout",
      errorMessage: "network timeout",
    });
  });

  it("marks cancelled runs as skipped and records cancelled history without retry state", async () => {
    const store = makeStore();
    await store.load();
    const history = makeHistory();
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate: vi.fn(async () => {
        throw new RunCancelledError("cancelled by test");
      }),
      logger: new Logger("error"),
      now: () => new Date("2026-06-25T10:23:00Z"),
      runHistory: history.store,
    });

    const result = await svc.runForDateNow("2026-06-24");

    expect(result).toEqual({ kind: "cancelled", reason: "cancelled by test" });
    expect(store.get("2026-06-24")).toMatchObject({
      status: "skipped",
      error: "cancelled by test",
    });
    expect(history.records.at(-1)).toMatchObject({
      event: "skipped",
      trigger: "manual",
      date: "2026-06-24",
      status: "skipped",
      resultKind: "cancelled",
      reason: "cancelled by test",
      errorMessage: "cancelled by test",
    });
  });

  it("marks preserved paper counts in run history", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-06-24");
    await store.setCompleted("2026-06-24", 10);
    const history = makeHistory();
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate: vi.fn(async () => ({ kind: "completed" as const, papersWritten: 0 })),
      logger: new Logger("error"),
      now: () => new Date("2026-06-25T10:23:00Z"),
      runHistory: history.store,
    });

    await svc.runForDateNow("2026-06-24");

    expect(history.records.at(-1)).toMatchObject({
      event: "completed",
      papersWritten: 10,
      requestedPapersWritten: 0,
      preservedPapersWritten: true,
    });
  });

  it("writes skipped run history when manual run is already running", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-06-24");
    const history = makeHistory();
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate: vi.fn(),
      logger: new Logger("error"),
      now: () => new Date("2026-06-25T10:23:00Z"),
      runHistory: history.store,
    });

    const result = await svc.runForDateNow("2026-06-24", { trigger: "calendar" });

    expect(result).toEqual({ kind: "skipped", reason: "already running" });
    expect(history.records).toMatchObject([
      {
        event: "skipped",
        trigger: "calendar",
        date: "2026-06-24",
        resultKind: "skipped",
        reason: "already running",
      },
    ]);
  });

  it("forceRunForDate clears existing state before running", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setFailed("2026-05-11", "permanent", "old");
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 2 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule,  },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    const result = await svc.forceRunForDate("2026-05-11");
    expect((result as any).kind).toBe("completed");
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
    expect(store.get("2026-05-11").attempts).toBe(1);
    expect(store.get("2026-05-11").error).toBeUndefined();
  });

  it("does nothing when schedule is disabled", async () => {
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
    await svc.tick();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("runAllPending runs every pending date in window and skips done", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-12");
    await store.setCompleted("2026-05-12", 3); // today: done, should skip
    // 2026-05-11 and 2026-05-10 left pending
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T05:00:00Z"), // 13:00 Shanghai
    });
    const results = await svc.runAllPending();
    // With lookbackDays=5, checks 05-12 (Tue), 05-11 (Mon), 05-10 (Sun), 05-09 (Sat), 05-08 (Fri)
    // 05-12 is done, 05-11 and 05-10 are pending, 05-09 and 05-08 are pending
    expect(runForDate).toHaveBeenCalledTimes(4);
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
    expect(runForDate).toHaveBeenCalledWith("2026-05-10");
    expect(runForDate).toHaveBeenCalledWith("2026-05-09");
    expect(runForDate).toHaveBeenCalledWith("2026-05-08");
    expect(results).toHaveLength(4);
    expect(results.every((r) => r.result.kind === "completed")).toBe(true);
  });

  it("runAllPending ignores scheduled-time gate", async () => {
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
          runAtLocal: "23:59",
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T00:00:00Z"), // 08:00 Shanghai, pre runAtLocal
    });
    const results = await svc.runAllPending();
    // With lookbackDays=5, checks 05-12, 05-11, 05-10, 05-09, 05-08
    // All dates should be run since we're ignoring the time gate
    expect(runForDate).toHaveBeenCalledTimes(5);
    expect(results[0].date).toBe("2026-05-12");
  });

  it("retryFailedInLookback reruns failed dates only", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-12");
    await store.setFailed("2026-05-12", "transient", "network");
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 1);
    await store.setRunning("2026-05-10");
    await store.setFailed("2026-05-10", "permanent", "old");
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 0 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule,  },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T05:00:00Z"),
    });
    const results = await svc.retryFailedInLookback();
    expect(results.map((r) => r.date)).toEqual(["2026-05-12", "2026-05-10"]);
    expect(runForDate).toHaveBeenCalledTimes(2);
    expect(runForDate).toHaveBeenCalledWith("2026-05-12");
    expect(runForDate).toHaveBeenCalledWith("2026-05-10");
    expect(store.get("2026-05-12").status).toBe("completed");
    expect(store.get("2026-05-10").status).toBe("completed");
  });

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
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true,  },
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
          enabled: true,
          runAtLocal: "09:00",
          runUntilLocal: "18:00",
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T00:00:00Z"), // 08:00 Shanghai, Monday
    });
    const result = await svc.tickToday();
    expect((result as any)?.kind).toBe("completed");
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
  });

  it("tickTodayScheduled respects runAtLocal gate", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn();
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
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T00:00:00Z"), // 08:00 Shanghai, Monday
    });
    const result = await svc.tickTodayScheduled();
    expect((result as any)?.kind).toBe("skipped");
    expect((result as any)?.reason).toBe("guarded");
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("tickTodayScheduled runs today after runAtLocal", async () => {
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
          enabled: true,
          runAtLocal: "00:01",
  
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"), // 13:00 Shanghai, Monday
    });
    const result = await svc.tickTodayScheduled();
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
        schedule: { ...DEFAULT_SETTINGS.schedule,  },
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
  
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    svc.start();
    await Promise.resolve();
    await Promise.resolve();
    expect(runForDate).not.toHaveBeenCalled();
    svc.stop();
  });

  it("tick calls progress.setBatch per date and setIdle at end", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const progress = {
      setTask: vi.fn(),
      setBatch: vi.fn(),
      setStage: vi.fn(),
      setComplete: vi.fn(),
      setError: vi.fn(),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    };
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          enabled: true,
          runAtLocal: "00:01",
  
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
    expect(progress.setBatch).toHaveBeenCalledTimes(5);
    expect(progress.setBatch).toHaveBeenCalledWith(1, 5, "2026-05-11");
    expect(progress.setBatch).toHaveBeenCalledWith(2, 5, "2026-05-10");
    expect(progress.setBatch).toHaveBeenCalledWith(3, 5, "2026-05-09");
    expect(progress.setBatch).toHaveBeenCalledWith(4, 5, "2026-05-08");
    expect(progress.setBatch).toHaveBeenCalledWith(5, 5, "2026-05-07");
    expect(progress.setTask).toHaveBeenCalledWith("arXiv Daily report", "2026-05-11");
    expect(progress.setComplete).toHaveBeenCalledWith("Daily report complete: 2026-05-11");
    expect(progress.setIdle).toHaveBeenCalled();
  });

  it("tickToday weekend skip emits setIdle with weekend reason", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const runForDate = vi.fn();
    const progress = {
      setTask: vi.fn(),
      setBatch: vi.fn(),
      setStage: vi.fn(),
      setComplete: vi.fn(),
      setError: vi.fn(),
      setIdle: vi.fn(),
      setDisabled: vi.fn(),
    };
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true,  },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-09T05:00:00Z"),
      progress: progress as any,
    });
    await svc.tickToday();
    expect(progress.setIdle).toHaveBeenCalledWith(undefined, "weekend");
    expect(progress.setBatch).not.toHaveBeenCalled();
  });

  it("cancels an active run and records a cancelled skipped result", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const cancellation = new RunCancellationService();
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const runForDate = vi.fn((_date: string, signal?: AbortSignal) => {
      markStarted();
      return new Promise((resolve) => {
        signal?.addEventListener("abort", () =>
          resolve({
            kind: "cancelled",
            reason: String((signal as any).reason ?? "cancelled by user"),
          }),
        );
      });
    });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true,  },
      }),
      store,
      lock,
      runForDate: runForDate as any,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
      cancellation,
    });

    const pending = svc.runForDateNow("2026-05-11");
    await started;
    expect(svc.activeRuns()).toEqual(["2026-05-11"]);
    expect(svc.cancelCurrentRun()).toEqual(["2026-05-11"]);

    const result = await pending;
    expect((result as any).kind).toBe("cancelled");
    expect((result as any).reason).toBe("cancelled by user");
    expect(store.get("2026-05-11").status).toBe("skipped");
    expect(store.get("2026-05-11").error).toBe("cancelled by user");
    expect(svc.activeRuns()).toEqual([]);
  });

  it("stops runAllPending after cancellation is requested", async () => {
    const store = makeStore();
    await store.load();
    const lock = new RunLock();
    const cancellation = new RunCancellationService();
    let svc!: SchedulerService;
    const runForDate = vi.fn(async () => {
      svc.cancelCurrentRun();
      return { kind: "cancelled", reason: "cancelled by user" };
    });
    svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule,  },
      }),
      store,
      lock,
      runForDate: runForDate as any,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T05:00:00Z"),
      cancellation,
    });

    const results = await svc.runAllPending();
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(results).toHaveLength(1);
    expect(results[0].date).toBe("2026-05-12");
    expect(results[0].result.kind).toBe("cancelled");
  });

  // --- Behavior pinning tests (must stay green through refactor) ---

  it("PIN: tick() skips lookback entirely when today is already done", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 5);
    const recentDates = { refresh: vi.fn(async () => undefined) };
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00" },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
      recentDates,
    });
    await svc.tick();
    expect(recentDates.refresh).not.toHaveBeenCalled();
    expect(runForDate).not.toHaveBeenCalled();
  });

  it("PIN: tickTodayScheduled() runs today AND refreshes recentDates when inside window", async () => {
    const store = makeStore();
    await store.load();
    const recentDates = { refresh: vi.fn(async () => undefined) };
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 2 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00" },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
      recentDates,
    });
    await svc.tickTodayScheduled();
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
    expect(recentDates.refresh).toHaveBeenCalledTimes(1);
  });

  it("PIN: tickTodayScheduled() does NOT run or refresh recentDates when outside window", async () => {
    const store = makeStore();
    await store.load();
    const recentDates = { refresh: vi.fn(async () => undefined) };
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 0 });
    // Settings window 09:00-18:00; "now" 19:01 Shanghai = outside.
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "09:00", runUntilLocal: "18:00" },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      // 11:01 UTC = 19:01 Shanghai (Asia/Shanghai UTC+8)
      now: () => new Date("2026-05-11T11:01:00Z"),
      recentDates,
    });
    const result = await svc.tickTodayScheduled();
    // Current legacy behavior: tickDate applies the time gate and returns guarded.
    expect((result as any)?.kind).toBe("skipped");
    expect((result as any)?.reason).toBe("guarded");
    expect(runForDate).not.toHaveBeenCalled();
    expect(recentDates.refresh).not.toHaveBeenCalled();
  });

  it("PIN: runAllPending() does NOT filter non-today dates when recentDates.hasDate is undefined (CLI path)", async () => {
    const store = makeStore();
    await store.load();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 1 });
    // recentDates entirely omitted (undefined) - mirrors CLI construction.
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "00:01" },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    const results = await svc.runAllPending();
    // With lookback 5 and no hasDate filter, all non-weekend prior dates get attempted:
    expect(results.length).toBeGreaterThan(0);
    expect(runForDate).toHaveBeenCalled();
  });

  it("PIN: pending result leaves date not-done and is retried on a subsequent tick", async () => {
    const store = makeStore();
    await store.load();
    const runForDate = vi
      .fn()
      .mockResolvedValueOnce({ kind: "pending", reason: "arxiv not ready" })
      .mockResolvedValueOnce({ kind: "completed", papersWritten: 3 });
    const svc = new SchedulerService({
      getSettings: () => ({ ...DEFAULT_SETTINGS, schedule: { ...DEFAULT_SETTINGS.schedule } }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await svc.runForDateNow("2026-05-11");
    expect(store.get("2026-05-11").status).not.toBe("completed");
    expect(store.get("2026-05-11").status).toBe("pending");
    // Second run: date is still pending (not done), so it runs again:
    await svc.runForDateNow("2026-05-11");
    expect(store.get("2026-05-11").status).toBe("completed");
  });

  it("PIN: tick() per-date weekend skip vs tickToday() per-now weekend check are distinct behaviors", async () => {
    // tick() loops lookback; a weekend DATE in the lookback is skipped even if "now" is a weekday.
    const store = makeStore();
    await store.load();
    const runForDate = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "00:01" },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      // 2026-05-11 is Monday; lookback includes Sat 05-09 and Sun 05-10 (skipped per-date).
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await svc.tick();
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
    expect(runForDate).not.toHaveBeenCalledWith("2026-05-10");
    expect(runForDate).not.toHaveBeenCalledWith("2026-05-09");

    // tickToday() checks isWeekend NOW; FF Monday 13:00 Shanghai is a weekday so it runs.
    const store2 = makeStore();
    await store2.load();
    const runForDate2 = vi.fn().mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc2 = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true },
      }),
      store: store2,
      lock: new RunLock(),
      runForDate: runForDate2,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await svc2.tickToday();
    expect(runForDate2).toHaveBeenCalledWith("2026-05-11");
  });
});
