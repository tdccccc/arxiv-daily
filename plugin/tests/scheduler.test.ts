import { describe, it, expect, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { Logger } from "../src/services/logger";
import { StateStore } from "../src/services/state-store";
import { RunLock } from "../src/services/run-lock";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function makeStore() {
  const data = { runState: {} as Record<string, any> };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

describe("SchedulerService", () => {
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
        lookbackDays: 1,
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
      schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, runAtLocal: "00:01", lookbackDays: 1 },
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
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(store.get("2026-05-11").status).toBe("completed");
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
          lookbackDays: 3,
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-11T05:00:00Z"), // Mon, includes Sun/Sat lookback
    });
    await svc.tick();
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
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
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 1 },
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
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 1 },
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
        lookbackDays: 1,
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
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 1 },
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
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 3 },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T05:00:00Z"), // 13:00 Shanghai
    });
    const results = await svc.runAllPending();
    expect(runForDate).toHaveBeenCalledTimes(2);
    expect(runForDate).toHaveBeenCalledWith("2026-05-11");
    expect(runForDate).toHaveBeenCalledWith("2026-05-10");
    expect(results).toHaveLength(2);
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
          lookbackDays: 1,
        },
      }),
      store,
      lock,
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T00:00:00Z"), // 08:00 Shanghai, pre runAtLocal
    });
    const results = await svc.runAllPending();
    expect(runForDate).toHaveBeenCalledTimes(1);
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
        schedule: { ...DEFAULT_SETTINGS.schedule, lookbackDays: 3 },
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
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, lookbackDays: 1 },
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
          runAtLocal: "23:59",
          lookbackDays: 1,
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
          runAtLocal: "23:59",
          lookbackDays: 1,
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
          lookbackDays: 1,
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
          enabled: true,
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
        schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true, lookbackDays: 1 },
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
});
