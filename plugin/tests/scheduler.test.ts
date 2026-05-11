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
      schedule: { ...DEFAULT_SETTINGS.schedule, runAtLocal: "00:01", lookbackDays: 1 },
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
      now: () => new Date(),
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
});
