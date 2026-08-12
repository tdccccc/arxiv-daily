import { describe, it, expect, vi, beforeEach } from "vitest";
import { SchedulerService, type SchedulerDeps } from "../../src/services/scheduler";
import type { PipelineResult } from "../../src/pipeline/pipeline";
import { RunLock } from "../../src/services/run-lock";


function makeStore(overrides: Record<string, unknown> = {}) {
  const entries: Record<string, { status: string; lastAttempt: number; attempts: number }> = {};
  return {
    loadAuthoritative: vi.fn(async () => {}),
    get: vi.fn((date: string) => entries[date] ?? { status: "pending", lastAttempt: 0, attempts: 0 }),
    isDone: vi.fn(() => false),
    setRunning: vi.fn(async (date: string) => {
      entries[date] = { status: "running", lastAttempt: Date.now(), attempts: 1 };
    }),
    setCompleted: vi.fn(async () => {}),
    setFailed: vi.fn(async () => {}),
    setPending: vi.fn(async (date: string, reason: string) => {
      entries[date] = { status: "pending", lastAttempt: Date.now(), attempts: entries[date]?.attempts ?? 1 };
      void reason;
    }),
    setSkipped: vi.fn(async () => {}),
    clearDate: vi.fn(async () => {}),
    snapshot: vi.fn(() => ({})),
    ...overrides,
  };
}

function makeDeps(overrides: Partial<SchedulerDeps> = {}): SchedulerDeps {
  return {
    getSettings: () => ({
      schedule: { enabled: true, runAtLocal: "08:00", tickIntervalMin: 5 },
      arxiv: { timezone: "UTC" },
    } as any),
    store: makeStore() as any,
    lock: new RunLock(),
    runForDate: vi.fn(async () => ({ kind: "completed", papersWritten: 5 }) as PipelineResult),
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), notice: vi.fn(), debug: vi.fn() } as any,
    now: () => new Date("2026-06-22T10:00:00Z"),
    progress: { setBatch: vi.fn(), setTask: vi.fn(), setComplete: vi.fn(), setError: vi.fn(), setIdle: vi.fn() } as any,
    ...overrides,
  };
}

describe("Scheduler pending result handling", () => {
  let deps: SchedulerDeps;

  beforeEach(() => {
    deps = makeDeps({
      runForDate: vi.fn(async (): Promise<PipelineResult> => ({ kind: "pending", reason: "no papers from arXiv" })),
    });
  });

  it("should not mark date as completed or failed when result is pending", async () => {
    const scheduler = new SchedulerService(deps);
    await scheduler.runForDateNow("2026-06-22");

    expect(deps.store.setCompleted).not.toHaveBeenCalled();
    expect(deps.store.setFailed).not.toHaveBeenCalled();
  });

  it("should mark date pending while preserving attempt history", async () => {
    const scheduler = new SchedulerService(deps);
    await scheduler.runForDateNow("2026-06-22");

    expect(deps.store.setPending).toHaveBeenCalledWith(
      "2026-06-22",
      "no papers from arXiv",
    );
    expect(deps.store.clearDate).not.toHaveBeenCalled();
  });

  it("should log info when result is pending", async () => {
    const scheduler = new SchedulerService(deps);
    await scheduler.runForDateNow("2026-06-22");

    expect(deps.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("pending"),
    );
  });

  it("should set progress to idle when result is pending", async () => {
    const scheduler = new SchedulerService(deps);
    await scheduler.runForDateNow("2026-06-22");

    expect(deps.progress?.setIdle).toHaveBeenCalled();
  });

  it("should return the pending result from runForDateNow", async () => {
    const scheduler = new SchedulerService(deps);
    const result = await scheduler.runForDateNow("2026-06-22");

    expect(result).toEqual({ kind: "pending", reason: "no papers from arXiv" });
  });
});
