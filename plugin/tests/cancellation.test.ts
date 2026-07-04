import { describe, expect, it } from "vitest";
import { RunCancellationService } from "../src/services/cancellation";
import { SchedulerService } from "../src/services/scheduler";
import { RunLock } from "../src/services/run-lock";
import { StateStore } from "../src/services/state-store";
import { vi } from "vitest";

describe("RunCancellationService", () => {
  it("scopes cancellation to dates active when cancellation was requested", () => {
    const cancellation = new RunCancellationService();
    const dateA = cancellation.begin("2026-05-11");

    expect(cancellation.cancelAll("stop A")).toEqual(["2026-05-11"]);
    const dateB = cancellation.begin("2026-05-12");
    cancellation.finish("2026-05-11");
    const dateC = cancellation.begin("2026-05-13");

    expect(dateA.aborted).toBe(true);
    expect(dateB.aborted).toBe(false);
    expect(dateC.aborted).toBe(false);
  });

  it("marks only the active batch as cancelled", () => {
    const cancellation = new RunCancellationService();
    const batchA = cancellation.beginBatch();
    const signalA = cancellation.begin("2026-05-11", batchA);

    cancellation.cancelAll("stop A");
    const batchB = cancellation.beginBatch();
    const signalB = cancellation.begin("2026-05-12", batchB);

    expect(signalA.aborted).toBe(true);
    expect(batchA.isCancellationRequested()).toBe(true);
    expect(signalB.aborted).toBe(false);
    expect(batchB.isCancellationRequested()).toBe(false);
  });

  it("aborts overlapping begin for the same cancelled batch only", () => {
    const cancellation = new RunCancellationService();
    const batch = cancellation.beginBatch();
    const first = cancellation.begin("2026-05-11", batch);
    cancellation.cancelAll("stop");
    const second = cancellation.begin("2026-05-11", batch);
    const futureBatch = cancellation.beginBatch();
    const future = cancellation.begin("2026-05-11", futureBatch);

    expect(first.aborted).toBe(true);
    expect(second.aborted).toBe(true);
    expect(future.aborted).toBe(false);
  });

  it("allows a later scheduler tick after cancelAll stopped an earlier batch", async () => {
    const cancellation = new RunCancellationService();
    const data: any = { runState: {} };
    const store = new StateStore(
      async () => ({ runState: { ...data.runState } }),
      async (next) => {
        data.runState = { ...next.runState };
      },
    );
    const runForDate = vi
      .fn()
      .mockImplementationOnce(async () => {
        cancellation.cancelAll("stop current tick");
        return { kind: "pending", reason: "cancelled test batch" };
      })
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const scheduler = new SchedulerService({
      getSettings: () => ({
        schedule: {
          enabled: true,
          runAtLocal: "00:00",
          runUntilLocal: "23:59",
          tickIntervalMin: 1,
        },
        arxiv: { timezone: "UTC" },
      } as any),
      store,
      lock: new RunLock(),
      runForDate,
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        notice: vi.fn(),
        debug: vi.fn(),
      } as any,
      progress: {
        setBatch: vi.fn(),
        setTask: vi.fn(),
        setComplete: vi.fn(),
        setError: vi.fn(),
        setIdle: vi.fn(),
      } as any,
      cancellation,
      history: undefined,
      now: () => new Date("2026-06-22T10:00:00Z"),
    } as any);

    await scheduler.tick();
    await scheduler.tick();

    expect(runForDate.mock.calls.slice(0, 2).map(([date]) => date))
      .toEqual(["2026-06-22", "2026-06-22"]);
    expect(store.get("2026-06-22").status).toBe("completed");
  });
});
