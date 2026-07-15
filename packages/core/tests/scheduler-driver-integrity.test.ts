import { describe, expect, it, vi } from "vitest";
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

describe("SchedulerDriver integrity guards", () => {
  it("continues a pending-date batch after one date fails to persist", async () => {
    const store = makeStore();
    await store.load();
    const setCompleted = vi.spyOn(store, "setCompleted");
    setCompleted.mockRejectedValueOnce(new Error("disk full"));
    const logger = new Logger("error");
    const logError = vi.spyOn(logger, "error");
    const runForDate = vi
      .fn()
      .mockResolvedValue({ kind: "completed", papersWritten: 1 });
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger,
      now: () => new Date("2026-05-12T05:00:00Z"),
    });

    const results = await svc.runAllPending();

    expect(runForDate).toHaveBeenCalledTimes(5);
    expect(results.map((r) => r.date)).toEqual([
      "2026-05-12",
      "2026-05-11",
      "2026-05-10",
      "2026-05-09",
      "2026-05-08",
    ]);
    expect(results.every((r) => r.result.kind === "completed")).toBe(true);
    expect(logError.mock.calls.flat().join(" ")).toContain(
      "failed to persist result for 2026-05-12",
    );
  });
});
