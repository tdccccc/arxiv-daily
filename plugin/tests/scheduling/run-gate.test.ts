import { describe, expect, it } from "vitest";
import { StateStore } from "../../src/services/state-store";
import {
  checkTickGate,
  isRunning,
  isWithinTimeGate,
  shouldBackoffTransient,
} from "../../src/services/scheduling/run-gate";

function makeStore() {
  const data = { runState: {} as Record<string, any> };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

describe("run-gate", () => {
  it("checkTickGate allows a fresh pending date inside the window", async () => {
    const store = makeStore();
    await store.load();
    const decision = checkTickGate("2026-05-11", store, {
      now: new Date("2026-05-11T05:00:00Z"),
      tickIntervalMin: 20,
      timeGate: { scheduledMin: 540, endMin: 1080, minutesNow: 780 }, // 13:00 inside 09:00-18:00
    });
    expect(decision.allow).toBe(true);
  });

  it("checkTickGate denies already-done dates", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 3);
    const decision = checkTickGate("2026-05-11", store, {
      now: new Date("2026-05-11T05:00:00Z"),
      tickIntervalMin: 20,
      timeGate: { scheduledMin: 540, endMin: 1080, minutesNow: 780 },
    });
    expect(decision.allow).toBe(false);
    if (!decision.allow) expect(decision.reason).toBe("already-done");
  });

  it("checkTickGate denies a running date", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    const decision = checkTickGate("2026-05-11", store, {
      now: new Date("2026-05-11T05:00:00Z"),
      tickIntervalMin: 20,
    });
    expect(decision.allow).toBe(false);
    if (!decision.allow) expect(decision.reason).toBe("running");
  });

  it("checkTickGate denies outside the timeGate window", async () => {
    const store = makeStore();
    await store.load();
    const decision = checkTickGate("2026-05-11", store, {
      now: new Date("2026-05-11T00:00:00Z"),
      tickIntervalMin: 20,
      timeGate: { scheduledMin: 540, endMin: 1080, minutesNow: 60 }, // 01:00 outside 09:00-18:00
    });
    expect(decision.allow).toBe(false);
    if (!decision.allow) expect(decision.reason).toBe("outside-window");
  });

  it("checkTickGate applies transient backoff when lastAttempt is too recent", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setFailed("2026-05-11", "transient", "x");
    const justNow = new Date(store.get("2026-05-11").lastAttempt + 5 * 60 * 1000);
    const decision = checkTickGate("2026-05-11", store, {
      now: justNow,
      tickIntervalMin: 20,
    });
    expect(decision.allow).toBe(false);
    if (!decision.allow) expect(decision.reason).toBe("transient-backoff");
  });

  it("checkTickGate allows transient after backoff elapsed", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    await store.setFailed("2026-05-11", "transient", "x");
    const later = new Date(store.get("2026-05-11").lastAttempt + 60 * 60 * 1000);
    const decision = checkTickGate("2026-05-11", store, {
      now: later,
      tickIntervalMin: 20,
    });
    expect(decision.allow).toBe(true);
  });

  it("atomic guards", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-11");
    expect(isRunning("2026-05-11", store)).toBe(true);
    expect(isWithinTimeGate(780, 540, 1080)).toBe(true);
    expect(isWithinTimeGate(60, 540, 1080)).toBe(false);

    await store.setFailed("2026-05-11", "transient", "x");
    const failedAt = store.get("2026-05-11").lastAttempt;
    expect(shouldBackoffTransient("2026-05-11", store, 20, new Date(failedAt + 10 * 60 * 1000))).toBe(true);
    expect(shouldBackoffTransient("2026-05-11", store, 20, new Date(failedAt + 21 * 60 * 1000))).toBe(false);
  });
});
