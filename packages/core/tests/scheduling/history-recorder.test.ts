import { describe, expect, it, vi } from "vitest";
import type { RunHistoryRecord } from "../../src/services/run-history";
import { HistoryRecorder } from "../../src/services/scheduling/history-recorder";
import { StateStore } from "../../src/services/state-store";

function makeStore() {
  const data = { runState: {} as Record<string, any> };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

describe("HistoryRecorder", () => {
  it("appends a RunHistoryRecord via safeAppend with schema/at/dailyPath", async () => {
    const store = makeStore();
    await store.load();
    const safeAppend = vi.fn(async (_record: RunHistoryRecord) => {});
    const recorder = new HistoryRecorder({
      runHistory: { safeAppend },
      store: () => store,
      dailyPathForDate: (d) => `vault/arxiv/${d}.md`,
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await store.setRunning("2026-05-11");
    await recorder.recordCompleted("2026-05-11", "scheduler", {
      papersWritten: 3,
      requestedPapersWritten: 3,
      preservedPapersWritten: false,
    });
    expect(safeAppend).toHaveBeenCalledTimes(1);
    const rec = safeAppend.mock.calls[0][0];
    expect(rec.schemaVersion).toBe(1);
    expect(rec.at).toBe("2026-05-11T05:00:00.000Z");
    expect(rec.event).toBe("completed");
    expect(rec.date).toBe("2026-05-11");
    expect(rec.dailyPath).toBe("vault/arxiv/2026-05-11.md");
    expect(rec.papersWritten).toBe(3);
  });

  it("uses store getter so replaceStore stays coherent", async () => {
    const s1 = makeStore();
    await s1.load();
    const s2 = makeStore();
    await s2.load();
    await s2.setRunning("2026-06-01");
    const safeAppend = vi.fn(async (_record: RunHistoryRecord) => {});
    let current = s1;
    const recorder = new HistoryRecorder({
      runHistory: { safeAppend },
      store: () => current,
      dailyPathForDate: () => undefined,
      now: () => new Date("2026-06-01T00:00:00Z"),
    });
    current = s2;
    await recorder.recordStarted("2026-06-01", "scheduler");
    const rec = safeAppend.mock.calls[0][0];
    expect(rec.attempts).toBe(1);
  });

  it("records skipped with reason and current status", async () => {
    const store = makeStore();
    await store.load();
    await store.setSkipped("2026-05-11", "disabled");
    const safeAppend = vi.fn(async (_record: RunHistoryRecord) => {});
    const recorder = new HistoryRecorder({
      runHistory: { safeAppend },
      store: () => store,
      now: () => new Date("2026-05-11T05:00:00Z"),
    });
    await recorder.recordSkipped("2026-05-11", "scheduler", "disabled");
    const rec = safeAppend.mock.calls[0][0];
    expect(rec.event).toBe("skipped");
    expect(rec.reason).toBe("disabled");
  });

  it("is a no-op when runHistory is undefined (CLI path)", async () => {
    const store = makeStore();
    await store.load();
    const recorder = new HistoryRecorder({ store: () => store, now: () => new Date() });
    await recorder.recordCompleted("2026-05-11", "manual", {
      papersWritten: 1,
      requestedPapersWritten: 1,
      preservedPapersWritten: false,
    });
  });
});
