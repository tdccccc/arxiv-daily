import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../../src/services/scheduler";
import { createStorageStateStore } from "../../src/services/state-store";
import { RunHistoryStore, formatRunHistoryRecords } from "../../src/services/run-history";
import { RunLock } from "../../src/services/run-lock";
import { DEFAULT_SETTINGS } from "../../src/settings/defaults";
import type { PipelineResult } from "../../src/pipeline/pipeline";
import type { RunState } from "../../src/settings/types";
import type { StorageAdapter } from "../../src/core/adapters";

const date = "2026-10-02";
function harness(initial: RunState = {}) {
  const files: Record<string, string> = {};
  const storage: StorageAdapter = {
    normalizePath: p => p,
    exists: async p => p in files,
    mkdir: async () => {},
    readText: async p => files[p],
    writeText: async (p, text) => { files[p] = text; },
    rename: async (a, b) => { files[b] = files[a]; delete files[a]; },
    remove: async p => { delete files[p]; },
  };
  const store = createStorageStateStore(storage, DEFAULT_SETTINGS.output);
  const history = RunHistoryStore.fromStorage(storage, DEFAULT_SETTINGS.output);
  const runForDate = vi.fn<() => Promise<PipelineResult>>(async () => ({ kind: "pending", reason: "announcement unavailable", outcome: "awaiting_announcement" }));
  const onDailyCompleted = vi.fn(async () => {});
  const scheduler = new SchedulerService({
    store, lock: new RunLock(), getSettings: () => DEFAULT_SETTINGS,
    runForDate, runHistory: history, onDailyCompleted,
    logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(), notice: vi.fn() } as any,
    now: () => new Date("2026-10-02T12:00:00Z"),
  });
  return { storage, store, history, runForDate, scheduler, onDailyCompleted,
    initialize: () => store.replaceAll(initial) };
}

describe("Structured run outcomes", () => {
  it("persists waiting across restarts without consuming the failure budget", async () => {
    const h = harness();
    await h.initialize();
    for (let i = 0; i < 12; i++) await h.scheduler.runForDateNow(date);
    const fresh = createStorageStateStore(h.storage, DEFAULT_SETTINGS.output);
    await fresh.load();
    expect(fresh.get(date)).toMatchObject({ status: "pending", outcome: "awaiting_announcement", attempts: 12, failureAttempts: 0 });
    expect(fresh.isDone(date)).toBe(false);
    expect(h.onDailyCompleted).not.toHaveBeenCalled();
    expect((await h.history.readLatest()).filter(r => r.event === "pending")).toHaveLength(12);
    expect((await h.history.readLatest()).filter(r => r.event === "pending")).toContainEqual(expect.objectContaining({ outcome: "awaiting_announcement", attempts: 12, failureAttempts: 0 }));
    h.runForDate.mockResolvedValue({ kind: "failed_transient", reason: "network offline" });
    await h.scheduler.runForDateNow(date);
    expect(h.store.get(date)).toMatchObject({ status: "failed_transient", attempts: 13, failureAttempts: 1 });
    expect(h.store.get(date).outcome).toBeUndefined();
  });

  it.each(["no_updates", "no_matches"] as const)("persists completed %s distinctly in state and history", async outcome => {
    const h = harness();
    await h.initialize();
    h.runForDate.mockResolvedValue({ kind: "completed", papersWritten: 0, outcome });
    await h.scheduler.runForDateNow(date);
    const fresh = createStorageStateStore(h.storage, DEFAULT_SETTINGS.output);
    await fresh.load();
    expect(fresh.get(date)).toMatchObject({ status: "completed", papersWritten: 0, outcome });
    expect(fresh.isDone(date)).toBe(true);
    expect((await h.history.readLatest()).find(r => r.event === "completed")).toMatchObject({ outcome });
  });

  it("counts a crash after many waits as one real failure", async () => {
    const h = harness({ [date]: { status: "pending", lastAttempt: 1, attempts: 12, failureAttempts: 0, outcome: "awaiting_announcement" } });
    await h.initialize();
    await h.store.setRunning(date);
    expect(h.store.get(date).outcome).toBeUndefined();
    await h.store.recoverStaleRunning(Date.now() + 3_600_001);
    expect(h.store.get(date)).toMatchObject({ status: "failed_transient", attempts: 13, failureAttempts: 1 });
  });

  it("preserves the legacy retry ceiling when no explicit failure counter exists", async () => {
    const h = harness({ [date]: { status: "failed_transient", lastAttempt: 1, attempts: 9 } });
    await h.initialize();
    h.runForDate.mockResolvedValue({ kind: "failed_transient", reason: "network offline" });
    await h.scheduler.runForDateNow(date);
    expect(h.store.get(date)).toMatchObject({ status: "failed_permanent", attempts: 10, failureAttempts: 10 });
  });
  it("still exhausts ten actual failures after waiting and preserves every attempt", async () => {
    const h = harness();
    await h.initialize();
    for (let i = 0; i < 12; i++) await h.scheduler.runForDateNow(date);
    h.runForDate.mockResolvedValue({ kind: "failed_transient", reason: "network offline" });
    for (let i = 0; i < 9; i++) {
      await h.scheduler.runForDateNow(date);
      expect(h.store.get(date).status).toBe("failed_transient");
    }
    await h.scheduler.runForDateNow(date);
    expect(h.store.get(date)).toMatchObject({ status: "failed_permanent", attempts: 22, failureAttempts: 10 });
    expect(h.store.get(date).error).toContain("after 10 attempts");
  });

  it.each(["pending", "completed"] as const)("recovers the structured %s terminal write without rerunning the source", async kind => {
    const h = harness();
    await h.initialize();
    const outcome = kind === "pending" ? "awaiting_announcement" : "no_updates";
    h.runForDate.mockResolvedValue(kind === "pending"
      ? { kind, reason: "later", outcome: "awaiting_announcement" }
      : { kind, papersWritten: 0, outcome: "no_updates" });
    vi.spyOn(h.store, kind === "pending" ? "setPending" : "setCompleted")
      .mockRejectedValueOnce(new Error("disk busy"));
    await h.scheduler.runForDateNow(date);
    expect(h.store.get(date).status).toBe("running");
    await h.scheduler.runForDateNow(date);
    expect(h.runForDate).toHaveBeenCalledTimes(1);
    expect(h.store.get(date)).toMatchObject({ status: kind, outcome, failureAttempts: 0 });
    expect((await h.history.readLatest()).filter(r => r.event === kind)).toHaveLength(1);
  });

  it("includes structured outcome and actual failures in text history", async () => {
    const h = harness();
    await h.initialize();
    await h.scheduler.runForDateNow(date);
    const text = formatRunHistoryRecords(await h.history.readLatest());
    expect(text).toContain("outcome=awaiting_announcement");
    expect(text).toContain("failures=0");
  });

});
