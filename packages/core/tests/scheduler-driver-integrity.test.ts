import { describe, expect, it, vi } from "vitest";
import { SchedulerService } from "../src/services/scheduler";
import { SchedulerDriver } from "../src/services/scheduling/scheduler-driver";
import { Logger } from "../src/services/logger";
import { StateStore } from "../src/services/state-store";
import { RunCancellationService } from "../src/services/cancellation";
import { RunLock } from "../src/services/run-lock";
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

function completedResult(date = "2026-05-12") {
  return {
    kind: "completed" as const,
    papersWritten: 2,
    digest: {
      date,
      summaryLanguage: "en" as const,
      categories: "astro-ph",
      dailyPath: `arxiv-daily/daily/${date}.md`,
      paperCount: 2,
      topics: [],
    },
  };
}

function makeProgress() {
  return {
    setTask: vi.fn(),
    setBatch: vi.fn(),
    setStage: vi.fn(),
    setComplete: vi.fn(),
    setError: vi.fn(),
    setIdle: vi.fn(),
    setDisabled: vi.fn(),
  };
}

function makeHistoryRecorder() {
  return {
    recorder: {
      recordStarted: vi.fn(async () => undefined),
      recordCompleted: vi.fn(async () => undefined),
      recordPending: vi.fn(async () => undefined),
      recordCancelled: vi.fn(async () => undefined),
      recordSkippedForDate: vi.fn(async () => undefined),
      recordSkipped: vi.fn(async () => undefined),
      recordFailed: vi.fn(async () => undefined),
    },
  };
}

function recordsFor(
  records: RunHistoryRecord[],
  date: string,
  event: RunHistoryRecord["event"],
): RunHistoryRecord[] {
  return records.filter((record) => record.date === date && record.event === event);
}

describe("SchedulerDriver integrity guards", () => {
  it.each(["save throws", "confirmation mismatch"] as const)(
    "does not expose a partially matching completion readback when %s",
    async (failureMode) => {
      const targetDate = "2026-05-12";
      const initialUnrelated = {
        status: "skipped" as const,
        lastAttempt: 2,
        attempts: 0,
        error: "operator",
      };
      const durable = {
        runState: {
          unrelated: { ...initialUnrelated },
        } as Record<string, any>,
      };
      const commitFailure = new Error("fsync failed");
      const store = new StateStore(
        async () => ({ runState: structuredClone(durable.runState) }),
        async ({ runState }) => {
          durable.runState = structuredClone(runState);
          if (runState[targetDate]?.status !== "completed") return;
          durable.runState.unrelated = {
            ...durable.runState.unrelated,
            lastAttempt: 3,
          };
          if (failureMode === "save throws") throw commitFailure;
        },
      );
      await store.load();
      const logger = new Logger("error");
      const notice = vi.spyOn(logger, "notice");
      const progress = makeProgress();
      const history = makeHistoryRecorder();
      const onDailyCompleted = vi.fn(async () => undefined);
      const targetResult = completedResult(targetDate);
      const runForDate = vi.fn(async () => targetResult);
      const driver = new SchedulerDriver({
        getSettings: () => DEFAULT_SETTINGS,
        store,
        lock: new RunLock(),
        runForDate,
        logger,
        progress,
        history: history.recorder,
        onDailyCompleted,
      });

      await expect(driver.runForDateNow(targetDate)).resolves.toEqual({
        kind: "failed_transient",
        reason: "scheduler completion commit failed",
      });

      expect(durable.runState[targetDate]?.status).toBe("completed");
      expect(store.get(targetDate).status).toBe("running");
      expect(store.get("unrelated")).toEqual(initialUnrelated);
      expect(runForDate).toHaveBeenCalledTimes(1);
      expect(notice).not.toHaveBeenCalledWith(
        expect.stringContaining(`arXiv ${targetDate}:`),
      );
      expect(progress.setComplete).not.toHaveBeenCalled();
      expect(progress.setIdle).not.toHaveBeenCalledWith(targetDate);
      expect(history.recorder.recordCompleted).not.toHaveBeenCalled();
      expect(onDailyCompleted).not.toHaveBeenCalled();
    },
  );

  it("retains a failed completion candidate across batch/manual retries and finalizes its original digest once", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    let completionStorageAvailable = false;
    const setCompleted = vi
      .spyOn(store, "setCompleted")
      .mockImplementation(async (date, papersWritten) => {
        if (date === targetDate && !completionStorageAvailable) {
          throw new Error("disk full");
        }
        await originalSetCompleted(date, papersWritten);
      });
    const setPending = vi.spyOn(store, "setPending");
    const logger = new Logger("error");
    const logNotice = vi.spyOn(logger, "notice");
    const progress = makeProgress();
    const historyRecords: RunHistoryRecord[] = [];
    const runHistory = {
      safeAppend: vi.fn(async (record: RunHistoryRecord) => {
        historyRecords.push(record);
      }),
    };
    const onDailyCompleted = vi.fn(async () => undefined);
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async (date: string) =>
      date === targetDate
        ? targetResult
        : { kind: "completed" as const, papersWritten: 1 },
    );
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger,
      progress,
      runHistory,
      onDailyCompleted,
      now: () => new Date("2026-05-12T05:00:00Z"),
    });

    const firstBatch = await svc.runAllPending();

    expect(firstBatch.map((entry) => entry.date)).toEqual([
      "2026-05-12",
      "2026-05-11",
      "2026-05-10",
      "2026-05-09",
      "2026-05-08",
    ]);
    expect(firstBatch[0]).toEqual({
      date: targetDate,
      result: {
        kind: "failed_transient",
        reason: "scheduler completion commit failed",
      },
    });
    expect(firstBatch.slice(1).every((entry) => entry.result.kind === "completed"))
      .toBe(true);
    expect(store.get(targetDate).status).toBe("running");
    expect(setPending).not.toHaveBeenCalledWith(
      targetDate,
      "scheduler completion commit failed",
    );
    expect(runForDate.mock.calls.filter(([date]) => date === targetDate)).toHaveLength(1);
    expect(onDailyCompleted).not.toHaveBeenCalledWith(targetDate, expect.anything());
    expect(logNotice).not.toHaveBeenCalledWith(
      expect.stringContaining(`arXiv ${targetDate}:`),
    );
    expect(progress.setComplete).not.toHaveBeenCalledWith(
      `Daily report complete: ${targetDate}`,
    );
    expect(recordsFor(historyRecords, targetDate, "started")).toHaveLength(1);
    expect(recordsFor(historyRecords, targetDate, "completed")).toHaveLength(0);
    expect(recordsFor(historyRecords, targetDate, "failed")).toEqual([
      expect.objectContaining({
        status: "failed_transient",
        resultKind: "failed_transient",
        reason: "scheduler completion commit failed",
      }),
    ]);

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    expect(runForDate.mock.calls.filter(([date]) => date === targetDate)).toHaveLength(1);
    expect(recordsFor(historyRecords, targetDate, "started")).toHaveLength(1);
    expect(recordsFor(historyRecords, targetDate, "failed")).toHaveLength(2);
    expect(recordsFor(historyRecords, targetDate, "completed")).toHaveLength(0);
    expect(onDailyCompleted).not.toHaveBeenCalledWith(targetDate, expect.anything());

    completionStorageAvailable = true;
    const recoveryBatch = await svc.runAllPending();

    expect(recoveryBatch).toHaveLength(1);
    expect(recoveryBatch[0]?.date).toBe(targetDate);
    expect(recoveryBatch[0]?.result).toBe(targetResult);
    expect(runForDate.mock.calls.filter(([date]) => date === targetDate)).toHaveLength(1);
    expect(store.get(targetDate)).toMatchObject({
      status: "completed",
      papersWritten: 2,
    });
    expect(recordsFor(historyRecords, targetDate, "completed")).toEqual([
      expect.objectContaining({
        status: "completed",
        resultKind: "completed",
        papersWritten: 2,
        requestedPapersWritten: 2,
      }),
    ]);
    const targetCallbacks = onDailyCompleted.mock.calls.filter(
      ([date]) => date === targetDate,
    );
    expect(targetCallbacks).toHaveLength(1);
    expect(targetCallbacks[0]?.[1]).toBe(targetResult);
    expect(targetCallbacks[0]?.[1].digest).toBe(targetResult.digest);

    await svc.tickToday();

    expect(setCompleted.mock.calls.filter(([date]) => date === targetDate)).toHaveLength(3);
    expect(runForDate.mock.calls.filter(([date]) => date === targetDate)).toHaveLength(1);
    expect(onDailyCompleted.mock.calls.filter(([date]) => date === targetDate)).toHaveLength(1);
    expect(recordsFor(historyRecords, targetDate, "completed")).toHaveLength(1);
  });

  it("retries a pending completion through a normal scheduled entry without rerunning the pipeline", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    vi.spyOn(store, "setCompleted")
      .mockRejectedValueOnce(new Error("disk full"))
      .mockImplementation(originalSetCompleted);
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const onDailyCompleted = vi.fn(async () => undefined);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      onDailyCompleted,
      now: () => new Date("2026-05-12T05:00:00Z"),
    });

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    await expect(svc.tickToday()).resolves.toBe(targetResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted.mock.calls[0]?.[1]).toBe(targetResult);
    expect(store.get(targetDate).status).toBe("completed");
  });

  it("retries an older pending completion on scheduler tick even after it leaves lookback", async () => {
    const targetDate = "2026-04-01";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    vi.spyOn(store, "setCompleted")
      .mockRejectedValueOnce(new Error("disk full"))
      .mockImplementation(originalSetCompleted);
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const onDailyCompleted = vi.fn(async () => undefined);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      onDailyCompleted,
      now: () => new Date("2026-05-12T05:00:00Z"),
    });

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    await svc.tick();

    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted.mock.calls[0]?.[1]).toBe(targetResult);
    expect(store.get(targetDate).status).toBe("completed");
  });

  it("coalesces concurrent pending retry waiters onto the original completed result", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    let markRecoveredCommitStarted!: () => void;
    const recoveredCommitStarted = new Promise<void>((resolve) => {
      markRecoveredCommitStarted = resolve;
    });
    let releaseRecoveredCommit!: () => void;
    const recoveredCommitCanFinish = new Promise<void>((resolve) => {
      releaseRecoveredCommit = resolve;
    });
    let completionAttempts = 0;
    vi.spyOn(store, "setCompleted").mockImplementation(
      async (date, papersWritten) => {
        completionAttempts += 1;
        if (completionAttempts === 1) throw new Error("disk full");
        if (completionAttempts === 2) {
          markRecoveredCommitStarted();
          await recoveredCommitCanFinish;
        }
        await originalSetCompleted(date, papersWritten);
      },
    );
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const historyRecords: RunHistoryRecord[] = [];
    const onDailyCompleted = vi.fn(async () => undefined);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      runHistory: {
        safeAppend: vi.fn(async (record: RunHistoryRecord) => {
          historyRecords.push(record);
        }),
      },
      onDailyCompleted,
    });

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    const firstWaiter = svc.runForDateNow(targetDate);
    await recoveredCommitStarted;
    const secondWaiter = svc.runForDateNow(targetDate);
    releaseRecoveredCommit();

    await expect(firstWaiter).resolves.toBe(targetResult);
    await expect(secondWaiter).resolves.toBe(targetResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted.mock.calls[0]?.[1]).toBe(targetResult);
    expect(recordsFor(historyRecords, targetDate, "completed")).toHaveLength(1);
    expect(store.get(targetDate).status).toBe("completed");
  });

  it("does not rerun a normal entry whose pre-lock eligibility became stale after successful completion", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    let releaseFirstRefresh!: () => void;
    const firstRefreshCanFinish = new Promise<void>((resolve) => {
      releaseFirstRefresh = resolve;
    });
    let markFirstRefreshStarted!: () => void;
    const firstRefreshStarted = new Promise<void>((resolve) => {
      markFirstRefreshStarted = resolve;
    });
    let refreshCalls = 0;
    const recentDates = {
      refresh: vi.fn(async () => {
        refreshCalls += 1;
        if (refreshCalls === 1) {
          markFirstRefreshStarted();
          await firstRefreshCanFinish;
        }
      }),
    };
    const targetResult = completedResult(targetDate);
    const rerunResult = { kind: "completed" as const, papersWritten: 99 };
    const runForDate = vi
      .fn()
      .mockResolvedValueOnce(targetResult)
      .mockResolvedValue(rerunResult);
    const historyRecords: RunHistoryRecord[] = [];
    const onDailyCompleted = vi.fn(async () => undefined);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      recentDates,
      runHistory: {
        safeAppend: vi.fn(async (record: RunHistoryRecord) => {
          historyRecords.push(record);
        }),
      },
      onDailyCompleted,
    });

    const staleEntry = svc.runForDateNow(targetDate);
    await firstRefreshStarted;

    await expect(svc.runForDateNow(targetDate)).resolves.toBe(targetResult);
    releaseFirstRefresh();

    await expect(staleEntry).resolves.toEqual({
      kind: "skipped",
      reason: "already done",
    });
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(recordsFor(historyRecords, targetDate, "started")).toHaveLength(1);
    expect(recordsFor(historyRecords, targetDate, "completed")).toHaveLength(1);
    expect(recordsFor(historyRecords, targetDate, "skipped")).toEqual([
      expect.objectContaining({ reason: "already done" }),
    ]);
    expect(store.get(targetDate)).toMatchObject({
      status: "completed",
      papersWritten: 2,
    });
  });

  it("rechecks pending completion inside the date lock before a concurrent entry can rerun the pipeline", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    vi.spyOn(store, "setCompleted")
      .mockRejectedValueOnce(new Error("disk full"))
      .mockImplementation(originalSetCompleted);
    let releaseFirstRefresh!: () => void;
    const firstRefreshCanFinish = new Promise<void>((resolve) => {
      releaseFirstRefresh = resolve;
    });
    let markFirstRefreshStarted!: () => void;
    const firstRefreshStarted = new Promise<void>((resolve) => {
      markFirstRefreshStarted = resolve;
    });
    let refreshCalls = 0;
    const recentDates = {
      refresh: vi.fn(async () => {
        refreshCalls += 1;
        if (refreshCalls === 1) {
          markFirstRefreshStarted();
          await firstRefreshCanFinish;
        }
      }),
    };
    const firstResult = completedResult(targetDate);
    const rerunResult = { kind: "completed" as const, papersWritten: 99 };
    const runForDate = vi
      .fn()
      .mockResolvedValueOnce(firstResult)
      .mockResolvedValue(rerunResult);
    const onDailyCompleted = vi.fn(async () => undefined);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      recentDates,
      onDailyCompleted,
    });

    const delayedEntry = svc.runForDateNow(targetDate);
    await firstRefreshStarted;

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });
    releaseFirstRefresh();

    await expect(delayedEntry).resolves.toBe(firstResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted.mock.calls[0]?.[1]).toBe(firstResult);
    expect(store.get(targetDate)).toMatchObject({
      status: "completed",
      papersWritten: 2,
    });
  });

  it("coalesces a waiter with pending finalization discovered inside the date lock", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    let markRecoveredCommitStarted!: () => void;
    const recoveredCommitStarted = new Promise<void>((resolve) => {
      markRecoveredCommitStarted = resolve;
    });
    let releaseRecoveredCommit!: () => void;
    const recoveredCommitCanFinish = new Promise<void>((resolve) => {
      releaseRecoveredCommit = resolve;
    });
    let completionAttempts = 0;
    vi.spyOn(store, "setCompleted").mockImplementation(
      async (date, papersWritten) => {
        completionAttempts += 1;
        if (completionAttempts === 1) throw new Error("disk full");
        if (completionAttempts === 2) {
          markRecoveredCommitStarted();
          await recoveredCommitCanFinish;
        }
        await originalSetCompleted(date, papersWritten);
      },
    );
    let releaseFirstRefresh!: () => void;
    const firstRefreshCanFinish = new Promise<void>((resolve) => {
      releaseFirstRefresh = resolve;
    });
    let markFirstRefreshStarted!: () => void;
    const firstRefreshStarted = new Promise<void>((resolve) => {
      markFirstRefreshStarted = resolve;
    });
    let refreshCalls = 0;
    const recentDates = {
      refresh: vi.fn(async () => {
        refreshCalls += 1;
        if (refreshCalls === 1) {
          markFirstRefreshStarted();
          await firstRefreshCanFinish;
        }
      }),
    };
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const historyRecords: RunHistoryRecord[] = [];
    const onDailyCompleted = vi.fn(async () => undefined);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      recentDates,
      runHistory: {
        safeAppend: vi.fn(async (record: RunHistoryRecord) => {
          historyRecords.push(record);
        }),
      },
      onDailyCompleted,
    });

    const delayedEntry = svc.runForDateNow(targetDate);
    await firstRefreshStarted;
    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    releaseFirstRefresh();
    await recoveredCommitStarted;
    const concurrentWaiter = svc.runForDateNow(targetDate);
    releaseRecoveredCommit();

    await expect(delayedEntry).resolves.toBe(targetResult);
    await expect(concurrentWaiter).resolves.toBe(targetResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted.mock.calls[0]?.[1]).toBe(targetResult);
    expect(recordsFor(historyRecords, targetDate, "completed")).toHaveLength(1);
    expect(store.get(targetDate).status).toBe("completed");
  });

  it.each(["logger", "progress", "history"] as const)(
    "never returns an unconfirmed completion when %s failure observability throws",
    async (throwingEffect) => {
      const targetDate = "2026-05-12";
      const store = makeStore();
      await store.load();
      vi.spyOn(store, "setCompleted").mockRejectedValue(new Error("disk full"));
      const logger = {
        debug: vi.fn(),
        info: vi.fn(),
        notice: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(() => {
          if (throwingEffect === "logger") throw new Error("logger failed");
        }),
      };
      const progress = makeProgress();
      if (throwingEffect === "progress") {
        progress.setError.mockImplementation(() => {
          throw new Error("progress failed");
        });
      }
      const history = {
        recordStarted: vi.fn(async () => undefined),
        recordCompleted: vi.fn(async () => undefined),
        recordPending: vi.fn(async () => undefined),
        recordCancelled: vi.fn(async () => undefined),
        recordSkippedForDate: vi.fn(async () => undefined),
        recordSkipped: vi.fn(async () => undefined),
        recordFailed: vi.fn(async () => {
          if (throwingEffect === "history") throw new Error("history failed");
        }),
      };
      const targetResult = completedResult(targetDate);
      const runForDate = vi.fn(async () => targetResult);
      const driver = new SchedulerDriver({
        getSettings: () => DEFAULT_SETTINGS,
        store,
        lock: new RunLock(),
        runForDate,
        logger: logger as never,
        progress,
        history: history as never,
      });

      await expect(driver.runForDateNow(targetDate)).resolves.toEqual({
        kind: "failed_transient",
        reason: "scheduler completion commit failed",
      });
      await expect(driver.runForDateNow(targetDate)).resolves.toEqual({
        kind: "failed_transient",
        reason: "scheduler completion commit failed",
      });

      expect(runForDate).toHaveBeenCalledTimes(1);
      expect(store.get(targetDate).status).toBe("running");
    },
  );

  it("returns the original completed object when cancellation batch cleanup throws", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const cancellation = new RunCancellationService();
    const finishBatch = vi
      .spyOn(cancellation, "finishBatch")
      .mockImplementation(() => {
        throw new Error("batch cleanup failed");
      });
    const progress = makeProgress();
    const history = makeHistoryRecorder();
    const onDailyCompleted = vi.fn(async () => undefined);
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const driver = new SchedulerDriver({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      progress,
      cancellation,
      history: history.recorder,
      onDailyCompleted,
    });

    await expect(driver.runForDateNow(targetDate)).resolves.toBe(targetResult);

    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(history.recorder.recordCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted.mock.calls[0]?.[1]).toBe(targetResult);
    expect(progress.setComplete).toHaveBeenCalledTimes(1);
    expect(progress.setIdle).toHaveBeenLastCalledWith(targetDate);
    expect(store.get(targetDate).status).toBe("completed");

    const laterDate = "2026-05-13";
    const laterResult = completedResult(laterDate);
    runForDate.mockResolvedValueOnce(laterResult);
    await expect(driver.runForDateNow(laterDate)).resolves.toBe(laterResult);

    expect(finishBatch).toHaveBeenCalledTimes(2);
    expect(runForDate).toHaveBeenCalledTimes(2);
    expect(history.recorder.recordCompleted).toHaveBeenCalledTimes(2);
    expect(onDailyCompleted).toHaveBeenCalledTimes(2);
    expect(store.get(laterDate).status).toBe("completed");
  });

  it.each(["beginBatch", "prepareRun", "begin", "finish"] as const)(
    "treats cancellation %s failure as best effort",
    async (throwingMethod) => {
      const targetDate = "2026-05-12";
      const store = makeStore();
      await store.load();
      const cancellation = new RunCancellationService();
      vi.spyOn(cancellation, throwingMethod).mockImplementation(() => {
        throw new Error(`${throwingMethod} failed`);
      });
      const targetResult = completedResult(targetDate);
      const runForDate = vi.fn(async () => targetResult);
      const history = makeHistoryRecorder();
      const driver = new SchedulerDriver({
        getSettings: () => DEFAULT_SETTINGS,
        store,
        lock: new RunLock(),
        runForDate,
        logger: new Logger("error"),
        cancellation,
        history: history.recorder,
      });

      await expect(driver.runForDateNow(targetDate)).resolves.toBe(targetResult);

      expect(runForDate).toHaveBeenCalledTimes(1);
      expect(history.recorder.recordCompleted).toHaveBeenCalledTimes(1);
      expect(store.get(targetDate).status).toBe("completed");
    },
  );

  it("runs every post-commit effect once even when earlier process-local best-effort effects throw", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const logger = {
      debug: vi.fn(),
      info: vi.fn(),
      notice: vi.fn(() => {
        throw new Error("notice failed");
      }),
      warn: vi.fn(),
      error: vi.fn(() => {
        throw new Error("error logger failed");
      }),
    };
    const progress = makeProgress();
    progress.setComplete.mockImplementation(() => {
      throw new Error("progress failed");
    });
    const history = {
      recordStarted: vi.fn(async () => undefined),
      recordCompleted: vi.fn(async () => {
        throw new Error("history failed");
      }),
      recordPending: vi.fn(async () => undefined),
      recordCancelled: vi.fn(async () => undefined),
      recordSkippedForDate: vi.fn(async () => undefined),
      recordSkipped: vi.fn(async () => undefined),
      recordFailed: vi.fn(async () => undefined),
    };
    const onDailyCompleted = vi.fn(async () => {
      throw new Error("callback failed");
    });
    const targetResult = completedResult(targetDate);
    const driver = new SchedulerDriver({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate: vi.fn(async () => targetResult),
      logger: logger as never,
      progress,
      history: history as never,
      onDailyCompleted,
    });

    await expect(driver.runForDateNow(targetDate)).resolves.toBe(targetResult);
    await expect(driver.runForDateNow(targetDate)).resolves.toEqual({
      kind: "skipped",
      reason: "already done",
    });

    expect(store.get(targetDate).status).toBe("completed");
    expect(logger.notice).toHaveBeenCalledTimes(1);
    expect(progress.setComplete).toHaveBeenCalledTimes(1);
    expect(history.recordCompleted).toHaveBeenCalledTimes(1);
    expect(onDailyCompleted).toHaveBeenCalledTimes(1);
  });

  it("reloads authoritative state under the date lock before final running and done gates", async () => {
    const targetDate = "2026-05-12";
    const durable = {
      runState: {
        [targetDate]: { status: "running" as const, lastAttempt: 1, attempts: 1 },
      },
    };
    const store = new StateStore(
      async () => ({ runState: structuredClone(durable.runState) }),
      async ({ runState }) => {
        durable.runState = structuredClone(runState) as typeof durable.runState;
      },
    );
    await store.load();
    delete (durable.runState as Record<string, unknown>)[targetDate];
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
    });

    await expect(svc.runForDateNow(targetDate)).resolves.toBe(targetResult);

    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(store.get(targetDate).status).toBe("completed");
  });

  it("reloads authoritative state before a scheduled backoff gate", async () => {
    const targetDate = "2026-05-12";
    const now = new Date("2026-05-12T05:00:00Z");
    const durable: { runState: Record<string, any> } = {
      runState: {
        [targetDate]: {
          status: "failed_transient",
          lastAttempt: now.getTime(),
          attempts: 1,
          error: "old",
        },
      },
    };
    const store = new StateStore(
      async () => ({ runState: structuredClone(durable.runState) }),
      async ({ runState }) => {
        durable.runState = structuredClone(runState);
      },
    );
    await store.load();
    durable.runState = {};
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const svc = new SchedulerService({
      getSettings: () => ({
        ...DEFAULT_SETTINGS,
        arxiv: { ...DEFAULT_SETTINGS.arxiv, timezone: "UTC" },
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          enabled: true,
          runAtLocal: "00:00",
          runUntilLocal: "23:59",
        },
      }),
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => now,
    });

    await expect(svc.tickTodayScheduled()).resolves.toBe(targetResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
  });

  it("rejects store replacement while a pipeline is active and commits to the original store", async () => {
    const targetDate = "2026-05-12";
    const oldStore = makeStore();
    const newStore = makeStore();
    await oldStore.load();
    await newStore.load();
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    let releaseRun!: () => void;
    const runCanFinish = new Promise<void>((resolve) => {
      releaseRun = resolve;
    });
    const targetResult = completedResult(targetDate);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store: oldStore,
      lock: new RunLock(),
      runForDate: vi.fn(async () => {
        markStarted();
        await runCanFinish;
        return targetResult;
      }),
      logger: new Logger("error"),
    });

    const running = svc.runForDateNow(targetDate);
    await started;
    expect(() => svc.replaceStore(newStore)).toThrow(
      "cannot replace scheduler store while work is active",
    );
    releaseRun();
    await expect(running).resolves.toBe(targetResult);

    expect(oldStore.get(targetDate).status).toBe("completed");
    expect(newStore.get(targetDate).status).toBe("pending");
  });

  it("rejects store replacement while an uncommitted completion is pending", async () => {
    const targetDate = "2026-05-12";
    const oldStore = makeStore();
    const newStore = makeStore();
    await oldStore.load();
    await newStore.load();
    vi.spyOn(oldStore, "setCompleted").mockRejectedValue(new Error("disk full"));
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store: oldStore,
      lock: new RunLock(),
      runForDate: vi.fn(async () => completedResult(targetDate)),
      logger: new Logger("error"),
    });

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    expect(() => svc.replaceStore(newStore)).toThrow(
      "cannot replace scheduler store while work is active",
    );
    expect(newStore.get(targetDate).status).toBe("pending");
  });

  it("turns a date load failure into failed_transient and continues a runAllPending batch", async () => {
    const store = makeStore();
    await store.load();
    const load = vi
      .spyOn(store, "loadAuthoritative")
      .mockRejectedValueOnce(new Error("state read failed"))
      .mockImplementation(StateStore.prototype.loadAuthoritative.bind(store));
    const runForDate = vi.fn(async (date: string) => completedResult(date));
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
      now: () => new Date("2026-05-12T05:00:00Z"),
    });

    const results = await svc.runAllPending();

    expect(results[0]).toEqual({
      date: "2026-05-12",
      result: { kind: "failed_transient", reason: "state read failed" },
    });
    expect(results.slice(1).some(({ result }) => result.kind === "completed")).toBe(true);
    expect(runForDate).not.toHaveBeenCalledWith("2026-05-12");
    expect(load).toHaveBeenCalled();
  });

  it("turns force clearDate failure into failed_transient without running the pipeline", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    await store.setCompleted(targetDate, 2);
    vi.spyOn(store, "clearDate").mockRejectedValueOnce(new Error("clear failed"));
    const runForDate = vi.fn(async () => completedResult(targetDate));
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
    });

    await expect(svc.forceRunForDate(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "clear failed",
    });

    expect(runForDate).not.toHaveBeenCalled();
  });

  it.each(["pending", "cancelled"] as const)(
    "retries an uncommitted %s transition without rerunning the pipeline",
    async (kind) => {
      const targetDate = "2026-05-12";
      const store = makeStore();
      await store.load();
      const originalSetPending = store.setPending.bind(store);
      vi.spyOn(store, "setPending")
        .mockRejectedValueOnce(new Error("pending write failed"))
        .mockImplementation(originalSetPending);
      const targetResult = {
        kind,
        reason: kind === "pending" ? "not ready" : "cancelled by user",
      } as const;
      const runForDate = vi.fn(async () => targetResult);
      const svc = new SchedulerService({
        getSettings: () => DEFAULT_SETTINGS,
        store,
        lock: new RunLock(),
        runForDate,
        logger: new Logger("error"),
      });

      await expect(svc.runForDateNow(targetDate)).resolves.toBe(targetResult);
      expect(store.get(targetDate).status).toBe("running");

      await expect(svc.runForDateNow(targetDate)).resolves.toBe(targetResult);
      expect(runForDate).toHaveBeenCalledTimes(1);
      expect(store.get(targetDate)).toMatchObject({
        status: "pending",
        error: targetResult.reason,
      });
    },
  );

  it("allows force to intentionally rerun a durably completed date", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    await store.setCompleted(targetDate, 2);
    const forcedResult = {
      ...completedResult(targetDate),
      papersWritten: 3,
    };
    const runForDate = vi.fn(async () => forcedResult);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
    });

    await expect(svc.forceRunForDate(targetDate)).resolves.toBe(forcedResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(store.get(targetDate)).toMatchObject({
      status: "completed",
      papersWritten: 3,
    });
  });

  it("retries a pending completion through force without clearing state or rerunning the pipeline", async () => {
    const targetDate = "2026-05-12";
    const store = makeStore();
    await store.load();
    const originalSetCompleted = store.setCompleted.bind(store);
    vi.spyOn(store, "setCompleted")
      .mockRejectedValueOnce(new Error("disk full"))
      .mockImplementation(originalSetCompleted);
    const clearDate = vi.spyOn(store, "clearDate");
    const targetResult = completedResult(targetDate);
    const runForDate = vi.fn(async () => targetResult);
    const svc = new SchedulerService({
      getSettings: () => DEFAULT_SETTINGS,
      store,
      lock: new RunLock(),
      runForDate,
      logger: new Logger("error"),
    });

    await expect(svc.runForDateNow(targetDate)).resolves.toEqual({
      kind: "failed_transient",
      reason: "scheduler completion commit failed",
    });

    await expect(svc.forceRunForDate(targetDate)).resolves.toBe(targetResult);
    expect(runForDate).toHaveBeenCalledTimes(1);
    expect(clearDate).not.toHaveBeenCalled();
    expect(store.get(targetDate).status).toBe("completed");
  });
});
