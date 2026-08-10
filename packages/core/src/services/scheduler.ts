import type { PipelineResult } from "../pipeline/pipeline";
import type { PluginSettings } from "../settings/types";
import type { RunCancellationService } from "./cancellation";
import type { Logger } from "./logger";
import { NoopProgressReporter, type ProgressReporter } from "./progress";
import type { RunHistoryStore, RunHistoryTrigger } from "./run-history";
import type { RunLock } from "./run-lock";
import { HistoryRecorder, type HistoryRecorderDeps } from "./scheduling/history-recorder";
import { SchedulerDriver } from "./scheduling/scheduler-driver";
import type { StateStore } from "./state-store";

export interface SchedulerRunOptions {
  trigger?: RunHistoryTrigger;
}

export interface SchedulerRecentDates {
  refresh: (signal?: AbortSignal) => Promise<unknown>;
  hasDate?: (date: string) => boolean;
}

export class SchedulerPersistenceBinding {
  #pair: {
    readonly stateStore: StateStore;
    readonly runHistory?: Pick<RunHistoryStore, "safeAppend">;
  };

  constructor(
    stateStore: StateStore,
    runHistory?: Pick<RunHistoryStore, "safeAppend">,
  ) {
    this.#pair = { stateStore, runHistory };
  }

  get stateStore(): StateStore {
    return this.#pair.stateStore;
  }

  get runHistory(): Pick<RunHistoryStore, "safeAppend"> | undefined {
    return this.#pair.runHistory;
  }

  replace(
    stateStore: StateStore,
    runHistory: Pick<RunHistoryStore, "safeAppend">,
  ): void {
    assertPersistenceStorePair(stateStore, runHistory);
    this.#pair = { stateStore, runHistory };
  }
}

export interface SchedulerDeps {
  getSettings: () => PluginSettings;
  store: StateStore;
  persistence?: SchedulerPersistenceBinding;
  lock: RunLock;
  runForDate: (date: string, signal?: AbortSignal) => Promise<PipelineResult>;
  logger: Logger;
  now?: () => Date;
  progress?: ProgressReporter;
  cancellation?: RunCancellationService;
  recentDates?: SchedulerRecentDates;
  runHistory?: Pick<RunHistoryStore, "safeAppend">;
  dailyPathForDate?: (date: string) => string;
  onDailyCompleted?: (
    date: string,
    result: Extract<PipelineResult, { kind: "completed" }>,
  ) => Promise<void>;
}

type SchedulerResult = PipelineResult | { kind: "skipped"; reason: string };

export class SchedulerService {
  private driver: SchedulerDriver;
  private readonly persistence: SchedulerPersistenceBinding;
  private readonly historyDeps: HistoryRecorderDeps;

  constructor(deps: SchedulerDeps) {
    this.persistence = deps.persistence ?? new SchedulerPersistenceBinding(
      deps.store,
      deps.runHistory,
    );
    const persistence = this.persistence;
    this.historyDeps = {
      get runHistory() {
        return persistence.runHistory;
      },
      store: () => persistence.stateStore,
      dailyPathForDate: deps.dailyPathForDate,
      now: deps.now,
      logger: deps.logger,
    };
    const history = new HistoryRecorder(this.historyDeps);
    this.driver = new SchedulerDriver({
      getSettings: deps.getSettings,
      store: deps.store,
      getStore: () => this.persistence.stateStore,
      lock: deps.lock,
      runForDate: deps.runForDate,
      logger: deps.logger,
      progress: deps.progress ?? new NoopProgressReporter(),
      cancellation: deps.cancellation,
      history,
      recentDates: deps.recentDates,
      now: deps.now,
      onDailyCompleted: deps.onDailyCompleted,
    });
    void this.recoverStaleRunning();
  }

  start(): void {
    void this.recoverStaleRunning();
    this.driver.start();
  }

  stop(): void {
    this.driver.stop();
  }

  replacePersistenceStores(
    stateStore: StateStore,
    runHistory: Pick<RunHistoryStore, "safeAppend">,
  ): void {
    if (this.activeRuns().length > 0) {
      throw new Error("Scheduler persistence stores cannot change while runs are active");
    }
    this.persistence.replace(stateStore, runHistory);
  }

  /** Compatibility for Core callers that replace only state persistence. */
  replaceStore(store: StateStore): void {
    this.replacePersistenceStores(
      store,
      this.persistence.runHistory ?? { safeAppend: async () => undefined },
    );
  }

  /** Compatibility for Core callers that replace only run history. */
  replaceRunHistory(runHistory: Pick<RunHistoryStore, "safeAppend">): void {
    this.replacePersistenceStores(this.persistence.stateStore, runHistory);
  }

  cancelCurrentRun(reason = "cancelled by user"): string[] {
    return this.driver.cancelCurrentRun(reason);
  }

  activeRuns(): string[] {
    return this.driver.activeRuns();
  }

  async tick(): Promise<void> {
    return this.driver.tick();
  }

  async tickToday(): Promise<SchedulerResult | undefined> {
    return this.driver.tickToday();
  }

  async tickTodayScheduled(): Promise<SchedulerResult | undefined> {
    return this.driver.tickTodayScheduled();
  }

  async runForDateNow(
    date: string,
    opts: SchedulerRunOptions = {},
  ): Promise<SchedulerResult> {
    return this.driver.runForDateNow(date, opts);
  }

  async forceRunForDate(date: string): Promise<SchedulerResult> {
    return this.driver.forceRunForDate(date);
  }

  async retryFailedInLookback(): Promise<Array<{ date: string; result: SchedulerResult }>> {
    return this.driver.retryFailedInLookback();
  }

  async runAllPending(): Promise<Array<{ date: string; result: SchedulerResult }>> {
    return this.driver.runAllPending();
  }

  private async recoverStaleRunning(): Promise<void> {
    try {
      const store = this.persistence.stateStore;
      if (typeof store.recoverStaleRunning !== "function") return;
      const recovered = await store.recoverStaleRunning(
        (this.historyDeps.now?.() ?? new Date()).getTime(),
      );
      if (recovered.length > 0) {
        this.historyDeps.logger?.warn(
          `scheduler: recovered stale running dates: ${recovered.join(", ")}`,
        );
      }
    } catch (e) {
      this.historyDeps.logger?.warn("scheduler: stale running recovery failed", e);
    }
  }
}

function assertPersistenceStorePair(
  stateStore: StateStore,
  runHistory: Pick<RunHistoryStore, "safeAppend">,
): void {
  if (
    !stateStore ||
    typeof stateStore.get !== "function" ||
    typeof stateStore.snapshot !== "function"
  ) {
    throw new Error("Invalid scheduler state store replacement");
  }
  if (!runHistory || typeof runHistory.safeAppend !== "function") {
    throw new Error("Invalid scheduler run history replacement");
  }
}
