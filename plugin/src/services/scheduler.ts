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
  refresh: () => Promise<unknown>;
  hasDate?: (date: string) => boolean;
}

export interface SchedulerDeps {
  getSettings: () => PluginSettings;
  store: StateStore;
  lock: RunLock;
  runForDate: (date: string, signal?: AbortSignal) => Promise<PipelineResult>;
  logger: Logger;
  now?: () => Date;
  progress?: ProgressReporter;
  cancellation?: RunCancellationService;
  recentDates?: SchedulerRecentDates;
  runHistory?: Pick<RunHistoryStore, "safeAppend">;
  dailyPathForDate?: (date: string) => string;
}

type SchedulerResult = PipelineResult | { kind: "skipped"; reason: string };

export class SchedulerService {
  private driver: SchedulerDriver;
  private store: StateStore;
  private readonly historyDeps: HistoryRecorderDeps;

  constructor(deps: SchedulerDeps) {
    this.store = deps.store;
    this.historyDeps = {
      runHistory: deps.runHistory,
      store: () => this.store,
      dailyPathForDate: deps.dailyPathForDate,
      now: deps.now,
      logger: deps.logger,
    };
    const history = new HistoryRecorder(this.historyDeps);
    this.driver = new SchedulerDriver({
      getSettings: deps.getSettings,
      store: deps.store,
      lock: deps.lock,
      runForDate: deps.runForDate,
      logger: deps.logger,
      progress: deps.progress ?? new NoopProgressReporter(),
      cancellation: deps.cancellation,
      history,
      recentDates: deps.recentDates,
      now: deps.now,
    });
  }

  start(): void {
    this.driver.start();
  }

  stop(): void {
    this.driver.stop();
  }

  replaceStore(store: StateStore): void {
    this.store = store;
    this.driver.replaceStore(store);
  }

  replaceRunHistory(runHistory: Pick<RunHistoryStore, "safeAppend">): void {
    this.historyDeps.runHistory = runHistory;
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
}
