import type { PipelineResult } from "../../pipeline/pipeline";
import type { PluginSettings } from "../../settings/types";
import {
  daysBefore,
  isTimeWithinLocalWindow,
  isWeekendDate,
  isWeekendInTz,
  minutesFromHHMM,
  minutesSinceMidnight,
  todayInTz,
} from "../../utils/time";
import type { RunCancellationBatch, RunCancellationService } from "../cancellation";
import { isCancellationError } from "../cancellation";
import type { Logger } from "../logger";
import { NoopProgressReporter, type ProgressReporter } from "../progress";
import type { RunHistoryTrigger } from "../run-history";
import type { RunLock } from "../run-lock";
import type { StateStore } from "../state-store";
import { lookbackDateStrings, todayDateString } from "./date-selector";
import { LOOKBACK_DAYS } from "./constants";
import type { HistoryRecorder } from "./history-recorder";
import { checkTickGate, isRunning } from "./run-gate";
import type { TimeGate } from "./types";

export interface SchedulerRunOptions {
  trigger?: RunHistoryTrigger;
}

export interface SchedulerRecentDates {
  refresh: (signal?: AbortSignal) => Promise<unknown>;
  hasDate?: (date: string) => boolean;
}

export interface SchedulerDriverDeps {
  getSettings: () => PluginSettings;
  store: StateStore;
  getStore?: () => StateStore;
  lock: RunLock;
  runForDate: (date: string, signal?: AbortSignal) => Promise<PipelineResult>;
  logger: Logger;
  progress?: ProgressReporter;
  cancellation?: RunCancellationService;
  history: HistoryRecorder;
  recentDates?: SchedulerRecentDates;
  now?: () => Date;
  /**
   * Optional post-completion hook (e.g. email delivery). Failures must be
   * swallowed by the callback — they must not rewrite pipeline run-state.
   */
  onDailyCompleted?: (
    date: string,
    result: Extract<PipelineResult, { kind: "completed" }>,
  ) => Promise<void>;
}

type SchedulerResult = PipelineResult | { kind: "skipped"; reason: string };

export class SchedulerDriver {
  private intervalHandle: number | null = null;
  private readonly progress: ProgressReporter;
  private ticking = false;

  constructor(private deps: SchedulerDriverDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }

  replaceStore(store: StateStore): void {
    this.deps.store = store;
  }

  start(): void {
    const min = this.deps.getSettings().schedule.tickIntervalMin;
    this.stop();
    const handle = setInterval(() => {
      this.runScheduledTick().catch((e) => this.deps.logger.error("scheduler tick failed", e));
    }, Math.max(1, min) * 60_000);
    this.intervalHandle = handle as unknown as number;
  }

  stop(): void {
    if (this.intervalHandle != null) {
      clearInterval(this.intervalHandle as unknown as ReturnType<typeof setInterval>);
      this.intervalHandle = null;
    }
  }

  cancelCurrentRun(reason = "cancelled by user"): string[] {
    return this.deps.cancellation?.cancelAll(reason) ?? [];
  }

  activeRuns(): string[] {
    return this.deps.cancellation?.activeDates() ?? [];
  }

  async tick(): Promise<void> {
    const now = this.now();
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) return;
    const tz = s.arxiv.timezone;

    const todayObj = todayInTz(now, tz);
    const today = todayDateString(tz, () => now);
    const dateStrings = lookbackDateStrings(tz, LOOKBACK_DAYS, () => now);
    const minutesNow = minutesSinceMidnight(now, tz);
    const scheduledMin = minutesFromHHMM(s.schedule.runAtLocal);
    const endMin = minutesFromHHMM(s.schedule.runUntilLocal);

    if (!isTimeWithinLocalWindow(now, tz, s.schedule.runAtLocal, s.schedule.runUntilLocal)) {
      this.progress.setIdle(this.latestCompleted());
      return;
    }

    // Today's report is already generated (or finalized). Stay idle for the
    // remainder of the run window to avoid re-querying arxiv on every tick.
    if (this.store().isDone(today)) {
      this.progress.setIdle(this.latestCompleted());
      return;
    }

    const batch = this.beginCancellationBatch();
    try {
      await this.deps.recentDates?.refresh(batch?.signal);
      for (let i = 0; i < dateStrings.length; i += 1) {
        if (this.isCancellationRequested(batch)) break;
        const dateObj = daysBefore(todayObj, i, tz);
        const date = dateStrings[i];
        if (!date) continue;
        const isToday = date === today;
        this.progress.setBatch(i + 1, LOOKBACK_DAYS, date);
        if (isWeekendDate(dateObj, tz)) continue;
        await this.tickDate(date, {
          now,
          timeGate: isToday ? { scheduledMin, endMin, minutesNow } : undefined,
          trigger: "scheduler",
          cancellationBatch: batch,
        });
        if (this.isCancellationRequested(batch)) break;
      }
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.progress.setIdle(this.latestCompleted());
  }

  async tickToday(): Promise<SchedulerResult | undefined> {
    const now = this.now();
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    if (!s.schedule.enabled) {
      await this.deps.history.recordSkippedForDate(todayInTz(now, tz), "scheduler", "disabled", now);
      return { kind: "skipped", reason: "disabled" };
    }
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      await this.deps.history.recordSkippedForDate(todayInTz(now, tz), "scheduler", "weekend", now);
      return { kind: "skipped", reason: "weekend" };
    }
    const today = todayDateString(tz, () => now);
    this.progress.setBatch(1, 1, today);
    const batch = this.beginCancellationBatch();
    let result: PipelineResult | undefined;
    try {
      await this.deps.recentDates?.refresh(batch?.signal);
      result = await this.tickDate(today, {
        now,
        trigger: "scheduler",
        cancellationBatch: batch,
      });
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      await this.deps.history.recordSkipped(today, "scheduler", "guarded", now);
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }

  async tickTodayScheduled(): Promise<SchedulerResult | undefined> {
    const now = this.now();
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    if (!s.schedule.enabled) {
      await this.deps.history.recordSkippedForDate(todayInTz(now, tz), "scheduler", "disabled", now);
      return { kind: "skipped", reason: "disabled" };
    }
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      await this.deps.history.recordSkippedForDate(todayInTz(now, tz), "scheduler", "weekend", now);
      return { kind: "skipped", reason: "weekend" };
    }
    const today = todayDateString(tz, () => now);
    const minutesNow = minutesSinceMidnight(now, tz);
    const scheduledMin = minutesFromHHMM(s.schedule.runAtLocal);
    const endMin = minutesFromHHMM(s.schedule.runUntilLocal);
    this.progress.setBatch(1, 1, today);
    const batch = this.beginCancellationBatch();
    let result: PipelineResult | undefined;
    try {
      if (isTimeWithinLocalWindow(now, tz, s.schedule.runAtLocal, s.schedule.runUntilLocal)) {
        await this.deps.recentDates?.refresh(batch?.signal);
      }
      result = await this.tickDate(today, {
        now,
        timeGate: { scheduledMin, endMin, minutesNow },
        trigger: "scheduler",
        cancellationBatch: batch,
      });
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      await this.deps.history.recordSkipped(today, "scheduler", "guarded", now);
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }

  /** Manual trigger: ignore scheduled-time gate, still respect lock and isDone. */
  async runForDateNow(
    date: string,
    opts: SchedulerRunOptions = {},
  ): Promise<SchedulerResult> {
    return this.runForDateNowAt(date, opts, this.now());
  }

  async forceRunForDate(date: string): Promise<SchedulerResult> {
    const now = this.now();
    const entry = this.store().get(date);
    if (entry.status === "running") {
      await this.deps.history.recordSkipped(date, "force", "already running", now);
      return { kind: "skipped", reason: "already running" };
    }
    return this.runForDateNowAt(date, { trigger: "force" }, now, {
      clearDateBeforeRun: true,
    });
  }

  async retryFailedInLookback(): Promise<Array<{ date: string; result: SchedulerResult }>> {
    const now = this.now();
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    const dateStrings = lookbackDateStrings(tz, LOOKBACK_DAYS, () => now);
    const results: Array<{ date: string; result: SchedulerResult }> = [];

    const batch = this.beginCancellationBatch();
    try {
      for (let i = 0; i < dateStrings.length; i += 1) {
        if (this.isCancellationRequested(batch)) break;
        const date = dateStrings[i];
        if (!date) continue;
        const entry = this.store().get(date);
        if (entry.status !== "failed_transient" && entry.status !== "failed_permanent") {
          continue;
        }
        this.progress.setBatch(i + 1, LOOKBACK_DAYS, date);
        const r = await this.tryRun(date, "retry", now, batch, {
          clearDateBeforeRun: true,
        });
        if (r === undefined) {
          await this.deps.history.recordSkipped(date, "retry", "lock held", now);
          results.push({ date, result: { kind: "skipped", reason: "lock held" } });
        } else {
          results.push({ date, result: r });
        }
        if (this.isCancellationRequested(batch)) break;
      }
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.progress.setIdle(this.latestCompleted());
    return results;
  }

  /**
   * Run every pending or failed_transient date within the current lookback
   * window, skipping completed / failed_permanent / running entries. Bypasses
   * the runAtLocal time gate (manual trigger). Returns one entry per attempted
   * date for caller-side reporting.
   */
  async runAllPending(): Promise<Array<{ date: string; result: SchedulerResult }>> {
    const now = this.now();
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    const today = todayDateString(tz, () => now);
    const dateStrings = lookbackDateStrings(tz, LOOKBACK_DAYS, () => now);

    const results: Array<{ date: string; result: SchedulerResult }> = [];

    const batch = this.beginCancellationBatch();
    try {
      await this.deps.recentDates?.refresh(batch?.signal);
      for (let i = 0; i < dateStrings.length; i += 1) {
        if (this.isCancellationRequested(batch)) break;
        const date = dateStrings[i];
        if (!date) continue;
        if (
          date !== today &&
          this.deps.recentDates?.hasDate &&
          !this.deps.recentDates.hasDate(date)
        ) {
          continue;
        }
        const entry = this.store().get(date);
        if (this.store().isDone(date)) continue;
        if (entry.status === "running") {
          await this.deps.history.recordSkipped(date, "run-all-pending", "already running", now);
          results.push({ date, result: { kind: "skipped", reason: "already running" } });
          continue;
        }
        this.progress.setBatch(i + 1, LOOKBACK_DAYS, date);
        const r = await this.tryRun(date, "run-all-pending", now, batch);
        if (r === undefined) {
          await this.deps.history.recordSkipped(date, "run-all-pending", "lock held", now);
          results.push({ date, result: { kind: "skipped", reason: "lock held" } });
        } else {
          results.push({ date, result: r });
        }
        if (this.isCancellationRequested(batch)) break;
      }
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.progress.setIdle(this.latestCompleted());
    return results;
  }

  private async runForDateNowAt(
    date: string,
    opts: SchedulerRunOptions,
    now: Date,
    runOpts: { clearDateBeforeRun?: boolean } = {},
  ): Promise<SchedulerResult> {
    const trigger = opts.trigger ?? "manual";
    if (isRunning(date, this.store())) {
      await this.deps.history.recordSkipped(date, trigger, "already running", now);
      return { kind: "skipped", reason: "already running" };
    }
    this.progress.setBatch(1, 1, date);
    const batch = this.beginCancellationBatch();
    let result: PipelineResult | undefined;
    try {
      await this.deps.recentDates?.refresh(batch?.signal);
      result = await this.tryRun(date, trigger, now, batch, runOpts);
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      await this.deps.history.recordSkipped(date, trigger, "lock held", now);
      return { kind: "skipped", reason: "lock held" };
    }
    return result;
  }

  private async tickDate(
    date: string,
    opts: {
      now: Date;
      timeGate?: TimeGate;
      trigger: RunHistoryTrigger;
      cancellationBatch?: RunCancellationBatch;
    },
  ): Promise<PipelineResult | undefined> {
    const s = this.deps.getSettings();
    const entry = this.store().get(date);
    const decision = checkTickGate(date, this.store(), {
      now: opts.now,
      timeGate: opts.timeGate,
      tickIntervalMin: s.schedule.tickIntervalMin,
    });
    if (!decision.allow) {
      if (decision.reason === "already-done") {
        this.deps.logger.debug(`tickDate: ${date} already done (${entry.status}), skip`);
      } else if (decision.reason === "running") {
        this.deps.logger.debug(`tickDate: ${date} currently running, skip`);
      }
      return undefined;
    }

    return await this.tryRun(date, opts.trigger, opts.now, opts.cancellationBatch);
  }

  private async tryRun(
    date: string,
    trigger: RunHistoryTrigger,
    now: Date,
    cancellationBatch?: RunCancellationBatch,
    opts: { clearDateBeforeRun?: boolean } = {},
  ): Promise<PipelineResult | undefined> {
    return this.deps.lock.withLock(date, async () => {
      if (opts.clearDateBeforeRun) {
        await this.store().clearDate(date);
      }
      const previousEntry = this.store().get(date);
      this.deps.cancellation?.prepareRun();
      const signal = this.deps.cancellation?.begin(date, cancellationBatch);
      let result: PipelineResult;
      try {
        this.progress.setTask("arXiv Daily report", date);
        await this.store().setRunning(date);
        await this.deps.history.recordStarted(date, trigger, now);
        result = signal
          ? await this.deps.runForDate(date, signal)
          : await this.deps.runForDate(date);
      } catch (e) {
        result = isCancellationError(e)
          ? {
              kind: "cancelled",
              reason: errorMessage(e),
            }
          : {
              kind: "failed_transient",
              reason: errorMessage(e),
            };
      }
      try {
        if (result.kind === "completed") {
          const papersWritten = preservedCompletedPaperCount(
            previousEntry,
            result.papersWritten,
          );
          const preservedPapersWritten = papersWritten !== result.papersWritten;
          await this.store().setCompleted(date, papersWritten);
          this.deps.logger.notice(`arXiv ${date}: ${papersWritten} papers written`);
          this.progress.setComplete(`Daily report complete: ${date}`);
          await this.deps.history.recordCompleted(date, trigger, {
            papersWritten,
            requestedPapersWritten: result.papersWritten,
            preservedPapersWritten,
          }, now);
          if (this.deps.onDailyCompleted) {
            try {
              await this.deps.onDailyCompleted(date, result);
            } catch (e) {
              this.deps.logger.error(
                `scheduler: onDailyCompleted failed for ${date}; pipeline remains completed`,
                e,
              );
            }
          }
        } else if (result.kind === "pending") {
          await this.store().setPending(date, result.reason);
          this.deps.logger.info(`arXiv ${date}: pending - ${result.reason}`);
          this.progress.setIdle(this.latestCompleted());
          await this.deps.history.recordPending(date, trigger, result.reason, now);
        } else if (result.kind === "cancelled") {
          await this.store().setPending(date, result.reason);
          const message = `Daily report cancelled: ${date} (${result.reason})`;
          this.deps.logger.info(`arXiv ${date}: cancelled - ${result.reason}`);
          this.progress.setError(message);
          await this.deps.history.recordCancelled(date, trigger, result.reason, now);
        } else if (result.kind === "failed_transient") {
          const persistedStatus = await this.persistFailed(date, "transient", result.reason);
          const persistedReason = this.store().get(date).error ?? result.reason;
          result = { kind: persistedStatus, reason: persistedReason };
          const severity = persistedStatus === "failed_permanent" ? "permanent" : "transient";
          const message = `Daily report failed ${severity}: ${date} (${persistedReason})`;
          if (persistedStatus === "failed_permanent") {
            this.deps.logger.error(`arXiv ${date} permanent: ${persistedReason}`);
            this.deps.logger.notice(`arXiv ${date}: failed (${persistedReason})`, 10_000);
          } else {
            this.deps.logger.warn(`arXiv ${date} transient: ${persistedReason}`);
          }
          this.progress.setError(message);
          await this.deps.history.recordFailed(date, trigger, persistedStatus, persistedReason, now);
        } else {
          const persistedStatus = await this.persistFailed(date, "permanent", result.reason);
          this.deps.logger.error(`arXiv ${date} permanent: ${result.reason}`);
          this.deps.logger.notice(`arXiv ${date}: failed (${result.reason})`, 10_000);
          this.progress.setError(`Daily report failed permanent: ${date} (${result.reason})`);
          await this.deps.history.recordFailed(date, trigger, persistedStatus, result.reason, now);
        }
      } catch (e) {
        if (isCancellationError(e)) throw e;
        this.deps.logger.error(
          `scheduler: failed to persist result for ${date}; continuing batch`,
          e,
        );
      } finally {
        this.deps.cancellation?.finish(date);
      }
      return result;
    });
  }

  private beginCancellationBatch(): RunCancellationBatch | undefined {
    return this.deps.cancellation?.beginBatch();
  }

  private finishCancellationBatch(batch: RunCancellationBatch | undefined): void {
    if (batch) this.deps.cancellation?.finishBatch(batch);
  }

  private isCancellationRequested(batch?: RunCancellationBatch): boolean {
    return batch?.isCancellationRequested() ?? false;
  }

  private async runScheduledTick(): Promise<void> {
    if (this.ticking) return;
    this.ticking = true;
    try {
      await this.tick();
    } finally {
      this.ticking = false;
    }
  }

  private async persistFailed(
    date: string,
    kind: "transient" | "permanent",
    reason: string,
  ): Promise<"failed_transient" | "failed_permanent"> {
    try {
      return await this.store().setFailed(date, kind, reason);
    } catch (e) {
      this.deps.logger.error(
        `scheduler: failed to persist failure for ${date}; clearing running state`,
        e,
      );
      await this.store().clearDate(date);
      return kind === "permanent" ? "failed_permanent" : "failed_transient";
    }
  }

  private latestCompleted(): string | undefined {
    const snap = this.store().snapshot();
    const done = Object.entries(snap)
      .filter(([, v]) => v.status === "completed")
      .map(([k]) => k)
      .sort();
    return done[done.length - 1];
  }

  private store(): StateStore {
    return this.deps.getStore?.() ?? this.deps.store;
  }

  private now(): Date {
    return (this.deps.now ?? (() => new Date()))();
  }
}

function preservedCompletedPaperCount(
  previousEntry: ReturnType<StateStore["get"]>,
  nextPapersWritten: number,
): number {
  if (
    nextPapersWritten === 0 &&
    previousEntry.status === "completed" &&
    (previousEntry.papersWritten ?? 0) > 0
  ) {
    return previousEntry.papersWritten ?? nextPapersWritten;
  }
  return nextPapersWritten;
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error.trim()) return error.trim();
  try {
    return JSON.stringify(error) || "unknown error";
  } catch {
    return "unknown error";
  }
}
