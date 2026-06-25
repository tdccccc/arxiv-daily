import type { Logger } from "./logger";
import type { StateStore } from "./state-store";
import type { RunLock } from "./run-lock";
import type { PluginSettings } from "../settings/types";
import {
  todayInTz,
  formatDate,
  isMinutesWithinWindow,
  isTimeWithinLocalWindow,
  minutesFromHHMM,
  minutesSinceMidnight,
  daysBefore,
  isWeekendInTz,
  isWeekendDate,
} from "../utils/time";
import type { PipelineResult } from "../pipeline/pipeline";
import type { ProgressReporter } from "./progress";
import { NoopProgressReporter } from "./progress";
import type { RunCancellationService } from "./cancellation";
import type {
  RunHistoryRecord,
  RunHistoryTrigger,
  RunHistoryStore,
} from "./run-history";

const LOOKBACK_DAYS = 5;

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

export class SchedulerService {
  private intervalHandle: number | null = null;
  private readonly progress: ProgressReporter;

  constructor(private deps: SchedulerDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }

  replaceStore(store: StateStore): void {
    this.deps.store = store;
  }

  replaceRunHistory(runHistory: Pick<RunHistoryStore, "safeAppend">): void {
    this.deps.runHistory = runHistory;
  }

  start(): void {
    const min = this.deps.getSettings().schedule.tickIntervalMin;
    this.stop();
    const handle = setInterval(() => {
      this.tick().catch((e) => this.deps.logger.error("scheduler tick failed", e));
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
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) return;
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();

    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const scheduledMin = minutesFromHHMM(s.schedule.runAtLocal);
    const endMin = minutesFromHHMM(s.schedule.runUntilLocal);

    if (!isTimeWithinLocalWindow(now, tz, s.schedule.runAtLocal, s.schedule.runUntilLocal)) {
      this.progress.setIdle(this.latestCompleted());
      return;
    }

    await this.deps.recentDates?.refresh();

    for (let i = 0; i < LOOKBACK_DAYS; i++) {
      if (this.isCancellationRequested()) break;
      const dateObj = daysBefore(todayObj, i);
      const date = formatDate(dateObj);
      const isToday = date === today;
      this.progress.setBatch(i + 1, LOOKBACK_DAYS, date);
      if (isWeekendDate(dateObj)) continue;
      await this.tickDate(date, {
        now,
        timeGate: isToday ? { scheduledMin, endMin, minutesNow } : undefined,
        trigger: "scheduler",
      });
      if (this.isCancellationRequested()) break;
    }
    this.progress.setIdle(this.latestCompleted());
  }

  async tickToday(): Promise<
    PipelineResult | { kind: "skipped"; reason: string } | undefined
  > {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) {
      await this.recordSkippedForToday("scheduler", "disabled");
      return { kind: "skipped", reason: "disabled" };
    }
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      await this.recordSkippedForDate(todayInTz(now, tz), "scheduler", "weekend");
      return { kind: "skipped", reason: "weekend" };
    }
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    this.progress.setBatch(1, 1, today);
    const result = await this.tickDate(today, { now, trigger: "scheduler" });
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      await this.recordSkipped(today, "scheduler", "guarded");
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }

  async tickTodayScheduled(): Promise<
    PipelineResult | { kind: "skipped"; reason: string } | undefined
  > {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) {
      await this.recordSkippedForToday("scheduler", "disabled");
      return { kind: "skipped", reason: "disabled" };
    }
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      await this.recordSkippedForDate(todayInTz(now, tz), "scheduler", "weekend");
      return { kind: "skipped", reason: "weekend" };
    }
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const scheduledMin = minutesFromHHMM(s.schedule.runAtLocal);
    const endMin = minutesFromHHMM(s.schedule.runUntilLocal);
    if (isTimeWithinLocalWindow(now, tz, s.schedule.runAtLocal, s.schedule.runUntilLocal)) {
      await this.deps.recentDates?.refresh();
    }
    this.progress.setBatch(1, 1, today);
    const result = await this.tickDate(today, {
      now,
      timeGate: { scheduledMin, endMin, minutesNow },
      trigger: "scheduler",
    });
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      await this.recordSkipped(today, "scheduler", "guarded");
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }

  private async tickDate(
    date: string,
    opts: {
      now: Date;
      timeGate?: { scheduledMin: number; endMin: number; minutesNow: number };
      trigger: RunHistoryTrigger;
    },
  ): Promise<PipelineResult | undefined> {
    const s = this.deps.getSettings();
    const entry = this.deps.store.get(date);
    if (this.deps.store.isDone(date)) return undefined;
    if (entry.status === "running") return undefined;

    if (
      opts.timeGate &&
      !isMinutesWithinWindow(
        opts.timeGate.minutesNow,
        opts.timeGate.scheduledMin,
        opts.timeGate.endMin,
      )
    ) {
      return undefined;
    }

    if (entry.status === "failed_transient") {
      const tickMs = s.schedule.tickIntervalMin * 60_000;
      if (opts.now.getTime() - entry.lastAttempt < tickMs) return undefined;
    }

    return await this.tryRun(date, opts.trigger);
  }

  /** Manual trigger: ignore scheduled-time gate, still respect lock and isDone. */
  async runForDateNow(
    date: string,
    opts: SchedulerRunOptions = {},
  ): Promise<PipelineResult | { kind: "skipped"; reason: string }> {
    const trigger = opts.trigger ?? "manual";
    const entry = this.deps.store.get(date);
    if (entry.status === "running") {
      await this.recordSkipped(date, trigger, "already running");
      return { kind: "skipped", reason: "already running" };
    }
    this.progress.setBatch(1, 1, date);
    const result = await this.tryRun(date, trigger);
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      await this.recordSkipped(date, trigger, "lock held");
      return { kind: "skipped", reason: "lock held" };
    }
    return result;
  }

  async forceRunForDate(
    date: string,
  ): Promise<PipelineResult | { kind: "skipped"; reason: string }> {
    const entry = this.deps.store.get(date);
    if (entry.status === "running") {
      await this.recordSkipped(date, "force", "already running");
      return { kind: "skipped", reason: "already running" };
    }
    await this.deps.store.clearDate(date);
    return this.runForDateNow(date, { trigger: "force" });
  }

  async retryFailedInLookback(): Promise<Array<{ date: string; result: PipelineResult | { kind: "skipped"; reason: string } }>> {
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    const todayObj = todayInTz(now, tz);
    const results: Array<{
      date: string;
      result: PipelineResult | { kind: "skipped"; reason: string };
    }> = [];

    for (let i = 0; i < LOOKBACK_DAYS; i++) {
      if (this.isCancellationRequested()) break;
      const date = formatDate(daysBefore(todayObj, i));
      const entry = this.deps.store.get(date);
      if (entry.status !== "failed_transient" && entry.status !== "failed_permanent") {
        continue;
      }
      await this.deps.store.clearDate(date);
      this.progress.setBatch(i + 1, LOOKBACK_DAYS, date);
      const r = await this.tryRun(date, "retry");
      if (r === undefined) {
        await this.recordSkipped(date, "retry", "lock held");
        results.push({ date, result: { kind: "skipped", reason: "lock held" } });
      } else {
        results.push({ date, result: r });
      }
      if (this.isCancellationRequested()) break;
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
  async runAllPending(): Promise<Array<{ date: string; result: PipelineResult | { kind: "skipped"; reason: string } }>> {
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);

    await this.deps.recentDates?.refresh();

    const results: Array<{
      date: string;
      result: PipelineResult | { kind: "skipped"; reason: string };
    }> = [];

    for (let i = 0; i < LOOKBACK_DAYS; i++) {
      if (this.isCancellationRequested()) break;
      const date = formatDate(daysBefore(todayObj, i));
      if (
        date !== today &&
        this.deps.recentDates?.hasDate &&
        !this.deps.recentDates.hasDate(date)
      ) {
        continue;
      }
      const entry = this.deps.store.get(date);
      if (this.deps.store.isDone(date)) continue;
      if (entry.status === "running") {
        await this.recordSkipped(date, "run-all-pending", "already running");
        results.push({ date, result: { kind: "skipped", reason: "already running" } });
        continue;
      }
      this.progress.setBatch(i + 1, LOOKBACK_DAYS, date);
      const r = await this.tryRun(date, "run-all-pending");
      if (r === undefined) {
        await this.recordSkipped(date, "run-all-pending", "lock held");
        results.push({ date, result: { kind: "skipped", reason: "lock held" } });
      } else {
        results.push({ date, result: r });
      }
      if (this.isCancellationRequested()) break;
    }
    this.progress.setIdle(this.latestCompleted());
    return results;
  }

  private async tryRun(
    date: string,
    trigger: RunHistoryTrigger,
  ): Promise<PipelineResult | undefined> {
    return this.deps.lock.withLock(date, async () => {
      const previousEntry = this.deps.store.get(date);
      this.deps.cancellation?.prepareRun();
      const signal = this.deps.cancellation?.begin(date);
      let result: PipelineResult;
      try {
        this.progress.setTask("arXiv Daily report", date);
        await this.deps.store.setRunning(date);
        await this.recordStarted(date, trigger);
        result = signal
          ? await this.deps.runForDate(date, signal)
          : await this.deps.runForDate(date);
      } catch (e) {
        result = {
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
          await this.deps.store.setCompleted(date, papersWritten);
          this.deps.logger.notice(`arXiv ${date}: ${papersWritten} papers written`);
          this.progress.setComplete(`Daily report complete: ${date}`);
          await this.recordCompleted(date, trigger, {
            papersWritten,
            requestedPapersWritten: result.papersWritten,
            preservedPapersWritten,
          });
        } else if (result.kind === "pending") {
          // Don't mark as completed - clear running state so scheduler can retry later
          await this.deps.store.clearDate(date);
          this.deps.logger.info(`arXiv ${date}: pending - ${result.reason}`);
          this.progress.setIdle(this.latestCompleted());
          await this.recordPending(date, trigger, result.reason);
        } else if (result.kind === "failed_transient") {
          await this.deps.store.setFailed(date, "transient", result.reason);
          this.deps.logger.warn(`arXiv ${date} transient: ${result.reason}`);
          this.progress.setError(`Daily report failed: ${date} (${result.reason})`);
          await this.recordFailed(date, trigger, result.kind, result.reason);
        } else {
          await this.deps.store.setFailed(date, "permanent", result.reason);
          this.deps.logger.error(`arXiv ${date} permanent: ${result.reason}`);
          this.deps.logger.notice(`arXiv ${date}: failed (${result.reason})`, 10_000);
          this.progress.setError(`Daily report failed: ${date} (${result.reason})`);
          await this.recordFailed(date, trigger, result.kind, result.reason);
        }
      } finally {
        this.deps.cancellation?.finish(date);
      }
      return result;
    });
  }

  private isCancellationRequested(): boolean {
    return this.deps.cancellation?.isCancellationRequested() ?? false;
  }

  private latestCompleted(): string | undefined {
    const snap = this.deps.store.snapshot();
    const done = Object.entries(snap)
      .filter(([, v]) => v.status === "completed")
      .map(([k]) => k)
      .sort();
    return done[done.length - 1];
  }

  private async recordStarted(
    date: string,
    trigger: RunHistoryTrigger,
  ): Promise<void> {
    await this.recordHistory({
      date,
      event: "started",
      trigger,
      status: "running",
      attempts: this.deps.store.get(date).attempts,
    });
  }

  private async recordCompleted(
    date: string,
    trigger: RunHistoryTrigger,
    detail: {
      papersWritten: number;
      requestedPapersWritten: number;
      preservedPapersWritten: boolean;
    },
  ): Promise<void> {
    const entry = this.deps.store.get(date);
    await this.recordHistory({
      date,
      event: "completed",
      trigger,
      status: "completed",
      resultKind: "completed",
      papersWritten: detail.papersWritten,
      requestedPapersWritten: detail.requestedPapersWritten,
      preservedPapersWritten: detail.preservedPapersWritten || undefined,
      attempts: entry.attempts,
    });
  }

  private async recordPending(
    date: string,
    trigger: RunHistoryTrigger,
    reason: string,
  ): Promise<void> {
    await this.recordHistory({
      date,
      event: "pending",
      trigger,
      status: "pending",
      resultKind: "pending",
      reason,
    });
  }

  private async recordFailed(
    date: string,
    trigger: RunHistoryTrigger,
    resultKind: "failed_transient" | "failed_permanent",
    reason: string,
  ): Promise<void> {
    const entry = this.deps.store.get(date);
    await this.recordHistory({
      date,
      event: "failed",
      trigger,
      status: entry.status,
      resultKind,
      reason,
      errorMessage: reason,
      attempts: entry.attempts,
    });
  }

  private async recordSkippedForToday(
    trigger: RunHistoryTrigger,
    reason: string,
  ): Promise<void> {
    const settings = this.deps.getSettings();
    await this.recordSkippedForDate(
      todayInTz((this.deps.now ?? (() => new Date()))(), settings.arxiv.timezone),
      trigger,
      reason,
    );
  }

  private async recordSkippedForDate(
    dateObj: { y: number; m: number; d: number },
    trigger: RunHistoryTrigger,
    reason: string,
  ): Promise<void> {
    await this.recordSkipped(formatDate(dateObj), trigger, reason);
  }

  private async recordSkipped(
    date: string,
    trigger: RunHistoryTrigger,
    reason: string,
  ): Promise<void> {
    await this.recordHistory({
      date,
      event: "skipped",
      trigger,
      status: this.deps.store.get(date).status,
      resultKind: "skipped",
      reason,
      errorMessage: reason,
      attempts: this.deps.store.get(date).attempts,
    });
  }

  private async recordHistory(
    record: Omit<RunHistoryRecord, "schemaVersion" | "at" | "dailyPath">,
  ): Promise<void> {
    await this.deps.runHistory?.safeAppend({
      schemaVersion: 1,
      at: (this.deps.now ?? (() => new Date()))().toISOString(),
      dailyPath: this.dailyPathForDate(record.date),
      ...record,
    });
  }

  private dailyPathForDate(date: string): string | undefined {
    try {
      return this.deps.dailyPathForDate?.(date);
    } catch (e) {
      this.deps.logger.warn(`daily path resolution failed for ${date}`, e);
      return undefined;
    }
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
