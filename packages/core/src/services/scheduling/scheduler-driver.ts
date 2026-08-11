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
import { checkTickGate } from "./run-gate";
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
type CompletedPipelineResult = Extract<PipelineResult, { kind: "completed" }>;

interface PendingCompletion {
  result: CompletedPipelineResult;
  papersWritten: number;
  requestedPapersWritten: number;
  preservedPapersWritten: boolean;
  trigger: RunHistoryTrigger;
  completedAt: Date;
  committed: boolean;
}

type PendingNonCompletionResult = Extract<
  PipelineResult,
  { kind: "pending" | "cancelled" }
>;

interface PendingNonCompletion {
  result: PendingNonCompletionResult;
  trigger: RunHistoryTrigger;
  at: Date;
}

interface PendingCompletionAttempt {
  handled: boolean;
  result?: SchedulerResult;
}

type RunMode = "normal" | "scheduled" | "run-all-pending" | "retry" | "force";
type TryRunResult = SchedulerResult | { kind: "not-eligible" };

const COMPLETION_COMMIT_FAILURE_REASON = "scheduler completion commit failed";
const STORE_REPLACEMENT_ACTIVE_ERROR =
  "cannot replace scheduler store while work is active";

export class SchedulerDriver {
  private intervalHandle: number | null = null;
  private readonly progress: ProgressReporter;
  private readonly pendingCompletions = new Map<string, PendingCompletion>();
  private readonly pendingNonCompletions = new Map<string, PendingNonCompletion>();
  private readonly pendingCompletionFinalizations = new Map<
    string,
    Promise<PipelineResult | undefined>
  >();
  private activeWork = 0;
  private ticking = false;

  constructor(private deps: SchedulerDriverDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
  }

  assertStoreReplacementAllowed(): void {
    if (
      this.activeWork > 0 ||
      this.pendingCompletions.size > 0 ||
      this.pendingNonCompletions.size > 0
    ) {
      throw new Error(STORE_REPLACEMENT_ACTIVE_ERROR);
    }
  }

  replaceStore(store: StateStore): void {
    this.assertStoreReplacementAllowed();
    this.deps.store = store;
  }

  start(): void {
    const min = this.deps.getSettings().schedule.tickIntervalMin;
    this.stop();
    const handle = setInterval(() => {
      this.runScheduledTick().catch((error) =>
        this.safeEffect(() =>
          this.deps.logger.error("scheduler tick failed", error),
        ),
      );
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
    const tz = s.arxiv.timezone;
    const todayObj = todayInTz(now, tz);
    const today = todayDateString(tz, () => now);
    const dateStrings = lookbackDateStrings(tz, LOOKBACK_DAYS, () => now);
    const pendingRetryDates = this.pendingRetryDates(dateStrings);
    if (!s.schedule.enabled) {
      await this.retryPendingCompletions(pendingRetryDates, now);
      return;
    }
    const minutesNow = minutesSinceMidnight(now, tz);
    const scheduledMin = minutesFromHHMM(s.schedule.runAtLocal);
    const endMin = minutesFromHHMM(s.schedule.runUntilLocal);

    if (!isTimeWithinLocalWindow(now, tz, s.schedule.runAtLocal, s.schedule.runUntilLocal)) {
      await this.retryPendingCompletions(pendingRetryDates, now);
      this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
      return;
    }

    // Today's report is already generated (or finalized). Stay idle for the
    // remainder of the run window to avoid re-querying arxiv on every tick,
    // but still finalize any in-memory completion candidates from prior dates.
    if (
      this.store().isDone(today) &&
      (await this.confirmDurablyDone(today))
    ) {
      await this.retryPendingCompletions(pendingRetryDates, now);
      this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
      return;
    }

    const batch = this.beginCancellationBatch();
    try {
      const retriedPending = await this.retryPendingCompletions(
        pendingRetryDates,
        now,
        "scheduler",
        batch,
      );
      await this.deps.recentDates?.refresh(batch?.signal);
      for (let i = 0; i < dateStrings.length; i += 1) {
        if (this.isCancellationRequested(batch)) break;
        const dateObj = daysBefore(todayObj, i, tz);
        const date = dateStrings[i];
        if (!date || retriedPending.has(date)) continue;
        const isToday = date === today;
        this.safeEffect(() =>
          this.progress.setBatch(i + 1, LOOKBACK_DAYS, date),
        );
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
    this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
  }

  async tickToday(): Promise<SchedulerResult | undefined> {
    const now = this.now();
    const s = this.deps.getSettings();
    const tz = s.arxiv.timezone;
    const today = todayDateString(tz, () => now);
    const pendingAttempt = await this.attemptPendingCompletion(
      today,
      "scheduler",
      now,
    );
    if (pendingAttempt.handled) {
      this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
      return pendingAttempt.result;
    }
    if (!s.schedule.enabled) {
      await this.safeAsyncEffect(() =>
        this.deps.history.recordSkippedForDate(
          todayInTz(now, tz),
          "scheduler",
          "disabled",
          now,
        ),
      );
      return { kind: "skipped", reason: "disabled" };
    }
    if (isWeekendInTz(now, tz)) {
      this.safeEffect(() =>
        this.progress.setIdle(this.latestCompleted(), "weekend"),
      );
      await this.safeAsyncEffect(() =>
        this.deps.history.recordSkippedForDate(
          todayInTz(now, tz),
          "scheduler",
          "weekend",
          now,
        ),
      );
      return { kind: "skipped", reason: "weekend" };
    }
    this.safeEffect(() => this.progress.setBatch(1, 1, today));
    const batch = this.beginCancellationBatch();
    let result: SchedulerResult | undefined;
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
    this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
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
    const today = todayDateString(tz, () => now);
    const pendingAttempt = await this.attemptPendingCompletion(
      today,
      "scheduler",
      now,
    );
    if (pendingAttempt.handled) {
      this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
      return pendingAttempt.result;
    }
    if (!s.schedule.enabled) {
      await this.safeAsyncEffect(() =>
        this.deps.history.recordSkippedForDate(
          todayInTz(now, tz),
          "scheduler",
          "disabled",
          now,
        ),
      );
      return { kind: "skipped", reason: "disabled" };
    }
    if (isWeekendInTz(now, tz)) {
      this.safeEffect(() =>
        this.progress.setIdle(this.latestCompleted(), "weekend"),
      );
      await this.safeAsyncEffect(() =>
        this.deps.history.recordSkippedForDate(
          todayInTz(now, tz),
          "scheduler",
          "weekend",
          now,
        ),
      );
      return { kind: "skipped", reason: "weekend" };
    }
    const minutesNow = minutesSinceMidnight(now, tz);
    const scheduledMin = minutesFromHHMM(s.schedule.runAtLocal);
    const endMin = minutesFromHHMM(s.schedule.runUntilLocal);
    this.safeEffect(() => this.progress.setBatch(1, 1, today));
    const batch = this.beginCancellationBatch();
    let result: SchedulerResult | undefined;
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
    this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
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
    return this.runForDateNowAt(date, { trigger: "force" }, this.now(), {
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
      const retriedPending = await this.retryPendingCompletions(
        dateStrings,
        now,
        "retry",
        batch,
      );
      for (const [date, result] of retriedPending) {
        results.push({
          date,
          result: result ?? { kind: "skipped", reason: "lock held" },
        });
      }
      for (let i = 0; i < dateStrings.length; i += 1) {
        if (this.isCancellationRequested(batch)) break;
        const date = dateStrings[i];
        if (!date || retriedPending.has(date)) continue;
        this.safeEffect(() =>
          this.progress.setBatch(i + 1, LOOKBACK_DAYS, date),
        );
        const result = await this.tryRun(date, "retry", now, batch, "retry");
        if (result === undefined) {
          await this.safeAsyncEffect(() =>
            this.deps.history.recordSkipped(date, "retry", "lock held", now),
          );
          results.push({
            date,
            result: { kind: "skipped", reason: "lock held" },
          });
        } else if (result.kind !== "not-eligible") {
          results.push({ date, result });
        }
        if (this.isCancellationRequested(batch)) break;
      }
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
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
      const retriedPending = await this.retryPendingCompletions(
        dateStrings,
        now,
        "run-all-pending",
        batch,
      );
      for (const [date, result] of retriedPending) {
        results.push({
          date,
          result: result ?? { kind: "skipped", reason: "lock held" },
        });
      }
      await this.deps.recentDates?.refresh(batch?.signal);
      for (let i = 0; i < dateStrings.length; i += 1) {
        if (this.isCancellationRequested(batch)) break;
        const date = dateStrings[i];
        if (!date || retriedPending.has(date)) continue;
        if (
          date !== today &&
          this.deps.recentDates?.hasDate &&
          !this.deps.recentDates.hasDate(date)
        ) {
          continue;
        }
        this.safeEffect(() =>
          this.progress.setBatch(i + 1, LOOKBACK_DAYS, date),
        );
        const result = await this.tryRun(
          date,
          "run-all-pending",
          now,
          batch,
          "run-all-pending",
        );
        if (result === undefined) {
          await this.safeAsyncEffect(() =>
            this.deps.history.recordSkipped(
              date,
              "run-all-pending",
              "lock held",
              now,
            ),
          );
          results.push({
            date,
            result: { kind: "skipped", reason: "lock held" },
          });
        } else if (result.kind !== "not-eligible") {
          results.push({ date, result });
        }
        if (this.isCancellationRequested(batch)) break;
      }
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
    return results;
  }

  private async runForDateNowAt(
    date: string,
    opts: SchedulerRunOptions,
    now: Date,
    runOpts: { clearDateBeforeRun?: boolean } = {},
  ): Promise<SchedulerResult> {
    const trigger = opts.trigger ?? "manual";
    const pendingAttempt = await this.attemptPendingCompletion(
      date,
      trigger,
      now,
    );
    if (pendingAttempt.handled) {
      return pendingAttempt.result ?? { kind: "skipped", reason: "lock held" };
    }
    this.safeEffect(() => this.progress.setBatch(1, 1, date));
    const batch = this.beginCancellationBatch();
    let result: TryRunResult | undefined;
    try {
      await this.deps.recentDates?.refresh(batch?.signal);
      result = await this.tryRun(
        date,
        trigger,
        now,
        batch,
        runOpts.clearDateBeforeRun ? "force" : "normal",
      );
    } finally {
      this.finishCancellationBatch(batch);
    }
    this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
    if (result === undefined) {
      await this.safeAsyncEffect(() =>
        this.deps.history.recordSkipped(date, trigger, "lock held", now),
      );
      return { kind: "skipped", reason: "lock held" };
    }
    if (result.kind === "not-eligible") {
      return { kind: "skipped", reason: "guarded" };
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
  ): Promise<SchedulerResult | undefined> {
    const result = await this.tryRun(
      date,
      opts.trigger,
      opts.now,
      opts.cancellationBatch,
      "scheduled",
      opts.timeGate,
    );
    return result?.kind === "not-eligible" ? undefined : result;
  }

  private async tryRun(
    date: string,
    trigger: RunHistoryTrigger,
    now: Date,
    cancellationBatch?: RunCancellationBatch,
    mode: RunMode = "normal",
    timeGate?: TimeGate,
  ): Promise<TryRunResult | undefined> {
    const store = this.store();
    this.activeWork += 1;
    try {
      return await this.deps.lock.withLock(date, async () => {
        const pendingCompletion = this.pendingCompletions.get(date);
        if (pendingCompletion) {
          return this.runPendingCompletionFinalization(date, () =>
            this.finalizePendingCompletionLocked(
              date,
              pendingCompletion,
              trigger,
              now,
              store,
            ),
          );
        }
        const pendingNonCompletion = this.pendingNonCompletions.get(date);
        if (pendingNonCompletion) {
          return this.finalizePendingNonCompletionLocked(
            date,
            pendingNonCompletion,
            store,
          );
        }

        try {
          await store.loadAuthoritative();
        } catch (error) {
          return this.storeFailureResult(date, trigger, now, error);
        }

        const currentEntry = store.get(date);
        if (mode === "scheduled") {
          const decision = checkTickGate(date, store, {
            now,
            timeGate,
            tickIntervalMin: this.deps.getSettings().schedule.tickIntervalMin,
          });
          if (!decision.allow) {
            this.safeEffect(() => {
              if (decision.reason === "already-done") {
                this.deps.logger.debug(
                  `tickDate: ${date} already done (${currentEntry.status}), skip`,
                );
              } else if (decision.reason === "running") {
                this.deps.logger.debug(`tickDate: ${date} currently running, skip`);
              }
            });
            return { kind: "not-eligible" as const };
          }
        } else if (mode === "retry") {
          if (
            currentEntry.status !== "failed_transient" &&
            currentEntry.status !== "failed_permanent"
          ) {
            return { kind: "not-eligible" as const };
          }
        } else if (mode === "run-all-pending") {
          if (store.isDone(date)) return { kind: "not-eligible" as const };
          if (currentEntry.status === "running") {
            await this.safeAsyncEffect(() =>
              this.deps.history.recordSkipped(
                date,
                trigger,
                "already running",
                now,
              ),
            );
            return { kind: "skipped" as const, reason: "already running" };
          }
        } else if (mode === "normal") {
          if (store.isDone(date)) {
            await this.safeAsyncEffect(() =>
              this.deps.history.recordSkipped(date, trigger, "already done", now),
            );
            return { kind: "skipped" as const, reason: "already done" };
          }
          if (currentEntry.status === "running") {
            await this.safeAsyncEffect(() =>
              this.deps.history.recordSkipped(
                date,
                trigger,
                "already running",
                now,
              ),
            );
            return { kind: "skipped" as const, reason: "already running" };
          }
        } else if (mode === "force" && currentEntry.status === "running") {
          await this.safeAsyncEffect(() =>
            this.deps.history.recordSkipped(
              date,
              trigger,
              "already running",
              now,
            ),
          );
          return { kind: "skipped" as const, reason: "already running" };
        }

        if (mode === "force" || mode === "retry") {
          try {
            await store.clearDate(date);
          } catch (error) {
            return this.storeFailureResult(date, trigger, now, error);
          }
        }

        const previousEntry = store.get(date);
        this.safeEffect(() => this.deps.cancellation?.prepareRun());
        const signal = this.beginCancellationRun(date, cancellationBatch);
        let result: PipelineResult;
        try {
          this.safeEffect(() =>
            this.progress.setTask("arXiv Daily report", date),
          );
          try {
            await store.setRunning(date);
          } catch (error) {
            this.safeEffect(() => this.deps.cancellation?.finish(date));
            return this.storeFailureResult(date, trigger, now, error);
          }
          await this.safeAsyncEffect(() =>
            this.deps.history.recordStarted(date, trigger, now),
          );
          result = signal
            ? await this.deps.runForDate(date, signal)
            : await this.deps.runForDate(date);
        } catch (error) {
          result = isCancellationError(error)
            ? { kind: "cancelled", reason: errorMessage(error) }
            : { kind: "failed_transient", reason: errorMessage(error) };
        }

        try {
          if (result.kind === "completed") {
            const papersWritten = preservedCompletedPaperCount(
              previousEntry,
              result.papersWritten,
            );
            const pending: PendingCompletion = {
              result,
              papersWritten,
              requestedPapersWritten: result.papersWritten,
              preservedPapersWritten: papersWritten !== result.papersWritten,
              trigger,
              completedAt: now,
              committed: false,
            };
            this.pendingCompletions.set(date, pending);
            const finalization = this.finalizePendingCompletionLocked(
              date,
              pending,
              trigger,
              now,
              store,
            );
            this.pendingCompletionFinalizations.set(date, finalization);
            try {
              return await finalization;
            } finally {
              if (this.pendingCompletionFinalizations.get(date) === finalization) {
                this.pendingCompletionFinalizations.delete(date);
              }
            }
          }
          if (result.kind === "pending" || result.kind === "cancelled") {
            const pending: PendingNonCompletion = { result, trigger, at: now };
            this.pendingNonCompletions.set(date, pending);
            return await this.finalizePendingNonCompletionLocked(date, pending, store);
          }
          return await this.finalizeFailureLocked(date, result, trigger, now, store);
        } finally {
          this.safeEffect(() => this.deps.cancellation?.finish(date));
        }
      });
    } finally {
      this.activeWork -= 1;
    }
  }

  private async attemptPendingCompletion(
    date: string,
    trigger: RunHistoryTrigger,
    now: Date,
  ): Promise<PendingCompletionAttempt> {
    const existingFinalization = this.pendingCompletionFinalizations.get(date);
    if (existingFinalization) {
      return { handled: true, result: await existingFinalization };
    }
    if (!this.pendingCompletions.has(date) && !this.pendingNonCompletions.has(date)) {
      return { handled: false };
    }
    const result = await this.tryRun(date, trigger, now);
    if (result?.kind === "not-eligible") return { handled: true };
    return { handled: true, result };
  }

  private async runPendingCompletionFinalization(
    date: string,
    finalize: () => Promise<PipelineResult | undefined>,
  ): Promise<PipelineResult | undefined> {
    const existing = this.pendingCompletionFinalizations.get(date);
    if (existing) return existing;

    const finalization = finalize();
    this.pendingCompletionFinalizations.set(date, finalization);
    try {
      return await finalization;
    } finally {
      if (this.pendingCompletionFinalizations.get(date) === finalization) {
        this.pendingCompletionFinalizations.delete(date);
      }
    }
  }

  private pendingRetryDates(preferredDates: string[]): string[] {
    const pendingDates = new Set([
      ...this.pendingCompletions.keys(),
      ...this.pendingNonCompletions.keys(),
    ]);
    return [
      ...preferredDates,
      ...[...pendingDates].filter((date) => !preferredDates.includes(date)),
    ];
  }

  private async retryPendingCompletions(
    dates: string[],
    now: Date,
    trigger: RunHistoryTrigger = "scheduler",
    cancellationBatch?: RunCancellationBatch,
  ): Promise<Map<string, SchedulerResult | undefined>> {
    const results = new Map<string, SchedulerResult | undefined>();
    for (const [index, date] of dates.entries()) {
      if (
        !this.pendingCompletions.has(date) &&
        !this.pendingNonCompletions.has(date)
      ) {
        continue;
      }
      if (this.isCancellationRequested(cancellationBatch)) break;
      this.safeEffect(() =>
        this.progress.setBatch(Math.max(1, index + 1), dates.length, date),
      );
      const attempt = await this.attemptPendingCompletion(date, trigger, now);
      if (attempt.handled) results.set(date, attempt.result);
    }
    return results;
  }

  private async finalizePendingCompletionLocked(
    date: string,
    pending: PendingCompletion,
    attemptTrigger: RunHistoryTrigger,
    now: Date,
    store: StateStore,
  ): Promise<PipelineResult> {
    if (!pending.committed) {
      try {
        await store.setCompleted(date, pending.papersWritten);
        pending.committed = true;
      } catch (commitFailure) {
        this.safeEffect(() =>
          this.deps.logger.error(
            `scheduler: completion commit failed for ${date}; retaining pending completion`,
            commitFailure,
          ),
        );
        this.safeEffect(() =>
          this.progress.setError(
            `Daily report failed transient: ${date} (${COMPLETION_COMMIT_FAILURE_REASON})`,
          ),
        );
        await this.safeAsyncEffect(() =>
          this.deps.history.recordFailed(
            date,
            attemptTrigger,
            "failed_transient",
            COMPLETION_COMMIT_FAILURE_REASON,
            now,
          ),
        );
        return {
          kind: "failed_transient",
          reason: COMPLETION_COMMIT_FAILURE_REASON,
        };
      }
    }

    this.safeEffect(() =>
      this.deps.logger.notice(
        `arXiv ${date}: ${pending.papersWritten} papers written`,
      ),
    );
    this.safeEffect(() =>
      this.progress.setComplete(`Daily report complete: ${date}`),
    );
    await this.safeAsyncEffect(() =>
      this.deps.history.recordCompleted(
        date,
        pending.trigger,
        {
          papersWritten: pending.papersWritten,
          requestedPapersWritten: pending.requestedPapersWritten,
          preservedPapersWritten: pending.preservedPapersWritten,
        },
        pending.completedAt,
      ),
    );
    if (this.deps.onDailyCompleted) {
      await this.safeAsyncEffect(
        () => this.deps.onDailyCompleted!(date, pending.result),
        `scheduler: onDailyCompleted failed for ${date}; pipeline remains completed`,
      );
    }
    this.pendingCompletions.delete(date);
    return pending.result;
  }

  private async finalizePendingNonCompletionLocked(
    date: string,
    pending: PendingNonCompletion,
    store: StateStore,
  ): Promise<PendingNonCompletionResult> {
    try {
      await store.setPending(date, pending.result.reason);
    } catch (error) {
      this.safeEffect(() =>
        this.deps.logger.error(
          `scheduler: failed to persist ${pending.result.kind} result for ${date}; retaining terminal transition`,
          error,
        ),
      );
      return pending.result;
    }

    if (pending.result.kind === "pending") {
      this.safeEffect(() =>
        this.deps.logger.info(
          `arXiv ${date}: pending - ${pending.result.reason}`,
        ),
      );
      this.safeEffect(() => this.progress.setIdle(this.latestCompleted()));
      await this.safeAsyncEffect(() =>
        this.deps.history.recordPending(
          date,
          pending.trigger,
          pending.result.reason,
          pending.at,
        ),
      );
    } else {
      this.safeEffect(() =>
        this.deps.logger.info(
          `arXiv ${date}: cancelled - ${pending.result.reason}`,
        ),
      );
      this.safeEffect(() =>
        this.progress.setError(
          `Daily report cancelled: ${date} (${pending.result.reason})`,
        ),
      );
      await this.safeAsyncEffect(() =>
        this.deps.history.recordCancelled(
          date,
          pending.trigger,
          pending.result.reason,
          pending.at,
        ),
      );
    }
    this.pendingNonCompletions.delete(date);
    return pending.result;
  }

  private async confirmDurablyDone(date: string): Promise<boolean> {
    const store = this.store();
    this.activeWork += 1;
    try {
      const confirmed = await this.deps.lock.withLock(date, async () => {
        try {
          await store.loadAuthoritative();
        } catch {
          return false;
        }
        return store.isDone(date);
      });
      return confirmed ?? false;
    } finally {
      this.activeWork -= 1;
    }
  }

  private async finalizeFailureLocked(
    date: string,
    result: Extract<
      PipelineResult,
      { kind: "failed_transient" | "failed_permanent" }
    >,
    trigger: RunHistoryTrigger,
    now: Date,
    store: StateStore,
  ): Promise<Extract<
    PipelineResult,
    { kind: "failed_transient" | "failed_permanent" }
  >> {
    const failureKind =
      result.kind === "failed_permanent" ? "permanent" : "transient";
    let persistedStatus: "failed_transient" | "failed_permanent" = result.kind;
    try {
      persistedStatus = await store.setFailed(date, failureKind, result.reason);
    } catch (error) {
      this.safeEffect(() =>
        this.deps.logger.error(
          `scheduler: failed to persist failure for ${date}; clearing running state`,
          error,
        ),
      );
      try {
        await store.clearDate(date);
      } catch (clearError) {
        this.safeEffect(() =>
          this.deps.logger.error(
            `scheduler: failed to clear running state for ${date}`,
            clearError,
          ),
        );
      }
    }

    const persistedReason = store.get(date).error ?? result.reason;
    const severity =
      persistedStatus === "failed_permanent" ? "permanent" : "transient";
    if (persistedStatus === "failed_permanent") {
      this.safeEffect(() =>
        this.deps.logger.error(`arXiv ${date} permanent: ${persistedReason}`),
      );
      this.safeEffect(() =>
        this.deps.logger.notice(
          `arXiv ${date}: failed (${persistedReason})`,
          10_000,
        ),
      );
    } else {
      this.safeEffect(() =>
        this.deps.logger.warn(`arXiv ${date} transient: ${persistedReason}`),
      );
    }
    this.safeEffect(() =>
      this.progress.setError(
        `Daily report failed ${severity}: ${date} (${persistedReason})`,
      ),
    );
    await this.safeAsyncEffect(() =>
      this.deps.history.recordFailed(
        date,
        trigger,
        persistedStatus,
        persistedReason,
        now,
      ),
    );
    return { kind: persistedStatus, reason: persistedReason };
  }

  private async storeFailureResult(
    date: string,
    trigger: RunHistoryTrigger,
    now: Date,
    error: unknown,
  ): Promise<Extract<PipelineResult, { kind: "failed_transient" }>> {
    const reason = errorMessage(error);
    this.safeEffect(() =>
      this.deps.logger.error(`scheduler: state store failed for ${date}`, error),
    );
    this.safeEffect(() =>
      this.progress.setError(`Daily report failed transient: ${date} (${reason})`),
    );
    await this.safeAsyncEffect(() =>
      this.deps.history.recordFailed(
        date,
        trigger,
        "failed_transient",
        reason,
        now,
      ),
    );
    return { kind: "failed_transient", reason };
  }

  private safeEffect(effect: () => void): void {
    try {
      effect();
    } catch {
      // In-memory observability is best effort and never changes run state.
    }
  }

  private async safeAsyncEffect(
    effect: () => Promise<void>,
    failureMessage?: string,
  ): Promise<void> {
    try {
      await effect();
    } catch (error) {
      if (failureMessage) {
        this.safeEffect(() => this.deps.logger.error(failureMessage, error));
      }
    }
  }

  private beginCancellationBatch(): RunCancellationBatch | undefined {
    try {
      return this.deps.cancellation?.beginBatch();
    } catch {
      return undefined;
    }
  }

  private beginCancellationRun(
    date: string,
    batch?: RunCancellationBatch,
  ): AbortSignal | undefined {
    try {
      return this.deps.cancellation?.begin(date, batch);
    } catch {
      return undefined;
    }
  }

  private finishCancellationBatch(batch: RunCancellationBatch | undefined): void {
    if (!batch) return;
    this.safeEffect(() => this.deps.cancellation?.finishBatch(batch));
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
