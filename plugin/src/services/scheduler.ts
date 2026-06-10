import type { Logger } from "./logger";
import type { StateStore } from "./state-store";
import type { RunLock } from "./run-lock";
import type { PluginSettings } from "../settings/types";
import {
  todayInTz,
  formatDate,
  parseHHMM,
  minutesSinceMidnight,
  daysBefore,
  isWeekendInTz,
  isWeekendDate,
} from "../utils/time";
import type { PipelineResult } from "../pipeline/pipeline";
import type { ProgressReporter } from "./progress";
import { NoopProgressReporter } from "./progress";

export interface SchedulerDeps {
  getSettings: () => PluginSettings;
  store: StateStore;
  lock: RunLock;
  runForDate: (date: string) => Promise<PipelineResult>;
  logger: Logger;
  now?: () => Date;
  progress?: ProgressReporter;
}

export class SchedulerService {
  private intervalHandle: number | null = null;
  private readonly progress: ProgressReporter;

  constructor(private deps: SchedulerDeps) {
    this.progress = deps.progress ?? new NoopProgressReporter();
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

  async tick(): Promise<void> {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) return;
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();

    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const t = parseHHMM(s.schedule.runAtLocal);
    const scheduledMin = t.hour * 60 + t.minute;

    for (let i = 0; i < s.schedule.lookbackDays; i++) {
      const dateObj = daysBefore(todayObj, i);
      const date = formatDate(dateObj);
      const isToday = date === today;
      this.progress.setBatch(i + 1, s.schedule.lookbackDays, date);
      if (isWeekendDate(dateObj)) continue;
      await this.tickDate(date, {
        now,
        timeGate: isToday ? { scheduledMin, minutesNow } : undefined,
      });
    }
    this.progress.setIdle(this.latestCompleted());
  }

  async tickToday(): Promise<
    PipelineResult | { kind: "skipped"; reason: string } | undefined
  > {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) {
      return { kind: "skipped", reason: "disabled" };
    }
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      return { kind: "skipped", reason: "weekend" };
    }
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    this.progress.setBatch(1, 1, today);
    const result = await this.tickDate(today, { now });
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }

  async tickTodayScheduled(): Promise<
    PipelineResult | { kind: "skipped"; reason: string } | undefined
  > {
    const s = this.deps.getSettings();
    if (!s.schedule.enabled) {
      return { kind: "skipped", reason: "disabled" };
    }
    const tz = s.arxiv.timezone;
    const now = (this.deps.now ?? (() => new Date()))();
    if (isWeekendInTz(now, tz)) {
      this.progress.setIdle(this.latestCompleted(), "weekend");
      return { kind: "skipped", reason: "weekend" };
    }
    const todayObj = todayInTz(now, tz);
    const today = formatDate(todayObj);
    const minutesNow = minutesSinceMidnight(now, tz);
    const t = parseHHMM(s.schedule.runAtLocal);
    const scheduledMin = t.hour * 60 + t.minute;
    this.progress.setBatch(1, 1, today);
    const result = await this.tickDate(today, {
      now,
      timeGate: { scheduledMin, minutesNow },
    });
    this.progress.setIdle(this.latestCompleted());
    if (result === undefined) {
      return { kind: "skipped", reason: "guarded" };
    }
    return result;
  }

  private async tickDate(
    date: string,
    opts: {
      now: Date;
      timeGate?: { scheduledMin: number; minutesNow: number };
    },
  ): Promise<PipelineResult | undefined> {
    const s = this.deps.getSettings();
    const entry = this.deps.store.get(date);
    if (this.deps.store.isDone(date)) return undefined;
    if (entry.status === "running") return undefined;

    if (opts.timeGate && opts.timeGate.minutesNow < opts.timeGate.scheduledMin) {
      return undefined;
    }

    if (entry.status === "failed_transient") {
      const tickMs = s.schedule.tickIntervalMin * 60_000;
      if (opts.now.getTime() - entry.lastAttempt < tickMs) return undefined;
    }

    return await this.tryRun(date);
  }

  /** Manual trigger: ignore scheduled-time gate, still respect lock and isDone. */
  async runForDateNow(
    date: string,
  ): Promise<PipelineResult | { kind: "skipped"; reason: string }> {
    const entry = this.deps.store.get(date);
    if (entry.status === "running") {
      return { kind: "skipped", reason: "already running" };
    }
    this.progress.setBatch(1, 1, date);
    const result = await this.tryRun(date);
    this.progress.setIdle(this.latestCompleted());
    return result ?? { kind: "skipped", reason: "lock held" };
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

    const results: Array<{
      date: string;
      result: PipelineResult | { kind: "skipped"; reason: string };
    }> = [];

    for (let i = 0; i < s.schedule.lookbackDays; i++) {
      const date = formatDate(daysBefore(todayObj, i));
      const entry = this.deps.store.get(date);
      if (this.deps.store.isDone(date)) continue;
      if (entry.status === "running") {
        results.push({ date, result: { kind: "skipped", reason: "already running" } });
        continue;
      }
      this.progress.setBatch(i + 1, s.schedule.lookbackDays, date);
      const r = await this.tryRun(date);
      results.push({ date, result: r ?? { kind: "skipped", reason: "lock held" } });
    }
    this.progress.setIdle(this.latestCompleted());
    return results;
  }

  private async tryRun(date: string): Promise<PipelineResult | undefined> {
    return this.deps.lock.withLock(date, async () => {
      await this.deps.store.setRunning(date);
      let result: PipelineResult;
      try {
        result = await this.deps.runForDate(date);
      } catch (e) {
        result = { kind: "failed_transient", reason: (e as Error).message };
      }
      if (result.kind === "completed") {
        await this.deps.store.setCompleted(date, result.papersWritten);
        this.deps.logger.notice(`arXiv ${date}: ${result.papersWritten} papers written`);
      } else if (result.kind === "failed_transient") {
        await this.deps.store.setFailed(date, "transient", result.reason);
        this.deps.logger.warn(`arXiv ${date} transient: ${result.reason}`);
      } else {
        await this.deps.store.setFailed(date, "permanent", result.reason);
        this.deps.logger.error(`arXiv ${date} permanent: ${result.reason}`);
        this.deps.logger.notice(`arXiv ${date}: failed (${result.reason})`, 10_000);
      }
      return result;
    });
  }

  private latestCompleted(): string | undefined {
    const snap = this.deps.store.snapshot();
    const done = Object.entries(snap)
      .filter(([, v]) => v.status === "completed")
      .map(([k]) => k)
      .sort();
    return done[done.length - 1];
  }
}
