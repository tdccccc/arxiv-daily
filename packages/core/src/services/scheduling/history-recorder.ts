import type { Logger } from "../logger";
import type {
  RunHistoryRecord,
  RunHistoryStore,
  RunHistoryTrigger,
} from "../run-history";
import type { StateStore } from "../state-store";
import { formatDate } from "../../utils/time";

export interface HistoryRecorderDeps {
  runHistory?: Pick<RunHistoryStore, "safeAppend">;
  /** Getter so replaceStore keeps this recorder coherent with the live store. */
  store: () => StateStore;
  dailyPathForDate?: (date: string) => string | undefined;
  now?: () => Date;
  logger?: Pick<Logger, "warn">;
}

export class HistoryRecorder {
  constructor(private readonly deps: HistoryRecorderDeps) {}

  async recordStarted(date: string, trigger: RunHistoryTrigger, at?: Date): Promise<void> {
    await this.record({
      date,
      event: "started",
      trigger,
      status: "running",
      attempts: this.deps.store().get(date).attempts,
    }, at);
  }

  async recordCompleted(
    date: string,
    trigger: RunHistoryTrigger,
    detail: {
      papersWritten: number;
      requestedPapersWritten: number;
      preservedPapersWritten: boolean;
    },
    at?: Date,
  ): Promise<void> {
    const entry = this.deps.store().get(date);
    await this.record({
      date,
      event: "completed",
      trigger,
      status: "completed",
      resultKind: "completed",
      papersWritten: detail.papersWritten,
      requestedPapersWritten: detail.requestedPapersWritten,
      preservedPapersWritten: detail.preservedPapersWritten || undefined,
      attempts: entry.attempts,
    }, at);
  }

  async recordPending(date: string, trigger: RunHistoryTrigger, reason: string, at?: Date): Promise<void> {
    await this.record({
      date,
      event: "pending",
      trigger,
      status: "pending",
      resultKind: "pending",
      reason,
    }, at);
  }

  async recordFailed(
    date: string,
    trigger: RunHistoryTrigger,
    resultKind: "failed_transient" | "failed_permanent",
    reason: string,
    at?: Date,
  ): Promise<void> {
    const entry = this.deps.store().get(date);
    await this.record({
      date,
      event: "failed",
      trigger,
      status: entry.status,
      resultKind,
      reason,
      errorMessage: reason,
      attempts: entry.attempts,
    }, at);
  }

  async recordCancelled(
    date: string,
    trigger: RunHistoryTrigger,
    reason: string,
    at?: Date,
  ): Promise<void> {
    const entry = this.deps.store().get(date);
    await this.record({
      date,
      event: "skipped",
      trigger,
      status: entry.status,
      resultKind: "cancelled",
      reason,
      errorMessage: reason,
      attempts: entry.attempts,
    }, at);
  }

  async recordSkippedForDate(
    dateObj: { y: number; m: number; d: number },
    trigger: RunHistoryTrigger,
    reason: string,
    at?: Date,
  ): Promise<void> {
    await this.recordSkipped(formatDate(dateObj), trigger, reason, at);
  }

  async recordSkipped(date: string, trigger: RunHistoryTrigger, reason: string, at?: Date): Promise<void> {
    const entry = this.deps.store().get(date);
    await this.record({
      date,
      event: "skipped",
      trigger,
      status: entry.status,
      resultKind: "skipped",
      reason,
      errorMessage: reason,
      attempts: entry.attempts,
    }, at);
  }

  private async record(
    record: Omit<RunHistoryRecord, "schemaVersion" | "at" | "dailyPath">,
    at?: Date,
  ): Promise<void> {
    if (!this.deps.runHistory) return;
    const now = at ?? (this.deps.now ?? (() => new Date()))();
    await this.deps.runHistory.safeAppend({
      schemaVersion: 1,
      at: now.toISOString(),
      dailyPath: this.dailyPathForDate(record.date),
      ...record,
    });
  }

  private dailyPathForDate(date: string): string | undefined {
    try {
      return this.deps.dailyPathForDate?.(date);
    } catch (e) {
      this.deps.logger?.warn(`daily path resolution failed for ${date}`, e);
      return undefined;
    }
  }
}
