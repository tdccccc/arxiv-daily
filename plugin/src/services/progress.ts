export type ProgressStage =
  | "fetch-recent"
  | "enrich-abstract"
  | "filter"
  | "fetch-content"
  | "summarize-daily"
  | "write-detail";

export type IdleReason = "weekend" | "disabled";

export interface ProgressReporter {
  setBatch(currentDay: number, totalDays: number, date: string): void;
  setStage(stage: ProgressStage, current?: number, total?: number): void;
  setIdle(lastCompletedDate?: string, reason?: IdleReason): void;
  setDisabled(): void;
}

export class NoopProgressReporter implements ProgressReporter {
  setBatch(_currentDay: number, _totalDays: number, _date: string): void {}
  setStage(_stage: ProgressStage, _current?: number, _total?: number): void {}
  setIdle(_lastCompletedDate?: string, _reason?: IdleReason): void {}
  setDisabled(): void {}
}
