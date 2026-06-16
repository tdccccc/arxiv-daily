import type {
  IdleReason,
  ProgressReporter,
  ProgressStage,
} from "../core/adapters";

export type {
  IdleReason,
  ProgressReporter,
  ProgressStage,
} from "../core/adapters";

export class NoopProgressReporter implements ProgressReporter {
  setTask(_title: string, _detail?: string): void {}
  setBatch(_currentDay: number, _totalDays: number, _date: string): void {}
  setStage(_stage: ProgressStage, _current?: number, _total?: number): void {}
  setComplete(_message?: string): void {}
  setError(_message: string): void {}
  setIdle(_lastCompletedDate?: string, _reason?: IdleReason): void {}
  setDisabled(): void {}
}
