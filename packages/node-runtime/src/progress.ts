import type {
  IdleReason,
  ProgressReporter,
  ProgressStage,
} from "@arxiv-daily/core";

export interface WritableTextStream {
  write(chunk: string): unknown;
}

export class StreamProgressReporter implements ProgressReporter {
  constructor(private stream: WritableTextStream = process.stderr) {}

  setTask(title: string, detail?: string): void {
    this.write(`task ${title}${detail ? ` ${detail}` : ""}`);
  }

  setBatch(currentDay: number, totalDays: number, date: string): void {
    this.write(`batch ${currentDay}/${totalDays} ${date}`);
  }

  setStage(stage: ProgressStage, current?: number, total?: number): void {
    const count =
      current !== undefined && total !== undefined ? ` ${current}/${total}` : "";
    this.write(`stage ${stage}${count}`);
  }

  setComplete(message?: string): void {
    this.write(`complete${message ? ` ${message}` : ""}`);
  }

  setError(message: string): void {
    this.write(`error ${message}`);
  }

  setIdle(lastCompletedDate?: string, reason?: IdleReason): void {
    const suffix = [
      lastCompletedDate ? `last=${lastCompletedDate}` : "",
      reason ? `reason=${reason}` : "",
    ].filter(Boolean);
    this.write(`idle${suffix.length ? ` ${suffix.join(" ")}` : ""}`);
  }

  setDisabled(): void {
    this.write("disabled");
  }

  private write(line: string): void {
    this.stream.write(`[arxiv-daily] ${line}\n`);
  }
}
