import type { PipelineResult, RunOutcome } from "@arxiv-daily/core";

export interface CliRunEvent {
  date: string;
  kind: PipelineResult["kind"] | "skipped";
  outcome?: RunOutcome;
  papersWritten?: number;
}

export interface WritableTextStream {
  write(chunk: string): unknown;
}

export interface CliIo {
  /** Typed host notification; never inferred from terminal output. */
  onRunResult?: (event: CliRunEvent) => void;
  stdout: WritableTextStream;
  stderr: WritableTextStream;
}

export function isCliRunEvent(value: unknown): value is CliRunEvent {
  if (!value || typeof value !== "object") return false;
  const event = value as Partial<CliRunEvent>;
  return typeof event.date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(event.date)
    && ["completed", "pending", "cancelled", "failed_transient", "failed_permanent", "skipped"].includes(event.kind ?? "")
    && (event.outcome === undefined || ["awaiting_announcement", "no_updates", "no_matches", "papers_written"].includes(event.outcome))
    && (event.papersWritten === undefined || Number.isSafeInteger(event.papersWritten) && event.papersWritten >= 0);
}
