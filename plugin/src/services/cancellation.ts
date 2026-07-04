export class RunCancelledError extends Error {
  constructor(message = "cancelled by user") {
    super(message);
    this.name = "RunCancelledError";
  }
}

export interface RunCancellationBatch {
  readonly id: number;
  isCancellationRequested(): boolean;
}

interface ActiveRun {
  controller: AbortController;
  batchId?: number;
}

export class RunCancellationService {
  private controllers = new Map<string, ActiveRun>();
  private batches = new Map<number, { cancelled: boolean }>();
  private nextBatchId = 1;
  private cancelReason = "cancelled by user";

  prepareRun(): void {}

  beginBatch(): RunCancellationBatch {
    const id = this.nextBatchId++;
    const state = { cancelled: false };
    this.batches.set(id, state);
    return {
      id,
      isCancellationRequested: () => state.cancelled,
    };
  }

  finishBatch(batch: RunCancellationBatch): void {
    this.batches.delete(batch.id);
  }

  begin(date: string, batch?: RunCancellationBatch): AbortSignal {
    const existing = this.controllers.get(date);
    if (existing && !existing.controller.signal.aborted) {
      existing.controller.abort("superseded by a new run for the same date");
    }
    if (existing) this.controllers.delete(date);
    const controller = new AbortController();
    this.controllers.set(date, { controller, batchId: batch?.id });
    if (batch?.isCancellationRequested()) {
      controller.abort(this.cancelReason);
    }
    return controller.signal;
  }

  finish(date: string): void {
    this.controllers.delete(date);
  }

  cancelAll(reason = "cancelled by user"): string[] {
    const dates = Array.from(this.controllers.keys());
    if (dates.length === 0) return [];
    this.cancelReason = reason;
    for (const active of this.controllers.values()) {
      if (active.batchId != null) {
        const batch = this.batches.get(active.batchId);
        if (batch) batch.cancelled = true;
      }
      if (!active.controller.signal.aborted) active.controller.abort(reason);
    }
    return dates;
  }

  activeDates(): string[] {
    return Array.from(this.controllers.keys()).sort();
  }

  isCancellationRequested(): boolean {
    return Array.from(this.batches.values()).some((batch) => batch.cancelled);
  }
}

export function throwIfCancelled(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw new RunCancelledError(cancelReason(signal));
}

export function isCancellationError(error: unknown): boolean {
  return error instanceof RunCancelledError || isAbortError(error);
}

function isAbortError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const anyError = error as any;
  return anyError.name === "AbortError" || anyError.code === "ABORT_ERR";
}

function cancelReason(signal: AbortSignal): string {
  const reason = (signal as any).reason;
  return typeof reason === "string" && reason ? reason : "cancelled by user";
}
