export class RunCancelledError extends Error {
  constructor(message = "cancelled by user") {
    super(message);
    this.name = "RunCancelledError";
  }
}

import { OperationRegistry, type OperationHandle } from "./operations";

export interface RunCancellationBatch {
  readonly id: number;
  readonly signal: AbortSignal;
  isCancellationRequested(): boolean;
}

interface ActiveRun {
  controller: AbortController;
  batchId?: number;
}

interface BatchState {
  cancelled: boolean;
  operation: OperationHandle;
}

export class RunCancellationService {
  private controllers = new Map<string, ActiveRun>();
  private batches = new Map<number, BatchState>();
  private nextBatchId = 1;
  private cancelReason = "cancelled by user";

  constructor(private readonly operations = new OperationRegistry()) {}

  prepareRun(): void {}

  beginBatch(label = "Daily run"): RunCancellationBatch {
    const id = this.nextBatchId++;
    const operation = this.operations.begin("daily-run", label);
    const state: BatchState = { cancelled: false, operation };
    operation.signal.addEventListener("abort", () => {
      state.cancelled = true;
      this.cancelBatchRuns(id, cancelReason(operation.signal));
    }, { once: true });
    this.batches.set(id, state);
    return {
      id,
      signal: operation.signal,
      isCancellationRequested: () => state.cancelled || operation.signal.aborted,
    };
  }

  finishBatch(batch: RunCancellationBatch): void {
    const state = this.batches.get(batch.id);
    this.batches.delete(batch.id);
    state?.operation.finish();
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
    this.cancelReason = reason;
    for (const batch of this.batches.values()) {
      batch.cancelled = true;
      this.operations.cancel(batch.operation.id, reason);
    }
    for (const active of this.controllers.values()) {
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

  private cancelBatchRuns(batchId: number, reason: string): void {
    for (const active of this.controllers.values()) {
      if (active.batchId === batchId && !active.controller.signal.aborted) {
        active.controller.abort(reason);
      }
    }
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
