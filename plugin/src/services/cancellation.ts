export class RunCancelledError extends Error {
  constructor(message = "cancelled by user") {
    super(message);
    this.name = "RunCancelledError";
  }
}

export class RunCancellationService {
  private controllers = new Map<string, AbortController>();
  private cancellationRequested = false;
  private cancelReason = "cancelled by user";

  prepareRun(): void {
    if (this.controllers.size === 0) {
      this.cancellationRequested = false;
      this.cancelReason = "cancelled by user";
    }
  }

  begin(date: string): AbortSignal {
    const controller = new AbortController();
    this.controllers.set(date, controller);
    if (this.cancellationRequested) {
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
    this.cancellationRequested = true;
    this.cancelReason = reason;
    for (const controller of this.controllers.values()) {
      if (!controller.signal.aborted) controller.abort(reason);
    }
    return dates;
  }

  activeDates(): string[] {
    return Array.from(this.controllers.keys()).sort();
  }

  isCancellationRequested(): boolean {
    return this.cancellationRequested;
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
