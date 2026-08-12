export type OperationKind =
  | "daily-run"
  | "detail-summary"
  | "pdf-download"
  | "personal-library-scan"
  | "personal-library-direction-generation"
  | "personal-library-fulltext-index"
  | "paper-index"
  | "paper-note";

export interface OperationSnapshot {
  readonly id: string;
  readonly kind: OperationKind;
  readonly label: string;
  readonly key?: string;
  readonly startedAt: number;
  readonly cancellationRequested: boolean;
}

export interface OperationHandle {
  readonly id: string;
  readonly signal: AbortSignal;
  finish(): void;
}

export type OperationListener = (snapshot: readonly OperationSnapshot[]) => void;

interface ActiveOperation {
  id: string;
  kind: OperationKind;
  label: string;
  key?: string;
  startedAt: number;
  cancellationRequested: boolean;
  controller: AbortController;
}

/** Host-neutral owner of user-visible long-running operations. */
export class OperationRegistry {
  private readonly active = new Map<string, ActiveOperation>();
  private readonly listeners = new Set<OperationListener>();
  private nextId = 1;

  begin(kind: OperationKind, label: string, key?: string): OperationHandle {
    const id = `${kind}:${this.nextId++}`;
    const controller = new AbortController();
    this.active.set(id, {
      id,
      kind,
      label,
      key,
      startedAt: Date.now(),
      cancellationRequested: false,
      controller,
    });
    this.notify();
    let finished = false;
    return {
      id,
      signal: controller.signal,
      finish: () => {
        if (finished) return;
        finished = true;
        this.finish(id);
      },
    };
  }

  snapshot(): readonly OperationSnapshot[] {
    return Array.from(this.active.values(), snapshotOf);
  }

  find(kind: OperationKind, key: string): OperationSnapshot | undefined {
    return this.snapshot().find((operation) => operation.kind === kind && operation.key === key);
  }

  cancel(id: string, reason = "cancelled by user"): boolean {
    const operation = this.active.get(id);
    if (!operation) return false;
    if (!operation.cancellationRequested) {
      operation.cancellationRequested = true;
      operation.controller.abort(reason);
      this.notify();
    }
    return true;
  }

  cancelAll(reason = "cancelled by user"): readonly OperationSnapshot[] {
    const operations = this.snapshot();
    for (const operation of operations) this.cancel(operation.id, reason);
    return operations;
  }

  finish(id: string): boolean {
    const removed = this.active.delete(id);
    if (removed) this.notify();
    return removed;
  }

  subscribe(listener: OperationListener): () => void {
    this.listeners.add(listener);
    this.callListener(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of Array.from(this.listeners)) this.callListener(listener);
  }

  private callListener(listener: OperationListener): void {
    try {
      listener(this.snapshot());
    } catch {
      // A host listener must never disrupt registry state or another listener.
    }
  }
}

function snapshotOf(operation: ActiveOperation): OperationSnapshot {
  return {
    id: operation.id,
    kind: operation.kind,
    label: operation.label,
    key: operation.key,
    startedAt: operation.startedAt,
    cancellationRequested: operation.cancellationRequested,
  };
}
