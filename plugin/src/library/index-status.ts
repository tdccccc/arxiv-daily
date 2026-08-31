/**
 * What the settings page is allowed to know about full-text indexing.
 *
 * Indexing already reported itself: `indexPersonalLibraryFullText` pushes every
 * step to the status bar. But the settings page is a modal over that status bar,
 * so a person who starts a run from the Library row is looking at the one place
 * the progress cannot reach — the button they just pressed goes quiet for hours.
 * This store is the second consumer that fixes it, and it is a store rather than
 * a direct call because the run and the settings tab do not know about each
 * other: the run publishes, the tab subscribes if it happens to be open.
 *
 * Two facts live here, for the same row and the same reason. `activity` is the
 * run in flight. `lastRun` is what the previous one left behind — the status bar
 * hides its completion after four seconds and the notice after ten, so an index
 * that ran for hours otherwise leaves nothing behind saying whether it finished.
 */

/** A run in flight, as much of it as a settings row can use. */
export interface LibraryIndexActivity {
  /** The registry operation to cancel; the row cancels this run, not "a run". */
  operationId: string;
  /** What the run is doing now, already phrased for a reader. */
  phase: string;
  /** Position within `total`, when the current phase counts anything. */
  completed?: number;
  total?: number;
  /**
   * Cancellation has been asked for and the run has not stopped yet. Separate
   * from simply ending, because aborting a step can take a while and a button
   * that still says "Cancel" invites a second press at nothing.
   */
  cancelling: boolean;
}

/**
 * The durable trace of the last finished run, read back from the knowledge base
 * manifest rather than remembered from the summary that produced it — the
 * manifest is what search actually reads, so it cannot claim an index that is
 * no longer there.
 */
export interface LibraryIndexRun {
  /** Manifest `updatedAt`: when the index was last committed. */
  updatedAt: string;
  /** Papers the manifest holds, i.e. what a search can currently reach. */
  papers: number;
}

export interface LibraryIndexStatus {
  activity?: LibraryIndexActivity;
  lastRun?: LibraryIndexRun;
}

export type LibraryIndexStatusListener = (status: LibraryIndexStatus) => void;

export class LibraryIndexStatusStore {
  private status: LibraryIndexStatus = {};
  private readonly listeners = new Set<LibraryIndexStatusListener>();

  snapshot(): LibraryIndexStatus {
    return this.status;
  }

  /**
   * Subscribe and receive the current status immediately, so a caller never has
   * to render once from `snapshot()` and again from the first event.
   */
  subscribe(listener: LibraryIndexStatusListener): () => void {
    this.listeners.add(listener);
    this.callListener(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  beginRun(operationId: string, phase: string): void {
    this.status = {
      ...this.status,
      activity: { operationId, phase, cancelling: false },
    };
    this.notify();
  }

  /**
   * Report where the run has got to. Counts are dropped rather than carried
   * forward when a phase stops counting, so the row cannot show a stale
   * fraction from the phase before.
   */
  report(update: { phase?: string; completed?: number; total?: number }): void {
    const activity = this.status.activity;
    if (!activity) return;
    const next: LibraryIndexActivity = {
      operationId: activity.operationId,
      phase: update.phase ?? activity.phase,
      cancelling: activity.cancelling,
      ...(update.completed !== undefined && update.total !== undefined
        ? { completed: update.completed, total: update.total }
        : {}),
    };
    if (sameActivity(activity, next)) return;
    this.status = { ...this.status, activity: next };
    this.notify();
  }

  markCancelling(): void {
    const activity = this.status.activity;
    if (!activity || activity.cancelling) return;
    this.status = { ...this.status, activity: { ...activity, cancelling: true } };
    this.notify();
  }

  endRun(): void {
    if (!this.status.activity) return;
    const { activity: _activity, ...rest } = this.status;
    this.status = rest;
    this.notify();
  }

  setLastRun(run: LibraryIndexRun | undefined): void {
    if (sameRun(this.status.lastRun, run)) return;
    this.status = run ? { ...this.status, lastRun: run } : dropLastRun(this.status);
    this.notify();
  }

  private notify(): void {
    for (const listener of Array.from(this.listeners)) this.callListener(listener);
  }

  private callListener(listener: LibraryIndexStatusListener): void {
    try {
      listener(this.status);
    } catch {
      // A settings row that throws while rendering progress must not take the
      // indexing run down with it, nor stop the other listeners.
    }
  }
}

function dropLastRun(status: LibraryIndexStatus): LibraryIndexStatus {
  const { lastRun: _lastRun, ...rest } = status;
  return rest;
}

function sameActivity(left: LibraryIndexActivity, right: LibraryIndexActivity): boolean {
  return left.operationId === right.operationId
    && left.phase === right.phase
    && left.completed === right.completed
    && left.total === right.total
    && left.cancelling === right.cancelling;
}

function sameRun(left: LibraryIndexRun | undefined, right: LibraryIndexRun | undefined): boolean {
  if (!left || !right) return left === right;
  return left.updatedAt === right.updatedAt && left.papers === right.papers;
}
