/**
 * Single-flight gate for user-initiated runs of one date. The clicked button is
 * disabled while its own action runs, but a dashboard re-render (for example the
 * recent-dates refresh repainting the toolbar) can replace it with an enabled
 * copy, and the same date can also be requested from the command palette, the
 * "More" menu, or the settings guide. Without a shared gate those paths reach
 * the scheduler, which answers a same-date second request with its generic
 * "lock held" skip (or "already done" if the first run finished in between),
 * next to the second request's own "running…" notice.
 *
 * The gate is keyed per plugin instance so separate windows/instances keep the
 * scheduler's own cross-process lock as their coordination point.
 */
export class ManualRunGate {
  private readonly active = new Set<string>();

  /**
   * Marks `date` as running. Returns the release function, or `undefined` when
   * a manual run for the same date is already in flight.
   */
  tryBegin(date: string): (() => void) | undefined {
    if (this.active.has(date)) return undefined;
    this.active.add(date);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.active.delete(date);
    };
  }

  isActive(date: string): boolean {
    return this.active.has(date);
  }
}

const gates = new WeakMap<object, ManualRunGate>();

/** The gate shared by every manual run surface of one plugin instance. */
export function manualRunGateFor(plugin: object): ManualRunGate {
  let gate = gates.get(plugin);
  if (!gate) {
    gate = new ManualRunGate();
    gates.set(plugin, gate);
  }
  return gate;
}

/** The subset of Obsidian's `Notice` the run feedback needs. */
export interface ManualRunNotice {
  hide(): void;
}

export interface ManualRunFeedback<T> {
  /** Shows and logs one notice; returns a handle so the interim one can be hidden. */
  notice: (message: string, timeoutMs?: number) => ManualRunNotice;
  /** The "running…" notice shown while the scheduler call is in flight. */
  runningMessage: string;
  run: () => Promise<T>;
}

/**
 * Runs one user-initiated manual run for `date`. It acquires the shared gate,
 * shows the interim notice, and always hides that notice and releases the gate
 * when the run settles, so the final result or the caller's error notice is the
 * only feedback left. Returns the run's result, or `undefined` when the date
 * already has a manual run in flight (a notice says so).
 */
export async function runManualDate<T>(
  plugin: object,
  date: string,
  { notice, runningMessage, run }: ManualRunFeedback<T>,
): Promise<T | undefined> {
  const release = manualRunGateFor(plugin).tryBegin(date);
  if (!release) {
    notice(`arXiv Daily: ${date} is already running`);
    return undefined;
  }
  const runningNotice = notice(runningMessage);
  try {
    return await run();
  } finally {
    runningNotice.hide();
    release();
  }
}
