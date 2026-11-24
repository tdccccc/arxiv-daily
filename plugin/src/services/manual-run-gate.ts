/**
 * Single-flight gate for user-initiated runs of one date. The clicked button is
 * disabled while its own action runs, but a dashboard re-render (for example the
 * recent-dates refresh repainting the toolbar) can replace it with an enabled
 * copy, and the same date can also be requested from the command palette, the
 * "More" menu, or the settings guide. Those paths all reach the scheduler, which
 * otherwise queues the second request on the date lock and reports the stale
 * "skipped: already done" result once the first run finishes.
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
