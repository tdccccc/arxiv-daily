import type { RunStateEntry } from "../../settings/types";
import { isMinutesWithinWindow } from "../../utils/time";
import type { RunGateDecision, StateStoreRead, TickGateOptions } from "./types";

/**
 * Composite per-date gate used by scheduled paths (tick, tickTodayScheduled).
 * Mirrors the four guards of the legacy tickDate. Advisory optimization; the
 * lock in tryRun is the real correctness guard.
 */
export function checkTickGate(
  date: string,
  store: StateStoreRead,
  opts: TickGateOptions,
): RunGateDecision {
  const entry = store.get(date);
  if (store.isDone(date)) {
    return { allow: false, reason: "already-done" };
  }
  if (entry.status === "running") {
    return { allow: false, reason: "running" };
  }
  if (
    opts.timeGate &&
    !isMinutesWithinWindow(opts.timeGate.minutesNow, opts.timeGate.scheduledMin, opts.timeGate.endMin)
  ) {
    return { allow: false, reason: "outside-window" };
  }
  if (shouldBackoffTransient(date, store, opts.tickIntervalMin, opts.now)) {
    return { allow: false, reason: "transient-backoff" };
  }
  return { allow: true };
}

export function isRunning(date: string, store: StateStoreRead): boolean {
  return store.get(date).status === "running";
}

export function isWithinTimeGate(minutesNow: number, scheduledMin: number, endMin: number): boolean {
  return isMinutesWithinWindow(minutesNow, scheduledMin, endMin);
}

export function shouldBackoffTransient(
  date: string,
  store: StateStoreRead,
  tickIntervalMin: number,
  now: Date,
): boolean {
  const entry: RunStateEntry = store.get(date);
  if (entry.status !== "failed_transient") return false;
  const tickMs = Math.max(1, tickIntervalMin) * 60_000;
  return now.getTime() - entry.lastAttempt < tickMs;
}
