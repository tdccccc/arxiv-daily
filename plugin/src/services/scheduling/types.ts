import type { PipelineResult } from "../../pipeline/pipeline";
import type { RunStateEntry, RunStatus } from "../../settings/types";
import type { RunHistoryTrigger } from "../run-history";

/** Per-date gate inputs used by the scheduled paths. */
export interface TimeGate {
  scheduledMin: number;
  endMin: number;
  minutesNow: number;
}

/** Options passed to the per-date gate check. */
export interface TickGateOptions {
  now: Date;
  timeGate?: TimeGate;
  tickIntervalMin: number;
}

/** Result of a per-date gate check. */
export type RunGateDecision =
  | { allow: true }
  | { allow: false; reason: "already-done" | "running" | "outside-window" | "transient-backoff" };

/** Narrow read-only view of the state store needed by gates. */
export interface StateStoreRead {
  get(date: string): RunStateEntry;
  isDone(date: string): boolean;
}

// Re-export shared types so modules import from one place without sibling cycles.
export type { PipelineResult, RunHistoryTrigger, RunStateEntry, RunStatus };
