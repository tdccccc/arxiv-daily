import type { RunOutcome, RunState, RunStateEntry, RunStatus } from "../settings/types";
import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";

const MAX_TRANSIENT_ATTEMPTS = 10;
export const STALE_RUNNING_RECOVERY_MS = 60 * 60 * 1000;

/**
 * A completed date whose run wrote a daily report (legacy entries have no
 * outcome, so a positive paper count stands in for it). Dates that finished
 * with no matches or no updates never had a daily file.
 */
export function isReportWrittenRunState(entry: RunStateEntry | undefined): boolean {
  if (entry?.status !== "completed") return false;
  if (entry.outcome) return entry.outcome === "papers_written";
  return (entry.papersWritten ?? 0) > 0;
}

/** A run-state file that cannot be parsed, so every write to it is refused. */
export interface RunStateDamage {
  path: string;
  reason: string;
}

export type StateLoadFn = () => Promise<{ runState: RunState; damaged?: RunStateDamage }>;
export type StateSaveFn = (data: { runState: RunState }) => Promise<void>;
type StateStoreLogger = { warn(message: string, ...rest: unknown[]): void };

export interface StorageStateStorePaths {
  indexDir: string;
  runStatePath: string;
  runHistoryPath: string;
}

const stateMutationQueues = new Map<string, Promise<unknown>>();

/**
 * The file that mutations build on holds no valid run state. Distinct from an
 * unsupported schema or invalid optional field, which may be newer data that
 * must survive and so are never cleared.
 */
export class CorruptRunStateError extends Error {
  readonly code = "CORRUPT_RUN_STATE" as const;

  constructor(readonly path: string, readonly reason: string) {
    super(
      `run-state is damaged and was left untouched: ${path} (${reason}). ` +
        `Run "Clear arXiv Daily run state" in Obsidian, or move the file aside, to start a new history; ` +
        `clearing keeps the damaged file as ${path}.corrupt-<time>.`,
    );
    this.name = "CorruptRunStateError";
  }
}

class UnsupportedRunStateSchemaError extends Error {
  readonly code = "UNSUPPORTED_RUN_STATE_SCHEMA" as const;

  constructor(schemaVersion: unknown) {
    super(`unsupported run-state schema version: ${String(schemaVersion)}`);
    this.name = "UnsupportedRunStateSchemaError";
  }
}

class InvalidRunStateOptionalFieldError extends Error {
  readonly code = "INVALID_RUN_STATE_OPTIONAL_FIELD" as const;

  constructor(date: string) {
    super(`invalid run-state entry: ${date}`);
    this.name = "InvalidRunStateOptionalFieldError";
  }
}

export class StateStore {
  private state: RunState = {};
  private damaged: RunStateDamage | undefined;
  private mutationQueue: Promise<unknown> = Promise.resolve();

  /**
   * `quarantineFn` moves an unreadable durable file aside (keeping it) so that
   * clearAll can start over; without it a damaged file blocks every mutation.
   */
  constructor(
    private readonly loadFn: StateLoadFn,
    private readonly saveFn: StateSaveFn,
    private readonly mutationQueueKey?: string,
    private readonly authoritativeLoadFn: StateLoadFn = loadFn,
    private readonly quarantineFn?: (path: string) => Promise<void>,
  ) {}

  async load(): Promise<void> {
    await this.enqueueStateOperation(async () => {
      const data = await this.loadFn();
      this.state = cloneRunState(data.runState);
      this.damaged = data.damaged;
    });
  }

  async loadAuthoritative(): Promise<void> {
    await this.enqueueStateOperation(async () => {
      const data = await this.authoritativeLoadFn();
      this.state = cloneRunState(data.runState);
      this.damaged = undefined;
    });
  }

  /**
   * Set when the last load found a run-state file it could not parse. Loading
   * still succeeds, with whatever the backup held or an empty history, but
   * every mutation fails until the file is cleared, so hosts should say so.
   */
  damage(): RunStateDamage | undefined {
    return this.damaged;
  }

  get(date: string): RunStateEntry {
    return (
      this.state[date] ?? {
        status: "pending",
        lastAttempt: 0,
        attempts: 0,
      }
    );
  }

  isDone(date: string): boolean {
    const s = this.get(date).status;
    return s === "completed" || s === "failed_permanent" || s === "skipped";
  }

  async setSkipped(date: string, reason: string): Promise<void> {
    await this.enqueueMutation((candidate) => {
      candidate[date] = skippedEntry(getRunStateEntry(candidate, date), reason);
      return { result: undefined, persist: true };
    });
  }

  /**
   * Skips a date on the user's say-so, but only while it still has no outcome
   * to lose: a completed, failed-for-good or running date is left untouched
   * and false is returned. The check is made on the durable state, so a run
   * that finished while the user was deciding is never overwritten.
   */
  async skipIfOpen(date: string, reason: string): Promise<boolean> {
    return await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      if (prev.status !== "pending" && prev.status !== "failed_transient") {
        return { result: false, persist: false };
      }
      candidate[date] = skippedEntry(prev, reason);
      return { result: true, persist: true };
    });
  }

  async setRunning(date: string): Promise<void> {
    await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      candidate[date] = {
        ...prev,
        status: "running",
        lastAttempt: Date.now(),
        attempts: prev.attempts + 1,
        failureAttempts: prev.failureAttempts ?? prev.attempts,
        outcome: undefined,
      };
      return { result: undefined, persist: true };
    });
  }

  async setCompleted(date: string, papersWritten: number, outcome?: RunOutcome): Promise<void> {
    await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      candidate[date] = {
        ...prev,
        status: "completed",
        lastAttempt: Date.now(),
        papersWritten,
        outcome,
        error: undefined,
        rerunOf: undefined,
      };
      return { result: undefined, persist: true };
    });
  }

  async setPending(date: string, reason: string, outcome?: RunOutcome): Promise<void> {
    await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      candidate[date] = {
        ...prev,
        status: "pending",
        outcome,
        lastAttempt: Date.now(),
        error: reason,
      };
      return { result: undefined, persist: true };
    });
  }

  async setFailed(
    date: string,
    kind: "transient" | "permanent",
    message: string,
  ): Promise<Extract<RunStatus, "failed_transient" | "failed_permanent">> {
    return await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      let status: Extract<RunStatus, "failed_transient" | "failed_permanent"> =
        kind === "permanent" ? "failed_permanent" : "failed_transient";
      const failureAttempts = priorFailureAttempts(prev) + 1;
      const retriesExhausted =
        status === "failed_transient" && failureAttempts >= MAX_TRANSIENT_ATTEMPTS;
      if (retriesExhausted) status = "failed_permanent";
      const persistedMessage = retriesExhausted
        ? `retries exhausted after ${failureAttempts} attempts: ${message}`
        : message;
      candidate[date] = {
        ...prev,
        status,
        failureAttempts,
        outcome: undefined,
        lastAttempt: Date.now(),
        error: persistedMessage,
      };
      return { result: status, persist: true };
    });
  }

  /**
   * Starts the rerun of a completed date whose daily report was deleted. One
   * write replaces the completed entry with a `running` one that carries it as
   * `rerunOf`, so no crash can leave the date cleared but unmarked. Returns
   * false, writing nothing, when the durable entry is no longer completed.
   */
  async beginDeletedReportRerun(date: string): Promise<boolean> {
    return await this.enqueueMutation((candidate) => {
      const prev = candidate[date];
      if (prev?.status !== "completed") return { result: false, persist: false };
      const origin = copyKnownEntryFields(prev);
      candidate[date] = {
        status: "running",
        lastAttempt: Date.now(),
        attempts: 1,
        failureAttempts: 0,
        rerunOf: origin,
      };
      return { result: true, persist: true };
    });
  }

  /**
   * Puts back the completed entry a deleted-report rerun replaced, unless the
   * rerun has completed since. With minAgeMs the restore also waits for a
   * `running` entry to be that old (a run that may still be alive keeps its
   * state). Returns whether an entry was restored.
   */
  async restoreRerunOrigin(date: string, now = Date.now(), minAgeMs = 0): Promise<boolean> {
    return await this.enqueueMutation((candidate) => {
      const entry = candidate[date];
      if (!entry?.rerunOf || entry.status === "completed") {
        return { result: false, persist: false };
      }
      if (entry.status === "running" && minAgeMs > 0 && now - entry.lastAttempt < minAgeMs) {
        return { result: false, persist: false };
      }
      candidate[date] = { ...entry.rerunOf };
      return { result: true, persist: true };
    });
  }

  async clearDate(date: string): Promise<void> {
    await this.enqueueMutation((candidate) => {
      if (!(date in candidate)) return { result: undefined, persist: false };
      delete candidate[date];
      return { result: undefined, persist: true };
    });
  }

  /**
   * Also the way out of a damaged run-state file: the user asked for the old
   * content to go, so it is moved aside instead of blocking the clear.
   * replaceAll does not do this; it only serves the automatic legacy
   * migration, which must not discard a file on its own.
   */
  async clearAll(): Promise<void> {
    await this.enqueueMutation((candidate) => {
      for (const date of Object.keys(candidate)) delete candidate[date];
      return { result: undefined, persist: true };
    }, true);
  }

  async replaceAll(runState: RunState): Promise<void> {
    await this.enqueueMutation((candidate) => {
      for (const date of Object.keys(candidate)) delete candidate[date];
      Object.assign(candidate, cloneRunState(runState));
      return { result: undefined, persist: true };
    });
  }

  async recoverStaleRunning(
    now = Date.now(),
    maxAgeMs = STALE_RUNNING_RECOVERY_MS,
  ): Promise<string[]> {
    return await this.enqueueMutation((candidate) => {
      const recovered: string[] = [];
      for (const [date, entry] of Object.entries(candidate)) {
        if (entry.status !== "running") continue;
        if (now - entry.lastAttempt < maxAgeMs) continue;
        candidate[date] = entry.rerunOf
          ? { ...entry.rerunOf }
          : recoveredRunningEntry(
            entry,
            "recovered stale running state after startup",
            now,
          );
        recovered.push(date);
      }
      return {
        result: recovered.sort(),
        persist: recovered.length > 0,
      };
    });
  }

  /**
   * Recovers one date left `running` by a run that no longer exists. The caller
   * sets the minimum age: 0 when it holds the Vault-wide run lock (nothing can
   * be running), STALE_RUNNING_RECOVERY_MS when it cannot prove that. Returns
   * false when the durable record is not `running` or is younger than that.
   * The crashed attempt keeps its own time, so the retry backoff counts from
   * when it started rather than from the recovery.
   */
  async recoverRunningDate(date: string, now: number, minAgeMs: number): Promise<boolean> {
    return await this.enqueueMutation((candidate) => {
      const entry = candidate[date];
      if (entry?.status !== "running" || (minAgeMs > 0 && now - entry.lastAttempt < minAgeMs)) {
        return { result: false, persist: false };
      }
      candidate[date] = recoveredRunningEntry(
        entry,
        "recovered stale running state: the previous run did not finish",
      );
      return { result: true, persist: true };
    });
  }

  failedDates(): string[] {
    return Object.entries(this.state)
      .filter(([, v]) => v.status === "failed_transient" || v.status === "failed_permanent")
      .map(([date]) => date)
      .sort();
  }

  snapshot(): RunState {
    const out: RunState = {};
    for (const [k, v] of Object.entries(this.state)) {
      out[k] = { ...v };
    }
    return out;
  }

  private async enqueueMutation<T>(
    mutate: (
      candidate: RunState,
    ) => Promise<{ result: T; persist: boolean }> | { result: T; persist: boolean },
    discardDamaged = false,
  ): Promise<T> {
    return this.enqueueStateOperation(async () => {
      const data = await this.loadForMutation(discardDamaged);
      const durableState = cloneRunState(data.runState);
      const candidate = cloneRunState(durableState);
      const { result, persist } = await mutate(candidate);
      if (!persist) {
        this.state = durableState;
        return result;
      }

      try {
        await this.saveFn({ runState: candidate });
      } catch (commitFailure) {
        try {
          const readback = await this.authoritativeLoadFn();
          const durableReadback = cloneRunState(readback.runState);
          if (runStatesEqual(durableReadback, candidate)) {
            this.state = durableReadback;
            return result;
          }
        } catch {
          // The original save failure remains authoritative for this mutation.
        }
        this.state = durableState;
        throw commitFailure;
      }

      let readback: { runState: RunState };
      try {
        readback = await this.authoritativeLoadFn();
      } catch (confirmationFailure) {
        this.state = durableState;
        throw confirmationFailure;
      }
      const durableReadback = cloneRunState(readback.runState);
      if (!runStatesEqual(durableReadback, candidate)) {
        this.state = durableState;
        throw new Error("run-state persistence confirmation mismatch");
      }
      this.state = durableReadback;
      return result;
    });
  }

  private async loadForMutation(discardDamaged: boolean): Promise<{ runState: RunState }> {
    try {
      const data = await this.authoritativeLoadFn();
      this.damaged = undefined;
      return data;
    } catch (error) {
      if (!discardDamaged || !(error instanceof CorruptRunStateError) || !this.quarantineFn) throw error;
      await this.quarantineFn(error.path);
      this.damaged = undefined;
      return { runState: {} };
    }
  }

  private async enqueueStateOperation<T>(job: () => Promise<T>): Promise<T> {
    if (!this.mutationQueueKey) {
      const next = this.mutationQueue.catch(() => undefined).then(job);
      this.mutationQueue = next.catch(() => undefined);
      return next;
    }
    return enqueueStatePathMutation(this.mutationQueueKey, job);
  }
}

/**
 * True when a transient failure of the run that is now in flight (entry as left
 * by setRunning) would exhaust the retry budget. That run is the last chance,
 * so it should commit what it has rather than ask to be retried.
 */
export function isFinalTransientAttempt(entry: RunStateEntry): boolean {
  return priorFailureAttempts(entry) + 1 >= MAX_TRANSIENT_ATTEMPTS;
}

// Legacy running attempts already include the in-flight attempt.
function priorFailureAttempts(entry: RunStateEntry): number {
  return entry.failureAttempts ?? Math.max(0, entry.attempts - (entry.status === "running" ? 1 : 0));
}

// A crash or force-quit is transient; the retry cap still applies.
function recoveredRunningEntry(
  entry: RunStateEntry,
  message: string,
  lastAttempt = entry.lastAttempt,
): RunStateEntry {
  const failureAttempts = priorFailureAttempts(entry) + 1;
  const retriesExhausted = failureAttempts >= MAX_TRANSIENT_ATTEMPTS;
  return {
    ...entry,
    status: retriesExhausted ? "failed_permanent" : "failed_transient",
    failureAttempts,
    outcome: undefined,
    lastAttempt,
    error: retriesExhausted
      ? `retries exhausted after ${failureAttempts} attempts: ${message}`
      : message,
  };
}

function skippedEntry(prev: RunStateEntry, reason: string): RunStateEntry {
  return {
    ...prev,
    status: "skipped",
    outcome: undefined,
    lastAttempt: Date.now(),
    error: reason,
  };
}

function getRunStateEntry(runState: RunState, date: string): RunStateEntry {
  return (
    runState[date] ?? {
      status: "pending",
      lastAttempt: 0,
      attempts: 0,
    }
  );
}

function runStatesEqual(left: RunState, right: RunState): boolean {
  const leftDates = Object.keys(left).sort();
  const rightDates = Object.keys(right).sort();
  if (leftDates.length !== rightDates.length) return false;
  return leftDates.every((date, index) => {
    if (date !== rightDates[index]) return false;
    const leftEntry = left[date];
    const rightEntry = right[date];
    return (
      leftEntry?.status === rightEntry?.status &&
      leftEntry?.lastAttempt === rightEntry?.lastAttempt &&
      leftEntry?.attempts === rightEntry?.attempts &&
      leftEntry?.failureAttempts === rightEntry?.failureAttempts &&
      leftEntry?.outcome === rightEntry?.outcome &&
      leftEntry?.error === rightEntry?.error &&
      leftEntry?.papersWritten === rightEntry?.papersWritten &&
      sameRerunOrigin(leftEntry?.rerunOf, rightEntry?.rerunOf)
    );
  });
}

function sameRerunOrigin(
  left: RunStateEntry["rerunOf"],
  right: RunStateEntry["rerunOf"],
): boolean {
  const normalize = (origin: RunStateEntry["rerunOf"]) =>
    origin ? JSON.stringify(copyKnownEntryFields(origin)) : undefined;
  return normalize(left) === normalize(right);
}

function cloneRunState(runState: RunState): RunState {
  const out: RunState = {};
  for (const [date, entry] of Object.entries(runState)) {
    out[date] = { ...entry };
  }
  return out;
}

export function deriveStorageStateStorePaths(
  output: OutputSettings,
  normalizePath: (path: string) => string,
): StorageStateStorePaths {
  const indexDir = derivePaperInboxPaths(output, normalizePath).indexDir;
  return {
    indexDir,
    runStatePath: normalizePath(`${indexDir}/run-state.json`),
    runHistoryPath: normalizePath(`${indexDir}/run-history.jsonl`),
  };
}

export function createStorageStateStore(
  storage: StorageAdapter,
  output: OutputSettings,
  logger?: StateStoreLogger,
): StateStore {
  const paths = deriveStorageStateStorePaths(output, (path) =>
    storage.normalizePath(path),
  );
  return new StateStore(
    async () => {
      const loaded = await loadRunStateWithFallback(storage, paths.runStatePath, logger);
      const damaged = await findRunStateDamage(storage, paths.runStatePath);
      return damaged ? { ...loaded, damaged } : loaded;
    },
    async ({ runState }) => {
      await ensureDirDeep(storage, paths.indexDir);
      await writeAtomic(storage, paths.runStatePath, {
        schemaVersion: 1,
        updatedAt: new Date().toISOString(),
        runState,
      });
    },
    paths.runStatePath,
    async () => loadAuthoritativeRunState(storage, paths.runStatePath),
    async (damagedPath) => quarantineRunStateFile(storage, damagedPath),
  );
}

/** The lenient load papers over damage; the strict one that mutations use does not. */
async function findRunStateDamage(
  storage: StorageAdapter,
  path: string,
): Promise<RunStateDamage | undefined> {
  try {
    await loadAuthoritativeRunState(storage, path);
  } catch (error) {
    if (error instanceof CorruptRunStateError) return { path: error.path, reason: error.reason };
  }
  return undefined;
}

/** Keeps the damaged file for inspection instead of deleting it. */
async function quarantineRunStateFile(storage: StorageAdapter, path: string): Promise<void> {
  const stamp = new Date().toISOString().replace(/[-:.]/g, "");
  let target = `${path}.corrupt-${stamp}`;
  for (let n = 1; await storage.exists(target); n += 1) target = `${path}.corrupt-${stamp}-${n}`;
  await storage.rename(path, target);
}

async function loadAuthoritativeRunState(
  storage: StorageAdapter,
  path: string,
): Promise<{ runState: RunState }> {
  if (!(await storage.exists(path))) {
    // A crash between the two renames of writeAtomic leaves only the backup.
    // It is the last committed state, so mutations must build on it; an
    // unusable backup fails closed instead of silently starting from nothing.
    const backupPath = `${path}.bak`;
    if (await storage.exists(backupPath)) {
      return parseAuthoritativeRunState(await storage.readText(backupPath), backupPath);
    }
    return { runState: {} };
  }
  return parseAuthoritativeRunState(await storage.readText(path), path);
}

function parseAuthoritativeRunState(raw: string, path: string): { runState: RunState } {
  try {
    return parseRunState(raw, true);
  } catch (error) {
    // A newer schema or an unreadable optional field may be data worth keeping.
    if (
      error instanceof UnsupportedRunStateSchemaError ||
      error instanceof InvalidRunStateOptionalFieldError
    ) throw error;
    throw new CorruptRunStateError(path, error instanceof Error ? error.message : String(error));
  }
}

async function loadRunStateWithFallback(
  storage: StorageAdapter,
  path: string,
  logger?: StateStoreLogger,
): Promise<{ runState: RunState }> {
  const backupPath = `${path}.bak`;
  if (!(await storage.exists(path))) {
    if (await storage.exists(backupPath)) {
      try {
        return parseRunState(await storage.readText(backupPath));
      } catch (e) {
        if (
          e instanceof UnsupportedRunStateSchemaError ||
          e instanceof InvalidRunStateOptionalFieldError
        ) throw e;
        warnRunStateFallback(logger, `failed to read run-state backup, using empty state: ${backupPath}`, e);
      }
    }
    return { runState: {} };
  }
  try {
    return parseRunState(await storage.readText(path));
  } catch (e) {
    if (
      e instanceof UnsupportedRunStateSchemaError ||
      e instanceof InvalidRunStateOptionalFieldError
    ) throw e;
    warnRunStateFallback(logger, `failed to read run-state.json, trying backup: ${path}`, e);
  }

  if (await storage.exists(backupPath)) {
    try {
      return parseRunState(await storage.readText(backupPath));
    } catch (e) {
      if (
      e instanceof UnsupportedRunStateSchemaError ||
      e instanceof InvalidRunStateOptionalFieldError
    ) throw e;
      warnRunStateFallback(logger, `failed to read run-state backup, using empty state: ${backupPath}`, e);
    }
  }
  return { runState: {} };
}

function parseRunState(
  raw: string,
  strict = false,
): { runState: RunState } {
  const parsed: unknown = JSON.parse(raw);
  const isRecord = parsed && typeof parsed === "object" && !Array.isArray(parsed);
  if (
    isRecord &&
    "schemaVersion" in parsed &&
    parsed.schemaVersion !== 1
  ) {
    throw new UnsupportedRunStateSchemaError(parsed.schemaVersion);
  }
  const hasRunState = isRecord && "runState" in parsed;
  const rawStateValue = hasRunState ? parsed.runState : undefined;
  if (
    strict &&
    (!hasRunState ||
      !rawStateValue ||
      typeof rawStateValue !== "object" ||
      Array.isArray(rawStateValue))
  ) {
    throw new Error("invalid run-state root");
  }
  const rawState =
    rawStateValue &&
    typeof rawStateValue === "object" &&
    !Array.isArray(rawStateValue)
      ? (rawStateValue as Record<string, unknown>)
      : {};
  return {
    runState: parseRunStateEntries(rawState, strict),
  };
}

function parseRunStateEntries(
  rawState: Record<string, unknown>,
  strict = false,
): RunState {
  const out: RunState = {};
  if (!rawState || typeof rawState !== "object" || Array.isArray(rawState)) {
    return out;
  }
  for (const [date, rawEntry] of Object.entries(rawState)) {
    if (!rawEntry || typeof rawEntry !== "object" || Array.isArray(rawEntry)) {
      if (strict) throw new Error(`invalid run-state entry: ${date}`);
      continue;
    }
    const entry = rawEntry as Partial<RunStateEntry>;
    if (
      hasInvalidOptionalFields(entry) ||
      ("rerunOf" in entry && parseRerunOrigin(entry.rerunOf) === undefined)
    ) {
      throw new InvalidRunStateOptionalFieldError(date);
    }
    if (
      !isRunStatus(entry.status) ||
      typeof entry.lastAttempt !== "number" ||
      typeof entry.attempts !== "number"
    ) {
      if (strict) throw new Error(`invalid run-state entry: ${date}`);
      continue;
    }
    const rerunOf = "rerunOf" in entry ? parseRerunOrigin(entry.rerunOf) : undefined;
    out[date] = {
      ...copyKnownEntryFields(entry as RunStateEntry),
      ...(rerunOf ? { rerunOf } : {}),
    };
  }
  return out;
}

function hasInvalidOptionalFields(entry: Partial<RunStateEntry>): boolean {
  return (
    ("failureAttempts" in entry &&
      (typeof entry.failureAttempts !== "number" ||
        !Number.isSafeInteger(entry.failureAttempts) || entry.failureAttempts < 0)) ||
    ("outcome" in entry && !isRunOutcome(entry.outcome)) ||
    ("error" in entry && typeof entry.error !== "string") ||
    ("papersWritten" in entry &&
      (typeof entry.papersWritten !== "number" ||
        !Number.isFinite(entry.papersWritten)))
  );
}

function copyKnownEntryFields(entry: Omit<RunStateEntry, "rerunOf">): Omit<RunStateEntry, "rerunOf"> {
  return {
    status: entry.status,
    lastAttempt: entry.lastAttempt,
    attempts: entry.attempts,
    ...(entry.failureAttempts !== undefined ? { failureAttempts: entry.failureAttempts } : {}),
    ...(entry.outcome !== undefined ? { outcome: entry.outcome } : {}),
    error: typeof entry.error === "string" ? entry.error : undefined,
    papersWritten:
      typeof entry.papersWritten === "number" ? entry.papersWritten : undefined,
  };
}

/** The saved completed entry; undefined when the marker is not a valid one. */
function parseRerunOrigin(raw: unknown): Omit<RunStateEntry, "rerunOf"> | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const origin = raw as Partial<RunStateEntry>;
  if (
    origin.status !== "completed" ||
    typeof origin.lastAttempt !== "number" ||
    typeof origin.attempts !== "number" ||
    "rerunOf" in origin ||
    hasInvalidOptionalFields(origin)
  ) {
    return undefined;
  }
  return copyKnownEntryFields(origin as RunStateEntry);
}

function isRunStatus(value: unknown): value is RunStatus {
  return (
    value === "pending" ||
    value === "running" ||
    value === "completed" ||
    value === "failed_transient" ||
    value === "failed_permanent" ||
    value === "skipped"
  );
}

function warnRunStateFallback(
  logger: StateStoreLogger | undefined,
  message: string,
  error: unknown,
): void {
  if (logger) {
    logger.warn(message, error);
    return;
  }
  if (typeof console !== "undefined") {
    console.warn(`[arxiv-daily] ${message}`, error);
  }
}

function enqueueStatePathMutation<T>(
  key: string,
  job: () => Promise<T>,
): Promise<T> {
  const next = (stateMutationQueues.get(key) ?? Promise.resolve())
    .catch(() => undefined)
    .then(job);
  stateMutationQueues.set(key, next.catch(() => undefined));
  return next;
}

async function ensureDirDeep(
  storage: StorageAdapter,
  dir: string,
): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let cur = "";
  for (const part of parts) {
    cur = cur ? `${cur}/${part}` : part;
    if (!(await storage.exists(cur))) await storage.mkdir(cur);
  }
}

async function writeAtomic(
  storage: StorageAdapter,
  path: string,
  value: unknown,
): Promise<void> {
  const tmp = `${path}.tmp`;
  const bak = `${path}.bak`;
  const content = `${JSON.stringify(value, null, 2)}\n`;
  if (await storage.exists(tmp)) await storage.remove(tmp);
  await storage.writeText(tmp, content);
  if (!(await storage.exists(path))) {
    await storage.rename(tmp, path);
    return;
  }

  if (await storage.exists(bak)) await storage.remove(bak);
  await storage.rename(path, bak);
  try {
    await storage.rename(tmp, path);
    await storage.remove(bak);
  } catch (e) {
    if (await storage.exists(bak)) await storage.rename(bak, path);
    throw e;
  }
}

function isRunOutcome(value: unknown): value is RunOutcome {
  return value === "awaiting_announcement" || value === "no_updates" || value === "no_matches" || value === "papers_written";
}
