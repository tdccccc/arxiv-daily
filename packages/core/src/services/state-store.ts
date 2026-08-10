import type { RunState, RunStateEntry, RunStatus } from "../settings/types";
import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";

const MAX_TRANSIENT_ATTEMPTS = 10;
export const STALE_RUNNING_RECOVERY_MS = 60 * 60 * 1000;

export type StateLoadFn = () => Promise<{ runState: RunState }>;
export type StateSaveFn = (data: { runState: RunState }) => Promise<void>;
type StateStoreLogger = { warn(message: string, ...rest: unknown[]): void };

export interface StorageStateStorePaths {
  indexDir: string;
  runStatePath: string;
  runHistoryPath: string;
}

const stateMutationQueues = new Map<string, Promise<unknown>>();

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
  private mutationQueue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly loadFn: StateLoadFn,
    private readonly saveFn: StateSaveFn,
    private readonly mutationQueueKey?: string,
    private readonly authoritativeLoadFn: StateLoadFn = loadFn,
  ) {}

  async load(): Promise<void> {
    await this.enqueueStateOperation(async () => {
      const data = await this.loadFn();
      this.state = cloneRunState(data.runState);
    });
  }

  async loadAuthoritative(): Promise<void> {
    await this.enqueueStateOperation(async () => {
      const data = await this.authoritativeLoadFn();
      this.state = cloneRunState(data.runState);
    });
  }

  get(date: string): RunStateEntry {
    return (
      this.state[date] ?? {
        status: "pending" as RunStatus,
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
      const prev = getRunStateEntry(candidate, date);
      candidate[date] = {
        ...prev,
        status: "skipped",
        lastAttempt: Date.now(),
        error: reason,
      };
      return { result: undefined, persist: true };
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
      };
      return { result: undefined, persist: true };
    });
  }

  async setCompleted(date: string, papersWritten: number): Promise<void> {
    await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      candidate[date] = {
        ...prev,
        status: "completed",
        lastAttempt: Date.now(),
        papersWritten,
        error: undefined,
      };
      return { result: undefined, persist: true };
    });
  }

  async setPending(date: string, reason: string): Promise<void> {
    await this.enqueueMutation((candidate) => {
      const prev = getRunStateEntry(candidate, date);
      candidate[date] = {
        ...prev,
        status: "pending",
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
      const retriesExhausted =
        status === "failed_transient" && prev.attempts >= MAX_TRANSIENT_ATTEMPTS;
      if (retriesExhausted) status = "failed_permanent";
      const persistedMessage = retriesExhausted
        ? `retries exhausted after ${prev.attempts} attempts: ${message}`
        : message;
      candidate[date] = {
        ...prev,
        status,
        lastAttempt: Date.now(),
        error: persistedMessage,
      };
      return { result: status, persist: true };
    });
  }

  async clearDate(date: string): Promise<void> {
    await this.enqueueMutation((candidate) => {
      if (!(date in candidate)) return { result: undefined, persist: false };
      delete candidate[date];
      return { result: undefined, persist: true };
    });
  }

  async clearAll(): Promise<void> {
    await this.enqueueMutation((candidate) => {
      for (const date of Object.keys(candidate)) delete candidate[date];
      return { result: undefined, persist: true };
    });
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
        candidate[date] = {
          ...entry,
          status: "failed_permanent",
          lastAttempt: now,
          error: "recovered stale running state after startup",
        };
        recovered.push(date);
      }
      return {
        result: recovered.sort(),
        persist: recovered.length > 0,
      };
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
  ): Promise<T> {
    return this.enqueueStateOperation(async () => {
      const data = await this.authoritativeLoadFn();
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

  private async enqueueStateOperation<T>(job: () => Promise<T>): Promise<T> {
    if (!this.mutationQueueKey) {
      const next = this.mutationQueue.catch(() => undefined).then(job);
      this.mutationQueue = next.catch(() => undefined);
      return next;
    }
    return enqueueStatePathMutation(this.mutationQueueKey, job);
  }
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
      leftEntry?.error === rightEntry?.error &&
      leftEntry?.papersWritten === rightEntry?.papersWritten
    );
  });
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
      return loadRunStateWithFallback(storage, paths.runStatePath, logger);
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
  );
}

async function loadAuthoritativeRunState(
  storage: StorageAdapter,
  path: string,
): Promise<{ runState: RunState }> {
  if (!(await storage.exists(path))) return { runState: {} };
  return parseRunState(await storage.readText(path), true);
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
  const parsed = JSON.parse(raw);
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
    const optionalFieldsInvalid =
      ("error" in entry && typeof entry.error !== "string") ||
      ("papersWritten" in entry &&
        (typeof entry.papersWritten !== "number" ||
          !Number.isFinite(entry.papersWritten)));
    if (optionalFieldsInvalid) {
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
    out[date] = {
      status: entry.status,
      lastAttempt: entry.lastAttempt,
      attempts: entry.attempts,
      error: typeof entry.error === "string" ? entry.error : undefined,
      papersWritten:
        typeof entry.papersWritten === "number" ? entry.papersWritten : undefined,
    };
  }
  return out;
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
