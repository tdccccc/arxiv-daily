import type { RunState, RunStateEntry, RunStatus } from "../settings/types";
import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";

const MAX_TRANSIENT_ATTEMPTS = 10;

export type StateLoadFn = () => Promise<{ runState: RunState }>;
export type StateSaveFn = (data: { runState: RunState }) => Promise<void>;
type StateStoreLogger = { warn(message: string, ...rest: unknown[]): void };

export interface StorageStateStorePaths {
  indexDir: string;
  runStatePath: string;
  runHistoryPath: string;
}

const stateMutationQueues = new Map<string, Promise<unknown>>();

export class StateStore {
  private state: RunState = {};
  private mutationQueue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly loadFn: StateLoadFn,
    private readonly saveFn: StateSaveFn,
    private readonly mutationQueueKey?: string,
  ) {}

  async load(): Promise<void> {
    const data = await this.loadFn();
    this.state = data?.runState ?? {};
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
    await this.enqueueMutation(async () => {
      const prev = this.get(date);
      this.state[date] = {
        ...prev,
        status: "skipped",
        lastAttempt: Date.now(),
        error: reason,
      };
      await this.saveFn({ runState: this.state });
    });
  }

  async setRunning(date: string): Promise<void> {
    await this.enqueueMutation(async () => {
      const prev = this.get(date);
      const att = prev.attempts + 1;
      if (typeof console !== "undefined") {
        console.log(`[arxiv-daily] state: ${date} → running (attempt ${att})`);
      }
      this.state[date] = {
        ...prev,
        status: "running",
        lastAttempt: Date.now(),
        attempts: att,
      };
      await this.saveFn({ runState: this.state });
    });
  }

  async setCompleted(date: string, papersWritten: number): Promise<void> {
    await this.enqueueMutation(async () => {
      const prev = this.get(date);
      if (typeof console !== "undefined") {
        console.log(`[arxiv-daily] state: ${date} → completed (${papersWritten} papers)`);
      }
      this.state[date] = {
        ...prev,
        status: "completed",
        lastAttempt: Date.now(),
        papersWritten,
        error: undefined,
      };
      await this.saveFn({ runState: this.state });
    });
  }

  async setFailed(
    date: string,
    kind: "transient" | "permanent",
    message: string,
  ): Promise<void> {
    await this.enqueueMutation(async () => {
      const prev = this.get(date);
      let status: RunStatus = kind === "permanent" ? "failed_permanent" : "failed_transient";
      if (status === "failed_transient" && prev.attempts >= MAX_TRANSIENT_ATTEMPTS) {
        status = "failed_permanent";
      }
      if (typeof console !== "undefined") {
        console.log(`[arxiv-daily] state: ${date} → ${status}: ${message}`);
      }
      this.state[date] = {
        ...prev,
        status,
        lastAttempt: Date.now(),
        error: message,
      };
      await this.saveFn({ runState: this.state });
    });
  }

  async clearDate(date: string): Promise<void> {
    await this.enqueueMutation(async () => {
      if (!(date in this.state)) return;
      delete this.state[date];
      await this.saveFn({ runState: this.state });
    });
  }

  async clearAll(): Promise<void> {
    await this.enqueueMutation(async () => {
      this.state = {};
      await this.saveFn({ runState: this.state });
    });
  }

  async replaceAll(runState: RunState): Promise<void> {
    await this.enqueueMutation(async () => {
      this.state = cloneRunState(runState);
      await this.saveFn({ runState: this.state });
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

  private async enqueueMutation<T>(job: () => Promise<T>): Promise<T> {
    const run = async () => {
      const data = await this.loadFn();
      this.state = cloneRunState(data?.runState ?? {});
      return job();
    };
    if (!this.mutationQueueKey) {
      const next = this.mutationQueue.catch(() => undefined).then(run);
      this.mutationQueue = next.catch(() => undefined);
      return next;
    }
    return enqueueStatePathMutation(this.mutationQueueKey, run);
  }
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
  );
}

async function loadRunStateWithFallback(
  storage: StorageAdapter,
  path: string,
  logger?: StateStoreLogger,
): Promise<{ runState: RunState }> {
  if (!(await storage.exists(path))) return { runState: {} };
  try {
    return parseRunState(await storage.readText(path));
  } catch (e) {
    warnRunStateFallback(logger, `failed to read run-state.json, trying backup: ${path}`, e);
  }

  const backupPath = `${path}.bak`;
  if (await storage.exists(backupPath)) {
    try {
      return parseRunState(await storage.readText(backupPath));
    } catch (e) {
      warnRunStateFallback(logger, `failed to read run-state backup, using empty state: ${backupPath}`, e);
    }
  }
  return { runState: {} };
}

function parseRunState(raw: string): { runState: RunState } {
  const parsed = JSON.parse(raw);
  return {
    runState:
      parsed && typeof parsed === "object" && parsed.runState
        ? (parsed.runState as RunState)
        : {},
  };
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
