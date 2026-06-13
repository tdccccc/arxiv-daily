import type { RunState, RunStateEntry, RunStatus } from "../settings/types";
import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";

const MAX_TRANSIENT_ATTEMPTS = 10;

export type StateLoadFn = () => Promise<{ runState: RunState }>;
export type StateSaveFn = (data: { runState: RunState }) => Promise<void>;

export interface StorageStateStorePaths {
  indexDir: string;
  runStatePath: string;
}

export class StateStore {
  private state: RunState = {};

  constructor(
    private readonly loadFn: StateLoadFn,
    private readonly saveFn: StateSaveFn,
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
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "skipped",
      lastAttempt: Date.now(),
      error: reason,
    };
    await this.saveFn({ runState: this.state });
  }

  async setRunning(date: string): Promise<void> {
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "running",
      lastAttempt: Date.now(),
      attempts: prev.attempts + 1,
    };
    await this.saveFn({ runState: this.state });
  }

  async setCompleted(date: string, papersWritten: number): Promise<void> {
    const prev = this.get(date);
    this.state[date] = {
      ...prev,
      status: "completed",
      lastAttempt: Date.now(),
      papersWritten,
      error: undefined,
    };
    await this.saveFn({ runState: this.state });
  }

  async setFailed(
    date: string,
    kind: "transient" | "permanent",
    message: string,
  ): Promise<void> {
    const prev = this.get(date);
    let status: RunStatus = kind === "permanent" ? "failed_permanent" : "failed_transient";
    if (status === "failed_transient" && prev.attempts >= MAX_TRANSIENT_ATTEMPTS) {
      status = "failed_permanent";
    }
    this.state[date] = {
      ...prev,
      status,
      lastAttempt: Date.now(),
      error: message,
    };
    await this.saveFn({ runState: this.state });
  }

  async clearDate(date: string): Promise<void> {
    if (!(date in this.state)) return;
    delete this.state[date];
    await this.saveFn({ runState: this.state });
  }

  async clearAll(): Promise<void> {
    this.state = {};
    await this.saveFn({ runState: this.state });
  }

  async replaceAll(runState: RunState): Promise<void> {
    this.state = cloneRunState(runState);
    await this.saveFn({ runState: this.state });
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
  };
}

export function createStorageStateStore(
  storage: StorageAdapter,
  output: OutputSettings,
): StateStore {
  const paths = deriveStorageStateStorePaths(output, (path) =>
    storage.normalizePath(path),
  );
  return new StateStore(
    async () => {
      if (!(await storage.exists(paths.runStatePath))) return { runState: {} };
      const raw = await storage.readText(paths.runStatePath);
      const parsed = JSON.parse(raw);
      return {
        runState:
          parsed && typeof parsed === "object" && parsed.runState
            ? (parsed.runState as RunState)
            : {},
      };
    },
    async ({ runState }) => {
      await ensureDirDeep(storage, paths.indexDir);
      await writeAtomic(storage, paths.runStatePath, {
        schemaVersion: 1,
        updatedAt: new Date().toISOString(),
        runState,
      });
    },
  );
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
  const content = `${JSON.stringify(value, null, 2)}\n`;
  await storage.writeText(tmp, content);
  await storage.rename(tmp, path);
}
