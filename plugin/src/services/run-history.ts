import type { PipelineResult } from "../pipeline/pipeline";
import type { OutputSettings, RunStatus } from "../settings/types";
import type { StorageAdapter } from "../core/adapters";
import { deriveStorageStateStorePaths } from "./state-store";
import type { Logger } from "./logger";

export type RunHistoryEvent =
  | "started"
  | "completed"
  | "pending"
  | "failed"
  | "skipped";

export type RunHistoryTrigger =
  | "scheduler"
  | "manual"
  | "calendar"
  | "force"
  | "retry"
  | "run-all-pending";

export interface RunHistoryRecord {
  schemaVersion: 1;
  at: string;
  date: string;
  event: RunHistoryEvent;
  trigger: RunHistoryTrigger;
  status?: RunStatus;
  resultKind?: PipelineResult["kind"] | "skipped";
  papersWritten?: number;
  requestedPapersWritten?: number;
  preservedPapersWritten?: boolean;
  reason?: string;
  errorMessage?: string;
  dailyPath?: string;
  attempts?: number;
}

export interface RunHistoryStorePaths {
  indexDir: string;
  runHistoryPath: string;
}

interface RunHistoryPersistence {
  append(record: RunHistoryRecord): Promise<void>;
  readLatest(limit: number): Promise<RunHistoryRecord[]>;
}

export interface RunHistoryStorageOptions {
  maxBytes?: number;
  maxRotations?: number;
}

const DEFAULT_RUN_HISTORY_MAX_BYTES = 512 * 1024;
const DEFAULT_RUN_HISTORY_MAX_ROTATIONS = 3;

export function deriveRunHistoryStorePaths(
  output: OutputSettings,
  normalizePath: (path: string) => string,
): RunHistoryStorePaths {
  const paths = deriveStorageStateStorePaths(output, normalizePath);
  return {
    indexDir: paths.indexDir,
    runHistoryPath: paths.runHistoryPath,
  };
}

export class RunHistoryStore {
  private appendQueue: Promise<void> = Promise.resolve();

  constructor(
    private readonly persistence: RunHistoryPersistence,
    private readonly logger?: Pick<Logger, "warn">,
  ) {}

  static fromStorage(
    storage: StorageAdapter,
    output: OutputSettings,
    logger?: Pick<Logger, "warn">,
    options: RunHistoryStorageOptions = {},
  ): RunHistoryStore {
    const paths = deriveRunHistoryStorePaths(output, (path) =>
      storage.normalizePath(path),
    );
    const maxBytes = options.maxBytes ?? DEFAULT_RUN_HISTORY_MAX_BYTES;
    const maxRotations =
      options.maxRotations ?? DEFAULT_RUN_HISTORY_MAX_ROTATIONS;
    return new RunHistoryStore(
      {
        append: async (record) => {
          await ensureDirDeep(storage, paths.indexDir);
          let current = (await storage.exists(paths.runHistoryPath))
            ? await storage.readText(paths.runHistoryPath)
            : "";
          const prefix = current.length > 0 && !current.endsWith("\n") ? "\n" : "";
          let addition = `${prefix}${JSON.stringify(record)}\n`;
          if (
            current.length > 0 &&
            byteLength(current) + byteLength(addition) > maxBytes
          ) {
            await rotateHistoryFiles(
              storage,
              paths.runHistoryPath,
              maxRotations,
            );
            current = "";
            addition = `${JSON.stringify(record)}\n`;
          }
          if (storage.appendText) {
            await storage.appendText(paths.runHistoryPath, addition);
          } else {
            await storage.writeText(
              paths.runHistoryPath,
              `${current}${addition}`,
            );
          }
        },
        readLatest: async (limit) => {
          const raw = await readHistoryFiles(
            storage,
            paths.runHistoryPath,
            maxRotations,
          );
          return decodeRunHistoryLines(raw.join("\n"), logger)
            .sort((a, b) => b.at.localeCompare(a.at))
            .slice(0, Math.max(0, limit));
        },
      },
      logger,
    );
  }

  append(record: RunHistoryRecord): Promise<void> {
    const next = this.appendQueue
      .catch(() => undefined)
      .then(() => this.persistence.append(record));
    this.appendQueue = next.catch(() => undefined);
    return next;
  }

  async safeAppend(record: RunHistoryRecord): Promise<void> {
    try {
      await this.append(record);
    } catch (e) {
      this.logger?.warn("run history append failed", e);
    }
  }

  readLatest(limit = 50): Promise<RunHistoryRecord[]> {
    return this.persistence.readLatest(limit);
  }
}

export function formatRunHistoryRecords(records: RunHistoryRecord[]): string {
  if (records.length === 0) return "No run history yet.";
  return records.map(formatRunHistoryRecord).join("\n");
}

function formatRunHistoryRecord(record: RunHistoryRecord): string {
  const parts = [
    record.at,
    record.date,
    record.event,
    `trigger=${record.trigger}`,
  ];
  if (record.status) parts.push(`status=${record.status}`);
  if (record.resultKind) parts.push(`result=${record.resultKind}`);
  if (record.papersWritten != null) parts.push(`papers=${record.papersWritten}`);
  if (record.requestedPapersWritten != null) {
    parts.push(`requestedPapers=${record.requestedPapersWritten}`);
  }
  if (record.preservedPapersWritten) parts.push("preservedPapers=true");
  if (record.attempts != null) parts.push(`attempts=${record.attempts}`);
  if (record.reason) parts.push(`reason=${record.reason}`);
  if (record.errorMessage) parts.push(`error=${record.errorMessage}`);
  if (record.dailyPath) parts.push(`dailyPath=${record.dailyPath}`);
  return parts.join(" | ");
}

function decodeRunHistoryLines(
  raw: string,
  logger?: Pick<Logger, "warn">,
): RunHistoryRecord[] {
  const out: RunHistoryRecord[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (isRunHistoryRecord(parsed)) out.push(parsed);
    } catch (e) {
      logger?.warn(
        "run history: skipped malformed line",
        truncateHistoryLine(trimmed),
        e,
      );
    }
  }
  return out;
}

function truncateHistoryLine(line: string): string {
  return line.length > 200 ? `${line.slice(0, 200)}...` : line;
}

function isRunHistoryRecord(value: unknown): value is RunHistoryRecord {
  const candidate = value as Partial<RunHistoryRecord> | null;
  return (
    Boolean(candidate) &&
    candidate?.schemaVersion === 1 &&
    typeof candidate.at === "string" &&
    typeof candidate.date === "string" &&
    isRunHistoryEvent(candidate.event) &&
    isRunHistoryTrigger(candidate.trigger)
  );
}

function isRunHistoryEvent(value: unknown): value is RunHistoryEvent {
  return (
    value === "started" ||
    value === "completed" ||
    value === "pending" ||
    value === "failed" ||
    value === "skipped"
  );
}

function isRunHistoryTrigger(value: unknown): value is RunHistoryTrigger {
  return (
    value === "scheduler" ||
    value === "manual" ||
    value === "calendar" ||
    value === "force" ||
    value === "retry" ||
    value === "run-all-pending"
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

async function readHistoryFiles(
  storage: StorageAdapter,
  path: string,
  maxRotations: number,
): Promise<string[]> {
  const raw: string[] = [];
  if (await storage.exists(path)) raw.push(await storage.readText(path));
  for (let i = 1; i <= maxRotations; i += 1) {
    const rotated = `${path}.${i}`;
    if (await storage.exists(rotated)) raw.push(await storage.readText(rotated));
  }
  return raw;
}

async function rotateHistoryFiles(
  storage: StorageAdapter,
  path: string,
  maxRotations: number,
): Promise<void> {
  if (maxRotations <= 0) {
    await storage.remove(path);
    return;
  }
  await storage.remove(`${path}.${maxRotations}`);
  for (let i = maxRotations - 1; i >= 1; i -= 1) {
    const from = `${path}.${i}`;
    const to = `${path}.${i + 1}`;
    if (await storage.exists(from)) {
      await storage.remove(to);
      await storage.rename(from, to);
    }
  }
  if (await storage.exists(path)) {
    await storage.remove(`${path}.1`);
    await storage.rename(path, `${path}.1`);
  }
}

function byteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}
