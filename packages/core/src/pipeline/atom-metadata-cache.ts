import type { StorageAdapter } from "../core/adapters";
import type { AtomPaperMeta } from "./atom-parser";
import { modernArxivResources } from "../utils/arxiv";
import { isCancellationError, throwIfCancelled } from "../services/cancellation";

export interface AtomMetadataCacheOptions {
  rootDir: string;
  expiryDays: number;
  storage: StorageAdapter;
  now?: () => Date;
  operationLeaseMs?: number;
}

interface AtomMetadataCacheEnvelope {
  schemaVersion: 1;
  cachedAt: string;
  paper: AtomPaperMeta;
}

const DEFAULT_CACHE_OPERATION_LEASE_MS = 30_000;
const MIN_CACHE_OPERATION_LEASE_MS = 1;
let cacheOperationQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

export class AtomMetadataCache {
  private readonly queueRoot: string;

  constructor(private opts: AtomMetadataCacheOptions) {
    this.queueRoot = opts.storage.normalizePath(opts.rootDir);
  }

  async get(id: string, signal?: AbortSignal): Promise<AtomPaperMeta | null> {
    try {
      return await this.serialize(() => this.getUnlocked(id), signal);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      return null;
    }
  }

  set(id: string, paper: AtomPaperMeta, signal?: AbortSignal): Promise<void> {
    return this.serialize(() => this.setUnlocked(id, paper), signal);
  }

  async cleanupExpired(signal?: AbortSignal): Promise<number> {
    try {
      return await this.serialize(() => this.cleanupExpiredUnlocked(), signal);
    } catch (error) {
      if (isCancellationError(error)) throw error;
      return 0;
    }
  }

  private serialize<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    return serializeCacheOperation(
      this.opts.storage,
      this.queueRoot,
      operation,
      boundedLeaseMs(this.opts.operationLeaseMs),
      signal,
    );
  }

  private async getUnlocked(id: string): Promise<AtomPaperMeta | null> {
    const canonicalId = canonicalArxivId(id);
    if (!canonicalId) return null;
    const path = this.pathFor(canonicalId);
    try {
      if (!(await this.opts.storage.exists(path))) return null;
      const envelope = parseEnvelope(await this.opts.storage.readText(path));
      if (
        !envelope ||
        envelope.paper.id !== canonicalId ||
        isExpired(envelope.cachedAt, this.opts.expiryDays, this.now())
      ) {
        await this.removeBestEffort(path);
        return null;
      }
      return envelope.paper;
    } catch {
      return null;
    }
  }

  private async setUnlocked(id: string, paper: AtomPaperMeta): Promise<void> {
    const canonicalId = canonicalArxivId(id);
    if (!canonicalId || paper.id !== canonicalId || !isAtomPaperMeta(paper)) {
      throw new Error(`invalid Atom metadata cache entry for ${id}`);
    }
    const path = this.pathFor(canonicalId);
    await ensureDirDeep(this.opts.storage, parentDir(path));
    const envelope: AtomMetadataCacheEnvelope = {
      schemaVersion: 1,
      cachedAt: this.now().toISOString(),
      paper,
    };
    const content = `${JSON.stringify(envelope)}\n`;
    if (this.opts.storage.writeTextAtomic) {
      await this.opts.storage.writeTextAtomic(path, content);
    } else {
      await this.opts.storage.writeText(path, content);
    }
  }

  private async cleanupExpiredUnlocked(): Promise<number> {
    const storage = this.opts.storage;
    if (!storage.list) return 0;
    const dir = this.cacheDir();
    try {
      if (!(await storage.exists(dir))) return 0;
      let removed = 0;
      for (const entry of await storage.list(dir)) {
        if (entry.type !== "file" || !entry.path.endsWith(".json")) continue;
        try {
          const envelope = parseEnvelope(await storage.readText(entry.path));
          const filenameId = decodeFilenameId(entry.path);
          if (
            !envelope ||
            !filenameId ||
            envelope.paper.id !== filenameId ||
            isExpired(envelope.cachedAt, this.opts.expiryDays, this.now())
          ) {
            await storage.remove(entry.path).catch(() => {});
            removed += 1;
          }
        } catch {
          await storage.remove(entry.path).catch(() => {});
          removed += 1;
        }
      }
      return removed;
    } catch {
      return 0;
    }
  }

  private pathFor(canonicalId: string): string {
    return this.opts.storage.normalizePath(
      `${this.cacheDir()}/${encodeURIComponent(canonicalId)}.json`,
    );
  }

  private cacheDir(): string {
    return this.opts.storage.normalizePath(`${this.opts.rootDir}/atom-metadata`);
  }

  private now(): Date {
    return this.opts.now?.() ?? new Date();
  }

  private async removeBestEffort(path: string): Promise<void> {
    await this.opts.storage.remove(path).catch(() => {});
  }
}

export function isAtomPaperMeta(value: unknown): value is AtomPaperMeta {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const paper = value as Partial<AtomPaperMeta>;
  const canonicalId = typeof paper.id === "string" ? canonicalArxivId(paper.id) : null;
  return (
    canonicalId === paper.id &&
    isNonEmptyString(paper.title) &&
    isNonEmptyString(paper.authors) &&
    Array.isArray(paper.authorNames) &&
    paper.authorNames.length > 0 &&
    paper.authorNames.every(isNonEmptyString) &&
    isNonEmptyString(paper.abstract) &&
    isNonEmptyString(paper.published) &&
    isNonEmptyString(paper.updated) &&
    isNonEmptyString(paper.primaryCategory) &&
    Array.isArray(paper.categories) &&
    paper.categories.length > 0 &&
    paper.categories.every(isNonEmptyString) &&
    paper.categories.includes(paper.primaryCategory)
  );
}

function parseEnvelope(raw: string): AtomMetadataCacheEnvelope | null {
  try {
    const parsed = JSON.parse(raw) as Partial<AtomMetadataCacheEnvelope>;
    if (
      parsed.schemaVersion !== 1 ||
      typeof parsed.cachedAt !== "string" ||
      !Number.isFinite(Date.parse(parsed.cachedAt)) ||
      !isAtomPaperMeta(parsed.paper)
    ) return null;
    return parsed as AtomMetadataCacheEnvelope;
  } catch {
    return null;
  }
}

function canonicalArxivId(id: string): string | null {
  return modernArxivResources(id)?.id ?? null;
}

function isExpired(cachedAt: string, expiryDays: number, now: Date): boolean {
  const timestamp = Date.parse(cachedAt);
  const nowMs = now.getTime();
  const ttlMs = expiryDays * 86_400_000;
  if (
    !Number.isFinite(timestamp) ||
    !Number.isFinite(nowMs) ||
    !Number.isFinite(expiryDays) ||
    expiryDays < 0 ||
    !Number.isFinite(ttlMs) ||
    timestamp > nowMs
  ) return true;
  return nowMs - timestamp > ttlMs;
}

function serializeCacheOperation<T>(
  storage: StorageAdapter,
  root: string,
  operation: () => Promise<T>,
  leaseMs: number,
  signal?: AbortSignal,
): Promise<T> {
  throwIfCancelled(signal);
  let queues = cacheOperationQueues.get(storage);
  if (!queues) {
    queues = new Map();
    cacheOperationQueues.set(storage, queues);
  }
  const predecessor = queues.get(root) ?? Promise.resolve();
  const scheduled = predecessor.catch(() => undefined).then(async () => {
    throwIfCancelled(signal);
    let physical: Promise<T>;
    try {
      physical = Promise.resolve(operation());
    } catch (error) {
      physical = Promise.reject(error);
    }
    // The lease recovers the logical queue. An adapter that ignores cancellation
    // can still settle later and physically overlap a newer operation; consuming
    // that settlement prevents rejection leaks but cannot cancel unsupported I/O.
    void physical.catch(() => undefined);
    return operationLease(physical, leaseMs, signal);
  });
  const result = abortablePromise(scheduled, signal);
  const queueTail = scheduled.then(() => undefined, () => undefined);
  queues.set(root, queueTail);
  void queueTail.finally(() => {
    if (queues?.get(root) === queueTail) queues.delete(root);
  });
  return result;
}

function operationLease<T>(
  physical: Promise<T>,
  leaseMs: number,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
    };
    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onAbort = () => settle(() => reject(cancellationError(signal)));
    const timeout = setTimeout(
      () => settle(() => reject(new Error(
        `Atom metadata cache operation lease expired after ${leaseMs}ms`,
      ))),
      leaseMs,
    );
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
    physical.then(
      (value) => settle(() => resolve(value)),
      (error) => settle(() => reject(error)),
    );
  });
}

function abortablePromise<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  throwIfCancelled(signal);
  if (!signal) return promise;
  return new Promise<T>((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      reject(cancellationError(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) return onAbort();
    promise.then(
      (value) => { cleanup(); resolve(value); },
      (error) => { cleanup(); reject(error); },
    );
  });
}

function cancellationError(signal?: AbortSignal): unknown {
  try {
    throwIfCancelled(signal);
  } catch (error) {
    return error;
  }
  return new Error("cancelled by user");
}

function boundedLeaseMs(value: number | undefined): number {
  if (value == null || !Number.isFinite(value)) return DEFAULT_CACHE_OPERATION_LEASE_MS;
  return Math.max(MIN_CACHE_OPERATION_LEASE_MS, value);
}

/** Reset shared cache operation state between tests. Not intended for runtime use. */
export function resetAtomMetadataCacheForTests(): Promise<void> {
  cacheOperationQueues = new WeakMap();
  return Promise.resolve();
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function decodeFilenameId(path: string): string | null {
  const filename = path.split("/").pop();
  if (!filename?.endsWith(".json")) return null;
  try {
    const id = decodeURIComponent(filename.slice(0, -5));
    return canonicalArxivId(id) === id ? id : null;
  } catch {
    return null;
  }
}

function parentDir(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.length <= 1 ? "" : parts.slice(0, -1).join("/");
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  if (!dir) return;
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}
