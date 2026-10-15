import type { StorageAdapter } from "../core/adapters";
import type { FilterDirectionRef } from "../pipeline/paper-filter-contract";
import type { CheckpointGenerationIdentity } from "./daily-summary-checkpoint-store";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";

export const FILTER_DECISION_RECORD_SCHEMA_VERSION = 2 as const;
/** Records older than this many days (by report date) are pruned on each write. */
export const FILTER_DECISION_RETENTION_DAYS = 180;

const OUTCOMES = ["in_report", "cut_by_limit", "excluded_ignored", "skipped_by_model"] as const;

/**
 * What happened to a candidate after the model classified it: it reached the
 * report, fell below the daily paper limit, was dropped because the reader
 * had marked it ignored, or the model judged it irrelevant (`skip`).
 */
export type FilterDecisionOutcome = (typeof OUTCOMES)[number];

/** A direction as resolved from the model's reference, without the prompt-local `ref`. */
export interface FilterDecisionDirection {
  id: string;
  text: string;
  tag: string;
}

export interface FilterDecisionCandidate {
  id: string;
  title: string;
  /** Topic tag chosen by the model, or `skip`. */
  category: string;
  directions: FilterDecisionDirection[];
  relevanceScore: number;
  outcome: FilterDecisionOutcome;
  /** Whether the classifier was shown an abstract; false means it judged on the title alone. */
  hadAbstract: boolean;
}

export interface FilterDecisionRecordDocument {
  schemaVersion: typeof FILTER_DECISION_RECORD_SCHEMA_VERSION;
  reportDate: string;
  writtenAt: string;
  /** Every direction the classifier was offered, as configured when it ran. */
  directions: FilterDirectionRef[];
  promptContractVersion: number;
  resultContractVersion: number;
  /** Provider, model and mode only; the endpoint is a digest and no key is kept. */
  generation: CheckpointGenerationIdentity;
  candidates: FilterDecisionCandidate[];
}

export type FilterDecisionRecordInput = Omit<
  FilterDecisionRecordDocument,
  "schemaVersion" | "writtenAt"
>;

/** The part of the store the pipeline depends on. */
export interface FilterDecisionRecordPort {
  save(input: FilterDecisionRecordInput): Promise<unknown>;
}

export interface FilterDecisionRecordPaths {
  directory: string;
  documentPath: string;
}

export interface FilterDecisionRecordStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
  retentionDays?: number;
}

export class FilterDecisionRecordStoreError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "FilterDecisionRecordStoreError";
  }
}

const mutationQueues = new WeakMap<StorageAdapter, Map<string, Promise<unknown>>>();

function deriveFilterDecisionRecordDirectory(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
): string {
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  return storage.normalizePath(`${indexDir}/filter-decisions`);
}

export function deriveFilterDecisionRecordPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  reportDate: string,
): FilterDecisionRecordPaths {
  requireReportDate(reportDate);
  const directory = deriveFilterDecisionRecordDirectory(storage, output);
  return { directory, documentPath: storage.normalizePath(`${directory}/${reportDate}.json`) };
}

/**
 * Durable, compact record of every filter decision for one report date. Unlike
 * a filter checkpoint it outlives the committed report and is never read back
 * to skip a model call; it exists so filter quality can be evaluated later.
 */
export class FilterDecisionRecordStore implements FilterDecisionRecordPort {
  private readonly retentionDays: number;

  constructor(
    private readonly storage: StorageAdapter,
    private readonly output: OutputSettings,
    private readonly options: FilterDecisionRecordStoreOptions = {},
  ) {
    this.retentionDays = options.retentionDays ?? FILTER_DECISION_RETENTION_DAYS;
    if (!Number.isSafeInteger(this.retentionDays) || this.retentionDays < 1) {
      throw new FilterDecisionRecordStoreError("filter decision retention must be a positive integer of days");
    }
  }

  pathsFor(reportDate: string): FilterDecisionRecordPaths {
    return deriveFilterDecisionRecordPaths(this.storage, this.output, reportDate);
  }

  /** Corrupt, invalid or unreadable documents read as absent, with a warning. */
  async load(reportDate: string): Promise<FilterDecisionRecordDocument | null> {
    const { documentPath } = this.pathsFor(reportDate);
    let raw: string;
    try {
      if (!(await this.storage.exists(documentPath))) return null;
      raw = await this.storage.readText(documentPath);
    } catch (error) {
      this.warn(`unreadable filter decision record ignored: ${documentPath}`, error);
      return null;
    }
    try {
      const document = decodeDocument(JSON.parse(raw), reportDate);
      if (!document) {
        this.warn(`invalid filter decision record ignored: ${documentPath}`);
        return null;
      }
      return document;
    } catch (error) {
      this.warn(`corrupt filter decision record ignored: ${documentPath}`, error);
      return null;
    }
  }

  /** Report dates that have a record file, oldest first. Hosts without listing return none. */
  async listReportDates(): Promise<string[]> {
    if (!this.storage.list) return [];
    const directory = deriveFilterDecisionRecordDirectory(this.storage, this.output);
    try {
      if (!(await this.storage.exists(directory))) return [];
      const dates: string[] = [];
      for (const entry of await this.storage.list(directory)) {
        const date = /([^/]+)\.json$/.exec(entry.path)?.[1];
        if (entry.type === "file" && date && isReportDate(date)) dates.push(date);
      }
      return dates.sort();
    } catch (error) {
      this.warn(`cannot list filter decision records: ${directory}`, error);
      return [];
    }
  }

  /** Replaces the record of `input.reportDate`, then prunes records beyond the retention window. */
  async save(input: FilterDecisionRecordInput): Promise<FilterDecisionRecordDocument> {
    const paths = this.pathsFor(input.reportDate);
    const document: FilterDecisionRecordDocument = {
      schemaVersion: FILTER_DECISION_RECORD_SCHEMA_VERSION,
      ...clone(input),
      writtenAt: this.now().toISOString(),
    };
    if (!decodeDocument(document, input.reportDate)) {
      throw new FilterDecisionRecordStoreError("invalid filter decision record");
    }
    await this.enqueue(paths.documentPath, () => this.persist(paths, document));
    await this.prune(input.reportDate);
    return clone(document);
  }

  /** Removes records dated more than the retention window before `referenceDate`. */
  async prune(referenceDate: string): Promise<string[]> {
    const cutoff = retentionCutoff(referenceDate, this.retentionDays);
    const removed: string[] = [];
    for (const date of await this.listReportDates()) {
      if (date >= cutoff) continue;
      const { documentPath } = this.pathsFor(date);
      try {
        await this.enqueue(documentPath, () => removeIfExists(this.storage, documentPath));
        removed.push(date);
      } catch (error) {
        this.warn(`filter decision record prune failed: ${documentPath}`, error);
      }
    }
    return removed;
  }

  private async persist(
    paths: FilterDecisionRecordPaths,
    document: FilterDecisionRecordDocument,
  ): Promise<void> {
    const tmp = `${paths.documentPath}.tmp`;
    try {
      await ensureDirDeep(this.storage, paths.directory);
      await removeIfExists(this.storage, tmp);
      await writePrivateText(this.storage, tmp, `${JSON.stringify(document, null, 2)}\n`);
      let previous: string | null = null;
      if (await this.storage.exists(paths.documentPath)) {
        const raw = await this.storage.readText(paths.documentPath);
        if (decodeRawDocument(raw, document.reportDate)) previous = raw;
        await this.storage.remove(paths.documentPath);
      }
      try {
        await this.storage.rename(tmp, paths.documentPath);
      } catch (error) {
        // The target was removed to make room; put the last good record back.
        if (previous !== null) {
          await removeIfExists(this.storage, tmp);
          await writePrivateText(this.storage, tmp, previous);
          await this.storage.rename(tmp, paths.documentPath);
        }
        throw error;
      }
    } catch (error) {
      throw new FilterDecisionRecordStoreError(
        `failed to save filter decision record: ${paths.documentPath}`,
        error,
      );
    } finally {
      await removeIfExists(this.storage, tmp).catch(() => undefined);
    }
  }

  private enqueue<T>(path: string, operation: () => Promise<T>): Promise<T> {
    let queues = mutationQueues.get(this.storage);
    if (!queues) {
      queues = new Map();
      mutationQueues.set(this.storage, queues);
    }
    const next = (queues.get(path) ?? Promise.resolve()).catch(() => undefined).then(operation);
    const tail = next.then(() => undefined, () => undefined);
    queues.set(path, tail);
    void tail.finally(() => {
      if (queues?.get(path) === tail) queues.delete(path);
    });
    return next;
  }

  private now(): Date {
    return this.options.now?.() ?? new Date();
  }

  private warn(message: string, error?: unknown): void {
    this.options.onWarning?.(message, error);
  }
}

function decodeRawDocument(raw: string, reportDate: string): FilterDecisionRecordDocument | null {
  try {
    return decodeDocument(JSON.parse(raw), reportDate);
  } catch {
    return null;
  }
}

function decodeDocument(value: unknown, reportDate: string): FilterDecisionRecordDocument | null {
  if (!isExactObject(value, [
    "schemaVersion", "reportDate", "writtenAt", "directions",
    "promptContractVersion", "resultContractVersion", "generation", "candidates",
  ])) return null;
  if (
    value.schemaVersion !== FILTER_DECISION_RECORD_SCHEMA_VERSION ||
    value.reportDate !== reportDate ||
    typeof value.writtenAt !== "string" || !Number.isFinite(Date.parse(value.writtenAt)) ||
    !Number.isSafeInteger(value.promptContractVersion) ||
    !Number.isSafeInteger(value.resultContractVersion) ||
    !isDirectionRefs(value.directions) ||
    !isGeneration(value.generation) ||
    !Array.isArray(value.candidates) || !value.candidates.every(isCandidate)
  ) return null;
  return clone(value) as unknown as FilterDecisionRecordDocument;
}

function isDirectionRefs(value: unknown): value is FilterDirectionRef[] {
  return Array.isArray(value) && value.every((item) =>
    isExactObject(item, ["ref", "tag", "id", "text"]) &&
    typeof item.ref === "string" && typeof item.tag === "string" &&
    typeof item.id === "string" && typeof item.text === "string");
}

/** Looser than a checkpoint's identity check: an old record must stay readable after the request contract moves on. */
function isGeneration(value: unknown): value is CheckpointGenerationIdentity {
  return isExactObject(value, ["requestContractVersion", "provider", "endpointDigest", "model", "mode"]) &&
    Number.isSafeInteger(value.requestContractVersion) &&
    typeof value.provider === "string" && typeof value.model === "string" &&
    typeof value.endpointDigest === "string" && /^sha256:[0-9a-f]{64}$/.test(value.endpointDigest) &&
    isPlainObject(value.mode) && typeof value.mode.kind === "string";
}

function isCandidate(value: unknown): value is FilterDecisionCandidate {
  return isExactObject(value, ["id", "title", "category", "directions", "relevanceScore", "outcome", "hadAbstract"]) &&
    typeof value.id === "string" && typeof value.title === "string" &&
    typeof value.category === "string" && typeof value.hadAbstract === "boolean" &&
    typeof value.relevanceScore === "number" && Number.isFinite(value.relevanceScore) &&
    typeof value.outcome === "string" && (OUTCOMES as readonly string[]).includes(value.outcome) &&
    Array.isArray(value.directions) && value.directions.every((direction) =>
      isExactObject(direction, ["id", "text", "tag"]) &&
      typeof direction.id === "string" && typeof direction.text === "string" &&
      typeof direction.tag === "string");
}

function retentionCutoff(referenceDate: string, days: number): string {
  requireReportDate(referenceDate);
  const [year, month, day] = referenceDate.split("-").map(Number);
  const cutoff = Date.UTC(year!, month! - 1, day!) - days * 86_400_000;
  return new Date(cutoff).toISOString().slice(0, 10);
}

function isReportDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  const [year, month, day] = [Number(match[1]), Number(match[2]), Number(match[3])];
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day;
}

function requireReportDate(value: string): void {
  if (!isReportDate(value)) {
    throw new FilterDecisionRecordStoreError(`invalid filter decision report date: ${value}`);
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isExactObject(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

async function writePrivateText(
  storage: StorageAdapter,
  path: string,
  content: string,
): Promise<void> {
  if (storage.writeTextWithMode) {
    await storage.writeTextWithMode(path, content, 0o600);
    return;
  }
  await storage.writeText(path, content);
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}

async function removeIfExists(storage: StorageAdapter, path: string): Promise<void> {
  if (await storage.exists(path)) await storage.remove(path);
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
