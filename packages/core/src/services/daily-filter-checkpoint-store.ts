import type { StorageAdapter } from "../core/adapters";
import { LLM_REQUEST_CONTRACT_VERSION, type ChatMessage } from "../llm/client";
import type { CheckpointGenerationIdentity } from "./daily-summary-checkpoint-store";
import {
  DAILY_FILTER_FINGERPRINT_VERSION,
  DAILY_FILTER_PROMPT_CONTRACT_VERSION,
  DAILY_FILTER_RESULT_CONTRACT_VERSION,
  decodePaperFilterRecords,
  isPreparedDailyFilterCheckpoint,
  prepareDailyFilterCheckpoint,
  type DailyFilterCheckpointCompatibilityInput,
  type DailyFilterCheckpointFingerprintInput,
  type FilterDirectionRef,
  type FilterRecord,
  type PreparedDailyFilterCheckpoint,
} from "../pipeline/paper-filter-contract";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "./paper-index";
import { sha256ForCheckpointTests } from "./daily-summary-checkpoint-store";

export const DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION = 1 as const;
export {
  DAILY_FILTER_FINGERPRINT_VERSION,
  DAILY_FILTER_PROMPT_CONTRACT_VERSION,
  DAILY_FILTER_RESULT_CONTRACT_VERSION,
  prepareDailyFilterCheckpoint,
  type DailyFilterCheckpointCompatibilityInput,
  type DailyFilterCheckpointFingerprintInput,
  type PreparedDailyFilterCheckpoint,
} from "../pipeline/paper-filter-contract";
export interface DailyFilterCheckpointDocument {
  schemaVersion: typeof DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION;
  reportDate: string;
  fingerprint: string;
  fingerprintInput: DailyFilterCheckpointFingerprintInput;
  completedAt: string;
  result: FilterRecord[];
}

export interface DailyFilterCheckpointPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
  /**
   * The retired library-profile classifier wrote its own checkpoint beside the
   * filter's. Nothing reads or writes it any more, but `removeAll` still has to
   * clean up the files it left on disk, so the paths stay.
   */
  personalizedDocumentPath: string;
  personalizedBackupPath: string;
  /**
   * The withdrawn library novelty stage (ADR 0015) also kept a checkpoint here
   * during an in-flight run. Nothing reads or writes it any more; a run that
   * crashed before commit can leave the files behind, so `removeAll` still
   * deletes them.
   */
  legacyNoveltyDocumentPath: string;
  legacyNoveltyBackupPath: string;
}

export interface DailyFilterCheckpointStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export class DailyFilterCheckpointStoreError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "DailyFilterCheckpointStoreError";
  }
}

/** Batches probed on adapters that cannot list a directory. */
const FILTER_CHECKPOINT_UNLISTED_BATCH_SCAN = 64;

const mutationQueues = new WeakMap<StorageAdapter, Map<string, Promise<unknown>>>();

type DocumentReadResult =
  | { kind: "missing" }
  | { kind: "corrupt"; error?: unknown }
  | { kind: "unreadable"; error: unknown }
  | { kind: "valid"; document: DailyFilterCheckpointDocument };

/**
 * A day whose candidates exceed the single-request budget is classified in
 * several batches. Batch 0 keeps the historical `<date>.json` path, so an
 * ordinary one-request day is stored exactly as before; later batches use
 * `<date>.batch-<n>.json`.
 */
export function deriveDailyFilterCheckpointPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  reportDate: string,
  batch = 0,
): DailyFilterCheckpointPaths {
  requireReportDate(reportDate);
  if (!Number.isSafeInteger(batch) || batch < 0) {
    throw new DailyFilterCheckpointStoreError(`invalid checkpoint batch: ${batch}`);
  }
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(`${indexDir}/filter-checkpoints`);
  const documentPath = storage.normalizePath(
    batch === 0 ? `${directory}/${reportDate}.json` : `${directory}/${reportDate}.batch-${batch}.json`,
  );
  const personalizedDocumentPath = storage.normalizePath(
    `${directory}/${reportDate}.personalized.json`,
  );
  const legacyNoveltyDocumentPath = storage.normalizePath(`${directory}/${reportDate}.novelty.json`);
  return {
    directory,
    documentPath,
    backupPath: storage.normalizePath(`${documentPath}.bak`),
    personalizedDocumentPath,
    personalizedBackupPath: storage.normalizePath(`${personalizedDocumentPath}.bak`),
    legacyNoveltyDocumentPath,
    legacyNoveltyBackupPath: storage.normalizePath(`${legacyNoveltyDocumentPath}.bak`),
  };
}

export function buildDailyFilterCheckpointFingerprintInput(
  input: DailyFilterCheckpointCompatibilityInput | PreparedDailyFilterCheckpoint,
): DailyFilterCheckpointFingerprintInput {
  try {
    const prepared = isPreparedDailyFilterCheckpoint(input)
      ? input
      : prepareDailyFilterCheckpoint(input);
    return clone(prepared.fingerprintInput);
  } catch (error) {
    throw new DailyFilterCheckpointStoreError(
      "invalid daily filter generation identity",
      error,
    );
  }
}

export function createDailyFilterCompatibilityFingerprint(
  input: DailyFilterCheckpointCompatibilityInput | PreparedDailyFilterCheckpoint,
): string {
  return fingerprint(buildDailyFilterCheckpointFingerprintInput(input));
}

export class DailyFilterCheckpointStore {
  constructor(
    private readonly storage: StorageAdapter,
    private readonly output: OutputSettings,
    private readonly options: DailyFilterCheckpointStoreOptions = {},
  ) {}

  pathsFor(reportDate: string, batch = 0): DailyFilterCheckpointPaths {
    return deriveDailyFilterCheckpointPaths(this.storage, this.output, reportDate, batch);
  }

  async load(reportDate: string, batch = 0): Promise<DailyFilterCheckpointDocument | null> {
    const paths = this.pathsFor(reportDate, batch);
    const primary = await this.readDocument(paths.documentPath, reportDate);
    if (primary.kind === "valid") return primary.document;
    if (primary.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot read daily filter checkpoint: ${paths.documentPath}`,
        primary.error,
      );
    }
    const backup = await this.readDocument(paths.backupPath, reportDate);
    if (backup.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot read daily filter checkpoint: ${paths.backupPath}`,
        backup.error,
      );
    }
    if (backup.kind === "valid") {
      this.warn(`daily filter checkpoint recovered from backup: ${paths.backupPath}`);
      return backup.document;
    }
    return null;
  }

  async lookupReusable(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
    batch = 0,
  ): Promise<FilterRecord[] | null> {
    requirePrepared(prepared);
    const document = await this.load(reportDate, batch);
    if (!document) return null;
    if (document.fingerprint !== createDailyFilterCompatibilityFingerprint(prepared)) return null;
    return clone(document.result);
  }

  save(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
    result: unknown,
    batch = 0,
  ): Promise<DailyFilterCheckpointDocument> {
    const paths = this.pathsFor(reportDate, batch);
    return this.enqueue(paths.documentPath, async () => {
      requirePrepared(prepared);
      await this.assertMutable(reportDate, batch);
      const fingerprintInput = buildDailyFilterCheckpointFingerprintInput(prepared);
      if (
        fingerprintInput.promptContractVersion !== DAILY_FILTER_PROMPT_CONTRACT_VERSION ||
        fingerprintInput.resultContractVersion !== DAILY_FILTER_RESULT_CONTRACT_VERSION
      ) {
        throw new DailyFilterCheckpointStoreError(
          "cannot persist unsupported daily filter contract versions",
        );
      }
      const decoded = decodePaperFilterRecords(
        { papers: result },
        new Set(fingerprintInput.request.identity.knownIds),
        new Set(fingerprintInput.request.identity.validTags),
        fingerprintInput.request.identity.directions,
      );
      if (!decoded.ok) {
        throw new DailyFilterCheckpointStoreError(
          `invalid daily filter checkpoint result: ${decoded.reason}`,
        );
      }
      const document: DailyFilterCheckpointDocument = {
        schemaVersion: DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION,
        reportDate,
        fingerprint: fingerprint(fingerprintInput),
        fingerprintInput,
        completedAt: this.now().toISOString(),
        result: decoded.value,
      };
      await this.persist(paths, document);
      return clone(document);
    });
  }

  async removeAll(reportDate: string): Promise<void> {
    const paths = this.pathsFor(reportDate);
    await this.enqueue(paths.documentPath, async () => {
      await removeIfExists(this.storage, paths.documentPath);
      await removeIfExists(this.storage, paths.backupPath);
      await removeIfExists(this.storage, `${paths.documentPath}.tmp`);
      await removeIfExists(this.storage, `${paths.backupPath}.tmp`);
    });
    await this.enqueue(paths.personalizedDocumentPath, async () => {
      await removeIfExists(this.storage, paths.personalizedDocumentPath);
      await removeIfExists(this.storage, paths.personalizedBackupPath);
      await removeIfExists(this.storage, `${paths.personalizedDocumentPath}.tmp`);
      await removeIfExists(this.storage, `${paths.personalizedBackupPath}.tmp`);
    });
    await this.enqueue(paths.legacyNoveltyDocumentPath, async () => {
      await removeIfExists(this.storage, paths.legacyNoveltyDocumentPath);
      await removeIfExists(this.storage, paths.legacyNoveltyBackupPath);
      await removeIfExists(this.storage, `${paths.legacyNoveltyDocumentPath}.tmp`);
      await removeIfExists(this.storage, `${paths.legacyNoveltyBackupPath}.tmp`);
    });
    for (const batchPath of await this.batchDocumentPaths(reportDate, paths.directory)) {
      await this.enqueue(batchPath, async () => {
        for (const path of [batchPath, `${batchPath}.bak`, `${batchPath}.tmp`, `${batchPath}.bak.tmp`]) {
          await removeIfExists(this.storage, path);
        }
      });
    }
  }

  /** Every `<date>.batch-<n>.json` left by a multi-batch day, however many there were. */
  private async batchDocumentPaths(reportDate: string, directory: string): Promise<string[]> {
    const prefix = `${reportDate}.batch-`;
    if (this.storage.list) {
      if (!(await this.storage.exists(directory))) return [];
      const bases = new Set<string>();
      for (const entry of await this.storage.list(directory)) {
        const name = entry.path.slice(entry.path.lastIndexOf("/") + 1);
        if (!name.startsWith(prefix)) continue;
        const base = /^(.*\.json)(?:\.bak)?(?:\.tmp)?$/.exec(entry.path)?.[1];
        if (base) bases.add(base);
      }
      return [...bases];
    }
    const found: string[] = [];
    for (let batch = 1; batch <= FILTER_CHECKPOINT_UNLISTED_BATCH_SCAN; batch += 1) {
      const path = this.pathsFor(reportDate, batch).documentPath;
      if (await this.storage.exists(path)) found.push(path);
    }
    return found;
  }

  private async assertMutable(reportDate: string, batch = 0): Promise<void> {
    const paths = this.pathsFor(reportDate, batch);
    const primary = await this.readDocument(paths.documentPath, reportDate);
    if (primary.kind === "valid") return;
    if (primary.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot mutate unreadable daily filter checkpoint: ${paths.documentPath}`,
        primary.error,
      );
    }
    const backup = await this.readDocument(paths.backupPath, reportDate);
    if (backup.kind === "valid") {
      this.warn(`daily filter checkpoint recovered from backup: ${paths.backupPath}`);
      return;
    }
    if (backup.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot mutate unreadable daily filter checkpoint: ${paths.backupPath}`,
        backup.error,
      );
    }
    // Missing or readable-corrupt documents carry no durable state. A newly
    // strict-valid result may replace them; persist will never rotate corrupt
    // bytes into the backup or use them as promotion recovery content.
    return;
  }

  private async readDocument(path: string, reportDate: string): Promise<DocumentReadResult> {
    let exists: boolean;
    try {
      exists = await this.storage.exists(path);
    } catch (error) {
      this.warn(`unreadable daily filter checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    if (!exists) return { kind: "missing" };
    let raw: string;
    try {
      raw = await this.storage.readText(path);
    } catch (error) {
      this.warn(`unreadable daily filter checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    try {
      const document = decodeDocument(JSON.parse(raw), reportDate);
      if (!document) {
        this.warn(`invalid daily filter checkpoint ignored: ${path}`);
        return { kind: "corrupt" };
      }
      return { kind: "valid", document };
    } catch (error) {
      this.warn(`corrupt daily filter checkpoint ignored: ${path}`, error);
      return { kind: "corrupt", error };
    }
  }

  private async persist(
    paths: DailyFilterCheckpointPaths,
    document: DailyFilterCheckpointDocument,
  ): Promise<void> {
    await ensureDirDeep(this.storage, paths.directory);
    try {
      await replaceWithBackup(
        this.storage,
        paths,
        `${JSON.stringify(document, null, 2)}\n`,
        document.reportDate,
      );
    } catch (error) {
      throw new DailyFilterCheckpointStoreError(
        `failed to save daily filter checkpoint: ${paths.documentPath}`,
        error,
      );
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

function decodeDocument(value: unknown, reportDate: string): DailyFilterCheckpointDocument | null {
  if (!isExactObject(value, [
    "schemaVersion", "reportDate", "fingerprint", "fingerprintInput", "completedAt", "result",
  ])) return null;
  if (
    value.schemaVersion !== DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION ||
    value.reportDate !== reportDate ||
    typeof value.fingerprint !== "string" ||
    !/^sha256:[0-9a-f]{64}$/.test(value.fingerprint) ||
    !isIsoDate(value.completedAt)
  ) return null;
  const fingerprintInput = decodeFingerprintInput(value.fingerprintInput);
  if (!fingerprintInput || fingerprint(fingerprintInput) !== value.fingerprint) return null;
  const decoded = decodePaperFilterRecords(
    { papers: value.result },
    new Set(fingerprintInput.request.identity.knownIds),
    new Set(fingerprintInput.request.identity.validTags),
    fingerprintInput.request.identity.directions,
  );
  if (!decoded.ok) return null;
  return {
    schemaVersion: DAILY_FILTER_CHECKPOINT_SCHEMA_VERSION,
    reportDate,
    fingerprint: value.fingerprint,
    fingerprintInput,
    completedAt: value.completedAt,
    result: decoded.value,
  };
}

function decodeFingerprintInput(value: unknown): DailyFilterCheckpointFingerprintInput | null {
  if (!isExactObject(value, [
    "fingerprintVersion", "request", "generation", "promptContractVersion", "resultContractVersion",
  ])) return null;
  if (
    value.fingerprintVersion !== DAILY_FILTER_FINGERPRINT_VERSION ||
    value.promptContractVersion !== DAILY_FILTER_PROMPT_CONTRACT_VERSION ||
    value.resultContractVersion !== DAILY_FILTER_RESULT_CONTRACT_VERSION ||
    !isExactObject(value.request, ["messages", "identity"]) ||
    !isMessages(value.request.messages) ||
    !isRequestIdentity(value.request.identity) ||
    !isGeneration(value.generation)
  ) return null;
  return clone(value) as unknown as DailyFilterCheckpointFingerprintInput;
}

function isRequestIdentity(value: unknown): value is {
  knownIds: string[];
  validTags: string[];
  directions: FilterDirectionRef[];
} {
  return isExactObject(value, ["knownIds", "validTags", "directions"]) &&
    isStringArray(value.knownIds) &&
    isStringArray(value.validTags) &&
    isDirectionRefs(value.directions);
}

function isDirectionRefs(value: unknown): value is FilterDirectionRef[] {
  return Array.isArray(value) && value.every((item) =>
    isExactObject(item, ["ref", "tag", "id", "text"]) &&
    typeof item.ref === "string" &&
    typeof item.tag === "string" &&
    typeof item.id === "string" &&
    typeof item.text === "string");
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isMessages(value: unknown): value is ChatMessage[] {
  return Array.isArray(value) && value.length === 2 && value.every((message) =>
    isExactObject(message, ["role", "content"]) &&
    (message.role === "system" || message.role === "user" || message.role === "assistant") &&
    typeof message.content === "string"
  );
}

function isGeneration(value: unknown): value is CheckpointGenerationIdentity {
  if (!isExactObject(value, ["requestContractVersion", "provider", "endpointDigest", "model", "mode"]) ||
      value.requestContractVersion !== LLM_REQUEST_CONTRACT_VERSION ||
      typeof value.provider !== "string" || typeof value.model !== "string" ||
      typeof value.endpointDigest !== "string" ||
      !/^sha256:[0-9a-f]{64}$/.test(value.endpointDigest) || !isPlainObject(value.mode)) return false;
  if (value.mode.kind === "temperature") {
    return isExactObject(value.mode, ["kind", "temperature"]) &&
      typeof value.mode.temperature === "number" && Number.isFinite(value.mode.temperature);
  }
  if (value.mode.kind === "anthropic-thinking") {
    return value.provider === "anthropic" && isExactObject(value.mode, ["kind", "budgetTokens"]) &&
      typeof value.mode.budgetTokens === "number" &&
      Number.isSafeInteger(value.mode.budgetTokens) && value.mode.budgetTokens > 0;
  }
  if (value.mode.kind === "anthropic-adaptive") {
    return value.provider === "anthropic" && isExactObject(value.mode, ["kind", "reasoningEffort"]) &&
      typeof value.mode.reasoningEffort === "string";
  }
  return value.mode.kind === "reasoning-thinking" && value.provider !== "anthropic" &&
    isExactObject(value.mode, ["kind", "reasoningEffort"]) &&
    typeof value.mode.reasoningEffort === "string";
}

async function replaceWithBackup(
  storage: StorageAdapter,
  paths: DailyFilterCheckpointPaths,
  content: string,
  reportDate: string,
): Promise<void> {
  const tmp = `${paths.documentPath}.tmp`;
  const backupTmp = `${paths.backupPath}.tmp`;
  await removeIfExists(storage, tmp);
  await removeIfExists(storage, backupTmp);
  try {
    await writePrivateCheckpointText(storage, tmp, content);
    let previous: string | null = null;
    if (await storage.exists(paths.documentPath)) {
      const raw = await storage.readText(paths.documentPath);
      if (decodeRawDocument(raw, reportDate)) previous = raw;
    }
    let recoveryContent = previous;
    if (recoveryContent === null && await storage.exists(paths.backupPath)) {
      const raw = await storage.readText(paths.backupPath);
      if (decodeRawDocument(raw, reportDate)) {
        recoveryContent = raw;
      } else {
        await storage.remove(paths.backupPath);
      }
    }
    if (previous !== null) {
      await writePrivateCheckpointText(storage, backupTmp, previous);
      await removeIfExists(storage, paths.backupPath);
      await storage.rename(backupTmp, paths.backupPath);
    }
    if (await storage.exists(paths.documentPath)) await storage.remove(paths.documentPath);
    try {
      await storage.rename(tmp, paths.documentPath);
    } catch (error) {
      if (recoveryContent !== null) {
        await removeIfExists(storage, tmp);
        await writePrivateCheckpointText(storage, tmp, recoveryContent);
        await storage.rename(tmp, paths.documentPath);
      }
      throw error;
    }
  } finally {
    await removeIfExists(storage, tmp);
    await removeIfExists(storage, backupTmp);
  }
}

function fingerprint(input: DailyFilterCheckpointFingerprintInput): string {
  return `sha256:${sha256ForCheckpointTests(JSON.stringify(input))}`;
}

function decodeRawDocument(raw: string, reportDate: string): DailyFilterCheckpointDocument | null {
  try {
    return decodeDocument(JSON.parse(raw), reportDate);
  } catch {
    return null;
  }
}

function requirePrepared(value: unknown): asserts value is PreparedDailyFilterCheckpoint {
  if (!isPreparedDailyFilterCheckpoint(value)) {
    throw new DailyFilterCheckpointStoreError(
      "daily filter checkpoint requires a prepared exact request snapshot",
    );
  }
}

export function isPersonalizedCheckpointStoreError(
  error: unknown,
): error is DailyFilterCheckpointStoreError {
  return error instanceof DailyFilterCheckpointStoreError;
}

function requireReportDate(value: string): void {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  const year = Number(match?.[1]);
  const month = Number(match?.[2]);
  const day = Number(match?.[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (!match || date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 ||
      date.getUTCDate() !== day) {
    throw new DailyFilterCheckpointStoreError(`invalid checkpoint report date: ${value}`);
  }
}

function isIsoDate(value: unknown): value is string {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
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

async function writePrivateCheckpointText(
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
