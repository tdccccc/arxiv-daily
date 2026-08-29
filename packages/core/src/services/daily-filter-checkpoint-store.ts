import type { StorageAdapter } from "../core/adapters";
import type { ChatMessage } from "../llm/client";
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
  type FilterRecord,
  type PreparedDailyFilterCheckpoint,
} from "../pipeline/paper-filter-contract";
import type { OutputSettings } from "../settings/types";
import {
  PERSONALIZED_FILTER_PROMPT_CONTRACT_VERSION,
  PERSONALIZED_FILTER_RESULT_CONTRACT_VERSION,
  decodePersonalizedDirectionRecords,
  decodePersonalizedFilterFingerprintInput,
  fingerprintPersonalizedFilterInput,
  isPreparedPersonalizedFilterCheckpoint,
  type PersonalizedDirectionRecord,
  type PreparedPersonalizedFilterCheckpoint,
} from "../pipeline/personalized-paper-filter";
import {
  PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION,
  PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION,
  decodeNoveltyCheckpointRecords,
  decodeNoveltyFingerprintInput,
  fingerprintNoveltyCheckpointInput,
  isPreparedNoveltyCheckpoint,
  prepareNoveltyCheckpoint,
  type NoveltyCheckpointRecord,
  type PreparedNoveltyCheckpoint,
} from "../pipeline/personalized-novelty";
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
export {
  PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION,
  PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION,
  decodeNoveltyCheckpointRecords,
  decodeNoveltyFingerprintInput,
  fingerprintNoveltyCheckpointInput,
  isPreparedNoveltyCheckpoint,
  prepareNoveltyCheckpoint,
  type NoveltyCheckpointPort,
  type NoveltyCheckpointRecord,
  type PreparedNoveltyCheckpoint,
} from "../pipeline/personalized-novelty";

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
  personalizedDocumentPath: string;
  personalizedBackupPath: string;
  noveltyDocumentPath: string;
  noveltyBackupPath: string;
}

export const PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION = 1 as const;

export interface PersonalizedFilterCheckpointDocument {
  schemaVersion: typeof PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION;
  reportDate: string;
  fingerprint: string;
  fingerprintInput: PreparedPersonalizedFilterCheckpoint["fingerprintInput"];
  completedAt: string;
  result: PersonalizedDirectionRecord[];
}

export const NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION = 1 as const;

export interface NoveltyFilterCheckpointDocument {
  schemaVersion: typeof NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION;
  reportDate: string;
  fingerprint: string;
  fingerprintInput: PreparedNoveltyCheckpoint["fingerprintInput"];
  completedAt: string;
  result: NoveltyCheckpointRecord[];
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

const mutationQueues = new WeakMap<StorageAdapter, Map<string, Promise<unknown>>>();

type DocumentReadResult =
  | { kind: "missing" }
  | { kind: "corrupt"; error?: unknown }
  | { kind: "unreadable"; error: unknown }
  | { kind: "valid"; document: DailyFilterCheckpointDocument };

export function deriveDailyFilterCheckpointPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  reportDate: string,
): DailyFilterCheckpointPaths {
  requireReportDate(reportDate);
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(`${indexDir}/filter-checkpoints`);
  const documentPath = storage.normalizePath(`${directory}/${reportDate}.json`);
  const personalizedDocumentPath = storage.normalizePath(
    `${directory}/${reportDate}.personalized.json`,
  );
  const noveltyDocumentPath = storage.normalizePath(`${directory}/${reportDate}.novelty.json`);
  return {
    directory,
    documentPath,
    backupPath: storage.normalizePath(`${documentPath}.bak`),
    personalizedDocumentPath,
    personalizedBackupPath: storage.normalizePath(`${personalizedDocumentPath}.bak`),
    noveltyDocumentPath,
    noveltyBackupPath: storage.normalizePath(`${noveltyDocumentPath}.bak`),
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

  pathsFor(reportDate: string): DailyFilterCheckpointPaths {
    return deriveDailyFilterCheckpointPaths(this.storage, this.output, reportDate);
  }

  async load(reportDate: string): Promise<DailyFilterCheckpointDocument | null> {
    const paths = this.pathsFor(reportDate);
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
  ): Promise<FilterRecord[] | null> {
    requirePrepared(prepared);
    const document = await this.load(reportDate);
    if (!document) return null;
    if (document.fingerprint !== createDailyFilterCompatibilityFingerprint(prepared)) return null;
    return clone(document.result);
  }

  async loadPersonalized(reportDate: string): Promise<PersonalizedFilterCheckpointDocument | null> {
    const paths = this.pathsFor(reportDate);
    const primary = await this.readPersonalizedDocument(paths.personalizedDocumentPath, reportDate);
    if (primary.kind === "valid") return primary.document;
    if (primary.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot read personalized filter checkpoint: ${paths.personalizedDocumentPath}`,
        primary.error,
      );
    }
    const backup = await this.readPersonalizedDocument(paths.personalizedBackupPath, reportDate);
    if (backup.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot read personalized filter checkpoint: ${paths.personalizedBackupPath}`,
        backup.error,
      );
    }
    if (backup.kind === "valid") {
      this.warn(`personalized filter checkpoint recovered from backup: ${paths.personalizedBackupPath}`);
      return backup.document;
    }
    return null;
  }

  async lookupPersonalizedReusable(
    reportDate: string,
    prepared: PreparedPersonalizedFilterCheckpoint,
  ): Promise<PersonalizedDirectionRecord[] | null> {
    requirePersonalizedPrepared(prepared);
    const document = await this.loadPersonalized(reportDate);
    if (!document || document.fingerprint !== prepared.fingerprint) return null;
    return clone(document.result);
  }

  savePersonalized(
    reportDate: string,
    prepared: PreparedPersonalizedFilterCheckpoint,
    result: unknown,
  ): Promise<PersonalizedFilterCheckpointDocument> {
    const paths = this.pathsFor(reportDate);
    return this.enqueue(paths.personalizedDocumentPath, async () => {
      requirePersonalizedPrepared(prepared);
      await this.assertPersonalizedMutable(reportDate);
      if (prepared.fingerprintInput.promptContractVersion
          !== PERSONALIZED_FILTER_PROMPT_CONTRACT_VERSION
        || prepared.fingerprintInput.resultContractVersion
          !== PERSONALIZED_FILTER_RESULT_CONTRACT_VERSION) {
        throw new DailyFilterCheckpointStoreError(
          "cannot persist unsupported personalized filter contract versions",
        );
      }
      const decoded = decodePersonalizedDirectionRecords(
        { papers: result },
        prepared.fingerprintInput.plan.paperKeys,
        new Set(prepared.fingerprintInput.plan.directionIds),
      );
      if (!decoded.ok) {
        throw new DailyFilterCheckpointStoreError(
          `invalid personalized filter checkpoint result: ${decoded.reason}`,
        );
      }
      const document: PersonalizedFilterCheckpointDocument = {
        schemaVersion: PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION,
        reportDate,
        fingerprint: prepared.fingerprint,
        fingerprintInput: clone(prepared.fingerprintInput),
        completedAt: this.now().toISOString(),
        result: decoded.value,
      };
      await this.persistPersonalized(paths, document);
      return clone(document);
    });
  }

  async loadNovelty(reportDate: string): Promise<NoveltyFilterCheckpointDocument | null> {
    const paths = this.pathsFor(reportDate);
    const primary = await this.readNoveltyDocument(paths.noveltyDocumentPath, reportDate);
    if (primary.kind === "valid") return primary.document;
    if (primary.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot read personal novelty checkpoint: ${paths.noveltyDocumentPath}`,
        primary.error,
      );
    }
    const backup = await this.readNoveltyDocument(paths.noveltyBackupPath, reportDate);
    if (backup.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot read personal novelty checkpoint: ${paths.noveltyBackupPath}`,
        backup.error,
      );
    }
    if (backup.kind === "valid") {
      this.warn(`personal novelty checkpoint recovered from backup: ${paths.noveltyBackupPath}`);
      return backup.document;
    }
    return null;
  }

  async lookupNoveltyReusable(
    reportDate: string,
    prepared: PreparedNoveltyCheckpoint,
  ): Promise<NoveltyCheckpointRecord[] | null> {
    requireNoveltyPrepared(prepared);
    const document = await this.loadNovelty(reportDate);
    if (!document || document.fingerprint !== prepared.fingerprint) return null;
    return clone(document.result);
  }

  saveNovelty(
    reportDate: string,
    prepared: PreparedNoveltyCheckpoint,
    result: unknown,
  ): Promise<NoveltyFilterCheckpointDocument> {
    const paths = this.pathsFor(reportDate);
    return this.enqueue(paths.noveltyDocumentPath, async () => {
      requireNoveltyPrepared(prepared);
      await this.assertNoveltyMutable(reportDate);
      if (prepared.fingerprintInput.promptContractVersion
          !== PERSONAL_NOVELTY_PROMPT_CONTRACT_VERSION
        || prepared.fingerprintInput.resultContractVersion
          !== PERSONAL_NOVELTY_RESULT_CONTRACT_VERSION) {
        throw new DailyFilterCheckpointStoreError(
          "cannot persist unsupported personal novelty contract versions",
        );
      }
      // Every planned call paper must carry exactly one strict terminal outcome
      // (novelty, validation-exhausted, or plan-too-large); input-invalid,
      // transport, output-limit, and checkpoint reasons are never durable, and
      // partial coverage is rejected so a hit can only reuse complete results.
      const decoded = decodeNoveltyCheckpointRecords(result, prepared.fingerprintInput.plan);
      if (!decoded.ok) {
        throw new DailyFilterCheckpointStoreError(
          `invalid personal novelty checkpoint result: ${decoded.reason}`,
        );
      }
      const document: NoveltyFilterCheckpointDocument = {
        schemaVersion: NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION,
        reportDate,
        fingerprint: prepared.fingerprint,
        fingerprintInput: clone(prepared.fingerprintInput),
        completedAt: this.now().toISOString(),
        result: decoded.value,
      };
      await this.persistNovelty(paths, document);
      return clone(document);
    });
  }

  save(
    reportDate: string,
    prepared: PreparedDailyFilterCheckpoint,
    result: unknown,
  ): Promise<DailyFilterCheckpointDocument> {
    const paths = this.pathsFor(reportDate);
    return this.enqueue(paths.documentPath, async () => {
      requirePrepared(prepared);
      await this.assertMutable(reportDate);
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
    await this.enqueue(paths.noveltyDocumentPath, async () => {
      await removeIfExists(this.storage, paths.noveltyDocumentPath);
      await removeIfExists(this.storage, paths.noveltyBackupPath);
      await removeIfExists(this.storage, `${paths.noveltyDocumentPath}.tmp`);
      await removeIfExists(this.storage, `${paths.noveltyBackupPath}.tmp`);
    });
  }

  private async assertPersonalizedMutable(reportDate: string): Promise<void> {
    const paths = this.pathsFor(reportDate);
    const primary = await this.readPersonalizedDocument(paths.personalizedDocumentPath, reportDate);
    if (primary.kind === "valid") return;
    if (primary.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot mutate unreadable personalized filter checkpoint: ${paths.personalizedDocumentPath}`,
        primary.error,
      );
    }
    const backup = await this.readPersonalizedDocument(paths.personalizedBackupPath, reportDate);
    if (backup.kind === "valid") return;
    if (backup.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot mutate unreadable personalized filter checkpoint: ${paths.personalizedBackupPath}`,
        backup.error,
      );
    }
  }

  private async assertMutable(reportDate: string): Promise<void> {
    const paths = this.pathsFor(reportDate);
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

  private async assertNoveltyMutable(reportDate: string): Promise<void> {
    const paths = this.pathsFor(reportDate);
    const primary = await this.readNoveltyDocument(paths.noveltyDocumentPath, reportDate);
    if (primary.kind === "valid") return;
    if (primary.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot mutate unreadable personal novelty checkpoint: ${paths.noveltyDocumentPath}`,
        primary.error,
      );
    }
    const backup = await this.readNoveltyDocument(paths.noveltyBackupPath, reportDate);
    if (backup.kind === "valid") {
      this.warn(`personal novelty checkpoint recovered from backup: ${paths.noveltyBackupPath}`);
      return;
    }
    if (backup.kind === "unreadable") {
      throw new DailyFilterCheckpointStoreError(
        `cannot mutate unreadable personal novelty checkpoint: ${paths.noveltyBackupPath}`,
        backup.error,
      );
    }
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

  private async readPersonalizedDocument(
    path: string,
    reportDate: string,
  ): Promise<{ kind: "missing" } | { kind: "corrupt"; error?: unknown }
    | { kind: "unreadable"; error: unknown }
    | { kind: "valid"; document: PersonalizedFilterCheckpointDocument }> {
    let exists: boolean;
    try {
      exists = await this.storage.exists(path);
    } catch (error) {
      this.warn(`unreadable personalized filter checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    if (!exists) return { kind: "missing" };
    let raw: string;
    try {
      raw = await this.storage.readText(path);
    } catch (error) {
      this.warn(`unreadable personalized filter checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    try {
      const document = decodePersonalizedDocument(JSON.parse(raw), reportDate);
      if (!document) return { kind: "corrupt" };
      return { kind: "valid", document };
    } catch (error) {
      return { kind: "corrupt", error };
    }
  }

  private async readNoveltyDocument(
    path: string,
    reportDate: string,
  ): Promise<{ kind: "missing" } | { kind: "corrupt"; error?: unknown }
    | { kind: "unreadable"; error: unknown }
    | { kind: "valid"; document: NoveltyFilterCheckpointDocument }> {
    let exists: boolean;
    try {
      exists = await this.storage.exists(path);
    } catch (error) {
      this.warn(`unreadable personal novelty checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    if (!exists) return { kind: "missing" };
    let raw: string;
    try {
      raw = await this.storage.readText(path);
    } catch (error) {
      this.warn(`unreadable personal novelty checkpoint ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    try {
      const document = decodeNoveltyDocument(JSON.parse(raw), reportDate);
      if (!document) {
        this.warn(`invalid personal novelty checkpoint ignored: ${path}`);
        return { kind: "corrupt" };
      }
      return { kind: "valid", document };
    } catch (error) {
      this.warn(`corrupt personal novelty checkpoint ignored: ${path}`, error);
      return { kind: "corrupt", error };
    }
  }

  private async persistPersonalized(
    paths: DailyFilterCheckpointPaths,
    document: PersonalizedFilterCheckpointDocument,
  ): Promise<void> {
    await ensureDirDeep(this.storage, paths.directory);
    try {
      await replacePersonalizedWithBackup(
        this.storage, paths, `${JSON.stringify(document, null, 2)}\n`, document.reportDate,
      );
    } catch (error) {
      throw new DailyFilterCheckpointStoreError(
        `failed to save personalized filter checkpoint: ${paths.personalizedDocumentPath}`,
        error,
      );
    }
  }

  private async persistNovelty(
    paths: DailyFilterCheckpointPaths,
    document: NoveltyFilterCheckpointDocument,
  ): Promise<void> {
    await ensureDirDeep(this.storage, paths.directory);
    try {
      await replaceNoveltyWithBackup(
        this.storage, paths, `${JSON.stringify(document, null, 2)}\n`, document.reportDate,
      );
    } catch (error) {
      throw new DailyFilterCheckpointStoreError(
        `failed to save personal novelty checkpoint: ${paths.noveltyDocumentPath}`,
        error,
      );
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

function decodePersonalizedDocument(
  value: unknown,
  reportDate: string,
): PersonalizedFilterCheckpointDocument | null {
  if (!isExactObject(value, [
    "schemaVersion", "reportDate", "fingerprint", "fingerprintInput", "completedAt", "result",
  ]) || value.schemaVersion !== PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION
    || value.reportDate !== reportDate
    || typeof value.fingerprint !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(value.fingerprint)
    || !isIsoDate(value.completedAt)) return null;
  const fingerprintInput = decodePersonalizedFilterFingerprintInput(value.fingerprintInput);
  if (!fingerprintInput
    || fingerprintPersonalizedFilterInput(fingerprintInput) !== value.fingerprint) return null;
  const decoded = decodePersonalizedDirectionRecords(
    { papers: value.result },
    fingerprintInput.plan.paperKeys,
    new Set(fingerprintInput.plan.directionIds),
  );
  if (!decoded.ok) return null;
  return {
    schemaVersion: PERSONALIZED_FILTER_CHECKPOINT_SCHEMA_VERSION,
    reportDate,
    fingerprint: value.fingerprint,
    fingerprintInput,
    completedAt: value.completedAt,
    result: decoded.value,
  };
}

function decodeNoveltyDocument(
  value: unknown,
  reportDate: string,
): NoveltyFilterCheckpointDocument | null {
  if (!isExactObject(value, [
    "schemaVersion", "reportDate", "fingerprint", "fingerprintInput", "completedAt", "result",
  ]) || value.schemaVersion !== NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION
    || value.reportDate !== reportDate
    || typeof value.fingerprint !== "string"
    || !/^sha256:[0-9a-f]{64}$/.test(value.fingerprint)
    || !isIsoDate(value.completedAt)) return null;
  const fingerprintInput = decodeNoveltyFingerprintInput(value.fingerprintInput);
  if (!fingerprintInput
    || fingerprintNoveltyCheckpointInput(fingerprintInput) !== value.fingerprint) return null;
  const decoded = decodeNoveltyCheckpointRecords(value.result, fingerprintInput.plan);
  if (!decoded.ok) return null;
  return {
    schemaVersion: NOVELTY_FILTER_CHECKPOINT_SCHEMA_VERSION,
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
  return clone(value) as DailyFilterCheckpointFingerprintInput;
}

function isRequestIdentity(value: unknown): value is {
  knownIds: string[];
  validTags: string[];
} {
  return isExactObject(value, ["knownIds", "validTags"]) &&
    isStringArray(value.knownIds) &&
    isStringArray(value.validTags);
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
  if (!isExactObject(value, ["provider", "endpointDigest", "model", "mode"]) ||
      typeof value.provider !== "string" || typeof value.model !== "string" ||
      typeof value.endpointDigest !== "string" ||
      !/^sha256:[0-9a-f]{64}$/.test(value.endpointDigest) || !isPlainObject(value.mode)) return false;
  if (value.mode.kind === "temperature") {
    return isExactObject(value.mode, ["kind", "temperature"]) &&
      typeof value.mode.temperature === "number" && Number.isFinite(value.mode.temperature);
  }
  if (value.mode.kind === "anthropic-thinking") {
    return value.provider === "anthropic" && isExactObject(value.mode, ["kind", "budgetTokens"]) &&
      Number.isSafeInteger(value.mode.budgetTokens) && value.mode.budgetTokens > 0;
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

async function replacePersonalizedWithBackup(
  storage: StorageAdapter,
  paths: DailyFilterCheckpointPaths,
  content: string,
  reportDate: string,
): Promise<void> {
  const tmp = `${paths.personalizedDocumentPath}.tmp`;
  const backupTmp = `${paths.personalizedBackupPath}.tmp`;
  await removeIfExists(storage, tmp);
  await removeIfExists(storage, backupTmp);
  try {
    await writePrivateCheckpointText(storage, tmp, content);
    let previous: string | null = null;
    if (await storage.exists(paths.personalizedDocumentPath)) {
      const raw = await storage.readText(paths.personalizedDocumentPath);
      if (decodeRawPersonalizedDocument(raw, reportDate)) previous = raw;
    }
    let recoveryContent = previous;
    if (recoveryContent === null && await storage.exists(paths.personalizedBackupPath)) {
      const raw = await storage.readText(paths.personalizedBackupPath);
      if (decodeRawPersonalizedDocument(raw, reportDate)) recoveryContent = raw;
      else await storage.remove(paths.personalizedBackupPath);
    }
    if (previous !== null) {
      await writePrivateCheckpointText(storage, backupTmp, previous);
      await removeIfExists(storage, paths.personalizedBackupPath);
      await storage.rename(backupTmp, paths.personalizedBackupPath);
    }
    if (await storage.exists(paths.personalizedDocumentPath)) {
      await storage.remove(paths.personalizedDocumentPath);
    }
    try {
      await storage.rename(tmp, paths.personalizedDocumentPath);
    } catch (error) {
      if (recoveryContent !== null) {
        await removeIfExists(storage, tmp);
        await writePrivateCheckpointText(storage, tmp, recoveryContent);
        await storage.rename(tmp, paths.personalizedDocumentPath);
      }
      throw error;
    }
  } finally {
    await removeIfExists(storage, tmp);
    await removeIfExists(storage, backupTmp);
  }
}

async function replaceNoveltyWithBackup(
  storage: StorageAdapter,
  paths: DailyFilterCheckpointPaths,
  content: string,
  reportDate: string,
): Promise<void> {
  const tmp = `${paths.noveltyDocumentPath}.tmp`;
  const backupTmp = `${paths.noveltyBackupPath}.tmp`;
  await removeIfExists(storage, tmp);
  await removeIfExists(storage, backupTmp);
  try {
    await writePrivateCheckpointText(storage, tmp, content);
    let previous: string | null = null;
    if (await storage.exists(paths.noveltyDocumentPath)) {
      const raw = await storage.readText(paths.noveltyDocumentPath);
      if (decodeRawNoveltyDocument(raw, reportDate)) previous = raw;
    }
    let recoveryContent = previous;
    if (recoveryContent === null && await storage.exists(paths.noveltyBackupPath)) {
      const raw = await storage.readText(paths.noveltyBackupPath);
      if (decodeRawNoveltyDocument(raw, reportDate)) recoveryContent = raw;
      else await storage.remove(paths.noveltyBackupPath);
    }
    if (previous !== null) {
      await writePrivateCheckpointText(storage, backupTmp, previous);
      await removeIfExists(storage, paths.noveltyBackupPath);
      await storage.rename(backupTmp, paths.noveltyBackupPath);
    }
    if (await storage.exists(paths.noveltyDocumentPath)) {
      await storage.remove(paths.noveltyDocumentPath);
    }
    try {
      await storage.rename(tmp, paths.noveltyDocumentPath);
    } catch (error) {
      if (recoveryContent !== null) {
        await removeIfExists(storage, tmp);
        await writePrivateCheckpointText(storage, tmp, recoveryContent);
        await storage.rename(tmp, paths.noveltyDocumentPath);
      }
      throw error;
    }
  } finally {
    await removeIfExists(storage, tmp);
    await removeIfExists(storage, backupTmp);
  }
}

function decodeRawPersonalizedDocument(
  raw: string,
  reportDate: string,
): PersonalizedFilterCheckpointDocument | null {
  try {
    return decodePersonalizedDocument(JSON.parse(raw), reportDate);
  } catch {
    return null;
  }
}

function decodeRawNoveltyDocument(
  raw: string,
  reportDate: string,
): NoveltyFilterCheckpointDocument | null {
  try {
    return decodeNoveltyDocument(JSON.parse(raw), reportDate);
  } catch {
    return null;
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

function requirePersonalizedPrepared(
  value: unknown,
): asserts value is PreparedPersonalizedFilterCheckpoint {
  if (!isPreparedPersonalizedFilterCheckpoint(value)) {
    throw new DailyFilterCheckpointStoreError(
      "personalized filter checkpoint requires a prepared exact call-plan snapshot",
    );
  }
}

function requireNoveltyPrepared(
  value: unknown,
): asserts value is PreparedNoveltyCheckpoint {
  if (!isPreparedNoveltyCheckpoint(value)) {
    throw new DailyFilterCheckpointStoreError(
      "personal novelty checkpoint requires a prepared exact call-plan snapshot",
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

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isExactObject(value: unknown, keys: readonly string[]): value is Record<string, any> {
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
