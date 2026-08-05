import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { modernArxivResources } from "../utils/arxiv";
import { sha256Hex } from "../utils/digest";
import { derivePaperInboxPaths } from "../services/paper-index";
import { paperKeyFromArxivId } from "../services/paper-key";

export const PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION = 1 as const;
export const PERSONAL_LIBRARY_IDENTIFICATION_VERSION = 2 as const;

export type PersonalLibraryFileRecord =
  | {
      path: string;
      status: "ready";
      observationFingerprint: string;
      paperKey: string;
      arxivId: string;
      updatedAt: string;
    }
  | {
      path: string;
      status: "unresolved";
      observationFingerprint: string;
      reason: "unrecognized-filename";
      updatedAt: string;
    }
  | {
      path: string;
      status: "unrelated";
      observationFingerprint: string;
      reason: "unsupported-file-type";
      updatedAt: string;
    }
  | {
      path: string;
      status: "failed";
      observationFingerprint: string;
      reason: "metadata-unavailable" | "metadata-fetch-failed";
      arxivId?: string;
      updatedAt: string;
    };

export interface PersonalLibraryPaperRecord {
  paperKey: string;
  source: "arxiv";
  externalId: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  updated: string;
  primaryCategory: string;
  categories: string[];
  evidenceDepth: "metadata-and-abstract";
  filePaths: string[];
}

export interface PersonalLibraryScanSummary {
  ready: number;
  unresolved: number;
  unrelated: number;
  failed: number;
  papers: number;
  truncated: boolean;
}

export interface PersonalLibraryCatalog {
  schemaVersion: typeof PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION;
  revision: number;
  scopeFingerprint: string;
  identificationFingerprint: string;
  updatedAt: string;
  lastScan: PersonalLibraryScanSummary | null;
  files: Record<string, PersonalLibraryFileRecord>;
  papers: Record<string, PersonalLibraryPaperRecord>;
}

export interface PersonalLibraryCatalogPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface PersonalLibraryCatalogStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export class PersonalLibraryCatalogStoreError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "PersonalLibraryCatalogStoreError";
  }
}

type DocumentReadResult =
  | { kind: "missing" }
  | { kind: "valid"; document: PersonalLibraryCatalog }
  | { kind: "corrupt"; error?: unknown }
  | { kind: "unreadable"; error: unknown };

const mutationQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

export function createPersonalLibraryScopeFingerprint(input: {
  rootIdentity: string;
  eligibleExtensions: readonly string[];
}): string {
  const rootIdentity = requireNonEmpty("rootIdentity", input.rootIdentity);
  const eligibleExtensions = Array.from(new Set(input.eligibleExtensions.map(normalizeExtension))).sort();
  if (eligibleExtensions.length === 0) {
    throw new PersonalLibraryCatalogStoreError("eligibleExtensions must not be empty");
  }
  return `sha256:${sha256Hex(JSON.stringify({ rootIdentity, eligibleExtensions }))}`;
}

export function createPersonalLibraryIdentificationFingerprint(
  eligibleExtensions: readonly string[],
): string {
  const extensions = Array.from(new Set(eligibleExtensions.map(normalizeExtension))).sort();
  if (extensions.length === 0) {
    throw new PersonalLibraryCatalogStoreError("eligibleExtensions must not be empty");
  }
  return `sha256:${sha256Hex(JSON.stringify({
    version: PERSONAL_LIBRARY_IDENTIFICATION_VERSION,
    strategy: "modern-arxiv-id-in-filename|pdf-text-evidence",
    eligibleExtensions: extensions,
  }))}`;
}

export function derivePersonalLibraryCatalogPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
): PersonalLibraryCatalogPaths {
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const documentPath = storage.normalizePath(`${indexDir}/personal-library-catalog.json`);
  return {
    directory: indexDir,
    documentPath,
    // Host atomic writers reserve the `.bak` sibling for their own rollback.
    backupPath: `${documentPath}.backup`,
  };
}

export function createEmptyPersonalLibraryCatalog(
  scopeFingerprint: string,
  identificationFingerprint: string,
  now: Date = new Date(),
): PersonalLibraryCatalog {
  return {
    schemaVersion: PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION,
    revision: 0,
    scopeFingerprint: requireFingerprint("scopeFingerprint", scopeFingerprint),
    identificationFingerprint: requireFingerprint(
      "identificationFingerprint",
      identificationFingerprint,
    ),
    updatedAt: now.toISOString(),
    lastScan: null,
    files: {},
    papers: {},
  };
}

export function decodePersonalLibraryCatalog(value: unknown): PersonalLibraryCatalog | null {
  if (!isExactObject(value, [
    "schemaVersion",
    "revision",
    "scopeFingerprint",
    "identificationFingerprint",
    "updatedAt",
    "lastScan",
    "files",
    "papers",
  ])) return null;
  if (
    value.schemaVersion !== PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION
    || !isNonNegativeSafeInteger(value.revision)
    || !isFingerprint(value.scopeFingerprint)
    || !isFingerprint(value.identificationFingerprint)
    || !isIsoDate(value.updatedAt)
    || !isPlainObject(value.files)
    || !isPlainObject(value.papers)
  ) return null;

  const lastScan = value.lastScan === null ? null : decodeScanSummary(value.lastScan);
  if (value.lastScan !== null && !lastScan) return null;

  const files: Record<string, PersonalLibraryFileRecord> = {};
  for (const [path, raw] of Object.entries(value.files)) {
    const record = decodeFileRecord(raw, path);
    if (!record) return null;
    defineRecordEntry(files, path, record);
  }

  const papers: Record<string, PersonalLibraryPaperRecord> = {};
  for (const [paperKey, raw] of Object.entries(value.papers)) {
    const record = decodePaperRecord(raw, paperKey);
    if (!record) return null;
    defineRecordEntry(papers, paperKey, record);
  }

  for (const record of Object.values(files)) {
    if (record.status === "ready" && !papers[record.paperKey]?.filePaths.includes(record.path)) {
      return null;
    }
  }
  for (const paper of Object.values(papers)) {
    if (paper.filePaths.some((path) => files[path]?.status !== "ready"
      || files[path].paperKey !== paper.paperKey)) return null;
  }

  return {
    schemaVersion: PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION,
    revision: value.revision,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    updatedAt: value.updatedAt,
    lastScan,
    files,
    papers,
  };
}

export class PersonalLibraryCatalogStore {
  readonly paths: PersonalLibraryCatalogPaths;

  constructor(
    private readonly storage: StorageAdapter,
    output: OutputSettings,
    private readonly options: PersonalLibraryCatalogStoreOptions = {},
  ) {
    this.paths = derivePersonalLibraryCatalogPaths(storage, output);
  }

  async load(
    scopeFingerprint: string,
    identificationFingerprint: string,
  ): Promise<PersonalLibraryCatalog> {
    return await this.enqueue(() => this.loadUnlocked(
      scopeFingerprint,
      identificationFingerprint,
    ));
  }

  private async loadUnlocked(
    scopeFingerprint: string,
    identificationFingerprint: string,
  ): Promise<PersonalLibraryCatalog> {
    const primary = await this.readDocument(this.paths.documentPath);
    if (primary.kind === "valid" && isCompatible(
      primary.document,
      scopeFingerprint,
      identificationFingerprint,
    )) return cloneCatalog(primary.document);
    const backup = await this.readDocument(this.paths.backupPath);
    if (backup.kind === "valid" && isCompatible(
      backup.document,
      scopeFingerprint,
      identificationFingerprint,
    )) {
      this.warn(`personal library catalog recovered from backup: ${this.paths.backupPath}`);
      await this.repairPrimary(backup.document);
      return cloneCatalog(backup.document);
    }
    // A document that is valid under a different scope/identification policy
    // is not data for the current strategy: it must never block a rescan and
    // never be promoted. Treat it like a missing document for this policy.
    const primaryUnavailable = primary.kind === "missing"
      || (primary.kind === "valid" && !isCompatible(
        primary.document,
        scopeFingerprint,
        identificationFingerprint,
      ));
    const backupUnavailable = backup.kind === "missing"
      || (backup.kind === "valid" && !isCompatible(
        backup.document,
        scopeFingerprint,
        identificationFingerprint,
      ));
    if (primaryUnavailable && backupUnavailable) {
      return createEmptyPersonalLibraryCatalog(
        scopeFingerprint,
        identificationFingerprint,
        this.now(),
      );
    }
    throw new PersonalLibraryCatalogStoreError(
      `cannot load unreadable personal library catalog: ${this.paths.documentPath}`,
      readError(backup) ?? readError(primary),
    );
  }

  replace(next: PersonalLibraryCatalog): Promise<PersonalLibraryCatalog> {
    return this.enqueue(async () => {
      const validated = decodePersonalLibraryCatalog(next);
      if (!validated) {
        throw new PersonalLibraryCatalogStoreError("cannot persist invalid personal library catalog");
      }
      const current = await this.loadForMutation(
        validated.scopeFingerprint,
        validated.identificationFingerprint,
      );
      const candidate = cloneCatalog(validated);
      candidate.schemaVersion = PERSONAL_LIBRARY_CATALOG_SCHEMA_VERSION;
      candidate.revision = current.revision;
      candidate.updatedAt = current.updatedAt;
      if (semanticContent(candidate) === semanticContent(current)) return cloneCatalog(current);
      if (current.revision === Number.MAX_SAFE_INTEGER) {
        throw new PersonalLibraryCatalogStoreError("personal library catalog revision is exhausted");
      }
      candidate.revision = current.revision + 1;
      candidate.updatedAt = this.now().toISOString();
      if (!decodePersonalLibraryCatalog(candidate)) {
        throw new PersonalLibraryCatalogStoreError("cannot persist invalid personal library catalog");
      }
      await this.save(candidate);
      return cloneCatalog(candidate);
    });
  }

  private async loadForMutation(
    scopeFingerprint: string,
    identificationFingerprint: string,
  ): Promise<PersonalLibraryCatalog> {
    const primary = await this.readDocument(this.paths.documentPath);
    if (primary.kind === "valid" && isCompatible(
      primary.document,
      scopeFingerprint,
      identificationFingerprint,
    )) return primary.document;
    const backup = await this.readDocument(this.paths.backupPath);
    if (backup.kind === "valid" && isCompatible(
      backup.document,
      scopeFingerprint,
      identificationFingerprint,
    )) {
      this.warn(`personal library catalog recovered from backup: ${this.paths.backupPath}`);
      await this.repairPrimary(backup.document);
      return backup.document;
    }
    // Same policy as loadUnlocked: valid-but-incompatible documents belong to
    // an older identification/scope strategy and must not block a mutation.
    const primaryUnavailable = primary.kind === "missing"
      || (primary.kind === "valid" && !isCompatible(
        primary.document,
        scopeFingerprint,
        identificationFingerprint,
      ));
    const backupUnavailable = backup.kind === "missing"
      || (backup.kind === "valid" && !isCompatible(
        backup.document,
        scopeFingerprint,
        identificationFingerprint,
      ));
    if (primaryUnavailable && backupUnavailable) {
      return createEmptyPersonalLibraryCatalog(
        scopeFingerprint,
        identificationFingerprint,
        this.now(),
      );
    }
    throw new PersonalLibraryCatalogStoreError(
      `cannot mutate unreadable personal library catalog: ${this.paths.documentPath}`,
      readError(backup) ?? readError(primary),
    );
  }

  private async readDocument(path: string): Promise<DocumentReadResult> {
    let exists: boolean;
    try {
      exists = await this.storage.exists(path);
    } catch (error) {
      this.warn(`unreadable personal library catalog ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    if (!exists) return { kind: "missing" };
    let raw: string;
    try {
      raw = await this.storage.readText(path);
    } catch (error) {
      this.warn(`unreadable personal library catalog ignored: ${path}`, error);
      return { kind: "unreadable", error };
    }
    try {
      const document = decodePersonalLibraryCatalog(JSON.parse(raw));
      if (document) return { kind: "valid", document };
      this.warn(`invalid personal library catalog ignored: ${path}`);
      return { kind: "corrupt" };
    } catch (error) {
      this.warn(`corrupt personal library catalog ignored: ${path}`, error);
      return { kind: "corrupt", error };
    }
  }

  private async repairPrimary(document: PersonalLibraryCatalog): Promise<void> {
    if (!this.storage.writeTextAtomic) {
      throw new PersonalLibraryCatalogStoreError(
        "personal library catalog storage does not support atomic writes",
      );
    }
    try {
      await ensureDirDeep(this.storage, this.paths.directory);
      await this.storage.writeTextAtomic(
        this.paths.documentPath,
        `${JSON.stringify(document, null, 2)}\n`,
      );
    } catch (error) {
      throw new PersonalLibraryCatalogStoreError(
        `failed to repair personal library catalog: ${this.paths.documentPath}`,
        error,
      );
    }
  }

  private async save(document: PersonalLibraryCatalog): Promise<void> {
    await ensureDirDeep(this.storage, this.paths.directory);
    const content = `${JSON.stringify(document, null, 2)}\n`;
    try {
      await replaceWithBackup(this.storage, this.paths, content);
    } catch (error) {
      throw new PersonalLibraryCatalogStoreError(
        `failed to save personal library catalog: ${this.paths.documentPath}`,
        error,
      );
    }
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    let queues = mutationQueues.get(this.storage);
    if (!queues) {
      queues = new Map();
      mutationQueues.set(this.storage, queues);
    }
    const previous = queues.get(this.paths.documentPath) ?? Promise.resolve();
    const next = previous.catch(() => undefined).then(operation);
    const tail = next.then(() => undefined, () => undefined);
    queues.set(this.paths.documentPath, tail);
    void tail.finally(() => {
      if (queues?.get(this.paths.documentPath) === tail) queues.delete(this.paths.documentPath);
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

function decodeFileRecord(value: unknown, path: string): PersonalLibraryFileRecord | null {
  if (!isLogicalRelativePath(path) || !isPlainObject(value) || value.path !== path) return null;
  const common = isFingerprint(value.observationFingerprint) && isIsoDate(value.updatedAt);
  if (!common) return null;
  if (value.status === "ready") {
    if (!isExactObject(value, [
      "path", "status", "observationFingerprint", "paperKey", "arxivId", "updatedAt",
    ])) return null;
    const resources = typeof value.arxivId === "string" ? modernArxivResources(value.arxivId) : null;
    if (!resources || value.arxivId !== resources.id || value.paperKey !== paperKeyFromArxivId(resources.id)) {
      return null;
    }
    return value as unknown as PersonalLibraryFileRecord;
  }
  if (value.status === "unresolved") {
    if (!isExactObject(value, ["path", "status", "observationFingerprint", "reason", "updatedAt"])
      || value.reason !== "unrecognized-filename") return null;
    return value as unknown as PersonalLibraryFileRecord;
  }
  if (value.status === "unrelated") {
    if (!isExactObject(value, ["path", "status", "observationFingerprint", "reason", "updatedAt"])
      || value.reason !== "unsupported-file-type") return null;
    return value as unknown as PersonalLibraryFileRecord;
  }
  if (value.status === "failed") {
    const keys = value.arxivId === undefined
      ? ["path", "status", "observationFingerprint", "reason", "updatedAt"]
      : ["path", "status", "observationFingerprint", "reason", "arxivId", "updatedAt"];
    if (!isExactObject(value, keys)
      || (value.reason !== "metadata-unavailable" && value.reason !== "metadata-fetch-failed")) {
      return null;
    }
    if (value.arxivId !== undefined) {
      const resources = typeof value.arxivId === "string" ? modernArxivResources(value.arxivId) : null;
      if (!resources || resources.id !== value.arxivId) return null;
    }
    return value as unknown as PersonalLibraryFileRecord;
  }
  return null;
}

function decodePaperRecord(value: unknown, paperKey: string): PersonalLibraryPaperRecord | null {
  if (!isExactObject(value, [
    "paperKey", "source", "externalId", "title", "authors", "abstract", "published", "updated",
    "primaryCategory", "categories", "evidenceDepth", "filePaths",
  ])) return null;
  const resources = typeof value.externalId === "string" ? modernArxivResources(value.externalId) : null;
  if (
    value.paperKey !== paperKey
    || value.source !== "arxiv"
    || !resources
    || resources.id !== value.externalId
    || paperKeyFromArxivId(resources.id) !== paperKey
    || !isNonEmptyString(value.title)
    || !isStringArray(value.authors, true)
    || typeof value.abstract !== "string"
    || !isIsoDate(value.published)
    || !isIsoDate(value.updated)
    || !isNonEmptyString(value.primaryCategory)
    || !isUniqueStringArray(value.categories, true)
    || !value.categories.includes(value.primaryCategory)
    || value.evidenceDepth !== "metadata-and-abstract"
    || !isLogicalPathArray(value.filePaths)
  ) return null;
  return value as unknown as PersonalLibraryPaperRecord;
}

function decodeScanSummary(value: unknown): PersonalLibraryScanSummary | null {
  if (!isExactObject(value, ["ready", "unresolved", "unrelated", "failed", "papers", "truncated"])) {
    return null;
  }
  if (![value.ready, value.unresolved, value.unrelated, value.failed, value.papers]
    .every(isNonNegativeSafeInteger) || typeof value.truncated !== "boolean") return null;
  return value as unknown as PersonalLibraryScanSummary;
}

async function replaceWithBackup(
  storage: StorageAdapter,
  paths: PersonalLibraryCatalogPaths,
  content: string,
): Promise<void> {
  if (!storage.writeTextAtomic) {
    throw new PersonalLibraryCatalogStoreError(
      "personal library catalog storage does not support atomic writes",
    );
  }
  let previous: string | null = null;
  if (await storage.exists(paths.documentPath)) {
    const raw = await storage.readText(paths.documentPath);
    if (decodeRawCatalog(raw)) previous = raw;
  }
  let recoveryContent = previous;
  if (recoveryContent === null && await storage.exists(paths.backupPath)) {
    const raw = await storage.readText(paths.backupPath);
    if (decodeRawCatalog(raw)) recoveryContent = raw;
  }
  if (previous !== null) {
    await storage.writeTextAtomic(paths.backupPath, previous);
  }
  try {
    await storage.writeTextAtomic(paths.documentPath, content);
    if (previous === null && recoveryContent === null) {
      await storage.writeTextAtomic(paths.backupPath, content).catch(() => undefined);
    }
  } catch (error) {
    if (recoveryContent !== null) {
      await storage.writeTextAtomic(paths.documentPath, recoveryContent);
    }
    throw error;
  }
}

function semanticContent(document: PersonalLibraryCatalog): string {
  return JSON.stringify({
    scopeFingerprint: document.scopeFingerprint,
    identificationFingerprint: document.identificationFingerprint,
    lastScan: document.lastScan,
    files: sortRecord(document.files),
    papers: sortRecord(document.papers),
  });
}

function sortRecord<T>(record: Record<string, T>): Record<string, T> {
  return Object.fromEntries(Object.entries(record).sort(([left], [right]) => left.localeCompare(right)));
}

function isCompatible(
  document: PersonalLibraryCatalog,
  scopeFingerprint: string,
  identificationFingerprint: string,
): boolean {
  return document.scopeFingerprint === requireFingerprint("scopeFingerprint", scopeFingerprint)
    && document.identificationFingerprint === requireFingerprint(
      "identificationFingerprint",
      identificationFingerprint,
    );
}

function normalizeExtension(extension: string): string {
  const normalized = extension.trim().toLowerCase();
  if (!/^\.[a-z0-9]+$/.test(normalized)) {
    throw new PersonalLibraryCatalogStoreError(`invalid eligible extension: ${extension}`);
  }
  return normalized;
}

function requireNonEmpty(name: string, value: string): string {
  if (!value.trim()) throw new PersonalLibraryCatalogStoreError(`${name} must be non-empty`);
  return value;
}

function requireFingerprint(name: string, value: string): string {
  if (!isFingerprint(value)) {
    throw new PersonalLibraryCatalogStoreError(`${name} must be a SHA-256 fingerprint`);
  }
  return value;
}

function isFingerprint(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}

function isLogicalRelativePath(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && !value.includes("\\")
    && !value.includes("\0")
    && !value.startsWith("/")
    && !/^[A-Za-z]:/.test(value)
    && value.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== "..");
}

function isLogicalPathArray(value: unknown): value is string[] {
  return Array.isArray(value)
    && value.length > 0
    && value.every(isLogicalRelativePath)
    && new Set(value).size === value.length
    && [...value].sort().every((path, index) => path === value[index]);
}

function isStringArray(value: unknown, requireNonEmptyItems: boolean): value is string[] {
  return Array.isArray(value)
    && value.every((item) => typeof item === "string" && (!requireNonEmptyItems || item.trim().length > 0));
}

function isUniqueStringArray(value: unknown, requireNonEmptyItems: boolean): value is string[] {
  return isStringArray(value, requireNonEmptyItems) && new Set(value).size === value.length;
}

function isIsoDate(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
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
  return actual.length === expected.length
    && actual.every((key, index) => key === expected[index]);
}

function decodeRawCatalog(raw: string): PersonalLibraryCatalog | null {
  try {
    return decodePersonalLibraryCatalog(JSON.parse(raw));
  } catch {
    return null;
  }
}

function cloneCatalog(document: PersonalLibraryCatalog): PersonalLibraryCatalog {
  return JSON.parse(JSON.stringify(document)) as PersonalLibraryCatalog;
}

function defineRecordEntry<T>(record: Record<string, T>, key: string, value: T): void {
  Object.defineProperty(record, key, {
    value,
    enumerable: true,
    configurable: true,
    writable: true,
  });
}

function readError(result: DocumentReadResult): unknown {
  return result.kind === "corrupt" || result.kind === "unreadable" ? result.error : undefined;
}

async function writePrivateText(
  storage: StorageAdapter,
  path: string,
  content: string,
): Promise<void> {
  if (storage.writeTextWithMode) {
    await storage.writeTextWithMode(path, content, 0o600);
  } else {
    await storage.writeText(path, content);
  }
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
