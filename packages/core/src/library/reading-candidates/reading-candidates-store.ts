/**
 * File-backed CAS store for reading candidates.
 *
 * A bypass store in the same spirit as the incremental suggestions store:
 * sharded by the personal library scope / identification fingerprints,
 * primary + backup with atomic promotion, expectedRevision CAS with exact
 * semantic replay idempotency, and strict decode on every read. The document
 * is replaceable derived state — deleting it only loses saved candidates, no
 * authoritative research record.
 */

import type { StorageAdapter } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import { derivePaperInboxPaths } from "../../services/paper-index";
import {
  decodeReadingCandidatesDocument,
  emptyReadingCandidatesDocument,
  type ReadingCandidatesDocument,
} from "./reading-candidates";

export interface ReadingCandidatesDocumentPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface ReadingCandidatesStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export type ReadingCandidatesStoreErrorCode =
  | "invalid"
  | "stale"
  | "incompatible"
  | "corrupt-or-unreadable"
  | "atomic-write-unsupported"
  | "repair-failed"
  | "save-failed";

interface RevisionConflictFields {
  expectedRevision?: number | null;
  currentRevision?: number | null;
}

export class ReadingCandidatesStoreError extends Error {
  readonly expectedRevision?: number | null;
  readonly currentRevision?: number | null;

  constructor(
    message: string,
    readonly code: ReadingCandidatesStoreErrorCode,
    options: { cause?: unknown } & RevisionConflictFields = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "ReadingCandidatesStoreError";
    this.expectedRevision = options.expectedRevision;
    this.currentRevision = options.currentRevision;
  }
}

/**
 * Derive reading-candidate paths sharded by scope/identification fingerprints,
 * mirroring the interest-profile and knowledge-base layout:
 * `<indexDir>/personal-library-reading-candidates/<scopeHex>/<idHex>/`.
 */
export function deriveReadingCandidatesPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  scopeFingerprint: string,
  identificationFingerprint: string,
): ReadingCandidatesDocumentPaths {
  const scopeHex = fingerprintHex("scopeFingerprint", scopeFingerprint);
  const identificationHex = fingerprintHex("identificationFingerprint", identificationFingerprint);
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(
    `${indexDir}/personal-library-reading-candidates/${scopeHex}/${identificationHex}`,
  );
  return documentPaths(storage, directory);
}

// Runtime-local serialization is guaranteed only for the same StorageAdapter object and
// normalized document path. Cross-adapter and multi-runtime locking belongs to host composition.
const documentQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

export class ReadingCandidatesStore {
  readonly paths: ReadingCandidatesDocumentPaths;
  private readonly scopeFingerprint: string;
  private readonly identificationFingerprint: string;

  constructor(
    private readonly storage: StorageAdapter,
    output: OutputSettings,
    scopeFingerprint: string,
    identificationFingerprint: string,
    private readonly options: ReadingCandidatesStoreOptions = {},
  ) {
    validateBoundFingerprints(scopeFingerprint, identificationFingerprint);
    this.scopeFingerprint = scopeFingerprint;
    this.identificationFingerprint = identificationFingerprint;
    this.paths = deriveReadingCandidatesPaths(
      storage,
      output,
      scopeFingerprint,
      identificationFingerprint,
    );
  }

  load(): Promise<ReadingCandidatesDocument> {
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDurableDocument();
      return clone(loaded?.document ?? this.emptyDocument());
    });
  }

  replace(
    next: ReadingCandidatesDocument,
    expectedRevision: number,
  ): Promise<ReadingCandidatesDocument> {
    const validated = decodeReadingCandidatesDocument(next);
    if (!validated || !this.matchesBoundIdentity(validated)) {
      return Promise.reject(error("invalid",
        "cannot persist invalid or identity-mismatched reading candidates document"));
    }
    if (!isNonNegativeSafeInteger(expectedRevision)) {
      return Promise.reject(error("invalid",
        "reading candidates expected revision must be a non-negative safe integer"));
    }
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDurableDocument();
      const current = loaded?.document ?? this.emptyDocument();
      const candidate = clone(validated);
      candidate.revision = current.revision;

      // Check replay before CAS: an exact requested semantic state may be the result of a commit
      // whose success response was lost. Any changed stale state remains a conflict.
      if (loaded && semanticDocument(candidate) === semanticDocument(current)) return clone(current);
      if (expectedRevision !== current.revision) {
        throw stale(expectedRevision, current.revision);
      }
      if (loaded === null && semanticDocument(candidate) === semanticDocument(current)) {
        return clone(current);
      }
      if (current.revision === Number.MAX_SAFE_INTEGER) {
        throw error("invalid", "reading candidates document revision is exhausted");
      }
      candidate.revision = loaded === null ? 1 : current.revision + 1;
      candidate.updatedAt = latestTimestamp(this.validNow(), current);
      if (!decodeReadingCandidatesDocument(candidate)) {
        throw error("invalid", "cannot persist invalid reading candidates document");
      }
      await saveDocument(this.storage, this.paths, candidate, loaded?.raw ?? null);
      return clone(candidate);
    });
  }

  private loadDurableDocument(): Promise<{ document: ReadingCandidatesDocument; raw: string } | null> {
    return loadDurableDocument({
      storage: this.storage,
      paths: this.paths,
      scopeFingerprint: this.scopeFingerprint,
      identificationFingerprint: this.identificationFingerprint,
      onWarning: this.options.onWarning,
    });
  }

  private emptyDocument(): ReadingCandidatesDocument {
    return emptyReadingCandidatesDocument(
      this.scopeFingerprint,
      this.identificationFingerprint,
      this.validNow().toISOString(),
    );
  }

  private matchesBoundIdentity(document: ReadingCandidatesDocument): boolean {
    return document.scopeFingerprint === this.scopeFingerprint
      && document.identificationFingerprint === this.identificationFingerprint;
  }

  private validNow(): Date {
    const now = this.options.now?.() ?? new Date();
    if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
      throw error("invalid", "reading candidates clock returned an invalid date");
    }
    return now;
  }
}

type Decoder<T> = (value: unknown) => T | null;
type ReadResult<T> =
  | { kind: "missing" }
  | { kind: "valid"; document: T; raw: string }
  | { kind: "corrupt"; raw?: string; error?: unknown }
  | { kind: "unreadable"; error: unknown };

function documentPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  directory: string,
): ReadingCandidatesDocumentPaths {
  const normalizedDirectory = storage.normalizePath(directory);
  const documentPath = storage.normalizePath(`${normalizedDirectory}/reading-candidates.json`);
  return {
    directory: normalizedDirectory,
    documentPath,
    backupPath: storage.normalizePath(`${documentPath}.backup`),
  };
}

async function loadDurableDocument(input: {
  storage: StorageAdapter;
  paths: ReadingCandidatesDocumentPaths;
  scopeFingerprint: string;
  identificationFingerprint: string;
  onWarning?: (message: string, error?: unknown) => void;
}): Promise<{ document: ReadingCandidatesDocument; raw: string } | null> {
  const primary = await readDocument(input.storage, input.paths.documentPath, decodeReadingCandidatesDocument);
  if (primary.kind === "valid") {
    if (!compatible(primary.document, input)) {
      throw error("incompatible",
        `incompatible reading candidates document: ${input.paths.documentPath}`);
    }
    return { document: primary.document, raw: primary.raw };
  }
  const backup = await readDocument(input.storage, input.paths.backupPath, decodeReadingCandidatesDocument);
  if (backup.kind === "valid") {
    if (!compatible(backup.document, input)) {
      throw error("incompatible",
        `incompatible reading candidates document backup: ${input.paths.backupPath}`);
    }
    input.onWarning?.(
      `reading candidates document recovered from backup: ${input.paths.backupPath}`,
      readCause(primary),
    );
    await repairPrimary(input.storage, input.paths, backup.raw);
    return { document: backup.document, raw: backup.raw };
  }
  if (primary.kind === "missing" && backup.kind === "missing") return null;
  throw error("corrupt-or-unreadable",
    `corrupt or unreadable reading candidates document: ${input.paths.documentPath}`,
    { cause: readCause(backup) ?? readCause(primary) });
}

async function readDocument<T>(storage: StorageAdapter, path: string, decoder: Decoder<T>): Promise<ReadResult<T>> {
  try {
    if (!(await storage.exists(path))) return { kind: "missing" };
  } catch (caught) {
    return { kind: "unreadable", error: caught };
  }
  let raw: string;
  try {
    raw = await storage.readText(path);
  } catch (caught) {
    return { kind: "unreadable", error: caught };
  }
  try {
    const document = decoder(JSON.parse(raw));
    return document ? { kind: "valid", document, raw } : { kind: "corrupt", raw };
  } catch (caught) {
    return { kind: "corrupt", error: caught };
  }
}

async function repairPrimary(
  storage: StorageAdapter,
  paths: ReadingCandidatesDocumentPaths,
  raw: string,
): Promise<void> {
  requireAtomic(storage);
  try {
    await ensureDirDeep(storage, paths.directory);
    await storage.writeTextAtomic!(paths.documentPath, canonicalRaw(raw));
  } catch (caught) {
    throw error("repair-failed", `failed to repair reading candidates document: ${paths.documentPath}`,
      { cause: caught });
  }
}

async function saveDocument(
  storage: StorageAdapter,
  paths: ReadingCandidatesDocumentPaths,
  document: ReadingCandidatesDocument,
  priorPrimaryRaw: string | null,
): Promise<void> {
  requireAtomic(storage);
  const content = `${JSON.stringify(document, null, 2)}\n`;
  try {
    await ensureDirDeep(storage, paths.directory);
    await storage.writeTextAtomic!(
      paths.backupPath,
      priorPrimaryRaw === null ? content : canonicalRaw(priorPrimaryRaw),
    );
    // Atomic promotion is commit-wins. Never blindly overwrite a possibly committed primary.
    await storage.writeTextAtomic!(paths.documentPath, content);
  } catch (caught) {
    throw error("save-failed", `failed to save reading candidates document: ${paths.documentPath}`,
      { cause: caught });
  }
}

function requireAtomic(storage: StorageAdapter): void {
  if (!storage.writeTextAtomic) {
    throw error("atomic-write-unsupported",
      "reading candidates storage does not support atomic writes");
  }
}

function enqueue<T>(storage: StorageAdapter, normalizedPath: string, operation: () => Promise<T>): Promise<T> {
  let queues = documentQueues.get(storage);
  if (!queues) {
    queues = new Map();
    documentQueues.set(storage, queues);
  }
  const previous = queues.get(normalizedPath) ?? Promise.resolve();
  const next = previous.catch(() => undefined).then(operation);
  const tail = next.then(() => undefined, () => undefined);
  queues.set(normalizedPath, tail);
  void tail.finally(() => {
    if (queues?.get(normalizedPath) === tail) queues.delete(normalizedPath);
  });
  return next;
}

function validateBoundFingerprints(scope: string, identification: string): void {
  fingerprintHex("scopeFingerprint", scope);
  fingerprintHex("identificationFingerprint", identification);
}

function fingerprintHex(name: string, value: string): string {
  const match = /^sha256:([a-f0-9]{64})$/.exec(value);
  if (!match) throw error("invalid", `reading candidates ${name} must be a SHA-256 fingerprint`);
  return match[1]!;
}

function latestTimestamp(now: Date, current: ReadingCandidatesDocument): string {
  return new Date(Math.max(now.getTime(), Date.parse(current.updatedAt))).toISOString();
}

interface Fingerprinted { scopeFingerprint: string; identificationFingerprint: string }

function compatible(document: Fingerprinted, input: Fingerprinted): boolean {
  return document.scopeFingerprint === input.scopeFingerprint
    && document.identificationFingerprint === input.identificationFingerprint;
}

/** Canonical semantic snapshot: revision and updatedAt are store-owned, not content. */
function semanticDocument(document: ReadingCandidatesDocument): string {
  const { revision: _revision, updatedAt: _updatedAt, ...semantic } = document;
  return canonicalJson(semantic);
}

/** Key-sorted canonical JSON so record insertion order never breaks idempotency. */
function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (typeof value === "object" && value !== null) {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(value);
}

function stale(expectedRevision: number | null, currentRevision: number | null): Error {
  return error(
    "stale",
    `stale reading candidates revision: expected ${String(expectedRevision)}, current ${String(currentRevision)}`,
    { expectedRevision, currentRevision },
  );
}

function error(
  code: ReadingCandidatesStoreErrorCode,
  message: string,
  options: { cause?: unknown } & RevisionConflictFields = {},
): ReadingCandidatesStoreError {
  return new ReadingCandidatesStoreError(message, code, options);
}

function readCause<T>(result: ReadResult<T>): unknown {
  return result.kind === "corrupt" || result.kind === "unreadable" ? result.error : undefined;
}

function canonicalRaw(raw: string): string {
  return `${JSON.stringify(JSON.parse(raw), null, 2)}\n`;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function isNonNegativeSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

async function ensureDirDeep(storage: StorageAdapter, directory: string): Promise<void> {
  const parts = storage.normalizePath(directory).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) {
      await storage.mkdir(current);
    }
  }
}
