/**
 * Incremental suggestions document model and file-backed store.
 *
 * A direction diff run produces `DirectionDiffSuggestion` entries that are
 * persisted here before any mutation is applied to the confirmed interest
 * profile. The document is a bypass store in the same spirit as the
 * knowledge base: it is sharded by the same scope / identification
 * fingerprints that bind the personal library catalog, can be deleted and
 * rebuilt, and concurrency follows the interest-profile store pattern
 * (expectedRevision CAS with exact semantic replay idempotency).
 *
 * The persisted suggestions are untrusted LLM-shaped data, so the strict
 * decoder re-applies the T3 per-suggestion validation rules at the document
 * level: kind enum, non-empty unique paper keys, bounded reason text,
 * exactly two merge direction ids, and the cross-suggestion conflict rules
 * (a paper appears in only one suggestion; a split target never also
 * participates in a merge). Direction existence cannot be checked here —
 * that is a profile-level concern resolved when suggestions are applied.
 * Suggestions are stored in canonical code-unit order.
 */

import type { StorageAdapter } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import { derivePaperInboxPaths } from "../../services/paper-index";
import {
  PERSONAL_LIBRARY_DIRECTION_DIFF_MAX_REASON_LENGTH,
  type DirectionDiffSuggestion,
} from "./diff-suggestions";
import { PERSONAL_LIBRARY_MAX_ID_LENGTH } from "../personal-library-interest-profile";

export const INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION = 1 as const;

export interface IncrementalSuggestionsDocument {
  schemaVersion: typeof INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION;
  revision: number;
  scopeFingerprint: string;
  identificationFingerprint: string;
  updatedAt: string;
  suggestions: DirectionDiffSuggestion[];
}

export interface IncrementalSuggestionsDocumentPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface IncrementalSuggestionsStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export type IncrementalSuggestionsStoreErrorCode =
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

export class IncrementalSuggestionsStoreError extends Error {
  readonly expectedRevision?: number | null;
  readonly currentRevision?: number | null;

  constructor(
    message: string,
    readonly code: IncrementalSuggestionsStoreErrorCode,
    options: { cause?: unknown } & RevisionConflictFields = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "IncrementalSuggestionsStoreError";
    this.expectedRevision = options.expectedRevision;
    this.currentRevision = options.currentRevision;
  }
}

/**
 * Shape-only strict decode of one suggestion: no direction existence check
 * (the document carries no directions) and no cross-suggestion checks. Paper
 * keys are code-unit sorted and unique; merge ids are code-unit sorted and
 * distinct; direction ids follow the profile's opaque id rules.
 */
export function decodeIncrementalSuggestion(value: unknown): DirectionDiffSuggestion | null {
  if (!isPlainObject(value) || typeof value.kind !== "string") return null;
  switch (value.kind) {
    case "attach": {
      if (!isExactObject(value, ["kind", "directionId", "paperKeys", "reason"])) return null;
      if (!isOpaqueId(value.directionId)) return null;
      const paperKeys = decodePaperKeys(value.paperKeys);
      if (!paperKeys) return null;
      if (!isValidReason(value.reason)) return null;
      return { kind: "attach", directionId: value.directionId, paperKeys, reason: value.reason };
    }
    case "new": {
      if (!isExactObject(value, ["kind", "paperKeys", "reason"])) return null;
      const paperKeys = decodePaperKeys(value.paperKeys);
      if (!paperKeys) return null;
      if (!isValidReason(value.reason)) return null;
      return { kind: "new", paperKeys, reason: value.reason };
    }
    case "split": {
      if (!isExactObject(value, ["kind", "directionId", "paperKeys", "reason"])) return null;
      if (!isOpaqueId(value.directionId)) return null;
      const paperKeys = decodePaperKeys(value.paperKeys);
      if (!paperKeys) return null;
      if (!isValidReason(value.reason)) return null;
      return { kind: "split", directionId: value.directionId, paperKeys, reason: value.reason };
    }
    case "merge": {
      if (!isExactObject(value, ["kind", "directionIds", "reason"])) return null;
      const ids = value.directionIds;
      if (!Array.isArray(ids) || ids.length !== 2
        || typeof ids[0] !== "string" || typeof ids[1] !== "string"
        || !isOpaqueId(ids[0]) || !isOpaqueId(ids[1])
        || ids[0] === ids[1]
        || codeUnitCompare(ids[0], ids[1]) >= 0) return null;
      if (!isValidReason(value.reason)) return null;
      return { kind: "merge", directionIds: [ids[0], ids[1]], reason: value.reason };
    }
    default:
      return null;
  }
}

export function decodeIncrementalSuggestionsDocument(
  value: unknown,
): IncrementalSuggestionsDocument | null {
  if (!isExactObject(value, [
    "schemaVersion", "revision", "scopeFingerprint", "identificationFingerprint", "updatedAt",
    "suggestions",
  ]) || value.schemaVersion !== INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION
    || !isNonNegativeSafeInteger(value.revision)
    || !isFingerprint(value.scopeFingerprint)
    || !isFingerprint(value.identificationFingerprint)
    || !isCanonicalTimestamp(value.updatedAt)
    || !Array.isArray(value.suggestions)) return null;

  const suggestions: DirectionDiffSuggestion[] = [];
  for (const raw of value.suggestions) {
    const suggestion = decodeIncrementalSuggestion(raw);
    if (!suggestion) return null;
    suggestions.push(suggestion);
  }
  // Persisted suggestions are canonical: strictly code-unit ordered.
  for (let index = 1; index < suggestions.length; index += 1) {
    if (compareSuggestions(suggestions[index - 1]!, suggestions[index]!) >= 0) return null;
  }
  // Cross-suggestion conflicts mirror the T3 validation rules: a paper may
  // appear in only one suggestion, and a direction may not be both a split
  // target and a merge participant.
  const claimedPapers = new Set<string>();
  for (const suggestion of suggestions) {
    if (suggestion.kind === "merge") continue;
    for (const paperKey of suggestion.paperKeys) {
      if (claimedPapers.has(paperKey)) return null;
      claimedPapers.add(paperKey);
    }
  }
  const splitTargets = new Set(
    suggestions.filter((suggestion) => suggestion.kind === "split").map((suggestion) => suggestion.directionId),
  );
  for (const suggestion of suggestions) {
    if (suggestion.kind === "merge"
      && suggestion.directionIds.some((directionId) => splitTargets.has(directionId))) {
      return null;
    }
  }

  return {
    schemaVersion: INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION,
    revision: value.revision,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    updatedAt: value.updatedAt,
    suggestions,
  };
}

export function createEmptyIncrementalSuggestionsDocument(
  scopeFingerprint: string,
  identificationFingerprint: string,
  now: Date = new Date(),
): IncrementalSuggestionsDocument {
  const document: IncrementalSuggestionsDocument = {
    schemaVersion: INCREMENTAL_SUGGESTIONS_SCHEMA_VERSION,
    revision: 0,
    scopeFingerprint,
    identificationFingerprint,
    updatedAt: now.toISOString(),
    suggestions: [],
  };
  const decoded = decodeIncrementalSuggestionsDocument(document);
  if (!decoded) throw new TypeError("cannot create empty incremental suggestions document");
  return decoded;
}

/**
 * Derive incremental suggestions paths sharded by scope/identification
 * fingerprints, mirroring the interest-profile and knowledge-base layout:
 * `<indexDir>/personal-library-incremental-suggestions/<scopeHex>/<idHex>/`.
 */
export function deriveIncrementalSuggestionsPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  scopeFingerprint: string,
  identificationFingerprint: string,
): IncrementalSuggestionsDocumentPaths {
  const scopeHex = fingerprintHex("scopeFingerprint", scopeFingerprint);
  const identificationHex = fingerprintHex("identificationFingerprint", identificationFingerprint);
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(
    `${indexDir}/personal-library-incremental-suggestions/${scopeHex}/${identificationHex}`,
  );
  return documentPaths(storage, directory);
}

// Runtime-local serialization is guaranteed only for the same StorageAdapter object and
// normalized document path. Cross-adapter and multi-runtime locking belongs to host composition.
const documentQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

export class IncrementalSuggestionsStore {
  readonly paths: IncrementalSuggestionsDocumentPaths;
  private readonly scopeFingerprint: string;
  private readonly identificationFingerprint: string;

  constructor(
    private readonly storage: StorageAdapter,
    output: OutputSettings,
    scopeFingerprint: string,
    identificationFingerprint: string,
    private readonly options: IncrementalSuggestionsStoreOptions = {},
  ) {
    validateBoundFingerprints(scopeFingerprint, identificationFingerprint);
    this.scopeFingerprint = scopeFingerprint;
    this.identificationFingerprint = identificationFingerprint;
    this.paths = deriveIncrementalSuggestionsPaths(
      storage,
      output,
      scopeFingerprint,
      identificationFingerprint,
    );
  }

  load(): Promise<IncrementalSuggestionsDocument> {
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDurableDocument();
      return clone(loaded?.document ?? this.emptyDocument());
    });
  }

  replace(
    next: IncrementalSuggestionsDocument,
    expectedRevision: number,
  ): Promise<IncrementalSuggestionsDocument> {
    const validated = decodeIncrementalSuggestionsDocument(next);
    if (!validated || !this.matchesBoundIdentity(validated)) {
      return Promise.reject(error("invalid",
        "cannot persist invalid or identity-mismatched incremental suggestions document"));
    }
    if (!isNonNegativeSafeInteger(expectedRevision)) {
      return Promise.reject(error("invalid",
        "incremental suggestions expected revision must be a non-negative safe integer"));
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
        throw error("invalid", "incremental suggestions document revision is exhausted");
      }
      candidate.revision = loaded === null ? 1 : current.revision + 1;
      candidate.updatedAt = latestTimestamp(this.validNow(), current);
      if (!decodeIncrementalSuggestionsDocument(candidate)) {
        throw error("invalid", "cannot persist invalid incremental suggestions document");
      }
      await saveDocument(this.storage, this.paths, candidate, loaded?.raw ?? null);
      return clone(candidate);
    });
  }

  private loadDurableDocument(): Promise<{ document: IncrementalSuggestionsDocument; raw: string } | null> {
    return loadDurableDocument({
      storage: this.storage,
      paths: this.paths,
      scopeFingerprint: this.scopeFingerprint,
      identificationFingerprint: this.identificationFingerprint,
      onWarning: this.options.onWarning,
    });
  }

  private emptyDocument(): IncrementalSuggestionsDocument {
    return createEmptyIncrementalSuggestionsDocument(
      this.scopeFingerprint,
      this.identificationFingerprint,
      this.validNow(),
    );
  }

  private matchesBoundIdentity(document: IncrementalSuggestionsDocument): boolean {
    return document.scopeFingerprint === this.scopeFingerprint
      && document.identificationFingerprint === this.identificationFingerprint;
  }

  private validNow(): Date {
    const now = this.options.now?.() ?? new Date();
    if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
      throw error("invalid", "incremental suggestions clock returned an invalid date");
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
): IncrementalSuggestionsDocumentPaths {
  const normalizedDirectory = storage.normalizePath(directory);
  const documentPath = storage.normalizePath(`${normalizedDirectory}/incremental-suggestions.json`);
  return {
    directory: normalizedDirectory,
    documentPath,
    backupPath: storage.normalizePath(`${documentPath}.backup`),
  };
}

async function loadDurableDocument(input: {
  storage: StorageAdapter;
  paths: IncrementalSuggestionsDocumentPaths;
  scopeFingerprint: string;
  identificationFingerprint: string;
  onWarning?: (message: string, error?: unknown) => void;
}): Promise<{ document: IncrementalSuggestionsDocument; raw: string } | null> {
  const primary = await readDocument(input.storage, input.paths.documentPath, decodeIncrementalSuggestionsDocument);
  if (primary.kind === "valid") {
    if (!compatible(primary.document, input)) {
      throw error("incompatible",
        `incompatible incremental suggestions document: ${input.paths.documentPath}`);
    }
    return { document: primary.document, raw: primary.raw };
  }
  const backup = await readDocument(input.storage, input.paths.backupPath, decodeIncrementalSuggestionsDocument);
  if (backup.kind === "valid") {
    if (!compatible(backup.document, input)) {
      throw error("incompatible",
        `incompatible incremental suggestions document backup: ${input.paths.backupPath}`);
    }
    input.onWarning?.(
      `incremental suggestions document recovered from backup: ${input.paths.backupPath}`,
      readCause(primary),
    );
    await repairPrimary(input.storage, input.paths, backup.raw);
    return { document: backup.document, raw: backup.raw };
  }
  if (primary.kind === "missing" && backup.kind === "missing") return null;
  throw error("corrupt-or-unreadable",
    `corrupt or unreadable incremental suggestions document: ${input.paths.documentPath}`,
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
  paths: IncrementalSuggestionsDocumentPaths,
  raw: string,
): Promise<void> {
  requireAtomic(storage);
  try {
    await ensureDirDeep(storage, paths.directory);
    await storage.writeTextAtomic!(paths.documentPath, canonicalRaw(raw));
  } catch (caught) {
    throw error("repair-failed", `failed to repair incremental suggestions document: ${paths.documentPath}`,
      { cause: caught });
  }
}

async function saveDocument(
  storage: StorageAdapter,
  paths: IncrementalSuggestionsDocumentPaths,
  document: IncrementalSuggestionsDocument,
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
    throw error("save-failed", `failed to save incremental suggestions document: ${paths.documentPath}`,
      { cause: caught });
  }
}

function requireAtomic(storage: StorageAdapter): void {
  if (!storage.writeTextAtomic) {
    throw error("atomic-write-unsupported",
      "incremental suggestions storage does not support atomic writes");
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
  if (!match) throw error("invalid", `incremental suggestions ${name} must be a SHA-256 fingerprint`);
  return match[1]!;
}

function latestTimestamp(now: Date, current: IncrementalSuggestionsDocument): string {
  return new Date(Math.max(now.getTime(), Date.parse(current.updatedAt))).toISOString();
}

interface Fingerprinted { scopeFingerprint: string; identificationFingerprint: string }

function compatible(document: Fingerprinted, input: Fingerprinted): boolean {
  return document.scopeFingerprint === input.scopeFingerprint
    && document.identificationFingerprint === input.identificationFingerprint;
}

/** Canonical semantic snapshot: revision and updatedAt are store-owned, not content. */
function semanticDocument(document: IncrementalSuggestionsDocument): string {
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
    `stale incremental suggestions revision: expected ${String(expectedRevision)}, current ${String(currentRevision)}`,
    { expectedRevision, currentRevision },
  );
}

function error(
  code: IncrementalSuggestionsStoreErrorCode,
  message: string,
  options: { cause?: unknown } & RevisionConflictFields = {},
): IncrementalSuggestionsStoreError {
  return new IncrementalSuggestionsStoreError(message, code, options);
}

function readCause<T>(result: ReadResult<T>): unknown {
  return result.kind === "corrupt" || result.kind === "unreadable" ? result.error : undefined;
}

function canonicalRaw(raw: string): string {
  return `${JSON.stringify(JSON.parse(raw), null, 2)}\n`;
}

/** Paper keys must be non-empty, code-unit sorted, unique strings. */
function decodePaperKeys(value: unknown): string[] | null {
  if (!Array.isArray(value) || value.length === 0
    || !value.every((key: unknown) => typeof key === "string")
    || !isStrictlyOrderedUnique(value)) {
    return null;
  }
  return [...value];
}

function isValidReason(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && value.length <= PERSONAL_LIBRARY_DIRECTION_DIFF_MAX_REASON_LENGTH
    && value.trim() === value
    && !/[\u0000-\u001F\u007F]/.test(value);
}

const SUGGESTION_KIND_ORDER: Readonly<Record<DirectionDiffSuggestion["kind"], number>> = {
  attach: 0,
  merge: 1,
  new: 2,
  split: 3,
};

function compareSuggestions(left: DirectionDiffSuggestion, right: DirectionDiffSuggestion): number {
  const leftKey = suggestionSortKey(left);
  const rightKey = suggestionSortKey(right);
  for (let index = 0; index < leftKey.length; index += 1) {
    const diff = codeUnitCompare(leftKey[index]!, rightKey[index]!);
    if (diff !== 0) return diff;
  }
  return 0;
}

function suggestionSortKey(suggestion: DirectionDiffSuggestion): string[] {
  switch (suggestion.kind) {
    case "attach":
      return [String(SUGGESTION_KIND_ORDER.attach), suggestion.directionId, suggestion.paperKeys[0] ?? ""];
    case "merge":
      return [String(SUGGESTION_KIND_ORDER.merge), suggestion.directionIds[0], suggestion.directionIds[1]];
    case "new":
      return [String(SUGGESTION_KIND_ORDER.new), suggestion.paperKeys[0] ?? "", ""];
    case "split":
      return [String(SUGGESTION_KIND_ORDER.split), suggestion.directionId, suggestion.paperKeys[0] ?? ""];
  }
}

function isOpaqueId(value: unknown): value is string {
  return typeof value === "string"
    && value.length >= 1
    && value.length <= PERSONAL_LIBRARY_MAX_ID_LENGTH
    && /^[A-Za-z0-9._~-]+$/.test(value);
}

function isFingerprint(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}

function isCanonicalTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isStrictlyOrderedUnique(value: readonly string[]): boolean {
  return value.every((item, index) => index === 0 || codeUnitCompare(value[index - 1]!, item) < 0);
}

function codeUnitCompare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isExactObject(value: unknown, keys: readonly string[]): value is Record<string, any> {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort(codeUnitCompare);
  const expected = [...keys].sort(codeUnitCompare);
  return actual.length === expected.length
    && actual.every((key, index) => key === expected[index]);
}

function clone(value: IncrementalSuggestionsDocument): IncrementalSuggestionsDocument {
  return JSON.parse(JSON.stringify(value)) as IncrementalSuggestionsDocument;
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}
