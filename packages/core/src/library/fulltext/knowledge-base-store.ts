/**
 * Full-text knowledge base bypass store (side catalog) — file-backed
 * implementation of `FullTextKnowledgeBaseStore`.
 *
 * The knowledge base never lives inside papers.json: it can be deleted and
 * rebuilt, and its paths are sharded by the same scope / identification
 * fingerprints that bind the personal library catalog. Concurrency follows
 * the interest-profile store pattern (expectedRevision CAS on the manifest,
 * with exact semantic replay idempotency), NOT the catalog's whole-document
 * replace: the manifest is the authoritative index, per-paper files are
 * derived, content-addressed data that may be rewritten idempotently.
 *
 * Design decisions:
 * - Manifest writes rotate a `.backup` before atomic promotion of the primary
 *   (commit-wins; the backup always holds the previous primary content).
 * - `modelId`/`dimension` are global to the knowledge base: a different model
 *   indexing the same scope/identification produces different content. This
 *   stage treats a model switch under the same scope/id as delete-and-rebuild:
 *   `replaceManifest` never migrates; when the loaded manifest already has
 *   papers and `next.modelId` differs, the replacement is rejected as
 *   `invalid` and the caller must call `removeAll()` first (or the knowledge
 *   base directory is deleted externally). An empty knowledge base (no
 *   papers) may adopt any model id.
 * - `removeAll` removes every file under `papers/` (recursively via
 *   `storage.list` when available, otherwise enumerated through the
 *   manifest's records) plus the manifest and its backup. `StorageAdapter`
 *   has no rmdir, so empty directories are removed best-effort with
 *   `storage.remove` and failures are tolerated; files are authoritative.
 * - Runtime-local serialization is per (StorageAdapter object, normalized path):
 *   manifest operations queue on the manifest path, per-paper operations on
 *   their own file path. Cross-path races (e.g. `savePaper` vs `removeAll`)
 *   are not serialized — host composition owns cross-path locking, matching
 *   the interest-profile store contract.
 */

import type { StorageAdapter, StorageEntry } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  createFullTextKnowledgeBasePaperPath,
  decodeFullTextKnowledgeBaseManifest,
  decodeFullTextPaperDocument,
  deriveFullTextKnowledgeBasePaths,
  serializeFullTextPaperDocument,
  type FullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBasePaths,
  type FullTextKnowledgeBaseStore,
  type FullTextKnowledgeBaseStorePathsOptions,
  type FullTextPaperDocument,
} from "./knowledge-base";

export type FullTextKnowledgeBaseStoreErrorCode =
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

export class FullTextKnowledgeBaseStoreError extends Error {
  readonly expectedRevision?: number | null;
  readonly currentRevision?: number | null;

  constructor(
    message: string,
    readonly code: FullTextKnowledgeBaseStoreErrorCode,
    options: { cause?: unknown } & RevisionConflictFields = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "FullTextKnowledgeBaseStoreError";
    this.expectedRevision = options.expectedRevision;
    this.currentRevision = options.currentRevision;
  }
}

type Decoder<T> = (value: unknown) => T | null;
type ReadResult<T> =
  | { kind: "missing" }
  | { kind: "valid"; document: T; raw: string }
  | { kind: "corrupt"; raw?: string; error?: unknown }
  | { kind: "incompatible"; schemaVersion: number; raw: string }
  | { kind: "unreadable"; error: unknown };

// Runtime-local serialization is guaranteed only for the same StorageAdapter object and
// normalized document path. Cross-adapter and multi-runtime locking belongs to host composition.
const documentQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

export class FullTextKnowledgeBaseFileStore implements FullTextKnowledgeBaseStore {
  readonly paths: FullTextKnowledgeBasePaths;
  private readonly scopeFingerprint: string;
  private readonly identificationFingerprint: string;

  constructor(
    private readonly storage: StorageAdapter,
    output: OutputSettings,
    scopeFingerprint: string,
    identificationFingerprint: string,
    private readonly options: FullTextKnowledgeBaseStorePathsOptions = {},
  ) {
    validateBoundFingerprints(scopeFingerprint, identificationFingerprint);
    this.scopeFingerprint = scopeFingerprint;
    this.identificationFingerprint = identificationFingerprint;
    this.paths = deriveFullTextKnowledgeBasePaths(
      storage,
      output,
      scopeFingerprint,
      identificationFingerprint,
    );
  }

  loadManifest(): Promise<FullTextKnowledgeBaseManifest> {
    return enqueue(this.storage, this.paths.manifest.documentPath, async () => {
      const loaded = await this.loadDurableManifest();
      return clone(loaded?.document ?? this.emptyManifest());
    });
  }

  replaceManifest(
    next: FullTextKnowledgeBaseManifest,
    expectedRevision: number,
  ): Promise<FullTextKnowledgeBaseManifest> {
    const validated = decodeFullTextKnowledgeBaseManifest(next);
    if (!validated || !this.matchesBoundIdentity(validated)) {
      return Promise.reject(error("invalid",
        "cannot persist invalid or identity-mismatched full-text knowledge base manifest"));
    }
    if (!isNonNegativeSafeInteger(expectedRevision)) {
      return Promise.reject(error("invalid",
        "full-text knowledge base manifest expected revision must be a non-negative safe integer"));
    }
    return enqueue(this.storage, this.paths.manifest.documentPath, async () => {
      const loaded = await this.loadDurableManifest();
      const current = loaded?.document ?? this.emptyManifest();
      const candidate = clone(validated);
      candidate.revision = current.revision;

      // Check replay before CAS: an exact requested semantic state may be the result of a commit
      // whose success response was lost. Any changed stale state remains a conflict.
      if (loaded && semanticManifest(candidate) === semanticManifest(current)) {
        return clone(current);
      }
      if (expectedRevision !== current.revision) {
        throw stale(expectedRevision, current.revision);
      }
      // Model switch policy: the manifest's modelId/dimension are global. A different model
      // re-indexing the same scope/id produces different content, so a switch while the
      // knowledge base holds papers requires delete-and-rebuild (removeAll first).
      if (loaded !== null
        && Object.keys(loaded.document.papers).length > 0
        && candidate.modelId !== current.modelId) {
        throw error("invalid",
          "full-text knowledge base model switch requires rebuilding: remove all papers before indexing with a different model");
      }
      if (current.revision === Number.MAX_SAFE_INTEGER) {
        throw error("invalid", "full-text knowledge base manifest revision is exhausted");
      }
      candidate.revision = loaded === null ? 1 : current.revision + 1;
      candidate.updatedAt = latestTimestamp(this.validNow(), current);
      if (!decodeFullTextKnowledgeBaseManifest(candidate)) {
        throw error("invalid", "cannot persist invalid full-text knowledge base manifest");
      }
      await saveManifestDocument(this.storage, this.paths.manifest, candidate, loaded?.raw ?? null);
      return clone(candidate);
    });
  }

  loadPaper(paperKey: string): Promise<FullTextPaperDocument | null> {
    const path = createFullTextKnowledgeBasePaperPath(this.storage, this.paths, paperKey);
    return enqueue(this.storage, path, async () => {
      const read = await readDocument(this.storage, path, decodeFullTextPaperDocument);
      if (read.kind === "missing") return null;
      if (read.kind === "incompatible") {
        throw error("incompatible",
          `incompatible full-text paper schema version ${read.schemaVersion}: ${path}`);
      }
      if (read.kind !== "valid") {
        // Paper files are derived data: the owner may re-index to rebuild them.
        throw error("corrupt-or-unreadable",
          `corrupt or unreadable full-text paper document (rebuild by re-indexing the paper): ${path}`,
          { cause: readCause(read) });
      }
      return clonePaperDocument(read.document);
    });
  }

  savePaper(document: FullTextPaperDocument): Promise<void> {
    // decodeFullTextPaperDocument reads the persisted JSON form (vectors as
    // base64), so validate the exact serialized bytes we are about to write.
    const serialized = serializeFullTextPaperDocument(document);
    if (!decodeFullTextPaperDocument(JSON.parse(serialized))) {
      return Promise.reject(error("invalid", "cannot persist invalid full-text paper document"));
    }
    const path = createFullTextKnowledgeBasePaperPath(this.storage, this.paths, document.paperKey);
    return enqueue(this.storage, path, async () => {
      requireAtomic(this.storage);
      const existing = await readDocument(this.storage, path, decodeFullTextPaperDocument);
      if (existing.kind === "incompatible") {
        throw error("incompatible",
          `refusing to overwrite incompatible full-text paper schema version ${existing.schemaVersion}: ${path}`);
      }
      try {
        await ensureDirDeep(this.storage, this.paths.papersDirectory);
        // Derived, content-addressed data: idempotent rewrite, no backup needed.
        await this.storage.writeTextAtomic!(path, serialized);
      } catch (caught) {
        throw error("save-failed", `failed to save full-text paper document: ${path}`, { cause: caught });
      }
    });
  }

  removePaper(paperKey: string): Promise<void> {
    const path = createFullTextKnowledgeBasePaperPath(this.storage, this.paths, paperKey);
    return enqueue(this.storage, path, async () => {
      if (!(await this.storage.exists(path))) return; // idempotent: missing paper is already gone
      try {
        await this.storage.remove(path);
      } catch (caught) {
        throw error("save-failed", `failed to remove full-text paper document: ${path}`, { cause: caught });
      }
    });
  }

  /** Delete the whole knowledge base for this scope/identification (rebuild path). */
  removeAll(): Promise<void> {
    // Serialize with manifest operations so a concurrent replaceManifest cannot
    // resurrect a manifest under a concurrent removeAll.
    return enqueue(this.storage, this.paths.manifest.documentPath, async () => {
      await this.removePapersDirectory();
      for (const path of [this.paths.manifest.documentPath, this.paths.manifest.backupPath]) {
        if (!(await this.storage.exists(path))) continue;
        try {
          await this.storage.remove(path);
        } catch (caught) {
          throw error("save-failed", `failed to remove full-text knowledge base manifest: ${path}`,
            { cause: caught });
        }
      }
    });
  }

  private loadDurableManifest(): Promise<{ document: FullTextKnowledgeBaseManifest; raw: string } | null> {
    return loadDurableManifest({
      storage: this.storage,
      paths: this.paths.manifest,
      scopeFingerprint: this.scopeFingerprint,
      identificationFingerprint: this.identificationFingerprint,
      onWarning: this.options.onWarning,
    });
  }

  private emptyManifest(): FullTextKnowledgeBaseManifest {
    // In-memory empty state (never persisted): modelId/dimension are filled by
    // the first replaceManifest call; updatedAt is decided by the clock.
    return {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      revision: 0,
      scopeFingerprint: this.scopeFingerprint,
      identificationFingerprint: this.identificationFingerprint,
      modelId: "",
      dimension: 0,
      updatedAt: this.validNow().toISOString(),
      papers: {},
    };
  }

  private validNow(): Date {
    const now = this.options.now?.() ?? new Date();
    if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
      throw error("invalid", "full-text knowledge base clock returned an invalid date");
    }
    return now;
  }

  private matchesBoundIdentity(document: FullTextKnowledgeBaseManifest): boolean {
    return document.scopeFingerprint === this.scopeFingerprint
      && document.identificationFingerprint === this.identificationFingerprint;
  }

  private async removePapersDirectory(): Promise<void> {
    if (!(await this.storage.exists(this.paths.papersDirectory))) return;
    if (this.storage.list) {
      for (const entry of await this.storage.list(this.paths.papersDirectory)) {
        await this.removeEntry(entry);
      }
    } else {
      // No listing support: enumerate paper files through the manifest records.
      try {
        const manifest = await this.loadDurableManifest();
        if (manifest) {
          for (const paperKey of Object.keys(manifest.document.papers)) {
            const path = createFullTextKnowledgeBasePaperPath(this.storage, this.paths, paperKey);
            if (await this.storage.exists(path)) {
              await this.removePaperFile(path);
            }
          }
        }
      } catch {
        // Unreadable manifest without list support: nothing else enumerable;
        // the manifest files themselves are still removed by removeAll.
      }
    }
    // Best-effort empty-directory cleanup: StorageAdapter has no rmdir, and hosts may
    // refuse to remove directories (non-empty or rooted); files are authoritative.
    await this.storage.remove(this.paths.papersDirectory).catch(() => undefined);
    await this.storage.remove(this.paths.manifest.directory).catch(() => undefined);
  }

  private async removeEntry(entry: StorageEntry): Promise<void> {
    if (entry.type === "folder") {
      if (this.storage.list) {
        for (const nested of await this.storage.list(entry.path)) {
          await this.removeEntry(nested);
        }
      }
      await this.storage.remove(entry.path).catch(() => undefined);
      return;
    }
    await this.removePaperFile(entry.path);
  }

  private async removePaperFile(path: string): Promise<void> {
    try {
      await this.storage.remove(path);
    } catch (caught) {
      throw error("save-failed", `failed to remove full-text paper document: ${path}`, { cause: caught });
    }
  }
}

async function loadDurableManifest(input: {
  storage: StorageAdapter;
  paths: FullTextKnowledgeBasePaths["manifest"];
  scopeFingerprint: string;
  identificationFingerprint: string;
  onWarning?: (message: string, error?: unknown) => void;
}): Promise<{ document: FullTextKnowledgeBaseManifest; raw: string } | null> {
  const primary = await readDocument(input.storage, input.paths.documentPath, decodeFullTextKnowledgeBaseManifest);
  if (primary.kind === "valid") {
    if (!compatible(primary.document, input)) {
      throw error("incompatible",
        `incompatible full-text knowledge base manifest: ${input.paths.documentPath}`);
    }
    return { document: primary.document, raw: primary.raw };
  }
  if (primary.kind === "incompatible") {
    throw error("incompatible",
      `incompatible full-text knowledge base manifest schema version ${primary.schemaVersion}: ${input.paths.documentPath}`);
  }
  const backup = await readDocument(input.storage, input.paths.backupPath, decodeFullTextKnowledgeBaseManifest);
  if (backup.kind === "incompatible") {
    throw error("incompatible",
      `incompatible full-text knowledge base manifest backup schema version ${backup.schemaVersion}: ${input.paths.backupPath}`);
  }
  if (backup.kind === "valid") {
    if (!compatible(backup.document, input)) {
      throw error("incompatible",
        `incompatible full-text knowledge base manifest backup: ${input.paths.backupPath}`);
    }
    input.onWarning?.(
      `full-text knowledge base manifest recovered from backup: ${input.paths.backupPath}`,
      readCause(primary),
    );
    await repairPrimary(input.storage, input.paths, backup.raw);
    return { document: backup.document, raw: backup.raw };
  }
  if (primary.kind === "missing" && backup.kind === "missing") return null;
  throw error("corrupt-or-unreadable",
    `corrupt or unreadable full-text knowledge base manifest: ${input.paths.documentPath}`,
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
    const parsed = JSON.parse(raw);
    if (isPlainObject(parsed)
      && typeof parsed.schemaVersion === "number"
      && parsed.schemaVersion > FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION) {
      return { kind: "incompatible", schemaVersion: parsed.schemaVersion, raw };
    }
    const document = decoder(parsed);
    return document ? { kind: "valid", document, raw } : { kind: "corrupt", raw };
  } catch (caught) {
    return { kind: "corrupt", error: caught };
  }
}

async function repairPrimary(
  storage: StorageAdapter,
  paths: FullTextKnowledgeBasePaths["manifest"],
  raw: string,
): Promise<void> {
  requireAtomic(storage);
  try {
    await ensureDirDeep(storage, paths.directory);
    await storage.writeTextAtomic!(paths.documentPath, canonicalRaw(raw));
  } catch (caught) {
    throw error("repair-failed", `failed to repair full-text knowledge base manifest: ${paths.documentPath}`,
      { cause: caught });
  }
}

async function saveManifestDocument(
  storage: StorageAdapter,
  paths: FullTextKnowledgeBasePaths["manifest"],
  document: FullTextKnowledgeBaseManifest,
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
    throw error("save-failed", `failed to save full-text knowledge base manifest: ${paths.documentPath}`,
      { cause: caught });
  }
}

function requireAtomic(storage: StorageAdapter): void {
  if (!storage.writeTextAtomic) {
    throw error("atomic-write-unsupported",
      "full-text knowledge base storage does not support atomic writes");
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
  if (!match) throw error("invalid", `full-text knowledge base ${name} must be a SHA-256 fingerprint`);
  return match[1]!;
}

function latestTimestamp(now: Date, current: FullTextKnowledgeBaseManifest): string {
  return new Date(Math.max(now.getTime(), Date.parse(current.updatedAt))).toISOString();
}

interface Fingerprinted { scopeFingerprint: string; identificationFingerprint: string }

function compatible(document: Fingerprinted, input: Fingerprinted): boolean {
  return document.scopeFingerprint === input.scopeFingerprint
    && document.identificationFingerprint === input.identificationFingerprint;
}

/** Canonical semantic snapshot: revision and updatedAt are store-owned, not content. */
function semanticManifest(document: FullTextKnowledgeBaseManifest): string {
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
    `stale full-text knowledge base manifest revision: expected ${String(expectedRevision)}, current ${String(currentRevision)}`,
    { expectedRevision, currentRevision },
  );
}

function error(
  code: FullTextKnowledgeBaseStoreErrorCode,
  message: string,
  options: { cause?: unknown } & RevisionConflictFields = {},
): FullTextKnowledgeBaseStoreError {
  return new FullTextKnowledgeBaseStoreError(message, code, options);
}

function readCause<T>(result: ReadResult<T>): unknown {
  return result.kind === "corrupt" || result.kind === "unreadable" ? result.error : undefined;
}

function canonicalRaw(raw: string): string {
  return `${JSON.stringify(JSON.parse(raw), null, 2)}\n`;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function clone(value: FullTextKnowledgeBaseManifest): FullTextKnowledgeBaseManifest {
  return JSON.parse(JSON.stringify(value)) as FullTextKnowledgeBaseManifest;
}

function clonePaperDocument(document: FullTextPaperDocument): FullTextPaperDocument {
  return {
    schemaVersion: document.schemaVersion,
    paperKey: document.paperKey,
    modelId: document.modelId,
    dimension: document.dimension,
    textHash: document.textHash,
    contentHash: document.contentHash,
    title: document.title,
    titleVersion: document.titleVersion,
    filePaths: [...document.filePaths],
    observationFingerprints: [...document.observationFingerprints],
    derivation: document.derivation === undefined ? undefined : {
      parser: { ...document.derivation.parser },
      chunkerVersion: document.derivation.chunkerVersion,
      embeddingInputVersion: document.derivation.embeddingInputVersion,
    },
    chunks: document.chunks.map((chunk) => ({ ...chunk })),
    vectors: new Float32Array(document.vectors),
    updatedAt: document.updatedAt,
  };
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}
