import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "../services/paper-index";
import {
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  decodePersonalLibraryDirectionProposal,
  type PersonalLibraryDirectionProposal,
} from "./personal-library-interest-profile";

export interface PersonalLibraryInterestProfileDocumentPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface PersonalLibraryInterestProfileStorePaths {
  directory: string;
  proposal: PersonalLibraryInterestProfileDocumentPaths;
}

export interface PersonalLibraryInterestProfileStoreOptions {
  now?: () => Date;
  onWarning?: (message: string, error?: unknown) => void;
}

export type PersonalLibraryStoreErrorCode =
  | "invalid"
  | "stale"
  | "incompatible"
  | "corrupt-or-unreadable"
  | "regeneration-required"
  | "atomic-write-unsupported"
  | "repair-failed"
  | "save-failed";

interface RevisionConflictFields {
  expectedRevision?: number | null;
  currentRevision?: number | null;
}

export class PersonalLibraryDirectionProposalStoreError extends Error {
  readonly expectedRevision?: number | null;
  readonly currentRevision?: number | null;

  constructor(
    message: string,
    readonly code: PersonalLibraryStoreErrorCode,
    options: { cause?: unknown } & RevisionConflictFields = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "PersonalLibraryDirectionProposalStoreError";
    this.expectedRevision = options.expectedRevision;
    this.currentRevision = options.currentRevision;
  }
}

type StoreKind = "proposal";
type Decoder<T> = (value: unknown) => T | null;
type ReadResult<T> =
  | { kind: "missing" }
  | { kind: "valid"; document: T; raw: string }
  | { kind: "corrupt"; raw?: string; error?: unknown }
  | { kind: "unreadable"; error: unknown };

// Runtime-local serialization is guaranteed only for the same StorageAdapter object and
// normalized document path. Cross-adapter and multi-runtime locking belongs to host composition.
const documentQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

export function derivePersonalLibraryInterestProfileStorePaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  output: OutputSettings,
  scopeFingerprint: string,
  identificationFingerprint: string,
): PersonalLibraryInterestProfileStorePaths {
  const scopeHex = fingerprintHex("proposal", "scopeFingerprint", scopeFingerprint);
  const identificationHex = fingerprintHex(
    "proposal",
    "identificationFingerprint",
    identificationFingerprint,
  );
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(
    `${indexDir}/personal-library-profiles/${scopeHex}/${identificationHex}`,
  );
  return {
    directory,
    proposal: documentPaths(storage, directory, "direction-proposal.json"),
  };
}

export class PersonalLibraryDirectionProposalStore {
  readonly paths: PersonalLibraryInterestProfileDocumentPaths;
  private readonly scopeFingerprint: string;
  private readonly identificationFingerprint: string;

  constructor(
    private readonly storage: StorageAdapter,
    output: OutputSettings,
    scopeFingerprint: string,
    identificationFingerprint: string,
    private readonly options: PersonalLibraryInterestProfileStoreOptions = {},
  ) {
    validateBoundFingerprints("proposal", scopeFingerprint, identificationFingerprint);
    this.scopeFingerprint = scopeFingerprint;
    this.identificationFingerprint = identificationFingerprint;
    this.paths = derivePersonalLibraryInterestProfileStorePaths(
      storage,
      output,
      scopeFingerprint,
      identificationFingerprint,
    ).proposal;
  }

  load(): Promise<PersonalLibraryDirectionProposal | null> {
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDocument();
      return loaded === null ? null : clone(loaded.document);
    });
  }

  replace(
    next: PersonalLibraryDirectionProposal,
    expectedRevision: number | null,
  ): Promise<PersonalLibraryDirectionProposal> {
    const validated = decodePersonalLibraryDirectionProposal(next);
    if (!validated || !this.matchesBoundIdentity(validated)) {
      return Promise.reject(error("proposal", "invalid",
        "cannot persist invalid or identity-mismatched personal library direction proposal"));
    }
    if (expectedRevision !== null && !isNonNegativeSafeInteger(expectedRevision)) {
      return Promise.reject(error("proposal", "invalid",
        "personal library direction proposal expected revision must be null or a non-negative safe integer"));
    }
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDocument();
      const candidate = clone(validated);
      candidate.schemaVersion = PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION;
      candidate.revision = loaded?.document.revision ?? 0;

      // Exact semantic replay is idempotent even when the caller did not observe an ambiguous commit.
      if (loaded && semanticProposal(candidate) === semanticProposal(loaded.document)) {
        return clone(loaded.document);
      }
      const currentRevision = loaded?.document.revision ?? null;
      if ((loaded === null && expectedRevision !== null)
        || (loaded !== null && expectedRevision !== loaded.document.revision)) {
        throw stale("proposal", expectedRevision, currentRevision);
      }
      if (loaded) {
        if (loaded.document.revision === Number.MAX_SAFE_INTEGER) {
          throw error("proposal", "invalid", "personal library direction proposal revision is exhausted");
        }
        candidate.revision += 1;
      }
      if (!decodePersonalLibraryDirectionProposal(candidate)) {
        throw error("proposal", "invalid", "cannot persist invalid personal library direction proposal");
      }
      await saveDocument(this.storage, this.paths, "proposal", candidate, loaded?.raw ?? null);
      return clone(candidate);
    });
  }

  private loadDocument() {
    return loadDurableDocument({
      storage: this.storage,
      paths: this.paths,
      kind: "proposal",
      decoder: decodePersonalLibraryDirectionProposal,
      scopeFingerprint: this.scopeFingerprint,
      identificationFingerprint: this.identificationFingerprint,
      onWarning: this.options.onWarning,
    });
  }

  private matchesBoundIdentity(document: PersonalLibraryDirectionProposal): boolean {
    return document.scopeFingerprint === this.scopeFingerprint
      && document.identificationFingerprint === this.identificationFingerprint;
  }
}

function documentPaths(
  storage: Pick<StorageAdapter, "normalizePath">,
  directory: string,
  filename: string,
): PersonalLibraryInterestProfileDocumentPaths {
  const normalizedDirectory = storage.normalizePath(directory);
  const documentPath = storage.normalizePath(`${normalizedDirectory}/${filename}`);
  return {
    directory: normalizedDirectory,
    documentPath,
    backupPath: storage.normalizePath(`${documentPath}.backup`),
  };
}

async function loadDurableDocument<T>(input: {
  storage: StorageAdapter;
  paths: PersonalLibraryInterestProfileDocumentPaths;
  kind: StoreKind;
  decoder: Decoder<T>;
  canonicalDecoder?: Decoder<T>;
  migrateOnLoad?: boolean;
  scopeFingerprint: string;
  identificationFingerprint: string;
  onWarning?: (message: string, error?: unknown) => void;
}): Promise<{ document: T & Fingerprinted; raw: string } | null> {
  const primary = await readDocument(input.storage, input.paths.documentPath, input.decoder);
  if (primary.kind === "valid") {
    if (!compatible(primary.document as T & Fingerprinted, input)) {
      throw error(input.kind, "incompatible",
        `incompatible personal library ${label(input.kind)}: ${input.paths.documentPath}`);
    }
    const migrated = input.migrateOnLoad && input.canonicalDecoder
      && input.canonicalDecoder(JSON.parse(primary.raw)) === null;
    if (migrated) {
      const raw = `${JSON.stringify(primary.document, null, 2)}\n`;
      await repairPrimary(input.storage, input.paths, input.kind, raw);
      return { document: primary.document as T & Fingerprinted, raw };
    }
    return { document: primary.document as T & Fingerprinted, raw: primary.raw };
  }
  const backup = await readDocument(input.storage, input.paths.backupPath, input.decoder);
  if (backup.kind === "valid") {
    if (!compatible(backup.document as T & Fingerprinted, input)) {
      throw error(input.kind, "incompatible",
        `incompatible personal library ${label(input.kind)} backup: ${input.paths.backupPath}`);
    }
    input.onWarning?.(`personal library ${label(input.kind)} recovered from backup: ${input.paths.backupPath}`,
      readCause(primary));
    const migrated = input.migrateOnLoad && input.canonicalDecoder
      && input.canonicalDecoder(JSON.parse(backup.raw)) === null;
    const raw = migrated ? `${JSON.stringify(backup.document, null, 2)}\n` : backup.raw;
    await repairPrimary(input.storage, input.paths, input.kind, raw);
    return { document: backup.document as T & Fingerprinted, raw };
  }
  if (primary.kind === "missing" && backup.kind === "missing") return null;
  if (input.kind === "proposal" && (isLegacyProposalRead(primary) || isLegacyProposalRead(backup))) {
    throw error("proposal", "regeneration-required",
      `legacy personal library direction proposal must be regenerated: ${input.paths.documentPath}`);
  }
  throw error(input.kind, "corrupt-or-unreadable",
    `corrupt or unreadable personal library ${label(input.kind)}: ${input.paths.documentPath}`,
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
  paths: PersonalLibraryInterestProfileDocumentPaths,
  kind: StoreKind,
  raw: string,
): Promise<void> {
  requireAtomic(storage, kind);
  try {
    await ensureDirDeep(storage, paths.directory);
    await storage.writeTextAtomic!(paths.documentPath, canonicalRaw(raw));
  } catch (caught) {
    throw error(kind, "repair-failed", `failed to repair personal library ${label(kind)}: ${paths.documentPath}`,
      { cause: caught });
  }
}

async function saveDocument<T>(
  storage: StorageAdapter,
  paths: PersonalLibraryInterestProfileDocumentPaths,
  kind: StoreKind,
  document: T,
  priorPrimaryRaw: string | null,
): Promise<void> {
  requireAtomic(storage, kind);
  const content = `${JSON.stringify(document, null, 2)}\n`;
  try {
    await ensureDirDeep(storage, paths.directory);
    await storage.writeTextAtomic!(paths.backupPath,
      priorPrimaryRaw === null ? content : canonicalRaw(priorPrimaryRaw));
    // Atomic promotion is commit-wins. Never blindly overwrite a possibly committed primary.
    await storage.writeTextAtomic!(paths.documentPath, content);
  } catch (caught) {
    throw error(kind, "save-failed", `failed to save personal library ${label(kind)}: ${paths.documentPath}`,
      { cause: caught });
  }
}

function requireAtomic(storage: StorageAdapter, kind: StoreKind): void {
  if (!storage.writeTextAtomic) {
    throw error(kind, "atomic-write-unsupported",
      `personal library ${label(kind)} storage does not support atomic writes`);
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

function validateBoundFingerprints(kind: StoreKind, scope: string, identification: string): void {
  fingerprintHex(kind, "scopeFingerprint", scope);
  fingerprintHex(kind, "identificationFingerprint", identification);
}

function fingerprintHex(kind: StoreKind, name: string, value: string): string {
  const match = /^sha256:([a-f0-9]{64})$/.exec(value);
  if (!match) throw error(kind, "invalid", `${name} must be a SHA-256 fingerprint`);
  return match[1]!;
}

interface Fingerprinted { scopeFingerprint: string; identificationFingerprint: string }

function compatible(document: Fingerprinted, input: Fingerprinted): boolean {
  return document.scopeFingerprint === input.scopeFingerprint
    && document.identificationFingerprint === input.identificationFingerprint;
}

function semanticProposalIgnoringRevision(document: PersonalLibraryDirectionProposal): string {
  const { revision: _revision, ...semantic } = document;
  return JSON.stringify(semantic);
}

function semanticProposal(document: PersonalLibraryDirectionProposal): string {
  return semanticProposalIgnoringRevision(document);
}

function stale(kind: StoreKind, expectedRevision: number | null, currentRevision: number | null): Error {
  return error(kind, "stale",
    `stale personal library ${label(kind)} revision: expected ${String(expectedRevision)}, current ${String(currentRevision)}`,
    { expectedRevision, currentRevision });
}

function error(
  kind: StoreKind,
  code: PersonalLibraryStoreErrorCode,
  message: string,
  options: { cause?: unknown } & RevisionConflictFields = {},
): PersonalLibraryDirectionProposalStoreError {
  return new PersonalLibraryDirectionProposalStoreError(message, code, options);
}

function label(kind: StoreKind): string {
  return "direction proposal";
}

function canonicalRaw(raw: string): string {
  return `${JSON.stringify(JSON.parse(raw), null, 2)}\n`;
}

function isLegacyProposalRead<T>(result: ReadResult<T>): boolean {
  if (result.kind !== "corrupt" || typeof result.raw !== "string") return false;
  try {
    const value = JSON.parse(result.raw);
    return typeof value === "object" && value !== null && value.schemaVersion === 1;
  } catch {
    return false;
  }
}

function readCause<T>(result: ReadResult<T>): unknown {
  return result.kind === "corrupt" || result.kind === "unreadable" ? result.error : undefined;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}
