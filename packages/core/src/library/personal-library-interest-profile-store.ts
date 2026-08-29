import type { StorageAdapter } from "../core/adapters";
import type { OutputSettings } from "../settings/types";
import { derivePaperInboxPaths } from "../services/paper-index";
import type { PersonalLibraryCatalog } from "./personal-library-catalog";
import {
  confirmPersonalLibraryDirectionCandidate,
  type PersonalLibraryReviewedDirectionDraft,
} from "./personal-library-interest-profile-review";
import {
  PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION,
  PERSONAL_LIBRARY_PROPOSAL_SCHEMA_VERSION,
  createEmptyPersonalLibraryInterestProfile,
  decodeDurablePersonalLibraryInterestProfile,
  decodePersonalLibraryDirectionProposal,
  decodePersonalLibraryInterestProfile,
  decodePersistedPersonalLibraryInterestProfile,
  type PersonalLibraryDirectionProposal,
  type PersonalLibraryInterestProfile,
} from "./personal-library-interest-profile";

export interface PersonalLibraryInterestProfileDocumentPaths {
  directory: string;
  documentPath: string;
  backupPath: string;
}

export interface PersonalLibraryInterestProfileStorePaths {
  directory: string;
  proposal: PersonalLibraryInterestProfileDocumentPaths;
  profile: PersonalLibraryInterestProfileDocumentPaths;
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

export class PersonalLibraryConfirmationCoordinatorError extends Error {
  readonly code = "partial-confirmation-conflict" as const;

  constructor(
    message: string,
    readonly details: Readonly<Record<string, unknown>> = {},
    options: { cause?: unknown } = {},
  ) {
    super(message, options);
    this.name = "PersonalLibraryConfirmationCoordinatorError";
  }
}

export class PersonalLibraryInterestProfileStoreError extends Error {
  readonly expectedRevision?: number | null;
  readonly currentRevision?: number | null;

  constructor(
    message: string,
    readonly code: PersonalLibraryStoreErrorCode,
    options: { cause?: unknown } & RevisionConflictFields = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "PersonalLibraryInterestProfileStoreError";
    this.expectedRevision = options.expectedRevision;
    this.currentRevision = options.currentRevision;
  }
}

type StoreKind = "proposal" | "profile";
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
    profile: documentPaths(storage, directory, "interest-profile.json"),
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

export interface ConfirmPersonalLibraryDirectionWithStoresInput {
  proposalStore: PersonalLibraryDirectionProposalStore;
  profileStore: PersonalLibraryInterestProfileStore;
  proposal: PersonalLibraryDirectionProposal;
  profile: PersonalLibraryInterestProfile;
  catalog: PersonalLibraryCatalog;
  candidateId: string;
  directionId: string;
  status: "active" | "disabled";
  draft: PersonalLibraryReviewedDirectionDraft;
  now: Date;
  expectedProposalRevision: number;
  expectedProfileRevision: number;
}

export async function confirmPersonalLibraryDirectionWithStores(
  input: ConfirmPersonalLibraryDirectionWithStoresInput,
): Promise<{ proposal: PersonalLibraryDirectionProposal; profile: PersonalLibraryInterestProfile }> {
  const requested = confirmPersonalLibraryDirectionCandidate({
    proposal: input.proposal,
    profile: input.profile,
    catalog: input.catalog,
    candidateId: input.candidateId,
    directionId: input.directionId,
    status: input.status,
    draft: input.draft,
    now: input.now,
  });
  const originalProfile = clone(input.profile);
  const requestedProfile = clone(requested.profile);
  const originalProposal = clone(input.proposal);
  const requestedProposal = clone(requested.proposal);
  const savedProfile = await establishConfirmedProfile({
    store: input.profileStore,
    original: originalProfile,
    requested: requestedProfile,
    expectedRevision: input.expectedProfileRevision,
    directionId: input.directionId,
  });
  const savedProposal = await consumeConfirmedProposal({
    store: input.proposalStore,
    original: originalProposal,
    requested: requestedProposal,
    expectedRevision: input.expectedProposalRevision,
    directionId: input.directionId,
  });
  return { proposal: savedProposal, profile: savedProfile };
}

async function establishConfirmedProfile(input: {
  store: PersonalLibraryInterestProfileStore;
  original: PersonalLibraryInterestProfile;
  requested: PersonalLibraryInterestProfile;
  expectedRevision: number;
  directionId: string;
}): Promise<PersonalLibraryInterestProfile> {
  try {
    return await input.store.replace(input.requested, input.expectedRevision);
  } catch (caught) {
    if (!(caught instanceof PersonalLibraryInterestProfileStoreError)
      || caught.code !== "save-failed") throw caught;
    const durable = await loadProfileForCoordinator(input.store, input.directionId, caught);
    if (semanticProfile(durable) === semanticProfile(input.requested)) return durable;
    if (semanticProfile(durable) !== semanticProfile(input.original)) {
      throw coordinatorConflict("profile-divergent", input.directionId, caught, durable.revision);
    }
    try {
      return await input.store.replace(input.requested, durable.revision);
    } catch (retryCaught) {
      if (!(retryCaught instanceof PersonalLibraryInterestProfileStoreError)
        || retryCaught.code !== "save-failed") {
        throw coordinatorConflict("profile-retry-failed", input.directionId, retryCaught, durable.revision);
      }
      const final = await loadProfileForCoordinator(input.store, input.directionId, retryCaught);
      if (semanticProfile(final) === semanticProfile(input.requested)) return final;
      throw coordinatorConflict("profile-commit-uncertain", input.directionId, retryCaught, final.revision);
    }
  }
}

async function consumeConfirmedProposal(input: {
  store: PersonalLibraryDirectionProposalStore;
  original: PersonalLibraryDirectionProposal;
  requested: PersonalLibraryDirectionProposal;
  expectedRevision: number;
  directionId: string;
}): Promise<PersonalLibraryDirectionProposal> {
  try {
    return await input.store.replace(input.requested, input.expectedRevision);
  } catch (caught) {
    if (!(caught instanceof PersonalLibraryDirectionProposalStoreError)
      || caught.code !== "save-failed") {
      throw coordinatorConflict("proposal-write-failed-after-profile-commit", input.directionId, caught);
    }
    const durable = await loadProposalForCoordinator(input.store, input.directionId, caught);
    if (semanticProposalIgnoringRevision(durable) === semanticProposalIgnoringRevision(input.requested)) {
      return durable;
    }
    if (semanticProposalIgnoringRevision(durable) !== semanticProposalIgnoringRevision(input.original)) {
      throw coordinatorConflict("proposal-divergent-after-profile-commit", input.directionId, caught, durable.revision);
    }
    try {
      return await input.store.replace(input.requested, durable.revision);
    } catch (retryCaught) {
      if (!(retryCaught instanceof PersonalLibraryDirectionProposalStoreError)
        || retryCaught.code !== "save-failed") {
        throw coordinatorConflict("proposal-retry-failed-after-profile-commit", input.directionId,
          retryCaught, durable.revision);
      }
      const final = await loadProposalForCoordinator(input.store, input.directionId, retryCaught);
      if (semanticProposalIgnoringRevision(final) === semanticProposalIgnoringRevision(input.requested)) {
        return final;
      }
      throw coordinatorConflict("proposal-commit-uncertain-after-profile-commit", input.directionId,
        retryCaught, final.revision);
    }
  }
}

async function loadProfileForCoordinator(
  store: PersonalLibraryInterestProfileStore,
  directionId: string,
  cause: unknown,
): Promise<PersonalLibraryInterestProfile> {
  try {
    return await store.load();
  } catch (caught) {
    throw coordinatorConflict("profile-state-unreadable", directionId, caught ?? cause);
  }
}

async function loadProposalForCoordinator(
  store: PersonalLibraryDirectionProposalStore,
  directionId: string,
  cause: unknown,
): Promise<PersonalLibraryDirectionProposal> {
  try {
    const proposal = await store.load();
    if (proposal) return proposal;
  } catch (caught) {
    throw coordinatorConflict("proposal-state-unreadable-after-profile-commit", directionId, caught);
  }
  throw coordinatorConflict("proposal-missing-after-profile-commit", directionId, cause);
}

function coordinatorConflict(
  stage: string,
  directionId: string,
  cause: unknown,
  currentRevision?: number,
): PersonalLibraryConfirmationCoordinatorError {
  return new PersonalLibraryConfirmationCoordinatorError(
    `personal library confirmation coordinator conflict at ${stage}`,
    { stage, directionId, ...(currentRevision === undefined ? {} : { currentRevision }) },
    { cause },
  );
}

export class PersonalLibraryInterestProfileStore {
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
    validateBoundFingerprints("profile", scopeFingerprint, identificationFingerprint);
    this.scopeFingerprint = scopeFingerprint;
    this.identificationFingerprint = identificationFingerprint;
    this.paths = derivePersonalLibraryInterestProfileStorePaths(
      storage,
      output,
      scopeFingerprint,
      identificationFingerprint,
    ).profile;
  }

  load(): Promise<PersonalLibraryInterestProfile> {
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDocument();
      return clone(loaded?.document ?? createEmptyPersonalLibraryInterestProfile(
        this.scopeFingerprint,
        this.identificationFingerprint,
        this.validNow(),
      ));
    });
  }

  replace(
    next: PersonalLibraryInterestProfile,
    expectedRevision: number,
  ): Promise<PersonalLibraryInterestProfile> {
    const validated = decodePersonalLibraryInterestProfile(next);
    if (!validated || !this.matchesBoundIdentity(validated)) {
      return Promise.reject(error("profile", "invalid",
        "cannot persist invalid or identity-mismatched personal library interest profile"));
    }
    if (!isNonNegativeSafeInteger(expectedRevision)) {
      return Promise.reject(error("profile", "invalid",
        "personal library interest profile expected revision must be a non-negative safe integer"));
    }
    return enqueue(this.storage, this.paths.documentPath, async () => {
      const loaded = await this.loadDocument();
      const current = loaded?.document ?? createEmptyPersonalLibraryInterestProfile(
        this.scopeFingerprint,
        this.identificationFingerprint,
        this.validNow(),
      );
      const candidate = clone(validated);
      candidate.schemaVersion = PERSONAL_LIBRARY_INTEREST_PROFILE_SCHEMA_VERSION;
      candidate.revision = current.revision;
      candidate.updatedAt = current.updatedAt;

      // Check replay before CAS: an exact requested semantic state may be the result of a commit
      // whose success response was lost. Any changed stale state remains a conflict.
      if (loaded && semanticProfile(candidate) === semanticProfile(current)) return clone(current);
      if (expectedRevision !== current.revision) {
        throw stale("profile", expectedRevision, current.revision);
      }
      if (loaded === null && semanticProfile(candidate) === semanticProfile(current)) {
        return clone(current);
      }
      if (current.revision === Number.MAX_SAFE_INTEGER) {
        throw error("profile", "invalid", "personal library interest profile revision is exhausted");
      }
      candidate.revision = loaded === null ? 1 : current.revision + 1;
      candidate.updatedAt = latestTimestamp(this.validNow(), current, candidate);
      if (!decodePersistedPersonalLibraryInterestProfile(candidate)) {
        throw error("profile", "invalid", "cannot persist invalid personal library interest profile");
      }
      await saveDocument(this.storage, this.paths, "profile", candidate, loaded?.raw ?? null);
      return clone(candidate);
    });
  }

  private loadDocument() {
    return loadDurableDocument({
      storage: this.storage,
      paths: this.paths,
      kind: "profile",
      decoder: decodeDurablePersonalLibraryInterestProfile,
      canonicalDecoder: decodePersistedPersonalLibraryInterestProfile,
      migrateOnLoad: true,
      scopeFingerprint: this.scopeFingerprint,
      identificationFingerprint: this.identificationFingerprint,
      onWarning: this.options.onWarning,
    });
  }

  private matchesBoundIdentity(document: PersonalLibraryInterestProfile): boolean {
    return document.scopeFingerprint === this.scopeFingerprint
      && document.identificationFingerprint === this.identificationFingerprint;
  }

  private validNow(): Date {
    const now = this.options.now?.() ?? new Date();
    if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
      throw error("profile", "invalid", "personal library interest profile clock returned an invalid date");
    }
    return now;
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

function latestTimestamp(
  now: Date,
  current: PersonalLibraryInterestProfile,
  candidate: PersonalLibraryInterestProfile,
): string {
  let latest = Math.max(now.getTime(), Date.parse(current.updatedAt));
  for (const direction of candidate.directions) latest = Math.max(latest, Date.parse(direction.updatedAt));
  return new Date(latest).toISOString();
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

function semanticProfile(document: PersonalLibraryInterestProfile): string {
  const { revision: _revision, updatedAt: _updatedAt, ...semantic } = document;
  return JSON.stringify(semantic);
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
): PersonalLibraryDirectionProposalStoreError | PersonalLibraryInterestProfileStoreError {
  return kind === "proposal"
    ? new PersonalLibraryDirectionProposalStoreError(message, code, options)
    : new PersonalLibraryInterestProfileStoreError(message, code, options);
}

function label(kind: StoreKind): string {
  return kind === "proposal" ? "direction proposal" : "interest profile";
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
