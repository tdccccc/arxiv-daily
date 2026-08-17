import type { StorageAdapter } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import { sha256Hex } from "../../utils/digest";
import {
  MAX_BINARY_OBJECT_BYTES,
  MAX_GENERATION_DESCRIPTOR_BYTES,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeGenerationDescriptor,
  decodeVectorBlock,
  deriveFullTextGenerationIndexPaths,
  deriveFullTextGenerationPaths,
  encodeGenerationDescriptor,
  finishEvidenceStreamClosure,
  validateEvidenceStreamClosure,
  type EvidenceBlock,
  type EvidenceStreamClosureState,
  type FullTextGenerationPaths,
  type GenerationDescriptor,
  type GenerationObjectReference,
  type VectorBlock,
} from "./generation-index-format";

export const CURRENT_GENERATION_POINTER_FORMAT_VERSION = 1 as const;
export const CURRENT_GENERATION_POINTER_SCHEMA_VERSION = 1 as const;
const MAX_CURRENT_POINTER_BYTES = 16 * 1024;
const STAGING_CLAIM_FORMAT_VERSION = 1 as const;
const STAGING_CLAIM_SCHEMA_VERSION = 1 as const;
const STAGING_CLAIM_FILE = ".staging-claim.json";
const PROMOTION_CLAIM_FORMAT_VERSION = 1 as const;
const PROMOTION_CLAIM_SCHEMA_VERSION = 1 as const;
const PROMOTION_CLAIM_FILE = ".current-promotion-claim.json";

export interface CurrentGenerationPointer {
  readonly formatVersion: typeof CURRENT_GENERATION_POINTER_FORMAT_VERSION;
  readonly schemaVersion: typeof CURRENT_GENERATION_POINTER_SCHEMA_VERSION;
  readonly generationId: string;
  readonly sourceRevision: number;
  readonly scopeFingerprint: string;
  readonly identificationFingerprint: string;
  readonly descriptorChecksum: string;
  readonly checksum: string;
}

interface StagingClaim {
  readonly formatVersion: typeof STAGING_CLAIM_FORMAT_VERSION;
  readonly schemaVersion: typeof STAGING_CLAIM_SCHEMA_VERSION;
  readonly generationId: string;
  readonly sourceRevision: number;
  readonly scopeFingerprint: string;
  readonly identificationFingerprint: string;
  readonly descriptorChecksum: string;
  readonly writerToken: string;
}

interface PromotionClaim {
  readonly formatVersion: typeof PROMOTION_CLAIM_FORMAT_VERSION;
  readonly schemaVersion: typeof PROMOTION_CLAIM_SCHEMA_VERSION;
  readonly operation: "promote" | "recover";
  readonly writerToken: string;
  readonly candidateGenerationId: string;
  readonly sourceRevision: number;
  readonly expectedCurrent: null | { readonly generationId: string; readonly sourceRevision: number };
  readonly observedPrimaryChecksum: string;
  readonly scopeFingerprint: string;
  readonly identificationFingerprint: string;
}

export interface FullTextGenerationIndexStorePaths {
  readonly directory: string;
  readonly generationsDirectory: string;
  readonly currentPath: string;
  readonly backupPath: string;
  readonly promotionClaimPath: string;
}

export interface GenerationObjectWrite {
  readonly path: string;
  readonly bytes: Uint8Array;
}

export interface StageAndPromoteGenerationInput {
  readonly descriptor: GenerationDescriptor;
  /** High-entropy caller identity used to arbitrate same-generation writers. */
  readonly writerToken: string;
  /** Consumed once in descriptor order, allowing builders to release each bounded object after yield. */
  readonly objects: Iterable<GenerationObjectWrite> | AsyncIterable<GenerationObjectWrite>;
  readonly expectedCurrent: null | { readonly generationId: string; readonly sourceRevision: number };
  readonly sourceCurrentRevision: () => number | Promise<number>;
}

export interface GenerationStoreDiagnostics {
  /** Largest complete object read by this opened handle's store operations. */
  maxObjectBytes: number;
  /** Number of complete object reads performed by this opened handle. */
  objectReads: number;
}

export interface VerifiedGenerationObject {
  readonly reference: GenerationObjectReference;
  readonly bytes: Uint8Array;
}

export type OpenedGenerationObject =
  | { readonly reference: GenerationObjectReference & { readonly kind: "vector" }; readonly block: VectorBlock }
  | { readonly reference: GenerationObjectReference & { readonly kind: "evidence" }; readonly block: EvidenceBlock };

export type OpenedVectorObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "vector" } }>;
export type OpenedEvidenceObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "evidence" } }>;

export interface OpenedFullTextGeneration {
  readonly pointer: CurrentGenerationPointer;
  readonly descriptor: GenerationDescriptor;
  readonly diagnostics: GenerationStoreDiagnostics;
  /** Format-neutral verified bytes; remains usable when future formats add object kinds. */
  readRawObject(reference: GenerationObjectReference): Promise<VerifiedGenerationObject>;
  readObject(reference: GenerationObjectReference): Promise<OpenedGenerationObject>;
  /** Explicit bounded full-closure scan; ordinary open does not call this. */
  validateClosure(): Promise<void>;
  iterateObjects(kind?: GenerationObjectReference["kind"]): AsyncIterable<OpenedGenerationObject>;
  iterateVectorBlocks(): AsyncIterable<OpenedVectorObject>;
  iterateEvidenceBlocks(): AsyncIterable<OpenedEvidenceObject>;
}

export interface FullTextGenerationIndexStoreOptions {
  readonly onWarning?: (message: string, error?: unknown) => void;
  readonly beforePointerPromotion?: () => void | Promise<void>;
  /** Test seam after system-wide promotion ownership is acquired. */
  readonly afterPromotionClaimAcquired?: () => void | Promise<void>;
  readonly afterPointerPromotion?: () => void | Promise<void>;
  /** Failure-injection seam immediately before entering queued recovery. */
  readonly beforeRecoveryQueue?: () => void | Promise<void>;
}

export type FullTextGenerationIndexStoreErrorCode =
  | "invalid"
  | "capability-unsupported"
  | "generation-exists"
  | "generation-conflict"
  | "concurrent"
  | "stale-claim"
  | "stale-current"
  | "stale-source"
  | "commit-uncertain"
  | "claim-uncertain"
  | "incompatible"
  | "corrupt-or-unreadable"
  | "write-failed"
  | "repair-failed";

export class FullTextGenerationIndexStoreError extends Error {
  constructor(
    message: string,
    readonly code: FullTextGenerationIndexStoreErrorCode,
    readonly expectedRevision?: number | null,
    readonly currentRevision?: number | null,
    options: ErrorOptions = {},
  ) {
    super(message, options);
    this.name = "FullTextGenerationIndexStoreError";
  }
}

const writerQueues = new WeakMap<StorageAdapter, Map<string, Promise<void>>>();

/**
 * Immutable generation store. A generation-local staging claim owns writes and
 * cleanup for one immutable directory. A separate root promotion claim serializes
 * CURRENT/backup mutation across adapters/runtimes only when createTextExclusive
 * is system-wide for their shared backend. Fixed claims fail closed and are never
 * stolen by time. Task 5 still owns safe stale-claim repair and serialization with
 * the authoritative knowledge-base commit; these claims are not a cross-store transaction.
 */
export class FullTextGenerationIndexStore {
  readonly paths: FullTextGenerationIndexStorePaths;

  constructor(
    private readonly storage: StorageAdapter,
    private readonly output: OutputSettings,
    private readonly scopeFingerprint: string,
    private readonly identificationFingerprint: string,
    private readonly options: FullTextGenerationIndexStoreOptions = {},
  ) {
    validateFingerprint(scopeFingerprint, "scopeFingerprint");
    validateFingerprint(identificationFingerprint, "identificationFingerprint");
    const base = deriveFullTextGenerationIndexPaths(storage, output, scopeFingerprint, identificationFingerprint);
    const currentPath = storage.normalizePath(`${base.directory}/current.json`);
    this.paths = {
      directory: base.directory,
      generationsDirectory: base.generationsDirectory,
      currentPath,
      backupPath: storage.normalizePath(`${currentPath}.backup`),
      promotionClaimPath: storage.normalizePath(`${base.directory}/${PROMOTION_CLAIM_FILE}`),
    };
  }

  stageAndPromote(input: StageAndPromoteGenerationInput): Promise<OpenedFullTextGeneration> {
    return enqueueWriter(this.storage, this.paths.currentPath, () => this.stageAndPromoteSerial(input));
  }

  async openCurrent(): Promise<OpenedFullTextGeneration | null> {
    const primary = await this.readPointer(this.paths.currentPath);
    if (primary.kind === "incompatible") throw incompatible(`incompatible current generation pointer: ${this.paths.currentPath}`);
    if (primary.kind === "valid") {
      try {
        return await this.openPinned(primary.pointer);
      } catch (caught) {
        if (isIncompatible(caught)) throw caught;
        return this.queueRecovery(primary.raw, caught);
      }
    }
    if (primary.kind === "missing") {
      const backup = await this.readPointer(this.paths.backupPath);
      if (backup.kind === "missing") return null;
    }
    return this.queueRecovery(primary.kind === "corrupt" ? primary.raw : null, primary.kind === "corrupt" ? primary.error : undefined);
  }

  private async stageAndPromoteSerial(input: StageAndPromoteGenerationInput): Promise<OpenedFullTextGeneration> {
    requireWriteCapabilities(this.storage);
    let descriptor: GenerationDescriptor;
    let descriptorText: string;
    try {
      descriptorText = encodeGenerationDescriptor(input.descriptor);
      descriptor = decodeGenerationDescriptor(descriptorText);
      this.assertDescriptorIdentity(descriptor);
      validatePairedObjectCoverage(descriptor);
      validateWriterToken(input.writerToken);
      validateExpectedCurrent(input.expectedCurrent);
      if (typeof input.sourceCurrentRevision !== "function") throw new Error("sourceCurrentRevision callback is required");
      if (!isIterable(input.objects) && !isAsyncIterable(input.objects)) throw new Error("generation objects must be iterable");
    } catch (caught) {
      if (isIncompatible(caught)) throw caught;
      throw wrap("invalid", "invalid generation promotion input", caught);
    }
    const generation = deriveFullTextGenerationPaths(
      this.storage, this.output, this.scopeFingerprint, this.identificationFingerprint, descriptor.generationId,
    );

    // Exact committed replay precedes the exclusive-claim requirement so a
    // caller can acknowledge a previously committed generation idempotently.
    const replay = await this.tryExactCommittedReplay(descriptor);
    if (replay) return replay;
    requireExclusiveCreate(this.storage);

    const claimPath = this.storage.normalizePath(`${generation.directory}/${STAGING_CLAIM_FILE}`);
    const claim = stagingClaim(descriptor, descriptorText, input.writerToken);
    const claimText = encodeStagingClaim(claim);
    let ownsClaim = false;
    let ownershipLost = false;
    let committed = false;
    let cleanupWholeGeneration = true;

    try {
      await ensureDirDeep(this.storage, generation.directory);
      let created: boolean;
      try {
        created = await this.storage.createTextExclusive!(claimPath, claimText);
      } catch (caught) {
        // A thrown exclusive create has an unknowable outcome. Matching bytes do
        // not prove this call created the claim, so ownership is never inferred.
        throw wrap("claim-uncertain", `generation staging claim outcome is uncertain: ${claimPath}`, caught);
      }
      if (!created) {
        throw wrap("concurrent", `generation staging claim already exists: ${claimPath}`);
      }
      ownsClaim = true;

      // A claim-less partial directory from an earlier writer is not safe to
      // adopt. Check every v1 authoritative child before writing any object.
      const existingChildren = [generation.descriptorPath,
        ...descriptor.objects.map((reference) => this.objectPath(generation, reference.path))];
      for (const path of existingChildren) {
        if (await this.safeExists(path)) {
          cleanupWholeGeneration = false;
          throw wrap("generation-exists", `generation contains pre-existing content: ${generation.directory}`);
        }
      }

      await ensureDirDeep(this.storage, generation.objectsDirectory);
      let objectIndex = 0;
      for await (const object of input.objects) {
        const reference = descriptor.objects[objectIndex];
        this.validateObjectWrite(descriptor, reference, object, objectIndex);
        const path = this.objectPath(generation, object.path);
        await this.storage.writeBinary!(path, exactArrayBuffer(object.bytes));
        const verified = await this.readAndValidateObject(generation, reference!);
        if (verified.reference.kind === "vector"
          && (verified.block as VectorBlock).dimension !== descriptor.dimension) {
          throw new Error("vector block dimension does not match descriptor");
        }
        objectIndex += 1;
      }
      if (objectIndex !== descriptor.objects.length) {
        throw new Error("generation object writes must exactly match descriptor references");
      }
      await this.storage.writeTextAtomic!(generation.descriptorPath, descriptorText);
      const reread = await this.storage.readText(generation.descriptorPath);
      if (utf8Length(reread) > MAX_GENERATION_DESCRIPTOR_BYTES) throw new Error("generation descriptor exceeds its byte limit");
      const pinned = decodeGenerationDescriptor(reread);
      this.assertDescriptorIdentity(pinned);
      if (descriptorDigest(reread) !== descriptorDigest(descriptorText)) throw new Error("generation descriptor readback differs");

      const candidate = this.createOpenedHandle(pointerFromDescriptor(pinned, reread), pinned, generation);
      await this.validateGenerationClosure(candidate);
      await this.assertSourceRevision(input.sourceCurrentRevision, pinned.sourceRevision);
      await this.options.beforePointerPromotion?.();
      try {
        await this.assertStagingClaim(claimPath, claim);
      } catch (caught) {
        ownershipLost = true;
        throw wrap("generation-conflict", "generation staging claim ownership changed before promotion", caught);
      }

      // The generation-local claim protects immutable staging. This root claim
      // separately owns only the final CURRENT/backup promotion window.
      const promotionClaim = promotionClaimForCandidate(
        pinned,
        input.writerToken,
        input.expectedCurrent,
        this.scopeFingerprint,
        this.identificationFingerprint,
      );
      const promotionClaimText = encodePromotionClaim(promotionClaim);
      await this.acquirePromotionClaim(promotionClaimText);
      try {
        await this.options.afterPromotionClaimAcquired?.();
        const currentRead = await this.readPointer(this.paths.currentPath);
        const current = this.pointerForGuard(currentRead);
        if (!matchesExpected(current, input.expectedCurrent)) {
          throw new FullTextGenerationIndexStoreError(
            "current generation changed before promotion", "stale-current",
            input.expectedCurrent?.sourceRevision ?? null, current?.sourceRevision ?? null,
          );
        }
        await this.assertSourceRevision(input.sourceCurrentRevision, pinned.sourceRevision);
        await this.assertPromotionClaim(promotionClaimText);
        if (current !== null) {
          await this.validateCompleteGeneration(current);
          await this.storage.writeTextAtomic!(this.paths.backupPath, encodeCurrentGenerationPointer(current));
        }
        await this.assertPromotionClaim(promotionClaimText);
        const latest = await this.readPointer(this.paths.currentPath);
        if (!samePointerObservation(latest, currentRead)) {
          throw wrap("concurrent", "current generation pointer changed while promotion claim was held");
        }
        // This is the final awaited guard. The CURRENT commit invocation follows
        // synchronously so no other asynchronous recheck can stale its result.
        await this.assertSourceRevision(input.sourceCurrentRevision, pinned.sourceRevision);
        try {
          await this.storage.writeTextAtomic!(this.paths.currentPath, encodeCurrentGenerationPointer(candidate.pointer));
          committed = true;
        } catch (caught) {
          const outcome = await this.resolveCommitOutcome(pinned);
          if (outcome.kind === "committed") {
            committed = true;
            return outcome.opened;
          }
          if (outcome.kind === "uncertain") {
            throw wrap("commit-uncertain", "CURRENT commit outcome could not be determined", caught);
          }
          throw caught;
        }
        try {
          await this.options.afterPointerPromotion?.();
        } catch (caught) {
          this.warn("post-commit generation promotion observer failed", caught);
        }
        return candidate;
      } finally {
        await this.releasePromotionClaimIfOwned(promotionClaimText);
      }
    } catch (caught) {
      const cleanupForbidden = caught instanceof FullTextGenerationIndexStoreError
        && (caught.code === "claim-uncertain" || caught.code === "commit-uncertain");
      if (!committed && !cleanupForbidden && ownsClaim && !ownershipLost) {
        await this.cleanupGenerationIfOwned(
          generation,
          descriptor.generationId,
          claimPath,
          claimText,
          cleanupWholeGeneration,
        );
      }
      if (caught instanceof FullTextGenerationIndexStoreError) throw caught;
      throw wrap("write-failed", `failed to stage generation ${descriptor.generationId}`, caught);
    }
  }

  private async queueRecovery(observedPrimaryRaw: string | null, primaryError?: unknown): Promise<OpenedFullTextGeneration | null> {
    await this.options.beforeRecoveryQueue?.();
    return enqueueWriter(this.storage, this.paths.currentPath, async () => {
      const current = await this.readPointer(this.paths.currentPath);
      if (current.kind === "incompatible") throw incompatible(`incompatible current generation pointer: ${this.paths.currentPath}`);
      if (current.kind === "valid") {
        if (current.raw !== observedPrimaryRaw) {
          // A same-runtime writer won before recovery acquired the queue.
          return this.openPinned(current.pointer);
        }
        // The pointer is unchanged, but its descriptor may have been repaired.
        const repaired = await this.openPinned(current.pointer).catch(() => null);
        if (repaired) return repaired;
      }
      const currentRaw = current.kind === "valid" || current.kind === "corrupt" ? current.raw : null;
      if (currentRaw !== observedPrimaryRaw) {
        throw wrap("corrupt-or-unreadable", "current generation pointer changed during recovery", current.kind === "corrupt" ? current.error : undefined);
      }
      const backup = await this.readPointer(this.paths.backupPath);
      if (backup.kind === "incompatible") throw incompatible(`incompatible backup generation pointer: ${this.paths.backupPath}`);
      if (backup.kind === "missing" && current.kind === "missing") return null;
      if (backup.kind !== "valid") {
        throw wrap("corrupt-or-unreadable", "current and backup generation pointers are corrupt or unreadable",
          backup.kind === "corrupt" ? backup.error : primaryError);
      }
      requireRecoveryCapabilities(this.storage);
      requireExclusiveCreate(this.storage);
      const promotionClaimText = encodePromotionClaim(recoveryPromotionClaim(backup.pointer, currentRaw));
      await this.acquirePromotionClaim(promotionClaimText);
      try {
        const claimedCurrent = await this.readPointer(this.paths.currentPath);
        if (!samePointerObservation(claimedCurrent, current)) {
          if (claimedCurrent.kind === "valid") return this.openPinned(claimedCurrent.pointer);
          throw wrap("concurrent", "current generation pointer changed before recovery acquired promotion ownership");
        }
        const claimedBackup = await this.readPointer(this.paths.backupPath);
        if (claimedBackup.kind !== "valid" || !samePointerObservation(claimedBackup, backup)) {
          throw wrap("concurrent", "backup generation pointer changed during recovery");
        }
        let opened: OpenedFullTextGeneration;
        try {
          opened = await this.validateCompleteGeneration(claimedBackup.pointer);
        } catch (caught) {
          if (isIncompatible(caught)) throw caught;
          throw wrap("corrupt-or-unreadable", "backup pointer generation is not complete and valid", caught);
        }
        await this.assertPromotionClaim(promotionClaimText);
        const immediatelyBeforeWrite = await this.readPointer(this.paths.currentPath);
        if (!samePointerObservation(immediatelyBeforeWrite, claimedCurrent)) {
          throw wrap("concurrent", "current generation pointer changed immediately before recovery write");
        }
        try {
          await this.storage.writeTextAtomic!(this.paths.currentPath, encodeCurrentGenerationPointer(claimedBackup.pointer));
        } catch (caught) {
          throw wrap("repair-failed", `failed to repair current generation pointer: ${this.paths.currentPath}`, caught);
        }
        this.warn(`full-text generation pointer recovered from backup: ${this.paths.backupPath}`, primaryError);
        return opened;
      } finally {
        await this.releasePromotionClaimIfOwned(promotionClaimText);
      }
    });
  }

  private async openPinned(pointer: CurrentGenerationPointer): Promise<OpenedFullTextGeneration> {
    this.assertPointerIdentity(pointer);
    const generation = deriveFullTextGenerationPaths(
      this.storage, this.output, this.scopeFingerprint, this.identificationFingerprint, pointer.generationId,
    );
    let raw: string;
    try {
      raw = await this.storage.readText(generation.descriptorPath);
    } catch (caught) {
      throw wrap("corrupt-or-unreadable", `cannot read generation descriptor: ${generation.descriptorPath}`, caught);
    }
    if (descriptorDigest(raw) !== pointer.descriptorChecksum) {
      throw wrap("corrupt-or-unreadable", "generation descriptor checksum mismatch");
    }
    let descriptor: GenerationDescriptor;
    try {
      descriptor = decodeGenerationDescriptor(raw);
      this.assertDescriptorIdentity(descriptor);
      validatePairedObjectCoverage(descriptor);
      if (descriptor.generationId !== pointer.generationId || descriptor.sourceRevision !== pointer.sourceRevision) {
        throw new Error("generation pointer and descriptor identity disagree");
      }
    } catch (caught) {
      if (isIncompatible(caught)) throw caught;
      if (/unsupported generation descriptor (?:format|schema) version/i.test(String((caught as Error)?.message))) {
        throw incompatible("incompatible generation descriptor", caught);
      }
      throw wrap("corrupt-or-unreadable", "invalid generation descriptor", caught);
    }
    return this.createOpenedHandle(pointer, descriptor, generation);
  }

  private async validateCompleteGeneration(pointer: CurrentGenerationPointer): Promise<OpenedFullTextGeneration> {
    const opened = await this.openPinned(pointer);
    await this.validateGenerationClosure(opened);
    return opened;
  }

  private async validateGenerationClosure(opened: OpenedFullTextGeneration): Promise<void> {
    const descriptor = opened.descriptor;
    let evidenceState: EvidenceStreamClosureState | null = null;
    const mean = createCanonicalMeanAccumulator(descriptor.dimension);
    const vectors = descriptor.objects.filter((reference) => reference.kind === "vector");
    const evidence = descriptor.objects.filter((reference) => reference.kind === "evidence");
    let previousOrdinal: number | null = null;
    for (let pairIndex = 0; pairIndex < vectors.length; pairIndex += 1) {
      const vectorObject = await opened.readObject(vectors[pairIndex]!);
      const evidenceObject = await opened.readObject(evidence[pairIndex]!);
      const vectorBlock = vectorObject.block as VectorBlock;
      const evidenceBlock = evidenceObject.block as EvidenceBlock;
      mean.add(vectorBlock);
      for (let row = 0; row < vectorBlock.rowCount; row += 1) {
        const ordinal = vectorBlock.paperOrdinals[row]!;
        if (ordinal !== evidenceBlock.records[row]!.paperIndex) {
          throw new Error("vector paper ordinal does not match paired evidence paperIndex");
        }
        if (previousOrdinal === null) {
          if (ordinal !== 0) throw new Error("first vector paper ordinal must be zero");
        } else if (ordinal !== previousOrdinal && ordinal !== previousOrdinal + 1) {
          throw new Error("vector paper ordinals must be continuous across blocks");
        }
        previousOrdinal = ordinal;
      }
      evidenceState = validateEvidenceStreamClosure(evidenceState, evidenceBlock.records);
    }
    finishEvidenceStreamClosure(evidenceState, descriptor.corpusStats.indexedPaperCount);
    const actualPaperCount = previousOrdinal === null ? 0 : previousOrdinal + 1;
    if (actualPaperCount !== descriptor.corpusStats.indexedPaperCount) {
      throw new Error("vector paper ordinals do not match indexedPaperCount");
    }
    const actualMean = mean.finish();
    if (mean.rowCount !== descriptor.corpusStats.chunkCount) {
      throw new Error("vector rows do not match chunkCount");
    }
    if (!exactNumberArrayEqual(actualMean, descriptor.corpusMean)) {
      throw new Error("vector arithmetic mean does not match descriptor corpusMean");
    }
  }

  private createOpenedHandle(
    pointer: CurrentGenerationPointer,
    descriptor: GenerationDescriptor,
    generation: FullTextGenerationPaths,
  ): OpenedFullTextGeneration {
    const privateDescriptor = deepFreeze(cloneDescriptor(descriptor));
    const publicDescriptor = privateDescriptor;
    const references = new Map(privateDescriptor.objects.map((reference) => [referenceIdentity(reference), reference]));
    const diagnostics: GenerationStoreDiagnostics = { maxObjectBytes: 0, objectReads: 0 };
    const pinnedReference = (requested: GenerationObjectReference): GenerationObjectReference => {
      let identity: string;
      try { identity = referenceIdentity(requested); } catch (caught) { throw wrap("invalid", "invalid generation object reference", caught); }
      const pinned = references.get(identity);
      if (!pinned) throw wrap("invalid", "generation object reference is not part of the opened descriptor snapshot");
      return pinned;
    };
    const readRawObject = async (requested: GenerationObjectReference): Promise<VerifiedGenerationObject> => {
      try { return await this.readVerifiedObjectBytes(generation, pinnedReference(requested), diagnostics); }
      catch (caught) { throw normalizePublicReadError("failed to read generation object", caught); }
    };
    const readObject = async (requested: GenerationObjectReference): Promise<OpenedGenerationObject> => {
      try { return await this.readAndValidateObject(generation, pinnedReference(requested), diagnostics, privateDescriptor.dimension); }
      catch (caught) { throw normalizePublicReadError("failed to decode generation object", caught); }
    };
    const iterate = (kind?: GenerationObjectReference["kind"]) =>
      this.iteratePublicGenerationObjects(generation, privateDescriptor, diagnostics, kind);
    return {
      pointer: deepFreeze({ ...pointer }),
      descriptor: publicDescriptor,
      diagnostics,
      readRawObject,
      readObject,
      validateClosure: async () => {
        try {
          await this.validateGenerationClosure({
            pointer, descriptor: privateDescriptor, diagnostics, readRawObject, readObject,
            validateClosure: async () => undefined,
            iterateObjects: iterate,
            iterateVectorBlocks: () => iterate("vector") as AsyncIterable<OpenedVectorObject>,
            iterateEvidenceBlocks: () => iterate("evidence") as AsyncIterable<OpenedEvidenceObject>,
          });
        } catch (caught) {
          throw normalizePublicReadError("generation closure validation failed", caught);
        }
      },
      iterateObjects: iterate,
      iterateVectorBlocks: () => iterate("vector") as AsyncIterable<OpenedVectorObject>,
      iterateEvidenceBlocks: () => iterate("evidence") as AsyncIterable<OpenedEvidenceObject>,
    };
  }

  private async *iteratePublicGenerationObjects(
    generation: FullTextGenerationPaths,
    descriptor: GenerationDescriptor,
    diagnostics: GenerationStoreDiagnostics,
    kind?: GenerationObjectReference["kind"],
  ): AsyncIterable<OpenedGenerationObject> {
    try {
      yield* this.iterateGenerationObjects(generation, descriptor, diagnostics, kind);
    } catch (caught) {
      throw normalizePublicReadError("failed to iterate generation objects", caught);
    }
  }

  private async *iterateGenerationObjects(
    generation: FullTextGenerationPaths,
    descriptor: GenerationDescriptor,
    diagnostics: GenerationStoreDiagnostics,
    kind?: GenerationObjectReference["kind"],
  ): AsyncIterable<OpenedGenerationObject> {
    for (const reference of descriptor.objects) {
      if (kind !== undefined && reference.kind !== kind) continue;
      yield await this.readAndValidateObject(generation, reference, diagnostics, descriptor.dimension);
    }
  }

  private async readVerifiedObjectBytes(
    generation: FullTextGenerationPaths,
    reference: GenerationObjectReference,
    diagnostics?: GenerationStoreDiagnostics,
  ): Promise<VerifiedGenerationObject> {
    requireObjectReadCapability(this.storage);
    validateReferencePath(reference);
    const path = this.objectPath(generation, reference.path);
    let buffer: ArrayBuffer;
    try { buffer = await this.storage.readBinary!(path); }
    catch (caught) { throw wrap("corrupt-or-unreadable", `cannot read generation object: ${path}`, caught); }
    if (buffer.byteLength > MAX_BINARY_OBJECT_BYTES) throw new Error("generation object exceeds its byte limit");
    const bytes = new Uint8Array(buffer);
    if (diagnostics) {
      diagnostics.objectReads += 1;
      diagnostics.maxObjectBytes = Math.max(diagnostics.maxObjectBytes, bytes.byteLength);
    }
    if (bytes.byteLength !== reference.byteLength) throw new Error("generation object byteLength does not match reference");
    if (blockObjectChecksum(bytes) !== reference.checksum) throw new Error("generation object checksum does not match reference");
    return { reference, bytes };
  }

  private async readAndValidateObject(
    generation: FullTextGenerationPaths,
    reference: GenerationObjectReference,
    diagnostics?: GenerationStoreDiagnostics,
    descriptorDimension?: number,
  ): Promise<OpenedGenerationObject> {
    const { bytes } = await this.readVerifiedObjectBytes(generation, reference, diagnostics);
    const path = this.objectPath(generation, reference.path);
    try {
      if (reference.kind === "vector") {
        const block = decodeVectorBlock(bytes);
        if (block.rowStart !== reference.recordStart || block.rowCount !== reference.recordCount) {
          throw new Error("vector block rowStart/count does not match reference");
        }
        if (descriptorDimension !== undefined && block.dimension !== descriptorDimension) {
          throw new Error("vector block dimension does not match descriptor");
        }
        return { reference: reference as GenerationObjectReference & { kind: "vector" }, block };
      }
      const block = decodeEvidenceBlock(bytes);
      if (block.rowStart !== reference.recordStart || block.records.length !== reference.recordCount) {
        throw new Error("evidence block rowStart/count does not match reference");
      }
      return { reference: reference as GenerationObjectReference & { kind: "evidence" }, block };
    } catch (caught) {
      throw wrap("corrupt-or-unreadable", `invalid ${reference.kind} generation object: ${path}`, caught);
    }
  }

  private validateObjectWrite(
    descriptor: GenerationDescriptor,
    reference: GenerationObjectReference | undefined,
    write: GenerationObjectWrite,
    index: number,
  ): void {
    if (!reference || !write || typeof write.path !== "string" || !(write.bytes instanceof Uint8Array)) {
      throw new Error(`invalid generation object write at index ${index}`);
    }
    if (write.path !== reference.path || write.bytes.byteLength !== reference.byteLength
      || blockObjectChecksum(write.bytes) !== reference.checksum) {
      throw new Error(`generation object write disagrees with reference: ${reference.path}`);
    }
    const decoded = reference.kind === "vector" ? decodeVectorBlock(write.bytes) : decodeEvidenceBlock(write.bytes);
    const count = reference.kind === "vector" ? (decoded as VectorBlock).rowCount : (decoded as EvidenceBlock).records.length;
    if (decoded.rowStart !== reference.recordStart || count !== reference.recordCount) {
      throw new Error("generation object metadata disagrees with reference");
    }
    if (reference.kind === "vector" && (decoded as VectorBlock).dimension !== descriptor.dimension) {
      throw new Error("vector block dimension disagrees with descriptor");
    }
  }

  private async loadCurrentForGuard(): Promise<CurrentGenerationPointer | null> {
    const result = await this.readPointer(this.paths.currentPath);
    if (result.kind === "missing") return null;
    if (result.kind === "incompatible") throw incompatible("incompatible current generation pointer");
    if (result.kind !== "valid") throw wrap("corrupt-or-unreadable", "current generation pointer is corrupt or unreadable", result.error);
    this.assertPointerIdentity(result.pointer);
    return result.pointer;
  }

  private async readPointer(path: string): Promise<PointerReadResult> {
    try {
      if (!(await this.storage.exists(path))) return { kind: "missing" };
      const raw = await this.storage.readText(path);
      try {
        const parsed = JSON.parse(raw) as Record<string, unknown>;
        if (typeof parsed.formatVersion === "number" && parsed.formatVersion > CURRENT_GENERATION_POINTER_FORMAT_VERSION) return { kind: "incompatible", raw };
        if (typeof parsed.schemaVersion === "number" && parsed.schemaVersion > CURRENT_GENERATION_POINTER_SCHEMA_VERSION) return { kind: "incompatible", raw };
        return { kind: "valid", pointer: decodeCurrentGenerationPointer(raw), raw };
      } catch (error) {
        return { kind: "corrupt", error, raw };
      }
    } catch (error) {
      return { kind: "corrupt", error, raw: null };
    }
  }

  private assertPointerIdentity(pointer: CurrentGenerationPointer): void {
    if (pointer.scopeFingerprint !== this.scopeFingerprint || pointer.identificationFingerprint !== this.identificationFingerprint) {
      throw incompatible("current generation pointer identity does not match store binding");
    }
  }

  private assertDescriptorIdentity(descriptor: GenerationDescriptor): void {
    if (descriptor.scopeFingerprint !== this.scopeFingerprint || descriptor.identificationFingerprint !== this.identificationFingerprint) {
      throw incompatible("generation descriptor identity does not match store binding");
    }
  }

  private objectPath(generation: FullTextGenerationPaths, logicalPath: string): string {
    return this.storage.normalizePath(`${generation.directory}/${logicalPath}`);
  }

  private async safeExists(path: string): Promise<boolean> {
    try { return await this.storage.exists(path); } catch (caught) { throw wrap("write-failed", `cannot check generation path: ${path}`, caught); }
  }

  private async assertStagingClaim(path: string, expected: StagingClaim): Promise<void> {
    const raw = await this.storage.readText(path);
    const actual = decodeStagingClaim(raw);
    if (encodeStagingClaim(actual) !== encodeStagingClaim(expected)) {
      throw new Error("generation staging claim does not match its writer");
    }
  }

  private pointerForGuard(result: PointerReadResult): CurrentGenerationPointer | null {
    if (result.kind === "missing") return null;
    if (result.kind === "incompatible") throw incompatible("incompatible current generation pointer");
    if (result.kind === "corrupt") {
      throw wrap("corrupt-or-unreadable", "current generation pointer is corrupt or unreadable", result.error);
    }
    this.assertPointerIdentity(result.pointer);
    return result.pointer;
  }

  private async acquirePromotionClaim(claimText: string): Promise<void> {
    await ensureDirDeep(this.storage, this.paths.directory);
    let created: boolean;
    try {
      created = await this.storage.createTextExclusive!(this.paths.promotionClaimPath, claimText);
    } catch (caught) {
      throw wrap("claim-uncertain", `generation promotion claim outcome is uncertain: ${this.paths.promotionClaimPath}`, caught);
    }
    if (!created) {
      // A fixed claim may be live or left by a crashed writer. Core cannot prove
      // either case and therefore never steals it, including by elapsed time.
      throw wrap("concurrent", `generation promotion claim already exists: ${this.paths.promotionClaimPath}`);
    }
  }

  private async assertPromotionClaim(expectedText: string): Promise<void> {
    const actual = await this.storage.readText(this.paths.promotionClaimPath).catch(() => null);
    if (actual !== expectedText) throw wrap("stale-claim", "generation promotion claim ownership changed");
  }

  private async releasePromotionClaimIfOwned(expectedText: string): Promise<void> {
    const actual = await this.storage.readText(this.paths.promotionClaimPath).catch(() => null);
    if (actual !== expectedText) return;
    await this.storage.remove(this.paths.promotionClaimPath).catch((caught) => {
      this.warn("failed to release generation promotion claim", caught);
    });
  }

  private warn(message: string, error?: unknown): void {
    try { this.options.onWarning?.(message, error); } catch { /* diagnostics must not change storage outcome */ }
  }

  private async cleanupGenerationIfOwned(
    generation: FullTextGenerationPaths,
    generationId: string,
    claimPath: string,
    claimText: string,
    cleanupWholeGeneration: boolean,
  ): Promise<void> {
    const actualClaim = await this.storage.readText(claimPath).catch(() => null);
    if (actualClaim !== claimText) return;
    const current = await this.readPointer(this.paths.currentPath);
    // Cleanup is permitted only for a conclusive safe observation. Corrupt,
    // incompatible, or unreadable CURRENT may already contain the candidate.
    if (current.kind === "corrupt" || current.kind === "incompatible") return;
    if (current.kind === "valid" && current.pointer.generationId === generationId) return;
    const cleanupPath = cleanupWholeGeneration ? generation.directory : claimPath;
    await this.storage.remove(cleanupPath).catch(() => undefined);
  }

  private async assertSourceRevision(current: () => number | Promise<number>, expected: number): Promise<void> {
    const actual = await current();
    if (actual !== expected) {
      throw new FullTextGenerationIndexStoreError(
        `stale source revision: expected ${expected}, current ${String(actual)}`,
        "stale-source", expected, validRevisionOrNull(actual),
      );
    }
  }

  private async resolveCommitOutcome(descriptor: GenerationDescriptor): Promise<
    | { readonly kind: "committed"; readonly opened: OpenedFullTextGeneration }
    | { readonly kind: "not-committed" }
    | { readonly kind: "uncertain" }
  > {
    const current = await this.readPointer(this.paths.currentPath);
    if (current.kind === "missing") return { kind: "not-committed" };
    if (current.kind !== "valid") return { kind: "uncertain" };
    if (current.pointer.generationId !== descriptor.generationId
      || current.pointer.sourceRevision !== descriptor.sourceRevision) {
      return { kind: "not-committed" };
    }
    const opened = await this.validateCompleteGeneration(current.pointer).catch(() => null);
    if (!opened || encodeGenerationDescriptor(opened.descriptor) !== encodeGenerationDescriptor(descriptor)) {
      return { kind: "uncertain" };
    }
    return { kind: "committed", opened };
  }

  private async tryExactCommittedReplay(descriptor: GenerationDescriptor): Promise<OpenedFullTextGeneration | null> {
    const current = await this.readPointer(this.paths.currentPath);
    if (current.kind !== "valid" || current.pointer.generationId !== descriptor.generationId
      || current.pointer.sourceRevision !== descriptor.sourceRevision) return null;
    const opened = await this.validateCompleteGeneration(current.pointer).catch(() => null);
    if (!opened || encodeGenerationDescriptor(opened.descriptor) !== encodeGenerationDescriptor(descriptor)) return null;
    return opened;
  }
}

type PointerReadResult =
  | { kind: "missing" }
  | { kind: "valid"; pointer: CurrentGenerationPointer; raw: string }
  | { kind: "incompatible"; raw: string }
  | { kind: "corrupt"; error: unknown; raw: string | null };

export function encodeCurrentGenerationPointer(pointer: CurrentGenerationPointer): string {
  const semantic = validatePointer(pointer, false);
  const checksum = pointerChecksum(semantic);
  return JSON.stringify({ ...semantic, checksum });
}

export function decodeCurrentGenerationPointer(text: string): CurrentGenerationPointer {
  if (typeof text !== "string" || text.length > MAX_CURRENT_POINTER_BYTES || utf8Length(text) > MAX_CURRENT_POINTER_BYTES) {
    throw new Error("current generation pointer exceeds its byte limit");
  }
  let value: unknown;
  try { value = JSON.parse(text); } catch { throw new Error("current generation pointer is not valid JSON"); }
  const pointer = validatePointer(value, true);
  if (pointer.checksum !== pointerChecksum(pointer)) throw new Error("current generation pointer checksum mismatch");
  return pointer;
}

function validatePointer(value: unknown, verifyChecksum: boolean): CurrentGenerationPointer {
  if (!isPlainObject(value)) throw new Error("current generation pointer must be an object");
  const keys = ["formatVersion", "schemaVersion", "generationId", "sourceRevision", "scopeFingerprint", "identificationFingerprint", "descriptorChecksum", "checksum"];
  if (Object.keys(value).length !== keys.length || Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error("current generation pointer has an unknown field or missing required field");
  }
  if (value.formatVersion !== CURRENT_GENERATION_POINTER_FORMAT_VERSION) throw new Error("unsupported current generation pointer format version");
  if (value.schemaVersion !== CURRENT_GENERATION_POINTER_SCHEMA_VERSION) throw new Error("unsupported current generation pointer schema version");
  requireGenerationId(value.generationId);
  if (!Number.isSafeInteger(value.sourceRevision) || (value.sourceRevision as number) < 0) throw new Error("invalid current generation source revision");
  validateFingerprint(value.scopeFingerprint, "scopeFingerprint");
  validateFingerprint(value.identificationFingerprint, "identificationFingerprint");
  validateFingerprint(value.descriptorChecksum, "descriptorChecksum");
  validateFingerprint(value.checksum, "checksum");
  const pointer = value as unknown as CurrentGenerationPointer;
  if (!verifyChecksum) return pointer;
  return pointer;
}

function pointerChecksum(pointer: Omit<CurrentGenerationPointer, "checksum"> | CurrentGenerationPointer): string {
  const semantic = {
    formatVersion: pointer.formatVersion,
    schemaVersion: pointer.schemaVersion,
    generationId: pointer.generationId,
    sourceRevision: pointer.sourceRevision,
    scopeFingerprint: pointer.scopeFingerprint,
    identificationFingerprint: pointer.identificationFingerprint,
    descriptorChecksum: pointer.descriptorChecksum,
  };
  return `sha256:${sha256Hex(JSON.stringify(semantic))}`;
}

function pointerFromDescriptor(descriptor: GenerationDescriptor, raw: string): CurrentGenerationPointer {
  return decodeCurrentGenerationPointer(encodeCurrentGenerationPointer({
    formatVersion: CURRENT_GENERATION_POINTER_FORMAT_VERSION,
    schemaVersion: CURRENT_GENERATION_POINTER_SCHEMA_VERSION,
    generationId: descriptor.generationId,
    sourceRevision: descriptor.sourceRevision,
    scopeFingerprint: descriptor.scopeFingerprint,
    identificationFingerprint: descriptor.identificationFingerprint,
    descriptorChecksum: descriptorDigest(raw),
    checksum: `sha256:${"0".repeat(64)}`,
  }));
}

function descriptorDigest(raw: string): string { return `sha256:${sha256Hex(new TextEncoder().encode(raw))}`; }

export function computeCanonicalVectorMean(
  blocks: Iterable<Pick<VectorBlock, "dimension" | "vectors">>,
  dimension: number,
): number[] {
  const accumulator = createCanonicalMeanAccumulator(dimension);
  for (const block of blocks) accumulator.add(block);
  return accumulator.finish();
}

function createCanonicalMeanAccumulator(dimension: number): {
  readonly rowCount: number;
  add(block: Pick<VectorBlock, "dimension" | "vectors">): void;
  finish(): number[];
} {
  if (!Number.isSafeInteger(dimension) || dimension < 1) throw new Error("canonical vector mean dimension must be positive");
  const sums = new Float64Array(dimension);
  let rows = 0;
  return {
    get rowCount() { return rows; },
    add(block) {
      if (block.dimension !== dimension || block.vectors.length % dimension !== 0) {
        throw new Error("canonical vector mean block dimension is invalid");
      }
      // Generation v1 canonical arithmetic: descriptor vector-ref order, then
      // row-major float32 values, accumulated left-to-right in float64.
      for (let offset = 0; offset < block.vectors.length; offset += dimension) {
        for (let column = 0; column < dimension; column += 1) sums[column]! += block.vectors[offset + column]!;
        rows += 1;
      }
    },
    finish() {
      return Array.from(sums, (sum) => rows === 0 ? 0 : sum / rows);
    },
  };
}

function exactNumberArrayEqual(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => Object.is(value, right[index]));
}

function validateReferencePath(reference: GenerationObjectReference): void {
  if (!reference || typeof reference !== "object"
    || typeof reference.path !== "string"
    || !/^objects\/[a-z0-9][a-z0-9._-]{0,127}$/.test(reference.path)) {
    throw new Error("generation object reference path is invalid");
  }
}

function referenceIdentity(reference: GenerationObjectReference): string {
  validateReferencePath(reference);
  return JSON.stringify({
    kind: reference.kind,
    path: reference.path,
    byteLength: reference.byteLength,
    recordStart: reference.recordStart,
    recordCount: reference.recordCount,
    checksum: reference.checksum,
  });
}

function cloneDescriptor(descriptor: GenerationDescriptor): GenerationDescriptor {
  return JSON.parse(encodeGenerationDescriptor(descriptor)) as GenerationDescriptor;
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value;
  for (const nested of Object.values(value as Record<string, unknown>)) deepFreeze(nested);
  return Object.freeze(value);
}

function stagingClaim(descriptor: GenerationDescriptor, descriptorText: string, writerToken: string): StagingClaim {
  return {
    formatVersion: STAGING_CLAIM_FORMAT_VERSION,
    schemaVersion: STAGING_CLAIM_SCHEMA_VERSION,
    generationId: descriptor.generationId,
    sourceRevision: descriptor.sourceRevision,
    scopeFingerprint: descriptor.scopeFingerprint,
    identificationFingerprint: descriptor.identificationFingerprint,
    descriptorChecksum: descriptorDigest(descriptorText),
    writerToken,
  };
}

function promotionClaimForCandidate(
  descriptor: GenerationDescriptor,
  writerToken: string,
  expectedCurrent: StageAndPromoteGenerationInput["expectedCurrent"],
  scopeFingerprint: string,
  identificationFingerprint: string,
): PromotionClaim {
  return {
    formatVersion: PROMOTION_CLAIM_FORMAT_VERSION,
    schemaVersion: PROMOTION_CLAIM_SCHEMA_VERSION,
    operation: "promote",
    writerToken,
    candidateGenerationId: descriptor.generationId,
    sourceRevision: descriptor.sourceRevision,
    expectedCurrent,
    observedPrimaryChecksum: pointerObservationChecksum(null),
    scopeFingerprint,
    identificationFingerprint,
  };
}

function recoveryPromotionClaim(
  pointer: CurrentGenerationPointer,
  observedPrimaryRaw: string | null,
): PromotionClaim {
  const observation = pointerObservationChecksum(observedPrimaryRaw);
  return {
    formatVersion: PROMOTION_CLAIM_FORMAT_VERSION,
    schemaVersion: PROMOTION_CLAIM_SCHEMA_VERSION,
    operation: "recover",
    writerToken: `recovery-${sha256Hex(`${pointer.generationId}:${observation}`)}`,
    candidateGenerationId: pointer.generationId,
    sourceRevision: pointer.sourceRevision,
    expectedCurrent: null,
    observedPrimaryChecksum: observation,
    scopeFingerprint: pointer.scopeFingerprint,
    identificationFingerprint: pointer.identificationFingerprint,
  };
}

function encodePromotionClaim(claim: PromotionClaim): string {
  return JSON.stringify(validatePromotionClaim(claim));
}

function validatePromotionClaim(value: unknown): PromotionClaim {
  if (!isPlainObject(value)) throw new Error("generation promotion claim must be an object");
  const keys = ["formatVersion", "schemaVersion", "operation", "writerToken", "candidateGenerationId",
    "sourceRevision", "expectedCurrent", "observedPrimaryChecksum", "scopeFingerprint", "identificationFingerprint"];
  if (Object.keys(value).length !== keys.length || Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error("generation promotion claim has an unknown or missing field");
  }
  if (value.formatVersion !== PROMOTION_CLAIM_FORMAT_VERSION) throw new Error("unsupported promotion claim format version");
  if (value.schemaVersion !== PROMOTION_CLAIM_SCHEMA_VERSION) throw new Error("unsupported promotion claim schema version");
  if (value.operation !== "promote" && value.operation !== "recover") throw new Error("invalid promotion claim operation");
  validateWriterToken(value.writerToken);
  requireGenerationId(value.candidateGenerationId);
  if (!Number.isSafeInteger(value.sourceRevision) || (value.sourceRevision as number) < 0) throw new Error("invalid promotion claim source revision");
  validateExpectedCurrent(value.expectedCurrent as StageAndPromoteGenerationInput["expectedCurrent"]);
  validateFingerprint(value.observedPrimaryChecksum, "observedPrimaryChecksum");
  validateFingerprint(value.scopeFingerprint, "scopeFingerprint");
  validateFingerprint(value.identificationFingerprint, "identificationFingerprint");
  return value as unknown as PromotionClaim;
}

function pointerObservationChecksum(raw: string | null): string {
  return `sha256:${sha256Hex(raw === null ? "missing" : raw)}`;
}

function samePointerObservation(actual: PointerReadResult, expected: PointerReadResult): boolean {
  return pointerReadRaw(actual) === pointerReadRaw(expected);
}

function pointerReadRaw(result: PointerReadResult): string | null {
  return result.kind === "valid" || result.kind === "corrupt" || result.kind === "incompatible" ? result.raw : null;
}

function encodeStagingClaim(claim: StagingClaim): string {
  return JSON.stringify(validateStagingClaim(claim));
}

function decodeStagingClaim(raw: string): StagingClaim {
  let value: unknown;
  try { value = JSON.parse(raw); } catch { throw new Error("generation staging claim is not valid JSON"); }
  return validateStagingClaim(value);
}

function validateStagingClaim(value: unknown): StagingClaim {
  if (!isPlainObject(value)) throw new Error("generation staging claim must be an object");
  const keys = ["formatVersion", "schemaVersion", "generationId", "sourceRevision", "scopeFingerprint",
    "identificationFingerprint", "descriptorChecksum", "writerToken"];
  if (Object.keys(value).length !== keys.length || Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error("generation staging claim has an unknown or missing field");
  }
  if (value.formatVersion !== STAGING_CLAIM_FORMAT_VERSION) throw new Error("unsupported staging claim format version");
  if (value.schemaVersion !== STAGING_CLAIM_SCHEMA_VERSION) throw new Error("unsupported staging claim schema version");
  requireGenerationId(value.generationId);
  if (!Number.isSafeInteger(value.sourceRevision) || (value.sourceRevision as number) < 0) throw new Error("invalid staging claim source revision");
  validateFingerprint(value.scopeFingerprint, "scopeFingerprint");
  validateFingerprint(value.identificationFingerprint, "identificationFingerprint");
  validateFingerprint(value.descriptorChecksum, "descriptorChecksum");
  validateWriterToken(value.writerToken);
  return value as unknown as StagingClaim;
}

function validateWriterToken(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[a-z0-9](?:[a-z0-9-]{30,126}[a-z0-9])$/.test(value)) {
    throw new Error("writerToken must contain 32-128 lowercase alphanumeric/hyphen characters");
  }
}

function validatePairedObjectCoverage(descriptor: GenerationDescriptor): void {
  const vectors = descriptor.objects.filter((reference) => reference.kind === "vector");
  const evidence = descriptor.objects.filter((reference) => reference.kind === "evidence");
  if (vectors.length !== evidence.length) throw new Error("vector/evidence block ordinals must have equal coverage counts");
  for (let ordinal = 0; ordinal < vectors.length; ordinal += 1) {
    if (vectors[ordinal]!.recordStart !== evidence[ordinal]!.recordStart
      || vectors[ordinal]!.recordCount !== evidence[ordinal]!.recordCount) {
      throw new Error("same-ordinal vector/evidence block coverage must match");
    }
  }
}

function matchesExpected(current: CurrentGenerationPointer | null, expected: StageAndPromoteGenerationInput["expectedCurrent"]): boolean {
  return expected === null
    ? current === null
    : current?.generationId === expected.generationId && current.sourceRevision === expected.sourceRevision;
}

function validateExpectedCurrent(value: StageAndPromoteGenerationInput["expectedCurrent"]): void {
  if (value === null) return;
  requireGenerationId(value.generationId);
  if (!Number.isSafeInteger(value.sourceRevision) || value.sourceRevision < 0) throw new Error("invalid expected current source revision");
}

function requireExclusiveCreate(storage: StorageAdapter): void {
  if (!storage.createTextExclusive) {
    throw wrap("capability-unsupported", "generation staging requires createTextExclusive capability");
  }
}

function requireWriteCapabilities(storage: StorageAdapter): void {
  if (!storage.readBinary || !storage.writeBinary || !storage.writeTextAtomic) {
    throw wrap("capability-unsupported", "generation promotion requires readBinary, writeBinary, and writeTextAtomic capabilities");
  }
}

function requireRecoveryCapabilities(storage: StorageAdapter): void {
  if (!storage.readBinary || !storage.writeTextAtomic) {
    throw wrap("capability-unsupported", "generation recovery requires readBinary and writeTextAtomic capabilities");
  }
}

function requireObjectReadCapability(storage: StorageAdapter): void {
  if (!storage.readBinary) {
    throw wrap("capability-unsupported", "generation object reads require readBinary capability");
  }
}

function normalizePublicReadError(message: string, caught: unknown): FullTextGenerationIndexStoreError {
  if (caught instanceof FullTextGenerationIndexStoreError) return caught;
  return wrap("corrupt-or-unreadable", message, caught);
}

function enqueueWriter<T>(storage: StorageAdapter, path: string, operation: () => Promise<T>): Promise<T> {
  let queues = writerQueues.get(storage);
  if (!queues) { queues = new Map(); writerQueues.set(storage, queues); }
  const previous = queues.get(path) ?? Promise.resolve();
  const next = previous.catch(() => undefined).then(operation);
  const tail = next.then(() => undefined, () => undefined);
  queues.set(path, tail);
  void tail.finally(() => { if (queues?.get(path) === tail) queues.delete(path); });
  return next;
}

async function ensureDirDeep(storage: StorageAdapter, dir: string): Promise<void> {
  const parts = storage.normalizePath(dir).split("/").filter(Boolean);
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    if (!(await storage.exists(current))) await storage.mkdir(current);
  }
}

function exactArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.slice().buffer;
}

function requireGenerationId(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/.test(value)) {
    throw new Error("invalid generationId");
  }
}

function validateFingerprint(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !/^sha256:[a-f0-9]{64}$/.test(value)) throw new Error(`${name} must be a SHA-256 fingerprint`);
}

function utf8Length(value: string): number { return new TextEncoder().encode(value).byteLength; }
function validRevisionOrNull(value: unknown): number | null { return Number.isSafeInteger(value) && (value as number) >= 0 ? value as number : null; }
function isPlainObject(value: unknown): value is Record<string, unknown> { return typeof value === "object" && value !== null && !Array.isArray(value); }
function isIterable(value: unknown): value is Iterable<unknown> {
  return typeof value === "object" && value !== null && Symbol.iterator in value;
}
function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return typeof value === "object" && value !== null && Symbol.asyncIterator in value;
}
function isIncompatible(value: unknown): boolean { return value instanceof FullTextGenerationIndexStoreError && value.code === "incompatible"; }
function incompatible(message: string, cause?: unknown): FullTextGenerationIndexStoreError { return wrap("incompatible", message, cause); }
function wrap(code: FullTextGenerationIndexStoreErrorCode, message: string, cause?: unknown): FullTextGenerationIndexStoreError {
  return new FullTextGenerationIndexStoreError(message, code, undefined, undefined, cause === undefined ? {} : { cause });
}
