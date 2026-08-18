import type { StorageAdapter } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import { sha256Hex } from "../../utils/digest";
import { tokenizeUnicode, tokenizeUnicodeWithHanSingles } from "./bm25-retrieval";
import {
  MAX_BINARY_OBJECT_BYTES,
  MAX_GENERATION_DESCRIPTOR_BYTES,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeGenerationDescriptor,
  decodeLexicalDictionaryBlock,
  decodeLexicalPostingsBlock,
  decodePaperMetadataBlock,
  decodeVectorBlock,
  deriveFullTextGenerationIndexPaths,
  deriveFullTextGenerationPaths,
  encodeGenerationDescriptor,
  finishEvidenceStreamClosure,
  validateEvidenceStreamClosure,
  type EvidenceBlock,
  type EvidenceBlockRecord,
  type EvidenceStreamClosureState,
  type FullTextGenerationPaths,
  type GenerationDescriptor,
  type GenerationObjectReference,
  type LexicalDictionaryBlock,
  type LexicalDictionaryEntry,
  type LexicalNamespace,
  type LexicalOccurrence,
  type LexicalPostingsBlock,
  type PaperMetadataBlock,
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
  /** Test diagnostic: lexical closure never retains more than one block per zipper side. */
  maxLiveBlocks: number;
}

export interface VerifiedGenerationObject {
  readonly reference: GenerationObjectReference;
  readonly bytes: Uint8Array;
}

export type OpenedGenerationObject =
  | { readonly reference: GenerationObjectReference & { readonly kind: "vector" }; readonly block: VectorBlock }
  | { readonly reference: GenerationObjectReference & { readonly kind: "evidence" }; readonly block: EvidenceBlock }
  | { readonly reference: GenerationObjectReference & { readonly kind: "paper-metadata" }; readonly block: PaperMetadataBlock }
  | { readonly reference: GenerationObjectReference & { readonly kind: "lexical-dictionary" }; readonly block: LexicalDictionaryBlock }
  | { readonly reference: GenerationObjectReference & { readonly kind: "lexical-postings" }; readonly block: LexicalPostingsBlock };

export type OpenedVectorObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "vector" } }>;
export type OpenedEvidenceObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "evidence" } }>;
export type OpenedPaperMetadataObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "paper-metadata" } }>;
export type OpenedLexicalDictionaryObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "lexical-dictionary" } }>;
export type OpenedLexicalPostingsObject = Extract<OpenedGenerationObject, { readonly reference: { readonly kind: "lexical-postings" } }>;

export interface OpenedFullTextGeneration {
  readonly pointer: CurrentGenerationPointer;
  readonly descriptor: GenerationDescriptor;
  readonly diagnostics: GenerationStoreDiagnostics;
  readRawObject(reference: GenerationObjectReference): Promise<VerifiedGenerationObject>;
  readObject(reference: GenerationObjectReference): Promise<OpenedGenerationObject>;
  readPaperMetadata(reference: GenerationObjectReference & { readonly kind: "paper-metadata" }): Promise<OpenedPaperMetadataObject>;
  readLexicalDictionary(reference: GenerationObjectReference & { readonly kind: "lexical-dictionary" }): Promise<OpenedLexicalDictionaryObject>;
  readLexicalPostings(reference: GenerationObjectReference & { readonly kind: "lexical-postings" }): Promise<OpenedLexicalPostingsObject>;
  /** Validate dense and, for schema v4 BM25 generations, exact cross-object lexical closure. */
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
        const verified = await this.readAndValidateObject(generation, reference!, undefined, descriptor.dimension, descriptor);
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
    await this.validateDenseClosure(opened);
    await this.validateLexicalClosure(opened);
  }

  private async validateDenseClosure(opened: OpenedFullTextGeneration): Promise<void> {
    const descriptor = opened.descriptor;
    let evidenceState: EvidenceStreamClosureState | null = null;
    const mean = createCanonicalMeanAccumulator(descriptor.dimension);
    const vectors = descriptor.objects.filter((reference) => reference.kind === "vector");
    const evidence = descriptor.objects.filter((reference) => reference.kind === "evidence");
    let previousOrdinal: number | null = null;
    for (let pairIndex = 0; pairIndex < vectors.length; pairIndex += 1) {
      const vectorBlock = (await opened.readObject(vectors[pairIndex]!)).block as VectorBlock;
      const evidenceBlock = (await opened.readObject(evidence[pairIndex]!)).block as EvidenceBlock;
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
    if (mean.rowCount !== descriptor.corpusStats.chunkCount) throw new Error("vector rows do not match chunkCount");
    if (!exactNumberArrayEqual(actualMean, descriptor.corpusMean)) throw new Error("vector arithmetic mean does not match descriptor corpusMean");
  }

  private async validateLexicalClosure(opened: OpenedFullTextGeneration): Promise<void> {
    if (opened.descriptor.schemaVersion !== 4 || opened.descriptor.lexicalCapability === "none") return;
    await this.validateLexicalMetadata(opened);
    if (!opened.descriptor.objects.some((reference) => reference.kind === "lexical-postings")) return;
    await this.validateEvidencePostings(opened);
    await this.validatePostingsDictionary(opened);
  }

  private async validateLexicalMetadata(opened: OpenedFullTextGeneration): Promise<void> {
    const metadataRefs = refsOfKind(opened.descriptor, "paper-metadata");
    const evidenceRefs = refsOfKind(opened.descriptor, "evidence");
    let metadataRefIndex = 0; let evidenceRefIndex = 0;
    let metadataBlock: PaperMetadataBlock | null = null; let evidenceBlock: EvidenceBlock | null = null;
    let metadataIndex = 0; let evidenceIndex = 0; let chunkOrdinal = 0;
    let total = 0; let expandedTotal = 0;
    while (metadataRefIndex < metadataRefs.length || metadataBlock !== null) {
      if (metadataBlock === null) {
        metadataBlock = (await opened.readPaperMetadata(metadataRefs[metadataRefIndex++]!)).block;
        opened.diagnostics.maxLiveBlocks = Math.max(opened.diagnostics.maxLiveBlocks, 1 + Number(evidenceBlock !== null));
      }
      const metadata = metadataBlock.records[metadataIndex]!;
      let chunksSeen = 0;
      while (chunksSeen < metadata.chunkCount) {
        if (evidenceBlock === null) {
          if (evidenceRefIndex >= evidenceRefs.length) throw new Error("metadata coverage exceeds evidence EOF");
          evidenceBlock = (await opened.readObject(evidenceRefs[evidenceRefIndex++]!)).block as EvidenceBlock;
          evidenceIndex = 0;
          opened.diagnostics.maxLiveBlocks = Math.max(opened.diagnostics.maxLiveBlocks, 1 + Number(metadataBlock !== null));
        }
        const evidence = evidenceBlock.records[evidenceIndex]!;
        if (metadata.paperOrdinal !== evidence.paperIndex || metadata.paperKey !== evidence.paperKey
          || metadata.chunkStart !== chunkOrdinal - chunksSeen || evidence.chunk.index !== chunksSeen) {
          throw new Error("paper metadata identity or chunk coverage does not exactly match evidence");
        }
        total += tokenizeUnicode(evidence.chunk.text).length;
        expandedTotal += tokenizeUnicodeWithHanSingles(evidence.chunk.text).length;
        chunksSeen += 1; chunkOrdinal += 1; evidenceIndex += 1;
        if (evidenceIndex === evidenceBlock.records.length) { evidenceBlock = null; evidenceIndex = 0; }
      }
      metadataIndex += 1;
      if (metadataIndex === metadataBlock.records.length) { metadataBlock = null; metadataIndex = 0; }
    }
    if (metadataBlock !== null || evidenceBlock !== null || evidenceRefIndex !== evidenceRefs.length
      || chunkOrdinal !== opened.descriptor.corpusStats.chunkCount) throw new Error("metadata/evidence closure has trailing or missing records");
    const stats = opened.descriptor.corpusStats;
    if (total !== stats.totalLexicalTokenCount || expandedTotal !== stats.totalLexicalTokenCountWithHanSingles
      || !Object.is(stats.avgdl, total / stats.chunkCount)
      || !Object.is(stats.avgdlWithHanSingles, expandedTotal / stats.chunkCount)) {
      throw new Error("lexical statistics do not exactly match accepted tokenizer output");
    }
  }

  private async validateEvidencePostings(opened: OpenedFullTextGeneration): Promise<void> {
    const evidenceRefs = refsOfKind(opened.descriptor, "evidence");
    const postingRefs = refsOfKind(opened.descriptor, "lexical-postings");
    let evidenceRefIndex = 0; let postingRefIndex = 0;
    let evidenceBlock: EvidenceBlock | null = null; let postingBlock: LexicalPostingsBlock | null = null;
    let evidenceIndex = 0; let postingChunkIndex = 0; let occurrenceIndex = 0; let chunkOrdinal = 0;
    while (evidenceRefIndex < evidenceRefs.length || evidenceBlock !== null) {
      if (evidenceBlock === null) {
        evidenceBlock = (await opened.readObject(evidenceRefs[evidenceRefIndex++]!)).block as EvidenceBlock;
        evidenceIndex = 0;
        opened.diagnostics.maxLiveBlocks = Math.max(opened.diagnostics.maxLiveBlocks, 1 + Number(postingBlock !== null));
      }
      if (postingBlock === null) {
        if (postingRefIndex >= postingRefs.length) throw new Error("postings reached EOF before evidence");
        postingBlock = (await opened.readLexicalPostings(postingRefs[postingRefIndex++]!)).block;
        postingChunkIndex = 0; occurrenceIndex = 0;
        opened.diagnostics.maxLiveBlocks = Math.max(opened.diagnostics.maxLiveBlocks, 1 + Number(evidenceBlock !== null));
      }
      const evidence = evidenceBlock.records[evidenceIndex]!;
      const chunk = postingBlock.chunks[postingChunkIndex]!;
      if (postingBlock.chunkStart + postingChunkIndex !== chunkOrdinal
        || chunk.paperOrdinal !== evidence.paperIndex || chunk.chunkIndex !== evidence.chunk.index) {
        throw new Error("postings chunk identity does not exactly match evidence");
      }
      const derived = deriveLexicalChunk(evidence.chunk.text, chunkOrdinal);
      if (chunk.baseLength !== derived.baseLength || chunk.expandedLength !== derived.expandedLength
        || chunk.compactText !== derived.compactText) throw new Error("postings chunk lexical metadata does not match evidence");
      for (const expected of derived.occurrences) {
        const actual = postingBlock.occurrences[occurrenceIndex++];
        if (!actual || !sameOccurrence(actual, expected)) throw new Error("postings occurrence authority does not exactly match evidence");
      }
      if (postingBlock.occurrences[occurrenceIndex]?.chunkOrdinal === chunkOrdinal) {
        throw new Error("postings contains an extra occurrence for evidence chunk");
      }
      evidenceIndex += 1; postingChunkIndex += 1; chunkOrdinal += 1;
      if (evidenceIndex === evidenceBlock.records.length) { evidenceBlock = null; evidenceIndex = 0; }
      if (postingChunkIndex === postingBlock.chunks.length) {
        if (occurrenceIndex !== postingBlock.occurrences.length) throw new Error("postings occurrence stream has trailing records");
        postingBlock = null; postingChunkIndex = 0; occurrenceIndex = 0;
      }
    }
    if (postingBlock !== null || postingRefIndex !== postingRefs.length
      || chunkOrdinal !== opened.descriptor.corpusStats.chunkCount) throw new Error("evidence/postings closure has trailing or missing chunks");
  }

  private async validatePostingsDictionary(opened: OpenedFullTextGeneration): Promise<void> {
    const postingRefs = refsOfKind(opened.descriptor, "lexical-postings");
    const dictionaryRefs = refsOfKind(opened.descriptor, "lexical-dictionary");
    let dictionaryRefIndex = 0; let dictionaryBlock: LexicalDictionaryBlock | null = null; let dictionaryIndex = 0;
    const nextDictionaryEntry = async (): Promise<LexicalDictionaryEntry | null> => {
      while (dictionaryBlock === null || dictionaryIndex === dictionaryBlock.entries.length) {
        dictionaryBlock = null; dictionaryIndex = 0;
        if (dictionaryRefIndex >= dictionaryRefs.length) return null;
        dictionaryBlock = (await opened.readLexicalDictionary(dictionaryRefs[dictionaryRefIndex++]!)).block;
        opened.diagnostics.maxLiveBlocks = Math.max(opened.diagnostics.maxLiveBlocks, 2);
        if (dictionaryBlock.entries.length === 0) continue;
      }
      return dictionaryBlock.entries[dictionaryIndex++]!;
    };
    for (let postingOrdinal = 0; postingOrdinal < postingRefs.length; postingOrdinal += 1) {
      const postings = (await opened.readLexicalPostings(postingRefs[postingOrdinal]!)).block;
      opened.diagnostics.maxLiveBlocks = Math.max(opened.diagnostics.maxLiveBlocks, 1 + Number(dictionaryBlock !== null));
      let catalogIndex = 0;
      while (catalogIndex < postings.termCatalog.length) {
        const first = postings.occurrences[postings.termCatalog[catalogIndex]!]!;
        let chunkDf = 0; let totalTf = 0;
        do {
          const occurrence = postings.occurrences[postings.termCatalog[catalogIndex]!]!;
          chunkDf += 1; totalTf += occurrence.tf; catalogIndex += 1;
        } while (catalogIndex < postings.termCatalog.length
          && sameNamespaceTerm(first, postings.occurrences[postings.termCatalog[catalogIndex]!]!));
        const actual = await nextDictionaryEntry();
        if (!actual || actual.postingOrdinal !== postingOrdinal || actual.namespace !== first.namespace
          || actual.term !== first.term || actual.chunkDf !== chunkDf || actual.totalTf !== totalTf) {
          throw new Error("dictionary route authority does not exactly match postings term catalog");
        }
      }
    }
    if (await nextDictionaryEntry() !== null || dictionaryRefIndex !== dictionaryRefs.length) {
      throw new Error("postings/dictionary closure has trailing entries or pages");
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
    const diagnostics: GenerationStoreDiagnostics = { maxObjectBytes: 0, objectReads: 0, maxLiveBlocks: 0 };
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
      try { return await this.readAndValidateObject(generation, pinnedReference(requested), diagnostics, privateDescriptor.dimension, privateDescriptor); }
      catch (caught) { throw normalizePublicReadError("failed to decode generation object", caught); }
    };
    const typedRead = async <K extends OpenedGenerationObject["reference"]["kind"]>(
      reference: GenerationObjectReference & { readonly kind: K }, kind: K,
    ): Promise<Extract<OpenedGenerationObject, { readonly reference: { readonly kind: K } }>> => {
      const object = await readObject(reference);
      if (object.reference.kind !== kind) throw wrap("invalid", `generation object is not ${kind}`);
      return object as Extract<OpenedGenerationObject, { readonly reference: { readonly kind: K } }>;
    };
    const readPaperMetadata = (reference: GenerationObjectReference & { readonly kind: "paper-metadata" }) => typedRead(reference, "paper-metadata");
    const readLexicalDictionary = (reference: GenerationObjectReference & { readonly kind: "lexical-dictionary" }) => typedRead(reference, "lexical-dictionary");
    const readLexicalPostings = (reference: GenerationObjectReference & { readonly kind: "lexical-postings" }) => typedRead(reference, "lexical-postings");
    const iterate = (kind?: GenerationObjectReference["kind"]) =>
      this.iteratePublicGenerationObjects(generation, privateDescriptor, diagnostics, kind);
    return {
      pointer: deepFreeze({ ...pointer }),
      descriptor: publicDescriptor,
      diagnostics,
      readRawObject,
      readObject,
      readPaperMetadata,
      readLexicalDictionary,
      readLexicalPostings,
      validateClosure: async () => {
        try {
          await this.validateGenerationClosure({
            pointer, descriptor: privateDescriptor, diagnostics, readRawObject, readObject,
            readPaperMetadata, readLexicalDictionary, readLexicalPostings,
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
      yield await this.readAndValidateObject(generation, reference, diagnostics, descriptor.dimension, descriptor);
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
    descriptor?: GenerationDescriptor,
  ): Promise<OpenedGenerationObject> {
    const { bytes } = await this.readVerifiedObjectBytes(generation, reference, diagnostics);
    const path = this.objectPath(generation, reference.path);
    try {
      if (reference.kind === "vector") {
        const block = decodeVectorBlock(bytes, descriptor?.schemaVersion === 2 ? 2 : 4);
        if (block.rowStart !== reference.recordStart || block.rowCount !== reference.recordCount) {
          throw new Error("vector block rowStart/count does not match reference");
        }
        if (descriptorDimension !== undefined && block.dimension !== descriptorDimension) {
          throw new Error("vector block dimension does not match descriptor");
        }
        return { reference: reference as GenerationObjectReference & { kind: "vector" }, block };
      }
      if (reference.kind === "evidence") {
        const block = decodeEvidenceBlock(bytes, descriptor?.schemaVersion === 2 ? 2 : 4);
        if (block.rowStart !== reference.recordStart || block.records.length !== reference.recordCount) throw new Error("evidence block rowStart/count does not match reference");
        return { reference: reference as GenerationObjectReference & { kind: "evidence" }, block };
      }
      if (reference.kind === "paper-metadata") {
        const block = decodePaperMetadataBlock(bytes);
        if (block.paperStart !== reference.recordStart || block.records.length !== reference.recordCount) throw new Error("metadata block start/count does not match reference");
        return { reference: reference as GenerationObjectReference & { kind: "paper-metadata" }, block };
      }
      if (reference.kind === "lexical-dictionary") {
        const block = decodeLexicalDictionaryBlock(bytes);
        if (block.postingStart !== reference.recordStart || block.postingCount !== reference.recordCount) throw new Error("dictionary block postingStart/count does not match reference");
        const dictionaryRefs = descriptor?.objects.filter((candidate) => candidate.kind === "lexical-dictionary") ?? [];
        const ordinal = dictionaryRefs.findIndex((candidate) => candidate.path === reference.path);
        if (ordinal < 0 || block.dictionaryOrdinal !== ordinal) throw new Error("dictionary block dictionaryOrdinal does not match descriptor order");
        if (descriptor) validateDictionaryRouting(block, reference, descriptor);
        return { reference: reference as GenerationObjectReference & { kind: "lexical-dictionary" }, block };
      }
      const block = decodeLexicalPostingsBlock(bytes);
      if (block.chunkStart !== reference.recordStart || block.chunks.length !== reference.recordCount) throw new Error("postings block chunkStart/count does not match reference");
      const postingRefs = descriptor?.objects.filter((candidate) => candidate.kind === "lexical-postings") ?? [];
      const ordinal = postingRefs.findIndex((candidate) => candidate.path === reference.path);
      if (ordinal < 0 || block.postingOrdinal !== ordinal) throw new Error("postings block postingOrdinal does not match descriptor order");
      return { reference: reference as GenerationObjectReference & { kind: "lexical-postings" }, block };
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
    const decoded = reference.kind === "vector" ? decodeVectorBlock(write.bytes)
      : reference.kind === "evidence" ? decodeEvidenceBlock(write.bytes)
        : reference.kind === "paper-metadata" ? decodePaperMetadataBlock(write.bytes)
          : reference.kind === "lexical-dictionary" ? decodeLexicalDictionaryBlock(write.bytes)
            : decodeLexicalPostingsBlock(write.bytes);
    const count = reference.kind === "vector" ? (decoded as VectorBlock).rowCount
      : reference.kind === "evidence" ? (decoded as EvidenceBlock).records.length
        : reference.kind === "paper-metadata" ? (decoded as PaperMetadataBlock).records.length
          : reference.kind === "lexical-dictionary" ? (decoded as LexicalDictionaryBlock).postingCount
            : (decoded as LexicalPostingsBlock).chunks.length;
    const start = reference.kind === "vector" || reference.kind === "evidence" ? (decoded as VectorBlock | EvidenceBlock).rowStart
      : reference.kind === "paper-metadata" ? (decoded as PaperMetadataBlock).paperStart
        : reference.kind === "lexical-dictionary" ? (decoded as LexicalDictionaryBlock).postingStart
          : (decoded as LexicalPostingsBlock).chunkStart;
    if (start !== reference.recordStart || count !== reference.recordCount) throw new Error("generation object metadata disagrees with reference");
    if (reference.kind === "vector" && (decoded as VectorBlock).dimension !== descriptor.dimension) throw new Error("vector block dimension disagrees with descriptor");
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
        requireCurrentPointerTextWithinLimit(raw);
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
  requireCurrentPointerTextWithinLimit(text);
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

function refsOfKind<K extends GenerationObjectReference["kind"]>(
  descriptor: GenerationDescriptor,
  kind: K,
): Array<GenerationObjectReference & { readonly kind: K }> {
  return descriptor.objects.filter(
    (reference): reference is GenerationObjectReference & { readonly kind: K } => reference.kind === kind,
  );
}

function deriveLexicalChunk(text: string, chunkOrdinal: number): {
  readonly baseLength: number;
  readonly expandedLength: number;
  readonly compactText: string;
  readonly occurrences: readonly LexicalOccurrence[];
} {
  const base = tokenizeUnicode(text);
  const expanded = tokenizeUnicodeWithHanSingles(text);
  const compactText = text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, "");
  const occurrences: LexicalOccurrence[] = [];
  const appendFrequencies = (namespace: LexicalNamespace, tokens: readonly string[]) => {
    const frequencies = new Map<string, number>();
    for (const token of tokens) {
      frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
      if (occurrences.length + frequencies.size > 65_536) {
        throw new Error("evidence-derived lexical occurrences exceed 65536 per chunk");
      }
    }
    for (const [term, tf] of frequencies) occurrences.push({ chunkOrdinal, namespace, term, tf });
  };
  appendFrequencies("base", base);
  appendFrequencies("expanded", expanded);
  const characters = Array.from(compactText);
  const grams = new Set<string>();
  for (const size of [1, 2, 3]) {
    for (let offset = 0; offset + size <= characters.length; offset += 1) {
      grams.add(characters.slice(offset, offset + size).join(""));
      if (occurrences.length + grams.size > 65_536) {
        throw new Error("evidence-derived lexical occurrences exceed 65536 per chunk");
      }
    }
  }
  for (const term of grams) occurrences.push({ chunkOrdinal, namespace: "alias", term, tf: 1 });
  if (occurrences.length > 65_536) throw new Error("evidence-derived lexical occurrences exceed 65536 per chunk");
  occurrences.sort(compareLexicalOccurrences);
  return { baseLength: base.length, expandedLength: expanded.length, compactText, occurrences };
}

const LEXICAL_NAMESPACE_ORDER: Record<LexicalNamespace, number> = { alias: 0, base: 1, expanded: 2 };
function compareLexicalOccurrences(left: LexicalOccurrence, right: LexicalOccurrence): number {
  return left.chunkOrdinal - right.chunkOrdinal
    || LEXICAL_NAMESPACE_ORDER[left.namespace] - LEXICAL_NAMESPACE_ORDER[right.namespace]
    || compareUtf8Strings(left.term, right.term);
}
function compareUtf8Strings(left: string, right: string): number {
  const encoder = new TextEncoder(); const a = encoder.encode(left); const b = encoder.encode(right);
  const length = Math.min(a.length, b.length);
  for (let index = 0; index < length; index += 1) if (a[index] !== b[index]) return a[index]! - b[index]!;
  return a.length - b.length;
}
function sameOccurrence(left: LexicalOccurrence, right: LexicalOccurrence): boolean {
  return left.chunkOrdinal === right.chunkOrdinal && left.namespace === right.namespace
    && left.term === right.term && left.tf === right.tf;
}
function sameNamespaceTerm(left: LexicalOccurrence, right: LexicalOccurrence): boolean {
  return left.namespace === right.namespace && left.term === right.term;
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
  return {
    ...descriptor,
    corpusMean: [...descriptor.corpusMean],
    corpusStats: { ...descriptor.corpusStats },
    lexicalRouting: descriptor.lexicalRouting.map((route) => [...route]),
    indexDerivation: { ...descriptor.indexDerivation },
    objects: descriptor.objects.map((reference) => ({ ...reference })),
  };
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

function validateDictionaryRouting(block: LexicalDictionaryBlock, reference: GenerationObjectReference, descriptor: GenerationDescriptor): void {
  const actual = new Set<number>();
  for (const entryIndex of block.queryCatalog) {
    const entry = block.entries[entryIndex]!;
    actual.add(lexicalBucket(entry.namespace, entry.term));
  }
  const routed = new Set<number>();
  for (let bucket = 0; bucket < descriptor.lexicalRouting.length; bucket += 1) {
    if (descriptor.lexicalRouting[bucket]!.includes(reference.path)) routed.add(bucket);
  }
  if (actual.size !== routed.size || [...actual].some((bucket) => !routed.has(bucket))) throw new Error("dictionary routing membership does not exactly match queryCatalog buckets");
}

function lexicalBucket(namespace: string, term: string): number {
  const encoder = new TextEncoder(); const a = encoder.encode(namespace); const b = encoder.encode(term); const bytes = new Uint8Array(a.length + 1 + b.length); bytes.set(a); bytes.set(b, a.length + 1); return Number.parseInt(sha256Hex(bytes).slice(0, 2), 16);
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
function requireCurrentPointerTextWithinLimit(value: unknown): asserts value is string {
  if (typeof value !== "string" || value.length > MAX_CURRENT_POINTER_BYTES
    || utf8Length(value) > MAX_CURRENT_POINTER_BYTES) {
    throw new Error("current generation pointer exceeds its byte limit");
  }
}
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
