import type { StorageAdapter } from "../../core/adapters";
import type { OutputSettings } from "../../settings/types";
import { throwIfCancelled } from "../../services/cancellation";
import { sha256Hex } from "../../utils/digest";
import {
  buildFullTextGeneration,
  type GenerationIndexBuildDiagnostics,
  type GenerationIndexBuildProgress,
} from "./generation-index-builder";
import {
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  type GenerationIndexDerivation,
} from "./generation-index-format";
import { StorageGenerationObjectSpool } from "./generation-index-spool";
import { FullTextGenerationIndexStoreError, type FullTextGenerationIndexStore } from "./generation-index-store";
import type { FullTextKnowledgeBaseStore } from "./knowledge-base";

export const DEFAULT_FULL_TEXT_GENERATION_INDEX_DERIVATION = {
  builderVersion: 1,
  denseCenteringVersion: 1,
  tokenizerVersion: 1,
  postingsVersion: 1,
} as const satisfies GenerationIndexDerivation;

export interface SynchronizeFullTextGenerationIndexInput {
  readonly sourceStore: FullTextKnowledgeBaseStore;
  readonly generationStore: FullTextGenerationIndexStore;
  readonly storage: StorageAdapter;
  readonly output: OutputSettings;
  readonly scopeFingerprint: string;
  readonly identificationFingerprint: string;
  /** High-entropy identity for this build/promotion attempt. */
  readonly writerToken: string;
  readonly indexDerivation?: GenerationIndexDerivation;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: GenerationIndexBuildProgress) => void;
}

export interface FullTextGenerationSynchronizationResult {
  readonly kind: "reused" | "rebuilt";
  readonly generationId: string;
  readonly sourceRevision: number;
  readonly indexedPaperCount: number;
  readonly chunkCount: number;
  readonly diagnostics?: GenerationIndexBuildDiagnostics;
}

const MAX_POST_COMMIT_SOURCE_ATTEMPTS = 3;

export async function preflightFullTextGenerationSynchronization(input: {
  readonly storage: StorageAdapter;
  readonly generationStore: FullTextGenerationIndexStore;
}): Promise<"available" | "migration-fallback"> {
  const current = await input.generationStore.openCurrent();
  if (input.storage.readBinary && input.storage.writeBinary
    && input.storage.writeTextAtomic && input.storage.createTextExclusive
    && input.storage.list) {
    return "available";
  }
  if (current === null) return "migration-fallback";
  throw new FullTextGenerationIndexStoreError(
    "this host cannot update a full-text generation after cutover", "capability-unsupported",
  );
}

/**
 * Synchronize one immutable generation from an already committed legacy
 * manifest. The manifest remains the rebuild source; CURRENT changes only
 * after the builder and store have validated the complete generation.
 */
export async function synchronizeFullTextGenerationIndex(
  input: SynchronizeFullTextGenerationIndexInput,
): Promise<FullTextGenerationSynchronizationResult> {
  for (let attempt = 0; attempt < MAX_POST_COMMIT_SOURCE_ATTEMPTS; attempt += 1) {
    // A retry is a new promotion/build attempt. Keep attempt zero stable so a
    // caller can replay an interrupted operation, but derive a fresh token for
    // every retry in this synchronization call instead of reusing a
    // staging/lease identity.
    const synchronized = await synchronizeFullTextGenerationIndexOnce({
      ...input,
      writerToken: writerTokenForAttempt(input.writerToken, attempt),
    });
    const latest = await input.sourceStore.loadManifest();
    assertSourceIdentity(input, latest);
    if (latest.revision === synchronized.sourceRevision) {
      throwIfCancelled(input.signal);
      return synchronized;
    }
    if (attempt + 1 === MAX_POST_COMMIT_SOURCE_ATTEMPTS) {
      throw new FullTextGenerationIndexStoreError(
        "full-text source kept changing after generation promotion",
        "stale-source",
        synchronized.sourceRevision,
        latest.revision,
      );
    }
    throwIfCancelled(input.signal);
  }
  throw new Error("unreachable full-text generation synchronization retry state");
}

function writerTokenForAttempt(seed: string, attempt: number): string {
  if (attempt === 0) return seed;
  return `writer-${sha256Hex(`fulltext-generation-retry-v1\0${seed}\0${attempt}`).slice(0, 64)}`;
}

async function synchronizeFullTextGenerationIndexOnce(
  input: SynchronizeFullTextGenerationIndexInput,
): Promise<FullTextGenerationSynchronizationResult> {
  throwIfCancelled(input.signal);
  const manifest = await input.sourceStore.loadManifest();
  throwIfCancelled(input.signal);
  assertSourceIdentity(input, manifest);
  const indexDerivation = snapshotIndexDerivation(
    input.indexDerivation ?? DEFAULT_FULL_TEXT_GENERATION_INDEX_DERIVATION,
  );
  const sourceCurrentRevision = async (): Promise<number> => {
    const latest = await input.sourceStore.loadManifest();
    assertSourceIdentity(input, latest);
    return latest.revision;
  };
  const current = await input.generationStore.openCurrent();
  throwIfCancelled(input.signal);

  if (current
    && current.descriptor.schemaVersion === GENERATION_DESCRIPTOR_SCHEMA_VERSION
    && current.descriptor.sourceRevision === manifest.revision
    && current.descriptor.modelId === manifest.modelId
    && current.descriptor.dimension === manifest.dimension
    && sameIndexDerivation(current.descriptor.indexDerivation, indexDerivation)) {
    const replay = await input.generationStore.stageAndPromote({
      descriptor: current.descriptor,
      objects: [],
      writerToken: input.writerToken,
      expectedCurrent: {
        generationId: current.pointer.generationId,
        sourceRevision: current.pointer.sourceRevision,
      },
      sourceCurrentRevision,
    });
    return resultFromDescriptor("reused", replay.descriptor);
  }

  const generationId = deriveGenerationId(manifest, indexDerivation, input.writerToken);
  const spool = new StorageGenerationObjectSpool(
    input.storage,
    input.output,
    input.scopeFingerprint,
    input.identificationFingerprint,
    { generationId, writerToken: input.writerToken },
  );
  const built = await buildFullTextGeneration({
    manifest,
    loadPaper: (paperKey) => input.sourceStore.loadPaper(paperKey),
    generationId,
    indexDerivation,
    spool,
    signal: input.signal,
    onProgress: input.onProgress,
  });
  const opened = await input.generationStore.stageAndPromote({
    descriptor: built.descriptor,
    objects: built.objects(),
    writerToken: input.writerToken,
    expectedCurrent: current === null ? null : {
      generationId: current.pointer.generationId,
      sourceRevision: current.pointer.sourceRevision,
    },
    sourceCurrentRevision,
  });
  return {
    ...resultFromDescriptor("rebuilt", opened.descriptor),
    diagnostics: { ...built.diagnostics },
  };
}

function deriveGenerationId(
  manifest: Awaited<ReturnType<FullTextKnowledgeBaseStore["loadManifest"]>>,
  indexDerivation: GenerationIndexDerivation,
  writerToken: string,
): string {
  const identity = JSON.stringify({
    sourceRevision: manifest.revision,
    scopeFingerprint: manifest.scopeFingerprint,
    identificationFingerprint: manifest.identificationFingerprint,
    modelId: manifest.modelId,
    dimension: manifest.dimension,
    indexDerivation,
  });
  const attempt = `fulltext-generation-attempt-v1\0${identity}\0${writerToken}`;
  return `gen-${manifest.revision.toString(36)}-${sha256Hex(attempt).slice(0, 44)}`;
}

function snapshotIndexDerivation(value: GenerationIndexDerivation): GenerationIndexDerivation {
  return {
    builderVersion: value.builderVersion,
    denseCenteringVersion: value.denseCenteringVersion,
    tokenizerVersion: value.tokenizerVersion,
    postingsVersion: value.postingsVersion,
  };
}

function sameIndexDerivation(
  left: GenerationIndexDerivation,
  right: GenerationIndexDerivation,
): boolean {
  return left.builderVersion === right.builderVersion
    && left.denseCenteringVersion === right.denseCenteringVersion
    && left.tokenizerVersion === right.tokenizerVersion
    && left.postingsVersion === right.postingsVersion;
}

function assertSourceIdentity(
  input: Pick<SynchronizeFullTextGenerationIndexInput, "scopeFingerprint" | "identificationFingerprint">,
  manifest: Awaited<ReturnType<FullTextKnowledgeBaseStore["loadManifest"]>>,
): void {
  if (manifest.scopeFingerprint !== input.scopeFingerprint
    || manifest.identificationFingerprint !== input.identificationFingerprint) {
    throw new Error("full-text generation source manifest belongs to a different library scope");
  }
}

function resultFromDescriptor(
  kind: FullTextGenerationSynchronizationResult["kind"],
  descriptor: {
    readonly generationId: string;
    readonly sourceRevision: number;
    readonly corpusStats: { readonly indexedPaperCount: number; readonly chunkCount: number };
  },
): FullTextGenerationSynchronizationResult {
  return {
    kind,
    generationId: descriptor.generationId,
    sourceRevision: descriptor.sourceRevision,
    indexedPaperCount: descriptor.corpusStats.indexedPaperCount,
    chunkCount: descriptor.corpusStats.chunkCount,
  };
}
