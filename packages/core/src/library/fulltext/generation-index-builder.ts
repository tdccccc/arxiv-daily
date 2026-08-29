import { createEvidenceChunkId, type EvidenceChunk, type EvidenceDerivation } from "./evidence-chunk";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  decodeFullTextKnowledgeBaseManifest,
  type FullTextKnowledgeBaseManifest,
  type FullTextPaperDocument,
  type FullTextPaperKnowledgeRecord,
} from "./knowledge-base";
import type { GenerationObjectWrite } from "./generation-index-store";
import {
  BINARY_BLOCK_HEADER_BYTES,
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  LEXICAL_BUCKET_COUNT,
  MAX_BINARY_OBJECT_BYTES,
  MAX_GENERATION_OBJECTS,
  blockObjectChecksum,
  decodeLexicalPostingsBlock,
  encodeEvidenceBlock,
  encodeGenerationDescriptor,
  encodeLexicalDictionaryBlock,
  encodeLexicalPostingsBlock,
  encodePaperMetadataBlock,
  encodeVectorBlock,
  lexicalTermBucket,
  type EvidenceBlockRecord,
  type GenerationDescriptor,
  type GenerationIndexDerivation,
  type GenerationObjectReference,
  type LexicalChunkRecord,
  type LexicalDictionaryEntry,
  type LexicalOccurrence,
  type LexicalPostingsBlock,
  type PaperMetadataRecord,
} from "./generation-index-format";
import {
  compareNamespaceTerm,
  deriveLexicalChunk,
  deriveLexicalDictionaryEntries,
  lexicalBucketMask,
  lexicalQueryCatalog,
  lexicalTermBuckets,
  type DerivedLexicalChunk,
} from "./generation-lexical-derivation";

export interface GenerationObjectSpool {
  put(reference: Omit<GenerationObjectReference, "byteLength" | "checksum">, bytes: Uint8Array): Promise<GenerationObjectReference>;
  read(reference: GenerationObjectReference): Promise<Uint8Array>;
  removeAll(): Promise<void>;
}

export type GenerationIndexBuildErrorCode = "invalid-source" | "object-too-large" | "object-limit" | "spool-failed";

export class GenerationIndexBuildError extends Error {
  constructor(message: string, readonly code: GenerationIndexBuildErrorCode, options: ErrorOptions = {}) {
    super(message, options);
    this.name = "GenerationIndexBuildError";
  }
}

export interface GenerationIndexBuildDiagnostics {
  peakLoadedPapers: number;
  peakBufferedObjects: number;
  peakBufferedBytes: number;
  peakPairBytes: number;
  objectRefs: number;
  spoolReads: number;
  dictionarySpoolReads: number;
  replaySpoolReads: number;
  sourcePaperLoads: number;
  /** Complete production-codec calls, excluding constant-size budget probes. */
  encodeAttempts: number;
  /** Lexical chunk derivations; exactly one per indexed chunk. */
  derivedChunks: number;
}

export interface GenerationIndexBuildProgress {
  readonly phase: "papers" | "dictionary" | "complete";
  readonly completed: number;
  readonly total: number;
}

export type GenerationIndexBuildOperation =
  | { readonly operation: "derive-lexical-chunk"; readonly chunkOrdinal: number }
  | { readonly operation: "encode"; readonly codec: "vector" | "evidence" | "paper-metadata" | "lexical-postings" | "lexical-dictionary" | "descriptor" };

/** Optional observation seam; counters and callbacks are emitted by the wrappers that perform the real work. */
export interface GenerationIndexBuildInstrumentation {
  readonly onOperation?: (event: GenerationIndexBuildOperation) => void;
}

export interface BuildFullTextGenerationInput {
  readonly manifest: FullTextKnowledgeBaseManifest;
  readonly loadPaper: (paperKey: string) => FullTextPaperDocument | null | Promise<FullTextPaperDocument | null>;
  readonly generationId: string;
  readonly indexDerivation: GenerationIndexDerivation;
  readonly spool: GenerationObjectSpool;
  readonly titles?: ReadonlyMap<string, string>;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: GenerationIndexBuildProgress) => void;
  /** Deterministic soft row target. Production codecs retain the 4 MiB hard cap. */
  readonly blockTargetRows?: number;
  /** Test-only lower object limit; production always defaults to the format hard cap. */
  readonly maxObjects?: number;
  /** Optional passive test/diagnostic observation of actual wrapped operations. */
  readonly instrumentation?: GenerationIndexBuildInstrumentation;
}

export interface BuiltFullTextGeneration {
  readonly descriptor: GenerationDescriptor;
  /** One-shot stream. Its completion, failure, or early return releases the spool. */
  objects(): AsyncIterable<GenerationObjectWrite>;
  /** Releases an unconsumed build; idempotent. */
  dispose(): Promise<void>;
  readonly diagnostics: GenerationIndexBuildDiagnostics;
}

interface DenseRow {
  readonly paperOrdinal: number;
  readonly paperKey: string;
  readonly vectorRow: number;
  readonly chunk: EvidenceChunk;
  readonly vector: Float32Array;
  readonly evidenceJsonBytes: number;
}
interface PostingRow {
  readonly chunkOrdinal: number;
  readonly chunk: LexicalChunkRecord;
  readonly chunkJsonBytes: number;
  readonly occurrences: readonly LexicalOccurrence[];
  readonly occurrenceJsonBytes: readonly number[];
}

const encoder = new TextEncoder();
const MAX_PAYLOAD_BYTES = MAX_BINARY_OBJECT_BYTES - BINARY_BLOCK_HEADER_BYTES;

export async function buildFullTextGeneration(input: BuildFullTextGenerationInput): Promise<BuiltFullTextGeneration> {
  let disposed = false;
  let disposal: Promise<void> | undefined;
  const dispose = (): Promise<void> => {
    if (disposed) return Promise.resolve();
    if (disposal) return disposal;
    let resolveAttempt!: () => void;
    let rejectAttempt!: (reason: unknown) => void;
    const attempt = new Promise<void>((resolve, reject) => {
      resolveAttempt = resolve;
      rejectAttempt = reject;
    });
    disposal = attempt;
    let removal: Promise<void>;
    try {
      removal = Promise.resolve(input.spool.removeAll());
    } catch (caught) {
      disposal = undefined;
      rejectAttempt(buildError("spool-failed", "failed to clean generation object spool", caught));
      return attempt;
    }
    void removal.then(
      () => {
        disposed = true;
        disposal = undefined;
        resolveAttempt();
      },
      (caught) => {
        disposal = undefined;
        rejectAttempt(buildError("spool-failed", "failed to clean generation object spool", caught));
      },
    );
    return attempt;
  };
  try {
    return await buildGeneration(input, dispose, () => disposed);
  } catch (caught) {
    try { await dispose(); }
    catch (cleanup) {
      if (caught instanceof Error && caught.cause === undefined) Object.defineProperty(caught, "cause", { value: cleanup });
    }
    throw caught;
  }
}

async function buildGeneration(
  input: BuildFullTextGenerationInput,
  dispose: () => Promise<void>,
  isDisposed: () => boolean,
): Promise<BuiltFullTextGeneration> {
  validateInput(input);
  throwIfAborted(input.signal);
  const manifest = snapshotManifest(input.manifest);
  const maxObjects = input.maxObjects ?? MAX_GENERATION_OBJECTS;
  const diagnostics: GenerationIndexBuildDiagnostics = {
    peakLoadedPapers: 0, peakBufferedObjects: 0, peakBufferedBytes: 0, peakPairBytes: 0,
    objectRefs: 0, spoolReads: 0, dictionarySpoolReads: 0, replaySpoolReads: 0,
    sourcePaperLoads: 0, encodeAttempts: 0, derivedChunks: 0,
  };
  const observe = (event: GenerationIndexBuildOperation): void => {
    try { input.instrumentation?.onOperation?.(event); } catch { /* observation must not alter build outcome */ }
  };
  const encode = <T>(
    codec: Extract<GenerationIndexBuildOperation, { operation: "encode" }>["codec"],
    name: string,
    operation: () => T,
    failure: (name: string, caught: unknown) => GenerationIndexBuildError = encodeFailure,
  ): T => {
    diagnostics.encodeAttempts += 1;
    observe({ operation: "encode", codec });
    try { return operation(); }
    catch (caught) { throw failure(name, caught); }
  };
  const derive = (text: string, ordinal: number): DerivedLexicalChunk => {
    diagnostics.derivedChunks += 1;
    observe({ operation: "derive-lexical-chunk", chunkOrdinal: ordinal });
    return deriveSourceLexicalChunk(text, ordinal);
  };
  const refsByKind: Record<GenerationObjectReference["kind"], GenerationObjectReference[]> = {
    vector: [], evidence: [], "paper-metadata": [], "lexical-postings": [], "lexical-dictionary": [],
  };
  const allRefs = () => refsByKind.vector.length + refsByKind.evidence.length + refsByKind["paper-metadata"].length
    + refsByKind["lexical-postings"].length + refsByKind["lexical-dictionary"].length;
  const reserve = (slots: number) => {
    if (allRefs() + slots > maxObjects) throw buildError("object-limit", "generation object reference limit exceeded");
  };
  const putObject = async (
    kind: GenerationObjectReference["kind"], path: string, bytes: Uint8Array, recordStart: number, recordCount: number,
  ): Promise<GenerationObjectReference> => {
    const seed = { kind, path, recordStart, recordCount } as const;
    let reference: GenerationObjectReference;
    try {
      reference = await input.spool.put(seed, bytes);
      throwIfAborted(input.signal);
      if (!reference || typeof reference !== "object" || reference.kind !== kind || reference.path !== path
        || reference.recordStart !== recordStart || reference.recordCount !== recordCount
        || reference.byteLength !== bytes.byteLength || reference.checksum !== blockObjectChecksum(bytes)) {
        throw new Error("spool returned a mismatched reference");
      }
    } catch (caught) {
      if (isAbortError(caught)) throw caught;
      throw buildError("spool-failed", `failed to spool or verify generation object: ${path}`, caught);
    }
    refsByKind[kind].push(reference);
    diagnostics.objectRefs = allRefs();
    return reference;
  };

  const sums = new Float64Array(manifest.dimension);
  let chunkCount = 0;
  let indexedPaperCount = 0;
  let totalLexicalTokenCount = 0;
  let totalExpandedTokenCount = 0;
  let hasLexicalTerms = false;

  let denseRows: DenseRow[] = [];
  let denseEvidenceItemsBytes = 0;
  const flushDense = async () => {
    if (denseRows.length === 0) return;
    reserve(2);
    const rowStart = denseRows[0]!.vectorRow;
    const vectorBytes = encode("vector", "vector block", () => encodeVectorBlock({ rowStart, dimension: manifest.dimension,
      paperOrdinals: new Uint32Array(denseRows.map((row) => row.paperOrdinal)),
      vectors: new Float32Array(denseRows.flatMap((row) => Array.from(row.vector))) }));
    const evidenceBytes = encode("evidence", "evidence block", () => encodeEvidenceBlock({ rowStart, records: denseRows.map(evidenceRecord) }));
    trackPair(diagnostics, vectorBytes, evidenceBytes);
    const suffix = pad(refsByKind.vector.length);
    await putObject("vector", `objects/vector-${suffix}.bin`, vectorBytes, rowStart, denseRows.length);
    await putObject("evidence", `objects/evidence-${suffix}.bin`, evidenceBytes, rowStart, denseRows.length);
    denseRows = []; denseEvidenceItemsBytes = 0;
  };
  const appendDense = async (row: DenseRow) => {
    if (denseRows.length > 0 && input.blockTargetRows !== undefined && denseRows.length >= input.blockTargetRows) await flushDense();
    const nextCount = denseRows.length + 1;
    const vectorBytes = BINARY_BLOCK_HEADER_BYTES + 8 + nextCount * 4 + nextCount * manifest.dimension * 4;
    const evidencePayload = evidencePayloadBytes(row.vectorRow - denseRows.length, denseEvidenceItemsBytes + row.evidenceJsonBytes, nextCount);
    if (vectorBytes > MAX_BINARY_OBJECT_BYTES || evidencePayload > MAX_PAYLOAD_BYTES) {
      if (denseRows.length === 0) throw buildError("object-too-large", `single dense row ${row.vectorRow} exceeds the object cap`);
      await flushDense();
      const singleEvidence = evidencePayloadBytes(row.vectorRow, row.evidenceJsonBytes, 1);
      const singleVector = BINARY_BLOCK_HEADER_BYTES + 12 + manifest.dimension * 4;
      if (singleVector > MAX_BINARY_OBJECT_BYTES || singleEvidence > MAX_PAYLOAD_BYTES) {
        throw buildError("object-too-large", `single dense row ${row.vectorRow} exceeds the object cap`);
      }
    }
    denseRows.push(row); denseEvidenceItemsBytes += row.evidenceJsonBytes;
  };

  let metadataRows: PaperMetadataRecord[] = [];
  let metadataItemsBytes = 0;
  const flushMetadata = async () => {
    if (metadataRows.length === 0) return;
    reserve(1);
    const paperStart = metadataRows[0]!.paperOrdinal;
    const bytes = encode("paper-metadata", "paper metadata block", () => encodePaperMetadataBlock({ paperStart, records: metadataRows }));
    trackOneBuffer(diagnostics, bytes);
    await putObject("paper-metadata", `objects/metadata-${pad(refsByKind["paper-metadata"].length)}.bin`, bytes, paperStart, metadataRows.length);
    metadataRows = []; metadataItemsBytes = 0;
  };
  const appendMetadata = async (row: PaperMetadataRecord) => {
    if (metadataRows.length > 0 && input.blockTargetRows !== undefined && metadataRows.length >= input.blockTargetRows) await flushMetadata();
    const itemBytes = jsonBytes(row);
    const start = metadataRows[0]?.paperOrdinal ?? row.paperOrdinal;
    if (metadataPayloadBytes(start, metadataItemsBytes + itemBytes, metadataRows.length + 1) > MAX_PAYLOAD_BYTES) {
      if (metadataRows.length === 0) throw buildError("object-too-large", `single metadata paper ${row.paperKey} exceeds the object cap`);
      await flushMetadata();
      if (metadataPayloadBytes(row.paperOrdinal, itemBytes, 1) > MAX_PAYLOAD_BYTES) {
        throw buildError("object-too-large", `single metadata paper ${row.paperKey} exceeds the object cap`);
      }
    }
    metadataRows.push(row); metadataItemsBytes += itemBytes;
  };

  let postingRows: PostingRow[] = [];
  let postingChunkItemsBytes = 0;
  let postingOccurrenceItemsBytes = 0;
  let postingOccurrenceCount = 0;
  let postingCatalogItemsBytes = 0;
  let postingDictionaryEntryItemsBytes = 0;
  let postingDictionaryEntryCount = 0;
  let postingDictionaryCatalogItemsBytes = 0;
  const flushPostings = async () => {
    if (postingRows.length === 0) return;
    reserve(1);
    const postingOrdinal = refsByKind["lexical-postings"].length;
    const occurrences = postingRows.flatMap((row) => row.occurrences);
    const block: LexicalPostingsBlock = {
      postingOrdinal, chunkStart: postingRows[0]!.chunkOrdinal,
      chunks: postingRows.map((row) => row.chunk), occurrences,
      termCatalog: occurrences.map((_, index) => index).sort((a, b) => compareNamespaceTerm(occurrences[a]!, occurrences[b]!)
        || occurrences[a]!.chunkOrdinal - occurrences[b]!.chunkOrdinal),
    };
    const bytes = encode("lexical-postings", "lexical postings block", () => encodeLexicalPostingsBlock(block));
    trackOneBuffer(diagnostics, bytes);
    await putObject("lexical-postings", `objects/postings-${pad(postingOrdinal)}.bin`, bytes, block.chunkStart, block.chunks.length);
    postingRows = []; postingChunkItemsBytes = 0; postingOccurrenceItemsBytes = 0; postingOccurrenceCount = 0; postingCatalogItemsBytes = 0;
    postingDictionaryEntryItemsBytes = 0; postingDictionaryEntryCount = 0; postingDictionaryCatalogItemsBytes = 0;
  };
  const appendPosting = async (row: PostingRow) => {
    if (postingRows.length > 0 && input.blockTargetRows !== undefined && postingRows.length >= input.blockTargetRows) await flushPostings();
    const rowOccurrenceBytes = sum(row.occurrenceJsonBytes);
    const conservativeEntries = row.occurrences.map((occurrence) => ({ postingOrdinal: refsByKind["lexical-postings"].length,
      namespace: occurrence.namespace, term: occurrence.term, chunkDf: 1, totalTf: occurrence.tf }));
    const rowDictionaryBytes = conservativeEntries.reduce((total, entry) => total + jsonBytes(entry), 0);
    let nextOccurrences = postingOccurrenceCount + row.occurrences.length;
    const estimate = postingsPayloadBytes(
      refsByKind["lexical-postings"].length,
      postingRows[0]?.chunkOrdinal ?? row.chunkOrdinal,
      postingChunkItemsBytes + row.chunkJsonBytes,
      postingRows.length + 1,
      postingOccurrenceItemsBytes + rowOccurrenceBytes,
      nextOccurrences,
      postingCatalogItemsBytes + integerRangeBytes(postingOccurrenceCount, nextOccurrences),
    );
    let nextDictionaryCount = postingDictionaryEntryCount + conservativeEntries.length;
    const dictionaryEstimate = dictionaryPayloadBytes(0, refsByKind["lexical-postings"].length, 1,
      postingDictionaryEntryItemsBytes + rowDictionaryBytes, nextDictionaryCount,
      postingDictionaryCatalogItemsBytes + integerRangeBytes(postingDictionaryEntryCount, nextDictionaryCount));
    if (nextOccurrences > 65_536 || estimate > MAX_PAYLOAD_BYTES || nextDictionaryCount > 65_536 || dictionaryEstimate > MAX_PAYLOAD_BYTES) {
      if (postingRows.length === 0) throw buildError("object-too-large", `single lexical chunk ${row.chunkOrdinal} exceeds postings cap`);
      await flushPostings();
      nextOccurrences = row.occurrences.length;
      nextDictionaryCount = conservativeEntries.length;
      const single = postingsPayloadBytes(refsByKind["lexical-postings"].length, row.chunkOrdinal,
        row.chunkJsonBytes, 1, rowOccurrenceBytes, row.occurrences.length, integerRangeBytes(0, row.occurrences.length));
      const singleDictionary = dictionaryPayloadBytes(0, refsByKind["lexical-postings"].length, 1,
        rowDictionaryBytes, conservativeEntries.length, integerRangeBytes(0, conservativeEntries.length));
      if (row.occurrences.length > 65_536 || single > MAX_PAYLOAD_BYTES
        || conservativeEntries.length > 65_536 || singleDictionary > MAX_PAYLOAD_BYTES) {
        throw buildError("object-too-large", `single lexical chunk ${row.chunkOrdinal} exceeds postings cap`);
      }
    }
    postingRows.push(row); postingChunkItemsBytes += row.chunkJsonBytes;
    postingOccurrenceItemsBytes += rowOccurrenceBytes;
    postingCatalogItemsBytes += integerRangeBytes(postingOccurrenceCount, nextOccurrences);
    postingOccurrenceCount = nextOccurrences;
    postingDictionaryEntryItemsBytes += rowDictionaryBytes;
    postingDictionaryCatalogItemsBytes += integerRangeBytes(postingDictionaryEntryCount, nextDictionaryCount);
    postingDictionaryEntryCount = nextDictionaryCount;
  };

  const readyKeys = Object.keys(manifest.papers).filter((key) => manifest.papers[key]!.status === "ready").sort(compareCodeUnits);
  for (let sourceIndex = 0; sourceIndex < readyKeys.length; sourceIndex += 1) {
    throwIfAborted(input.signal);
    const paperKey = readyKeys[sourceIndex]!; const record = manifest.papers[paperKey]!;
    diagnostics.sourcePaperLoads += 1;
    let document: FullTextPaperDocument | null;
    try { document = await input.loadPaper(paperKey); }
    catch (caught) { if (isAbortError(caught)) throw caught; throw buildError("invalid-source", `failed to load ready source paper: ${paperKey}`, caught); }
    throwIfAborted(input.signal); diagnostics.peakLoadedPapers = Math.max(diagnostics.peakLoadedPapers, 1);
    validateSourceBinding(paperKey, record, document, manifest);
    const bound = document!;
    if (bound.chunks.length > 0) {
      const paperOrdinal = indexedPaperCount; const paperChunkStart = chunkCount;
      for (let chunkIndex = 0; chunkIndex < bound.chunks.length; chunkIndex += 1) {
        throwIfAborted(input.signal);
        const chunk = completeChunk(bound, chunkIndex);
        const vector = bound.vectors.slice(chunkIndex * manifest.dimension, (chunkIndex + 1) * manifest.dimension);
        for (let column = 0; column < manifest.dimension; column += 1) sums[column]! += vector[column]!;
        const recordForEvidence: EvidenceBlockRecord = { paperIndex: paperOrdinal, paperKey, vectorRow: chunkCount, chunk };
        await appendDense({ paperOrdinal, paperKey, vectorRow: chunkCount, chunk, vector, evidenceJsonBytes: jsonBytes(recordForEvidence) });
        const lexical = derive(chunk.text, chunkCount);
        totalLexicalTokenCount += lexical.baseLength; totalExpandedTokenCount += lexical.expandedLength;
        hasLexicalTerms ||= lexical.occurrences.length > 0;
        const lexicalRecord: LexicalChunkRecord = { paperOrdinal, chunkIndex, baseLength: lexical.baseLength,
          expandedLength: lexical.expandedLength, compactText: lexical.compactText };
        await appendPosting({ chunkOrdinal: chunkCount, chunk: lexicalRecord, chunkJsonBytes: jsonBytes(lexicalRecord),
          occurrences: lexical.occurrences, occurrenceJsonBytes: lexical.occurrences.map(jsonBytes) });
        chunkCount += 1;
      }
      const title = selectTitle(input.titles?.get(paperKey), record.title, bound.title);
      await appendMetadata({ paperOrdinal, paperKey, chunkStart: paperChunkStart, chunkCount: bound.chunks.length, ...(title === undefined ? {} : { title }) });
      indexedPaperCount += 1;
    }
    input.onProgress?.({ phase: "papers", completed: sourceIndex + 1, total: readyKeys.length });
  }
  await flushDense(); await flushMetadata();
  if (hasLexicalTerms) await flushPostings();
  else postingRows = [];

  const routing = Array.from({ length: LEXICAL_BUCKET_COUNT }, () => [] as string[]);
  if (hasLexicalTerms) await buildDictionaries(input, refsByKind, diagnostics, reserve, putObject, routing,
    (operation) => encode("lexical-dictionary", "lexical dictionary block", operation));
  const objects = [...refsByKind.vector, ...refsByKind.evidence, ...refsByKind["paper-metadata"],
    ...refsByKind["lexical-postings"], ...refsByKind["lexical-dictionary"]];
  const descriptor: GenerationDescriptor = {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION, schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId: input.generationId, sourceRevision: manifest.revision,
    scopeFingerprint: manifest.scopeFingerprint, identificationFingerprint: manifest.identificationFingerprint,
    modelId: manifest.modelId, dimension: manifest.dimension,
    corpusMean: Array.from(sums, (value) => chunkCount === 0 ? 0 : value / chunkCount),
    corpusStats: { indexedPaperCount, chunkCount, totalLexicalTokenCount,
      avgdl: chunkCount === 0 ? 0 : totalLexicalTokenCount / chunkCount,
      totalLexicalTokenCountWithHanSingles: totalExpandedTokenCount,
      avgdlWithHanSingles: chunkCount === 0 ? 0 : totalExpandedTokenCount / chunkCount },
    lexicalCapability: chunkCount === 0 ? "none" : "bm25-v1", lexicalRouting: routing,
    indexDerivation: { ...input.indexDerivation }, objects,
  };
  encode("descriptor", "generation descriptor", () => encodeGenerationDescriptor(descriptor), (_name, caught) => {
    if (/objects exceed|reference limit|descriptor exceeds/i.test(errorMessage(caught))) {
      return buildError("object-limit", "generation descriptor or object reference limit exceeded", caught);
    }
    return buildError("invalid-source", "built generation descriptor is invalid", caught);
  });
  diagnostics.objectRefs = objects.length;
  input.onProgress?.({ phase: "complete", completed: objects.length, total: objects.length });

  let streamState: "available" | "consuming" | "consumed" = "available";
  return {
    descriptor, diagnostics, dispose,
    objects() {
      let started = false;
      const generator = (async function* (): AsyncGenerator<GenerationObjectWrite> {
        if (streamState !== "available" || isDisposed()) throw buildError("invalid-source", "generation object stream is one-shot and unavailable");
        streamState = "consuming";
        let failure: unknown;
        try {
          for (const reference of descriptor.objects) {
            throwIfAborted(input.signal);
            const bytes = await readVerifiedSpool(input, reference, diagnostics, false, true);
            throwIfAborted(input.signal);
            yield { path: reference.path, bytes };
          }
        } catch (caught) {
          failure = caught;
        } finally {
          streamState = "consumed";
          try { await dispose(); }
          catch (cleanup) {
            if (failure instanceof Error && failure.cause === undefined) Object.defineProperty(failure, "cause", { value: cleanup });
            else if (failure === undefined) throw cleanup;
          }
        }
        if (failure !== undefined) throw failure;
      })();
      return {
        [Symbol.asyncIterator]() {
          return {
            next(value?: unknown) { started = true; return generator.next(value); },
            throw(error?: unknown) { started = true; return generator.throw(error); },
            async return(value?: unknown) {
              if (started) return generator.return(value);
              if (streamState === "available") streamState = "consumed";
              await dispose();
              return { done: true, value };
            },
          };
        },
      };
    },
  };
}

async function buildDictionaries(
  input: BuildFullTextGenerationInput,
  refs: Record<GenerationObjectReference["kind"], GenerationObjectReference[]>,
  diagnostics: GenerationIndexBuildDiagnostics,
  reserve: (slots: number) => void,
  put: (kind: GenerationObjectReference["kind"], path: string, bytes: Uint8Array, start: number, count: number) => Promise<GenerationObjectReference>,
  routing: string[][],
  encodeDictionary: (operation: () => Uint8Array) => Uint8Array,
): Promise<void> {
  let entries: LexicalDictionaryEntry[] = []; let entryItemsBytes = 0; let queryCatalogItemsBytes = 0;
  let postingStart = 0; let postingCount = 0; let routeRefCount = 0;
  const flush = async () => {
    if (postingCount === 0) return;
    reserve(1);
    const dictionaryOrdinal = refs["lexical-dictionary"].length;
    // One hash per entry, shared by the routing set, the query catalog and the
    // bucket mask; each of those used to derive it independently.
    const entryBuckets = lexicalTermBuckets(entries);
    const buckets = new Set(entryBuckets);
    if (routeRefCount + buckets.size > MAX_GENERATION_OBJECTS) throw buildError("object-limit", "generation lexical routing reference limit exceeded");
    const bytes = encodeDictionary(() => encodeLexicalDictionaryBlock({ dictionaryOrdinal, postingStart, postingCount, entries,
      queryCatalog: lexicalQueryCatalog(entries, entryBuckets), bucketMask: lexicalBucketMask(entries, entryBuckets) }));
    trackOneBuffer(diagnostics, bytes);
    const reference = await put("lexical-dictionary", `objects/dictionary-${pad(dictionaryOrdinal)}.bin`, bytes, postingStart, postingCount);
    for (const bucket of buckets) routing[bucket]!.push(reference.path);
    routeRefCount += buckets.size; entries = []; entryItemsBytes = 0; queryCatalogItemsBytes = 0; postingStart += postingCount; postingCount = 0;
  };
  for (let index = 0; index < refs["lexical-postings"].length; index += 1) {
    throwIfAborted(input.signal);
    const reference = refs["lexical-postings"][index]!; const bytes = await readVerifiedSpool(input, reference, diagnostics, true);
    let block: LexicalPostingsBlock;
    try { block = decodeLexicalPostingsBlock(bytes); }
    catch (caught) { throw buildError("spool-failed", `spooled postings failed decode: ${reference.path}`, caught); }
    const next = deriveLexicalDictionaryEntries(block); const nextBytes = next.map(jsonBytes); const candidateCount = entries.length + next.length;
    let nextCatalogBytes = integerRangeBytes(entries.length, candidateCount);
    const estimate = dictionaryPayloadBytes(refs["lexical-dictionary"].length, postingStart, postingCount + 1,
      entryItemsBytes + sum(nextBytes), candidateCount, queryCatalogItemsBytes + nextCatalogBytes);
    if (candidateCount > 65_536 || estimate > MAX_PAYLOAD_BYTES) {
      if (postingCount === 0) throw buildError("object-too-large", `dictionary for posting ${index} exceeds the object cap`);
      await flush();
      nextCatalogBytes = integerRangeBytes(0, next.length);
      const single = dictionaryPayloadBytes(refs["lexical-dictionary"].length, postingStart, 1, sum(nextBytes), next.length, nextCatalogBytes);
      if (next.length > 65_536 || single > MAX_PAYLOAD_BYTES) throw buildError("object-too-large", `dictionary for posting ${index} exceeds the object cap`);
    }
    entries.push(...next); entryItemsBytes += sum(nextBytes); queryCatalogItemsBytes += nextCatalogBytes; postingCount += 1;
    input.onProgress?.({ phase: "dictionary", completed: index + 1, total: refs["lexical-postings"].length });
  }
  await flush();
}

async function readVerifiedSpool(
  input: BuildFullTextGenerationInput,
  reference: GenerationObjectReference,
  diagnostics: GenerationIndexBuildDiagnostics,
  dictionary: boolean,
  replay = false,
): Promise<Uint8Array> {
  try {
    const bytes = await input.spool.read(reference);
    diagnostics.spoolReads += 1;
    if (dictionary) diagnostics.dictionarySpoolReads += 1;
    if (replay) diagnostics.replaySpoolReads += 1;
    if (!(bytes instanceof Uint8Array) || bytes.byteLength > MAX_BINARY_OBJECT_BYTES
      || bytes.byteLength !== reference.byteLength || blockObjectChecksum(bytes) !== reference.checksum) {
      throw new Error("spooled object failed reference verification");
    }
    return bytes;
  } catch (caught) {
    if (isAbortError(caught)) throw caught;
    if (caught instanceof GenerationIndexBuildError && caught.code === "spool-failed") throw caught;
    throw buildError("spool-failed", `failed to read or verify spooled object: ${reference.path}`, caught);
  }
}

function evidenceRecord(row: DenseRow): EvidenceBlockRecord { return { paperIndex: row.paperOrdinal, paperKey: row.paperKey, vectorRow: row.vectorRow, chunk: row.chunk }; }
function evidencePayloadBytes(rowStart: number, itemBytes: number, count: number): number { return jsonBytes({ rowStart, records: [] }) + itemBytes + commas(count); }
function metadataPayloadBytes(paperStart: number, itemBytes: number, count: number): number { return jsonBytes({ paperStart, records: [] }) + itemBytes + commas(count); }
function postingsPayloadBytes(postingOrdinal: number, chunkStart: number, chunkBytes: number, chunkCount: number, occurrenceBytes: number, occurrenceCount: number, catalogBytes: number): number {
  return jsonBytes({ postingOrdinal, chunkStart, chunks: [], occurrences: [], termCatalog: [] })
    + chunkBytes + commas(chunkCount) + occurrenceBytes + commas(occurrenceCount)
    + catalogBytes + commas(occurrenceCount);
}
function dictionaryPayloadBytes(dictionaryOrdinal: number, postingStart: number, postingCount: number, entryBytes: number, entryCount: number, queryCatalogBytes: number): number {
  return jsonBytes({ dictionaryOrdinal, postingStart, postingCount, entries: [], queryCatalog: [], bucketMask: "0".repeat(64) })
    + entryBytes + commas(entryCount) + queryCatalogBytes + commas(entryCount);
}
function integerRangeBytes(start: number, end: number): number { let total = 0; for (let index = start; index < end; index += 1) total += String(index).length; return total; }
function commas(count: number): number { return Math.max(0, count - 1); }
function jsonBytes(value: unknown): number { return encoder.encode(JSON.stringify(value)).byteLength; }
function sum(values: readonly number[]): number { return values.reduce((total, value) => total + value, 0); }

function completeChunk(document: FullTextPaperDocument, index: number): EvidenceChunk {
  const raw = document.chunks[index]!; const derivation = raw.derivation ?? document.derivation;
  if (!raw.id || !raw.headings || !raw.locator || !derivation) throw buildError("invalid-source", `paper ${document.paperKey} chunk ${index} is incomplete`);
  const identity = { text: raw.text, headings: [...raw.headings], locator: { ...raw.locator }, derivation: cloneDerivation(derivation) };
  if (raw.index !== index || raw.page !== raw.locator.pageStart || createEvidenceChunkId(identity) !== raw.id) throw buildError("invalid-source", `paper ${document.paperKey} chunk ${index} identity is invalid`);
  return { id: raw.id, index: raw.index, page: raw.page, ...identity };
}

function validateSourceBinding(paperKey: string, record: FullTextPaperKnowledgeRecord, document: FullTextPaperDocument | null, manifest: FullTextKnowledgeBaseManifest): asserts document is FullTextPaperDocument {
  if (!document) throw buildError("invalid-source", `ready source paper is missing: ${paperKey}`);
  const equal = document.schemaVersion === FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION
    && record.paperKey === paperKey && document.paperKey === paperKey
    && record.modelId === manifest.modelId && document.modelId === record.modelId
    && record.dimension === manifest.dimension && document.dimension === record.dimension
    && record.textHash === document.textHash && record.contentHash === document.contentHash
    && record.title === document.title && record.titleVersion === document.titleVersion
    && record.updatedAt === document.updatedAt
    && stringArraysEqual(record.filePaths, document.filePaths)
    && stringArraysEqual(record.observationFingerprints, document.observationFingerprints)
    && derivationsEqual(record.derivation, document.derivation)
    && record.chunkCount === document.chunks.length
    && document.vectors instanceof Float32Array
    && document.vectors.length === document.chunks.length * document.dimension;
  if (!equal) throw buildError("invalid-source", `ready source paper does not exactly match manifest: ${paperKey}`);
  for (const value of document.vectors) if (!Number.isFinite(value)) throw buildError("invalid-source", `ready source paper has a non-finite vector: ${paperKey}`);
}

function snapshotManifest(manifest: FullTextKnowledgeBaseManifest): FullTextKnowledgeBaseManifest {
  try { const validated = decodeFullTextKnowledgeBaseManifest(structuredClone(manifest)); if (!validated) throw new Error("manifest schema invalid"); return validated; }
  catch (caught) { throw buildError("invalid-source", "knowledge-base manifest cannot be snapshotted or validated", caught); }
}
function validateInput(input: BuildFullTextGenerationInput): void {
  if (!input || typeof input !== "object" || typeof input.loadPaper !== "function" || !input.spool
    || typeof input.spool.put !== "function" || typeof input.spool.read !== "function" || typeof input.spool.removeAll !== "function") throw buildError("invalid-source", "invalid generation builder input");
  for (const [name, value, maximum] of [["blockTargetRows", input.blockTargetRows, Number.MAX_SAFE_INTEGER], ["maxObjects", input.maxObjects, MAX_GENERATION_OBJECTS]] as const) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 1 || value > maximum)) throw buildError("invalid-source", `${name} must be a positive bounded safe integer`);
  }
}
function selectTitle(candidate: string | undefined, record: string | undefined, document: string | undefined): string | undefined {
  return validTitle(candidate) ? candidate : record ?? document;
}
function validTitle(value: unknown): value is string { return typeof value === "string" && value.trim().length > 0 && value.length <= 16_384; }
function deriveSourceLexicalChunk(text: string, ordinal: number): DerivedLexicalChunk {
  try { return deriveLexicalChunk(text, ordinal); }
  catch (caught) { throw buildError(isCapacityError(caught) ? "object-too-large" : "invalid-source", `source chunk ${ordinal} cannot be lexically derived`, caught); }
}
function trackPair(diagnostics: GenerationIndexBuildDiagnostics, left: Uint8Array, right: Uint8Array): void { diagnostics.peakBufferedObjects = Math.max(diagnostics.peakBufferedObjects, 2); diagnostics.peakBufferedBytes = Math.max(diagnostics.peakBufferedBytes, left.byteLength + right.byteLength); diagnostics.peakPairBytes = Math.max(diagnostics.peakPairBytes, left.byteLength + right.byteLength); }
function trackOneBuffer(diagnostics: GenerationIndexBuildDiagnostics, bytes: Uint8Array): void { diagnostics.peakBufferedObjects = Math.max(diagnostics.peakBufferedObjects, 1); diagnostics.peakBufferedBytes = Math.max(diagnostics.peakBufferedBytes, bytes.byteLength); }
function encodeFailure(name: string, caught: unknown): GenerationIndexBuildError { return buildError(isCapacityError(caught) ? "object-too-large" : "invalid-source", `${name} failed final encoding`, caught); }
function pad(value: number): string { return String(value).padStart(10, "0"); }
function compareCodeUnits(left: string, right: string): number { return left < right ? -1 : left > right ? 1 : 0; }
function stringArraysEqual(left: readonly string[], right: readonly string[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function derivationsEqual(left?: EvidenceDerivation, right?: EvidenceDerivation): boolean { return left === undefined ? right === undefined : right !== undefined && left.parser.id === right.parser.id && left.parser.version === right.parser.version && left.chunkerVersion === right.chunkerVersion && left.embeddingInputVersion === right.embeddingInputVersion; }
function cloneDerivation(value: EvidenceDerivation): EvidenceDerivation { return { parser: { ...value.parser }, chunkerVersion: value.chunkerVersion, embeddingInputVersion: value.embeddingInputVersion }; }
function isCapacityError(error: unknown): boolean { return /byte limit|exceeds? 65536|count exceeds? 65536/i.test(errorMessage(error)); }
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function buildError(code: GenerationIndexBuildErrorCode, message: string, cause?: unknown): GenerationIndexBuildError { return new GenerationIndexBuildError(message, code, cause === undefined ? {} : { cause }); }
function isAbortError(error: unknown): boolean { return !!error && typeof error === "object" && (error as { name?: unknown }).name === "AbortError"; }
function throwIfAborted(signal?: AbortSignal): void { if (signal?.aborted) throw new DOMException(typeof signal.reason === "string" ? signal.reason : "The operation was aborted", "AbortError"); }
