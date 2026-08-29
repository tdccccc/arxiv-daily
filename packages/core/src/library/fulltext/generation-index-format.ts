import type { OutputSettings } from "../../settings/types";
import { sha256Hex } from "../../utils/digest";
import { derivePaperInboxPaths } from "../../services/paper-index";
import { createEvidenceChunkId, type EvidenceChunk, type EvidenceDerivation } from "./evidence-chunk";

export const MAX_BINARY_OBJECT_BYTES = 4 * 1024 * 1024;
export const BINARY_BLOCK_HEADER_BYTES = 52;
export const BINARY_BLOCK_FORMAT_VERSION = 1 as const;
export const BINARY_BLOCK_SCHEMA_VERSION = 4 as const;
export const GENERATION_DESCRIPTOR_FORMAT_VERSION = 1 as const;
export const GENERATION_DESCRIPTOR_SCHEMA_VERSION = 5 as const;
export const MAX_GENERATION_DESCRIPTOR_BYTES = 1024 * 1024;
export const MAX_GENERATION_OBJECTS = 4096;
/**
 * Routing references are not objects. Every dictionary block is one object, yet
 * real prose spreads its terms across all 256 buckets, so a block contributes
 * 256 references. Sharing the object budget therefore capped the index at
 * sixteen dictionary blocks - roughly two thousand chunks - regardless of how
 * few objects it actually used. References are bounded here instead, by what
 * the descriptor can hold once they are stored as ordinals rather than paths.
 */
export const MAX_GENERATION_ROUTE_REFS = 100_000;
export const MAX_GENERATION_PAPERS = 1_000_000;
export const MAX_GENERATION_CHUNKS = 10_000_000;
export const MAX_GENERATION_DIMENSION = 4096;

const MAGIC = new Uint8Array([0x41, 0x44, 0x47, 0x49]); // ADGI
const textEncoder = new TextEncoder();
const CHECKSUM_OFFSET = 20;
const CHECKSUM_BYTES = 32;
const VECTOR_KIND = 1;
const EVIDENCE_KIND = 2;
const PAPER_METADATA_KIND = 3;
const LEXICAL_DICTIONARY_KIND = 4;
const LEXICAL_POSTINGS_KIND = 5;
export const LEXICAL_BUCKET_COUNT = 256;
const MAX_MODEL_ID_LENGTH = 256;
const MAX_PROVENANCE_LENGTH = 128;
const MAX_PAPER_KEY_LENGTH = 256;
const MAX_CHUNK_TEXT_LENGTH = 1024 * 1024;
const MAX_HEADINGS = 64;
const MAX_HEADING_LENGTH = 1024;

export interface VectorBlockInput {
  readonly rowStart: number;
  readonly dimension: number;
  /** Paper ordinal for every vector row. */
  readonly paperOrdinals: Uint32Array;
  readonly vectors: Float32Array;
}

export interface VectorBlock extends VectorBlockInput {
  readonly rowCount: number;
}

export interface EvidenceBlockRecord {
  readonly paperIndex: number;
  readonly paperKey: string;
  readonly vectorRow: number;
  readonly chunk: EvidenceChunk;
}

export interface EvidenceBlock {
  /** Global vector row represented by records[0]. */
  readonly rowStart: number;
  readonly records: readonly EvidenceBlockRecord[];
}

/** Constant-size state sufficient to validate ordered evidence blocks as one generation stream. */
export interface EvidenceStreamClosureState {
  readonly paperIndex: number;
  readonly paperKey: string;
  readonly chunkIndex: number;
  readonly vectorRow: number;
}

export interface PaperMetadataRecord {
  readonly paperOrdinal: number;
  readonly paperKey: string;
  readonly chunkStart: number;
  readonly chunkCount: number;
  readonly title?: string;
}

export interface PaperMetadataBlock {
  readonly paperStart: number;
  readonly records: readonly PaperMetadataRecord[];
}

export type LexicalNamespace = "base" | "expanded" | "alias";

export interface LexicalChunkRecord {
  readonly paperOrdinal: number;
  readonly chunkIndex: number;
  readonly baseLength: number;
  readonly expandedLength: number;
  readonly compactText: string;
}

export interface LexicalOccurrence {
  readonly chunkOrdinal: number;
  readonly namespace: LexicalNamespace;
  readonly term: string;
  readonly tf: number;
}

export interface LexicalPostingsBlock {
  readonly postingOrdinal: number;
  readonly chunkStart: number;
  readonly chunks: readonly LexicalChunkRecord[];
  readonly occurrences: readonly LexicalOccurrence[];
  readonly termCatalog: readonly number[];
}

export interface LexicalDictionaryEntry {
  readonly postingOrdinal: number;
  readonly namespace: LexicalNamespace;
  readonly term: string;
  readonly chunkDf: number;
  readonly totalTf: number;
}

export interface LexicalDictionaryBlock {
  readonly dictionaryOrdinal: number;
  readonly postingStart: number;
  readonly postingCount: number;
  readonly entries: readonly LexicalDictionaryEntry[];
  readonly queryCatalog: readonly number[];
  /** Canonical 256-bit bucket membership, lowercase hexadecimal. */
  readonly bucketMask: string;
}

export type GenerationObjectKind = "vector" | "evidence" | "paper-metadata" | "lexical-postings" | "lexical-dictionary";

export interface GenerationObjectReference {
  readonly kind: GenerationObjectKind;
  readonly path: string;
  readonly byteLength: number;
  readonly recordStart: number;
  readonly recordCount: number;
  readonly checksum: string;
}

export interface GenerationIndexDerivation {
  readonly builderVersion: number;
  readonly denseCenteringVersion: number;
  readonly tokenizerVersion: number;
  readonly postingsVersion: number;
}

export interface GenerationDescriptor {
  readonly formatVersion: typeof GENERATION_DESCRIPTOR_FORMAT_VERSION;
  /** Persisted source schema; v2 is accepted read-only and projected to the current dense shape. */
  readonly schemaVersion: 2 | typeof GENERATION_DESCRIPTOR_SCHEMA_VERSION;
  readonly generationId: string;
  readonly sourceRevision: number;
  readonly scopeFingerprint: string;
  readonly identificationFingerprint: string;
  readonly modelId: string;
  readonly dimension: number;
  /** Per-dimension arithmetic mean over exactly the vector rows covered by vector refs; all zero when empty. */
  readonly corpusMean: readonly number[];
  readonly corpusStats: {
    /**
     * Number of distinct non-empty evidence paper keys. Evidence order assigns
     * paperIndex 0..n-1 while paperKey strictly increases by UTF-16 code units,
     * allowing P4b.2 to prove global uniqueness with constant-size state.
     */
    readonly indexedPaperCount: number;
    readonly chunkCount: number;
    readonly totalLexicalTokenCount: number;
    readonly avgdl: number;
    readonly totalLexicalTokenCountWithHanSingles: number;
    readonly avgdlWithHanSingles: number;
  };
  readonly lexicalCapability: "none" | "bm25-v1";
  /** Per bucket, the ordinals of the lexical-dictionary objects that carry it. */
  readonly lexicalRouting: readonly (readonly number[])[];
  readonly indexDerivation: GenerationIndexDerivation;
  readonly objects: readonly GenerationObjectReference[];
}

export interface FullTextGenerationIndexPaths {
  readonly directory: string;
  readonly generationsDirectory: string;
}

export interface FullTextGenerationPaths {
  readonly directory: string;
  readonly descriptorPath: string;
  readonly objectsDirectory: string;
}

export function encodeVectorBlock(input: VectorBlockInput): Uint8Array {
  requireIntegerInRange(input.rowStart, "vector rowStart", 0, MAX_GENERATION_CHUNKS);
  requireIntegerInRange(input.dimension, "vector dimension", 1, MAX_GENERATION_DIMENSION);
  if (!(input.vectors instanceof Float32Array) || input.vectors.length % input.dimension !== 0) {
    throw new Error("vector block vectors must contain complete rows");
  }
  const rowCount = input.vectors.length / input.dimension;
  requireIntegerInRange(rowCount, "vector record count", 1, MAX_GENERATION_CHUNKS);
  if (!(input.paperOrdinals instanceof Uint32Array) || input.paperOrdinals.length !== rowCount) {
    throw new Error("vector block paperOrdinals must contain exactly one ordinal per row");
  }
  validatePaperOrdinals(input.paperOrdinals);
  if (input.rowStart + rowCount > MAX_GENERATION_CHUNKS) throw new Error("vector row range exceeds the chunk limit");
  const payloadLength = 8 + rowCount * 4 + input.vectors.length * 4;
  requireObjectLength(BINARY_BLOCK_HEADER_BYTES + payloadLength);
  const payload = new Uint8Array(payloadLength);
  const view = new DataView(payload.buffer);
  view.setUint32(0, input.dimension, true);
  view.setUint32(4, input.rowStart, true);
  for (let index = 0; index < rowCount; index += 1) view.setUint32(8 + index * 4, input.paperOrdinals[index]!, true);
  const vectorOffset = 8 + rowCount * 4;
  for (let index = 0; index < input.vectors.length; index += 1) {
    const value = input.vectors[index]!;
    if (!Number.isFinite(value)) throw new Error("vector block values must be finite float32 numbers");
    view.setFloat32(vectorOffset + index * 4, value, true);
  }
  return encodeBlock(VECTOR_KIND, rowCount, payload);
}

export function decodeVectorBlock(bytes: Uint8Array, schemaVersion: 2 | 4 = BINARY_BLOCK_SCHEMA_VERSION): VectorBlock {
  const block = decodeBlock(bytes, VECTOR_KIND, schemaVersion);
  if (block.payload.byteLength < 8) throw new Error("vector block payload is truncated");
  const view = new DataView(block.payload.buffer, block.payload.byteOffset, block.payload.byteLength);
  const dimension = view.getUint32(0, true);
  const rowStart = view.getUint32(4, true);
  requireIntegerInRange(dimension, "vector dimension", 1, MAX_GENERATION_DIMENSION);
  requireIntegerInRange(block.recordCount, "vector record count", 1, MAX_GENERATION_CHUNKS);
  if (rowStart > MAX_GENERATION_CHUNKS || rowStart + block.recordCount > MAX_GENERATION_CHUNKS) {
    throw new Error("vector row range exceeds the chunk limit");
  }
  const valueCount = block.recordCount * dimension;
  const expectedPayloadLength = 8 + block.recordCount * 4 + valueCount * 4;
  if (!Number.isSafeInteger(valueCount) || expectedPayloadLength !== block.payload.byteLength) {
    throw new Error("vector block record count, dimension, ordinals, and payload length disagree");
  }
  const paperOrdinals = new Uint32Array(block.recordCount);
  for (let index = 0; index < block.recordCount; index += 1) paperOrdinals[index] = view.getUint32(8 + index * 4, true);
  validatePaperOrdinals(paperOrdinals);
  const vectors = new Float32Array(valueCount);
  const vectorOffset = 8 + block.recordCount * 4;
  for (let index = 0; index < valueCount; index += 1) {
    const value = view.getFloat32(vectorOffset + index * 4, true);
    if (!Number.isFinite(value)) throw new Error("vector block contains a non-finite value");
    vectors[index] = value;
  }
  return { rowStart, rowCount: block.recordCount, dimension, paperOrdinals, vectors };
}

export function encodeEvidenceBlock(input: EvidenceBlock): Uint8Array {
  requireIntegerInRange(input.rowStart, "evidence rowStart", 0, MAX_GENERATION_CHUNKS - 1);
  if (!Array.isArray(input.records)) throw new Error("evidence block records must be an array");
  requireIntegerInRange(input.records.length, "evidence record count", 1, MAX_GENERATION_CHUNKS);
  if (input.rowStart + input.records.length > MAX_GENERATION_CHUNKS) {
    throw new Error("evidence row range exceeds the chunk limit");
  }
  const records = input.records.map((record) => validateEvidenceRecord(record));
  validateEvidenceBlockOrder(input.rowStart, records);
  const payload = new TextEncoder().encode(JSON.stringify({ rowStart: input.rowStart, records }));
  requireObjectLength(BINARY_BLOCK_HEADER_BYTES + payload.byteLength);
  return encodeBlock(EVIDENCE_KIND, records.length, payload);
}

export function encodePaperMetadataBlock(input: PaperMetadataBlock): Uint8Array {
  requireIntegerInRange(input.paperStart, "metadata paperStart", 0, MAX_GENERATION_PAPERS - 1);
  if (!Array.isArray(input.records) || input.records.length === 0) throw new Error("metadata records must be non-empty");
  const records = input.records.map((record, offset) => validatePaperMetadataRecord(record, input.paperStart + offset));
  const payload = new TextEncoder().encode(JSON.stringify({ paperStart: input.paperStart, records }));
  requireObjectLength(BINARY_BLOCK_HEADER_BYTES + payload.byteLength);
  return encodeBlock(PAPER_METADATA_KIND, records.length, payload);
}

export function decodePaperMetadataBlock(bytes: Uint8Array): PaperMetadataBlock {
  const block = decodeBlock(bytes, PAPER_METADATA_KIND);
  const value = decodeStrictJson(block.payload, "paper metadata block");
  requireExactObject(value, ["paperStart", "records"], "paper metadata block");
  requireIntegerInRange(value.paperStart, "metadata paperStart", 0, MAX_GENERATION_PAPERS - 1);
  if (!Array.isArray(value.records) || value.records.length !== block.recordCount || value.records.length === 0) {
    throw new Error("metadata record count does not match payload");
  }
  const records = value.records.map((record, offset) => validatePaperMetadataRecord(record, value.paperStart + offset));
  return { paperStart: value.paperStart, records };
}

export function encodeLexicalPostingsBlock(input: LexicalPostingsBlock): Uint8Array {
  const validated = validateLexicalPostings(input);
  const payload = new TextEncoder().encode(JSON.stringify(validated));
  requireObjectLength(BINARY_BLOCK_HEADER_BYTES + payload.byteLength);
  return encodeBlock(LEXICAL_POSTINGS_KIND, validated.occurrences.length, payload);
}

export function decodeLexicalPostingsBlock(bytes: Uint8Array): LexicalPostingsBlock {
  const block = decodeBlock(bytes, LEXICAL_POSTINGS_KIND);
  const value = decodeStrictJson(block.payload, "lexical postings block");
  const validated = validateLexicalPostings(value);
  if (block.recordCount !== validated.occurrences.length) throw new Error("postings occurrence count does not match header");
  return validated;
}

export function encodeLexicalDictionaryBlock(input: LexicalDictionaryBlock): Uint8Array {
  const validated = validateLexicalDictionary(input);
  const payload = new TextEncoder().encode(JSON.stringify(validated));
  requireObjectLength(BINARY_BLOCK_HEADER_BYTES + payload.byteLength);
  return encodeBlock(LEXICAL_DICTIONARY_KIND, validated.entries.length, payload);
}

export function decodeLexicalDictionaryBlock(bytes: Uint8Array): LexicalDictionaryBlock {
  const block = decodeBlock(bytes, LEXICAL_DICTIONARY_KIND);
  const value = decodeStrictJson(block.payload, "lexical dictionary block");
  const validated = validateLexicalDictionary(value);
  if (block.recordCount !== validated.entries.length) throw new Error("dictionary entry count does not match header");
  return validated;
}

/**
 * Deriving a bucket costs a canonicalization pass plus a SHA-256, and the same
 * terms recur across posting blocks and again through validation, which
 * independently re-derives them. The function is pure, so a bounded cache keyed
 * on its exact inputs returns the identical value for a fraction of the cost.
 */
const BUCKET_CACHE_LIMIT = 200_000;
const bucketCache = new Map<string, number>();

export function lexicalTermBucket(namespace: LexicalNamespace, term: string): number {
  const cacheKey = typeof term === "string" ? `${String(namespace)}\u0000${term}` : null;
  if (cacheKey !== null) {
    const cached = bucketCache.get(cacheKey);
    if (cached !== undefined) return cached;
  }
  const bucket = computeLexicalTermBucket(namespace, term);
  if (cacheKey !== null) {
    if (bucketCache.size >= BUCKET_CACHE_LIMIT) bucketCache.clear();
    bucketCache.set(cacheKey, bucket);
  }
  return bucket;
}

function computeLexicalTermBucket(namespace: LexicalNamespace, term: string): number {
  const ns = validateNamespace(namespace);
  const termBytes = strictTermBytes(term);
  const nsBytes = textEncoder.encode(ns);
  const input = new Uint8Array(nsBytes.length + 1 + termBytes.length);
  input.set(nsBytes); input.set(termBytes, nsBytes.length + 1);
  return Number.parseInt(sha256Hex(input).slice(0, 2), 16);
}

export function decodeEvidenceBlock(bytes: Uint8Array, schemaVersion: 2 | 4 = BINARY_BLOCK_SCHEMA_VERSION): EvidenceBlock {
  const block = decodeBlock(bytes, EVIDENCE_KIND, schemaVersion);
  requireIntegerInRange(block.recordCount, "evidence record count", 1, MAX_GENERATION_CHUNKS);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(block.payload);
  } catch {
    throw new Error("evidence block payload is not valid UTF-8");
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("evidence block payload is not valid JSON");
  }
  requireExactObject(value, ["rowStart", "records"], "evidence block");
  requireIntegerInRange(value.rowStart, "evidence rowStart", 0, MAX_GENERATION_CHUNKS - 1);
  if (!Array.isArray(value.records) || value.records.length !== block.recordCount) {
    throw new Error("evidence block record count does not match its payload");
  }
  if (value.rowStart + block.recordCount > MAX_GENERATION_CHUNKS) throw new Error("evidence row range exceeds the chunk limit");
  const records: EvidenceBlockRecord[] = [];
  for (const record of value.records) records.push(validateEvidenceRecord(record));
  validateEvidenceBlockOrder(value.rowStart, records);
  return { rowStart: value.rowStart, records };
}

/**
 * Validate one decoded evidence block against the preceding block using O(1)
 * state. Blocks must be supplied in descriptor recordStart order.
 */
export function validateEvidenceStreamClosure(
  previous: EvidenceStreamClosureState | null,
  records: readonly EvidenceBlockRecord[],
): EvidenceStreamClosureState {
  if (!Array.isArray(records) || records.length === 0) throw new Error("evidence closure requires a non-empty block");
  let state = previous;
  for (const record of records) {
    if (state === null) {
      if (record.vectorRow !== 0) throw new Error("first evidence vector row must be zero");
      if (record.paperIndex !== 0) throw new Error("first paper index must be zero");
      if (record.chunk.index !== 0) throw new Error("first paper chunk index must be zero");
    } else {
      if (record.vectorRow !== state.vectorRow + 1) throw new Error("evidence vector rows must be continuous across blocks");
      if (record.paperIndex === state.paperIndex) {
        if (record.paperKey !== state.paperKey) throw new Error("same paper index must keep the same paper key");
        if (record.chunk.index !== state.chunkIndex + 1) throw new Error("same paper chunk index must increase by one");
      } else {
        if (record.paperIndex !== state.paperIndex + 1) throw new Error("new paper index must increase by one");
        if (compareCodeUnitStrings(state.paperKey, record.paperKey) >= 0) {
          throw new Error("paper key order must strictly increase when paper index changes");
        }
        if (record.chunk.index !== 0) throw new Error("new paper chunk index must start at zero");
      }
    }
    state = closureState(record);
  }
  return state!;
}

/** Verify descriptor indexedPaperCount after the final evidence block. */
export function finishEvidenceStreamClosure(
  state: EvidenceStreamClosureState | null,
  indexedPaperCount: number,
): void {
  requireIntegerInRange(indexedPaperCount, "indexedPaperCount", 0, MAX_GENERATION_PAPERS);
  const actual = state === null ? 0 : state.paperIndex + 1;
  if (actual !== indexedPaperCount) throw new Error("evidence stream does not match indexedPaperCount");
}

/** Object-reference checksum covers the complete immutable object, including its embedded checksum. */
export function blockObjectChecksum(bytes: Uint8Array): string {
  requireObjectLength(bytes.byteLength);
  return `sha256:${sha256Hex(bytes)}`;
}

export function encodeGenerationDescriptor(descriptor: GenerationDescriptor): string {
  if (descriptor.schemaVersion !== GENERATION_DESCRIPTOR_SCHEMA_VERSION) {
    throw new Error("only generation descriptor schema v4 can be encoded");
  }
  const validated = validateDescriptor(descriptor);
  const encoded = JSON.stringify(validated);
  if (new TextEncoder().encode(encoded).byteLength > MAX_GENERATION_DESCRIPTOR_BYTES) {
    throw new Error("generation descriptor exceeds its byte limit");
  }
  return encoded;
}

export function decodeGenerationDescriptor(text: string): GenerationDescriptor {
  if (typeof text !== "string") throw new Error("generation descriptor must be JSON text");
  // UTF-8 is never shorter than the JavaScript code-unit count. This guard
  // rejects clearly oversized input before allocating an encoded copy.
  if (text.length > MAX_GENERATION_DESCRIPTOR_BYTES
    || new TextEncoder().encode(text).byteLength > MAX_GENERATION_DESCRIPTOR_BYTES) {
    throw new Error("generation descriptor exceeds its byte limit");
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("generation descriptor is not valid JSON");
  }
  if (!isPlainObject(value)) throw new Error("generation descriptor must be an object");
  if (value.schemaVersion === 2) return validateDescriptorV2(value);
  if (value.schemaVersion !== GENERATION_DESCRIPTOR_SCHEMA_VERSION) {
    throw new Error("unsupported generation descriptor schema version");
  }
  return validateDescriptor(value);
}

export function deriveFullTextGenerationIndexPaths(
  storage: { normalizePath(path: string): string },
  output: OutputSettings,
  scopeFingerprint: string,
  identificationFingerprint: string,
): FullTextGenerationIndexPaths {
  const scopeHex = fingerprintHex("scopeFingerprint", scopeFingerprint);
  const identificationHex = fingerprintHex("identificationFingerprint", identificationFingerprint);
  const { indexDir } = derivePaperInboxPaths(output, (path) => storage.normalizePath(path));
  const directory = storage.normalizePath(
    `${indexDir}/personal-library-search-index/${scopeHex}/${identificationHex}`,
  );
  return {
    directory,
    generationsDirectory: storage.normalizePath(`${directory}/generations`),
  };
}

export function deriveFullTextGenerationPaths(
  storage: { normalizePath(path: string): string },
  output: OutputSettings,
  scopeFingerprint: string,
  identificationFingerprint: string,
  generationId: string,
): FullTextGenerationPaths {
  requireGenerationId(generationId);
  const root = deriveFullTextGenerationIndexPaths(storage, output, scopeFingerprint, identificationFingerprint);
  const directory = storage.normalizePath(`${root.generationsDirectory}/${generationId}`);
  return {
    directory,
    descriptorPath: storage.normalizePath(`${directory}/descriptor.json`),
    objectsDirectory: storage.normalizePath(`${directory}/objects`),
  };
}

function encodeBlock(kind: number, recordCount: number, payload: Uint8Array): Uint8Array {
  const bytes = new Uint8Array(BINARY_BLOCK_HEADER_BYTES + payload.byteLength);
  bytes.set(MAGIC, 0);
  const view = new DataView(bytes.buffer);
  view.setUint16(4, BINARY_BLOCK_FORMAT_VERSION, true);
  view.setUint16(6, BINARY_BLOCK_SCHEMA_VERSION, true);
  view.setUint16(8, kind, true);
  view.setUint16(10, 0, true);
  view.setUint32(12, payload.byteLength, true);
  view.setUint32(16, recordCount, true);
  bytes.set(payload, BINARY_BLOCK_HEADER_BYTES);
  bytes.set(hexToBytes(sha256Hex(checksumInput(bytes))), CHECKSUM_OFFSET);
  return bytes;
}

function decodeBlock(bytes: Uint8Array, expectedKind: number, expectedSchemaVersion: 2 | 4 = BINARY_BLOCK_SCHEMA_VERSION): { payload: Uint8Array; recordCount: number } {
  if (!(bytes instanceof Uint8Array)) throw new Error("binary block must be a Uint8Array");
  requireObjectLength(bytes.byteLength);
  if (bytes.byteLength < BINARY_BLOCK_HEADER_BYTES) throw new Error("binary block header is truncated");
  for (let index = 0; index < MAGIC.length; index += 1) {
    if (bytes[index] !== MAGIC[index]) throw new Error("binary block magic does not match");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint16(4, true) !== BINARY_BLOCK_FORMAT_VERSION) throw new Error("unsupported binary block format version");
  if (view.getUint16(6, true) !== expectedSchemaVersion) throw new Error("unsupported binary block schema version");
  if (view.getUint16(8, true) !== expectedKind) throw new Error("binary block kind does not match decoder");
  if (view.getUint16(10, true) !== 0) throw new Error("binary block reserved header field must be zero");
  const payloadLength = view.getUint32(12, true);
  if (payloadLength > MAX_BINARY_OBJECT_BYTES - BINARY_BLOCK_HEADER_BYTES) {
    throw new Error("binary block payload length exceeds the byte limit");
  }
  const expectedLength = BINARY_BLOCK_HEADER_BYTES + payloadLength;
  if (bytes.byteLength < expectedLength) throw new Error("binary block payload is truncated");
  if (bytes.byteLength > expectedLength) throw new Error("binary block has trailing bytes");
  const actual = sha256Hex(checksumInput(bytes));
  const expected = bytesToHex(bytes.subarray(CHECKSUM_OFFSET, CHECKSUM_OFFSET + CHECKSUM_BYTES));
  if (actual !== expected) throw new Error("binary block checksum mismatch");
  return { payload: bytes.subarray(BINARY_BLOCK_HEADER_BYTES), recordCount: view.getUint32(16, true) };
}

/** Embedded checksum covers header bytes 0..19 followed by payload; checksum bytes 20..51 are excluded. */
function checksumInput(bytes: Uint8Array): Uint8Array {
  const input = new Uint8Array(bytes.byteLength - CHECKSUM_BYTES);
  input.set(bytes.subarray(0, CHECKSUM_OFFSET));
  input.set(bytes.subarray(BINARY_BLOCK_HEADER_BYTES), CHECKSUM_OFFSET);
  return input;
}

function validatePaperOrdinals(ordinals: Uint32Array): void {
  for (let index = 0; index < ordinals.length; index += 1) {
    const current = ordinals[index]!;
    if (current >= MAX_GENERATION_PAPERS) throw new Error("vector paper ordinal exceeds the paper limit");
    if (index === 0) continue;
    const previous = ordinals[index - 1]!;
    if (current !== previous && current !== previous + 1) {
      throw new Error("vector paper ordinals must be non-decreasing and change only by one");
    }
  }
}

function validateEvidenceBlockOrder(rowStart: number, records: readonly EvidenceBlockRecord[]): void {
  let previous: EvidenceBlockRecord | undefined;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index]!;
    if (record.vectorRow !== rowStart + index) throw new Error("evidence vector row must equal rowStart plus record offset");
    if (previous) {
      if (record.paperIndex < previous.paperIndex) throw new Error("evidence paper index must be non-decreasing");
      if (record.paperIndex === previous.paperIndex) {
        if (record.paperKey !== previous.paperKey) throw new Error("same paper index must keep the same paper key");
        if (record.chunk.index !== previous.chunk.index + 1) {
          throw new Error("same paper chunk index must increase by one within a block");
        }
      } else if (compareCodeUnitStrings(previous.paperKey, record.paperKey) >= 0) {
        throw new Error("paper key order must strictly increase when paper index changes");
      }
    }
    previous = record;
  }
}

/** Deterministic UTF-16 code-unit order; deliberately independent of host locale. */
function compareCodeUnitStrings(left: string, right: string): number {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function closureState(record: EvidenceBlockRecord): EvidenceStreamClosureState {
  return {
    paperIndex: record.paperIndex,
    paperKey: record.paperKey,
    chunkIndex: record.chunk.index,
    vectorRow: record.vectorRow,
  };
}

function decodeStrictUtf8(bytes: Uint8Array, name: string): string {
  try { return new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch { throw new Error(`${name} is not valid UTF-8`); }
}

function decodeStrictJson(bytes: Uint8Array, name: string): Record<string, any> {
  let value: unknown;
  try { value = JSON.parse(decodeStrictUtf8(bytes, name)); }
  catch (caught) { if (caught instanceof Error && /UTF-8/.test(caught.message)) throw caught; throw new Error(`${name} is not valid JSON`); }
  if (!isPlainObject(value)) throw new Error(`${name} must be an object`);
  return value;
}

function strictTermBytes(term: unknown): Uint8Array {
  if (typeof term !== "string" || term.length === 0 || term.normalize("NFKC").toLocaleLowerCase("und") !== term) {
    throw new Error("lexical term must be non-empty canonical NFKC lowercase text");
  }
  const bytes = textEncoder.encode(term);
  if (bytes.length === 0 || bytes.length > 65_536 || decodeStrictUtf8(bytes, "lexical term") !== term) throw new Error("lexical term bytes are invalid or too long");
  return bytes;
}

function compactNormalizedText(text: string): string {
  return text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, "");
}

function compareBytes(left: Uint8Array, right: Uint8Array): number {
  const length = Math.min(left.length, right.length);
  for (let index = 0; index < length; index += 1) if (left[index] !== right[index]) return left[index]! - right[index]!;
  return left.length - right.length;
}

const NAMESPACE_ORDER: Record<LexicalNamespace, number> = { alias: 0, base: 1, expanded: 2 };
function validateNamespace(value: unknown): LexicalNamespace {
  if (value !== "base" && value !== "expanded" && value !== "alias") throw new Error("lexical namespace is invalid");
  return value;
}
function compareNamespaceTerm(a: { namespace: LexicalNamespace; term: string }, b: { namespace: LexicalNamespace; term: string }): number {
  return NAMESPACE_ORDER[a.namespace] - NAMESPACE_ORDER[b.namespace] || compareBytes(strictTermBytes(a.term), strictTermBytes(b.term));
}
function compareOccurrenceAuthority(a: LexicalOccurrence, b: LexicalOccurrence): number {
  return a.chunkOrdinal - b.chunkOrdinal || compareNamespaceTerm(a, b);
}
function compareOccurrenceCatalog(a: LexicalOccurrence, b: LexicalOccurrence): number {
  return compareNamespaceTerm(a, b) || a.chunkOrdinal - b.chunkOrdinal;
}

function validateObjectLogicalPath(value: unknown, name: string): string {
  if (typeof value !== "string" || !/^objects\/[a-z0-9][a-z0-9._-]{0,127}$/.test(value)) throw new Error(`${name} is invalid`);
  return value;
}

function validateLexicalPostings(value: unknown): LexicalPostingsBlock {
  requireExactObject(value, ["postingOrdinal", "chunkStart", "chunks", "occurrences", "termCatalog"], "lexical postings block");
  requireIntegerInRange(value.postingOrdinal, "postingOrdinal", 0, MAX_GENERATION_OBJECTS - 1);
  requireIntegerInRange(value.chunkStart, "postings chunkStart", 0, MAX_GENERATION_CHUNKS - 1);
  if (!Array.isArray(value.chunks) || value.chunks.length === 0 || value.chunkStart + value.chunks.length > MAX_GENERATION_CHUNKS) throw new Error("postings chunks must be a bounded non-empty array");
  const chunks: LexicalChunkRecord[] = []; let prior: LexicalChunkRecord | undefined;
  for (const raw of value.chunks) {
    requireExactObject(raw, ["paperOrdinal", "chunkIndex", "baseLength", "expandedLength", "compactText"], "lexical chunk");
    requireIntegerInRange(raw.paperOrdinal, "lexical chunk paperOrdinal", 0, MAX_GENERATION_PAPERS - 1);
    requireIntegerInRange(raw.chunkIndex, "lexical chunk chunkIndex", 0, MAX_GENERATION_CHUNKS - 1);
    requireIntegerInRange(raw.baseLength, "lexical chunk baseLength", 0, 0xffff_ffff);
    requireIntegerInRange(raw.expandedLength, "lexical chunk expandedLength", 0, 0xffff_ffff);
    if (raw.expandedLength < raw.baseLength) throw new Error("expandedLength must not be less than baseLength");
    if (typeof raw.compactText !== "string" || compactNormalizedText(raw.compactText) !== raw.compactText) throw new Error("compactText must be canonical compact NFKC lowercase text");
    if (prior && (raw.paperOrdinal === prior.paperOrdinal ? raw.chunkIndex !== prior.chunkIndex + 1 : raw.paperOrdinal !== prior.paperOrdinal + 1 || raw.chunkIndex !== 0)) throw new Error("lexical chunk identity must be continuous");
    const chunk = { paperOrdinal: raw.paperOrdinal, chunkIndex: raw.chunkIndex, baseLength: raw.baseLength, expandedLength: raw.expandedLength, compactText: raw.compactText }; chunks.push(chunk); prior = chunk;
  }
  if (!Array.isArray(value.occurrences) || value.occurrences.length > 65_536) throw new Error("postings occurrence count exceeds 65536");
  const occurrences: LexicalOccurrence[] = []; let previous: LexicalOccurrence | undefined;
  for (const raw of value.occurrences) {
    requireExactObject(raw, ["chunkOrdinal", "namespace", "term", "tf"], "lexical occurrence");
    requireIntegerInRange(raw.chunkOrdinal, "occurrence chunkOrdinal", value.chunkStart, value.chunkStart + chunks.length - 1);
    const namespace = validateNamespace(raw.namespace); strictTermBytes(raw.term); requireIntegerInRange(raw.tf, "occurrence tf", 1, 0xffff_ffff);
    if (namespace === "alias" && raw.tf !== 1) throw new Error("alias occurrence tf must equal one");
    const occurrence = { chunkOrdinal: raw.chunkOrdinal, namespace, term: raw.term, tf: raw.tf };
    if (previous && compareOccurrenceAuthority(previous, occurrence) >= 0) throw new Error("occurrence authority must strictly increase without duplicates");
    occurrences.push(occurrence); previous = occurrence;
  }
  const termCatalog = validateExactPermutation(value.termCatalog, occurrences.length, "termCatalog");
  let catalogPrevious: LexicalOccurrence | undefined;
  for (const index of termCatalog) { const occurrence = occurrences[index]!; if (catalogPrevious && compareOccurrenceCatalog(catalogPrevious, occurrence) >= 0) throw new Error("termCatalog order must strictly increase"); catalogPrevious = occurrence; }
  return { postingOrdinal: value.postingOrdinal, chunkStart: value.chunkStart, chunks, occurrences, termCatalog };
}

function validateLexicalDictionary(value: unknown): LexicalDictionaryBlock {
  requireExactObject(value, ["dictionaryOrdinal", "postingStart", "postingCount", "entries", "queryCatalog", "bucketMask"], "lexical dictionary block");
  requireIntegerInRange(value.dictionaryOrdinal, "dictionaryOrdinal", 0, MAX_GENERATION_OBJECTS - 1);
  requireIntegerInRange(value.postingStart, "dictionary postingStart", 0, MAX_GENERATION_OBJECTS - 1);
  requireIntegerInRange(value.postingCount, "dictionary postingCount", 1, MAX_GENERATION_OBJECTS);
  if (value.postingStart + value.postingCount > MAX_GENERATION_OBJECTS) throw new Error("dictionary posting range exceeds object limit");
  if (!Array.isArray(value.entries)) throw new Error("dictionary entries must be an array");
  if (value.entries.length > 65_536) throw new Error("dictionary entry count exceeds 65536");
  if (!Array.isArray(value.queryCatalog) || value.queryCatalog.length > 65_536) {
    throw new Error("dictionary queryCatalog count exceeds 65536");
  }
  const entries: LexicalDictionaryEntry[] = []; let previous: LexicalDictionaryEntry | undefined;
  for (const raw of value.entries) {
    requireExactObject(raw, ["postingOrdinal", "namespace", "term", "chunkDf", "totalTf"], "dictionary entry");
    requireIntegerInRange(raw.postingOrdinal, "dictionary postingOrdinal", value.postingStart, value.postingStart + value.postingCount - 1);
    const namespace = validateNamespace(raw.namespace); strictTermBytes(raw.term);
    requireIntegerInRange(raw.chunkDf, "dictionary chunkDf", 1, MAX_GENERATION_CHUNKS); requireIntegerInRange(raw.totalTf, "dictionary totalTf", 1, Number.MAX_SAFE_INTEGER);
    if (raw.totalTf < raw.chunkDf) throw new Error("dictionary totalTf must be at least chunkDf");
    const entry = { postingOrdinal: raw.postingOrdinal, namespace, term: raw.term, chunkDf: raw.chunkDf, totalTf: raw.totalTf };
    const order = previous ? previous.postingOrdinal - entry.postingOrdinal || compareNamespaceTerm(previous, entry) : -1;
    if (previous && order >= 0) throw new Error("dictionary authority must strictly increase without duplicates"); entries.push(entry); previous = entry;
  }
  const queryCatalog = validateExactPermutation(value.queryCatalog, entries.length, "queryCatalog");
  let catalogPrevious: LexicalDictionaryEntry | undefined; let previousBucket = -1; const buckets = new Set<number>();
  for (const index of queryCatalog) { const entry = entries[index]!; const bucket = lexicalTermBucket(entry.namespace, entry.term); buckets.add(bucket);
    const order = catalogPrevious ? previousBucket - bucket || compareNamespaceTerm(catalogPrevious, entry) || catalogPrevious.postingOrdinal - entry.postingOrdinal : -1;
    if (catalogPrevious && order >= 0) throw new Error("queryCatalog order must strictly increase"); catalogPrevious = entry; previousBucket = bucket; }
  const bucketMask = bucketMaskHex(buckets); if (value.bucketMask !== bucketMask) throw new Error("dictionary bucketMask does not match queryCatalog");
  return { dictionaryOrdinal: value.dictionaryOrdinal, postingStart: value.postingStart, postingCount: value.postingCount, entries, queryCatalog, bucketMask };
}

function validateExactPermutation(value: unknown, count: number, name: string): number[] {
  if (!Array.isArray(value) || value.length !== count) throw new Error(`${name} must cover every authority index exactly once`);
  const bitmap = new Uint8Array(Math.ceil(count / 8)); const result: number[] = [];
  for (const raw of value) { requireIntegerInRange(raw, `${name} index`, 0, count - 1); const byte = raw >>> 3, mask = 1 << (raw & 7); if ((bitmap[byte]! & mask) !== 0) throw new Error(`${name} contains a duplicate index`); bitmap[byte]! |= mask; result.push(raw); }
  return result;
}
function bucketMaskHex(buckets: ReadonlySet<number>): string {
  const bytes = new Uint8Array(32); for (const bucket of buckets) bytes[bucket >>> 3]! |= 1 << (bucket & 7); return bytesToHex(bytes);
}

function validatePaperMetadataRecord(value: unknown, expectedOrdinal: number): PaperMetadataRecord {
  requireAllowedObject(value, ["paperOrdinal", "paperKey", "chunkStart", "chunkCount", "title"], ["paperOrdinal", "paperKey", "chunkStart", "chunkCount"], "paper metadata record");
  if (value.paperOrdinal !== expectedOrdinal) throw new Error("paper metadata ordinals must be continuous");
  if (!boundedString(value.paperKey, MAX_PAPER_KEY_LENGTH)) throw new Error("paper metadata paperKey is invalid");
  requireIntegerInRange(value.chunkStart, "paper metadata chunkStart", 0, MAX_GENERATION_CHUNKS - 1);
  requireIntegerInRange(value.chunkCount, "paper metadata chunkCount", 1, MAX_GENERATION_CHUNKS);
  if (value.chunkStart + value.chunkCount > MAX_GENERATION_CHUNKS) throw new Error("paper metadata chunk range exceeds corpus bounds");
  if (value.title !== undefined && (typeof value.title !== "string" || value.title.length > 16_384)) throw new Error("paper metadata title is invalid");
  return { paperOrdinal: value.paperOrdinal, paperKey: value.paperKey, chunkStart: value.chunkStart, chunkCount: value.chunkCount, ...(value.title === undefined ? {} : { title: value.title }) };
}

function validateEvidenceRecord(value: unknown): EvidenceBlockRecord {
  requireExactObject(value, ["paperIndex", "paperKey", "vectorRow", "chunk"], "evidence record");
  requireIntegerInRange(value.paperIndex, "evidence paper index", 0, MAX_GENERATION_PAPERS - 1);
  if (!boundedString(value.paperKey, MAX_PAPER_KEY_LENGTH)) throw new Error("evidence paper key exceeds its string limit");
  requireIntegerInRange(value.vectorRow, "evidence vector row", 0, MAX_GENERATION_CHUNKS - 1);
  requireEvidenceShapeBounds(value.chunk);
  const chunk = validateGenerationEvidenceChunk(value.chunk);
  if (!chunk) throw new Error("invalid evidence chunk identity, locator, or derivation");
  return { paperIndex: value.paperIndex, paperKey: value.paperKey, vectorRow: value.vectorRow, chunk };
}

function requireEvidenceShapeBounds(value: unknown): void {
  requireExactObject(value, ["id", "index", "page", "text", "headings", "locator", "derivation"], "evidence chunk");
  if (typeof value.text !== "string" || value.text.length > MAX_CHUNK_TEXT_LENGTH) {
    throw new Error("evidence chunk text exceeds its string limit");
  }
  if (!Array.isArray(value.headings) || value.headings.length > MAX_HEADINGS
    || value.headings.some((heading) => typeof heading !== "string" || heading.length > MAX_HEADING_LENGTH)) {
    throw new Error("evidence chunk headings exceed their limits");
  }
  requireAllowedObject(
    value.locator,
    ["pageStart", "pageEnd", "blockStart", "blockEnd", "bbox"],
    ["pageStart"],
    "evidence locator",
  );
  if (value.locator.bbox !== undefined) {
    requireExactObject(value.locator.bbox, ["left", "top", "right", "bottom"], "evidence bounding box");
  }
  const derivation = value.derivation;
  requireExactObject(derivation, ["parser", "chunkerVersion", "embeddingInputVersion"], "evidence derivation");
  requireExactObject(derivation.parser, ["id", "version"], "evidence parser provenance");
  if (!boundedString(derivation.parser.id, MAX_PROVENANCE_LENGTH)
    || !boundedString(derivation.parser.version, MAX_PROVENANCE_LENGTH)) {
    throw new Error("evidence chunk parser provenance exceeds its limits");
  }
}

function validateDescriptorV2(value: Record<string, any>): GenerationDescriptor {
  requireExactObject(value, [
    "formatVersion", "schemaVersion", "generationId", "sourceRevision", "scopeFingerprint",
    "identificationFingerprint", "modelId", "dimension", "corpusMean", "corpusStats", "indexDerivation", "objects",
  ], "generation descriptor schema v2");
  if (!Array.isArray(value.objects) || value.objects.some((object: unknown) => !isPlainObject(object)
    || (object.kind !== "vector" && object.kind !== "evidence"))) {
    throw new Error("generation descriptor schema v2 accepts dense vector/evidence objects only");
  }
  requireExactObject(value.corpusStats, ["indexedPaperCount", "chunkCount"], "generation schema v2 corpusStats");
  const projected = validateDescriptor({
    ...value,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    corpusStats: {
      indexedPaperCount: value.corpusStats.indexedPaperCount,
      chunkCount: value.corpusStats.chunkCount,
      totalLexicalTokenCount: 0,
      avgdl: 0,
      totalLexicalTokenCountWithHanSingles: 0,
      avgdlWithHanSingles: 0,
    },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: LEXICAL_BUCKET_COUNT }, () => [] as number[]),
  });
  return { ...projected, schemaVersion: 2 };
}

function validateDescriptor(value: unknown): GenerationDescriptor {
  requireExactObject(value, [
    "formatVersion", "schemaVersion", "generationId", "sourceRevision", "scopeFingerprint",
    "identificationFingerprint", "modelId", "dimension", "corpusMean", "corpusStats", "lexicalCapability", "lexicalRouting", "indexDerivation", "objects",
  ], "generation descriptor");
  if (value.formatVersion !== GENERATION_DESCRIPTOR_FORMAT_VERSION) throw new Error("unsupported generation descriptor format version");
  if (value.schemaVersion !== GENERATION_DESCRIPTOR_SCHEMA_VERSION) throw new Error("unsupported generation descriptor schema version");
  requireGenerationId(value.generationId);
  requireIntegerInRange(value.sourceRevision, "generation sourceRevision", 0, Number.MAX_SAFE_INTEGER);
  requireFingerprint(value.scopeFingerprint, "generation scopeFingerprint");
  requireFingerprint(value.identificationFingerprint, "generation identificationFingerprint");
  if (!boundedString(value.modelId, MAX_MODEL_ID_LENGTH)) throw new Error("generation modelId exceeds its string limit");
  requireIntegerInRange(value.dimension, "generation dimension", 1, MAX_GENERATION_DIMENSION);
  if (!Array.isArray(value.corpusMean) || value.corpusMean.length !== value.dimension
    || value.corpusMean.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))) {
    throw new Error("generation corpusMean must contain the finite per-dimension mean of all vector rows");
  }
  requireExactObject(value.corpusStats, ["indexedPaperCount", "chunkCount", "totalLexicalTokenCount", "avgdl", "totalLexicalTokenCountWithHanSingles", "avgdlWithHanSingles"], "generation corpusStats");
  requireIntegerInRange(value.corpusStats.indexedPaperCount, "generation indexed paper count", 0, MAX_GENERATION_PAPERS);
  requireIntegerInRange(value.corpusStats.chunkCount, "generation chunk count", 0, MAX_GENERATION_CHUNKS);
  requireIntegerInRange(value.corpusStats.totalLexicalTokenCount, "generation total lexical token count", 0, Number.MAX_SAFE_INTEGER);
  requireIntegerInRange(value.corpusStats.totalLexicalTokenCountWithHanSingles, "generation expanded lexical token count", 0, Number.MAX_SAFE_INTEGER);
  const expectedAvgdl = value.corpusStats.chunkCount === 0 ? 0 : value.corpusStats.totalLexicalTokenCount / value.corpusStats.chunkCount;
  const expectedExpandedAvgdl = value.corpusStats.chunkCount === 0 ? 0 : value.corpusStats.totalLexicalTokenCountWithHanSingles / value.corpusStats.chunkCount;
  if (!Object.is(value.corpusStats.avgdl, expectedAvgdl) || !Object.is(value.corpusStats.avgdlWithHanSingles, expectedExpandedAvgdl)) {
    throw new Error("generation avgdl values must exactly match their token totals/chunkCount");
  }
  if (value.lexicalCapability !== "none" && value.lexicalCapability !== "bm25-v1") throw new Error("generation lexicalCapability is unknown");
  if (!Array.isArray(value.lexicalRouting) || value.lexicalRouting.length !== LEXICAL_BUCKET_COUNT) throw new Error("generation lexicalRouting must contain exactly 256 route arrays");
  const lexicalRouting: number[][] = []; let routeRefs = 0;
  for (const route of value.lexicalRouting) {
    if (!Array.isArray(route)) throw new Error("generation lexicalRouting bucket must be an array");
    const ordinals: number[] = []; let previous = -1;
    for (const raw of route) {
      if (typeof raw !== "number" || !Number.isInteger(raw) || raw < 0) {
        throw new Error("lexical routing entry must be a dictionary ordinal");
      }
      if (raw <= previous) throw new Error("lexical routing ordinals must strictly increase");
      previous = raw; ordinals.push(raw);
    }
    routeRefs += ordinals.length; lexicalRouting.push(ordinals);
  }
  if (routeRefs > MAX_GENERATION_ROUTE_REFS) throw new Error("lexical routing exceeds its reference limit");
  if ((value.corpusStats.chunkCount === 0) !== (value.corpusStats.indexedPaperCount === 0)
    || value.corpusStats.indexedPaperCount > value.corpusStats.chunkCount) {
    throw new Error("generation indexed paper count must be positive only for a non-empty corpus and cannot exceed chunkCount");
  }
  if (value.corpusStats.chunkCount === 0 && value.corpusMean.some((entry) => entry !== 0)) {
    throw new Error("generation corpusMean must be all zero for an empty corpus");
  }
  requireExactObject(
    value.indexDerivation,
    ["builderVersion", "denseCenteringVersion", "tokenizerVersion", "postingsVersion"],
    "generation indexDerivation",
  );
  const indexDerivation: GenerationIndexDerivation = {
    builderVersion: requireBoundedVersion(value.indexDerivation.builderVersion, "builderVersion"),
    denseCenteringVersion: requireBoundedVersion(value.indexDerivation.denseCenteringVersion, "denseCenteringVersion"),
    tokenizerVersion: requireBoundedVersion(value.indexDerivation.tokenizerVersion, "tokenizerVersion"),
    postingsVersion: requireBoundedVersion(value.indexDerivation.postingsVersion, "postingsVersion"),
  };
  if (!Array.isArray(value.objects) || value.objects.length > MAX_GENERATION_OBJECTS) {
    throw new Error("generation objects exceed their count limit");
  }
  const paths = new Set<string>();
  const objects = value.objects.map((object) => {
    requireExactObject(
      object,
      ["kind", "path", "byteLength", "recordStart", "recordCount", "checksum"],
      "generation object reference",
    );
    if (object.kind !== "vector" && object.kind !== "evidence" && object.kind !== "paper-metadata"
      && object.kind !== "lexical-dictionary" && object.kind !== "lexical-postings") throw new Error("generation object kind is unknown");
    if (typeof object.path !== "string" || !/^objects\/[a-z0-9][a-z0-9._-]{0,127}$/.test(object.path)) {
      throw new Error("generation object path must be a bounded logical child of objects/");
    }
    if (paths.has(object.path)) throw new Error("generation object paths must be unique");
    paths.add(object.path);
    requireIntegerInRange(object.byteLength, "generation object byteLength", BINARY_BLOCK_HEADER_BYTES, MAX_BINARY_OBJECT_BYTES);
    requireIntegerInRange(object.recordStart, "generation object recordStart", 0, MAX_GENERATION_CHUNKS - 1);
    requireIntegerInRange(object.recordCount, "generation object recordCount", 1, MAX_GENERATION_CHUNKS);
    if (object.recordStart + object.recordCount > MAX_GENERATION_CHUNKS) {
      throw new Error("generation object record range exceeds the chunk limit");
    }
    requireFingerprint(object.checksum, "generation object checksum");
    return {
      kind: object.kind,
      path: object.path,
      byteLength: object.byteLength,
      recordStart: object.recordStart,
      recordCount: object.recordCount,
      checksum: object.checksum,
    };
  });
  const lexicalKinds: readonly GenerationObjectKind[] = ["lexical-postings", "lexical-dictionary"];
  const hasLexicalObjects = objects.some((object) => lexicalKinds.includes(object.kind));
  const dictionaryOrdinals = new Set(objects.filter((object) => object.kind === "lexical-dictionary").map((_, index) => index));
  const routedOrdinals = new Set(lexicalRouting.flat());
  if (value.corpusStats.chunkCount === 0 && value.lexicalCapability !== "none") throw new Error("empty generation lexicalCapability must be none");
  if (value.lexicalCapability === "none" && (hasLexicalObjects || routeRefs !== 0 || value.corpusStats.totalLexicalTokenCount !== 0 || value.corpusStats.totalLexicalTokenCountWithHanSingles !== 0)) throw new Error("dense-only generation cannot declare lexical objects, routing, or statistics");
  if (value.lexicalCapability === "bm25-v1") {
    if (!objects.some((object) => object.kind === "paper-metadata")) throw new Error("bm25 generation is missing paper metadata");
    const hasPostings = objects.some((object) => object.kind === "lexical-postings");
    const hasDictionaries = objects.some((object) => object.kind === "lexical-dictionary");
    if (hasPostings !== hasDictionaries) throw new Error("bm25 postings and dictionary objects must appear together");
    const hasIndexedTerms = value.corpusStats.totalLexicalTokenCount > 0
      || value.corpusStats.totalLexicalTokenCountWithHanSingles > 0;
    if (hasIndexedTerms && (!hasPostings || !hasDictionaries || routeRefs === 0)) {
      throw new Error("bm25 generation with lexical tokens requires postings, dictionary, and routing");
    }
    for (const ordinal of routedOrdinals) if (!dictionaryOrdinals.has(ordinal)) throw new Error("lexical routing must reference a dictionary object");
    for (const ordinal of dictionaryOrdinals) if (!routedOrdinals.has(ordinal)) throw new Error("every dictionary object requires at least one bucket route");
  }
  validateObjectCoverage(objects, value.corpusStats.chunkCount, value.corpusStats.indexedPaperCount);

  const dictionaryCount = objects.filter((object) => object.kind === "lexical-dictionary").length;
  for (const route of lexicalRouting) {
    for (const ordinal of route) {
      if (ordinal >= dictionaryCount) throw new Error("lexical routing names a dictionary ordinal that does not exist");
    }
  }

  return {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId: value.generationId,
    sourceRevision: value.sourceRevision,
    scopeFingerprint: value.scopeFingerprint,
    identificationFingerprint: value.identificationFingerprint,
    modelId: value.modelId,
    dimension: value.dimension,
    corpusMean: [...value.corpusMean],
    corpusStats: {
      indexedPaperCount: value.corpusStats.indexedPaperCount,
      chunkCount: value.corpusStats.chunkCount,
      totalLexicalTokenCount: value.corpusStats.totalLexicalTokenCount,
      avgdl: value.corpusStats.avgdl,
      totalLexicalTokenCountWithHanSingles: value.corpusStats.totalLexicalTokenCountWithHanSingles,
      avgdlWithHanSingles: value.corpusStats.avgdlWithHanSingles,
    },
    lexicalCapability: value.lexicalCapability,
    lexicalRouting,
    indexDerivation,
    objects,
  };
}

function validateObjectCoverage(objects: readonly GenerationObjectReference[], chunkCount: number, paperCount: number): void {
  if (chunkCount === 0) {
    if (objects.length !== 0) throw new Error("empty generation cannot reference objects");
    return;
  }
  const kindOrder: Record<GenerationObjectKind, number> = {
    vector: 0, evidence: 1, "paper-metadata": 2, "lexical-postings": 3, "lexical-dictionary": 4,
  };
  let previousKind: GenerationObjectKind | null = null;
  let previousStart = -1;
  let previousPath = "";
  for (const object of objects) {
    if (previousKind !== null && (kindOrder[object.kind] < kindOrder[previousKind]
      || (object.kind === previousKind && object.recordStart < previousStart)
      || (object.kind === previousKind && object.recordStart === previousStart && object.path <= previousPath))) {
      throw new Error("generation object references must use canonical kind, recordStart, and path order");
    }
    previousKind = object.kind;
    previousStart = object.recordStart;
    previousPath = object.path;
  }
  for (const kind of ["vector", "evidence"] as const) {
    const refs = objects.filter((object) => object.kind === kind);
    if (refs.length === 0) throw new Error(`non-empty generation is missing ${kind} object coverage`);
    let nextStart = 0;
    for (const ref of refs) {
      if (ref.recordStart !== nextStart) throw new Error(`${kind} object coverage must be continuous from zero without gaps or overlaps`);
      nextStart += ref.recordCount;
    }
    if (nextStart !== chunkCount) throw new Error(`${kind} object coverage must equal chunkCount`);
  }
  const metadata = objects.filter((object) => object.kind === "paper-metadata");
  if (metadata.length > 0) {
    let nextPaper = 0;
    for (const ref of metadata) {
      if (ref.recordStart !== nextPaper) throw new Error("paper-metadata object coverage must be continuous from zero");
      nextPaper += ref.recordCount;
    }
    if (nextPaper !== paperCount) throw new Error("paper-metadata object coverage must equal indexedPaperCount");
  }
  const postings = objects.filter((object) => object.kind === "lexical-postings");
  if (postings.length > 0) { let nextChunk = 0; for (const ref of postings) { if (ref.recordStart !== nextChunk) throw new Error("lexical-postings chunk coverage must be continuous from zero"); nextChunk += ref.recordCount; } if (nextChunk !== chunkCount) throw new Error("lexical-postings chunk coverage must equal chunkCount"); }
  const dictionaries = objects.filter((object) => object.kind === "lexical-dictionary");
  if (dictionaries.length > 0) { let nextPosting = 0; for (const ref of dictionaries) { if (ref.recordStart !== nextPosting) throw new Error("lexical-dictionary posting ranges must be continuous from zero"); nextPosting += ref.recordCount; } if (nextPosting !== postings.length) throw new Error("lexical-dictionary posting ranges must cover all postings ordinals"); }
}

function requireBoundedVersion(value: unknown, name: string): number {
  requireIntegerInRange(value, `generation indexDerivation ${name}`, 1, 0xffff_ffff);
  return value;
}

function validateGenerationEvidenceChunk(value: Record<string, any>): EvidenceChunk | null {
  if (!isFingerprint(value.id) || !isNonNegativeInteger(value.index) || !isPositiveInteger(value.page)) return null;
  if (typeof value.text !== "string" || !Array.isArray(value.headings) || !value.headings.every(isNonEmptyString)) return null;
  if (!isPlainObject(value.locator) || !isPositiveInteger(value.locator.pageStart) || value.page !== value.locator.pageStart) return null;
  if (value.locator.pageEnd !== undefined
    && (!isPositiveInteger(value.locator.pageEnd) || value.locator.pageEnd < value.locator.pageStart)) return null;
  if (value.locator.blockStart !== undefined && !isNonNegativeInteger(value.locator.blockStart)) return null;
  if (value.locator.blockEnd !== undefined && !isNonNegativeInteger(value.locator.blockEnd)) return null;
  if (value.locator.bbox !== undefined && !isNormalizedBoundingBox(value.locator.bbox)) return null;
  const derivation = validateGenerationEvidenceDerivation(value.derivation);
  if (!derivation) return null;
  const locator = {
    pageStart: value.locator.pageStart,
    ...(value.locator.pageEnd === undefined ? {} : { pageEnd: value.locator.pageEnd }),
    ...(value.locator.blockStart === undefined ? {} : { blockStart: value.locator.blockStart }),
    ...(value.locator.blockEnd === undefined ? {} : { blockEnd: value.locator.blockEnd }),
    ...(value.locator.bbox === undefined ? {} : { bbox: { ...value.locator.bbox } }),
  };
  const identity = { text: value.text, headings: [...value.headings], locator, derivation };
  if (createEvidenceChunkId(identity) !== value.id) return null;
  return { id: value.id, index: value.index, page: value.page, ...identity };
}

function validateGenerationEvidenceDerivation(value: unknown): EvidenceDerivation | null {
  if (!isPlainObject(value) || !isPlainObject(value.parser)) return null;
  if (!isNonEmptyString(value.parser.id) || !isNonEmptyString(value.parser.version)) return null;
  if (!isPositiveInteger(value.chunkerVersion) || !isPositiveInteger(value.embeddingInputVersion)) return null;
  return {
    parser: { id: value.parser.id, version: value.parser.version },
    chunkerVersion: value.chunkerVersion,
    embeddingInputVersion: value.embeddingInputVersion,
  };
}

function isNormalizedBoundingBox(value: unknown): boolean {
  if (!isPlainObject(value)) return false;
  const numbers = [value.left, value.top, value.right, value.bottom];
  return numbers.every((number) => typeof number === "number" && Number.isFinite(number) && number >= 0 && number <= 1)
    && value.left <= value.right && value.top <= value.bottom;
}

function isFingerprint(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[a-f0-9]{64}$/.test(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function requireGenerationId(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/.test(value)) {
    throw new Error("generationId must be 1-64 lowercase alphanumeric/hyphen characters without edge hyphens");
  }
}

function requireObjectLength(byteLength: number): void {
  if (!Number.isSafeInteger(byteLength) || byteLength > MAX_BINARY_OBJECT_BYTES) {
    throw new Error("binary object exceeds its byte limit");
  }
}

function requireIntegerInRange(value: unknown, name: string, minimum: number, maximum: number): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new Error(`${name} is outside its safe range`);
  }
}

function requireExactObject(value: unknown, keys: readonly string[], name: string): asserts value is Record<string, any> {
  if (!isPlainObject(value)) throw new Error(`${name} must be an object`);
  const actual = Object.keys(value);
  if (actual.length !== keys.length || actual.some((key) => !keys.includes(key))) {
    throw new Error(`${name} has an unknown field or missing required field`);
  }
}

function requireAllowedObject(
  value: unknown,
  allowed: readonly string[],
  required: readonly string[],
  name: string,
): asserts value is Record<string, any> {
  if (!isPlainObject(value)) throw new Error(`${name} must be an object`);
  const actual = Object.keys(value);
  if (actual.some((key) => !allowed.includes(key)) || required.some((key) => !actual.includes(key))) {
    throw new Error(`${name} has an unknown field or missing required field`);
  }
}

function requireFingerprint(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !/^sha256:[a-f0-9]{64}$/.test(value)) throw new Error(`${name} must be a SHA-256 fingerprint`);
}

function fingerprintHex(name: string, value: unknown): string {
  requireFingerprint(value, name);
  return value.slice("sha256:".length);
}

function boundedString(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.trim().length > 0 && value.length <= maximum;
}

function isPlainObject(value: unknown): value is Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  return bytes;
}

function bytesToHex(bytes: Uint8Array): string {
  let hex = "";
  for (const byte of bytes) hex += byte.toString(16).padStart(2, "0");
  return hex;
}
