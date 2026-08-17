import type { OutputSettings } from "../../settings/types";
import { sha256Hex } from "../../utils/digest";
import { derivePaperInboxPaths } from "../../services/paper-index";
import { createEvidenceChunkId, type EvidenceChunk, type EvidenceDerivation } from "./evidence-chunk";

export const MAX_BINARY_OBJECT_BYTES = 4 * 1024 * 1024;
export const BINARY_BLOCK_HEADER_BYTES = 52;
export const BINARY_BLOCK_FORMAT_VERSION = 1 as const;
export const BINARY_BLOCK_SCHEMA_VERSION = 2 as const;
export const GENERATION_DESCRIPTOR_FORMAT_VERSION = 1 as const;
export const GENERATION_DESCRIPTOR_SCHEMA_VERSION = 2 as const;
export const MAX_GENERATION_DESCRIPTOR_BYTES = 1024 * 1024;
export const MAX_GENERATION_OBJECTS = 4096;
export const MAX_GENERATION_PAPERS = 1_000_000;
export const MAX_GENERATION_CHUNKS = 10_000_000;
export const MAX_GENERATION_DIMENSION = 4096;

const MAGIC = new Uint8Array([0x41, 0x44, 0x47, 0x49]); // ADGI
const CHECKSUM_OFFSET = 20;
const CHECKSUM_BYTES = 32;
const VECTOR_KIND = 1;
const EVIDENCE_KIND = 2;
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

export type GenerationObjectKind = "vector" | "evidence";

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
  readonly schemaVersion: typeof GENERATION_DESCRIPTOR_SCHEMA_VERSION;
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
  };
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

export function decodeVectorBlock(bytes: Uint8Array): VectorBlock {
  const block = decodeBlock(bytes, VECTOR_KIND);
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

export function decodeEvidenceBlock(bytes: Uint8Array): EvidenceBlock {
  const block = decodeBlock(bytes, EVIDENCE_KIND);
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

function decodeBlock(bytes: Uint8Array, expectedKind: number): { payload: Uint8Array; recordCount: number } {
  if (!(bytes instanceof Uint8Array)) throw new Error("binary block must be a Uint8Array");
  requireObjectLength(bytes.byteLength);
  if (bytes.byteLength < BINARY_BLOCK_HEADER_BYTES) throw new Error("binary block header is truncated");
  for (let index = 0; index < MAGIC.length; index += 1) {
    if (bytes[index] !== MAGIC[index]) throw new Error("binary block magic does not match");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint16(4, true) !== BINARY_BLOCK_FORMAT_VERSION) throw new Error("unsupported binary block format version");
  if (view.getUint16(6, true) !== BINARY_BLOCK_SCHEMA_VERSION) throw new Error("unsupported binary block schema version");
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
  for (let index = 1; index < ordinals.length; index += 1) {
    const previous = ordinals[index - 1]!;
    const current = ordinals[index]!;
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

function validateDescriptor(value: unknown): GenerationDescriptor {
  requireExactObject(value, [
    "formatVersion", "schemaVersion", "generationId", "sourceRevision", "scopeFingerprint",
    "identificationFingerprint", "modelId", "dimension", "corpusMean", "corpusStats", "indexDerivation", "objects",
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
  requireExactObject(value.corpusStats, ["indexedPaperCount", "chunkCount"], "generation corpusStats");
  requireIntegerInRange(value.corpusStats.indexedPaperCount, "generation indexed paper count", 0, MAX_GENERATION_PAPERS);
  requireIntegerInRange(value.corpusStats.chunkCount, "generation chunk count", 0, MAX_GENERATION_CHUNKS);
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
    if (object.kind === "lexical-postings") {
      throw new Error("lexical-postings references require a future format version with its postings codec");
    }
    if (object.kind !== "vector" && object.kind !== "evidence") throw new Error("generation object kind is unknown");
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
  validateObjectCoverage(objects, value.corpusStats.chunkCount);
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
    },
    indexDerivation,
    objects,
  };
}

function validateObjectCoverage(objects: readonly GenerationObjectReference[], chunkCount: number): void {
  if (chunkCount === 0) {
    if (objects.length !== 0) throw new Error("empty generation cannot reference vector or evidence objects");
    return;
  }
  let previousKind = "";
  let previousStart = -1;
  let previousPath = "";
  for (const object of objects) {
    const kindOrder = object.kind === "vector" ? 0 : 1;
    const previousKindOrder = previousKind === "" ? -1 : previousKind === "vector" ? 0 : 1;
    if (kindOrder < previousKindOrder
      || (object.kind === previousKind && object.recordStart < previousStart)
      || (object.kind === previousKind && object.recordStart === previousStart && object.path <= previousPath)) {
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
      if (ref.recordStart !== nextStart) {
        throw new Error(`${kind} object coverage must be continuous from zero without gaps or overlaps`);
      }
      nextStart += ref.recordCount;
    }
    if (nextStart !== chunkCount) throw new Error(`${kind} object coverage must equal chunkCount`);
  }
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
