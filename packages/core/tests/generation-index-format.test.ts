import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { createEvidenceChunkId, type EvidenceChunk } from "../src/library/fulltext/evidence-chunk";
import {
  lexicalBucketMask,
  lexicalQueryCatalog,
  lexicalTermBuckets,
} from "../src/library/fulltext/generation-lexical-derivation";
import {
  BINARY_BLOCK_HEADER_BYTES,
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  MAX_BINARY_OBJECT_BYTES,
  MAX_GENERATION_DIMENSION,
  MAX_GENERATION_OBJECTS,
  MAX_GENERATION_ROUTE_REFS,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeGenerationDescriptor,
  decodeLexicalDictionaryBlock,
  decodeLexicalPostingsBlock,
  decodePaperMetadataBlock,
  decodeVectorBlock,
  deriveFullTextGenerationPaths,
  encodeEvidenceBlock,
  encodeGenerationDescriptor,
  encodeLexicalDictionaryBlock,
  encodeLexicalPostingsBlock,
  encodePaperMetadataBlock,
  encodeVectorBlock,
  compareTermCodePoints,
  lexicalTermBucket,
  finishEvidenceStreamClosure,
  validateEvidenceStreamClosure,
  type EvidenceBlockRecord,
  type GenerationDescriptor,
} from "../src/library/fulltext/generation-index-format";
import { sha256Hex } from "../src/utils/digest";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;
const OBJECT_CHECKSUM = `sha256:${"c".repeat(64)}`;

function makeChunk(overrides: Partial<EvidenceChunk> = {}): EvidenceChunk {
  const identity = {
    text: overrides.text ?? "Evidence text with μ and 中文",
    headings: overrides.headings ?? ["Methods", "Ablation"],
    locator: overrides.locator ?? {
      pageStart: 3,
      pageEnd: 4,
      blockStart: 7,
      blockEnd: 9,
      bbox: { left: 0.1, top: 0.2, right: 0.8, bottom: 0.4 },
    },
    derivation: overrides.derivation ?? {
      parser: { id: "fixture-parser", version: "2.1.0" },
      chunkerVersion: 2,
      embeddingInputVersion: 1,
    },
  };
  return {
    id: overrides.id ?? createEvidenceChunkId(identity),
    index: overrides.index ?? 0,
    page: overrides.page ?? identity.locator.pageStart,
    ...identity,
  };
}

function makeEvidenceRecord(
  vectorRow: number,
  paperIndex: number,
  chunkIndex: number,
  paperKey = `paper:${paperIndex}`,
): EvidenceBlockRecord {
  return { paperIndex, paperKey, vectorRow, chunk: makeChunk({ index: chunkIndex }) };
}

function makeDescriptor(overrides: Partial<GenerationDescriptor> = {}): GenerationDescriptor {
  return {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId: "gen-20260817-a1",
    sourceRevision: 42,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    modelId: "multilingual-e5-small-q8",
    dimension: 3,
    corpusMean: [0.25, -0.5, 0.75],
    corpusStats: { indexedPaperCount: 1, chunkCount: 2, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: 256 }, () => [] as number[]),
    indexDerivation: {
      builderVersion: 1,
      denseCenteringVersion: 1,
      tokenizerVersion: 1,
      postingsVersion: 1,
    },
    objects: [
      {
        kind: "vector",
        path: "objects/000001.vectors.bin",
        byteLength: 128,
        recordStart: 0,
        recordCount: 2,
        checksum: OBJECT_CHECKSUM,
      },
      {
        kind: "evidence",
        path: "objects/000001.evidence.bin",
        byteLength: 512,
        recordStart: 0,
        recordCount: 2,
        checksum: `sha256:${"d".repeat(64)}`,
      },
      {
        kind: "paper-metadata",
        path: "objects/000001.metadata.bin",
        byteLength: 256,
        recordStart: 0,
        recordCount: 1,
        checksum: `sha256:${"e".repeat(64)}`,
      },
    ],
    ...overrides,
  };
}

function rewriteBlockHeader(bytes: Uint8Array, mutate: (view: DataView) => void): Uint8Array {
  const copy = bytes.slice();
  const view = new DataView(copy.buffer, copy.byteOffset, copy.byteLength);
  mutate(view);
  const checksumInput = new Uint8Array(copy.length - 32);
  checksumInput.set(copy.subarray(0, 20), 0);
  checksumInput.set(copy.subarray(BINARY_BLOCK_HEADER_BYTES), 20);
  const digest = sha256Hex(checksumInput);
  for (let index = 0; index < 32; index += 1) {
    copy[20 + index] = Number.parseInt(digest.slice(index * 2, index * 2 + 2), 16);
  }
  return copy;
}

describe("generation vector binary block", () => {
  it("round-trips little-endian float32 rows with explicit row metadata", () => {
    const encoded = encodeVectorBlock({
      rowStart: 9,
      dimension: 3,
      paperOrdinals: new Uint32Array([4, 5]),
      vectors: new Float32Array([1.5, -2.25, 3.125, 4, 5, 6]),
    });
    const decoded = decodeVectorBlock(encoded);
    expect(decoded.rowStart).toBe(9);
    expect(decoded.rowCount).toBe(2);
    expect(decoded.dimension).toBe(3);
    expect(Array.from(decoded.paperOrdinals)).toEqual([4, 5]);
    expect(Array.from(decoded.vectors)).toEqual([1.5, -2.25, 3.125, 4, 5, 6]);
    expect(encoded.byteLength).toBe(BINARY_BLOCK_HEADER_BYTES + 8 + 2 * 4 + 6 * 4);
    expect(blockObjectChecksum(encoded)).toMatch(/^sha256:[a-f0-9]{64}$/);
  });

  it("accepts an object exactly at the byte cap and rejects one byte over it", () => {
    const floatCount = Math.floor((MAX_BINARY_OBJECT_BYTES - BINARY_BLOCK_HEADER_BYTES - 8) / 8);
    const encoded = encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      paperOrdinals: new Uint32Array(floatCount),
      vectors: new Float32Array(floatCount),
    });
    expect(encoded.byteLength).toBeLessThanOrEqual(MAX_BINARY_OBJECT_BYTES);
    expect(MAX_BINARY_OBJECT_BYTES - encoded.byteLength).toBeLessThan(8);
    expect(decodeVectorBlock(encoded).rowCount).toBe(floatCount);
    expect(() => encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      paperOrdinals: new Uint32Array(floatCount + 1),
      vectors: new Float32Array(floatCount + 1),
    })).toThrow(/byte limit/i);
    expect(() => decodeVectorBlock(new Uint8Array(MAX_BINARY_OBJECT_BYTES + 1))).toThrow(/byte limit/i);
  });

  it("rejects truncated, trailing, wrong-kind, future-version, checksum, count, and dimension tampering", () => {
    const valid = encodeVectorBlock({ rowStart: 2, dimension: 2, paperOrdinals: new Uint32Array([0, 1]), vectors: new Float32Array([1, 2, 3, 4]) });
    expect(() => decodeVectorBlock(valid.subarray(0, valid.length - 1))).toThrow(/truncated|length/i);
    const trailing = new Uint8Array(valid.length + 1);
    trailing.set(valid);
    expect(() => decodeVectorBlock(trailing)).toThrow(/trailing|length/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint16(8, 2, true))))
      .toThrow(/kind/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint16(4, 2, true))))
      .toThrow(/format version/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint16(6, 2, true))))
      .toThrow(/schema version/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint16(6, 5, true))))
      .toThrow(/schema version/i);
    const badChecksum = valid.slice();
    badChecksum[badChecksum.length - 1]! ^= 0xff;
    expect(() => decodeVectorBlock(badChecksum)).toThrow(/checksum/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint32(16, 0xffff_ffff, true))))
      .toThrow(/record count/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint32(BINARY_BLOCK_HEADER_BYTES, 0xffff_ffff, true))))
      .toThrow(/dimension/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint32(BINARY_BLOCK_HEADER_BYTES + 4, 0xffff_ffff, true))))
      .toThrow(/row range/i);
  });

  it("requires one little-endian paper ordinal per row with only same or +1 transitions", () => {
    expect(() => encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      paperOrdinals: new Uint32Array([0]),
      vectors: new Float32Array([1, 2]),
    })).toThrow(/one ordinal per row/i);
    expect(() => encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      paperOrdinals: new Uint32Array([0, 2]),
      vectors: new Float32Array([1, 2]),
    })).toThrow(/ordinals/i);
    const encoded = encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      paperOrdinals: new Uint32Array([0x0001_0203]),
      vectors: new Float32Array([1]),
    });
    expect(Array.from(encoded.subarray(BINARY_BLOCK_HEADER_BYTES + 8, BINARY_BLOCK_HEADER_BYTES + 12)))
      .toEqual([3, 2, 1, 0]);
    expect(() => encodeVectorBlock({
      rowStart: 0, dimension: 1, paperOrdinals: new Uint32Array([999_999]), vectors: new Float32Array([1]),
    })).not.toThrow();
    for (const ordinal of [1_000_000, 0xffff_ffff]) {
      expect(() => encodeVectorBlock({
        rowStart: 0, dimension: 1, paperOrdinals: new Uint32Array([ordinal]), vectors: new Float32Array([1]),
      })).toThrow(/paper ordinal/i);
    }

    const twoRows = encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      paperOrdinals: new Uint32Array([0, 1]),
      vectors: new Float32Array([1, 2]),
    });
    const skippedOrdinal = rewriteBlockHeader(twoRows, (view) => {
      view.setUint32(BINARY_BLOCK_HEADER_BYTES + 12, 2, true);
    });
    expect(() => decodeVectorBlock(skippedOrdinal)).toThrow(/ordinals/i);
    for (const ordinal of [1_000_000, 0xffff_ffff]) {
      const outOfRange = rewriteBlockHeader(twoRows, (view) => {
        view.setUint32(BINARY_BLOCK_HEADER_BYTES + 8, ordinal, true);
      });
      expect(() => decodeVectorBlock(outOfRange)).toThrow(/paper ordinal/i);
    }
    expect(() => decodeVectorBlock(rewriteBlockHeader(twoRows, (view) => view.setUint32(16, 1, true))))
      .toThrow(/payload length|ordinals/i);
  });

  it("rejects magic and payload-length tampering before payload loops", () => {
    const valid = encodeVectorBlock({ rowStart: 0, dimension: 1, paperOrdinals: new Uint32Array([0]), vectors: new Float32Array([1]) });
    const badMagic = valid.slice();
    badMagic[0] = 0;
    expect(() => decodeVectorBlock(badMagic)).toThrow(/magic/i);
    expect(() => decodeVectorBlock(rewriteBlockHeader(valid, (view) => view.setUint32(12, 0xffff_ffff, true))))
      .toThrow(/payload length|byte limit/i);
  });
});

describe("generation evidence binary block", () => {
  it("losslessly round-trips chunk identity, headings, locator, text, and derivation", () => {
    const chunk = makeChunk();
    const encoded = encodeEvidenceBlock({ rowStart: 9, records: [{ paperIndex: 5, paperKey: "arxiv:2403.19236", vectorRow: 9, chunk }] });
    expect(decodeEvidenceBlock(encoded)).toEqual({
      rowStart: 9,
      records: [{ paperIndex: 5, paperKey: "arxiv:2403.19236", vectorRow: 9, chunk }],
    });
  });

  it("rejects recomputed-identity and locator violations through the EvidenceChunk contract", () => {
    const encoded = encodeEvidenceBlock({ rowStart: 0, records: [{ paperIndex: 0, paperKey: "arxiv:2403.19236", vectorRow: 0, chunk: makeChunk() }] });
    const payload = JSON.parse(new TextDecoder().decode(encoded.subarray(BINARY_BLOCK_HEADER_BYTES)));
    payload.records[0].chunk.text = "tampered";
    const tamperedPayload = new TextEncoder().encode(JSON.stringify(payload));
    const tampered = new Uint8Array(BINARY_BLOCK_HEADER_BYTES + tamperedPayload.length);
    tampered.set(encoded.subarray(0, BINARY_BLOCK_HEADER_BYTES));
    tampered.set(tamperedPayload, BINARY_BLOCK_HEADER_BYTES);
    const resealed = rewriteBlockHeader(tampered, (view) => view.setUint32(12, tamperedPayload.length, true));
    expect(() => decodeEvidenceBlock(resealed)).toThrow(/evidence chunk/i);

    const invalid = makeChunk({ page: 2 });
    expect(() => encodeEvidenceBlock({ rowStart: 0, records: [{ paperIndex: 0, paperKey: "arxiv:2403.19236", vectorRow: 0, chunk: invalid }] }))
      .toThrow(/evidence chunk/i);
  });

  it("strictly rejects unsafe offsets, unknown JSON fields, wrong kind, and invalid UTF-8", () => {
    const valid = encodeEvidenceBlock({ rowStart: 0, records: [{ paperIndex: 0, paperKey: "arxiv:2403.19236", vectorRow: 0, chunk: makeChunk() }] });
    const payload = JSON.parse(new TextDecoder().decode(valid.subarray(BINARY_BLOCK_HEADER_BYTES)));
    payload.records[0].vectorRow = Number.MAX_SAFE_INTEGER;
    const bytes = new TextEncoder().encode(JSON.stringify(payload));
    const unsafe = new Uint8Array(BINARY_BLOCK_HEADER_BYTES + bytes.length);
    unsafe.set(valid.subarray(0, BINARY_BLOCK_HEADER_BYTES));
    unsafe.set(bytes, BINARY_BLOCK_HEADER_BYTES);
    expect(() => decodeEvidenceBlock(rewriteBlockHeader(unsafe, (view) => view.setUint32(12, bytes.length, true))))
      .toThrow(/vector row/i);

    payload.records[0].vectorRow = 0;
    payload.extra = true;
    const extraBytes = new TextEncoder().encode(JSON.stringify(payload));
    const extra = new Uint8Array(BINARY_BLOCK_HEADER_BYTES + extraBytes.length);
    extra.set(valid.subarray(0, BINARY_BLOCK_HEADER_BYTES));
    extra.set(extraBytes, BINARY_BLOCK_HEADER_BYTES);
    expect(() => decodeEvidenceBlock(rewriteBlockHeader(extra, (view) => view.setUint32(12, extraBytes.length, true))))
      .toThrow(/unknown field/i);

    expect(() => decodeEvidenceBlock(rewriteBlockHeader(valid, (view) => view.setUint16(8, 1, true))))
      .toThrow(/kind/i);
    const invalidUtf8 = valid.slice(0, BINARY_BLOCK_HEADER_BYTES + 1);
    invalidUtf8[BINARY_BLOCK_HEADER_BYTES] = 0xff;
    expect(() => decodeEvidenceBlock(rewriteBlockHeader(invalidUtf8, (view) => view.setUint32(12, 1, true))))
      .toThrow(/UTF-8/i);
  });

  it("requires vector rows to equal rowStart+i and enforces local paper/chunk order", () => {
    expect(() => encodeEvidenceBlock({
      rowStart: 10,
      records: [makeEvidenceRecord(10, 0, 0), makeEvidenceRecord(12, 0, 1)],
    })).toThrow(/vector row/i);
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [makeEvidenceRecord(0, 1, 0), makeEvidenceRecord(1, 0, 0)],
    })).toThrow(/paper index|paper order/i);
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [makeEvidenceRecord(0, 0, 0, "paper:same"), makeEvidenceRecord(1, 1, 0, "paper:same")],
    })).toThrow(/paper key/i);
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [
        makeEvidenceRecord(0, 0, 0, "paper:a"),
        makeEvidenceRecord(1, 1, 0, "paper:b"),
        makeEvidenceRecord(2, 2, 0, "paper:a"),
      ],
    })).toThrow(/paper key.*order/i);
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [makeEvidenceRecord(0, 0, 0, "paper:b"), makeEvidenceRecord(1, 1, 0, "paper:a")],
    })).toThrow(/paper key.*order/i);
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [makeEvidenceRecord(0, 0, 0, "paper:a"), makeEvidenceRecord(1, 1, 0, "paper:b")],
    })).not.toThrow();
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [makeEvidenceRecord(0, 0, 3), makeEvidenceRecord(1, 0, 5)],
    })).toThrow(/chunk index/i);
    expect(() => encodeEvidenceBlock({
      rowStart: 0,
      records: [
        makeEvidenceRecord(0, 0, 0),
        makeEvidenceRecord(1, 1, 0),
        makeEvidenceRecord(2, 0, 1),
      ],
    })).toThrow(/paper index|paper order/i);
  });

  it("validates generation closure across blocks without a corpus Set", () => {
    expect(() => validateEvidenceStreamClosure(null, [makeEvidenceRecord(0, 500, 0)]))
      .toThrow(/first paper index/i);

    const first = validateEvidenceStreamClosure(null, [
      makeEvidenceRecord(0, 0, 0, "paper:a"),
      makeEvidenceRecord(1, 0, 1, "paper:a"),
    ]);
    const continued = validateEvidenceStreamClosure(first, [
      makeEvidenceRecord(2, 0, 2, "paper:a"),
      makeEvidenceRecord(3, 1, 0, "paper:b"),
    ]);
    expect(continued).toEqual({ paperIndex: 1, paperKey: "paper:b", chunkIndex: 0, vectorRow: 3 });
    expect(() => finishEvidenceStreamClosure(continued, 2)).not.toThrow();
    expect(() => finishEvidenceStreamClosure(continued, 3)).toThrow(/indexedPaperCount/i);

    expect(() => validateEvidenceStreamClosure(continued, [makeEvidenceRecord(4, 0, 1, "paper:a")]))
      .toThrow(/paper index/i);
    expect(() => validateEvidenceStreamClosure(continued, [makeEvidenceRecord(4, 3, 0, "paper:c")]))
      .toThrow(/paper index/i);
    expect(() => validateEvidenceStreamClosure(continued, [makeEvidenceRecord(4, 2, 1, "paper:c")]))
      .toThrow(/chunk index/i);

    const paperA = validateEvidenceStreamClosure(null, [makeEvidenceRecord(0, 0, 0, "paper:a")]);
    expect(() => validateEvidenceStreamClosure(paperA, [makeEvidenceRecord(1, 1, 0, "paper:b")]))
      .not.toThrow();
    const paperB = validateEvidenceStreamClosure(null, [makeEvidenceRecord(0, 0, 0, "paper:b")]);
    expect(() => validateEvidenceStreamClosure(paperB, [makeEvidenceRecord(1, 1, 0, "paper:a")]))
      .toThrow(/paper key.*order/i);
  });
});

function lexicalBucket(namespace: "base" | "expanded" | "alias", term: string): number {
  return lexicalTermBucket(namespace, term);
}
function bucketMask(...buckets: number[]): string {
  const bytes = new Uint8Array(32); for (const bucket of buckets) bytes[bucket >>> 3]! |= 1 << (bucket & 7);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
function postingsInput() {
  return {
    postingOrdinal: 0, chunkStart: 0,
    chunks: [{ paperOrdinal: 0, chunkIndex: 0, baseLength: 2, expandedLength: 3, compactText: "alpha中文" }],
    occurrences: [
      { chunkOrdinal: 0, namespace: "alias" as const, term: "a", tf: 1 },
      { chunkOrdinal: 0, namespace: "base" as const, term: "alpha", tf: 2 },
      { chunkOrdinal: 0, namespace: "expanded" as const, term: "中", tf: 1 },
    ], termCatalog: [0, 1, 2],
  };
}
function dictionaryInput() {
  const entries = [
    { postingOrdinal: 0, namespace: "alias" as const, term: "a", chunkDf: 1, totalTf: 1 },
    { postingOrdinal: 0, namespace: "base" as const, term: "alpha", chunkDf: 1, totalTf: 2 },
  ];
  const queryCatalog = [0, 1].sort((a, b) => lexicalBucket(entries[a]!.namespace, entries[a]!.term) - lexicalBucket(entries[b]!.namespace, entries[b]!.term));
  return { dictionaryOrdinal: 0, postingStart: 0, postingCount: 1, entries, queryCatalog,
    bucketMask: bucketMask(...entries.map((entry) => lexicalBucket(entry.namespace, entry.term))) };
}

describe("generation linear lexical blocks", () => {
  it("strictly round-trips postings and dictionary blocks", () => {
    const postings = postingsInput(); expect(decodeLexicalPostingsBlock(encodeLexicalPostingsBlock(postings))).toEqual(postings);
    const dictionary = dictionaryInput(); expect(decodeLexicalDictionaryBlock(encodeLexicalDictionaryBlock(dictionary))).toEqual(dictionary);
  });
  it("rejects catalog duplicate, missing, out-of-range, and noncanonical order", () => {
    for (const termCatalog of [[0, 0, 2], [0, 1], [0, 1, 9], [1, 0, 2]])
      expect(() => encodeLexicalPostingsBlock({ ...postingsInput(), termCatalog })).toThrow(/termCatalog|index|order|cover/i);
    const dictionary = dictionaryInput();
    for (const queryCatalog of [[0, 0], [0], [0, 9], [...dictionary.queryCatalog].reverse()])
      expect(() => encodeLexicalDictionaryBlock({ ...dictionary, queryCatalog })).toThrow(/queryCatalog|index|order|cover/i);
  });
  it("rejects occurrence authority order/duplicates/range/tf/namespace and invalid chunk records", () => {
    const valid = postingsInput();
    expect(() => encodeLexicalPostingsBlock({ ...valid, occurrences: [...valid.occurrences].reverse() })).toThrow(/authority/i);
    expect(() => encodeLexicalPostingsBlock({ ...valid, occurrences: [valid.occurrences[0]!, valid.occurrences[0]!], termCatalog: [0, 1] })).toThrow(/duplicate|authority/i);
    for (const occurrence of [
      { ...valid.occurrences[0]!, chunkOrdinal: 1 }, { ...valid.occurrences[0]!, tf: 0 },
      { ...valid.occurrences[0]!, tf: 2 }, { ...valid.occurrences[0]!, namespace: "future" as any },
    ]) expect(() => encodeLexicalPostingsBlock({ ...valid, occurrences: [occurrence], termCatalog: [0] })).toThrow(/ordinal|tf|namespace|alias/i);
    expect(() => encodeLexicalPostingsBlock({ ...valid, chunks: [{ ...valid.chunks[0]!, compactText: "A-b" }] })).toThrow(/compact/i);
    expect(() => encodeLexicalPostingsBlock({ ...valid, chunks: [{ ...valid.chunks[0]!, expandedLength: 1 }] })).toThrow(/expandedLength/i);
    expect(() => encodeLexicalPostingsBlock({ ...valid, chunks: [...valid.chunks, { ...valid.chunks[0]!, chunkIndex: 2 }] })).toThrow(/continuous/i);
  });
  it("enforces 65536 occurrences, strict JSON fields, exact object cap and block envelope", () => {
    const valid = encodeLexicalPostingsBlock(postingsInput());
    const emptyCompact = encodeLexicalPostingsBlock({
      ...postingsInput(),
      chunks: [{ ...postingsInput().chunks[0]!, compactText: "" }],
    });
    const exactCap = encodeLexicalPostingsBlock({
      ...postingsInput(),
      chunks: [{ ...postingsInput().chunks[0]!, compactText: "x".repeat(MAX_BINARY_OBJECT_BYTES - emptyCompact.byteLength) }],
    });
    expect(exactCap.byteLength).toBe(MAX_BINARY_OBJECT_BYTES);
    expect(decodeLexicalPostingsBlock(exactCap).chunks[0]!.compactText.length)
      .toBe(MAX_BINARY_OBJECT_BYTES - emptyCompact.byteLength);
    expect(() => encodeLexicalPostingsBlock({
      ...postingsInput(),
      chunks: [{ ...postingsInput().chunks[0]!, compactText: `${exactCap.length}x${"x".repeat(MAX_BINARY_OBJECT_BYTES - emptyCompact.byteLength)}` }],
    })).toThrow(/byte limit/i);
    expect(() => decodeLexicalDictionaryBlock(valid)).toThrow(/kind/i);
    expect(() => decodeLexicalPostingsBlock(valid.subarray(0, valid.length - 1))).toThrow(/truncated|length/i);
    const trailing = new Uint8Array(valid.length + 1); trailing.set(valid); expect(() => decodeLexicalPostingsBlock(trailing)).toThrow(/trailing/i);
    const corrupt = valid.slice(); corrupt[corrupt.length - 1]! ^= 1; expect(() => decodeLexicalPostingsBlock(corrupt)).toThrow(/checksum/i);
    expect(() => decodeLexicalPostingsBlock(rewriteBlockHeader(valid, (view) => view.setUint16(6, 3, true)))).toThrow(/schema/i);
    expect(() => decodeLexicalPostingsBlock(rewriteBlockHeader(valid, (view) => view.setUint16(6, 5, true)))).toThrow(/schema/i);
    expect(() => encodeLexicalPostingsBlock({ ...postingsInput(), chunks: [{ ...postingsInput().chunks[0]!, compactText: "x".repeat(MAX_BINARY_OBJECT_BYTES) }] })).toThrow(/byte limit/i);
    expect(() => decodeLexicalPostingsBlock(new Uint8Array(MAX_BINARY_OBJECT_BYTES + 1))).toThrow(/byte limit/i);
    expect(() => encodeLexicalPostingsBlock({
      ...postingsInput(),
      occurrences: Array.from({ length: 65_537 }, () => postingsInput().occurrences[0]!),
      termCatalog: [],
    })).toThrow(/occurrence count/i);
  });
  it("validates dictionary authority, query buckets, mask, and posting ranges", () => {
    const valid = dictionaryInput();
    expect(() => encodeLexicalDictionaryBlock({ ...valid, entries: [...valid.entries].reverse() })).toThrow(/authority/i);
    expect(() => encodeLexicalDictionaryBlock({ ...valid, entries: [{ ...valid.entries[0]!, chunkDf: 0 }], queryCatalog: [0], bucketMask: bucketMask(lexicalBucket("alias", "a")) })).toThrow(/chunkDf/i);
    expect(() => encodeLexicalDictionaryBlock({ ...valid, bucketMask: "0".repeat(64) })).toThrow(/bucketMask/i);
    expect(() => encodeLexicalDictionaryBlock({ ...valid, postingStart: 1 })).toThrow(/postingOrdinal/i);
    const minimal = valid.entries[0]!;
    expect(() => encodeLexicalDictionaryBlock({
      ...valid,
      entries: Array.from({ length: 65_537 }, () => minimal),
      queryCatalog: [],
    })).toThrow(/entry count/i);
    expect(() => encodeLexicalDictionaryBlock({
      ...valid,
      entries: [],
      queryCatalog: Array.from({ length: 65_537 }, () => 0),
      bucketMask: "0".repeat(64),
    })).toThrow(/queryCatalog count/i);
  });
});

describe("generation descriptor codec and paths", () => {
  it("round-trips none, empty, tokenless bm25, and routed lexical descriptors", () => {
    const dense = makeDescriptor(); expect(decodeGenerationDescriptor(encodeGenerationDescriptor(dense))).toEqual(dense);
    const empty = makeDescriptor({ corpusMean: [0, 0, 0], corpusStats: { indexedPaperCount: 0, chunkCount: 0, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 }, objects: [] });
    expect(() => encodeGenerationDescriptor(empty)).not.toThrow();
    expect(() => encodeGenerationDescriptor({ ...empty, lexicalCapability: "bm25-v1" })).toThrow(/empty.*none/i);
    expect(() => encodeGenerationDescriptor({ ...dense, lexicalCapability: "bm25-v1" })).not.toThrow();
    const posting = { kind: "lexical-postings" as const, path: "objects/p.bin", byteLength: 128, recordStart: 0, recordCount: 2, checksum: `sha256:${"1".repeat(64)}` };
    const dictionary = { kind: "lexical-dictionary" as const, path: "objects/d.bin", byteLength: 128, recordStart: 0, recordCount: 1, checksum: `sha256:${"2".repeat(64)}` };
    const routing = Array.from({ length: 256 }, () => [] as number[]); routing[7] = [0];
    const bm25 = makeDescriptor({ lexicalCapability: "bm25-v1", lexicalRouting: routing, objects: [...makeDescriptor().objects, posting, dictionary] });
    expect(() => encodeGenerationDescriptor(bm25)).not.toThrow();
    expect(() => encodeGenerationDescriptor({ ...bm25, lexicalRouting: Array.from({ length: 256 }, () => [] as number[]) })).toThrow(/route/i);
  });
  it("validates object canonical order and kind-specific coverage", () => {
    const refs = makeDescriptor().objects;
    expect(() => encodeGenerationDescriptor(makeDescriptor({ objects: [...refs].reverse() }))).toThrow(/order/i);
    expect(() => encodeGenerationDescriptor(makeDescriptor({ objects: refs.map((ref) => ref.kind === "vector" ? { ...ref, recordStart: 1 } : ref) }))).toThrow(/coverage|continuous/i);
    expect(() => encodeGenerationDescriptor(makeDescriptor({ objects: refs.map((ref) => ref.kind === "paper-metadata" ? { ...ref, recordCount: 2 } : ref) }))).toThrow(/metadata|paper/i);
    expect(() => encodeGenerationDescriptor(makeDescriptor({ objects: [...refs, { kind: "directory" as any, path: "objects/x.bin", byteLength: 128, recordStart: 0, recordCount: 1, checksum: OBJECT_CHECKSUM }] }))).toThrow(/kind/i);
  });
  it("rejects old/future/unknown descriptor data and derives bounded paths", () => {
    const raw = JSON.parse(encodeGenerationDescriptor(makeDescriptor()));
    // Schema 4 stored routing as object paths; it is refused rather than
    // misread, so a store holding one rebuilds instead of reusing it.
    for (const schemaVersion of [3, 4, 6]) expect(() => decodeGenerationDescriptor(JSON.stringify({ ...raw, schemaVersion }))).toThrow(/schema/i);
    expect(() => decodeGenerationDescriptor(JSON.stringify({ ...raw, extra: true }))).toThrow(/unknown/i);
    const normalizePath = (path: string) => path.replaceAll("//", "/");
    expect(deriveFullTextGenerationPaths({ normalizePath }, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION, "gen-20260817-a1").descriptorPath).toContain("/descriptor.json");
    expect(() => deriveFullTextGenerationPaths({ normalizePath }, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION, "../escape")).toThrow(/generationId/i);
  });
});

describe("lexical bucket derivation is shareable", () => {
  const entries = Array.from({ length: 200 }, (_, index) => ({
    postingOrdinal: index % 3,
    namespace: (["base", "expanded", "alias"] as const)[index % 3]!,
    term: `term${index}`,
    chunkDf: 1,
    totalTf: 1,
  }));

  it("precomputed buckets produce the same query catalog as deriving them inline", () => {
    const buckets = lexicalTermBuckets(entries);
    expect(lexicalQueryCatalog(entries, buckets)).toEqual(lexicalQueryCatalog(entries));
  });

  it("precomputed buckets produce the same bucket mask", () => {
    const buckets = lexicalTermBuckets(entries);
    expect(lexicalBucketMask(entries, buckets)).toEqual(lexicalBucketMask(entries));
  });

  it("lexicalTermBuckets agrees with per-entry derivation", () => {
    expect(lexicalTermBuckets(entries)).toEqual(
      entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term)),
    );
  });

  it("a repeated derivation returns the same bucket", () => {
    expect(lexicalTermBucket("base", "recurring")).toBe(lexicalTermBucket("base", "recurring"));
    expect(lexicalTermBucket("alias", "recurring")).toBe(lexicalTermBucket("alias", "recurring"));
  });

  it("the same term in different namespaces can land in different buckets", () => {
    const derived = new Set((["base", "expanded", "alias"] as const).map((ns) => lexicalTermBucket(ns, "shared")));
    expect(derived.size).toBeGreaterThan(0);
  });
});

describe("lexical routing capacity is independent of the object budget", () => {
  /**
   * A consistent bm25 descriptor with `count` dictionary blocks: one postings
   * object per block, and vector/evidence coverage matching the chunk count the
   * postings declare.
   */
  function withDictionaries(count: number, routing: number[][]): GenerationDescriptor {
    const object = (kind: GenerationObjectReference["kind"], name: string, recordStart: number, recordCount: number, seed: string) => ({
      kind, path: `objects/${name}`, byteLength: 128, recordStart, recordCount,
      checksum: `sha256:${seed.repeat(64).slice(0, 64)}`,
    });
    const postings = Array.from({ length: count }, (_, index) =>
      object("lexical-postings", `postings-${String(index).padStart(10, "0")}.bin`, index, 1, "f"));
    const dictionaries = Array.from({ length: count }, (_, index) =>
      object("lexical-dictionary", `dictionary-${String(index).padStart(10, "0")}.bin`, index, 1, "e"));
    return {
      ...makeDescriptor(),
      lexicalCapability: "bm25-v1",
      corpusStats: {
        indexedPaperCount: 1, chunkCount: count,
        totalLexicalTokenCount: count, avgdl: 1,
        totalLexicalTokenCountWithHanSingles: count, avgdlWithHanSingles: 1,
      },
      lexicalRouting: routing,
      objects: [
        object("vector", "000001.vectors.bin", 0, count, "a"),
        object("evidence", "000001.evidence.bin", 0, count, "d"),
        object("paper-metadata", "000001.metadata.bin", 0, 1, "e"),
        ...postings,
        ...dictionaries,
      ],
    };
  }

  it("accepts a routing table larger than the object budget", () => {
    // Every dictionary block touches all 256 buckets in real prose, so a corpus
    // needing more than 16 blocks exceeds MAX_GENERATION_OBJECTS in references
    // while using only a handful of objects. That must not be an error.
    const blocks = 40;
    const routing = Array.from({ length: 256 }, () => Array.from({ length: blocks }, (_, index) => index));
    const total = routing.reduce((sum, bucket) => sum + bucket.length, 0);
    expect(total).toBeGreaterThan(MAX_GENERATION_OBJECTS);
    expect(() => encodeGenerationDescriptor(withDictionaries(blocks, routing))).not.toThrow();
  });

  it("round-trips routing entries as dictionary ordinals", () => {
    const routing = Array.from({ length: 256 }, (_, bucket) => (bucket % 2 === 0 ? [0, 1] : [1]));
    const descriptor = withDictionaries(2, routing);
    expect(decodeGenerationDescriptor(encodeGenerationDescriptor(descriptor))).toEqual(descriptor);
  });

  it("refuses a routing entry that names no dictionary object", () => {
    const routing = Array.from({ length: 256 }, () => [] as number[]);
    routing[0] = [7];
    expect(() => encodeGenerationDescriptor(withDictionaries(2, routing))).toThrow(/routing/i);
  });

  it("refuses a routing entry that is not a dictionary ordinal", () => {
    const routing = Array.from({ length: 256 }, () => [] as number[]);
    (routing[0] as unknown[])[0] = "objects/dictionary-0000000000.bin";
    expect(() => encodeGenerationDescriptor(withDictionaries(2, routing as number[][]))).toThrow(/routing/i);
  });

  it("still bounds the routing table by its own limit", () => {
    const blocks = 8;
    const perBucket = Math.ceil(MAX_GENERATION_ROUTE_REFS / 256) + 1;
    const routing = Array.from({ length: 256 }, () => Array.from({ length: perBucket }, () => 0));
    expect(() => encodeGenerationDescriptor(withDictionaries(blocks, routing))).toThrow(/routing/i);
  });
});

describe("term ordering does not require encoding", () => {
  const encoder = new TextEncoder();
  const byUtf8Bytes = (left: string, right: string) => {
    const a = encoder.encode(left);
    const b = encoder.encode(right);
    for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
      if (a[index] !== b[index]) return Math.sign(a[index]! - b[index]!);
    }
    return Math.sign(a.length - b.length);
  };

  const terms = [
    "a", "ab", "abc", "b", "z", "za",
    "é", "e", "ü", "ss",
    "中", "中文", "文", "国", "漢字",
    "α", "β", "ω",
    "0", "1", "9", "10",
    "𝄞", "😀", "a😀", "😀a",
    "ﬁ", "ｆｕｌｌ",
  ];

  it("code point order matches UTF-8 byte order for every pair", () => {
    for (const left of terms) {
      for (const right of terms) {
        expect(Math.sign(compareTermCodePoints(left, right))).toBe(byUtf8Bytes(left, right));
      }
    }
  });

  it("orders supplementary-plane terms the way their bytes do", () => {
    expect(Math.sign(compareTermCodePoints("😀", "中"))).toBe(byUtf8Bytes("😀", "中"));
    expect(Math.sign(compareTermCodePoints("中", "😀"))).toBe(byUtf8Bytes("中", "😀"));
  });

  it("treats equal terms as equal", () => {
    for (const term of terms) expect(compareTermCodePoints(term, term)).toBe(0);
  });
});
