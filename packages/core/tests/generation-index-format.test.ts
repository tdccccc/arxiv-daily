import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { createEvidenceChunkId, type EvidenceChunk } from "../src/library/fulltext/evidence-chunk";
import {
  BINARY_BLOCK_HEADER_BYTES,
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  MAX_BINARY_OBJECT_BYTES,
  MAX_GENERATION_DIMENSION,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeGenerationDescriptor,
  decodeVectorBlock,
  deriveFullTextGenerationPaths,
  encodeEvidenceBlock,
  encodeGenerationDescriptor,
  encodeVectorBlock,
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
    corpusStats: { indexedPaperCount: 1, chunkCount: 2 },
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
      vectors: new Float32Array([1.5, -2.25, 3.125, 4, 5, 6]),
    });
    const decoded = decodeVectorBlock(encoded);
    expect(decoded.rowStart).toBe(9);
    expect(decoded.rowCount).toBe(2);
    expect(decoded.dimension).toBe(3);
    expect(Array.from(decoded.vectors)).toEqual([1.5, -2.25, 3.125, 4, 5, 6]);
    expect(encoded.byteLength).toBe(BINARY_BLOCK_HEADER_BYTES + 8 + 6 * 4);
    expect(blockObjectChecksum(encoded)).toMatch(/^sha256:[a-f0-9]{64}$/);
  });

  it("accepts an object exactly at the byte cap and rejects one byte over it", () => {
    const floatCount = (MAX_BINARY_OBJECT_BYTES - BINARY_BLOCK_HEADER_BYTES - 8) / 4;
    const encoded = encodeVectorBlock({
      rowStart: 0,
      dimension: 1,
      vectors: new Float32Array(floatCount),
    });
    expect(encoded.byteLength).toBe(MAX_BINARY_OBJECT_BYTES);
    expect(decodeVectorBlock(encoded).rowCount).toBe(floatCount);
    expect(() => decodeVectorBlock(new Uint8Array(MAX_BINARY_OBJECT_BYTES + 1))).toThrow(/byte limit/i);
  });

  it("rejects truncated, trailing, wrong-kind, future-version, checksum, count, and dimension tampering", () => {
    const valid = encodeVectorBlock({ rowStart: 2, dimension: 2, vectors: new Float32Array([1, 2, 3, 4]) });
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

  it("rejects magic and payload-length tampering before payload loops", () => {
    const valid = encodeVectorBlock({ rowStart: 0, dimension: 1, vectors: new Float32Array([1]) });
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

describe("generation descriptor codec and paths", () => {
  it("round-trips canonical strict JSON with index-only derivation and sorted refs", () => {
    const descriptor = makeDescriptor();
    const first = encodeGenerationDescriptor(descriptor);
    const second = encodeGenerationDescriptor(decodeGenerationDescriptor(first));
    expect(second).toBe(first);
    expect(decodeGenerationDescriptor(first)).toEqual(descriptor);
  });

  it("accepts multiple contiguous blocks sorted by kind, recordStart, then path", () => {
    const descriptor = makeDescriptor({
      corpusStats: { indexedPaperCount: 2, chunkCount: 4 },
      objects: [
        { kind: "vector", path: "objects/a.bin", byteLength: 128, recordStart: 0, recordCount: 2, checksum: OBJECT_CHECKSUM },
        { kind: "vector", path: "objects/b.bin", byteLength: 128, recordStart: 2, recordCount: 2, checksum: `sha256:${"d".repeat(64)}` },
        { kind: "evidence", path: "objects/c.bin", byteLength: 128, recordStart: 0, recordCount: 2, checksum: `sha256:${"e".repeat(64)}` },
        { kind: "evidence", path: "objects/d.bin", byteLength: 128, recordStart: 2, recordCount: 2, checksum: `sha256:${"f".repeat(64)}` },
      ],
    });
    expect(decodeGenerationDescriptor(encodeGenerationDescriptor(descriptor))).toEqual(descriptor);
  });

  it("rejects incomplete, unsorted, gapped, overlapping, missing-kind, and zero-count refs", () => {
    const cases: GenerationDescriptor["objects"][] = [
      makeDescriptor().objects.slice(0, 1),
      [...makeDescriptor().objects].reverse(),
      makeDescriptor().objects.map((ref) => ref.kind === "vector" ? { ...ref, recordStart: 1 } : ref),
      [
        { ...makeDescriptor().objects[0]!, recordCount: 1 },
        { ...makeDescriptor().objects[0]!, path: "objects/second.bin", recordStart: 0, recordCount: 2 },
        makeDescriptor().objects[1]!,
      ],
      makeDescriptor().objects.map((ref) => ref.kind === "evidence" ? { ...ref, recordCount: 1 } : ref),
      makeDescriptor().objects.map((ref) => ({ ...ref, recordCount: 0 })),
    ];
    for (const objects of cases) {
      expect(() => encodeGenerationDescriptor(makeDescriptor({ objects }))).toThrow(/object|coverage|continuous|order|record/i);
    }
  });

  it("requires both block kinds for non-empty corpora and none for empty corpora with zero mean", () => {
    expect(() => encodeGenerationDescriptor(makeDescriptor({ objects: makeDescriptor().objects.slice(0, 1) })))
      .toThrow(/object|kind|coverage/i);
    const empty = makeDescriptor({
      corpusMean: [0, 0, 0],
      corpusStats: { indexedPaperCount: 0, chunkCount: 0 },
      objects: [],
    });
    expect(decodeGenerationDescriptor(encodeGenerationDescriptor(empty))).toEqual(empty);
    expect(() => encodeGenerationDescriptor({ ...empty, corpusMean: [0, 0.01, 0] })).toThrow(/corpusMean/i);
    expect(() => encodeGenerationDescriptor({ ...empty, objects: makeDescriptor().objects })).toThrow(/empty|object/i);
    expect(() => encodeGenerationDescriptor({
      ...empty,
      corpusStats: { indexedPaperCount: 1, chunkCount: 0 },
    })).toThrow(/indexed paper/i);
    expect(() => encodeGenerationDescriptor(makeDescriptor({
      corpusStats: { indexedPaperCount: 0, chunkCount: 2 },
    }))).toThrow(/indexed paper/i);
    expect(() => encodeGenerationDescriptor(makeDescriptor({
      corpusStats: { indexedPaperCount: 3, chunkCount: 2 },
    }))).toThrow(/indexed paper/i);
  });

  it("rejects lexical postings refs until their codec advances the format", () => {
    const lexical = {
      kind: "lexical-postings" as const,
      path: "objects/postings.bin",
      byteLength: 128,
      recordStart: 0,
      recordCount: 1,
      checksum: OBJECT_CHECKSUM,
    };
    expect(() => encodeGenerationDescriptor(makeDescriptor({ objects: [...makeDescriptor().objects, lexical] as any })))
      .toThrow(/lexical|kind/i);
  });

  it("rejects future versions, unknown or missing index derivation, unsafe counts, dimensions, strings, refs, and trailing JSON", () => {
    const valid = JSON.parse(encodeGenerationDescriptor(makeDescriptor()));
    for (const mutation of [
      (value: any) => { value.formatVersion += 1; },
      (value: any) => { value.schemaVersion += 1; },
      (value: any) => { value.extra = true; },
      (value: any) => { delete value.indexDerivation.builderVersion; },
      (value: any) => { value.indexDerivation.tokenizerVersion = Number.MAX_SAFE_INTEGER; },
      (value: any) => { value.corpusStats.chunkCount = Number.MAX_SAFE_INTEGER; },
      (value: any) => { value.dimension = MAX_GENERATION_DIMENSION + 1; },
      (value: any) => { value.modelId = "x".repeat(300); },
      (value: any) => { value.objects[0].path = "objects/../escape.bin"; },
      (value: any) => { value.objects[0].byteLength = MAX_BINARY_OBJECT_BYTES + 1; },
      (value: any) => { value.objects[0].kind = "unknown"; },
      (value: any) => { delete value.objects[0].recordStart; },
    ]) {
      const changed = structuredClone(valid);
      mutation(changed);
      expect(() => decodeGenerationDescriptor(JSON.stringify(changed))).toThrow();
    }
    expect(() => decodeGenerationDescriptor(`${JSON.stringify(valid)} trailing`)).toThrow(/JSON/i);
  });

  it("rejects generation IDs that could escape or create unbounded path names", () => {
    for (const generationId of ["../escape", "Gen-A", "a/b", "a".repeat(65), "-leading"] as const) {
      expect(() => encodeGenerationDescriptor(makeDescriptor({ generationId }))).toThrow(/generationId/i);
    }
  });

  it("derives paths through existing scope and identification fingerprint sharding", () => {
    const normalizePath = (path: string) => path.replaceAll("//", "/");
    const paths = deriveFullTextGenerationPaths(
      { normalizePath },
      DEFAULT_SETTINGS.output,
      SCOPE,
      IDENTIFICATION,
      "gen-20260817-a1",
    );
    const base = `arxiv-daily/.index/personal-library-knowledge-base/${"a".repeat(64)}/${"b".repeat(64)}`;
    expect(paths).toEqual({
      directory: `${base}/generations/gen-20260817-a1`,
      descriptorPath: `${base}/generations/gen-20260817-a1/descriptor.json`,
      objectsDirectory: `${base}/generations/gen-20260817-a1/objects`,
    });
    expect(() => deriveFullTextGenerationPaths(
      { normalizePath }, DEFAULT_SETTINGS.output, SCOPE, IDENTIFICATION, "../escape",
    )).toThrow(/generationId/i);
  });
});
