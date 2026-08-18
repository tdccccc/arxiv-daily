import { describe, expect, it } from "vitest";
import { createEvidenceChunkId, type EvidenceChunk } from "../src/library/fulltext/evidence-chunk";
import { FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION, type FullTextPaperDocument } from "../src/library/fulltext/knowledge-base";
import { searchGenerationDense, searchKnowledgeBase } from "../src/library/fulltext/retrieval";
import {
  GENERATION_DESCRIPTOR_FORMAT_VERSION,
  GENERATION_DESCRIPTOR_SCHEMA_VERSION,
  blockObjectChecksum,
  decodeEvidenceBlock,
  decodeVectorBlock,
  encodeEvidenceBlock,
  encodeVectorBlock,
  type EvidenceBlockRecord,
  type GenerationDescriptor,
  type GenerationObjectReference,
} from "../src/library/fulltext/generation-index-format";
import { FullTextGenerationIndexStoreError, type OpenedFullTextGeneration } from "../src/library/fulltext/generation-index-store";

const SCOPE = `sha256:${"a".repeat(64)}`;
const IDENTIFICATION = `sha256:${"b".repeat(64)}`;

function deferred<T = void>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((settle) => { resolve = settle; });
  return { promise, resolve };
}

function chunk(index: number, text: string): EvidenceChunk {
  const identity = {
    text,
    headings: ["Methods"],
    locator: { pageStart: index + 1 },
    derivation: { parser: { id: "fixture", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 },
  };
  return { id: createEvidenceChunkId(identity), index, page: index + 1, ...identity };
}

function paper(paperKey: string, rows: readonly number[][]): FullTextPaperDocument {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: "model-a",
    dimension: 2,
    textHash: `sha256:${"c".repeat(64)}`,
    filePaths: [`${paperKey}.pdf`],
    observationFingerprints: [`sha256:${"d".repeat(64)}`],
    chunks: rows.map((_, index) => chunk(index, `${paperKey} chunk ${index}`)),
    vectors: new Float32Array(rows.flat()),
    updatedAt: "2026-08-17T00:00:00.000Z",
  };
}

async function builtGeneration(papers: readonly FullTextPaperDocument[], maxRowsPerBlock = 2) {
  const nonEmpty = papers.filter((entry) => entry.chunks.length > 0);
  const rows: Array<{ paperOrdinal: number; paperKey: string; chunk: EvidenceChunk; vector: number[] }> = [];
  nonEmpty.forEach((entry, paperOrdinal) => entry.chunks.forEach((entryChunk, chunkIndex) => rows.push({
    paperOrdinal,
    paperKey: entry.paperKey,
    chunk: entryChunk as EvidenceChunk,
    vector: Array.from(entry.vectors.subarray(chunkIndex * entry.dimension, (chunkIndex + 1) * entry.dimension)),
  })));
  const writes = new Map<string, Uint8Array>();
  const vectorRefs: GenerationObjectReference[] = [];
  const evidenceRefs: GenerationObjectReference[] = [];
  for (let rowStart = 0, blockIndex = 0; rowStart < rows.length; rowStart += maxRowsPerBlock, blockIndex += 1) {
    const blockRows = rows.slice(rowStart, rowStart + maxRowsPerBlock);
    const suffix = String(blockIndex).padStart(6, "0");
    const vectorPath = `objects/${suffix}.vectors.bin`;
    const evidencePath = `objects/${suffix}.evidence.bin`;
    const vectorBytes = encodeVectorBlock({
      rowStart,
      dimension: 2,
      paperOrdinals: new Uint32Array(blockRows.map((entry) => entry.paperOrdinal)),
      vectors: new Float32Array(blockRows.flatMap((entry) => entry.vector)),
    });
    const records: EvidenceBlockRecord[] = blockRows.map((entry, offset) => ({
      paperIndex: entry.paperOrdinal,
      paperKey: entry.paperKey,
      vectorRow: rowStart + offset,
      chunk: entry.chunk,
    }));
    const evidenceBytes = encodeEvidenceBlock({ rowStart, records });
    writes.set(vectorPath, vectorBytes);
    writes.set(evidencePath, evidenceBytes);
    vectorRefs.push({
      kind: "vector", path: vectorPath, byteLength: vectorBytes.byteLength,
      recordStart: rowStart, recordCount: blockRows.length, checksum: blockObjectChecksum(vectorBytes),
    });
    evidenceRefs.push({
      kind: "evidence", path: evidencePath, byteLength: evidenceBytes.byteLength,
      recordStart: rowStart, recordCount: blockRows.length, checksum: blockObjectChecksum(evidenceBytes),
    });
  }
  const sums = new Float64Array(2);
  rows.forEach((entry) => entry.vector.forEach((value, column) => { sums[column]! += value; }));
  const descriptor: GenerationDescriptor = {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId: "gen-dense-test",
    sourceRevision: 7,
    scopeFingerprint: SCOPE,
    identificationFingerprint: IDENTIFICATION,
    modelId: "model-a",
    dimension: 2,
    corpusMean: Array.from(sums, (sum) => rows.length === 0 ? 0 : sum / rows.length),
    corpusStats: { indexedPaperCount: nonEmpty.length, chunkCount: rows.length, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: 256 }, () => [] as string[]),
    indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 },
    objects: [...vectorRefs, ...evidenceRefs],
  };
  let vectorReads = 0;
  let evidenceReads = 0;
  const opened = {
    descriptor,
    diagnostics: { maxObjectBytes: 0, objectReads: 0 },
    iterateVectorBlocks: async function* () {
      for (const reference of vectorRefs) {
        vectorReads += 1;
        yield { reference, block: decodeVectorBlock(writes.get(reference.path)!) } as any;
      }
    },
    readObject: async (reference: GenerationDescriptor["objects"][number]) => {
      if (reference.kind === "evidence") evidenceReads += 1;
      const bytes = writes.get(reference.path)!;
      return reference.kind === "vector"
        ? { reference, block: decodeVectorBlock(bytes) }
        : { reference, block: decodeEvidenceBlock(bytes) };
    },
  } as unknown as OpenedFullTextGeneration;
  return { descriptor, writes, opened, reads: () => ({ vectorReads, evidenceReads }) };
}

describe("dense generation reader", () => {
  it("matches the P3 oracle bit-for-bit across paper/block boundaries and materializes only selected evidence", async () => {
    const papers = [
      paper("paper:a", [[1, 0], [0.5, 0.5], [0, 1]]),
      paper("paper:b", [[-1, 0], [0.25, 0.75]]),
      paper("paper:c", [[0.75, 0.25]]),
    ];
    const query = new Float32Array([0.8, 0.2]);
    const built = await builtGeneration(papers, 2);
    const stats = { vectorReads: 0, evidenceReads: 0, peakCandidates: 0, peakHits: 0 };
    const actual = await searchGenerationDense({ generation: built.opened, queryVector: query, limit: 2, maxHitsPerPaper: 2, stats });
    const expected = searchKnowledgeBase({ papers, queryVector: query, limit: 2, maxHitsPerPaper: 2 });
    expect(actual).toEqual(expected);
    for (let paperIndex = 0; paperIndex < actual.length; paperIndex += 1) {
      expect(Object.is(actual[paperIndex]!.score, expected[paperIndex]!.score)).toBe(true);
      for (let hit = 0; hit < actual[paperIndex]!.hits.length; hit += 1) {
        expect(Object.is(actual[paperIndex]!.hits[hit]!.score, expected[paperIndex]!.hits[hit]!.score)).toBe(true);
      }
    }
    expect(stats.vectorReads).toBe(3);
    expect(stats.evidenceReads).toBeLessThan(3);
    expect(stats.peakCandidates).toBeLessThanOrEqual(2);
    expect(stats.peakHits).toBeLessThanOrEqual((2 + 1) * 2);
  });

  it("applies ordinal title/token scores, hard caps options, and skips all reads for empty/zero queries", async () => {
    const papers = [paper("paper:a", [[1, 0]]), paper("paper:b", [[0, 1]])];
    const built = await builtGeneration(papers, 1);
    const lifted = await searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      centerCorpus: false,
      titleScoresByPaperOrdinal: [0, 1],
      tokenScoresByPaperOrdinal: [0, 0.5],
      limit: 2,
    });
    expect(lifted.map((entry) => entry.paperKey)).toEqual(["paper:a", "paper:b"]);
    expect(lifted[1]!.score).toBe(1);
    await expect(searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      titleScoresByPaperOrdinal: [0],
    })).rejects.toThrow(/titleScoresByPaperOrdinal.*length/i);
    await expect(searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      tokenScoresByPaperOrdinal: [0, Number.NaN],
    })).rejects.toThrow(/tokenScoresByPaperOrdinal.*finite/i);
    await expect(searchGenerationDense({ generation: built.opened, queryVector: new Float32Array([1, 0]), limit: 1_001 }))
      .rejects.toThrow(/limit/i);
    await expect(searchGenerationDense({ generation: built.opened, queryVector: new Float32Array([1]), limit: 0 }))
      .rejects.toThrow(/dimension/i);
    const before = built.reads();
    expect(await searchGenerationDense({ generation: built.opened, queryVector: new Float32Array([1, 0]), limit: 0 })).toEqual([]);
    expect(built.reads()).toEqual(before);
    const empty = await builtGeneration([], 1);
    expect(await searchGenerationDense({ generation: empty.opened, queryVector: new Float32Array([1, 0]) })).toEqual([]);
    expect(empty.reads()).toEqual({ vectorReads: 0, evidenceReads: 0 });
  });

  it("is field-exact with P3 for zero/mean vectors, title/token-only lifts, and paper/hit ties", async () => {
    const papers = [
      paper("paper:a", [[1, 0], [1, 0]]),
      paper("paper:b", [[1, 0]]),
      paper("paper:c", [[0, 1]]),
    ];
    const built = await builtGeneration(papers, 2);
    const scenarios = [
      { name: "raw zero query", query: new Float32Array([0, 0]), centerCorpus: false },
      { name: "centered zero query", query: new Float32Array([0, 0]), centerCorpus: true },
      { name: "query equals center mean", query: new Float32Array(built.descriptor.corpusMean), centerCorpus: true },
      { name: "chunk equals center mean", query: new Float32Array([1, 0]), centerCorpus: true },
    ] as const;
    for (const scenario of scenarios) {
      const actual = await searchGenerationDense({
        generation: built.opened, queryVector: scenario.query, centerCorpus: scenario.centerCorpus,
        limit: 3, maxHitsPerPaper: 2,
      });
      const expected = searchKnowledgeBase({
        papers, queryVector: scenario.query, centerCorpus: scenario.centerCorpus,
        limit: 3, maxHitsPerPaper: 2,
      });
      expect(actual, scenario.name).toEqual(expected);
      actual.forEach((match, paperIndex) => {
        const oracle = expected[paperIndex]!;
        expect(Object.is(match.score, oracle.score), `${scenario.name} paper score`).toBe(true);
        expect(Object.is(match.rankingScore, oracle.rankingScore), `${scenario.name} ranking score`).toBe(true);
        expect(match.scoreKind).toBe(oracle.scoreKind);
        expect(match.rankingScoreKind).toBe(oracle.rankingScoreKind);
        expect(match.chunkCount).toBe(oracle.chunkCount);
        match.hits.forEach((hit, hitIndex) => {
          const oracleHit = oracle.hits[hitIndex]!;
          expect(hit).toEqual(oracleHit);
          expect(Object.is(hit.score, oracleHit.score), `${scenario.name} hit score`).toBe(true);
        });
      });
    }
    const titleOnly = await searchGenerationDense({
      generation: built.opened, queryVector: new Float32Array([0, 0]), centerCorpus: false,
      titleScoresByPaperOrdinal: [0, 0, 0.75], limit: 3,
    });
    const titleOracle = searchKnowledgeBase({
      papers, queryVector: new Float32Array([0, 0]), centerCorpus: false,
      titleScores: new Map([["paper:c", 0.75]]), limit: 3,
    });
    expect(titleOnly).toEqual(titleOracle);
    const tokenOnly = await searchGenerationDense({
      generation: built.opened, queryVector: new Float32Array([0, 0]), centerCorpus: false,
      tokenScoresByPaperOrdinal: [0, 0.5, 0], limit: 3,
    });
    const tokenOracle = searchKnowledgeBase({
      papers, queryVector: new Float32Array([0, 0]), centerCorpus: false,
      tokenScores: new Map([["paper:b", 0.5]]), limit: 3,
    });
    expect(tokenOnly).toEqual(tokenOracle);
    expect(titleOnly.map((entry) => entry.paperKey)).toEqual(["paper:c", "paper:a", "paper:b"]);
    expect(tokenOnly.map((entry) => entry.paperKey)).toEqual(["paper:b", "paper:a", "paper:c"]);
    expect(titleOnly[1]!.hits.map((hit) => hit.chunkIndex)).toEqual([0, 1]);
  });

  it("yields to a real timer and aborts a multi-block scan", async () => {
    const papers = Array.from({ length: 24 }, (_, index) => paper(
      `paper:${String(index).padStart(3, "0")}`,
      Array.from({ length: 64 }, () => [1, index / 24]),
    ));
    const built = await builtGeneration(papers, 16);
    const controller = new AbortController();
    const pending = searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      limit: 5,
      signal: controller.signal,
    });
    setTimeout(() => controller.abort(), 0);
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });

  it("aborts while a one-row selected evidence read is deferred", async () => {
    const built = await builtGeneration([paper("paper:a", [[1, 0]])], 1);
    const entered = deferred();
    const release = deferred();
    const originalRead = built.opened.readObject.bind(built.opened);
    (built.opened as { readObject: OpenedFullTextGeneration["readObject"] }).readObject = async (reference) => {
      entered.resolve();
      await release.promise;
      return originalRead(reference);
    };
    const controller = new AbortController();
    const pending = searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      centerCorpus: false,
      limit: 1,
      signal: controller.signal,
    });
    await entered.promise;
    controller.abort();
    release.resolve();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });

  it("treats a leading # as data and rejects a changed key across evidence blocks", async () => {
    const built = await builtGeneration([paper("#paper:a", [[1, 0], [0.9, 0.1]])], 1);
    const originalRead = built.opened.readObject.bind(built.opened);
    (built.opened as { readObject: OpenedFullTextGeneration["readObject"] }).readObject = async (reference) => {
      const object = await originalRead(reference);
      if (object.reference.kind !== "evidence") return object;
      const records = object.block.records.map((record) => record.vectorRow === 1
        ? { ...record, paperKey: "#paper:b" }
        : record);
      return { ...object, block: { ...object.block, records } };
    };
    await expect(searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      centerCorpus: false,
      limit: 1,
      maxHitsPerPaper: 2,
    })).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("materializes multiple hits for a legitimate leading-# paperKey", async () => {
    const papers = [paper("#paper:a", [[1, 0], [0.9, 0.1]])];
    const built = await builtGeneration(papers, 1);
    await expect(searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      centerCorpus: false,
      limit: 1,
      maxHitsPerPaper: 2,
    })).resolves.toEqual(searchKnowledgeBase({
      papers,
      queryVector: new Float32Array([1, 0]),
      centerCorpus: false,
      limit: 1,
      maxHitsPerPaper: 2,
    }));
  });

  it.each([2, 1])("rejects inconsistent evidence paperKey for one ordinal with %i rows per block", async (rowsPerBlock) => {
    const built = await builtGeneration([paper("paper:a", [[1, 0], [0.9, 0.1]])], rowsPerBlock);
    const originalRead = built.opened.readObject.bind(built.opened);
    (built.opened as { readObject: OpenedFullTextGeneration["readObject"] }).readObject = async (reference) => {
      const object = await originalRead(reference);
      if (object.reference.kind !== "evidence") return object;
      const records = object.block.records.map((record) => record.vectorRow === 1
        ? { ...record, paperKey: "paper:changed" }
        : record);
      return { ...object, block: { ...object.block, records } };
    };
    await expect(searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 0]),
      centerCorpus: false,
      limit: 1,
      maxHitsPerPaper: 2,
    })).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });

  it("never buffers limit+1 papers or maxHits+1 rows", async () => {
    const papers = Array.from({ length: 12 }, (_, paperIndex) => paper(
      `paper:${String(paperIndex).padStart(2, "0")}`,
      Array.from({ length: 9 }, (_, chunkIndex) => [paperIndex + 1, chunkIndex + 1]),
    ));
    const built = await builtGeneration(papers, 5);
    const stats = { vectorReads: 0, evidenceReads: 0, peakCandidates: 0, peakHits: 0 };
    await searchGenerationDense({
      generation: built.opened,
      queryVector: new Float32Array([1, 1]),
      centerCorpus: false,
      limit: 3,
      maxHitsPerPaper: 2,
      stats,
    });
    expect(stats.peakCandidates).toBe(3);
    expect(stats.peakHits).toBe((3 + 1) * 2);

    const largePapers = Array.from({ length: 1_005 }, (_, paperIndex) => paper(
      `large:${String(paperIndex).padStart(4, "0")}`,
      [[paperIndex + 1, 1], [paperIndex + 1, 2], [paperIndex + 1, 3]],
    ));
    const large = await builtGeneration(largePapers, 64);
    const largeStats = { vectorReads: 0, evidenceReads: 0, peakCandidates: 0, peakHits: 0 };
    const matches = await searchGenerationDense({
      generation: large.opened,
      queryVector: new Float32Array([1, 1]),
      centerCorpus: false,
      limit: 1_000,
      maxHitsPerPaper: 2,
      stats: largeStats,
    });
    expect(matches).toHaveLength(1_000);
    expect(largeStats.peakCandidates).toBe(1_000);
    expect(largeStats.peakHits).toBe((1_000 + 1) * 2);
  });

  it("does not read corrupt unselected evidence and propagates a typed selected-evidence failure", async () => {
    const papers = [paper("paper:a", [[1, 0]]), paper("paper:b", [[0, 1]])];
    const built = await builtGeneration(papers, 1);
    const originalRead = built.opened.readObject.bind(built.opened);
    const evidenceRefs = built.descriptor.objects.filter((entry) => entry.kind === "evidence");
    (built.opened as { readObject: OpenedFullTextGeneration["readObject"] }).readObject = async (reference) => {
      if (reference.path === evidenceRefs[1]!.path) {
        throw new FullTextGenerationIndexStoreError("selected evidence corrupt", "corrupt-or-unreadable");
      }
      return originalRead(reference);
    };
    await expect(searchGenerationDense({
      generation: built.opened, queryVector: new Float32Array([1, 0]), centerCorpus: false, limit: 1, maxHitsPerPaper: 1,
    })).resolves.toMatchObject([{ paperKey: "paper:a" }]);
    expect(built.reads().evidenceReads).toBe(1);
    await expect(searchGenerationDense({
      generation: built.opened, queryVector: new Float32Array([0, 1]), centerCorpus: false, limit: 1, maxHitsPerPaper: 1,
    })).rejects.toMatchObject({ code: "corrupt-or-unreadable" });
  });


});
