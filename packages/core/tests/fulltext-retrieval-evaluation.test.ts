import { describe, expect, it } from "vitest";
import { createEvidenceChunkId } from "../src/library/fulltext/evidence-chunk";
import { searchKnowledgeBaseBm25 } from "../src/library/fulltext/bm25-retrieval";
import { fusePaperRankingsRrf } from "../src/library/fulltext/hybrid-retrieval";
import {
  assertHybridRetrievalGates,
  evaluateRetrieval,
  type RetrievalJudgment,
} from "../src/library/fulltext/retrieval-evaluation";
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
import type { OpenedFullTextGeneration } from "../src/library/fulltext/generation-index-store";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";

const DIMENSION = 6;
const axis = (index: number): Float32Array => {
  const vector = new Float32Array(DIMENSION);
  vector[index] = 1;
  return vector;
};

function paper(paperKey: string, title: string, text: string, vector: Float32Array): FullTextPaperDocument {
  const identity = {
    text,
    headings: [] as string[],
    locator: { pageStart: 1 },
    derivation: { parser: { id: "evaluation", version: "1" }, chunkerVersion: 2, embeddingInputVersion: 1 },
  };
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    title,
    modelId: "fixed-evaluation-model",
    dimension: DIMENSION,
    textHash: `sha256:${"1".repeat(64)}`,
    filePaths: [`${paperKey}.pdf`],
    observationFingerprints: [`sha256:${"2".repeat(64)}`],
    chunks: [{ id: createEvidenceChunkId(identity), index: 0, page: 1, ...identity }],
    vectors: new Float32Array(vector),
    updatedAt: "2026-08-17T00:00:00.000Z",
  };
}

// Fixed independently authored text/vector corpus. Judgments below describe
// query intent and were written before executing any retrieval branch.
const CORPUS = [
  paper("attention", "Graph Attention Networks", "masked self-attention over graph neighborhoods", axis(0)),
  paper("panstarrs", "The Pan-STARRS1 Surveys", "Pan-STARRS photometric survey data products", axis(1)),
  paper("chinese", "中文科研文献检索", "中文检索与证据定位方法", axis(2)),
  paper("semantic", "Invariant Representation Alignment", "latent representation alignment across domains", axis(3)),
  paper("sky", "Deep Sky Calibration", "wide field photometric calibration for deep galaxy surveys", axis(4)),
  paper("robust", "Robust Estimation", "bounded influence estimation under adversarial contamination", axis(5)),
  paper("hard-negative", "Contamination Keywords", "robust estimation under contamination mentioned only as background", axis(0)),
  paper("survey-negative", "Survey Instrument Status", "survey calibration hardware status report", axis(1)),
] as const;

interface QueryFixture {
  id: string;
  category: string;
  queryText: string;
  lexicalQueryText?: string;
  queryVector: Float32Array;
}

const QUERIES: readonly QueryFixture[] = [
  { id: "exact", category: "exact-title", queryText: "Graph Attention Networks", queryVector: axis(1) },
  { id: "alias", category: "compact-alias", queryText: "panstarrs", queryVector: axis(0) },
  { id: "cjk", category: "cjk-keyword", queryText: "中文检索", queryVector: axis(1) },
  { id: "semantic", category: "semantic-rewrite", queryText: "meaning-preserving domain features", queryVector: axis(3) },
  {
    id: "long",
    category: "title-abstract",
    queryText: "Deep Sky Calibration\n\nA semantic description of galaxy photometry across a wide field.",
    lexicalQueryText: "Deep Sky Calibration",
    queryVector: axis(4),
  },
  { id: "hard", category: "hard-negative", queryText: "reliable statistics with outliers", queryVector: axis(5) },
];

const JUDGMENTS: readonly RetrievalJudgment[] = [
  { queryId: "exact", category: "exact-title", grades: { attention: 3 } },
  { queryId: "alias", category: "compact-alias", grades: { panstarrs: 3 } },
  { queryId: "cjk", category: "cjk-keyword", grades: { chinese: 3 } },
  { queryId: "semantic", category: "semantic-rewrite", grades: { semantic: 3 } },
  { queryId: "long", category: "title-abstract", grades: { sky: 3 } },
  { queryId: "hard", category: "hard-negative", grades: { robust: 3, "hard-negative": 0 } },
];

function fixedGeneration(papers: readonly FullTextPaperDocument[], rowsPerBlock = 2): OpenedFullTextGeneration {
  const canonical = [...papers].sort((left, right) => left.paperKey < right.paperKey ? -1 : 1);
  const rows = canonical.flatMap((entry, paperOrdinal) => entry.chunks.map((entryChunk, chunkIndex) => ({
    paperOrdinal,
    paperKey: entry.paperKey,
    chunk: entryChunk,
    vector: Array.from(entry.vectors.subarray(chunkIndex * entry.dimension, (chunkIndex + 1) * entry.dimension)),
  })));
  const writes = new Map<string, Uint8Array>();
  const vectorRefs: GenerationObjectReference[] = [];
  const evidenceRefs: GenerationObjectReference[] = [];
  for (let rowStart = 0, blockIndex = 0; rowStart < rows.length; rowStart += rowsPerBlock, blockIndex += 1) {
    const blockRows = rows.slice(rowStart, rowStart + rowsPerBlock);
    const suffix = String(blockIndex).padStart(6, "0");
    const vectorPath = `objects/${suffix}.vectors.bin`;
    const evidencePath = `objects/${suffix}.evidence.bin`;
    const vector = encodeVectorBlock({
      rowStart,
      dimension: DIMENSION,
      paperOrdinals: new Uint32Array(blockRows.map((entry) => entry.paperOrdinal)),
      vectors: new Float32Array(blockRows.flatMap((entry) => entry.vector)),
    });
    const evidence = encodeEvidenceBlock({
      rowStart,
      records: blockRows.map((entry, offset): EvidenceBlockRecord => ({
        paperIndex: entry.paperOrdinal,
        paperKey: entry.paperKey,
        vectorRow: rowStart + offset,
        chunk: entry.chunk as EvidenceBlockRecord["chunk"],
      })),
    });
    writes.set(vectorPath, vector);
    writes.set(evidencePath, evidence);
    vectorRefs.push({ kind: "vector", path: vectorPath, byteLength: vector.byteLength, recordStart: rowStart, recordCount: blockRows.length, checksum: blockObjectChecksum(vector) });
    evidenceRefs.push({ kind: "evidence", path: evidencePath, byteLength: evidence.byteLength, recordStart: rowStart, recordCount: blockRows.length, checksum: blockObjectChecksum(evidence) });
  }
  const sums = new Float64Array(DIMENSION);
  rows.forEach((entry) => entry.vector.forEach((value, column) => { sums[column]! += value; }));
  const descriptor: GenerationDescriptor = {
    formatVersion: GENERATION_DESCRIPTOR_FORMAT_VERSION,
    schemaVersion: GENERATION_DESCRIPTOR_SCHEMA_VERSION,
    generationId: "fixed-evaluation-generation",
    sourceRevision: 1,
    scopeFingerprint: `sha256:${"a".repeat(64)}`,
    identificationFingerprint: `sha256:${"b".repeat(64)}`,
    modelId: "fixed-evaluation-model",
    dimension: DIMENSION,
    corpusMean: Array.from(sums, (sum) => sum / rows.length),
    corpusStats: { indexedPaperCount: canonical.length, chunkCount: rows.length },
    indexDerivation: { builderVersion: 1, denseCenteringVersion: 1, tokenizerVersion: 1, postingsVersion: 1 },
    objects: [...vectorRefs, ...evidenceRefs],
  };
  return {
    descriptor,
    iterateVectorBlocks: async function* () {
      for (const reference of vectorRefs) yield { reference, block: decodeVectorBlock(writes.get(reference.path)!) } as any;
    },
    readObject: async (reference: (typeof descriptor.objects)[number]) => ({
      reference,
      block: reference.kind === "vector" ? decodeVectorBlock(writes.get(reference.path)!) : decodeEvidenceBlock(writes.get(reference.path)!),
    }),
  } as unknown as OpenedFullTextGeneration;
}

async function generationDenseRankings(centerCorpus?: false): Promise<Record<string, string[]>> {
  const generation = fixedGeneration(CORPUS);
  const rankings: Record<string, string[]> = {};
  for (const query of QUERIES) rankings[query.id] = (await searchGenerationDense({
    generation,
    queryVector: query.queryVector,
    ...(centerCorpus === false ? { centerCorpus: false } : {}),
    limit: CORPUS.length,
  })).map((entry) => entry.paperKey);
  return rankings;
}

function actualRankings(): Record<string, Record<string, string[]>> {
  const rankings: Record<string, Record<string, string[]>> = { dense: {}, bm25: {}, hybrid: {} };
  const titles = new Map(CORPUS.map((entry) => [entry.paperKey, entry.title!]));
  for (const query of QUERIES) {
    const dense = searchKnowledgeBase({
      papers: CORPUS,
      queryVector: query.queryVector,
      centerCorpus: false,
      limit: CORPUS.length,
    });
    const bm25 = searchKnowledgeBaseBm25({
      papers: CORPUS,
      queryText: query.lexicalQueryText ?? query.queryText,
      titles,
      limit: CORPUS.length,
    });
    const hybrid = fusePaperRankingsRrf({
      rankings: [dense, bm25],
      candidateLimit: CORPUS.length,
      limit: CORPUS.length,
    });
    rankings.dense[query.id] = dense.map((entry) => entry.paperKey);
    rankings.bm25[query.id] = bm25.map((entry) => entry.paperKey);
    rankings.hybrid[query.id] = hybrid.map((entry) => entry.paperKey);
  }
  return rankings;
}

describe("retrieval evaluation", () => {
  it("evaluates actual dense, BM25, and RRF rankings over the fixed corpus", () => {
    const report = evaluateRetrieval({ judgments: JUDGMENTS, rankings: actualRankings(), k: 5 });
    expect(report.modes.dense.overall.recall).toBe(1);
    expect(report.modes.dense.overall.mrr).toBeCloseTo(0.6388888888888888, 12);
    expect(report.modes.dense.overall.ndcg).toBeCloseTo(0.7268921860244643, 12);
    expect(report.modes.bm25.overall).toEqual({
      recall: 2 / 3,
      mrr: 2 / 3,
      ndcg: 2 / 3,
    });
    expect(report.modes.hybrid.overall).toEqual({ recall: 1, mrr: 1, ndcg: 1 });
    expect(() => assertHybridRetrievalGates(report, {
      denseMode: "dense",
      lexicalMode: "bm25",
      hybridMode: "hybrid",
      lexicalCategories: ["exact-title", "compact-alias", "cjk-keyword"],
      semanticCategories: ["semantic-rewrite", "title-abstract", "hard-negative"],
    })).not.toThrow();
  });

  it("keeps fixed-generation dense rankings and metrics equal to the P3 oracle", async () => {
    const legacy = actualRankings();
    const generationDense = await generationDenseRankings(false);
    expect(generationDense).toEqual(legacy.dense);
    const report = evaluateRetrieval({
      judgments: JUDGMENTS,
      rankings: { dense: generationDense, bm25: legacy.bm25, hybrid: legacy.hybrid },
      k: 5,
    });
    expect(report.modes.dense.overall).toEqual({ recall: 1, mrr: 0.6388888888888888, ndcg: 0.7268921860244643 });
    expect(report.modes.bm25.overall).toEqual({ recall: 2 / 3, mrr: 2 / 3, ndcg: 2 / 3 });
    expect(report.modes.hybrid.overall).toEqual({ recall: 1, mrr: 1, ndcg: 1 });
  });

  it("keeps fixed centered generation ranking equal to P3 while proving centering changes raw ranking", async () => {
    const generationCentered = await generationDenseRankings();
    const generationRaw = await generationDenseRankings(false);
    const legacyCentered: Record<string, string[]> = {};
    for (const query of QUERIES) legacyCentered[query.id] = searchKnowledgeBase({
      papers: CORPUS,
      queryVector: query.queryVector,
      centerCorpus: true,
      limit: CORPUS.length,
    }).map((entry) => entry.paperKey);
    expect(generationCentered).toEqual(legacyCentered);
    expect(generationCentered).not.toEqual(generationRaw);
    expect(generationCentered).toEqual({
      exact: ["panstarrs", "survey-negative", "chinese", "robust", "semantic", "sky", "attention", "hard-negative"],
      alias: ["attention", "hard-negative", "chinese", "robust", "semantic", "sky", "panstarrs", "survey-negative"],
      cjk: ["panstarrs", "survey-negative", "chinese", "robust", "semantic", "sky", "attention", "hard-negative"],
      semantic: ["semantic", "chinese", "robust", "sky", "attention", "hard-negative", "panstarrs", "survey-negative"],
      long: ["sky", "chinese", "robust", "semantic", "attention", "hard-negative", "panstarrs", "survey-negative"],
      hard: ["robust", "chinese", "semantic", "sky", "attention", "hard-negative", "panstarrs", "survey-negative"],
    });
    const centeredReport = evaluateRetrieval({ judgments: JUDGMENTS, rankings: { dense: generationCentered }, k: 5 });
    expect(centeredReport.modes.dense.overall).toEqual({
      recall: 2 / 3,
      mrr: 0.5555555555555555,
      ndcg: 7 / 12,
    });
  });

  it("fails a semantic category gate on an independently supplied regression", () => {
    const rankings = actualRankings();
    rankings.hybrid.semantic = ["hard-negative"];
    const report = evaluateRetrieval({ judgments: JUDGMENTS, rankings, k: 5 });
    expect(() => assertHybridRetrievalGates(report, {
      denseMode: "dense",
      lexicalMode: "bm25",
      hybridMode: "hybrid",
      lexicalCategories: ["exact-title", "compact-alias", "cjk-keyword"],
      semanticCategories: ["semantic-rewrite", "title-abstract", "hard-negative"],
    })).toThrow(/semantic-rewrite/);
  });
});
