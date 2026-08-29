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
    corpusStats: { indexedPaperCount: canonical.length, chunkCount: rows.length, totalLexicalTokenCount: 0, avgdl: 0, totalLexicalTokenCountWithHanSingles: 0, avgdlWithHanSingles: 0 },
    lexicalCapability: "none",
    lexicalRouting: Array.from({ length: 256 }, () => [] as number[]),
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

  it("defines empty judgments, modes, and all-irrelevant judgments as zero metrics", () => {
    expect(evaluateRetrieval({ judgments: [], rankings: { dense: {} }, k: 5 })).toEqual({
      k: 5,
      modes: { dense: { overall: { recall: 0, mrr: 0, ndcg: 0 }, categories: {} } },
    });
    expect(evaluateRetrieval({ judgments: [], rankings: {}, k: 5 })).toEqual({ k: 5, modes: {} });
    expect(evaluateRetrieval({
      judgments: [{ queryId: "query", category: "category", grades: { paper: 0 } }],
      rankings: { dense: { query: ["paper"] } },
      k: 5,
    }).modes.dense).toEqual({
      overall: { recall: 0, mrr: 0, ndcg: 0 },
      categories: { category: { recall: 0, mrr: 0, ndcg: 0 } },
    });
  });

  it("rejects invalid graded relevance values outside the explicit integer range 0 through 3", () => {
    for (const grade of [-1, Number.NaN, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, 4, Number.MAX_VALUE, 1.5]) {
      expect(() => evaluateRetrieval({
        judgments: [{ queryId: "query", category: "category", grades: { paper: grade } }],
        rankings: { dense: { query: ["paper"] } },
        k: 5,
      })).toThrowError(/grade.*finite integer.*0.*3/);
    }
  });

  it("rejects malformed judgment query IDs and paper keys", () => {
    expect(() => evaluateRetrieval({
      judgments: [{ queryId: "", category: "category", grades: { paper: 3 } }],
      rankings: {},
      k: 5,
    })).toThrowError(/queryId.*non-empty string/);
    expect(() => evaluateRetrieval({
      judgments: [{ queryId: "query", category: "category", grades: { "": 3 } }],
      rankings: {},
      k: 5,
    })).toThrowError(/paper key.*non-empty string/);
  });

  it("fails fast when judgments repeat a query ID", () => {
    expect(() => evaluateRetrieval({
      judgments: [
        { queryId: "query", category: "first", grades: { relevant: 3 } },
        { queryId: "query", category: "second", grades: { other: 3 } },
      ],
      rankings: { dense: { query: ["relevant", "other"] } },
      k: 5,
    })).toThrowError(new TypeError("evaluateRetrieval: duplicate judgment queryId query"));
  });

  it("fails fast when a ranking repeats a relevant or nonrelevant paper, including beyond k", () => {
    const judgments: readonly RetrievalJudgment[] = [
      { queryId: "query", category: "category", grades: { relevant: 3, other: 1 } },
    ];
    expect(() => evaluateRetrieval({
      judgments,
      rankings: { duplicateRelevant: { query: ["relevant", "relevant"] } },
      k: 5,
    })).toThrowError(new TypeError("evaluateRetrieval: duplicate paper key relevant in mode duplicateRelevant query query"));
    expect(() => evaluateRetrieval({
      judgments,
      rankings: { duplicateNonrelevant: { query: ["irrelevant", "irrelevant"] } },
      k: 5,
    })).toThrowError(new TypeError("evaluateRetrieval: duplicate paper key irrelevant in mode duplicateNonrelevant query query"));
    expect(() => evaluateRetrieval({
      judgments,
      rankings: { beyondK: { query: ["relevant", "a", "b", "c", "d", "outside", "outside"] } },
      k: 5,
    })).toThrowError(new TypeError("evaluateRetrieval: duplicate paper key outside in mode beyondK query query"));
  });

  it("keeps every calculated metric finite and within zero and one", () => {
    const report = evaluateRetrieval({ judgments: JUDGMENTS, rankings: actualRankings(), k: 5 });
    for (const mode of Object.values(report.modes)) {
      for (const metrics of [mode.overall, ...Object.values(mode.categories)]) {
        for (const value of Object.values(metrics)) {
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeGreaterThanOrEqual(0);
          expect(value).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it("rejects non-finite or out-of-range metrics at the final gate boundary", () => {
    const report = evaluateRetrieval({ judgments: JUDGMENTS, rankings: actualRankings(), k: 5 });
    const invalidReport = {
      ...report,
      modes: {
        ...report.modes,
        hybrid: { ...report.modes.hybrid, overall: { ...report.modes.hybrid.overall, ndcg: Number.NaN } },
      },
    };
    expect(() => assertHybridRetrievalGates(invalidReport, {
      denseMode: "dense",
      lexicalMode: "bm25",
      hybridMode: "hybrid",
      lexicalCategories: ["exact-title"],
      semanticCategories: ["semantic-rewrite"],
    })).toThrowError(/metric.*finite.*0.*1/);
  });

  it("requires semantic recall tolerance to be finite and between zero and one", () => {
    const report = evaluateRetrieval({ judgments: JUDGMENTS, rankings: actualRankings(), k: 5 });
    const gates = {
      denseMode: "dense",
      lexicalMode: "bm25",
      hybridMode: "hybrid",
      lexicalCategories: ["exact-title", "compact-alias", "cjk-keyword"],
      semanticCategories: ["semantic-rewrite", "title-abstract", "hard-negative"],
    } as const;
    for (const semanticRecallTolerance of [-0.01, 1.01, Number.NaN, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]) {
      expect(() => assertHybridRetrievalGates(report, { ...gates, semanticRecallTolerance }))
        .toThrowError(/semanticRecallTolerance.*finite.*0.*1/);
    }
    expect(() => assertHybridRetrievalGates(report, { ...gates, semanticRecallTolerance: 0 })).not.toThrow();
    expect(() => assertHybridRetrievalGates(report, { ...gates, semanticRecallTolerance: 1 })).not.toThrow();
  });

  it("propagates duplicate-ranking failure before hybrid gates can accept inflated metrics", () => {
    const rankings = actualRankings();
    rankings.hybrid.exact = ["attention", "attention", ...rankings.hybrid.exact];
    expect(() => {
      const report = evaluateRetrieval({ judgments: JUDGMENTS, rankings, k: 5 });
      assertHybridRetrievalGates(report, {
        denseMode: "dense",
        lexicalMode: "bm25",
        hybridMode: "hybrid",
        lexicalCategories: ["exact-title", "compact-alias", "cjk-keyword"],
        semanticCategories: ["semantic-rewrite", "title-abstract", "hard-negative"],
      });
    }).toThrowError(/duplicate paper key attention in mode hybrid query exact/);
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
