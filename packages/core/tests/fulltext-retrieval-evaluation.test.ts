import { describe, expect, it } from "vitest";
import { searchKnowledgeBaseBm25 } from "../src/library/fulltext/bm25-retrieval";
import { fusePaperRankingsRrf } from "../src/library/fulltext/hybrid-retrieval";
import {
  assertHybridRetrievalGates,
  evaluateRetrieval,
  type RetrievalJudgment,
} from "../src/library/fulltext/retrieval-evaluation";
import { searchKnowledgeBase } from "../src/library/fulltext/retrieval";
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
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    title,
    modelId: "fixed-evaluation-model",
    dimension: DIMENSION,
    textHash: `sha256:${"1".repeat(64)}`,
    filePaths: [`${paperKey}.pdf`],
    observationFingerprints: [`sha256:${"2".repeat(64)}`],
    chunks: [{ index: 0, page: 1, text }],
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
