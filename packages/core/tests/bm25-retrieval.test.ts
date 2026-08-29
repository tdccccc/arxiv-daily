import { describe, expect, it } from "vitest";
import {
  searchKnowledgeBaseBm25,
  tokenizeUnicode,
  type Bm25RetrievalStats,
} from "../src/library/fulltext/bm25-retrieval";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";

function paper(paperKey: string, chunks: string[], title?: string): FullTextPaperDocument {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey,
    modelId: "fixture",
    dimension: 2,
    textHash: `sha256:${"1".repeat(64)}`,
    filePaths: [`${paperKey}.pdf`],
    observationFingerprints: [`sha256:${"2".repeat(64)}`],
    title,
    chunks: chunks.map((text, index) => ({ index, page: index + 1, text })),
    vectors: new Float32Array(chunks.length * 2),
    updatedAt: "2026-08-17T00:00:00.000Z",
  };
}

describe("tokenizeUnicode", () => {
  it("normalizes Unicode words and emits Han bigrams with a single-character fallback", () => {
    expect(tokenizeUnicode("ＡI naïve Καλημέρα 中文检索 中")).toEqual([
      "ai", "naïve", "καλημέρα", "中文", "文检", "检索", "中",
    ]);
  });

  it("is deterministic and contains no host-specific segmentation", () => {
    expect(tokenizeUnicode("图神经网络 GNN")).toEqual(["图神", "神经", "经网", "网络", "gnn"]);
    expect(tokenizeUnicode("图神经网络 GNN")).toEqual(tokenizeUnicode("图神经网络 GNN"));
  });
});

describe("searchKnowledgeBaseBm25", () => {
  it("ranks chunk BM25 at paper level and lets each paper contribute once", () => {
    const papers = [
      paper("many", ["quantum sensor", "quantum sensor", "quantum sensor"]),
      paper("focused", ["quantum quantum quantum sensor calibration"]),
      paper("other", ["galaxy morphology"]),
    ];
    const result = searchKnowledgeBaseBm25({ papers, queryText: "quantum sensor", limit: 3, maxHitsPerPaper: 2 });
    expect(new Set(result.map((match) => match.paperKey))).toEqual(new Set(["focused", "many"]));
    expect(result.filter((match) => match.paperKey === "many")).toHaveLength(1);
    expect(result.find((match) => match.paperKey === "many")!.hits).toHaveLength(2);
  });

  it("supports CJK bigrams and single-character queries", () => {
    const papers = [paper("cjk", ["中文检索与证据定位"]), paper("other", ["英文语义搜索"])];
    expect(searchKnowledgeBaseBm25({ papers, queryText: "中文检索" })[0]!.paperKey).toBe("cjk");
    expect(searchKnowledgeBaseBm25({ papers, queryText: "证" })[0]!.paperKey).toBe("cjk");
  });

  it("folds exact/prefix titles and compact aliases into lexical rank", () => {
    const papers = [
      paper("body", ["Pan STARRS survey ".repeat(12)], "Another survey"),
      paper("exact", ["brief abstract"], "Pan-STARRS"),
      paper("prefix", ["brief abstract"], "Pan-STARRS Data Release"),
    ];
    const titles = new Map(papers.map((entry) => [entry.paperKey, entry.title!]));
    const result = searchKnowledgeBaseBm25({ papers, queryText: "panstarrs", titles });
    expect(result.slice(0, 2).map((match) => match.paperKey)).toEqual(["exact", "prefix"]);
    expect(result.find((match) => match.paperKey === "body")).toBeDefined();
  });

  it("keeps paper and hit worksets bounded and reports two scan passes", () => {
    const papers = Array.from({ length: 200 }, (_, index) => paper(
      `p-${String(index).padStart(3, "0")}`,
      Array.from({ length: 8 }, (__, chunk) => `target evidence ${index} ${chunk}`),
    ));
    const stats: Bm25RetrievalStats = { passes: 0, chunksScanned: 0, peakPaperCandidates: 0, peakHitsPerPaper: 0 };
    const result = searchKnowledgeBaseBm25({
      papers,
      queryText: "target evidence",
      limit: 7,
      maxHitsPerPaper: 2,
      stats,
    });
    expect(result).toHaveLength(7);
    expect(result.every((match) => match.hits.length <= 2)).toBe(true);
    expect(stats).toEqual({
      passes: 2,
      chunksScanned: 3200,
      peakPaperCandidates: 7,
      peakHitsPerPaper: 2,
    });
  });

  it("validates bounds and handles empty inputs", () => {
    expect(searchKnowledgeBaseBm25({ papers: [], queryText: "target" })).toEqual([]);
    expect(searchKnowledgeBaseBm25({ papers: [paper("a", ["target"])], queryText: "", limit: 3 })).toEqual([]);
    expect(() => searchKnowledgeBaseBm25({ papers: [paper("a", ["target"])], queryText: "target", limit: -1 }))
      .toThrow(TypeError);
    expect(() => searchKnowledgeBaseBm25({ papers: [paper("a", ["target"])], queryText: "target", maxHitsPerPaper: 0 }))
      .toThrow(TypeError);
  });
});
