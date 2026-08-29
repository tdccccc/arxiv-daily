import { describe, expect, it } from "vitest";
import type {
  FullTextKnowledgeBaseManifest,
  KnowledgeBasePaperMatch,
} from "@arxiv-daily/core";
import { projectLibraryFullTextMatches } from "../src/library/fulltext-results";

const hit = {
  source: "dense" as const,
  scoreKind: "cosine" as const,
  score: 0.8,
  chunkIndex: 2,
  chunkId: "chunk-2",
  headings: ["Methods"],
  locator: { pageStart: 4 },
  page: 4,
  text: "Evidence passage",
};

describe("projectLibraryFullTextMatches", () => {
  it("preserves ranking and hits while projecting the indexed local PDF path for every paper key", () => {
    const manifest = {
      papers: {
        "arxiv:2607.00001": {
          title: "Indexed arXiv title",
          filePaths: ["papers/arxiv-paper.pdf"],
        },
        "file:sha256:abc": {
          title: "Fallback title",
          filePaths: ["papers/fallback.pdf"],
        },
      },
    } as FullTextKnowledgeBaseManifest;
    const matches = [{
      paperKey: "arxiv:2607.00001",
      score: 0.8,
      scoreKind: "cosine" as const,
      rankingScore: 0.031,
      rankingScoreKind: "rrf" as const,
      hits: [hit],
      chunkCount: 4,
    }, {
      paperKey: "file:sha256:abc",
      score: 2.4,
      scoreKind: "bm25" as const,
      rankingScore: 0.02,
      rankingScoreKind: "rrf" as const,
      hits: [],
      chunkCount: 1,
    }] satisfies KnowledgeBasePaperMatch[];

    const projected = projectLibraryFullTextMatches({
      catalogTitles: new Map([["arxiv:2607.00001", "Catalog title"]]),
      manifest,
      matches,
    });

    expect(projected).toEqual([
      expect.objectContaining({
        paperKey: "arxiv:2607.00001",
        title: "Catalog title",
        filePath: "papers/arxiv-paper.pdf",
        score: 0.8,
        rankingScore: 0.031,
        hits: [hit],
      }),
      expect.objectContaining({
        paperKey: "file:sha256:abc",
        title: "Fallback title",
        filePath: "papers/fallback.pdf",
        score: 2.4,
        rankingScore: 0.02,
        hits: [],
      }),
    ]);
  });
});
