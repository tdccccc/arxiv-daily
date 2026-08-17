import { describe, expect, it } from "vitest";
import { fusePaperRankingsRrf } from "../src/library/fulltext/hybrid-retrieval";
import type { KnowledgeBasePaperMatch } from "../src/library/fulltext/retrieval";

function match(
  paperKey: string,
  score: number,
  hitIds: string[],
  source: "dense" | "lexical" = score <= 1 ? "dense" : "lexical",
): KnowledgeBasePaperMatch {
  const scoreKind = source === "dense" ? "cosine" : "bm25";
  return {
    paperKey,
    score,
    scoreKind,
    rankingScore: score,
    rankingScoreKind: scoreKind,
    chunkCount: hitIds.length,
    hits: hitIds.map((chunkId, chunkIndex) => ({
      source,
      scoreKind,
      chunkIndex,
      chunkId,
      headings: [],
      locator: { pageStart: chunkIndex + 1 },
      page: chunkIndex + 1,
      text: `${paperKey}-${chunkId}`,
      score: score - chunkIndex / 100,
    })),
  };
}

describe("fusePaperRankingsRrf", () => {
  it("fuses at paper level so duplicate chunks never cast extra votes", () => {
    const dense = [match("semantic", 0.9, ["d1"]), match("both", 0.8, ["b1"])];
    const lexical = [
      match("lexical", 9, ["l1"]),
      match("both", 8, ["b1", "b2", "b3"]),
      match("both", 7, ["b4"]),
    ];
    const fused = fusePaperRankingsRrf({ rankings: [dense, lexical], rrfK: 60, limit: 3 });
    expect(fused[0]!.paperKey).toBe("both");
    expect(fused.find((entry) => entry.paperKey === "both")!.rankingScore)
      .toBeCloseTo(1 / 62 + 1 / 62, 12);
  });

  it("deduplicates EvidenceChunk hits and fills the limit from later channel hits", () => {
    const dense = [match("paper", 0.9, ["same", "dense-only"])];
    const lexical = [match("paper", 5, ["same", "lexical-only"])];
    const fused = fusePaperRankingsRrf({ rankings: [dense, lexical], maxHitsPerPaper: 3 });
    expect(fused[0]!.hits.map((hit) => hit.chunkId)).toEqual(["same", "lexical-only", "dense-only"]);
    expect(new Set(fused[0]!.hits.map((hit) => hit.chunkId)).size).toBe(3);
  });

  it("skips repeated prefixes independently per channel and fills every available hit slot", () => {
    const dense = [match("paper", 0.9, ["same", "dense-only", "dense-second"])];
    const lexical = [match("paper", 5, ["same", "same", "same", "lexical-only"])];
    const fused = fusePaperRankingsRrf({ rankings: [dense, lexical], maxHitsPerPaper: 4 });
    expect(fused[0]!.hits.map((hit) => hit.chunkId)).toEqual([
      "same", "lexical-only", "dense-only", "dense-second",
    ]);
  });

  it("deduplicates each branch before applying candidateLimit", () => {
    const fused = fusePaperRankingsRrf({
      rankings: [[match("duplicate", 3, ["d1"]), match("duplicate", 2, ["d2"]), match("next", 1, ["n1"])]],
      candidateLimit: 2,
      limit: 2,
    });
    expect(fused.map((entry) => entry.paperKey)).toEqual(["duplicate", "next"]);
  });

  it("round-robins cross-channel evidence and identifies incomparable hit scores", () => {
    const dense = [match("paper", 0.9, ["d1", "d2", "d3"])];
    const lexical = [match("paper", 9, ["l1", "l2", "l3"])];
    const fused = fusePaperRankingsRrf({ rankings: [dense, lexical], maxHitsPerPaper: 4 });
    expect(fused[0]!.hits.map((hit) => hit.chunkId)).toEqual(["d1", "l1", "d2", "l2"]);
    expect(fused[0]!.hits.map((hit) => [hit.source, hit.scoreKind])).toEqual([
      ["dense", "cosine"], ["lexical", "bm25"], ["dense", "cosine"], ["lexical", "bm25"],
    ]);
    expect(fused[0]).toMatchObject({ scoreKind: "cosine", rankingScoreKind: "rrf" });
    expect(fused[0]!.rankingScore).toBeCloseTo(2 / 61, 12);
    expect(fused[0]!.score).toBe(0.9);
  });

  it("bounds candidates and breaks ties by paperKey", () => {
    const fused = fusePaperRankingsRrf({
      rankings: [[match("z", 1, ["z1"]), match("a", 0.9, ["a1"])], [match("a", 4, ["a2"]), match("z", 3, ["z2"])]],
      limit: 1,
      candidateLimit: 1,
    });
    expect(fused.map((entry) => entry.paperKey)).toEqual(["a"]);
  });

  it("validates bounds and handles empty rankings", () => {
    expect(fusePaperRankingsRrf({ rankings: [] })).toEqual([]);
    expect(() => fusePaperRankingsRrf({ rankings: [], rrfK: -1 })).toThrow(TypeError);
    expect(() => fusePaperRankingsRrf({ rankings: [], maxHitsPerPaper: 0 })).toThrow(TypeError);
  });
});
