import { describe, expect, it } from "vitest";
import {
  compactText,
  computeTokenHitScores,
  isKeywordQuery,
  significantQueryTokens,
} from "../src/library/fulltext/lexical-search";

describe("significantQueryTokens", () => {
  it("normalizes, drops stop words, and drops short tokens", () => {
    expect(significantQueryTokens("The Pan-STARRS Survey")).toEqual(["pan", "starrs", "survey"]);
    expect(significantQueryTokens("how does dropout work")).toEqual(["dropout", "work"]);
    expect(significantQueryTokens("a b c")).toEqual([]);
  });
});

describe("compactText", () => {
  it("lowercases and removes every non-alphanumeric", () => {
    expect(compactText("Pan-STARRS survey: the 1st data release!")).toBe("panstarrssurveythe1stdatarelease");
  });
});

describe("computeTokenHitScores", () => {
  const papers = [
    { paperKey: "arxiv:a", text: "The Pan-STARRS survey. Pan-STARRS data. Pan-STARRS imaging." },
    { paperKey: "arxiv:b", text: "The dropout regularization in the deep neural networks." },
    { paperKey: "arxiv:c", text: "The study of galaxy clusters and their mass profiles." },
  ];

  it("scores papers by frequency-graded hit ratio", () => {
    const scores = computeTokenHitScores(papers, ["panstarrs"]);
    expect(scores.get("arxiv:a")).toBeCloseTo(3 / 6, 6); // 3 mentions → 3/(3+3)
    expect(scores.has("arxiv:b")).toBe(false);
    expect(scores.has("arxiv:c")).toBe(false);
  });

  it("downgrades a passing single mention", () => {
    const passing = [
      { paperKey: "arxiv:p", text: "future surveys such as Pan-STARRS will help." },
      { paperKey: "arxiv:q", text: "unrelated text." },
      { paperKey: "arxiv:r", text: "more unrelated text." },
    ];
    const scores = computeTokenHitScores(passing, ["panstarrs"]);
    expect(scores.get("arxiv:p")).toBeCloseTo(1 / 4, 6); // 1 mention → 1/(1+3)
  });

  it("keeps title matches high while body frequency distinguishes topical ties", () => {
    const topical = [
      {
        paperKey: "arxiv:surveys",
        title: "The Pan-STARRS1 Surveys",
        text: "Pan-STARRS ".repeat(269),
      },
      {
        paperKey: "arxiv:database",
        title: "The Pan-STARRS1 Database and Data Products",
        text: "Pan-STARRS ".repeat(118),
      },
      { paperKey: "arxiv:u", text: "unrelated." },
      { paperKey: "arxiv:v", text: "unrelated too." },
      { paperKey: "arxiv:w", text: "still unrelated." },
    ];
    const scores = computeTokenHitScores(topical, ["panstarrs"]);
    expect(scores.get("arxiv:surveys")).toBeGreaterThan(scores.get("arxiv:database")!);
    expect(scores.get("arxiv:database")).toBeGreaterThan(0.95);
  });

  it("matches hyphenated terms with and without the hyphen", () => {
    // "Pan-STARRS" (paper) and "panstarrs" (query) both compact to the same form.
    const scores = computeTokenHitScores(papers, ["pan", "starrs"]);
    expect(scores.get("arxiv:a")).toBeCloseTo(3 / 6, 6);
  });

  it("drops tokens that appear in too many papers", () => {
    // "the" appears in every paper → dropped as common; "deep" stays rare.
    const common = computeTokenHitScores(papers, ["the", "deep"]);
    expect(common.get("arxiv:b")).toBeCloseTo(0.25, 6); // deep, single mention
    expect(common.has("arxiv:a")).toBe(false);
    expect(common.has("arxiv:c")).toBe(false);
  });

  it("returns an empty map for no tokens or no papers", () => {
    expect(computeTokenHitScores([], ["panstarrs"])).toEqual(new Map());
    expect(computeTokenHitScores(papers, [])).toEqual(new Map());
  });
});

describe("isKeywordQuery", () => {
  it("accepts short keyword queries and rejects title+abstract blobs", () => {
    expect(isKeywordQuery("panstarrs")).toBe(true);
    expect(isKeywordQuery("The Pan-STARRS1 Surveys")).toBe(true);
    expect(
      isKeywordQuery(
        "Deep galaxy surveys with wide-field imaging\n\nWe present a study of galaxies, surveys, imaging, photometry, and redshift measurements across a wide field.",
      ),
    ).toBe(false);
  });
});
