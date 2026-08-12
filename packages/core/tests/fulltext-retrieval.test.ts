import { describe, expect, it } from "vitest";
import { cosineSimilarity, searchKnowledgeBase } from "../src/library/fulltext/retrieval";
import {
  FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
  type FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";

/** Small shared dimension keeps fixtures readable while exercising real math. */
const DIMENSION = 12;

/**
 * Deterministic PRNG (mulberry32): the same seed always yields the same
 * sequence, so every fixture vector is random-but-fixed across runs.
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A random-but-fixed vector with values in [-1, 1); never the zero vector. */
function randomVector(seed: number, dimension: number = DIMENSION): Float32Array {
  const rng = mulberry32(seed);
  const vector = new Float32Array(dimension);
  for (let index = 0; index < dimension; index += 1) {
    const value = rng() * 2 - 1;
    vector[index] = value === 0 ? 0.001 : value;
  }
  return vector;
}

interface PaperFixture {
  paperKey: string;
  chunks: Array<{ page: number; text: string }>;
  /** One vector per chunk, in chunk order. */
  vectors: readonly Float32Array[];
  /** Optional explicit dimension (needed when there are no vectors). */
  dimension?: number;
}

/** Build a realistic `FullTextPaperDocument` with row-major concatenated vectors. */
function makePaper(fixture: PaperFixture): FullTextPaperDocument {
  const dimension = fixture.dimension ?? fixture.vectors[0]?.length ?? DIMENSION;
  const vectors = new Float32Array(fixture.vectors.length * dimension);
  fixture.vectors.forEach((vector, index) => vectors.set(vector, index * dimension));
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    paperKey: fixture.paperKey,
    modelId: "multilingual-e5-small-q8",
    dimension,
    textHash: `sha256:${"1".repeat(64)}`,
    filePaths: [`library/${fixture.paperKey}.pdf`],
    observationFingerprints: [`sha256:${"2".repeat(64)}`],
    chunks: fixture.chunks.map((chunk, index) => ({ index, page: chunk.page, text: chunk.text })),
    vectors,
    updatedAt: "2026-08-05T00:00:00.000Z",
  };
}

describe("cosineSimilarity", () => {
  it("returns ~1 for identical vectors", () => {
    const a = randomVector(1);
    expect(cosineSimilarity(a, new Float32Array(a))).toBeCloseTo(1, 6);
  });

  it("is symmetric", () => {
    const a = randomVector(7);
    const b = randomVector(9);
    expect(cosineSimilarity(a, b)).toBe(cosineSimilarity(b, a));
  });

  it("returns ~0 for orthogonal vectors", () => {
    expect(cosineSimilarity(new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0]))).toBeCloseTo(0, 6);
    expect(cosineSimilarity(new Float32Array([0.5, 0.5, 0]), new Float32Array([-0.5, 0.5, 0]))).toBeCloseTo(0, 6);
  });

  it("returns -1 for opposite vectors", () => {
    expect(cosineSimilarity(new Float32Array([2, 0]), new Float32Array([-1, 0]))).toBe(-1);
  });

  it("throws when lengths differ", () => {
    expect(() => cosineSimilarity(new Float32Array([1, 2, 3]), new Float32Array([1, 2])))
      .toThrow(/equal length/);
  });

  it("returns 0 when either vector is zero", () => {
    const zero = new Float32Array(4);
    const vector = new Float32Array([1, 2, 3, 4]);
    expect(cosineSimilarity(zero, vector)).toBe(0);
    expect(cosineSimilarity(vector, zero)).toBe(0);
    expect(cosineSimilarity(zero, zero)).toBe(0);
  });
});

describe("searchKnowledgeBase", () => {
  it("ranks a paper with an exactly matching chunk first with score ~1 and correct text/page", () => {
    const query = randomVector(101);
    const papers = [
      makePaper({
        paperKey: "paper-a",
        chunks: [{ page: 1, text: "Alpha intro" }, { page: 3, text: "Alpha methods" }],
        vectors: [randomVector(102), randomVector(103)],
      }),
      makePaper({
        paperKey: "paper-b",
        chunks: [
          { page: 2, text: "Beta target passage" },
          { page: 5, text: "Beta results" },
          { page: 7, text: "Beta discussion" },
        ],
        vectors: [new Float32Array(query), randomVector(104), randomVector(105)],
      }),
      makePaper({
        paperKey: "paper-c",
        chunks: [{ page: 1, text: "Gamma abstract" }],
        vectors: [randomVector(106)],
      }),
    ];
    const matches = searchKnowledgeBase({ papers, queryVector: query });
    expect(matches).toHaveLength(3);
    expect(matches[0]!.paperKey).toBe("paper-b");
    expect(matches[0]!.score).toBeCloseTo(1, 6);
    expect(matches[0]!.chunkCount).toBe(3);
    expect(matches[0]!.hits).toHaveLength(3); // default maxHitsPerPaper
    expect(matches[0]!.hits[0]!.chunkIndex).toBe(0);
    expect(matches[0]!.hits[0]!.page).toBe(2);
    expect(matches[0]!.hits[0]!.text).toBe("Beta target passage");
    expect(matches[0]!.hits[0]!.score).toBeCloseTo(1, 6);
    expect(matches[1]!.score).toBeLessThan(1);
    expect(matches[2]!.score).toBeLessThan(1);
  });

  it("uses the max chunk similarity as the paper score: one strong passage beats a higher average", () => {
    // query = [1, 0] in 2D; [0.9, sqrt(1 - 0.9^2)] has cosine 0.9, [0.6, 0.8] has 0.6.
    const query = new Float32Array([1, 0]);
    const strongPassage = makePaper({
      paperKey: "strong-passage",
      chunks: [{ page: 1, text: "Highly relevant passage" }, { page: 2, text: "Unrelated section" }],
      vectors: [
        new Float32Array([0.9, Math.sqrt(1 - 0.9 * 0.9)]),
        new Float32Array([0, 1]),
      ],
    });
    const broadAverage = makePaper({
      paperKey: "broad-average",
      chunks: [{ page: 1, text: "Even coverage one" }, { page: 2, text: "Even coverage two" }],
      vectors: [new Float32Array([0.6, 0.8]), new Float32Array([0.6, 0.8])],
    });
    const matches = searchKnowledgeBase({
      papers: [broadAverage, strongPassage],
      queryVector: query,
      centerCorpus: false,
    });
    expect(matches[0]!.paperKey).toBe("strong-passage");
    expect(matches[0]!.score).toBeCloseTo(0.9, 6);
    expect(matches[0]!.hits[0]!.text).toBe("Highly relevant passage");
    expect(matches[1]!.paperKey).toBe("broad-average");
    expect(matches[1]!.score).toBeCloseTo(0.6, 6);
    // The broad paper's hits are both mediocre, so a mean strategy would have
    // ranked it first (mean 0.6 > mean 0.45); max strategy keeps the passage
    // that actually matches the query on top.
    expect(matches[0]!.score).toBeGreaterThan(matches[1]!.score);
  });

  it("ranks a paper with one very high chunk above a paper with many mediocre chunks", () => {
    const query = randomVector(201);
    const focused = makePaper({
      paperKey: "focused",
      chunks: [{ page: 1, text: "Focused intro" }, { page: 2, text: "Focused key passage" }],
      vectors: [randomVector(202), new Float32Array(query)],
    });
    const diffuse = makePaper({
      paperKey: "diffuse",
      chunks: Array.from({ length: 10 }, (_, k) => ({ page: k + 1, text: `Diffuse paragraph ${k}` })),
      vectors: Array.from({ length: 10 }, (_, k) => randomVector(300 + k)),
    });
    const matches = searchKnowledgeBase({ papers: [diffuse, focused], queryVector: query });
    expect(matches[0]!.paperKey).toBe("focused");
    expect(matches[0]!.score).toBeCloseTo(1, 6);
    expect(matches[1]!.paperKey).toBe("diffuse");
    expect(matches[1]!.chunkCount).toBe(10);
    expect(matches[1]!.score).toBeLessThan(matches[0]!.score);
  });

  it("returns at most limit papers", () => {
    const query = randomVector(301);
    const papers = [
      makePaper({ paperKey: "p-limit-a", chunks: [{ page: 1, text: "A" }], vectors: [randomVector(302)] }),
      makePaper({ paperKey: "p-limit-b", chunks: [{ page: 1, text: "B" }], vectors: [new Float32Array(query)] }),
      makePaper({ paperKey: "p-limit-c", chunks: [{ page: 1, text: "C" }], vectors: [randomVector(303)] }),
    ];
    const matches = searchKnowledgeBase({ papers, queryVector: query, limit: 2 });
    expect(matches).toHaveLength(2);
    expect(matches[0]!.paperKey).toBe("p-limit-b");
    const all = searchKnowledgeBase({ papers, queryVector: query });
    expect(all).toHaveLength(3);
    expect(matches[1]!.paperKey).toBe(all[1]!.paperKey);
    expect(matches[1]!.score).toBe(all[1]!.score);
  });

  it("returns at most maxHitsPerPaper hits per paper", () => {
    const query = randomVector(401);
    const paper = makePaper({
      paperKey: "p-hits",
      chunks: Array.from({ length: 5 }, (_, k) => ({ page: k + 1, text: `Hit passage ${k}` })),
      vectors: [new Float32Array(query), randomVector(402), randomVector(403), randomVector(404), randomVector(405)],
    });
    const matches = searchKnowledgeBase({ papers: [paper], queryVector: query, maxHitsPerPaper: 2 });
    expect(matches).toHaveLength(1);
    expect(matches[0]!.hits).toHaveLength(2);
    expect(matches[0]!.hits[0]!.chunkIndex).toBe(0);
    expect(matches[0]!.hits[0]!.score).toBeCloseTo(1, 6);
    const full = searchKnowledgeBase({ papers: [paper], queryVector: query });
    expect(full[0]!.hits).toHaveLength(3); // default maxHitsPerPaper
  });

  it("sorts hit evidence by score descending", () => {
    const query = randomVector(501);
    const paper = makePaper({
      paperKey: "p-sorted",
      chunks: Array.from({ length: 5 }, (_, k) => ({ page: k + 1, text: `Sorted passage ${k}` })),
      vectors: [randomVector(502), new Float32Array(query), randomVector(503), randomVector(504), randomVector(505)],
    });
    const match = searchKnowledgeBase({ papers: [paper], queryVector: query, maxHitsPerPaper: 5 })[0]!;
    expect(match.hits).toHaveLength(5);
    expect(match.hits[0]!.chunkIndex).toBe(1);
    for (let index = 1; index < match.hits.length; index += 1) {
      expect(match.hits[index]!.score).toBeLessThanOrEqual(match.hits[index - 1]!.score);
    }
  });

  it("breaks score ties deterministically by paperKey ascending", () => {
    const query = randomVector(701);
    const papers = [
      makePaper({ paperKey: "zebra", chunks: [{ page: 1, text: "Zebra passage" }], vectors: [new Float32Array(query)] }),
      makePaper({ paperKey: "alpha", chunks: [{ page: 1, text: "Alpha passage" }], vectors: [new Float32Array(query)] }),
      makePaper({ paperKey: "mike", chunks: [{ page: 1, text: "Mike passage" }], vectors: [new Float32Array(query)] }),
    ];
    const matches = searchKnowledgeBase({ papers, queryVector: query });
    expect(matches.map((match) => match.paperKey)).toEqual(["alpha", "mike", "zebra"]);
    // Deterministic: identical input twice yields identical output.
    expect(searchKnowledgeBase({ papers, queryVector: query })).toEqual(matches);
  });

  it("is deterministic for non-trivial fixtures", () => {
    const query = randomVector(1001);
    const papers = [
      makePaper({
        paperKey: "det-a",
        chunks: [{ page: 1, text: "Det A one" }, { page: 2, text: "Det A two" }],
        vectors: [randomVector(1002), randomVector(1003)],
      }),
      makePaper({
        paperKey: "det-b",
        chunks: [{ page: 4, text: "Det B one" }, { page: 6, text: "Det B two" }, { page: 9, text: "Det B three" }],
        vectors: [randomVector(1004), randomVector(1005), randomVector(1006)],
      }),
      makePaper({
        paperKey: "det-c",
        chunks: [{ page: 2, text: "Det C one" }],
        vectors: [randomVector(1007)],
      }),
    ];
    const first = searchKnowledgeBase({ papers, queryVector: query });
    expect(searchKnowledgeBase({ papers, queryVector: query })).toEqual(first);
  });

  it("returns an empty list for empty papers or limit 0", () => {
    const query = randomVector(601);
    expect(searchKnowledgeBase({ papers: [], queryVector: query })).toEqual([]);
    const paper = makePaper({
      paperKey: "solo",
      chunks: [{ page: 1, text: "Solo" }],
      vectors: [new Float32Array(query)],
    });
    expect(searchKnowledgeBase({ papers: [paper], queryVector: query, limit: 0 })).toEqual([]);
  });

  it("skips papers with no chunks (no evidence)", () => {
    const query = randomVector(1101);
    const empty = makePaper({ paperKey: "no-chunks", chunks: [], vectors: [] });
    const withChunk = makePaper({
      paperKey: "has-chunks",
      chunks: [{ page: 1, text: "Evidence" }],
      vectors: [new Float32Array(query)],
    });
    const matches = searchKnowledgeBase({ papers: [empty, withChunk], queryVector: query });
    expect(matches.map((match) => match.paperKey)).toEqual(["has-chunks"]);
  });

  it("throws when the query dimension does not match a paper dimension", () => {
    const sevenDimensional = makePaper({
      paperKey: "dim-7",
      chunks: [{ page: 1, text: "Seven-dimensional passage" }],
      vectors: [randomVector(701, 7)],
      dimension: 7,
    });
    expect(() => searchKnowledgeBase({ papers: [sevenDimensional], queryVector: new Float32Array(12) }))
      .toThrow(/dimension/);
    // Mixed batch: the mismatch surfaces as an error, never as partial results.
    const query = randomVector(801);
    const ok = makePaper({
      paperKey: "dim-12",
      chunks: [{ page: 1, text: "Twelve-dimensional passage" }],
      vectors: [new Float32Array(query)],
    });
    expect(() => searchKnowledgeBase({ papers: [ok, sevenDimensional], queryVector: query }))
      .toThrow(/rebuilt/);
  });

  it("rejects invalid limit and maxHitsPerPaper values", () => {
    const query = randomVector(901);
    const paper = makePaper({
      paperKey: "opts",
      chunks: [{ page: 1, text: "Opts" }],
      vectors: [new Float32Array(query)],
    });
    expect(() => searchKnowledgeBase({ papers: [paper], queryVector: query, limit: -1 })).toThrow(TypeError);
    expect(() => searchKnowledgeBase({ papers: [paper], queryVector: query, limit: 1.5 })).toThrow(TypeError);
    expect(() => searchKnowledgeBase({ papers: [paper], queryVector: query, maxHitsPerPaper: 0 })).toThrow(TypeError);
    expect(() => searchKnowledgeBase({ papers: [paper], queryVector: query, maxHitsPerPaper: 2.5 })).toThrow(TypeError);
  });

  it("lifts a paper whose title matches the query above stronger chunk evidence", () => {
    // 2-D query; the chunk match scores 0.9, an exact lexical title match 1.
    const query = new Float32Array([1, 0]);
    const chunkMatch = makePaper({
      paperKey: "chunk-match",
      chunks: [{ page: 1, text: "Passage close to the query" }],
      vectors: [new Float32Array([0.9, Math.sqrt(1 - 0.9 * 0.9)])],
    });
    const titleMatch = makePaper({
      paperKey: "title-match",
      chunks: [{ page: 1, text: "Unrelated passage" }],
      vectors: [new Float32Array([0, 1])],
    });
    // Without title scores the best chunk wins, as before.
    const without = searchKnowledgeBase({ papers: [chunkMatch, titleMatch], queryVector: query });
    expect(without[0]!.paperKey).toBe("chunk-match");
    // With a title score of 1 the title match outranks chunk evidence.
    const withTitles = searchKnowledgeBase({
      papers: [chunkMatch, titleMatch],
      queryVector: query,
      titleScores: new Map([["title-match", 1]]),
    });
    expect(withTitles[0]!.paperKey).toBe("title-match");
    expect(withTitles[0]!.score).toBe(1);
    expect(withTitles[1]!.paperKey).toBe("chunk-match");
    // Chunk evidence stays the reason for the rank: hits are unchanged.
    expect(withTitles[0]!.hits[0]!.text).toBe("Unrelated passage");
  });

  it("leaves ranking and scores unchanged when no title matches the query", () => {
    const query = new Float32Array([1, 0]);
    const a = makePaper({
      paperKey: "a",
      chunks: [{ page: 1, text: "A passage" }],
      vectors: [new Float32Array([0.9, Math.sqrt(1 - 0.9 * 0.9)])],
    });
    const b = makePaper({
      paperKey: "b",
      chunks: [{ page: 1, text: "B passage" }],
      vectors: [new Float32Array([0.5, Math.sqrt(1 - 0.5 * 0.5)])],
    });
    const base = searchKnowledgeBase({ papers: [a, b], queryVector: query, centerCorpus: false });
    // Title scores (0 and 0.2) stay below each paper's chunk score.
    const fused = searchKnowledgeBase({
      papers: [a, b],
      queryVector: query,
      centerCorpus: false,
      titleScores: new Map([
        ["a", 0],
        ["b", 0.2],
      ]),
    });
    expect(fused.map((match) => match.paperKey)).toEqual(base.map((match) => match.paperKey));
    expect(fused[0]!.score).toBeCloseTo(base[0]!.score, 6);
    expect(fused[1]!.score).toBeCloseTo(base[1]!.score, 6);
  });

  it("lifts a paper with a literal token hit above stronger chunk evidence", () => {
    const query = new Float32Array([1, 0]);
    const chunkMatch = makePaper({
      paperKey: "chunk-match",
      chunks: [{ page: 1, text: "Passage close to the query" }],
      vectors: [new Float32Array([0.9, Math.sqrt(1 - 0.9 * 0.9)])],
    });
    const tokenMatch = makePaper({
      paperKey: "token-match",
      chunks: [{ page: 1, text: "Unrelated passage" }],
      vectors: [new Float32Array([0, 1])],
    });
    const matches = searchKnowledgeBase({
      papers: [chunkMatch, tokenMatch],
      queryVector: query,
      tokenScores: new Map([["token-match", 1]]),
    });
    expect(matches[0]!.paperKey).toBe("token-match");
    expect(matches[0]!.score).toBe(1);
    expect(matches[1]!.paperKey).toBe("chunk-match");
  });

  it("never surfaces a paper with no chunks even when its title matches", () => {
    const query = randomVector(1303);
    const empty = makePaper({ paperKey: "no-chunks", chunks: [], vectors: [] });
    const withChunk = makePaper({
      paperKey: "has-chunks",
      chunks: [{ page: 1, text: "Evidence" }],
      vectors: [new Float32Array(query)],
    });
    const matches = searchKnowledgeBase({
      papers: [empty, withChunk],
      queryVector: query,
      titleScores: new Map([["no-chunks", 1]]),
    });
    expect(matches.map((match) => match.paperKey)).toEqual(["has-chunks"]);
  });

  it("centers the corpus so a thematically aligned paper outranks a generic high-baseline chunk", () => {
    // Shared academic baseline on x, theme on y. Raw cosine prefers near-baseline
    // fillers; after centering, the theme paper rises to the top.
    const query = new Float32Array([1, 0.3]);
    const themeMatch = makePaper({
      paperKey: "theme-match",
      chunks: [{ page: 1, text: "Theme passage" }],
      vectors: [new Float32Array([0.5, 1])],
    });
    const genericPopular = makePaper({
      paperKey: "generic-popular",
      chunks: [{ page: 1, text: "Generic academic filler" }],
      vectors: [new Float32Array([1, 0])],
    });
    const filler = [
      makePaper({
        paperKey: "filler-a",
        chunks: [{ page: 1, text: "Filler A" }],
        vectors: [new Float32Array([0.95, 0.05])],
      }),
      makePaper({
        paperKey: "filler-b",
        chunks: [{ page: 1, text: "Filler B" }],
        vectors: [new Float32Array([0.9, -0.05])],
      }),
      makePaper({
        paperKey: "filler-c",
        chunks: [{ page: 1, text: "Filler C" }],
        vectors: [new Float32Array([0.98, 0.02])],
      }),
    ];
    const raw = searchKnowledgeBase({
      papers: [themeMatch, genericPopular, ...filler],
      queryVector: query,
      centerCorpus: false,
    });
    expect(raw[0]!.paperKey).not.toBe("theme-match");
    const centered = searchKnowledgeBase({
      papers: [themeMatch, genericPopular, ...filler],
      queryVector: query,
    });
    expect(centered[0]!.paperKey).toBe("theme-match");
    expect(centered[0]!.score).toBeGreaterThan(centered[1]!.score);
  });
});
