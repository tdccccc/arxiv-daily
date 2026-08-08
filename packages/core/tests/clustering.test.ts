import { describe, expect, it } from "vitest";
import {
  centerCorpusChunks,
  clusterPaperVectors,
  type ClusteringInputPaper,
} from "../src/library/clustering/clusterer";
import { buildClusteringInput } from "../src/library/clustering/paper-vector";
import type {
  FullTextKnowledgeBaseManifest,
  FullTextKnowledgeBaseStore,
  FullTextPaperDocument,
} from "../src/library/fulltext/knowledge-base";
import { FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION } from "../src/library/fulltext/knowledge-base";

const DIMENSION = 64;

/** Deterministic pseudo-random vector (unit-ish, not normalized). */
function randomVector(seed: number): Float32Array {
  let state = seed;
  const rand = (): number => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  const out = new Float32Array(DIMENSION);
  // Moderate-magnitude noise: in 64 dimensions random pairs top out around
  // cosine 0.4, below the 0.65 relative stop floor against theme edges.
  for (let index = 0; index < DIMENSION; index += 1) out[index] = (rand() - 0.5) * 0.5;
  return out;
}

function blend(aFactor: number, a: Float32Array, bFactor: number, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let index = 0; index < a.length; index += 1) {
    out[index] = a[index]! * aFactor + b[index]! * bFactor;
  }
  return out;
}

function oneHot(dimension: number): Float32Array {
  const out = new Float32Array(DIMENSION);
  out[dimension] = 1;
  return out;
}

/** Unit-norm noise (KB chunk vectors are stored unit-normalized). */
function unitVector(seed: number): Float32Array {
  const raw = randomVector(seed);
  let sum = 0;
  for (const value of raw) sum += value * value;
  const norm = Math.sqrt(sum);
  for (let index = 0; index < raw.length; index += 1) raw[index]! /= norm;
  return raw;
}

const THEME_A = oneHot(0);
const THEME_B = oneHot(1);
const THEME_C = oneHot(2);

function paper(paperKey: string, ...chunks: Float32Array[]): ClusteringInputPaper {
  return { paperKey, chunks };
}

describe("centerCorpusChunks", () => {
  it("does not mutate its input and preserves paper order", () => {
    const input = [
      paper("a", randomVector(1), randomVector(2)),
      paper("b", randomVector(3), randomVector(4), randomVector(5)),
    ];
    const snapshot = input.map((p) => ({
      paperKey: p.paperKey,
      chunks: p.chunks.map((c) => c.slice()),
    }));

    const output = centerCorpusChunks(input);

    expect(output).not.toBe(input);
    expect(output[0]).not.toBe(input[0]);
    expect(output[0]?.chunks[0]).not.toBe(input[0]?.chunks[0]);
    expect(input[0]?.chunks[0]).toEqual(snapshot[0]?.chunks[0]);
    expect(input[1]?.chunks).toEqual(snapshot[1]?.chunks);
    expect(output.map((p) => p.paperKey)).toEqual(["a", "b"]);
  });

  it("produces unit-norm chunks whose corpus mean is ~0", () => {
    const input = [
      paper("a", randomVector(1), randomVector(2)),
      paper("b", randomVector(3)),
      paper("c", randomVector(4), randomVector(5), randomVector(6)),
    ];

    const output = centerCorpusChunks(input);

    const mean = new Float64Array(DIMENSION);
    let count = 0;
    for (const p of output) {
      for (const chunk of p.chunks) {
        let sum = 0;
        for (let index = 0; index < DIMENSION; index += 1) {
          sum += chunk[index]! * chunk[index]!;
          mean[index]! += chunk[index]!;
        }
        expect(Math.sqrt(sum)).toBeCloseTo(1, 6);
        count += 1;
      }
    }
    for (let index = 0; index < DIMENSION; index += 1) {
      // Renormalization rescales each chunk by its own norm, so the corpus
      // mean after centering is only approximately zero — chunks near the
      // mean get amplified by normalization. Bound the drift loosely.
      expect(Math.abs(mean[index]! / count)).toBeLessThan(0.02);
    }
  });

  it("handles an empty corpus by returning the papers untouched", () => {
    const input = [paper("a"), paper("b", new Float32Array(0))];
    expect(centerCorpusChunks(input).map((p) => p.paperKey)).toEqual(["a", "b"]);
  });

  it("is the exact transform the clustering pipeline applies", () => {
    // The exported transform is the single implementation used inside
    // clusterPaperVectors; centering then clustering without a second
    // centering must reproduce the pipeline's own centered clustering.
    // Inputs are unit-norm, matching stored KB vectors.
    const input = [
      paper("a", THEME_A, unitVector(11), unitVector(12)),
      paper("b", THEME_A, unitVector(13), unitVector(14)),
      paper("c", THEME_B, unitVector(15), unitVector(16)),
      paper("d", THEME_C, unitVector(17)),
    ];
    const direct = clusterPaperVectors(input, { centerCorpus: true, minClusterSize: 2 });
    const viaExport = clusterPaperVectors(centerCorpusChunks(input), {
      centerCorpus: false,
      minClusterSize: 2,
    });
    expect(viaExport.clusters.map((c) => c.paperKeys)).toEqual(
      direct.clusters.map((c) => c.paperKeys),
    );
    expect(viaExport.outliers).toEqual(direct.outliers);
  });
});

describe("clusterPaperVectors (SNN)", () => {
  it("clusters same-theme papers and pools outliers", () => {
    const papers = [
      // Theme A: shared theme chunk + per-paper noise.
      paper("p-a1", THEME_A, randomVector(1), randomVector(2)),
      paper("p-a2", THEME_A, randomVector(3), randomVector(4)),
      paper("p-a3", THEME_A, randomVector(5), randomVector(6)),
      // Theme B.
      paper("p-b1", THEME_B, randomVector(7), randomVector(8)),
      paper("p-b2", THEME_B, randomVector(9), randomVector(10)),
      paper("p-b3", THEME_B, randomVector(11), randomVector(12)),
      // Theme C.
      paper("p-c1", THEME_C, randomVector(13), randomVector(14)),
      paper("p-c2", THEME_C, randomVector(15), randomVector(16)),
      // Outlier: no theme chunk, purely noise.
      paper("p-x1", randomVector(17), randomVector(18), randomVector(19)),
      paper("p-x2", randomVector(20), randomVector(21), randomVector(22)),
    ];
    const result = clusterPaperVectors(papers);

    expect(result.clusters.length).toBe(3);
    const byFirst = Object.fromEntries(result.clusters.map((cluster) => [cluster.paperKeys[0], cluster]));
    expect(byFirst["p-a1"]!.paperKeys.sort()).toEqual(["p-a1", "p-a2", "p-a3"]);
    expect(byFirst["p-b1"]!.paperKeys.sort()).toEqual(["p-b1", "p-b2", "p-b3"]);
    expect(byFirst["p-c1"]!.paperKeys.sort()).toEqual(["p-c1", "p-c2"]);
    expect(result.outliers.sort()).toEqual(["p-x1", "p-x2"]);
    // Same-theme members share the identical theme chunk: near-1 confidence.
    expect(byFirst["p-a1"]!.memberConfidence["p-a2"]!).toBeGreaterThan(0.9);
    expect(byFirst["p-a1"]!.id).toMatch(/^cluster-\d+$/);
  });

  it("is deterministic for identical input", () => {
    const papers = [
      paper("p-a1", THEME_A, randomVector(1)),
      paper("p-a2", THEME_A, randomVector(3)),
      paper("p-b1", THEME_B, randomVector(7)),
      paper("p-b2", THEME_B, randomVector(9)),
      paper("p-x1", randomVector(17), randomVector(18)),
    ];
    expect(clusterPaperVectors(papers)).toEqual(clusterPaperVectors(papers));
  });

  it("is independent of input order", () => {
    const papers = [
      paper("p-a1", THEME_A, randomVector(1)),
      paper("p-a2", THEME_A, randomVector(3)),
      paper("p-b1", THEME_B, randomVector(7)),
      paper("p-b2", THEME_B, randomVector(9)),
    ];
    const normal = clusterPaperVectors(papers);
    const shuffled = clusterPaperVectors([...papers].reverse());
    expect(shuffled).toEqual(normal);
  });

  it("buffers clusters below minClusterSize", () => {
    // Five papers, two theme pairs and one isolated theme; without corpus
    // centering the one-hot themes are exactly orthogonal so ties resolve
    // deterministically by paperKey and the isolated paper never links.
    const papers = [
      paper("p-a1", THEME_A),
      paper("p-a2", THEME_A),
      paper("p-b1", THEME_B),
      paper("p-b2", THEME_B),
      paper("p-solo", oneHot(3)),
    ];
    const result = clusterPaperVectors(papers, { minClusterSize: 2, centerCorpus: false });
    expect(result.clusters.length).toBe(2);
    const byFirst = Object.fromEntries(result.clusters.map((cluster) => [cluster.paperKeys[0], cluster]));
    expect(byFirst["p-a1"]!.paperKeys.sort()).toEqual(["p-a1", "p-a2"]);
    expect(byFirst["p-b1"]!.paperKeys.sort()).toEqual(["p-b1", "p-b2"]);
    expect(result.outliers).toEqual(["p-solo"]);
  });

  it("stops merging at the relative similarity gap", () => {
    // Theme A members share the identical theme chunk (score 1); a weak
    // bridge paper blends half theme A with half theme B (centered cosine to
    // theme A around 0.5-0.6 — below the 0.8 tight floor, above the 0.2
    // loose floor). A tight stop ratio (0.8) keeps the bridge out; a loose
    // one (0.2) pulls it in.
    const papers = [
      paper("p-a1", THEME_A, randomVector(1)),
      paper("p-a2", THEME_A, randomVector(3)),
      paper("p-bridge", blend(0.5, THEME_A, 0.5, oneHot(1))),
    ];
    const tight = clusterPaperVectors(papers, { relativeStopRatio: 0.8, minSimilarity: 0 });
    expect(tight.clusters.length).toBe(1);
    expect(tight.outliers).toEqual(["p-bridge"]);
    const loose = clusterPaperVectors(papers, { relativeStopRatio: 0.2 });
    expect(loose.clusters.length).toBe(1);
    expect(loose.clusters[0]!.paperKeys).toHaveLength(3);
  });

  it("returns an empty result for empty input", () => {
    expect(clusterPaperVectors([])).toEqual({ clusters: [], outliers: [] });
  });

  it("rejects invalid options", () => {
    expect(() => clusterPaperVectors([], { minClusterSize: 0 })).toThrow(TypeError);
    expect(() => clusterPaperVectors([], { relativeStopRatio: 1.5 })).toThrow(TypeError);
    expect(() => clusterPaperVectors([], { minSimilarity: -0.1 })).toThrow(TypeError);
  });
});

describe("buildClusteringInput", () => {
  class MemoryStore implements FullTextKnowledgeBaseStore {
    paths = { directory: "kb", manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/m.json.backup" }, papersDirectory: "kb/papers" };
    manifest: FullTextKnowledgeBaseManifest = {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      revision: 1,
      scopeFingerprint: `sha256:${"a".repeat(64)}`,
      identificationFingerprint: `sha256:${"b".repeat(64)}`,
      modelId: "fake",
      dimension: DIMENSION,
      updatedAt: "2026-08-05T00:00:00.000Z",
      papers: {
        "arxiv:1": {
          paperKey: "arxiv:1", status: "ready", modelId: "fake", dimension: DIMENSION,
          textHash: `sha256:${"1".repeat(64)}`, filePaths: ["a.pdf"],
          observationFingerprints: [`sha256:${"c".repeat(64)}`], chunkCount: 1,
          updatedAt: "2026-08-05T00:00:00.000Z",
        },
        "arxiv:2": {
          paperKey: "arxiv:2", status: "failed", modelId: "fake", dimension: DIMENSION,
          filePaths: [], observationFingerprints: [], chunkCount: 0, error: "boom",
          updatedAt: "2026-08-05T00:00:00.000Z",
        },
      },
    };
    private readonly documents = new Map<string, FullTextPaperDocument>();

    async loadManifest(): Promise<FullTextKnowledgeBaseManifest> { return this.manifest; }
    async replaceManifest(): Promise<FullTextKnowledgeBaseManifest> { throw new Error("not used"); }
    async loadPaper(paperKey: string): Promise<FullTextPaperDocument | null> {
      return this.documents.get(paperKey) ?? null;
    }
    async savePaper(document: FullTextPaperDocument): Promise<void> { this.documents.set(document.paperKey, document); }
    async removePaper(): Promise<void> {}
    async removeAll(): Promise<void> {}
  }

  function documentWithChunks(paperKey: string, count: number): FullTextPaperDocument {
    const vectors = new Float32Array(count * DIMENSION);
    for (let index = 0; index < count; index += 1) vectors[index * DIMENSION] = 1;
    return {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      paperKey,
      modelId: "fake",
      dimension: DIMENSION,
      textHash: `sha256:${"0".repeat(64)}`,
      filePaths: ["a.pdf"],
      observationFingerprints: [`sha256:${"1".repeat(64)}`],
      chunks: Array.from({ length: count }, (_, index) => ({ index, page: 1, text: `chunk ${index}` })),
      vectors,
      updatedAt: "2026-08-05T00:00:00.000Z",
    };
  }

  it("collects ready papers with chunk vectors, skipping failures", async () => {
    const store = new MemoryStore();
    await store.savePaper(documentWithChunks("arxiv:1", 3));
    const papers = await buildClusteringInput(store);
    expect(papers.length).toBe(1);
    expect(papers[0]!.paperKey).toBe("arxiv:1");
    expect(papers[0]!.chunks.length).toBe(3);
    expect(papers[0]!.chunks[0]!.length).toBe(DIMENSION);
  });
});
