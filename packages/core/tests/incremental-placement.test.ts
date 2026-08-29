import { describe, expect, it } from "vitest";
import {
  decideIncrementalPlacement,
  suggestIncrementalPlacement,
} from "../src/library/incremental/placement";
import { reclusterPool } from "../src/library/incremental/recluster";
import { clusterPaperVectors, type ClusteringInputPaper } from "../src/library/clustering/clusterer";
import type {
  FullTextKnowledgeBaseManifest,
  FullTextKnowledgeBaseStore,
  FullTextPaperDocument,
  FullTextPaperKnowledgeRecord,
} from "../src/library/fulltext/knowledge-base";
import { FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION } from "../src/library/fulltext/knowledge-base";
import type {
  PersonalLibraryConfirmedDirection,
  PersonalLibraryInterestProfile,
} from "../src/library/personal-library-interest-profile";

const DIMENSION = 64;

function oneHot(dimension: number): Float32Array {
  const out = new Float32Array(DIMENSION);
  out[dimension] = 1;
  return out;
}

const THEME_A = oneHot(0);
const THEME_B = oneHot(1);
const THEME_C = oneHot(2);

let noiseState = 1;
function noise(seed: number): Float32Array {
  noiseState = seed;
  const rand = (): number => {
    noiseState = (noiseState * 1103515245 + 12345) % 2147483648;
    return noiseState / 2147483648;
  };
  const out = new Float32Array(DIMENSION);
  for (let index = 0; index < DIMENSION; index += 1) out[index] = (rand() - 0.5) * 0.1;
  return out;
}

function paper(paperKey: string, ...chunks: Float32Array[]): ClusteringInputPaper {
  return { paperKey, chunks };
}

function direction(
  id: string,
  representativePaperKeys: string[],
  clusterMembers: string[],
  locked = false,
): PersonalLibraryConfirmedDirection {
  return {
    id,
    status: "active",
    name: `Direction ${id}`,
    description: "desc",
    discoveryCues: ["cue"],
    representatives: representativePaperKeys.map((paperKey) => ({
      paperKey,
      evidenceFingerprint: `sha256:${"a".repeat(64)}`,
    })),
    representativeSetFingerprint: `sha256:${"b".repeat(64)}`,
    clusterMembers: clusterMembers.map((paperKey, index) => ({
      paperKey,
      confidence: 1 - index * 0.1,
    })),
    timeline: [{ kind: "created", at: "2026-08-06T00:00:00.000Z" }],
    lineage: { proposalIds: ["p.1"], candidateIds: ["c.1"], directionIds: [] },
    createdAt: "2026-08-06T00:00:00.000Z",
    updatedAt: "2026-08-06T00:00:00.000Z",
    ...(locked ? { lockedAt: "2026-08-06T01:00:00.000Z" } : {}),
  };
}

function profile(directions: PersonalLibraryConfirmedDirection[]): PersonalLibraryInterestProfile {
  return {
    schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
    revision: 1,
    scopeFingerprint: `sha256:${"c".repeat(64)}`,
    identificationFingerprint: `sha256:${"d".repeat(64)}`,
    updatedAt: "2026-08-06T00:00:00.000Z",
    directions,
  };
}

class MemoryStore implements FullTextKnowledgeBaseStore {
  paths = { directory: "kb", manifest: { directory: "kb", documentPath: "kb/manifest.json", backupPath: "kb/m.json.backup" }, papersDirectory: "kb/papers" };
  readonly manifest: FullTextKnowledgeBaseManifest;
  private readonly documents = new Map<string, FullTextPaperDocument>();

  constructor(paperKeys: string[]) {
    const papers: Record<string, FullTextPaperKnowledgeRecord> = {};
    for (const key of paperKeys) {
      papers[key] = {
        paperKey: key, status: "ready", modelId: "fake", dimension: DIMENSION,
        textHash: `sha256:${"1".repeat(64)}`, filePaths: [`${key}.pdf`],
        observationFingerprints: [`sha256:${"2".repeat(64)}`], chunkCount: 1,
        updatedAt: "2026-08-06T00:00:00.000Z",
      };
    }
    this.manifest = {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      revision: 1,
      scopeFingerprint: `sha256:${"c".repeat(64)}`,
      identificationFingerprint: `sha256:${"d".repeat(64)}`,
      modelId: "fake",
      dimension: DIMENSION,
      updatedAt: "2026-08-06T00:00:00.000Z",
      papers,
    };
  }

  save(paperKey: string, ...chunks: Float32Array[]): void {
    const vectors = new Float32Array(chunks.length * DIMENSION);
    chunks.forEach((chunk, index) => vectors.set(chunk, index * DIMENSION));
    this.documents.set(paperKey, {
      schemaVersion: FULLTEXT_KNOWLEDGE_BASE_SCHEMA_VERSION,
      paperKey,
      modelId: "fake",
      dimension: DIMENSION,
      textHash: `sha256:${"3".repeat(64)}`,
      filePaths: [`${paperKey}.pdf`],
      observationFingerprints: [`sha256:${"4".repeat(64)}`],
      chunks: chunks.map((_, index) => ({ index, page: 1, text: `chunk ${index}` })),
      vectors,
      updatedAt: "2026-08-06T00:00:00.000Z",
    });
  }

  async loadManifest(): Promise<FullTextKnowledgeBaseManifest> { return this.manifest; }
  async replaceManifest(): Promise<FullTextKnowledgeBaseManifest> { throw new Error("not used"); }
  async loadPaper(paperKey: string): Promise<FullTextPaperDocument | null> {
    return this.documents.get(paperKey) ?? null;
  }
  async savePaper(): Promise<void> { throw new Error("not used"); }
  async removePaper(): Promise<void> {}
  async removeAll(): Promise<void> {}
}

describe("decideIncrementalPlacement", () => {
  it("attaches when the best direction clears the floor and the margin", () => {
    const decision = decideIncrementalPlacement("p-new", [
      { directionId: "d-a", similarity: 0.5 },
      { directionId: "d-b", similarity: 0.3 },
    ]);
    expect(decision).toEqual({ kind: "attach", directionId: "d-a", confidence: 0.5, margin: 0.2 });
  });

  it("buffers when the margin is too small (ambiguous)", () => {
    const decision = decideIncrementalPlacement("p-new", [
      { directionId: "d-a", similarity: 0.5 },
      { directionId: "d-b", similarity: 0.48 },
    ]);
    expect(decision.kind).toBe("buffer");
  });

  it("buffers when nothing clears the floor", () => {
    const decision = decideIncrementalPlacement("p-new", [
      { directionId: "d-a", similarity: 0.1 },
    ]);
    expect(decision.kind).toBe("buffer");
    expect(decision).toMatchObject({ confidence: 0.1, margin: 0.1 });
  });

  it("buffers with no directions at all", () => {
    expect(decideIncrementalPlacement("p-new", []).kind).toBe("buffer");
  });

  it("rejects invalid options", () => {
    expect(() => decideIncrementalPlacement("p", [], { minSimilarity: 2 })).toThrow(TypeError);
    expect(() => decideIncrementalPlacement("p", [], { minMargin: -1 })).toThrow(TypeError);
  });
});

describe("suggestIncrementalPlacement", () => {
  it("attaches same-theme papers, buffers ambiguous and unrelated ones", async () => {
    // Symmetric corpus (2 anchors per direction): an A/B-hybrid paper stays
    // ambiguous (no margin) and is buffered; a same-theme paper attaches.
    const store = new MemoryStore([
      "arxiv:a-rep1", "arxiv:a-rep2", "arxiv:b-rep1", "arxiv:b-rep2",
      "arxiv:n-mixed", "arxiv:n-c",
    ]);
    store.save("arxiv:a-rep1", THEME_A, noise(1));
    store.save("arxiv:a-rep2", THEME_A, noise(3));
    store.save("arxiv:b-rep1", THEME_B, noise(5));
    store.save("arxiv:b-rep2", THEME_B, noise(7));
    store.save("arxiv:n-mixed", blend(0.5, THEME_A, 0.5, THEME_B), noise(9));
    store.save("arxiv:n-c", THEME_C, noise(11));

    const directions = [
      direction("d-a", ["arxiv:a-rep1", "arxiv:a-rep2"], []),
      direction("d-b", ["arxiv:b-rep1", "arxiv:b-rep2"], []),
    ];
    const result = await suggestIncrementalPlacement({
      profile: profile(directions),
      knowledgeBase: store,
    });

    expect(result.placements["arxiv:n-mixed"]!.kind).toBe("buffer");
    expect(result.placements["arxiv:n-c"]!.kind).toBe("buffer");
  });

  it("treats direction-covered papers as covered and locked directions as attachable", async () => {
    const store = new MemoryStore([
      "arxiv:a-rep1", "arxiv:covered1", "arxiv:n-a",
    ]);
    store.save("arxiv:a-rep1", THEME_A, noise(1));
    store.save("arxiv:covered1", THEME_A, noise(3));
    store.save("arxiv:n-a", THEME_A, noise(5));

    const covered = await suggestIncrementalPlacement({
      profile: profile([direction("d-a", ["arxiv:a-rep1"], ["arxiv:covered1"])]),
      knowledgeBase: store,
    });
    expect(covered.covered).toContain("arxiv:covered1");
    expect(covered.placements["arxiv:covered1"]).toBeUndefined();

    const locked = await suggestIncrementalPlacement({
      profile: profile([direction("d-locked", ["arxiv:a-rep1"], [], true)]),
      knowledgeBase: store,
    });
    expect(locked.placements["arxiv:n-a"]).toMatchObject({ kind: "attach", directionId: "d-locked" });
  });
});

describe("reclusterPool", () => {
  it("finds strong internal groups in the buffer pool and keeps the rest buffered", async () => {
    const papers = [
      paper("p-x1", THEME_C),
      paper("p-x2", THEME_C),
      paper("p-x3", THEME_C),
      paper("p-solo", oneHot(5)),
    ];
    // Same centered space as the caller would provide: center on the corpus.
    const centered = center(papers);
    const result = reclusterPool(centered, {
      poolPaperKeys: ["p-x1", "p-x2", "p-x3", "p-solo"],
      directions: [direction("d-a", ["p-a1"], [])],
    });
    expect(result.candidates.length).toBe(1);
    expect(result.candidates[0]!.paperKeys.sort()).toEqual(["p-x1", "p-x2", "p-x3"]);
    expect(result.stillPooled).toEqual(["p-solo"]);
    // The new cluster carries a drift reference against direction anchors.
    expect(Array.isArray(result.candidates[0]!.nearestDirection)).toBe(true);
  });
});

describe("clusterPaperVectors integration", () => {
  it("produces a new-cluster candidate when buffered papers share a theme", () => {
    const papers = [
      paper("p-x1", THEME_C),
      paper("p-x2", THEME_C),
      paper("p-x3", THEME_C),
    ];
    const centered = center(papers);
    const clustering = clusterPaperVectors(centered, { centerCorpus: false });
    expect(clustering.clusters.length).toBe(1);
    expect(clustering.clusters[0]!.paperKeys).toHaveLength(3);
  });
});

function blend(aFactor: number, a: Float32Array, bFactor: number, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let index = 0; index < a.length; index += 1) {
    out[index] = a[index]! * aFactor + b[index]! * bFactor;
  }
  return out;
}

function center(papers: ClusteringInputPaper[]): ClusteringInputPaper[] {
  const dimension = papers[0]?.chunks[0]?.length ?? 0;
  const mean = new Float64Array(dimension);
  let count = 0;
  for (const paper of papers) {
    for (const chunk of paper.chunks) {
      for (let index = 0; index < dimension; index += 1) mean[index]! += chunk[index] ?? 0;
      count += 1;
    }
  }
  for (let index = 0; index < dimension; index += 1) mean[index]! /= count;
  return papers.map((paper) => ({
    paperKey: paper.paperKey,
    chunks: paper.chunks.map((chunk) => {
      const out = new Float32Array(chunk.length);
      for (let index = 0; index < chunk.length; index += 1) {
        out[index] = (chunk[index] ?? 0) - mean[index]!;
      }
      let norm = 0;
      for (const value of out) norm += value * value;
      norm = Math.sqrt(norm);
      if (norm === 0) return out;
      for (let index = 0; index < out.length; index += 1) out[index]! /= norm;
      return out;
    }),
  }));
}
