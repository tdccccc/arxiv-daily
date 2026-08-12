/**
 * Pool reclustering: papers that no direction claimed may still form
 * strong internal groups — those are new-direction candidates; the rest stay
 * buffered. Called after placement in the same centered chunk space (the
 * caller centers the corpus once and shares the papers).
 *
 * Drift reference: each new cluster is scored against every confirmed
 * direction's anchors so the LLM diff stage can tell a genuinely new theme
 * from an evolved (drifted) existing one.
 */

import { clusterPaperVectors, type ClusteringInputPaper, type ClusteringOptions } from "../clustering/clusterer";
import type { PersonalLibraryConfirmedDirection } from "../personal-library-interest-profile";

export interface NewClusterCandidate {
  clusterId: string;
  paperKeys: string[];
  memberConfidence: Readonly<Record<string, number>>;
  /** Nearest direction anchors with centered best-passage similarity. */
  nearestDirection: Array<{ directionId: string; similarity: number }>;
}

export interface ReclusterPoolResult {
  candidates: NewClusterCandidate[];
  /** Papers that stayed unclustered (still the buffer pool). */
  stillPooled: string[];
}

export interface ReclusterPoolOptions extends ClusteringOptions {
  /** Paper keys to consider (the buffer pool). */
  poolPaperKeys: readonly string[];
  /** Directions whose anchors provide the drift reference. */
  directions: readonly PersonalLibraryConfirmedDirection[];
}

/**
 * Pure reclustering over already-centered papers. Deterministic.
 *
 * Clustering runs on one centroid vector per paper instead of the full chunk
 * set: the buffer pool can hold hundreds of papers and chunk-level pairwise
 * similarity is quadratic in chunks (with the per-paper 80-chunk cap that is
 * ~1e4 chunks — minutes of synchronous CPU that freezes host UIs). The
 * centroid of a paper's chunks in the centered space is its theme mean;
 * multi-theme papers land in the middle, which the LLM-diff review stage
 * catches. The drift reference still uses the full chunk sets.
 */
export function reclusterPool(
  papers: readonly ClusteringInputPaper[],
  options: ReclusterPoolOptions,
): ReclusterPoolResult {
  const poolSet = new Set(options.poolPaperKeys);
  const poolPapers = papers
    .filter((paper) => poolSet.has(paper.paperKey))
    .map(centroidPaper);
  const clustering = clusterPaperVectors(poolPapers, {
    minClusterSize: options.minClusterSize,
    centerCorpus: false, // caller already centered the shared space
    minSimilarity: options.minSimilarity,
    relativeStopRatio: options.relativeStopRatio,
  });

  // Anchor chunks per direction for the drift reference.
  const paperByKey = new Map(papers.map((paper) => [paper.paperKey, paper]));
  const anchors = new Map<string, Float32Array[]>();
  for (const direction of options.directions) {
    const chunks: Float32Array[] = [];
    for (const representative of direction.representatives) {
      const paper = paperByKey.get(representative.paperKey);
      if (paper) chunks.push(...paper.chunks);
    }
    if (chunks.length > 0) anchors.set(direction.id, chunks);
  }

  const candidates: NewClusterCandidate[] = [];
  for (const cluster of clustering.clusters) {
    const nearestDirection: Array<{ directionId: string; similarity: number }> = [];
    for (const [directionId, anchor] of anchors) {
      let best = -Infinity;
      for (const paperKey of cluster.paperKeys) {
        const paper = paperByKey.get(paperKey);
        if (!paper) continue;
        const score = maxChunkCosine(paper.chunks, anchor);
        if (score > best) best = score;
      }
      if (best > -Infinity) nearestDirection.push({ directionId, similarity: best });
    }
    nearestDirection.sort((left, right) => right.similarity - left.similarity);
    candidates.push({
      clusterId: cluster.id,
      paperKeys: cluster.paperKeys,
      memberConfidence: cluster.memberConfidence,
      nearestDirection,
    });
  }
  return { candidates, stillPooled: [...clustering.outliers] };
}

function maxChunkCosine(a: readonly Float32Array[], b: readonly Float32Array[]): number {
  let best = -Infinity;
  for (const va of a) {
    for (const vb of b) {
      const score = dot(va, vb);
      if (score > best) best = score;
      if (best >= 1) return 1;
    }
  }
  return best;
}

/** Paper-level centroid: mean of the paper's chunks, renormalized (see the module comment). */
function centroidPaper(paper: ClusteringInputPaper): ClusteringInputPaper {
  const chunks = paper.chunks.filter((chunk) => chunk.length > 0);
  if (chunks.length === 0) return { paperKey: paper.paperKey, chunks: [] };
  const dimension = chunks[0]!.length;
  const mean = new Float32Array(dimension);
  for (const chunk of chunks) {
    for (let index = 0; index < dimension; index += 1) {
      mean[index] = (mean[index] ?? 0) + (chunk[index] ?? 0) / chunks.length;
    }
  }
  let norm = 0;
  for (const value of mean) norm += value * value;
  if (norm > 0) {
    const scale = 1 / Math.sqrt(norm);
    for (let index = 0; index < dimension; index += 1) mean[index] = (mean[index] ?? 0) * scale;
  }
  return { paperKey: paper.paperKey, chunks: [mean] };
}

function dot(a: Float32Array, b: Float32Array): number {
  const length = Math.min(a.length, b.length);
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    sum += (a[index] ?? 0) * (b[index] ?? 0);
  }
  return sum;
}
