/**
 * Incremental direction placement: assign newly indexed papers to confirmed
 * directions or the buffer pool.
 *
 * Direction anchor = the direction's representative papers (<= 5, an
 * existing field). Paper-to-direction similarity = the strongest chunk-pair
 * cosine against those anchors — the same best-passage evidence as full-text
 * retrieval and clustering.
 *
 * The similarity space is the corpus-centered chunk space (the KB-wide chunk
 * mean is subtracted, exactly like clustering): raw e5-small cosine is
 * saturated and cannot separate same-theme from cross-theme pairs, while the
 * centered space leaves a measurable gap (see P2 journal).
 *
 * Decision rule (relative, robust to the saturated distribution): the paper
 * attaches to its best direction only when that similarity clears an
 * absolute floor AND beats the second-best direction by a margin; otherwise
 * it goes to the buffer pool. Locked directions still accept attachments
 * (goal: 锁定的方向不参与自动合并/分裂/改名，但新论文仍可归入).
 */

import type { PersonalLibraryConfirmedDirection } from "../personal-library-interest-profile";
import type {
  FullTextKnowledgeBaseStore,
  FullTextPaperDocument,
} from "../fulltext/knowledge-base";
import type { ClusteringInputPaper } from "../clustering/clusterer";

export interface DirectionScore {
  directionId: string;
  /** Centered best-passage similarity between the paper and the anchor. */
  similarity: number;
}

export type PlacementDecision =
  | { kind: "attach"; directionId: string; confidence: number; margin: number }
  | { kind: "buffer"; confidence: number; margin: number };

export interface PlacementOptions {
  /**
   * Absolute floor for an attachment, in the centered cosine space. Below
   * this the paper is buffered even if it is clearly closest to one
   * direction. Default 0.25 (calibrated on real corpora; see journal).
   */
  minSimilarity?: number;
  /**
   * The best direction must beat the second-best by at least this margin,
   * otherwise the assignment is ambiguous and the paper is buffered.
   * Default 0.05.
   */
  minMargin?: number;
}

const DEFAULT_MIN_SIMILARITY = 0.25;
const DEFAULT_MIN_MARGIN = 0.05;

/** Pure decision over already-computed scores; deterministic. */
export function decideIncrementalPlacement(
  paperKey: string,
  scores: readonly DirectionScore[],
  options?: PlacementOptions,
): PlacementDecision {
  const minSimilarity = requireFiniteInRange(options?.minSimilarity, "minSimilarity", DEFAULT_MIN_SIMILARITY, 0, 1);
  const minMargin = requireFiniteInRange(options?.minMargin, "minMargin", DEFAULT_MIN_MARGIN, 0, 1);
  if (scores.length === 0) {
    return { kind: "buffer", confidence: 0, margin: 0 };
  }
  const ranked = [...scores].sort((left, right) =>
    right.similarity !== left.similarity ? right.similarity - left.similarity : left.directionId < right.directionId ? -1 : 1);
  const best = ranked[0]!;
  const second = ranked[1]?.similarity ?? 0;
  const margin = best.similarity - second;
  if (best.similarity >= minSimilarity && margin >= minMargin) {
    return {
      kind: "attach",
      directionId: best.directionId,
      confidence: best.similarity,
      margin,
    };
  }
  return { kind: "buffer", confidence: best.similarity, margin };
}

export interface IncrementalPlacementInput {
  profile: { directions: readonly PersonalLibraryConfirmedDirection[] };
  knowledgeBase: FullTextKnowledgeBaseStore;
  options?: PlacementOptions;
  signal?: AbortSignal;
}

export interface IncrementalPlacementResult {
  /** paperKey → decision for every indexed paper not covered by a direction. */
  placements: Readonly<Record<string, PlacementDecision>>;
  /** paperKeys already covered by a direction's clusterMembers. */
  covered: readonly string[];
  /** Paper keys with no usable vectors (skipped). */
  skipped: readonly string[];
}

/**
 * Orchestration: load all ready papers and the direction anchors from the
 * knowledge base, center the chunk space on the corpus mean, score every
 * uncovered paper against every direction, and decide placement.
 */
export async function suggestIncrementalPlacement(
  input: IncrementalPlacementInput,
): Promise<IncrementalPlacementResult> {
  const papers = await loadClusteringInput(input.knowledgeBase, input.signal);
  // Center the chunk space on the corpus mean (same transform as clustering):
  // raw e5-small cosine is saturated and cannot separate themes.
  centerCorpus(papers);
  const covered = coveredPaperKeys(input.profile.directions);
  const uncovered = papers.filter((paper) => !covered.has(paper.paperKey));

  // Direction anchors: representative papers of every confirmed direction
  // (active, disabled, and locked alike — locked still accepts attachments).
  const anchorByDirection = new Map<string, Float32Array[]>();
  const representativeKeys = new Set<string>();
  for (const direction of input.profile.directions) {
    for (const representative of direction.representatives) {
      representativeKeys.add(representative.paperKey);
    }
  }
  const paperByKey = new Map(papers.map((paper) => [paper.paperKey, paper]));
  for (const direction of input.profile.directions) {
    const anchor: Float32Array[] = [];
    for (const representative of direction.representatives) {
      const paper = paperByKey.get(representative.paperKey);
      if (paper) anchor.push(...paper.chunks);
    }
    if (anchor.length > 0) anchorByDirection.set(direction.id, anchor);
  }

  const placements: Record<string, PlacementDecision> = Object.create(null);
  const skipped: string[] = [];
  for (const paper of uncovered) {
    if (paper.chunks.length === 0) {
      skipped.push(paper.paperKey);
      continue;
    }
    const scores: DirectionScore[] = [];
    for (const [directionId, anchor] of anchorByDirection) {
      scores.push({
        directionId,
        similarity: maxChunkCosine(paper.chunks, anchor),
      });
    }
    placements[paper.paperKey] = decideIncrementalPlacement(paper.paperKey, scores, input.options);
  }
  return { placements, covered: [...covered], skipped };
}

export function coveredPaperKeys(
  directions: readonly PersonalLibraryConfirmedDirection[],
): Set<string> {
  const covered = new Set<string>();
  for (const direction of directions) {
    // Cluster members AND representative papers are direction-covered: a
    // representative is the direction's anchor, not a new arrival.
    for (const member of direction.clusterMembers) covered.add(member.paperKey);
    for (const representative of direction.representatives) covered.add(representative.paperKey);
  }
  return covered;
}

export async function loadClusteringInput(
  store: FullTextKnowledgeBaseStore,
  signal?: AbortSignal,
): Promise<ClusteringInputPaper[]> {
  const manifest = await store.loadManifest();
  const papers: ClusteringInputPaper[] = [];
  for (const paperKey of Object.keys(manifest.papers).sort()) {
    signal?.throwIfAborted();
    const record = manifest.papers[paperKey];
    if (!record || record.status !== "ready") continue;
    const document = await store.loadPaper(paperKey);
    if (!document) continue;
    const chunks = chunkVectors(document);
    if (chunks.length === 0) continue;
    papers.push({ paperKey, chunks });
    // Yield per paper so host UIs stay responsive while a large library's
    // vectors are decoded and parsed (the incremental auto-trigger runs this
    // right after indexing); harmless on Node hosts.
    await yieldToEventLoop();
  }
  return papers;
}

/** Row-major chunk vectors of a paper, capped like clustering input. */
function chunkVectors(document: FullTextPaperDocument): Float32Array[] {
  const chunks: Float32Array[] = [];
  for (let index = 0; index < document.chunks.length && index < 80; index += 1) {
    const offset = index * document.dimension;
    const chunk = document.vectors.subarray(offset, offset + document.dimension);
    if (chunk.length === 0) continue;
    chunks.push(chunk);
  }
  return chunks;
}

/** Subtract the corpus chunk mean (in place) and re-normalize. */
function centerCorpus(papers: ClusteringInputPaper[]): void {
  let count = 0;
  const dimension = papers[0]?.chunks[0]?.length ?? 0;
  const mean = new Float64Array(dimension);
  for (const paper of papers) {
    for (const chunk of paper.chunks) {
      for (let index = 0; index < dimension; index += 1) mean[index]! += chunk[index] ?? 0;
      count += 1;
    }
  }
  if (count === 0) return;
  for (let index = 0; index < dimension; index += 1) mean[index]! /= count;
  for (const paper of papers) {
    paper.chunks = paper.chunks.map((chunk) => {
      const out = new Float32Array(chunk.length);
      for (let index = 0; index < chunk.length; index += 1) {
        out[index] = (chunk[index] ?? 0) - mean[index]!;
      }
      return normalizedChunk(out);
    });
  }
}

function normalizedChunk(chunk: Float32Array): Float32Array {
  let norm = 0;
  for (const value of chunk) norm += value * value;
  norm = Math.sqrt(norm);
  if (norm === 0) return chunk.slice();
  const out = new Float32Array(chunk.length);
  for (let index = 0; index < chunk.length; index += 1) {
    out[index] = (chunk[index] ?? 0) / norm;
  }
  return out;
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

function dot(a: Float32Array, b: Float32Array): number {
  const length = Math.min(a.length, b.length);
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    sum += (a[index] ?? 0) * (b[index] ?? 0);
  }
  return sum;
}

function requireFiniteInRange(
  value: number | undefined,
  name: string,
  fallback: number,
  min: number,
  max: number,
): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new TypeError(`${name} must be a finite number in [${min}, ${max}]`);
  }
  return value;
}

/** Let queued host events run before continuing a long vector load. */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof setTimeout === "function") setTimeout(resolve, 0);
    else resolve();
  });
}
