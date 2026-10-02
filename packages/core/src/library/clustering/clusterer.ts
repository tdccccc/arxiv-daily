/**
 * Deterministic average-linkage clustering with an outlier pool.
 *
 * L2 reshape history (2026-08-06): absolute-threshold centroid clustering
 * and a mutual-top-k SNN graph were both measured unusable on real e5-small
 * embeddings — the cosine distribution is saturated (unrelated academic
 * papers score 0.85+ on raw vectors) and weak best-passage matches bridge
 * theme clusters in rank-based graphs. Corpus centering plus best-passage
 * similarity fixed both, and single linkage on the ranked edge order was the
 * first merging rule tried on top of them.
 *
 * Measured again on the frozen 207-paper corpus (2026-09-05), single linkage
 * did not survive. Two properties of it were doing the damage:
 *
 *   - It chains. A likes B and B likes C welds A to C however unrelated they
 *     are, so one weak bridge merges two themes. At every stop ratio below
 *     the shattering point, 176 of 189 papers landed in one component.
 *   - The stop floor was a fraction of the single strongest edge, and that
 *     edge is a near-duplicate pair. The whole scale hung on one number:
 *     max 0.786 against a p99 of 0.424 and a median of 0.027, so a ratio of
 *     0.6 cut above the 99th percentile and still produced a 24-paper blob.
 *
 * Together they left no usable setting: below the transition one blob, above
 * it fragments plus most of the library in the outlier pool. The evidence was
 * never the problem — three photometric-redshift method papers scored
 * 0.36–0.47 against a 0.027 median, and 0.017 against an unrelated survey
 * paper — the merging rule was.
 *
 * So:
 *
 *   1. corpus-level centering of chunk vectors (suppress the shared academic
 *      language direction that dominates raw cosine);
 *   2. paper-to-paper similarity = strongest chunk-pair cosine (best-passage
 *      evidence, same semantics as full-text retrieval);
 *   3. agglomerative merging by AVERAGE linkage: two groups merge on the mean
 *      similarity across every pair between them, so a single strong pair
 *      cannot drag two themes together;
 *   4. the floor is a QUANTILE of the corpus's own pairwise similarities, not
 *      a fraction of its strongest pair — it still adapts to saturated or
 *      low-scoring distributions, without one near-duplicate setting the
 *      scale for everything else;
 *   5. groups below `minClusterSize` land in the outlier pool (the P3
 *      buffering source);
 *   6. member confidence = the member's strongest in-cluster edge, clamped
 *      to [0, 1] for the proposal schema.
 *
 * Determinism: input order is normalized (paperKey sort), every tie resolves
 * by the lowest member index, and clusters are enumerated in paperKey order.
 * Same input, same output, always.
 *
 * Cost: O(n^2 * c^2) cosine evaluations to build the matrix, then merging
 * with a nearest-neighbour array — O(n^2) in the typical case for n papers.
 * For a personal library (hundreds of papers, one or two chunks each after
 * ADR 0013) this is well under a second; clustering is a low-frequency,
 * user-triggered operation, not on the daily report path.
 */

export interface ClusteringInputPaper {
  paperKey: string;
  /** Paper chunk vectors, one Float32Array per chunk. */
  chunks: readonly Float32Array[];
}

export interface PaperCluster {
  /** Deterministic id: `cluster-<ordinal>` in final cluster order. */
  id: string;
  /** Member paper keys in ascending order. */
  paperKeys: string[];
  /**
   * Strength of each member's strongest link inside the cluster, in the
   * corpus-centered cosine space, clamped to [0, 1].
   */
  memberConfidence: Readonly<Record<string, number>>;
}

export interface ClusteringResult {
  clusters: PaperCluster[];
  /** Paper keys that never joined any cluster (the buffering pool). */
  outliers: string[];
}

export interface ClusteringOptions {
  /**
   * Clusters smaller than this are moved to the outlier pool. Default 2.
   */
  minClusterSize?: number;
  /**
   * Subtract the corpus chunk mean before scoring. Default true.
   */
  centerCorpus?: boolean;
  /**
   * Absolute floor for merging an edge, in the (possibly centered) cosine
   * space; edges below it never merge. Default 0 (the relative stop does the
   * work; raise to 0.1+ to hard-exclude weak best-passage coincidences).
   */
  minSimilarity?: number;
  /**
   * Merging stops when the closest remaining pair of groups falls below this
   * quantile of the corpus's own pairwise similarities. 0.95 keeps the top
   * 5% of pairs as merge evidence; higher is tighter (more, smaller groups
   * and a larger outlier pool), lower is looser.
   *
   * A quantile rather than a fraction of the strongest pair: the strongest
   * pair is typically a near-duplicate, and anchoring on it let one paper
   * decide the scale for the whole library.
   */
  similarityQuantile?: number;
}

const DEFAULT_MIN_CLUSTER_SIZE = 2;
/**
 * Treats the bottom 80% of pairs as coincidence and lets the rest be merge
 * evidence — average linkage still refuses most of them, because a group only
 * grows while the mean similarity across every pair between two groups holds
 * up.
 *
 * On a fixture of three unambiguous themes plus noise, every value from 0.70
 * to 0.86 recovers exactly those three and pools the noise; below it themes
 * merge into each other, above it the smallest theme is lost. 0.8 is the
 * middle of that plateau.
 *
 * It is a granularity knob, not a universal constant: the right value tracks
 * what share of a corpus's pairs are genuinely related, which shrinks as a
 * library grows. The proposal's tight evidence-grouping pass uses its measured
 * cut; topic and direction breadth is decided later by semantic organization.
 */
const DEFAULT_SIMILARITY_QUANTILE = 0.8;

export function clusterPaperVectors(
  input: readonly ClusteringInputPaper[],
  options?: ClusteringOptions,
): ClusteringResult {
  const minClusterSize = requirePositiveInteger(options?.minClusterSize, "minClusterSize", DEFAULT_MIN_CLUSTER_SIZE);
  const centerCorpus = options?.centerCorpus ?? true;
  const minSimilarity = requireFiniteInRange(options?.minSimilarity, "minSimilarity", 0, 0, 1);
  const similarityQuantile = requireFiniteInRange(
    options?.similarityQuantile, "similarityQuantile", DEFAULT_SIMILARITY_QUANTILE, 0, 1,
  );

  const papers = input
    .filter((paper) => paper.chunks.some((chunk) => chunk.length > 0))
    .map((paper) => ({ paperKey: paper.paperKey, chunks: paper.chunks.map(normalizedChunk) }))
    .sort((left, right) => (left.paperKey < right.paperKey ? -1 : left.paperKey > right.paperKey ? 1 : 0));
  if (papers.length === 0) return { clusters: [], outliers: [] };

  // Corpus-level centering is the one transform host incremental passes also
  // need; it is exported as the non-mutating `centerCorpusChunks` so both
  // paths share a single implementation and cannot drift.
  let centered: ClusteringInputPaper[] = papers;
  if (centerCorpus) centered = centerCorpusChunks(papers);

  // Pairwise strongest chunk evidence (symmetric), as mergeable edges.
  const n = centered.length;
  const similarity = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    similarity[i * n + i] = 1;
    for (let j = i + 1; j < n; j += 1) {
      const score = maxChunkCosine(centered[i]!.chunks, centered[j]!.chunks);
      similarity[i * n + j] = score;
      similarity[j * n + i] = score;
    }
  }
  // The floor is a quantile of this corpus's own pairwise similarities, so a
  // saturated library and a sparse one both get a cut in the same place
  // relative to their own evidence.
  const pairScores: number[] = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) pairScores.push(similarity[i * n + j]!);
  }
  pairScores.sort((left, right) => left - right);
  const quantileFloor = pairScores.length === 0
    ? -Infinity
    : pairScores[Math.min(pairScores.length - 1, Math.floor(pairScores.length * similarityQuantile))]!;
  // `minSimilarity` is an absolute veto on top of the adaptive floor: a corpus
  // where even the top pairs are coincidental must not cluster just because
  // its own quantile says they are its best.
  const stop = Math.max(quantileFloor, minSimilarity);

  // Average-linkage agglomeration. `groupSimilarity` holds the mean similarity
  // between live groups and is updated by the Lance-Williams rule for average
  // linkage, so it never needs recomputing from members.
  const groupSimilarity = Float64Array.from(similarity);
  const size = new Int32Array(n).fill(1);
  const alive = new Uint8Array(n).fill(1);
  const membersOf: number[][] = Array.from({ length: n }, (_, index) => [index]);

  // Nearest-neighbour array: the best live partner of each live group. Finding
  // the global best pair is then a scan, and only groups whose partner was
  // just consumed need recomputing — O(n^2) in the typical case instead of the
  // O(n^3) a full rescan per merge would cost.
  const partner = new Int32Array(n).fill(-1);
  const partnerScore = new Float64Array(n).fill(-Infinity);
  const refreshPartner = (a: number): void => {
    let bestScore = -Infinity;
    let best = -1;
    for (let b = 0; b < n; b += 1) {
      if (b === a || !alive[b]) continue;
      const score = groupSimilarity[a * n + b]!;
      // Ties go to the lowest index, which is the lowest paperKey: the input
      // was sorted, so the same corpus always merges in the same order.
      if (score > bestScore) { bestScore = score; best = b; }
    }
    partner[a] = best;
    partnerScore[a] = bestScore;
  };
  for (let a = 0; a < n; a += 1) refreshPartner(a);

  for (let merges = 0; merges < n - 1; merges += 1) {
    let left = -1;
    let bestScore = -Infinity;
    for (let a = 0; a < n; a += 1) {
      if (!alive[a] || partner[a]! < 0) continue;
      if (partnerScore[a]! > bestScore) { bestScore = partnerScore[a]!; left = a; }
    }
    if (left < 0 || bestScore < stop) break;
    const right = partner[left]!;
    // Keep the lower index alive so cluster order stays paperKey order.
    const kept = Math.min(left, right);
    const dropped = Math.max(left, right);

    for (let other = 0; other < n; other += 1) {
      if (other === kept || other === dropped || !alive[other]) continue;
      const merged = (size[kept]! * groupSimilarity[kept * n + other]!
        + size[dropped]! * groupSimilarity[dropped * n + other]!)
        / (size[kept]! + size[dropped]!);
      groupSimilarity[kept * n + other] = merged;
      groupSimilarity[other * n + kept] = merged;
    }
    membersOf[kept] = [...membersOf[kept]!, ...membersOf[dropped]!];
    size[kept] = size[kept]! + size[dropped]!;
    alive[dropped] = 0;
    partner[dropped] = -1;
    partnerScore[dropped] = -Infinity;

    refreshPartner(kept);
    for (let a = 0; a < n; a += 1) {
      if (!alive[a] || a === kept) continue;
      if (partner[a] === dropped || partner[a] === kept) refreshPartner(a);
    }
  }

  const byRoot = new Map<number, number[]>();
  for (let index = 0; index < n; index += 1) {
    if (!alive[index]) continue;
    byRoot.set(index, membersOf[index]!.slice().sort((a, b) => a - b));
  }
  const roots = [...byRoot.keys()].sort((a, b) => a - b);
  const clusters: PaperCluster[] = [];
  const outliers: string[] = [];
  for (const root of roots) {
    const members = byRoot.get(root)!;
    if (members.length < minClusterSize) {
      outliers.push(...members.map((index) => papers[index]!.paperKey));
      continue;
    }
    // Member confidence = strongest in-cluster edge, clamped to [0, 1].
    const confidence: Record<string, number> = {};
    for (const member of members) {
      let best = -Infinity;
      for (const other of members) {
        if (other === member) continue;
        const score = similarity[member * n + other]!;
        if (score > best) best = score;
      }
      confidence[papers[member]!.paperKey] = best === -Infinity ? 0 : Math.min(1, Math.max(0, best));
    }
    clusters.push({
      id: `cluster-${String(clusters.length + 1).padStart(2, "0")}`,
      paperKeys: members.map((index) => papers[index]!.paperKey),
      memberConfidence: confidence,
    });
  }
  outliers.sort();
  return { clusters, outliers };
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

function normalizedChunk(chunk: Float32Array): Float32Array {
  const norm = Math.sqrt(sumOfSquares(chunk));
  if (norm === 0) return chunk.slice();
  const out = new Float32Array(chunk.length);
  for (let index = 0; index < chunk.length; index += 1) {
    out[index] = (chunk[index] ?? 0) / norm;
  }
  return out;
}

/**
 * Corpus-level centering of chunk vectors (public, non-mutating): subtract
 * the corpus chunk mean (Float64 accumulation, input order preserved), then
 * renormalize every chunk. Host incremental passes use this instead of
 * mirroring the transform, so clustering and placement cannot drift apart.
 * Returns new paper objects with new chunk arrays; the input is never
 * mutated.
 */
export function centerCorpusChunks(
  papers: readonly ClusteringInputPaper[],
): ClusteringInputPaper[] {
  let count = 0;
  const dimension = papers[0]?.chunks[0]?.length ?? 0;
  const mean = new Float64Array(dimension);
  for (const paper of papers) {
    for (const chunk of paper.chunks) {
      for (let index = 0; index < dimension; index += 1) mean[index]! += chunk[index] ?? 0;
      count += 1;
    }
  }
  if (count === 0) return [...papers];
  for (let index = 0; index < dimension; index += 1) mean[index]! /= count;
  return papers.map((paper) => ({
    paperKey: paper.paperKey,
    chunks: paper.chunks.map((chunk) => {
      const out = new Float32Array(chunk.length);
      for (let index = 0; index < chunk.length; index += 1) {
        out[index] = (chunk[index] ?? 0) - mean[index]!;
      }
      return normalizedChunk(out);
    }),
  }));
}

function sumOfSquares(values: Float32Array): number {
  let sum = 0;
  for (const value of values) sum += value * value;
  return sum;
}

function requirePositiveInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new TypeError(`clusterPaperVectors: ${name} must be a positive integer`);
  }
  return value;
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
    throw new TypeError(`clusterPaperVectors: ${name} must be a finite number in [${min}, ${max}]`);
  }
  return value;
}
