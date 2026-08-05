/**
 * Deterministic single-linkage clustering with an outlier pool — the
 * HDBSCAN equivalent for the knowledge base (goal: "HDBSCAN 或等价").
 *
 * L2 reshape history (2026-08-06): absolute-threshold centroid clustering
 * and a mutual-top-k SNN graph were both measured unusable on real e5-small
 * embeddings — the cosine distribution is saturated (unrelated academic
 * papers score 0.85+ on raw vectors) and weak best-passage matches bridge
 * theme clusters in rank-based graphs. Single-linkage clustering on the
 * ranked edge order is robust to both:
 *
 *   1. corpus-level centering of chunk vectors (suppress the shared academic
 *      language direction that dominates raw cosine);
 *   2. paper-to-paper similarity = strongest chunk-pair cosine (best-passage
 *      evidence, same semantics as full-text retrieval);
 *   3. edges sorted by descending similarity, merged in Kruskal fashion
 *      (union-find) while the edge is at or above the stop floor — the floor
 *      is RELATIVE (fraction of the strongest edge) so it adapts to
 *      saturated or low-scoring distributions instead of assuming an
 *      absolute semantic scale;
 *   4. the resulting components are the clusters; components below
 *      `minClusterSize` land in the outlier pool (the P3 buffering source);
 *   5. member confidence = the member's strongest in-cluster edge, clamped
 *      to [0, 1] for the proposal schema.
 *
 * Determinism: input order is normalized (paperKey sort), edge ties resolve
 * by paperKey index, and components are enumerated in paperKey order. Same
 * input, same output, always.
 *
 * Cost: O(n^2 * c^2) cosine evaluations + O(n^2 log n) edge sorting (n
 * papers, c chunks each). For a personal library (hundreds of papers, ~30
 * chunks each) this is seconds — clustering is a low-frequency,
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
   * Merging stops when the next edge falls below this fraction of the
   * strongest edge in the corpus — the "similarity gap" between theme
   * density and coincidental matches. Default 0.65 (tuned on a real
   * heterogeneous corpus: tight enough to keep weak bridges out, loose
   * enough to keep strong themes whole).
   */
  relativeStopRatio?: number;
}

const DEFAULT_MIN_CLUSTER_SIZE = 2;
const DEFAULT_RELATIVE_STOP_RATIO = 0.65;

export function clusterPaperVectors(
  input: readonly ClusteringInputPaper[],
  options?: ClusteringOptions,
): ClusteringResult {
  const minClusterSize = requirePositiveInteger(options?.minClusterSize, "minClusterSize", DEFAULT_MIN_CLUSTER_SIZE);
  const centerCorpus = options?.centerCorpus ?? true;
  const minSimilarity = requireFiniteInRange(options?.minSimilarity, "minSimilarity", 0, 0, 1);
  const relativeStopRatio = requireFiniteInRange(
    options?.relativeStopRatio, "relativeStopRatio", DEFAULT_RELATIVE_STOP_RATIO, 0, 1,
  );

  const papers = input
    .filter((paper) => paper.chunks.some((chunk) => chunk.length > 0))
    .map((paper) => ({ paperKey: paper.paperKey, chunks: paper.chunks.map(normalizedChunk) }))
    .sort((left, right) => (left.paperKey < right.paperKey ? -1 : left.paperKey > right.paperKey ? 1 : 0));
  if (papers.length === 0) return { clusters: [], outliers: [] };

  if (centerCorpus) centerChunks(papers);

  // Pairwise strongest chunk evidence (symmetric), as mergeable edges.
  const n = papers.length;
  const similarity = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    similarity[i * n + i] = 1;
    for (let j = i + 1; j < n; j += 1) {
      const score = maxChunkCosine(papers[i]!.chunks, papers[j]!.chunks);
      similarity[i * n + j] = score;
      similarity[j * n + i] = score;
    }
  }
  const edges: Array<{ i: number; j: number; score: number }> = [];
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const score = similarity[i * n + j]!;
      if (score >= minSimilarity) edges.push({ i, j, score });
    }
  }
  edges.sort((left, right) =>
    right.score !== left.score ? right.score - left.score : left.i !== right.i ? left.i - right.i : left.j - right.j);

  // Kruskal merging: union edges from strongest to weakest, stopping at the
  // relative floor (a fraction of the strongest edge).
  const parent = Array.from({ length: n }, (_, index) => index);
  const find = (x: number): number => {
    let root = x;
    while (parent[root] !== root) root = parent[root]!;
    while (parent[x] !== x) {
      const next = parent[x]!;
      parent[x] = root;
      x = next;
    }
    return root;
  };
  const stop = edges.length === 0 ? -1 : edges[0]!.score * relativeStopRatio;
  for (const edge of edges) {
    if (edge.score < stop) break;
    const rootI = find(edge.i);
    const rootJ = find(edge.j);
    if (rootI !== rootJ) parent[rootI] = rootJ;
  }

  // Components, enumerated in paperKey order (deterministic).
  const byRoot = new Map<number, number[]>();
  for (let index = 0; index < n; index += 1) {
    const root = find(index);
    const members = byRoot.get(root);
    if (members) members.push(index);
    else byRoot.set(root, [index]);
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

function centerChunks(papers: Array<{ paperKey: string; chunks: Float32Array[] }>): void {
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
