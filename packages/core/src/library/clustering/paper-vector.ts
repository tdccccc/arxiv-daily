/**
 * Clustering input from the full-text knowledge base: every ready paper's
 * chunk vectors, deterministically (paperKey order), up to a bound.
 *
 * L2 reshape (2026-08-06): paper-level mean pooling was removed — measured
 * unusable on real e5-small embeddings (normalized-vector means collapse
 * every paper into one direction). Clustering now consumes the chunk vectors
 * directly and ranks paper pairs by strongest chunk evidence.
 */

import type { FullTextKnowledgeBaseStore } from "../fulltext/knowledge-base";
import type { ClusteringInputPaper } from "./clusterer";

/** Upper bound on papers fed to clustering (matches catalog selection scale). */
export const MAX_CLUSTERING_INPUT_PAPERS = 2_000 as const;

/**
 * Chunk cap per paper for clustering: longest papers (hundreds of chunks)
 * would dominate the O(n^2 * c^2) similarity matrix; the earliest chunks
 * carry the abstract/introduction theme signal.
 */
export const MAX_CLUSTERING_CHUNKS_PER_PAPER = 80 as const;

/**
 * Recency rank for a knowledge-base paperKey: arXiv keys carry a YYMM prefix
 * (newest month sorts first; deterministic tiebreak on the full key), and
 * undatable fallback (`file:sha256:…`) keys come last in stable hash order.
 */
export function comparePaperKeysByRecency(left: string, right: string): number {
  const leftMatch = /^arxiv:(\d{4})\.\d{4,5}$/.exec(left);
  const rightMatch = /^arxiv:(\d{4})\.\d{4,5}$/.exec(right);
  if (leftMatch && rightMatch) {
    const leftMonth = leftMatch[1] ?? "";
    const rightMonth = rightMatch[1] ?? "";
    if (leftMonth !== rightMonth) return rightMonth.localeCompare(leftMonth);
    return left.localeCompare(right);
  }
  if (leftMatch) return -1;
  if (rightMatch) return 1;
  return left.localeCompare(right);
}

/**
 * Load every ready paper's chunk vectors from the knowledge base,
 * deterministically (newest arXiv papers first, fallback keys last), up to
 * `limit`. Papers without usable chunks are skipped; long papers are
 * truncated to the chunk cap.
 */
export async function buildClusteringInput(
  store: FullTextKnowledgeBaseStore,
  limit: number = MAX_CLUSTERING_INPUT_PAPERS,
): Promise<ClusteringInputPaper[]> {
  const manifest = await store.loadManifest();
  const papers: ClusteringInputPaper[] = [];
  for (const paperKey of Object.keys(manifest.papers).sort(comparePaperKeysByRecency)) {
    if (papers.length >= limit) break;
    const record = manifest.papers[paperKey];
    if (!record || record.status !== "ready") continue;
    const document = await store.loadPaper(paperKey);
    if (!document) continue;
    const chunks: Float32Array[] = [];
    for (let index = 0; index < document.chunks.length && index < MAX_CLUSTERING_CHUNKS_PER_PAPER; index += 1) {
      const offset = index * document.dimension;
      const chunk = document.vectors.subarray(offset, offset + document.dimension);
      if (chunk.length === 0) continue;
      chunks.push(chunk);
    }
    if (chunks.length === 0) continue;
    papers.push({ paperKey, chunks });
  }
  return papers;
}
