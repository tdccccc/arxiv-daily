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
 * Shortest normalized title treated as identifying a specific work.
 *
 * Content hashing already collapses byte-identical copies, so what reaches
 * title matching is "same paper, different file" — a re-download, a publisher
 * copy versus a preprint. Equal titles are strong evidence there, but generic
 * stubs (`Erratum`, `Introduction`, a bare journal name) are not, and merging
 * on those would silently fuse unrelated papers into one cluster member.
 *
 * This is a conservatism floor, not a tuned parameter: no corpus was fit to
 * it. It sits far above generic stubs and far below real paper titles, whose
 * median length in the frozen corpus is well over a hundred characters.
 */
const MIN_IDENTIFYING_TITLE_CHARS = 30;

/** One group of papers collapsed to a single clustering member by title. */
export interface MergedDuplicatePapers {
  /** The paper that stayed in the clustering input. */
  readonly keptPaperKey: string;
  /** Papers dropped as the same work, in input order. */
  readonly droppedPaperKeys: readonly string[];
  /** The kept paper's title, verbatim, so a caller can show what was matched. */
  readonly title: string;
}

export interface ClusteringInputResult {
  readonly papers: ClusteringInputPaper[];
  /** Empty when nothing was merged. Reported so a merge is never silent. */
  readonly mergedDuplicates: readonly MergedDuplicatePapers[];
}

/** Case, spacing and punctuation carry no identity; the words do. */
function normalizeTitle(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

/**
 * Load every ready paper's chunk vectors from the knowledge base,
 * deterministically (newest arXiv papers first, fallback keys last), up to
 * `limit`. Papers without usable chunks are skipped; long papers are
 * truncated to the chunk cap.
 *
 * Papers sharing a normalized title collapse to their first occurrence: the
 * same work filed twice would otherwise contribute two near-identical members,
 * inflating a real cluster's apparent coherence and fabricating two-member
 * clusters out of nothing but a duplicate. What was merged is returned rather
 * than logged away, because a wrong merge has to be answerable.
 */
export async function buildClusteringInput(
  store: FullTextKnowledgeBaseStore,
  limit: number = MAX_CLUSTERING_INPUT_PAPERS,
): Promise<ClusteringInputResult> {
  const manifest = await store.loadManifest();
  const papers: ClusteringInputPaper[] = [];
  const merges = new Map<string, { keptPaperKey: string; droppedPaperKeys: string[]; title: string }>();
  const seenTitles = new Map<string, string>();
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

    const normalized = document.title === undefined ? "" : normalizeTitle(document.title);
    if (normalized.length >= MIN_IDENTIFYING_TITLE_CHARS) {
      const keptPaperKey = seenTitles.get(normalized);
      if (keptPaperKey !== undefined) {
        const group = merges.get(normalized);
        if (group) group.droppedPaperKeys.push(paperKey);
        continue;
      }
      seenTitles.set(normalized, paperKey);
      merges.set(normalized, { keptPaperKey: paperKey, droppedPaperKeys: [], title: document.title! });
    }
    papers.push({ paperKey, chunks });
  }
  return {
    papers,
    mergedDuplicates: [...merges.values()].filter((group) => group.droppedPaperKeys.length > 0),
  };
}
