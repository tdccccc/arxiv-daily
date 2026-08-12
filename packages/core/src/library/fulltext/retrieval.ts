/**
 * Brute-force cosine similarity retrieval over the full-text knowledge base.
 *
 * Pure, deterministic, side-effect-free: the same papers and query vector
 * always produce the same matches. Performs no I/O and uses no randomness.
 *
 * Paper-level scoring — maximum chunk similarity:
 * A paper's score is the similarity of its single most similar chunk (its
 * best evidence passage), so the top hit is always the reason for the
 * paper's rank — results stay explainable end to end.
 *
 * Alternatives considered and rejected:
 * - Mean similarity: a paper with one strong passage and many mediocre ones
 *   would sink below a paper whose chunks are uniformly "okay", even though
 *   the first paper clearly contains the sought content. Means also depend
 *   on the chunker's arbitrary boundaries.
 * - Top-k average: reintroduces a k choice and mixes evidence of different
 *   quality into one opaque number.
 * Maximum similarity is invariant to how finely a paper is chunked: splitting
 * or merging passages does not lower the score of a paper that contains a
 * perfect passage, while means and top-k averages shift with the chunker.
 *
 * Title fusion — short-query robustness:
 * Short queries (a paper title, a few keywords) embed into a different
 * region of the embedding space than long passage chunks, so a paper's
 * highest chunk similarity can lose to an unrelated paper whose many chunks
 * happen to include a coincidentally similar passage (observed with a
 * 262-chunk model paper beating every title query). Embedding similarity is
 * also collapsed for short texts with some remote models, so the title
 * signal is computed lexically by the caller (`title-similarity.ts`): the
 * paper score is the maximum of its best chunk similarity and its title
 * score. A title-length query lands on its own paper; a free-text query
 * scores 0 against every title and is unaffected. Title evidence only
 * re-ranks papers that already have chunk evidence — it never surfaces an
 * unindexed paper with empty hits.
 *
 * Ordering: papers by score descending, ties by paperKey ascending; hits by
 * score descending, ties by chunk index ascending. Output is fully
 * deterministic. `limit: 0` or an empty papers list yields an empty result.
 * Papers without chunks contribute no evidence and are skipped.
 *
 * Dimension contract: the query vector must match each paper's `dimension`
 * (the embedding model's output width). A mismatch means the knowledge base
 * was built with a different embedding model and must be rebuilt; it is
 * reported as an error rather than silently producing meaningless scores.
 */

import type { FullTextPaperDocument } from "./knowledge-base";

/** One matching chunk of a paper, with its similarity score. */
export interface KnowledgeBaseChunkHit {
  /** Index of the chunk within its paper (0-based, matches `chunks`). */
  chunkIndex: number;
  /** One-based page of the chunk's first character. */
  page: number;
  text: string;
  score: number;
}

/** A paper-level match: the paper, its score, and its best evidence passages. */
export interface KnowledgeBasePaperMatch {
  paperKey: string;
  /** Maximum chunk similarity — the score of the best evidence passage. */
  score: number;
  /** Best evidence passages, score descending (ties by chunk index). */
  hits: KnowledgeBaseChunkHit[];
  /** Total number of chunks in the paper. */
  chunkCount: number;
}

export interface SearchKnowledgeBaseInput {
  papers: readonly FullTextPaperDocument[];
  queryVector: Float32Array;
  /**
   * Optional per-paper title scores (paperKey → similarity in [0, 1],
   * computed lexically by the caller; see `title-similarity.ts`). When
   * present, a paper's score is the maximum of its best chunk similarity
   * and its title score; see the module comment for the short-query
   * rationale. Papers without an entry are scored by chunk similarity
   * alone.
   */
  titleScores?: ReadonlyMap<string, number>;
  /**
   * Optional per-paper lexical token-hit scores (paperKey → hit ratio in
   * [0, 1], computed by the caller; see `lexical-search.ts`). Keyword
   * queries land here when embedding similarity is collapsed. Same max
   * fusion as `titleScores`.
   */
  tokenScores?: ReadonlyMap<string, number>;
  /**
   * Subtract the corpus chunk mean and renormalize before scoring (same
   * transform placement/clustering use). Default true. Disable only for
   * diagnostics or pure unit tests of the raw cosine path.
   */
  centerCorpus?: boolean;
  /** Maximum number of matching papers to return. Default 10. 0 yields no matches. */
  limit?: number;
  /** Maximum number of hit chunks to report per paper. Default 3. */
  maxHitsPerPaper?: number;
}

const DEFAULT_LIMIT = 10;
const DEFAULT_MAX_HITS_PER_PAPER = 3;

/**
 * Cosine similarity between two equal-length vectors, in [-1, 1]. A zero
 * vector has no direction, so its similarity is defined as 0 regardless of
 * the other vector.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`cosineSimilarity: vectors must have equal length, got ${a.length} and ${b.length}`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < a.length; index += 1) {
    const x = a[index]!;
    const y = b[index]!;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function searchKnowledgeBase(input: SearchKnowledgeBaseInput): KnowledgeBasePaperMatch[] {
  const limit = requireNonNegativeInteger(input.limit, "limit", DEFAULT_LIMIT);
  const maxHitsPerPaper = requirePositiveInteger(input.maxHitsPerPaper, "maxHitsPerPaper", DEFAULT_MAX_HITS_PER_PAPER);
  if (limit === 0 || input.papers.length === 0) return [];

  const centerCorpus = input.centerCorpus !== false;
  const centered = centerCorpus
    ? centerSearchSpace(input.papers, input.queryVector)
    : { papers: input.papers, queryVector: input.queryVector };

  const matches: KnowledgeBasePaperMatch[] = [];
  for (const paper of centered.papers) {
    if (paper.dimension !== centered.queryVector.length) {
      throw new Error(
        `searchKnowledgeBase: query vector dimension ${centered.queryVector.length} does not match paper ` +
          `"${paper.paperKey}" dimension ${paper.dimension}; the knowledge base was built with a different ` +
          "embedding model and must be rebuilt",
      );
    }
    const hits = rankChunkHits(paper, centered.queryVector, maxHitsPerPaper);
    if (hits.length === 0) continue;
    let score = hits[0]!.score;
    const titleScore = input.titleScores?.get(paper.paperKey);
    if (titleScore !== undefined) score = Math.max(score, titleScore);
    const tokenScore = input.tokenScores?.get(paper.paperKey);
    if (tokenScore !== undefined) score = Math.max(score, tokenScore);
    matches.push({
      paperKey: paper.paperKey,
      score,
      hits,
      chunkCount: paper.chunks.length,
    });
  }
  matches.sort((left, right) => {
    if (right.score !== left.score) return right.score - left.score;
    return left.paperKey < right.paperKey ? -1 : left.paperKey > right.paperKey ? 1 : 0;
  });
  return matches.slice(0, limit);
}

/**
 * Corpus-level centering for retrieval: subtract the mean of every chunk in
 * the candidate set from both the query and every chunk, then renormalize.
 * Same transform used by clustering/placement; keeps similar-paper ranking in
 * the space that already separates themes from the shared academic baseline.
 */
function centerSearchSpace(
  papers: readonly FullTextPaperDocument[],
  queryVector: Float32Array,
): { papers: FullTextPaperDocument[]; queryVector: Float32Array } {
  const dimension = queryVector.length;
  let count = 0;
  const mean = new Float64Array(dimension);
  for (const paper of papers) {
    if (paper.dimension !== dimension) {
      throw new Error(
        `searchKnowledgeBase: query vector dimension ${dimension} does not match paper ` +
          `"${paper.paperKey}" dimension ${paper.dimension}; the knowledge base was built with a different ` +
          "embedding model and must be rebuilt",
      );
    }
    for (const chunk of paper.chunks) {
      const offset = chunk.index * dimension;
      for (let index = 0; index < dimension; index += 1) {
        mean[index]! += paper.vectors[offset + index] ?? 0;
      }
      count += 1;
    }
  }
  if (count === 0) {
    return { papers: [...papers], queryVector: new Float32Array(queryVector) };
  }
  for (let index = 0; index < dimension; index += 1) mean[index]! /= count;

  const centeredQuery = new Float32Array(dimension);
  for (let index = 0; index < dimension; index += 1) {
    centeredQuery[index] = queryVector[index]! - mean[index]!;
  }
  normalizeInPlace(centeredQuery);

  const centeredPapers = papers.map((paper) => {
    const vectors = new Float32Array(paper.vectors.length);
    for (const chunk of paper.chunks) {
      const offset = chunk.index * dimension;
      for (let index = 0; index < dimension; index += 1) {
        vectors[offset + index] = (paper.vectors[offset + index] ?? 0) - mean[index]!;
      }
      normalizeSliceInPlace(vectors, offset, dimension);
    }
    return { ...paper, vectors };
  });
  return { papers: centeredPapers, queryVector: centeredQuery };
}

function normalizeInPlace(vector: Float32Array): void {
  let sum = 0;
  for (let index = 0; index < vector.length; index += 1) sum += vector[index]! * vector[index]!;
  if (sum === 0) return;
  const scale = 1 / Math.sqrt(sum);
  for (let index = 0; index < vector.length; index += 1) vector[index]! *= scale;
}

function normalizeSliceInPlace(vectors: Float32Array, offset: number, dimension: number): void {
  let sum = 0;
  for (let index = 0; index < dimension; index += 1) {
    const value = vectors[offset + index]!;
    sum += value * value;
  }
  if (sum === 0) return;
  const scale = 1 / Math.sqrt(sum);
  for (let index = 0; index < dimension; index += 1) vectors[offset + index]! *= scale;
}

/** Score every chunk of a paper against the query and keep the best `maxHitsPerPaper`. */
function rankChunkHits(
  paper: FullTextPaperDocument,
  queryVector: Float32Array,
  maxHitsPerPaper: number,
): KnowledgeBaseChunkHit[] {
  const scored: Array<{ chunkIndex: number; score: number }> = [];
  paper.chunks.forEach((chunk) => {
    const offset = chunk.index * paper.dimension;
    const vector = paper.vectors.subarray(offset, offset + paper.dimension);
    scored.push({ chunkIndex: chunk.index, score: cosineSimilarity(vector, queryVector) });
  });
  scored.sort((left, right) => {
    if (right.score !== left.score) return right.score - left.score;
    return left.chunkIndex - right.chunkIndex;
  });
  return scored.slice(0, maxHitsPerPaper).map((entry) => {
    const chunk = paper.chunks[entry.chunkIndex]!;
    return {
      chunkIndex: chunk.index,
      page: chunk.page,
      text: chunk.text,
      score: entry.score,
    };
  });
}

function requirePositiveInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new TypeError(`searchKnowledgeBase: ${name} must be a positive integer, got ${JSON.stringify(value)}`);
  }
  return value;
}

function requireNonNegativeInteger(value: number | undefined, name: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`searchKnowledgeBase: ${name} must be a non-negative integer, got ${JSON.stringify(value)}`);
  }
  return value;
}
