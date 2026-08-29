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

import { createEvidenceChunkId, type EvidenceLocator } from "./evidence-chunk";
import { LEGACY_EVIDENCE_DERIVATION, type FullTextPaperDocument } from "./knowledge-base";
import type { EvidenceBlock } from "./generation-index-format";
import { FullTextGenerationIndexStoreError, type OpenedFullTextGeneration } from "./generation-index-store";

/** One matching chunk of a paper, with its similarity score. */
export interface KnowledgeBaseChunkHit {
  /** Retrieval branch that produced this evidence score. */
  source: "dense" | "lexical";
  /** Hit scores are meaningful only within their source channel. */
  scoreKind: "cosine" | "bm25";
  /** Index of the chunk within its paper (0-based, matches `chunks`). */
  chunkIndex: number;
  chunkId: string;
  headings: readonly string[];
  locator: EvidenceLocator;
  /** One-based page of the chunk's first character. */
  page: number;
  text: string;
  score: number;
}

/** A paper-level match: the paper, its score, and its best evidence passages. */
export interface KnowledgeBasePaperMatch {
  paperKey: string;
  /** Best dense evidence similarity, retained for compatibility/display. */
  score: number;
  scoreKind: "cosine" | "bm25";
  /** Score that determines final ordering; RRF is never presented as similarity. */
  rankingScore: number;
  rankingScoreKind: "cosine" | "bm25" | "rrf";
  /**
   * Best evidence passages. Dense/lexical modes use channel score order;
   * hybrid interleaves channel-local top hits deterministically. Hit scores are
   * comparable only when their `scoreKind` and `source` match.
   */
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
export const MAX_DENSE_GENERATION_LIMIT = 1_000;
export const MAX_DENSE_GENERATION_HITS_PER_PAPER = 100;
const DENSE_YIELD_ROWS = 256;

export interface DenseGenerationSearchStats {
  vectorReads: number;
  evidenceReads: number;
  peakCandidates: number;
  peakHits: number;
}

export interface SearchGenerationDenseInput {
  readonly generation: OpenedFullTextGeneration;
  readonly queryVector: Float32Array;
  /** One [0,1] score per indexed paper in canonical paper ordinal order. */
  readonly titleScoresByPaperOrdinal?: readonly number[];
  /** One [0,1] score per indexed paper in canonical paper ordinal order. */
  readonly tokenScoresByPaperOrdinal?: readonly number[];
  readonly centerCorpus?: boolean;
  readonly limit?: number;
  readonly maxHitsPerPaper?: number;
  readonly signal?: AbortSignal;
  readonly stats?: DenseGenerationSearchStats;
}

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
      scoreKind: "cosine",
      rankingScore: score,
      rankingScoreKind: "cosine",
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

interface DenseRowCandidate { row: number; chunkIndex: number; score: number }
interface DensePaperCandidate {
  ordinal: number;
  paperKey: string | null;
  chunkCount: number;
  score: number;
  hits: DenseRowCandidate[];
}

/** Exact bounded scan over one pinned immutable generation. */
export async function searchGenerationDense(input: SearchGenerationDenseInput): Promise<KnowledgeBasePaperMatch[]> {
  const limit = requireBoundedNonNegativeInteger(input.limit, "limit", DEFAULT_LIMIT, MAX_DENSE_GENERATION_LIMIT);
  const maxHits = requireBoundedPositiveInteger(
    input.maxHitsPerPaper, "maxHitsPerPaper", DEFAULT_MAX_HITS_PER_PAPER, MAX_DENSE_GENERATION_HITS_PER_PAPER,
  );
  const descriptor = input.generation.descriptor;
  validateOrdinalScores(input.titleScoresByPaperOrdinal, "titleScoresByPaperOrdinal", descriptor.corpusStats.indexedPaperCount);
  validateOrdinalScores(input.tokenScoresByPaperOrdinal, "tokenScoresByPaperOrdinal", descriptor.corpusStats.indexedPaperCount);
  if (!(input.queryVector instanceof Float32Array) || input.queryVector.length !== descriptor.dimension) {
    throw new Error(`searchGenerationDense: query vector dimension ${input.queryVector.length} does not match generation dimension ${descriptor.dimension}`);
  }
  throwIfCancelled(input.signal);
  const stats = input.stats;
  if (stats) { stats.vectorReads = 0; stats.evidenceReads = 0; stats.peakCandidates = 0; stats.peakHits = 0; }
  if (limit === 0 || descriptor.corpusStats.chunkCount === 0) return [];

  const query = new Float32Array(input.queryVector);
  if (input.centerCorpus !== false) {
    for (let column = 0; column < query.length; column += 1) query[column] = query[column]! - descriptor.corpusMean[column]!;
    normalizeInPlace(query);
  }
  const topPapers: DensePaperCandidate[] = [];
  let retainedPaperHitCount = 0;
  let currentOrdinal: number | null = null;
  let currentHits: DenseRowCandidate[] = [];
  let currentChunkCount = 0;
  let previousOrdinal: number | null = null;
  let expectedRow = 0;
  let rowsSinceYield = 0;

  const finalize = () => {
    if (currentOrdinal === null || currentHits.length === 0) return;
    currentHits.sort(compareDenseRows);
    const hits = currentHits.slice(0, maxHits);
    let score = hits[0]!.score;
    const titleScore = input.titleScoresByPaperOrdinal?.[currentOrdinal];
    if (titleScore !== undefined) score = Math.max(score, titleScore);
    const tokenScore = input.tokenScoresByPaperOrdinal?.[currentOrdinal];
    if (tokenScore !== undefined) score = Math.max(score, tokenScore);
    const change = insertBounded(
      topPapers,
      { ordinal: currentOrdinal, paperKey: null, chunkCount: currentChunkCount, score, hits },
      limit,
      compareDensePapers,
    );
    if (change.inserted) retainedPaperHitCount += hits.length - (change.removed?.hits.length ?? 0);
    // Ownership of currentHits has either moved into retained papers or was discarded.
    observeDenseBuffers(stats, topPapers.length, retainedPaperHitCount, 0);
  };

  for await (const object of input.generation.iterateVectorBlocks()) {
    if (stats) stats.vectorReads += 1;
    const block = object.block;
    if (block.dimension !== descriptor.dimension || block.rowStart !== expectedRow) {
      throw corruptDense("dense vector block metadata is not continuous");
    }
    const scratch = new Float32Array(block.dimension);
    for (let row = 0; row < block.rowCount; row += 1) {
      const ordinal = block.paperOrdinals[row]!;
      if (previousOrdinal === null) {
        if (ordinal !== 0) throw corruptDense("first dense paper ordinal must be zero");
      } else if (ordinal !== previousOrdinal && ordinal !== previousOrdinal + 1) {
        throw corruptDense("dense paper ordinals are not continuous");
      }
      if (currentOrdinal !== ordinal) {
        finalize();
        currentOrdinal = ordinal;
        currentHits = [];
        currentChunkCount = 0;
      }
      const offset = row * block.dimension;
      for (let column = 0; column < block.dimension; column += 1) {
        scratch[column] = input.centerCorpus === false
          ? block.vectors[offset + column]!
          : block.vectors[offset + column]! - descriptor.corpusMean[column]!;
      }
      if (input.centerCorpus !== false) normalizeInPlace(scratch);
      const chunkIndex = currentChunkCount++;
      insertBounded(
        currentHits,
        { row: block.rowStart + row, chunkIndex, score: cosineSimilarity(scratch, query) },
        maxHits,
        compareDenseRows,
      );
      observeDenseBuffers(stats, topPapers.length, retainedPaperHitCount, currentHits.length);
      previousOrdinal = ordinal;
      rowsSinceYield += 1;
      if (rowsSinceYield === DENSE_YIELD_ROWS) {
        await yieldToTimer(input.signal);
        rowsSinceYield = 0;
      }
    }
    expectedRow += block.rowCount;
  }
  finalize();
  if (expectedRow !== descriptor.corpusStats.chunkCount || previousOrdinal! + 1 !== descriptor.corpusStats.indexedPaperCount) {
    throw corruptDense("dense vector stream does not match descriptor corpusStats");
  }
  // Covers the final partial scan batch before any evidence I/O begins.
  throwIfCancelled(input.signal);

  // Resolve paper keys and selected evidence late. Evidence blocks are routed by
  // descriptor ranges and each distinct selected ref is read at most once.
  const selectedRows = new Map<number, { paper: DensePaperCandidate; hit: DenseRowCandidate }>();
  for (const paper of topPapers) for (const hit of paper.hits) selectedRows.set(hit.row, { paper, hit });
  const evidenceRefs = descriptor.objects.filter((reference) => reference.kind === "evidence");
  const materialized = new Map<number, KnowledgeBaseChunkHit>();
  for (const reference of evidenceRefs) {
    const selected = [...selectedRows.keys()].filter((row) => row >= reference.recordStart && row < reference.recordStart + reference.recordCount);
    if (selected.length === 0) continue;
    throwIfCancelled(input.signal);
    const object = await input.generation.readObject(reference);
    if (stats) stats.evidenceReads += 1;
    throwIfCancelled(input.signal);
    if (object.reference.kind !== "evidence") throw corruptDense("selected evidence reference decoded as the wrong kind");
    for (const row of selected) {
      const record = (object.block as EvidenceBlock).records[row - reference.recordStart];
      const candidate = selectedRows.get(row)!;
      if (!record || record.vectorRow !== row || record.paperIndex !== candidate.paper.ordinal
        || record.chunk.index !== candidate.hit.chunkIndex) {
        throw corruptDense("selected evidence row does not match dense candidate");
      }
      if (candidate.paper.paperKey === null) candidate.paper.paperKey = record.paperKey;
      else if (candidate.paper.paperKey !== record.paperKey) {
        throw corruptDense("selected evidence paperKey changed within one paper ordinal");
      }
      materialized.set(row, {
        source: "dense", scoreKind: "cosine", chunkIndex: record.chunk.index, chunkId: record.chunk.id,
        headings: record.chunk.headings, locator: record.chunk.locator, page: record.chunk.page,
        text: record.chunk.text, score: candidate.hit.score,
      });
    }
    // Materialization remains cancellable even for one-row evidence blocks.
    await yieldToTimer(input.signal);
  }
  topPapers.sort(compareDensePapers);
  return topPapers.map((paper) => {
    if (paper.paperKey === null) throw corruptDense("selected dense paperKey was not materialized");
    return {
      paperKey: paper.paperKey,
      score: paper.score,
      scoreKind: "cosine" as const,
      rankingScore: paper.score,
      rankingScoreKind: "cosine" as const,
      hits: paper.hits.map((hit) => {
        const value = materialized.get(hit.row);
        if (!value) throw corruptDense("selected dense evidence was not materialized");
        return value;
      }),
      chunkCount: paper.chunkCount,
    };
  });
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
    const locator = chunk.locator ?? { pageStart: chunk.page };
    const headings = chunk.headings ?? [];
    const derivation = chunk.derivation ?? paper.derivation ?? LEGACY_EVIDENCE_DERIVATION;
    return {
      source: "dense",
      scoreKind: "cosine",
      chunkIndex: chunk.index,
      chunkId: chunk.id ?? createEvidenceChunkId({ text: chunk.text, headings, locator, derivation }),
      headings,
      locator,
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

function requireBoundedNonNegativeInteger(value: number | undefined, name: string, fallback: number, cap: number): number {
  const actual = value ?? fallback;
  if (!Number.isSafeInteger(actual) || actual < 0 || actual > cap) {
    throw new TypeError(`searchGenerationDense: ${name} must be an integer from 0 through ${cap}`);
  }
  return actual;
}

function requireBoundedPositiveInteger(value: number | undefined, name: string, fallback: number, cap: number): number {
  const actual = value ?? fallback;
  if (!Number.isSafeInteger(actual) || actual < 1 || actual > cap) {
    throw new TypeError(`searchGenerationDense: ${name} must be an integer from 1 through ${cap}`);
  }
  return actual;
}

function compareDenseRows(left: DenseRowCandidate, right: DenseRowCandidate): number {
  return right.score !== left.score ? right.score - left.score : left.chunkIndex - right.chunkIndex;
}

function compareDensePapers(left: DensePaperCandidate, right: DensePaperCandidate): number {
  return right.score !== left.score ? right.score - left.score : left.ordinal - right.ordinal;
}

function insertBounded<T>(
  items: T[],
  candidate: T,
  capacity: number,
  compare: (left: T, right: T) => number,
): { inserted: boolean; removed?: T } {
  if (items.length === capacity && compare(candidate, items[items.length - 1]!) >= 0) return { inserted: false };
  let index = 0;
  while (index < items.length && compare(items[index]!, candidate) <= 0) index += 1;
  if (items.length < capacity) {
    items.splice(index, 0, candidate);
    return { inserted: true };
  }
  const removed = items[items.length - 1]!;
  // Replace the old worst item in place before shifting; length never exceeds capacity.
  for (let position = items.length - 1; position > index; position -= 1) items[position] = items[position - 1]!;
  items[index] = candidate;
  return { inserted: true, removed };
}

function observeDenseBuffers(
  stats: DenseGenerationSearchStats | undefined,
  candidateCount: number,
  retainedPaperHitCount: number,
  currentHitCount: number,
): void {
  if (!stats) return;
  stats.peakCandidates = Math.max(stats.peakCandidates, candidateCount);
  stats.peakHits = Math.max(stats.peakHits, retainedPaperHitCount + currentHitCount);
}

function validateOrdinalScores(scores: readonly number[] | undefined, name: string, paperCount: number): void {
  if (scores === undefined) return;
  if (!Array.isArray(scores) || scores.length !== paperCount) {
    throw new TypeError(`searchGenerationDense: ${name} length must equal indexedPaperCount ${paperCount}`);
  }
  if (scores.some((score) => !Number.isFinite(score) || score < 0 || score > 1)) {
    throw new TypeError(`searchGenerationDense: ${name} scores must be finite numbers in [0, 1]`);
  }
}

function corruptDense(message: string): FullTextGenerationIndexStoreError {
  return new FullTextGenerationIndexStoreError(message, "corrupt-or-unreadable");
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  if (typeof DOMException === "function") throw new DOMException("The operation was aborted", "AbortError");
  const error = new Error("The operation was aborted");
  error.name = "AbortError";
  throw error;
}

async function yieldToTimer(signal?: AbortSignal): Promise<void> {
  throwIfCancelled(signal);
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  throwIfCancelled(signal);
}
